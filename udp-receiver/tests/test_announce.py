import urllib.error
import urllib.request
from types import SimpleNamespace as NS

import pytest

from open_lexa import announce as announce_module
from open_lexa.announce import (FAILED, NOT_UNDERSTOOD, Announcer, ClipCache, error_phrase,
                                phrases_from, restore_mode, start_clip_server)
from open_lexa.handlers import Dispatcher
from open_lexa.handlers.base import HandlerResult, MCPServer, MCPTool
from open_lexa.handlers.sonos import sonos as sonos_module
from open_lexa.handlers.sonos.sonos import SonosSpeaker
from open_lexa.store import DirStore
from open_lexa.tool_agent import process


# --- tool agent hook ---------------------------------------------------------------

class Server(MCPServer):
    id = 'test'
    tools = (MCPTool('play_spotify', 'Włącz Spotify'),)

    def __init__(self, events, result=None):
        super().__init__()
        self.events = events
        self.result = result or HandlerResult.handled('OK')

    def handle(self, request):
        self.events.append(('dispatch', request.tool))
        return self.result


def _llm(calls, content=None):
    message = NS(tool_calls=calls, content=content)
    client = NS(chat=NS(completions=NS(create=lambda **kwargs: NS(
        choices=[NS(message=message, finish_reason='stop')]))))
    return NS(model='test', client=client, stats={'calls': 0, 'failures': 0})


def _call(name='play_spotify', arguments='{}'):
    return NS(function=NS(name=name, arguments=arguments))


def _setup(result=None):
    events = []
    dispatcher = Dispatcher([Server(events, result)])
    dispatcher.apply_config({'test': {'enabled': True}})

    def announce(key, session=None, **kwargs):
        events.append(('announce', key, *kwargs.values()))
    return events, dispatcher, announce


def test_announcement_plays_before_the_tool_runs():
    events, dispatcher, announce = _setup()
    result = process('włącz spotify', _llm([_call()]), dispatcher, announce=announce)
    assert result['status'] == 'handled'
    assert events == [('announce', 'play_spotify'), ('dispatch', 'play_spotify')]


def test_not_understood_when_no_tool_and_failed_on_errors():
    events, dispatcher, announce = _setup()
    assert process('bla bla', _llm([], 'Nie wiem'), dispatcher,
                   announce=announce)['status'] == 'no_tool'
    assert process('włącz', _llm([_call('unknown')]), dispatcher,
                   announce=announce)['status'] == 'failed'
    assert events == [('announce', NOT_UNDERSTOOD),
                      ('announce', FAILED, "unknown MCP tool 'unknown'")]


@pytest.mark.parametrize('result, error', [
    (HandlerResult.failed('UPnP Error 714'), 'UPnP Error 714'),
    (HandlerResult.skipped('set default_favorite'), 'set default_favorite'),
])
def test_tool_failure_is_announced_after_the_confirmation(result, error):
    events, dispatcher, announce = _setup(result)
    assert process('włącz', _llm([_call()]), dispatcher,
                   announce=announce)['status'] == result.status
    assert events == [('announce', 'play_spotify'), ('dispatch', 'play_spotify'),
                      ('announce', FAILED, error)]


def test_error_phrase_placeholder():
    assert error_phrase('Przepraszam.', 'boom') == 'Przepraszam.'
    assert error_phrase('Błąd: {error}', ' a\n b ') == 'Błąd: a b'
    assert error_phrase('Błąd: {error}', None) == 'Błąd:'
    long = error_phrase('{error}', 'słowo ' * 100)
    assert len(long) <= 161 and long.endswith('…')


def test_empty_transcript_is_silent_and_announcer_errors_do_not_block():
    events, dispatcher, _ = _setup()
    assert process('  ', _llm([_call()]), dispatcher,
                   announce=lambda *a, **k: events.append('called'))['status'] == 'empty'
    assert events == []

    def broken(key, session=None):
        raise RuntimeError('speaker gone')
    assert process('włącz', _llm([_call()]), dispatcher, announce=broken)['status'] == 'handled'
    assert events == [('dispatch', 'play_spotify')]


# --- announcer ---------------------------------------------------------------------

class FakeSpeaker:
    instances = []

    def __init__(self, ip, timeout=5.0):
        self.ip, self.timeout, self.clips = ip, timeout, []
        FakeSpeaker.instances.append(self)

    def play_clip(self, url, title, *, after, max_seconds):
        self.clips.append((url, title, after))
        return True


def _announcer(tmp_path, **announce):
    synthesised = []

    def tts(text, model, voice):
        synthesised.append((text, model, voice))
        return b'ID3' + text.encode()
    clips = ClipCache(DirStore(tmp_path), tts)
    clips.prefetch = lambda *args, **kwargs: None
    announcer = Announcer(clips, speaker_factory=FakeSpeaker)
    announcer.configure({'enabled': True, 'speaker_ip': '10.0.0.5',
                         'announce': {'enabled': True, 'base_url': 'http://10.0.0.2:8081/',
                                      **announce}})
    return announcer, clips, synthesised


def test_announcer_plays_cached_clip_with_restore_policy(tmp_path):
    announcer, clips, synthesised = _announcer(tmp_path, phrases={'pause': ''})
    speaker = announcer.speaker
    assert announcer('play_spotify') is True
    assert announcer('play_spotify') is True
    assert announcer(NOT_UNDERSTOOD) is True
    assert announcer('pause') is False  # explicitly emptied phrase
    assert announcer('unknown_tool') is False
    name = ClipCache.name('Okej, uruchamiam Spotify.', 'gpt-4o-mini-tts', 'nova')
    assert speaker.clips[0] == (f'http://10.0.0.2:8081/clips/{name}',
                                'Okej, uruchamiam Spotify.', 'none')
    assert speaker.clips[2][1:] == ('Sorry, nie zrozumiałam.', 'full')
    assert [text for text, _, _ in synthesised] == ['Okej, uruchamiam Spotify.',
                                                    'Sorry, nie zrozumiałam.']
    # Persisted on the archive: a fresh cache (after a restart) skips TTS.
    assert (tmp_path / '_clips' / name).read_bytes().startswith(b'ID3')
    fresh = ClipCache(DirStore(tmp_path), tts=None)
    assert fresh.get('Okej, uruchamiam Spotify.', 'gpt-4o-mini-tts', 'nova') == name
    assert clips.data(name) == fresh.data(name)


def test_announcer_speaks_error_details_without_persisting(tmp_path, monkeypatch):
    announcer, clips, synthesised = _announcer(tmp_path)
    assert announcer(FAILED, error='UPnP 714') is True
    assert announcer.speaker.clips[-1][1] == 'Przepraszam, coś poszło nie tak, spróbuj ponownie.'
    announcer.phrases[FAILED] = 'Błąd: {error}'
    monkeypatch.setattr(announce_module, 'MAX_EPHEMERAL_CLIPS', 2)
    for number in range(3):
        assert announcer(FAILED, error=f'błąd {number}') is True
    assert announcer.speaker.clips[-1][1:] == ('Błąd: błąd 2', 'full')
    names = [ClipCache.name(f'Błąd: błąd {n}', 'gpt-4o-mini-tts', 'nova') for n in range(3)]
    assert clips.data(names[0]) is None and clips.data(names[2]) is not None
    assert not any((tmp_path / '_clips' / name).exists() for name in names)


@pytest.mark.parametrize('sonos', [
    {},
    {'enabled': False, 'speaker_ip': '10.0.0.5',
     'announce': {'enabled': True, 'base_url': 'http://x:8081'}},
    {'enabled': True, 'speaker_ip': '10.0.0.5', 'announce': {'enabled': False}},
    {'enabled': True, 'speaker_ip': '10.0.0.5', 'announce': {'enabled': True, 'base_url': ''}},
])
def test_announcer_disabled(sonos):
    announcer = Announcer(ClipCache(), speaker_factory=FakeSpeaker)
    announcer.configure(sonos)
    assert announcer.enabled is False
    assert announcer('play_spotify') is False


def test_announcer_swallows_tts_failure_and_backs_off(tmp_path, monkeypatch):
    attempts = []

    def tts(text, model, voice):
        attempts.append(text)
        raise RuntimeError('quota')
    clips = ClipCache(DirStore(tmp_path), tts)
    clips.prefetch = lambda *args, **kwargs: None
    announcer = Announcer(clips, speaker_factory=FakeSpeaker)
    announcer.configure({'enabled': True, 'speaker_ip': '10.0.0.5',
                         'announce': {'enabled': True, 'base_url': 'http://h:8081'}})
    assert announcer('stop') is False
    assert announcer('stop') is False  # within the backoff: no second TTS request
    assert announcer.speaker.clips == []
    assert attempts == ['Zatrzymuję.']
    clock = announce_module.time.monotonic() + announce_module.FAILURE_BACKOFF + 1
    monkeypatch.setattr(announce_module.time, 'monotonic', lambda: clock)
    assert announcer('stop') is False
    assert attempts == ['Zatrzymuję.', 'Zatrzymuję.']


def test_restore_mode_and_phrases():
    assert restore_mode('play_radio') == 'none'
    assert restore_mode('pause') == restore_mode('stop') == 'paused'
    assert restore_mode('volume_up') == restore_mode(NOT_UNDERSTOOD) == 'full'
    phrases = phrases_from({'stop': '  Stop,   już.  ', 'resume': None})
    assert phrases['stop'] == 'Stop, już.'
    assert phrases['resume'] == ''
    assert phrases['play_spotify'] == 'Okej, uruchamiam Spotify.'


# --- clip HTTP server --------------------------------------------------------------

def test_clip_server_serves_cached_clips_only():
    clips = ClipCache()
    name = ClipCache.name('a', 'm', 'v')
    clips.memory[name] = b'0123456789'
    server = start_clip_server(clips, 0, host='127.0.0.1')
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        with urllib.request.urlopen(f'{base}/clips/{name}') as response:
            assert response.headers['Content-Type'] == 'audio/mpeg'
            assert response.read() == b'0123456789'
        request = urllib.request.Request(f'{base}/clips/{name}', headers={'Range': 'bytes=2-4'})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert response.headers['Content-Range'] == 'bytes 2-4/10'
            assert response.read() == b'234'
        request = urllib.request.Request(f'{base}/clips/{name}', method='HEAD')
        with urllib.request.urlopen(request) as response:
            assert response.headers['Content-Length'] == '10'
        for path in (f'/clips/{"0" * 24}.mp3', '/clips/../etc/passwd', '/healthz'):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


# --- Sonos playback ----------------------------------------------------------------

class Coordinator:
    def __init__(self, states):
        self.states = list(states)
        self.calls = []

    def get_current_transport_info(self):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return {'current_transport_state': state}

    def play_uri(self, uri, meta='', title='', **kwargs):
        self.calls.append(('play_uri', uri, title))

    def pause(self):
        self.calls.append(('pause',))

    def stop(self):
        self.calls.append(('stop',))


def _speaker(monkeypatch, coordinator, *, transport='PLAYING', uri='x-rincon-queue:RINCON#0',
             cloud=False):
    snapshots = []

    class Snapshot:
        def __init__(self, device):
            self.device = device
            self.transport_state = None
            self.media_uri = None
            self.is_playing_cloud_queue = False
            snapshots.append(self)

        def snapshot(self):
            self.transport_state = transport
            self.media_uri = uri
            self.is_playing_cloud_queue = cloud

        def restore(self):
            coordinator.calls.append(('restore', self.transport_state))

    import soco.snapshot
    monkeypatch.setattr(soco.snapshot, 'Snapshot', Snapshot)
    monkeypatch.setattr(sonos_module, 'CLIP_POLL_SECONDS', 0)
    monkeypatch.setattr(SonosSpeaker, 'coordinator', property(lambda self: coordinator))
    return SonosSpeaker('10.0.0.5'), snapshots


@pytest.mark.parametrize('after, restored', [
    ('full', [('restore', 'PLAYING')]),
    ('paused', [('restore', 'PAUSED_PLAYBACK')]),
    ('none', []),
])
def test_play_clip_waits_then_applies_restore_policy(monkeypatch, after, restored):
    coordinator = Coordinator(['TRANSITIONING', 'PLAYING', 'PLAYING', 'STOPPED'])
    speaker, snapshots = _speaker(monkeypatch, coordinator)
    assert speaker.play_clip('http://h/clips/a.mp3', 'Hej', after=after) is True
    assert coordinator.calls[0] == ('play_uri', 'http://h/clips/a.mp3', 'Hej')
    assert coordinator.calls[1:] == restored
    assert coordinator.states == ['STOPPED']  # polled until the clip ended
    assert len(snapshots) == (0 if after == 'none' else 1)


def test_play_clip_skips_sources_it_cannot_restore(monkeypatch):
    for kwargs in ({'uri': 'x-sonos-vli:RINCON_1:2,spotify:abc'}, {'cloud': True}):
        coordinator = Coordinator(['PLAYING'])
        speaker, _ = _speaker(monkeypatch, coordinator, **kwargs)
        assert speaker.play_clip('http://h/c.mp3', 'x', after='full') is False
        assert coordinator.calls == []
    # Paused Spotify Connect: nothing to lose, the clip plays.
    coordinator = Coordinator(['PLAYING', 'STOPPED'])
    speaker, _ = _speaker(monkeypatch, coordinator, transport='PAUSED_PLAYBACK',
                          uri='x-sonos-vli:RINCON_1:2,spotify:abc')
    assert speaker.play_clip('http://h/c.mp3', 'x', after='full') is True
    # Replacing playback does not need a restore, so even Connect is interrupted.
    coordinator = Coordinator(['PLAYING', 'STOPPED'])
    speaker, snapshots = _speaker(monkeypatch, coordinator, uri='x-sonos-vli:x')
    assert speaker.play_clip('http://h/c.mp3', 'x', after='none') is True
    assert snapshots == []


def test_play_clip_restores_music_when_the_clip_fails(monkeypatch):
    coordinator = Coordinator(['PLAYING'])
    speaker, _ = _speaker(monkeypatch, coordinator)

    def fail(uri, meta='', title='', **kwargs):
        coordinator.calls.append(('play_uri', uri))
        raise RuntimeError('UPnP Error 714: illegal MIME type')
    coordinator.play_uri = fail
    with pytest.raises(sonos_module.SonosError):
        speaker.play_clip('http://h/c.mp3', 'x', after='full')
    assert coordinator.calls == [('play_uri', 'http://h/c.mp3'), ('restore', 'PLAYING')]


def test_play_clip_gives_up_after_max_seconds(monkeypatch):
    coordinator = Coordinator(['PLAYING'])
    speaker, _ = _speaker(monkeypatch, coordinator)
    assert speaker.play_clip('http://h/c.mp3', 'x', after='full', max_seconds=0.05) is True
    assert coordinator.calls[-1] == ('restore', 'PLAYING')


def test_pause_and_stop_ignore_an_already_stopped_speaker(monkeypatch):
    coordinator = Coordinator(['STOPPED'])
    speaker, _ = _speaker(monkeypatch, coordinator)
    speaker.pause()
    speaker.stop()
    assert coordinator.calls == []
    coordinator = Coordinator(['PLAYING'])
    speaker, _ = _speaker(monkeypatch, coordinator)
    speaker.pause()
    speaker.stop()
    assert coordinator.calls == [('pause',), ('stop',)]


def test_tts_from_env_requires_a_key():
    assert announce_module.OpenAITTS.from_env({}) is None
    created = []
    speech = NS(create=lambda **kwargs: created.append(kwargs) or NS(content=b'mp3'))
    tts = announce_module.OpenAITTS('k', client=NS(audio=NS(speech=speech)))
    assert tts('Hej', 'gpt-4o-mini-tts', 'nova') == b'mp3'
    assert created[0]['response_format'] == 'mp3' and 'instructions' in created[0]
    tts('Hej', 'tts-1', 'nova')
    assert 'instructions' not in created[1]


def test_unconfigured_llm_and_stt_failure_are_announced():
    from test_commands_archive import run_worker
    from test_server import FakeTranscriber

    events, dispatcher, announce = _setup()
    assert process('włącz', None, dispatcher, announce=announce)['status'] == 'unavailable'
    assert events == [('announce', FAILED, 'LLM is not configured')]

    calls = []
    state = run_worker(FakeTranscriber([RuntimeError('whisper down')]),
                       announcer=lambda key, session=None, **kw: calls.append((key, session, kw)))
    assert state.failures == 1
    assert calls[0][0] == FAILED and calls[0][1] == '00000001'
    assert 'whisper down' in calls[0][2]['error']


def test_radio_favorite_with_container_meta_is_not_queued():
    radio = {'uri': 'x-sonosapi-radio:artist?sid=303', 'meta': 'object.container.radio'}
    assert SonosSpeaker.is_container(radio) is False
    assert SonosSpeaker.is_container({'uri': 'x-rincon-cpcontainer:1006286c', 'meta': ''})
    assert SonosSpeaker.is_container(
        {'uri': 'file:///jffs/settings/savedqueues.rsq#3', 'meta': 'object.container.playlistContainer'})
