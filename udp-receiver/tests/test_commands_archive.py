import io
import json
import queue
import threading
import wave

import pytest

from open_lexa.archive import SmbArchive, create_archive
from open_lexa.commands import DEFAULT_REPLY, decide, normalize
from open_lexa.protocol import Recording
from open_lexa.server import HealthState, worker
from open_lexa.stt import ConfigError, DualTranscriber

from test_server import FakeTranscriber


class MemoryArchive:
    def __init__(self, fail=False):
        self.saved = []
        self.fail = fail
        self.closed = False

    def save(self, stem, wav, meta):
        if self.fail:
            raise OSError('share unavailable')
        self.saved.append((stem, wav, meta))

    def close(self):
        self.closed = True


class FakeSmbFile(io.BytesIO):
    def __init__(self, files, path):
        super().__init__()
        self.files, self.path = files, path

    def __exit__(self, *exc):
        self.files[self.path] = self.getvalue()
        return super().__exit__(*exc)


class FakeSmbClient:
    def __init__(self, fail_writes=0, port=445):
        self.port = port
        self.files = {}
        self.sessions = []
        self.dirs = []
        self.resets = 0
        self.fail_writes = fail_writes

    def register_session(self, server, **kwargs):
        self.sessions.append((server, kwargs))

    def makedirs(self, path, exist_ok=False, **kwargs):
        assert kwargs['port'] == self.port
        self.dirs.append(path)

    def open_file(self, path, mode='r', **kwargs):
        assert kwargs['port'] == self.port
        if self.fail_writes:
            self.fail_writes -= 1
            raise ConnectionError('connection reset')
        return FakeSmbFile(self.files, path)

    def reset_connection_cache(self, fail_on_error=True, connection_cache=None):
        self.resets += 1


def run_worker(transcriber, **kwargs):
    state = HealthState(transcriber.name)
    jobs = queue.Queue()
    thread = threading.Thread(target=worker, args=(jobs, transcriber, 'pl', None, state),
                              kwargs=kwargs, daemon=True)
    thread.start()
    for session in range(len(transcriber.results)):
        jobs.put(Recording(('127.0.0.1', 1234), session + 1, bytes(6400)))
    jobs.put(None)
    thread.join(timeout=5)
    assert not thread.is_alive()
    return state


def test_normalize_drops_punctuation_and_case():
    assert normalize('  Włącz, ŚWIATŁO!  ') == 'włącz światło'


@pytest.mark.parametrize('text, command', [
    ('Włącz światło.', 'lights_on'),
    ('Wyłącz światło w kuchni', 'lights_off'),
    ('Puść radio', 'play_radio'),
    ('Ciszej!', 'volume_down'),
])
def test_decide_matches_rules(text, command):
    assert decide(text).command == command


@pytest.mark.parametrize('text', ['Jaka jest pogoda?', '', None])
def test_decide_defaults_to_sorry(text):
    decision = decide(text)
    assert decision.command is None
    assert decision.reply == DEFAULT_REPLY == 'sorry ale nie rozumiem'


def test_worker_decides_and_archives_wav_with_transcript():
    handled = []
    archive = MemoryArchive()
    state = run_worker(FakeTranscriber(['Włącz światło.', '']),
                       handle=lambda text: handled.append(text) or decide(text), archive=archive)

    assert handled == ['Włącz światło.', '']
    assert state.archived == 2 and state.failures == 0
    assert archive.closed
    stem, wav, meta = archive.saved[0]
    assert stem.endswith('-00000001')
    with wave.open(io.BytesIO(wav), 'rb') as source:
        assert source.readframes(source.getnframes()) == bytes(6400)
    assert meta['text'] == 'Włącz światło.'
    assert meta['command'] == 'lights_on'
    assert meta['language'] == 'pl' and meta['backend'] == 'fake'
    assert meta['duration_seconds'] == 0.2
    assert archive.saved[1][2]['reply'] == DEFAULT_REPLY


def test_archive_failure_does_not_block_command_or_count_as_failure():
    handled = []
    state = run_worker(FakeTranscriber(['Głośniej']),
                       handle=lambda text: handled.append(text) or decide(text),
                       archive=MemoryArchive(fail=True))
    assert handled == ['Głośniej']
    assert state.failures == 0 and state.transcribed == 1
    assert state.archive_failures == 1


def test_stt_failure_is_archived_with_error():
    archive = MemoryArchive()
    state = run_worker(FakeTranscriber([RuntimeError('timeout')]), archive=archive)
    assert state.failures == 1
    assert archive.saved[0][2]['text'] is None
    assert archive.saved[0][2]['error'] == 'timeout'


def test_smb_archive_writes_wav_then_json_sidecar():
    client = FakeSmbClient(port=4455)
    archive = SmbArchive('nas', 'data', 'open-lexa/raw', username='u', password='p', port=4455,
                         client=client)
    meta = {'created': '2026-11-02T10:00:00.000+00:00', 'text': 'Żółw'}
    archive.save('1-00000001', b'RIFF', meta)
    archive.save('2-00000002', b'RIFF', meta)

    base = r'\\nas\data\open-lexa\raw\2026-11-02'
    assert client.dirs == [base]
    assert len(client.sessions) == 1
    assert client.sessions[0][0] == 'nas'
    assert client.sessions[0][1]['username'] == 'u'
    assert list(client.files)[:2] == [base + r'\1-00000001.wav', base + r'\1-00000001.json']
    assert json.loads(client.files[base + r'\1-00000001.json']) == meta


def test_smb_archive_retries_once_with_fresh_connection_and_directory():
    client = FakeSmbClient(fail_writes=1)
    archive = SmbArchive('nas', 'data', client=client)
    meta = {'created': '2026-11-02T10:00:00.000+00:00'}
    archive.save('1-00000001', b'RIFF', meta)
    assert client.resets == 1
    assert len(client.sessions) == 2
    # The day directory is re-created in case it was deleted on the server.
    assert client.dirs == [r'\\nas\data\open-lexa\2026-11-02'] * 2
    assert r'\\nas\data\open-lexa\2026-11-02\1-00000001.json' in client.files


def test_smb_archive_gives_up_after_second_failure():
    client = FakeSmbClient(fail_writes=2)
    archive = SmbArchive('nas', 'data', client=client)
    with pytest.raises(ConnectionError):
        archive.save('1-00000001', b'RIFF', {'created': '2026-11-02T10:00:00.000+00:00'})
    archive.save('2-00000002', b'RIFF', {'created': '2026-11-02T10:00:00.000+00:00'})
    assert r'\\nas\data\open-lexa\2026-11-02\2-00000002.json' in client.files


def test_create_archive_from_env():
    assert create_archive({}) is None
    with pytest.raises(ConfigError):
        create_archive({'SMB_SERVER': 'nas'})
    archive = create_archive({'SMB_SERVER': 'nas', 'SMB_SHARE': 'data', 'SMB_PATH': 'lexa',
                              'SMB_USERNAME': 'u', 'SMB_PASSWORD': 'p'})
    assert archive.root == r'\\nas\data\lexa'
    assert (archive.username, archive.password, archive.timeout) == ('u', 'p', 10)


def test_stalled_archive_does_not_block_commands():
    release = threading.Event()

    class StalledArchive(MemoryArchive):
        def save(self, stem, wav, meta):
            release.wait(5)

    handled = []
    transcriber = FakeTranscriber(['Głośniej'] * 20)
    state = HealthState(transcriber.name)
    jobs = queue.Queue()
    thread = threading.Thread(target=worker, args=(jobs, transcriber, 'pl', None, state),
                              kwargs={'handle': lambda text: handled.append(text) or decide(text),
                                      'archive': StalledArchive()}, daemon=True)
    thread.start()
    for session in range(20):
        jobs.put(Recording(('127.0.0.1', 1234), session, bytes(64)))
    jobs.join()

    assert len(handled) == 20
    assert state.archive_failures >= 3  # queue of 16 + one in flight
    release.set()
    jobs.put(None)
    thread.join(timeout=5)
    assert not thread.is_alive()


class RoleTranscriber:
    def __init__(self, name, text=None, error=None, model=None):
        self.name, self.text, self.error, self.model = name, text, error, model
        self.closed = False

    def warmup(self):
        pass

    def transcribe(self, pcm, language):
        if self.error:
            raise self.error
        return self.text

    def close(self):
        self.closed = True


def test_worker_archives_both_transcripts_and_uses_gpt_for_the_command():
    archive = MemoryArchive()
    transcriber = DualTranscriber(RoleTranscriber('http', 'Włącz światło lokalnie', model='whisper-1'),
                                  RoleTranscriber('openai', 'Włącz światło.', model='gpt-transcribe'))
    state = HealthState(transcriber.name)
    jobs = queue.Queue()
    thread = threading.Thread(target=worker, args=(jobs, transcriber, 'pl', None, state),
                              kwargs={'archive': archive}, daemon=True)
    thread.start()
    jobs.put(Recording(('127.0.0.1', 1234), 1, bytes(6400)))
    jobs.put(None)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert state.transcribed == 1 and state.failures == 0
    meta = archive.saved[0][2]
    assert meta['text'] == 'Włącz światło.' and meta['command'] == 'lights_on'
    assert meta['corrected_text'] == 'Włącz światło.' and meta['corrected_source'] == 'cloud'
    assert meta['backend'] == 'dual' and meta['model'] == 'gpt-transcribe'
    assert meta['transcripts']['local']['text'] == 'Włącz światło lokalnie'
    assert meta['transcripts']['cloud']['text'] == 'Włącz światło.'


def test_worker_archives_dual_failure_with_per_backend_errors():
    archive = MemoryArchive()
    transcriber = DualTranscriber(RoleTranscriber('http', error=RuntimeError('local down')),
                                  RoleTranscriber('openai', error=RuntimeError('cloud down')))
    state = HealthState(transcriber.name)
    jobs = queue.Queue()
    thread = threading.Thread(target=worker, args=(jobs, transcriber, 'pl', None, state),
                              kwargs={'archive': archive}, daemon=True)
    thread.start()
    jobs.put(Recording(('127.0.0.1', 1234), 1, bytes(6400)))
    jobs.put(None)
    thread.join(timeout=5)

    assert state.transcribed == 0 and state.failures == 1
    meta = archive.saved[0][2]
    assert meta['text'] is None
    assert 'local down' in meta['error'] and 'cloud down' in meta['error']
    assert meta['transcripts']['local']['error'] == 'local down'
    assert meta['transcripts']['cloud']['error'] == 'cloud down'
