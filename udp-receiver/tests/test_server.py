import io
import queue
import threading
import wave

import pytest

from open_lexa.audio import wav_bytes
from open_lexa.protocol import Recording
from open_lexa.server import HealthState, worker
from open_lexa.stt import OpenAITranscriber


class FakeTranscriber:
    name = 'fake'

    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.warmed_up = False
        self.closed = False

    def warmup(self):
        self.warmed_up = True

    def transcribe(self, pcm, language):
        self.calls.append((pcm, language))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, text):
        self.audio = type('Audio', (), {})()
        self.audio.transcriptions = type('Transcriptions', (), {})()
        self.audio.transcriptions.create = self.create
        self.text = text

    def create(self, **kwargs):
        self.kwargs = kwargs
        return type('Result', (), {'text': self.text})()


def test_wav_contains_correct_format_and_samples():
    pcm = b'\x01\x00\xff\xff' * 1600
    with wave.open(io.BytesIO(wav_bytes(pcm)), 'rb') as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
        assert source.readframes(source.getnframes()) == pcm


def test_transcription_request_uses_wav_and_language_hint():
    client = FakeClient('  Włącz światło.  ')
    transcriber = OpenAITranscriber('gpt-transcribe', client=client)
    assert transcriber.transcribe(b'\x01\x00' * 800, 'pl') == 'Włącz światło.'
    assert client.kwargs['model'] == 'gpt-transcribe'
    assert client.kwargs['file'][0] == 'command.wav'
    assert client.kwargs['file'][1].startswith(b'RIFF')
    assert client.kwargs['response_format'] == 'json'
    assert client.kwargs['extra_body'] == {'languages': ['pl']}


def test_worker_survives_backend_failure_and_processes_next_recording(capsys):
    transcriber = FakeTranscriber([RuntimeError('network unavailable'), 'Działa'])
    state = HealthState(transcriber.name)
    jobs = queue.Queue()
    thread = threading.Thread(target=worker, args=(jobs, transcriber, 'pl', None, state), daemon=True)
    thread.start()
    for session in (1, 2):
        jobs.put(Recording(('127.0.0.1', 1234), session, bytes(6400)))
    jobs.put(None)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert transcriber.warmed_up
    assert transcriber.closed
    assert len(transcriber.calls) == 2
    assert state.transcribed == 1
    assert state.failures == 1
    assert 'Działa' in capsys.readouterr().out


def test_dry_run_worker_writes_wav_without_backend(tmp_path, capsys):
    jobs = queue.Queue()
    state = HealthState('dry-run', ready=True)
    thread = threading.Thread(target=worker, args=(jobs, None, 'pl', tmp_path, state), daemon=True)
    thread.start()
    jobs.put(Recording(('127.0.0.1', 1234), 7, bytes(6400)))
    jobs.put(None)
    thread.join(timeout=5)

    files = list(tmp_path.glob('*.wav'))
    assert len(files) == 1
    with wave.open(str(files[0]), 'rb') as result:
        assert result.readframes(result.getnframes()) == bytes(6400)
    output = capsys.readouterr().out
    assert '0.20 s' in output
    assert '6400 bytes PCM' in output


class FakeSettings:
    """SettingsWatcher stand-in: ``version`` bumps select the next build result."""

    def __init__(self):
        self.version = 0
        self.next = None

    def build(self):
        if isinstance(self.next, Exception):
            raise self.next
        return self.next


def test_worker_switches_backend_when_settings_change(caplog):
    env = FakeTranscriber(['z env'])
    admin = FakeTranscriber(['z admina'])
    settings = FakeSettings()
    state = HealthState(env.name)
    jobs = queue.Queue()
    thread = threading.Thread(target=worker, args=(jobs, env, 'pl', None, state),
                              kwargs={'settings': settings}, daemon=True)
    thread.start()
    jobs.put(Recording(('127.0.0.1', 1234), 1, bytes(640)))
    jobs.join()
    settings.next, settings.version = (admin, 'en'), 1
    jobs.put(Recording(('127.0.0.1', 1234), 2, bytes(640)))
    jobs.join()
    # A broken settings file keeps the working backend.
    settings.next, settings.version = ValueError('invalid _config/stt.json'), 2
    admin.results.append('nadal admin')
    jobs.put(Recording(('127.0.0.1', 1234), 3, bytes(640)))
    jobs.put(None)
    thread.join(timeout=5)

    assert env.calls == [(bytes(640), 'pl')] and env.closed
    assert [language for _, language in admin.calls] == ['en', 'en'] and admin.warmed_up
    assert state.transcribed == 3 and 'New STT settings rejected' in caplog.text


def test_settings_watcher_builds_from_file_and_falls_back_to_env(tmp_path):
    from open_lexa.store import DirStore
    from open_lexa.settings import SETTINGS_PATH, SettingsWatcher

    store = DirStore(tmp_path)
    env = FakeTranscriber([])
    watcher = SettingsWatcher(store, lambda: env)
    watcher.poll()
    assert watcher.version == 0 and watcher.build() == (env, None)
    store.write_atomic(SETTINGS_PATH, b'{"mode": "cloud", "cloud_url": "https://api.example.com/v1", '
                                      b'"cloud_api_key": "sk-x", "language": "de"}')
    watcher.poll()
    watcher.poll()
    transcriber, language = watcher.build()
    assert watcher.version == 1 and language == 'de'
    assert str(transcriber._client.base_url).startswith('https://api.example.com/v1')
    assert transcriber._client.api_key == 'sk-x' and transcriber.model == 'gpt-transcribe'
    store.write_atomic(SETTINGS_PATH, b'{"mode": "cloud"}')
    watcher.poll()
    with pytest.raises(ValueError, match='API key'):
        watcher.build()
    (tmp_path / SETTINGS_PATH).unlink()
    watcher.poll()
    assert watcher.version == 3 and watcher.build() == (env, None)


def test_settings_watcher_builds_dual_from_the_published_file(tmp_path):
    from open_lexa.settings import SETTINGS_PATH, SettingsWatcher
    from open_lexa.stt import DualTranscriber
    from open_lexa.store import DirStore

    store = DirStore(tmp_path)
    watcher = SettingsWatcher(store, lambda: FakeTranscriber([]))
    store.write_atomic(SETTINGS_PATH, b'{"mode": "dual", '
                                      b'"local_url": "http://open-lexa-whisper:8080/v1", '
                                      b'"local_model": "whisper-1", '
                                      b'"cloud_url": "https://api.example.com/v1", '
                                      b'"cloud_model": "gpt-transcribe", '
                                      b'"cloud_api_key": "sk-dual", "language": "pl"}')
    watcher.poll()
    transcriber, language = watcher.build()
    try:
        assert isinstance(transcriber, DualTranscriber)
        assert transcriber.local.model == 'whisper-1'
        assert transcriber.cloud.model == 'gpt-transcribe'
        assert language == 'pl'
    finally:
        transcriber.close()
