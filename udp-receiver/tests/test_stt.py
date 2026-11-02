import pytest

from open_lexa.stt import (ConfigError, DualTranscriber, LocalOpenAITranscriber,
                           OpenAITranscriber, create_transcriber, preferred_result,
                           transcribe_all)


class FakeTranscriber:
    def __init__(self, name, text=None, error=None):
        self.name = name
        self.text = text
        self.error = error
        self.calls = []
        self.closed = False

    def warmup(self):
        pass

    def transcribe(self, pcm, language):
        self.calls.append((pcm, language))
        if self.error:
            raise self.error
        return self.text

    def close(self):
        self.closed = True


def test_openai_backend_requires_an_api_key():
    with pytest.raises(ConfigError, match='OPENAI_API_KEY'):
        create_transcriber('openai', env={})


def test_local_backend_requires_a_url_and_uses_compatible_defaults():
    with pytest.raises(ConfigError, match='STT_HTTP_URL'):
        create_transcriber('local', env={})
    transcriber = create_transcriber(
        'local', env={'STT_HTTP_URL': 'http://127.0.0.1:8080/v1'})
    assert isinstance(transcriber, LocalOpenAITranscriber)
    assert transcriber.model == 'whisper-1'


def test_unknown_backend_is_rejected():
    with pytest.raises(ConfigError, match='Unknown STT_BACKEND'):
        create_transcriber('telepathy', env={})


def test_hailo_is_not_a_receiver_backend():
    # Hailo runs only in the separate Whisper API; the receiver reaches it over HTTP.
    with pytest.raises(ConfigError, match='Unknown STT_BACKEND'):
        create_transcriber('hailo', env={})


def test_dual_backend_requires_both_endpoints():
    with pytest.raises(ConfigError, match='STT_HTTP_URL'):
        create_transcriber('dual', env={'OPENAI_API_KEY': 'sk-x'})
    with pytest.raises(ConfigError, match='OPENAI_API_KEY'):
        create_transcriber('dual', env={'STT_HTTP_URL': 'http://whisper:8080/v1'})


def test_dual_backend_builds_local_and_cloud_clients():
    transcriber = create_transcriber('dual', env={
        'STT_HTTP_URL': 'http://whisper:8080/v1',
        'STT_HTTP_MODEL': 'whisper-small',
        'OPENAI_API_KEY': 'sk-x',
        'OPENAI_BASE_URL': 'https://api.example.com/v1',
        'OPENAI_STT_MODEL': 'gpt-transcribe',
    })
    try:
        assert isinstance(transcriber, DualTranscriber)
        assert transcriber.name == 'dual'
        assert isinstance(transcriber.local, LocalOpenAITranscriber)
        assert transcriber.local.model == 'whisper-small'
        assert isinstance(transcriber.cloud, OpenAITranscriber)
        assert transcriber.cloud.model == 'gpt-transcribe'
    finally:
        transcriber.close()


def test_dual_is_automatic_when_both_endpoints_are_configured():
    assert isinstance(create_transcriber(None, env={'OPENAI_API_KEY': 'sk-x'}), OpenAITranscriber)
    assert isinstance(create_transcriber(None, env={
        'OPENAI_API_KEY': 'sk-x', 'STT_HTTP_URL': 'http://whisper:8080/v1'}), DualTranscriber)


def test_dual_prefers_the_cloud_transcript():
    local = FakeTranscriber('http', 'lokalna wersja')
    cloud = FakeTranscriber('openai', 'poprawna wersja')
    transcriber = DualTranscriber(local, cloud)
    try:
        results = transcribe_all(transcriber, b'pcm', 'pl')
        assert [(r.role, r.text) for r in results] == [
            ('local', 'lokalna wersja'), ('cloud', 'poprawna wersja')]
        assert preferred_result(results).role == 'cloud'
        assert local.calls == [(b'pcm', 'pl')] and cloud.calls == [(b'pcm', 'pl')]
        assert transcriber.transcribe(b'pcm', 'pl') == 'poprawna wersja'
        assert len(local.calls) == 2 and len(cloud.calls) == 2
    finally:
        transcriber.close()
    assert local.closed and cloud.closed


def test_dual_falls_back_to_local_when_cloud_fails():
    local = FakeTranscriber('http', 'lokalna wersja')
    cloud = FakeTranscriber('openai', error=RuntimeError('cloud down'))
    transcriber = DualTranscriber(local, cloud)
    try:
        results = transcribe_all(transcriber, b'pcm', 'pl')
        assert results[1].error == 'cloud down'
        assert preferred_result(results).role == 'local'
        assert transcriber.transcribe(b'pcm', 'pl') == 'lokalna wersja'
    finally:
        transcriber.close()


def test_dual_keeps_local_when_cloud_returns_empty_text():
    local = FakeTranscriber('http', 'lokalna wersja')
    cloud = FakeTranscriber('openai', '')
    transcriber = DualTranscriber(local, cloud)
    try:
        assert transcriber.transcribe(b'pcm', 'pl') == 'lokalna wersja'
    finally:
        transcriber.close()
