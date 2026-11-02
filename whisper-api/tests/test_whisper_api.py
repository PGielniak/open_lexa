from __future__ import annotations

from contextlib import contextmanager
import queue
import threading

from open_lexa.protocol import Recording
from open_lexa.server import HealthState, worker
from open_lexa.stt import OpenAITranscriber, create_transcriber
from open_lexa_whisper.api import WhisperAPI, create_server


class FakeHailoTranscriber:
    name = 'hailo'

    def __init__(self, result='Włącz światło'):
        self.result = result
        self.calls = []
        self.warmed_up = False
        self.closed = False

    def warmup(self):
        self.warmed_up = True

    def transcribe(self, pcm, language):
        self.calls.append((pcm, language))
        return self.result

    def close(self):
        self.closed = True


@contextmanager
def running_whisper_api(*, api_key=None):
    transcriber = FakeHailoTranscriber()
    api = WhisperAPI(transcriber, api_key=api_key)
    server = create_server(api, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    assert api.state.wait_ready(2), api.state.snapshot()
    try:
        yield api, transcriber, f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        api.close()
        thread.join(timeout=2)


def test_receiver_local_backend_calls_the_standalone_whisper_api():
    pcm = b'\x01\x00\xff\xff' * 1600
    with running_whisper_api() as (api, hailo, address):
        receiver_backend = create_transcriber('local', env={
            'STT_HTTP_URL': f'{address}/v1',
            'STT_HTTP_MODEL': 'whisper-1',
        })
        assert receiver_backend.transcribe(pcm, 'pl') == 'Włącz światło'

        assert hailo.warmed_up
        assert hailo.calls == [(pcm, 'pl')]
        assert api.state.snapshot()['transcriptions'] == 1


def test_receiver_worker_transcribes_through_the_standalone_whisper_api():
    pcm = b'\x04\x00' * 1600
    with running_whisper_api() as (_, hailo, address):
        receiver_backend = create_transcriber('http', env={
            'STT_HTTP_URL': f'{address}/v1',
            'STT_HTTP_MODEL': 'whisper-1',
        })
        state = HealthState(receiver_backend.name)
        jobs = queue.Queue()
        thread = threading.Thread(
            target=worker, args=(jobs, receiver_backend, 'pl', None, state), daemon=True)
        thread.start()
        jobs.put(Recording(('127.0.0.1', 12345), 1, pcm))
        jobs.put(None)
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert state.ready and state.transcribed == 1 and state.failures == 0
        assert hailo.calls == [(pcm, 'pl')]


def test_receiver_http_backend_uses_cloud_style_url_and_api_key():
    pcm = b'\x02\x00' * 1600
    with running_whisper_api(api_key='cloud-secret') as (_, hailo, address):
        receiver_backend = create_transcriber('http', env={
            'STT_HTTP_URL': f'{address}/v1',
            'STT_HTTP_MODEL': 'cloud-whisper-v1',
            'STT_HTTP_API_KEY': 'cloud-secret',
        })
        assert receiver_backend.model == 'cloud-whisper-v1'
        assert receiver_backend.transcribe(pcm, 'pl') == 'Włącz światło'
        # Success proves that the generic HTTP mode sent the Bearer API key.
        assert hailo.calls == [(pcm, 'pl')]


def test_api_accepts_openai_gpt_transcribe_languages_list():
    pcm = b'\x03\x00' * 1600
    with running_whisper_api() as (_, hailo, address):
        cloud_backend = OpenAITranscriber(
            'gpt-transcribe', api_key='not-needed', base_url=f'{address}/v1')
        assert cloud_backend.transcribe(pcm, 'pl') == 'Włącz światło'
        assert hailo.calls == [(pcm, 'pl')]
