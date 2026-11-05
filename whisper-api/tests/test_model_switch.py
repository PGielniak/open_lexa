"""Multi-model routing is exercised over HTTP without a Hailo device."""
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

from open_lexa.stt import OpenAICompatibleTranscriber
from open_lexa_whisper.api import WhisperAPI, create_server


class Models:
    name = 'fake-hailo'
    models = ('whisper-base', 'whisper-small')
    model = 'whisper-base'

    def __init__(self):
        self.calls = []

    def warmup(self):
        pass

    def select_model(self, model):
        self.model = model

    def transcribe(self, pcm, language):
        self.calls.append((self.model, language, threading.get_ident()))
        return self.model

    def close(self):
        pass


def test_model_switch_and_legacy_alias():
    transcriber = Models()
    api = WhisperAPI(transcriber, model='whisper-base')
    server = create_server(api, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    assert api.state.wait_ready(2)
    address = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(f'{address}/v1/models') as response:
            data = json.load(response)
        assert [item['id'] for item in data['data']] == list(transcriber.models)
        for model, expected in [('whisper-small', 'whisper-small'),
                                ('whisper-base', 'whisper-base'),
                                ('whisper-1', 'whisper-base')]:
            client = OpenAICompatibleTranscriber(f'{address}/v1', model)
            assert client.transcribe(bytes(3200), 'pl') == expected
            client.close()
        assert len({call[2] for call in transcriber.calls}) == 1
        assert api.state.snapshot()['model'] == 'whisper-base'
        client = OpenAICompatibleTranscriber(f'{address}/v1', 'nonexistent')
        try:
            try:
                client.transcribe(bytes(3200), 'pl')
                assert False, 'unknown model should be rejected'
            except Exception as exc:
                assert '404' in str(exc)
        finally:
            client.close()
    finally:
        server.shutdown()
        server.server_close()
        api.close()
        thread.join(timeout=2)
