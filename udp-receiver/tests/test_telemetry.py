import queue
import threading
import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from open_lexa import server, stt
from open_lexa.protocol import Recording
from open_lexa_whisper import api as whisper


class Backend:
    name = 'fake'
    def warmup(self): pass
    def close(self): pass
    def transcribe(self, pcm, language): return 'test'


def test_trace_crosses_receiver_queue_and_stt_threads(monkeypatch, caplog):
    import logging
    from open_lexa.telemetry import TraceLogFilter
    caplog.handler.addFilter(TraceLogFilter())
    caplog.set_level(logging.INFO, logger='open_lexa')
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('test')
    monkeypatch.setattr(server, 'tracer', tracer)
    monkeypatch.setattr(stt, 'TRACER', tracer)
    jobs = queue.Queue()
    backend = stt.DualTranscriber(Backend(), Backend())
    state = server.HealthState('dual')
    thread = threading.Thread(target=server.worker, args=(jobs, backend, 'pl', None, state))
    thread.start()
    now = time.monotonic()
    rec = Recording(('127.0.0.1', 1234), 1, bytes(6400), now - 0.2, now - 0.1)
    span, received_ns = server.start_utterance(rec)
    jobs.put((rec, span, received_ns))
    jobs.put(None)
    thread.join(timeout=5)
    assert not thread.is_alive()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert {'utterance', 'receive_audio', 'receiver_queue_wait', 'stt', 'stt.local', 'stt.cloud', 'llm'} <= spans.keys()
    assert spans['stt.local'].parent.span_id == spans['stt'].context.span_id
    assert spans['stt.cloud'].parent.span_id == spans['stt'].context.span_id
    assert spans['receive_audio'].parent.span_id == spans['utterance'].context.span_id
    assert len({span.context.trace_id for span in spans.values()}) == 1
    received_log = next(r for r in caplog.records if 'Received 0.20 s' in r.getMessage())
    assert received_log.trace_id == format(spans['utterance'].context.trace_id, '032x')
    assert received_log.span_id == format(spans['receive_audio'].context.span_id, '016x')
    provider.shutdown()


def test_whisper_propagates_context_to_inference_thread(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('test')
    monkeypatch.setattr(whisper, 'tracer', tracer)
    service = whisper.WhisperAPI(Backend())
    service.start()
    assert service.state.wait_ready(2)
    with tracer.start_as_current_span('request'):
        assert service.transcribe(bytes(6400)) == 'test'
    service.close()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans['whisper.inference'].parent.span_id == spans['request'].context.span_id
    assert spans['whisper.transcribe'].parent.span_id == spans['whisper.inference'].context.span_id
    provider.shutdown()


def test_local_http_propagates_trace_without_cloud_headers(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('test')
    monkeypatch.setattr(whisper, 'tracer', tracer)
    service = whisper.WhisperAPI(Backend())
    httpd = whisper.create_server(service, '127.0.0.1', 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    assert service.state.wait_ready(2)
    client = stt.OpenAICompatibleTranscriber(f'http://127.0.0.1:{httpd.server_port}/v1', 'whisper-1')
    try:
        with tracer.start_as_current_span('client'):
            assert client.transcribe(bytes(6400), 'pl') == 'test'
    finally:
        client.close()
        httpd.shutdown()
        httpd.server_close()
        service.close()
        thread.join(timeout=2)
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans['whisper.http'].parent.span_id == spans['client'].context.span_id
    assert spans['whisper.inference'].parent.span_id == spans['whisper.http'].context.span_id
    provider.shutdown()
