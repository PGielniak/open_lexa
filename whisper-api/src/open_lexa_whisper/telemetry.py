"""Optional asynchronous tracing for the Whisper service."""
import logging
import os
from opentelemetry import trace

tracer = trace.get_tracer('open_lexa.whisper')


class TraceLogFilter(logging.Filter):
    def filter(self, record):
        ctx = trace.get_current_span().get_span_context()
        record.trace_id = format(ctx.trace_id, '032x') if ctx.is_valid else '-'
        record.span_id = format(ctx.span_id, '016x') if ctx.is_valid else '-'
        return True


def configure():
    if not os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT'):
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider = TracerProvider(resource=Resource.create({'service.name': 'open-lexa-whisper'}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(timeout=2)))
        trace.set_tracer_provider(provider)
        return provider
    except Exception:
        logging.getLogger(__name__).exception('Tracing unavailable; continuing without telemetry')
        return None
