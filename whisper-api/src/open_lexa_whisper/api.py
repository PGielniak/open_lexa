"""OpenAI-compatible HTTP API for a Whisper model owned by one process.

The UDP receiver sends canonical WAV data to ``POST /v1/audio/transcriptions``.
This process owns the Hailo device and performs every HailoRT call on one worker
thread, which is required because HailoRT device handles are thread-affine.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
import hmac
import io
import json
import logging
import os
import queue
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.parse import urlsplit
import wave

from dotenv import load_dotenv

from . import MAX_BYTES, RATE
from .hailo import ConfigError, create_transcriber

LOG = logging.getLogger('open_lexa.whisper_api')
DEFAULT_MAX_UPLOAD_BYTES = MAX_BYTES + 4096


class Transcriber(Protocol):
    """The model owned by the inference thread (see :mod:`open_lexa_whisper.hailo`)."""

    name: str

    def warmup(self) -> None: ...

    def transcribe(self, pcm: bytes, language: str) -> str: ...

    def close(self) -> None: ...


class APIRequestError(ValueError):
    """A client request that cannot be sent to the transcription backend."""

    def __init__(self, message: str, *, status: int = 400, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.param = param


class APIUnavailableError(RuntimeError):
    """The model is still starting, busy, or unavailable."""

    def __init__(self, message: str, *, status: int = 503):
        super().__init__(message)
        self.status = status


@dataclass
class APIState:
    """Thread-safe health and operational state exposed by the API."""

    backend: str
    model: str
    started: float = field(default_factory=time.time)
    ready: bool = False
    transcriptions: int = 0
    failures: int = 0
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _ready_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def mark_ready(self) -> None:
        with self._lock:
            self.ready = True
            self.last_error = None
            self._ready_event.set()

    def mark_unavailable(self, error: Exception | str | None = None) -> None:
        with self._lock:
            self.ready = False
            if error:
                self.last_error = str(error)
            self._ready_event.clear()

    def record_success(self) -> None:
        with self._lock:
            self.transcriptions += 1
            self.last_error = None

    def record_failure(self, error: Exception | str) -> None:
        with self._lock:
            self.failures += 1
            self.last_error = str(error)

    def is_ready(self) -> bool:
        with self._lock:
            return self.ready

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready_event.wait(timeout)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                'status': 'ok' if self.ready else 'starting',
                'backend': self.backend,
                'model': self.model,
                'ready': self.ready,
                'uptime_seconds': round(time.time() - self.started, 1),
                'transcriptions': self.transcriptions,
                'failures': self.failures,
                'last_error': self.last_error,
            }


@dataclass
class _Job:
    pcm: bytes
    language: str
    done: threading.Event = field(default_factory=threading.Event)
    text: str | None = None
    error: Exception | None = None


class WhisperAPI:
    """Own one transcriber in one worker thread and expose synchronous jobs."""

    def __init__(self, transcriber: Transcriber, *, model: str = 'whisper-1',
                 default_language: str = 'pl', api_key: str | None = None,
                 queue_size: int = 4, request_timeout: float = 45.0):
        if queue_size < 1:
            raise ValueError('queue_size must be at least one')
        if request_timeout <= 0:
            raise ValueError('request_timeout must be positive')
        self.transcriber = transcriber
        self.model = model
        self.default_language = default_language
        self.api_key = api_key or None
        self.request_timeout = request_timeout
        self.state = APIState(transcriber.name, model)
        self._jobs: queue.Queue[_Job | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name='whisper-inference', daemon=True)
        self._close_lock = threading.Lock()
        self._started = False
        self._closed = False

    def start(self) -> None:
        with self._close_lock:
            if self._closed:
                raise RuntimeError('Whisper API has already stopped')
            if not self._started:
                self._started = True
                self._thread.start()

    def _run(self) -> None:
        try:
            try:
                # Do this in the worker, not an HTTP request thread: HailoRT
                # requires device creation and inference to use the same thread.
                self.transcriber.warmup()
            except Exception as exc:
                self.state.mark_unavailable(exc)
                LOG.exception('Whisper backend initialization failed')
                return

            self.state.mark_ready()
            LOG.info('Whisper API backend ready: %s', self.transcriber.name)
            while True:
                job = self._jobs.get()
                try:
                    if job is None:
                        return
                    try:
                        job.text = self.transcriber.transcribe(job.pcm, job.language)
                        self.state.record_success()
                    except Exception as exc:
                        job.error = exc
                        self.state.record_failure(exc)
                        LOG.exception('Whisper transcription failed')
                    finally:
                        job.done.set()
                finally:
                    self._jobs.task_done()
        finally:
            self.state.mark_unavailable()
            try:
                self.transcriber.close()
            except Exception:
                LOG.exception('Whisper backend shutdown failed')

    def authorised(self, header: str | None) -> bool:
        """Accept all callers unless a dedicated API key was configured."""
        if not self.api_key:
            return True
        scheme, _, supplied = (header or '').partition(' ')
        return scheme.lower() == 'bearer' and hmac.compare_digest(supplied, self.api_key)

    def transcribe(self, pcm: bytes, language: str | None = None) -> str:
        if not self.state.is_ready():
            detail = self.state.snapshot()['last_error'] or 'model is still starting'
            raise APIUnavailableError(str(detail))
        job = _Job(pcm, (language or self.default_language or 'auto').strip() or 'auto')
        try:
            self._jobs.put_nowait(job)
        except queue.Full as exc:
            raise APIUnavailableError('transcription queue is full') from exc
        if not job.done.wait(self.request_timeout):
            raise APIUnavailableError('transcription timed out', status=504)
        if job.error is not None:
            raise job.error
        return (job.text or '').strip()

    def close(self, timeout: float = 20.0) -> None:
        """Drain accepted work, then release the device on its owner thread."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._started:
                try:
                    self._jobs.put(None, timeout=timeout)
                except queue.Full:
                    LOG.warning('Whisper queue did not drain before shutdown')
                self._thread.join(timeout=timeout)
                if self._thread.is_alive():
                    LOG.warning('Whisper inference thread did not stop before shutdown')


def _read_multipart(handler: BaseHTTPRequestHandler, max_bytes: int) -> tuple[dict[str, list[str]], bytes]:
    content_type = handler.headers.get('Content-Type', '')
    if not content_type.lower().startswith('multipart/form-data'):
        raise APIRequestError('Content-Type must be multipart/form-data', param='Content-Type')
    try:
        length = int(handler.headers.get('Content-Length', ''))
    except ValueError as exc:
        raise APIRequestError('Content-Length must be an integer', param='Content-Length') from exc
    if length < 1:
        raise APIRequestError('Content-Length is required', status=411, param='Content-Length')
    if length > max_bytes:
        raise APIRequestError(f'upload exceeds {max_bytes} bytes', status=413, param='file')
    body = handler.rfile.read(length)
    if len(body) != length:
        raise APIRequestError('incomplete request body')

    # The email parser handles quoted MIME boundaries and binary file parts;
    # constructing a minimal MIME document avoids the removed ``cgi`` module.
    raw = (f'Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n'.encode('ascii') + body)
    message = BytesParser(policy=policy.default).parsebytes(raw)
    if not message.is_multipart():
        raise APIRequestError('invalid multipart/form-data body', param='file')

    fields: dict[str, list[str]] = {}
    audio: bytes | None = None
    for part in message.iter_parts():
        if part.get_content_disposition() != 'form-data':
            continue
        name = part.get_param('name', header='content-disposition')
        if not name:
            continue
        payload = part.get_payload(decode=True) or b''
        if name == 'file':
            if audio is not None:
                raise APIRequestError('only one file field is supported', param='file')
            audio = payload
        else:
            fields.setdefault(name, []).append(payload.decode('utf-8', errors='replace'))
    if audio is None:
        raise APIRequestError('missing file field', param='file')
    return fields, audio


def _form_value(fields: dict[str, list[str]], *names: str) -> str | None:
    for name in names:
        values = fields.get(name)
        if values:
            return values[-1].strip()
    return None


def _request_language(fields: dict[str, list[str]], default: str) -> str:
    language = _form_value(fields, 'language', 'languages[]', 'languages')
    if language and language.startswith('['):
        try:
            listed = json.loads(language)
            if isinstance(listed, list) and listed:
                language = str(listed[0])
        except json.JSONDecodeError:
            pass
    return language or default


def _pcm_from_wav(data: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(data), 'rb') as source:
            format_ok = (
                source.getnchannels() == 1
                and source.getsampwidth() == 2
                and source.getframerate() == RATE
                and source.getcomptype() == 'NONE'
            )
            pcm = source.readframes(source.getnframes())
    except (EOFError, OSError, wave.Error) as exc:
        raise APIRequestError('file must be a valid WAV file', param='file') from exc
    if not format_ok:
        raise APIRequestError('WAV must be mono PCM16LE at 16000 Hz', param='file')
    if not pcm:
        raise APIRequestError('WAV contains no audio samples', param='file')
    if len(pcm) > MAX_BYTES:
        raise APIRequestError('WAV exceeds the 12 second receiver limit', status=413, param='file')
    return pcm


class WhisperRequestHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI audio transcription endpoint plus Kubernetes probes."""

    server: 'WhisperHTTPServer'
    protocol_version = 'HTTP/1.1'

    @property
    def api(self) -> WhisperAPI:
        return self.server.api

    def _send(self, status: int, body: bytes, content_type: str = 'application/json', *,
              headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header('Content-Type', f'{content_type}; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, body: dict[str, Any], *, headers: dict[str, str] | None = None) -> None:
        self._send(status, json.dumps(body, ensure_ascii=False).encode(), headers=headers)

    def _error(self, status: int, message: str, *, param: str | None = None,
               headers: dict[str, str] | None = None) -> None:
        self._json(status, {
            'error': {
                'message': message,
                'type': 'invalid_request_error' if status < 500 else 'server_error',
                'param': param,
                'code': None,
            }
        }, headers=headers)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        if path in ('/healthz', '/readyz'):
            ready = self.api.state.is_ready()
            self._json(200 if path == '/healthz' or ready else 503, self.api.state.snapshot())
            return
        if path == '/v1/models':
            self._json(200, {'object': 'list', 'data': [
                {'id': self.api.model, 'object': 'model', 'owned_by': 'open-lexa'},
            ]})
            return
        self._error(404, 'not found')

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlsplit(self.path).path != '/v1/audio/transcriptions':
            self._error(404, 'not found')
            return
        if not self.api.authorised(self.headers.get('Authorization')):
            self._error(401, 'invalid API key', headers={'WWW-Authenticate': 'Bearer'})
            return
        try:
            fields, wav = _read_multipart(self, DEFAULT_MAX_UPLOAD_BYTES)
            pcm = _pcm_from_wav(wav)
            text = self.api.transcribe(pcm, _request_language(fields, self.api.default_language))
        except APIRequestError as exc:
            self._error(exc.status, str(exc), param=exc.param)
            return
        except APIUnavailableError as exc:
            self._error(exc.status, str(exc))
            return
        except Exception as exc:
            LOG.exception('Unhandled transcription request failure')
            self._error(500, 'transcription failed')
            return

        response_format = (_form_value(fields, 'response_format') or 'json').lower()
        if response_format == 'text':
            self._send(200, text.encode(), 'text/plain')
        else:
            self._json(200, {'text': text})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        LOG.debug('%s - %s', self.address_string(), format % args)


class WhisperHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], api: WhisperAPI):
        super().__init__(address, WhisperRequestHandler)
        self.api = api


def create_server(api: WhisperAPI, bind: str = '0.0.0.0', port: int = 8080) -> WhisperHTTPServer:
    """Start the inference owner and return a server ready for ``serve_forever``."""
    api.start()
    return WhisperHTTPServer((bind, port), api)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Expose one Hailo Whisper model as an OpenAI-compatible transcription API.')
    parser.add_argument('--env-file', default='.env')
    parser.add_argument('--bind', help='HTTP bind address (default: WHISPER_API_BIND or 0.0.0.0)')
    parser.add_argument('--port', type=int, help='HTTP port (default: WHISPER_API_PORT or 8080)')
    parser.add_argument('--model', help='Advertised model name (default: WHISPER_API_MODEL or whisper-1)')
    parser.add_argument('--language', help='Fallback language (default: WHISPER_API_LANGUAGE or pl)')
    return parser


def _env_int(parser: argparse.ArgumentParser, name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        parser.error(f'{name} must be an integer')
        raise AssertionError('parser.error does not return')


def _env_float(parser: argparse.ArgumentParser, name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        parser.error(f'{name} must be numeric')
        raise AssertionError('parser.error does not return')


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv(args.env_file)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    bind = args.bind or os.environ.get('WHISPER_API_BIND', '0.0.0.0')
    port = args.port if args.port is not None else _env_int(parser, 'WHISPER_API_PORT', 8080)
    queue_size = _env_int(parser, 'WHISPER_API_QUEUE_SIZE', 4)
    request_timeout = _env_float(parser, 'WHISPER_API_REQUEST_TIMEOUT_SECONDS', 45.0)
    if not 1 <= port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if queue_size < 1:
        parser.error('WHISPER_API_QUEUE_SIZE must be at least one')
    if request_timeout <= 0:
        parser.error('WHISPER_API_REQUEST_TIMEOUT_SECONDS must be positive')

    backend = os.environ.get('WHISPER_API_BACKEND', 'hailo').strip().lower()
    if backend != 'hailo':
        parser.error('WHISPER_API_BACKEND must be hailo')
    try:
        transcriber = create_transcriber()
    except ConfigError as exc:
        parser.error(str(exc))

    api = WhisperAPI(
        transcriber,
        model=args.model or os.environ.get('WHISPER_API_MODEL', 'whisper-1'),
        default_language=args.language or os.environ.get('WHISPER_API_LANGUAGE', 'pl'),
        api_key=os.environ.get('WHISPER_API_KEY') or None,
        queue_size=queue_size,
        request_timeout=request_timeout,
    )
    server = create_server(api, bind, port)

    def stop(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    LOG.info('Whisper API listening on %s:%d; backend=%s; model=%s',
             bind, port, transcriber.name, api.model)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOG.info('Stopping Whisper API')
    finally:
        server.server_close()
        api.close()


if __name__ == '__main__':
    main()
