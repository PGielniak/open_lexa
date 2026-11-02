from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import queue
import signal
import socket
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from .store import create_store
from .archive import Archive, create_archive
from .audio import wav_bytes
from .commands import Decision, decide
from .protocol import CHUNK, HEADER, RATE, Receiver, Recording, parse_token
from .settings import SettingsWatcher
from .stt import (BACKENDS, ConfigError, Transcriber, create_transcriber, format_errors,
                  preferred_result, transcribe_all, transcripts_meta)

LOG = logging.getLogger('open_lexa')


@dataclass
class HealthState:
    backend: str
    started: float = field(default_factory=time.time)
    ready: bool = False
    received: int = 0
    transcribed: int = 0
    failures: int = 0
    archived: int = 0
    archive_failures: int = 0
    last_recording: float | None = None
    last_error: str | None = None

    def snapshot(self) -> dict:
        return {
            'status': 'ok' if self.ready else 'starting',
            'backend': self.backend,
            'ready': self.ready,
            'uptime_seconds': round(time.time() - self.started, 1),
            'recordings_received': self.received,
            'transcriptions': self.transcribed,
            'failures': self.failures,
            'archived': self.archived,
            'archive_failures': self.archive_failures,
            'last_recording': self.last_recording,
            'last_error': self.last_error,
        }


def start_health_server(port: int, state: HealthState) -> ThreadingHTTPServer:
    """Serve /healthz (process alive) and /readyz (STT backend initialised)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path not in ('/healthz', '/readyz'):
                self.send_error(404)
                return
            ok = True if path == '/healthz' else state.ready
            body = json.dumps(state.snapshot()).encode() + b'\n'
            self.send_response(200 if ok else 503)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            pass

    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name='health', daemon=True).start()
    return server


class ArchiveWriter:
    """Archive on a separate thread: smbprotocol waits for responses without a timeout,
    so a share that vanishes mid-write must not stall transcription and commands."""

    def __init__(self, archive: Archive, state: HealthState, maxsize: int = 16):
        self.archive = archive
        self.state = state
        self.jobs: queue.Queue = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._run, name='archive', daemon=True)
        self.thread.start()

    def submit(self, stem: str, recording: Recording, **meta) -> None:
        item = (stem, wav_bytes(recording.pcm), {
            'created': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            'session': f'{recording.session:08x}',
            'peer': recording.peer[0],
            'duration_seconds': round(len(recording.pcm) / (RATE * 2), 3),
            **meta,
        })
        try:
            self.jobs.put_nowait(item)
        except queue.Full:
            self.state.archive_failures += 1
            LOG.warning('Archive queue full; dropped session %08x', recording.session)

    def _run(self) -> None:
        while (item := self.jobs.get()) is not None:
            try:
                self.archive.save(*item)
                self.state.archived += 1
            except Exception:
                self.state.archive_failures += 1
                LOG.exception('Archiving %s to %r failed', item[0], self.archive)

    def close(self, timeout: float = 10) -> None:
        try:
            self.jobs.put(None, timeout=timeout)
            self.thread.join(timeout=timeout)
        except queue.Full:
            pass
        if self.thread.is_alive():
            LOG.warning('Archive still busy on shutdown; %d recording(s) not archived',
                        self.jobs.qsize())
        else:
            self.archive.close()


def reload_transcriber(settings: SettingsWatcher, current: Transcriber | None,
                       state: HealthState) -> tuple[Transcriber, str | None] | None:
    """Build and warm up the backend for new settings; None keeps ``current``."""
    try:
        replacement, language = settings.build()
    except Exception as exc:
        state.last_error = str(exc)
        LOG.exception('New STT settings rejected; keeping %s', current.name if current else 'no backend')
        return None
    try:
        replacement.warmup()
    except Exception as exc:
        state.last_error = str(exc)
        LOG.exception('New STT backend failed to initialise; keeping the current one')
        replacement.close()
        return None
    if current is not None:
        try:
            current.close()
        except Exception:
            LOG.exception('STT backend shutdown failed')
    state.backend = replacement.name
    state.ready = True
    LOG.info('STT backend switched to %s', replacement.name)
    return replacement, language


def worker(jobs: queue.Queue, transcriber: Transcriber | None, language: str,
           save_dir: Path | None, state: HealthState, *,
           handle: Callable[[str | None], Decision] = decide, archive: Archive | None = None,
           settings: SettingsWatcher | None = None) -> None:
    writer = ArchiveWriter(archive, state) if archive is not None else None
    applied = 0  # settings.version the current transcriber was built from; 0 = environment
    override: str | None = None
    if transcriber is not None:
        try:
            transcriber.warmup()
            state.ready = True
            LOG.info('STT backend ready: %s', transcriber.name)
        except Exception as exc:
            state.last_error = str(exc)
            LOG.exception('STT backend initialization failed')

    while True:
        recording = jobs.get()
        try:
            if recording is None:
                break
            # No peer-controlled path components. ns timestamp prevents overwrites.
            stem = f'{time.time_ns()}-{recording.session:08x}'
            if save_dir is not None:
                path = save_dir / f'{stem}.wav'
                path.write_bytes(wav_bytes(recording.pcm))
                LOG.info('WAV: %s', path)
            if settings is not None and settings.version != applied:
                applied = settings.version
                if (reloaded := reload_transcriber(settings, transcriber, state)) is not None:
                    transcriber, override = reloaded
            lang = override or language
            if transcriber is None:
                print(f'[dry-run] {recording.peer[0]}: {len(recording.pcm) / (RATE * 2):.2f} s, '
                      f'{len(recording.pcm)} bytes PCM', flush=True)
            else:
                results = transcribe_all(transcriber, recording.pcm, lang)
                chosen = preferred_result(results)
                for result in results:
                    if result.error:
                        LOG.warning('STT %s failed: %s', result.role or result.backend, result.error)
                if chosen is None:
                    # Every backend failed. Keep per-backend details for retraining/debugging.
                    error = format_errors(results) or 'STT returned no transcript'
                    if writer is not None:
                        writer.submit(stem, recording, text=None, language=lang,
                                      backend=transcriber.name, error=error,
                                      transcripts=transcripts_meta(results))
                    raise RuntimeError(error)
                text = chosen.text or ''
                state.transcribed += 1
                state.last_error = None
                if text:
                    print(text, flush=True)
                else:
                    LOG.info('STT returned an empty transcript')
                decision = handle(text)
                if writer is not None:
                    writer.submit(stem, recording, text=text, language=lang,
                                  backend=transcriber.name,
                                  model=chosen.model or getattr(transcriber, 'model', None),
                                  corrected_text=text,
                                  corrected_source=chosen.role or transcriber.name,
                                  transcripts=transcripts_meta(results),
                                  command=decision.command, reply=decision.reply)
        except Exception as exc:
            # One failed request must not kill the receiver or stop the next recording.
            state.failures += 1
            state.last_error = str(exc)
            LOG.exception('Recording processing failed (session %08x)', recording.session)
        finally:
            jobs.task_done()

    if transcriber is not None:
        try:
            transcriber.close()
        except Exception:
            LOG.exception('STT backend shutdown failed')
    if writer is not None:
        writer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Receive Open Lexa recordings over UDP and transcribe them with a pluggable STT backend.')
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--bind', help='UDP bind address (default: OPEN_LEXA_BIND or 0.0.0.0)')
    parser.add_argument('--port', type=int, help='UDP port (default: OPEN_LEXA_PORT or 12345)')
    parser.add_argument('--health-port', type=int,
                        help='HTTP health port, 0 disables (default: OPEN_LEXA_HEALTH_PORT or 8080)')
    parser.add_argument('--device-ip', help='Only accept packets from this IPv4 address; '
                                            'requires a direct/hostNetwork path (ServiceLB SNATs peers)')
    parser.add_argument('--backend', choices=BACKENDS,
                        help='STT backend (default: STT_BACKEND; dual when both endpoints are configured)')
    parser.add_argument('--model', help='Backend model name; defaults to OPENAI_STT_MODEL/STT_HTTP_MODEL')
    parser.add_argument('--language', help='Language hint (default: STT_LANGUAGE or pl); auto disables it')
    parser.add_argument('--save-dir', type=Path,
                        help='Optionally keep received WAVs (default: OPEN_LEXA_SAVE_DIR)')
    parser.add_argument('--settings-seconds', type=float,
                        help='Poll the admin STT settings on SMB every N s, 0 disables '
                             '(default: OPEN_LEXA_SETTINGS_SECONDS or 30)')
    parser.add_argument('--dry-run', action='store_true', help='Receive audio without initialising an STT backend')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv(args.env_file)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # smbprotocol logs every SMB message at INFO; keep only its warnings/errors.
    logging.getLogger('smbprotocol').setLevel(logging.WARNING)
    logging.getLogger('spnego').setLevel(logging.WARNING)

    def env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            parser.error(f'{name} must be an integer')

    bind = args.bind or os.environ.get('OPEN_LEXA_BIND', '0.0.0.0')
    port = args.port if args.port is not None else env_int('OPEN_LEXA_PORT', 12345)
    health_port = (args.health_port if args.health_port is not None
                   else env_int('OPEN_LEXA_HEALTH_PORT', 8080))
    language = args.language or os.environ.get('STT_LANGUAGE', 'pl')
    save_dir = args.save_dir
    if save_dir is None and os.environ.get('OPEN_LEXA_SAVE_DIR'):
        save_dir = Path(os.environ['OPEN_LEXA_SAVE_DIR'])
    if not 1 <= port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if not 0 <= health_port <= 65535:
        parser.error('--health-port must be between 0 and 65535')

    try:
        token = parse_token(os.environ.get('OPEN_LEXA_TOKEN', ''))
    except ValueError as exc:
        parser.error(str(exc))

    transcriber: Transcriber | None = None
    backend = args.backend or os.environ.get('STT_BACKEND') or None
    if not args.dry_run:
        try:
            transcriber = create_transcriber(backend, model=args.model)
        except ConfigError as exc:
            parser.error(str(exc))
    archive = None
    if not args.dry_run:
        try:
            archive = create_archive()
        except ConfigError as exc:
            parser.error(str(exc))
    settings = None
    # --backend/--model pin the backend; otherwise the admin's settings override the env.
    settings_seconds = args.settings_seconds
    if settings_seconds is None:
        try:
            settings_seconds = float(os.environ.get('OPEN_LEXA_SETTINGS_SECONDS', '30'))
        except ValueError:
            parser.error('OPEN_LEXA_SETTINGS_SECONDS must be a number')
    if archive is not None and settings_seconds > 0 and not args.backend and not args.model:
        settings = SettingsWatcher(create_store(), lambda: create_transcriber(backend),
                                   interval=settings_seconds)
    state = HealthState(transcriber.name if transcriber is not None else 'dry-run',
                        ready=args.dry_run)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    jobs: queue.Queue = queue.Queue(maxsize=4)
    receiver = Receiver(token)

    def deliver(recording: Recording) -> bool:
        try:
            jobs.put_nowait(recording)
        except queue.Full:
            return False
        state.received += 1
        state.last_recording = time.time()
        LOG.info('Received %.2f s from %s (session %08x)',
                 len(recording.pcm) / (RATE * 2), recording.peer[0], recording.session)
        return True

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    # Kubernetes sends SIGTERM; treat it like Ctrl-C so queued recordings drain.
    signal.signal(signal.SIGTERM, stop)

    httpd = None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        sock.bind((bind, port))
        sock.settimeout(0.5)
        thread = threading.Thread(
            target=worker, args=(jobs, transcriber, language, save_dir, state),
            kwargs={'archive': archive, 'settings': settings},
            daemon=True, name='stt')
        thread.start()
        if settings is not None:
            settings.start()
        if health_port:
            httpd = start_health_server(health_port, state)
        LOG.info('Listening on %s:%d/UDP; backend=%s; language=%s; archive=%r',
                 bind, port, state.backend, language, archive)
        if health_port:
            LOG.info('Health endpoints on :%d/healthz and /readyz', health_port)
        try:
            while True:
                try:
                    # One extra byte lets validation reject oversized/truncated datagrams.
                    data, peer = sock.recvfrom(HEADER.size + CHUNK + 1)
                except socket.timeout:
                    expired = receiver.expire(time.monotonic())
                    if expired:
                        LOG.warning('Discarded %d incomplete recording(s)', expired)
                    continue
                if args.device_ip and peer[0] != args.device_ip:
                    continue
                ack = receiver.feed(data, peer, time.monotonic(), deliver)
                if ack:
                    try:
                        sock.sendto(ack, peer)
                    except OSError as exc:
                        LOG.warning('Could not send ACK: %s', exc)
        except KeyboardInterrupt:
            LOG.info('Stopping; waiting for queued transcriptions')
        finally:
            jobs.join()
            jobs.put(None)
            # Leaves room for the archive drain within terminationGracePeriodSeconds.
            thread.join(timeout=20)
            if settings is not None:
                settings.stop()
            if httpd is not None:
                httpd.shutdown()
                httpd.server_close()


if __name__ == '__main__':
    main()
