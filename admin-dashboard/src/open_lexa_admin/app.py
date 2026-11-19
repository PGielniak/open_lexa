"""Admin UI for recordings, transcription review, and receiver settings."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Callable
from urllib.parse import urlencode
import wave

from fastapi import Depends, FastAPI, HTTPException, Request, status as http
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from open_lexa import settings as stt_settings
from open_lexa.audio import wav_bytes
from open_lexa.handlers import CONFIG_PATH as HANDLERS_CONFIG_PATH
from open_lexa.handlers import discover as discover_handlers
from open_lexa.llm import SETTINGS_PATH as LLM_SETTINGS_PATH
from open_lexa.announce import (DEFAULT_MODEL as DEFAULT_TTS_MODEL, DEFAULT_VOICE as DEFAULT_TTS_VOICE,
                                PHRASES as ANNOUNCE_PHRASES, phrases_from)
from open_lexa.protocol import MAX_BYTES, RATE
from open_lexa.store import Store, create_store, missing
from open_lexa.stt import (ConfigError, Transcriber, format_errors, preferred_result,
                           transcribe_all, transcripts_meta)

from . import db, llm

LOG = logging.getLogger('open_lexa.admin')
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / 'templates'))
DATASET_PATH = '_dataset/transcripts.jsonl'
PAGE_SIZE = 50
# Same limit as UDP recordings and the local Whisper API.
MAX_UPLOAD_SECONDS = MAX_BYTES // (RATE * 2)
MAX_UPLOAD_BYTES = MAX_BYTES + 4096
ORIGINS = ('browser', 'upload')
# Flat form fields for the sonos section of _config/handlers.json.
SONOS_DEFAULTS = {'enabled': '', 'speaker_ip': '', 'default_favorite': '', 'default_radio': '',
                  'radio_stations': '', 'volume_step': '5',
                  'volume_max': '60', 'timeout': '5', 'spotify_client_id': '',
                  'spotify_client_secret': '', 'spotify_market': 'PL',
                  'announce_enabled': '', 'announce_base_url': '',
                  'announce_model': DEFAULT_TTS_MODEL, 'announce_voice': DEFAULT_TTS_VOICE}


class Worker:
    """Background ingest and debounced settings/dataset export."""

    def __init__(self, db_path: str, store: Store, interval: float = 60, debounce: float = 2):
        self.db_path = db_path
        self.store = store
        self.interval = interval
        self.debounce = debounce
        self.export_due: float | None = 0.0  # export once at startup
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.last_ingest: float | None = None
        self.last_export: float | None = None
        self.errors: dict[str, str] = {}
        self.thread = threading.Thread(target=self._run, name='admin-worker', daemon=True)

    @property
    def last_error(self) -> str | None:
        return '; '.join(f'{task}: {error}' for task, error in self.errors.items()) or None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.wake.set()
        self.thread.join(timeout=15)

    def mark_changed(self) -> None:
        self.export_due = time.monotonic() + self.debounce
        self.wake.set()

    def ingest(self, *, full: bool) -> int:
        with self.lock:
            conn = db.connect(self.db_path)
            try:
                added = db.ingest(conn, self.store, full=full)
            finally:
                conn.close()
            self.last_ingest = time.time()
            return added

    def export(self) -> None:
        with self.lock:
            conn = db.connect(self.db_path)
            try:
                dataset = db.export_dataset(conn)
                settings = db.get_setting(conn, 'stt')
                handlers = db.get_setting(conn, 'handlers')
                llm_settings = db.get_setting(conn, 'llm')
            finally:
                conn.close()
            if settings is not None:
                # The receiver polls this file; see open_lexa.settings.
                self.store.write_atomic(stt_settings.SETTINGS_PATH, json.dumps(
                    stt_settings.validate(settings), ensure_ascii=False, indent=1).encode())
            if handlers is not None:
                # Handler plugin configuration; see open_lexa.handlers.
                self.store.write_atomic(HANDLERS_CONFIG_PATH, json.dumps(
                    handlers, ensure_ascii=False, indent=1).encode())
            self.store.write_atomic(DATASET_PATH, ''.join(
                json.dumps(row, ensure_ascii=False) + '\n' for row in dataset).encode())
            if llm_settings is not None:
                self.store.write_atomic(LLM_SETTINGS_PATH, json.dumps(
                    llm.publish_payload(llm_settings), ensure_ascii=False, indent=1).encode())
            self.last_export = time.time()
            LOG.info('Exported %d dataset rows', len(dataset))

    def _task(self, name: str, operation) -> bool:
        try:
            operation()
        except Exception as exc:
            # Keep serving the UI; the task is retried on a later tick.
            self.errors[name] = str(exc)
            LOG.exception('Admin background %s failed', name)
            return False
        self.errors.pop(name, None)
        return True

    def _run(self) -> None:
        full = True
        next_ingest = 0.0
        while not self.stopping.is_set():
            now = time.monotonic()
            if now >= next_ingest:
                added = []
                if self._task('ingest', lambda: added.append(self.ingest(full=full))):
                    full = False
                    if added[0]:
                        self.mark_changed()
                next_ingest = now + self.interval
            due = self.export_due
            if due is not None and now >= due:
                ok = self._task('export', self.export)
                if self.export_due == due:  # no newer change arrived meanwhile
                    self.export_due = None if ok else now + self.interval
            wake_at = min(next_ingest,
                          self.export_due if self.export_due is not None else next_ingest)
            self.wake.wait(max(0.05, wake_at - time.monotonic()))
            self.wake.clear()

def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Single 'bytes=a-b' range -> inclusive (start, end); None when absent or unsupported."""
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header or '')
    if not match or match.groups() == ('', ''):
        return None
    first, last = match.groups()
    if first == '':
        start, end = max(0, size - int(last)), size - 1
    else:
        start, end = int(first), min(int(last) if last else size - 1, size - 1)
    if start > end or start >= size:
        raise HTTPException(http.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                            headers={'Content-Range': f'bytes */{size}'})
    return start, end


def read_pcm16k(data: bytes) -> bytes:
    """PCM of a mono 16-bit 16 kHz WAV; the browser converts anything else before upload."""
    try:
        with wave.open(io.BytesIO(data), 'rb') as source:
            params = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            pcm = source.readframes(source.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f'to nie jest poprawny plik WAV ({exc})') from exc
    if params != (1, 2, RATE):
        raise ValueError(f'oczekiwano WAV mono 16 bit {RATE} Hz, jest {params[0]} kan., '
                         f'{params[1] * 8} bit, {params[2]} Hz')
    if not pcm:
        raise ValueError('nagranie jest puste')
    if len(pcm) > MAX_BYTES:
        raise ValueError(f'nagranie dłuższe niż {MAX_UPLOAD_SECONDS} s')
    return pcm


def create_app(*, db_path: str, store: Store, username: str, password: str,
               background: bool = True, interval: float = 60,
               transcriber_factory: Callable[[dict], Transcriber] = stt_settings.build_transcriber,
               llm_test_factory: Callable[[dict], 'llm.ConnectionTest'] = llm.ConnectionTest) -> FastAPI:
    conn = db.open_db(db_path)
    conn.close()
    worker = Worker(db_path, store, interval=interval)
    basic = HTTPBasic(realm='Open Lexa admin')
    # Discovered once: the receiver executes these; the admin only shows them.
    handler_plugins = discover_handlers()

    def auth(credentials: HTTPBasicCredentials = Depends(basic)) -> None:
        ok_user = secrets.compare_digest(credentials.username.encode(), username.encode())
        ok_password = secrets.compare_digest(credentials.password.encode(), password.encode())
        if not (ok_user and ok_password):
            raise HTTPException(http.HTTP_401_UNAUTHORIZED, headers={'WWW-Authenticate': 'Basic'})

    def database():
        connection = db.connect(db_path)
        try:
            yield connection
        finally:
            connection.close()

    @asynccontextmanager
    async def lifespan(_app):
        if background:
            worker.start()
        try:
            yield
        finally:
            if background:
                worker.stop()
            store.close()

    app = FastAPI(title='Open Lexa admin', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.worker = worker
    app.state.handler_plugins = handler_plugins

    @app.get('/healthz', include_in_schema=False)
    def healthz():
        return {'status': 'ok', 'last_ingest': worker.last_ingest, 'last_export': worker.last_export,
                'last_error': worker.last_error}

    protected = [Depends(auth)]

    def render(request: Request, template: str, **context) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, template, {'worker': worker, **context})

    @app.get('/', dependencies=protected)
    def index():
        return RedirectResponse('/recordings?status=new', http.HTTP_303_SEE_OTHER)

    @app.get('/recordings', dependencies=protected, response_class=HTMLResponse)
    def recordings(request: Request, status: str | None = None, page: int = 1,
                   conn=Depends(database)):
        status = status if status in db.STATUSES else None
        page = max(page, 1)
        rows = db.list_recordings(conn, status=status,
                                  limit=PAGE_SIZE + 1, offset=(page - 1) * PAGE_SIZE)

        def link(**changes) -> str:
            params = {'status': status, 'page': page, **changes}
            return '/recordings?' + urlencode({k: v for k, v in params.items() if v})

        return render(request, 'list.html', rows=rows[:PAGE_SIZE], counts=db.counts(conn), status=status,
                      page=page, link=link,
                      has_next=len(rows) > PAGE_SIZE)

    @app.get('/recordings/{recording_id}', dependencies=protected, response_class=HTMLResponse)
    def recording(request: Request, recording_id: int, saved: int = 0, conn=Depends(database)):
        row = db.get_recording(conn, recording_id)
        if row is None:
            raise HTTPException(http.HTTP_404_NOT_FOUND)
        warnings = request.query_params.getlist('warning')
        return render(request, 'detail.html', r=row,
                       transcripts=db.transcript_map(row['transcripts']),
                       saved=saved, warnings=warnings,
                       agent=json.loads(row['agent']) if row['agent'] else None)

    @app.post('/recordings/{recording_id}', dependencies=protected)
    async def save(request: Request, recording_id: int, conn=Depends(database)):
        if db.get_recording(conn, recording_id) is None:
            raise HTTPException(http.HTTP_404_NOT_FOUND)
        form = await request.form()
        action = form.get('action', 'save')
        new_status = 'skipped' if action == 'skip' else ('new' if action == 'reset' else 'labeled')
        with conn:
            conn.execute('UPDATE recordings SET corrected_transcript = ?, status = ?, notes = ?, labeled_at = ? WHERE id = ?',
                         (str(form.get('corrected_transcript') or '').strip() or None,
                          new_status, str(form.get('notes') or '').strip() or None,
                          db.utcnow(), recording_id))
        worker.mark_changed()
        # Preserve provenance but mark a subsequent human correction as reviewed.
        if action != 'reset':
            annotation = db.annotation_of(db.get_recording(conn, recording_id))
            if annotation.get('source') == 'llm':
                annotation['reviewed_at'] = db.utcnow()
                db.set_annotation(conn, recording_id, annotation)
        target = db.next_unlabeled(conn, recording_id) if action in ('next', 'skip') else None
        return RedirectResponse(f'/recordings/{target or recording_id}?saved=1', 303)

    @app.get('/recordings/{recording_id}/audio', dependencies=protected)
    def audio(request: Request, recording_id: int, conn=Depends(database)):
        row = db.get_recording(conn, recording_id)
        if row is None:
            raise HTTPException(http.HTTP_404_NOT_FOUND)
        try:
            data = store.read(row['audio_path'])  # path from the DB, never from the request
        except OSError as exc:
            if missing(exc):
                raise HTTPException(http.HTTP_404_NOT_FOUND, 'WAV missing on the archive') from exc
            raise
        headers = {'Accept-Ranges': 'bytes', 'Cache-Control': 'private, max-age=3600'}
        selected = parse_range(request.headers.get('range'), len(data))
        if selected is None:
            return Response(data, media_type='audio/wav', headers=headers)
        start, end = selected
        headers['Content-Range'] = f'bytes {start}-{end}/{len(data)}'
        return Response(data[start:end + 1], status_code=http.HTTP_206_PARTIAL_CONTENT,
                        media_type='audio/wav', headers=headers)

    def mask_key(key: str) -> str:
        return f'…{key[-4:]}' if len(key) >= 12 else ('ustawiony' if key else '')

    @app.get('/settings', dependencies=protected, response_class=HTMLResponse)
    def settings_page(request: Request, saved: str = '', error: str = '', conn=Depends(database)):
        stored = db.get_setting(conn, 'stt')
        current = {**stt_settings.DEFAULTS, **(stored or {})}
        key = current.pop('cloud_api_key')
        llm_stored = db.get_setting(conn, 'llm')
        llm_current = {**llm.LLM_DEFAULTS, **(llm_stored or {})}
        llm_key = llm_current.pop('api_key')
        handlers_stored = db.get_setting(conn, 'handlers') or {}
        sonos_stored = handlers_stored.get('sonos') or {}
        spotify_stored = sonos_stored.get('spotify') or {}
        announce_stored = sonos_stored.get('announce') or {}
        sonos = {**SONOS_DEFAULTS,
                 **{k: v for k, v in sonos_stored.items() if k not in ('spotify', 'announce')},
                 'enabled': '1' if sonos_stored.get('enabled') else '',
                 'spotify_client_id': spotify_stored.get('client_id', ''),
                 'spotify_market': spotify_stored.get('market', 'PL'),
                 'announce_enabled': '1' if announce_stored.get('enabled') else '',
                 'announce_base_url': announce_stored.get('base_url', ''),
                 'announce_model': announce_stored.get('model') or DEFAULT_TTS_MODEL,
                 'announce_voice': announce_stored.get('voice') or DEFAULT_TTS_VOICE}
        phrases = phrases_from(announce_stored.get('phrases'))
        announce_phrases = [{'key': key, 'label': label, 'default': default,
                             'value': phrases.get(key, '')}
                            for key, label, default in ANNOUNCE_PHRASES]
        spotify_secret = str(spotify_stored.get('client_secret') or '')
        return render(request, 'settings.html', s=current, stored=stored is not None,
                      key_hint=f'…{key[-4:]}' if len(key) >= 12 else ('ustawiony' if key else ''),
                      llm=llm_current, llm_stored=llm_stored is not None, llm_key_hint=mask_key(llm_key),
                      sonos=sonos, sonos_stored=bool(handlers_stored),
                      announce_phrases=announce_phrases,
                      spotify_secret_hint=mask_key(spotify_secret),
                       plugins=[{'id': h.id, 'version': h.version, 'intents': sorted(h.intents),
                                'enabled': bool(handlers_stored.get(h.id, {}).get('enabled'))}
                               for h in handler_plugins],
                       mcp_servers=[{'id': sid, 'url': spec.get('url', ''),
                                     'enabled': spec.get('enabled', False),
                                     'has_key': bool(spec.get('api_key'))}
                                    for sid, spec in (handlers_stored.get('mcp_servers') or {}).items()],
                       saved=saved, error=error)

    @app.post('/settings/mcp', dependencies=protected)
    async def save_mcp(request: Request, conn=Depends(database)):
        form = await request.form()
        server_id = str(form.get('server_id') or '').strip()
        url = str(form.get('url') or '').strip()
        if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_-]{0,39}', server_id) or server_id == 'sonos':
            return RedirectResponse('/settings?' + urlencode({'error': 'MCP: niepoprawne ID serwera'}), 303)
        handlers = db.get_setting(conn, 'handlers') or {}
        servers = handlers.setdefault('mcp_servers', {})
        if form.get('delete'):
            servers.pop(server_id, None)
        else:
            if not url.startswith(('http://', 'https://')):
                return RedirectResponse('/settings?' + urlencode({'error': 'MCP: URL musi zaczynać się od http:// lub https://'}), 303)
            old = servers.get(server_id, {})
            servers[server_id] = {'url': url, 'enabled': bool(form.get('enabled')),
                                  'api_key': str(form.get('api_key') or '').strip() or old.get('api_key', '')}
        db.set_setting(conn, 'handlers', handlers)
        worker.mark_changed()
        return RedirectResponse('/settings?saved=mcp', 303)

    @app.post('/settings', dependencies=protected)
    async def save_settings(request: Request, conn=Depends(database)):
        form = await request.form()
        stored = db.get_setting(conn, 'stt') or {}
        # The inactive mode's fieldset is disabled and not submitted: keep its stored values.
        raw = {key: (form.get(key) or '').strip() or stored.get(key, '') for key in stt_settings.DEFAULTS}
        # The key is never rendered back: an empty field keeps the stored one.
        raw['cloud_api_key'] = ('' if form.get('clear_api_key') else
                                (raw['cloud_api_key'].strip() or stored.get('cloud_api_key', '')))
        try:
            settings = stt_settings.validate(raw)
        except ConfigError as exc:
            return RedirectResponse('/settings?' + urlencode({'error': str(exc)}), http.HTTP_303_SEE_OTHER)
        db.set_setting(conn, 'stt', settings)
        worker.mark_changed()
        LOG.info('STT settings saved (%s)', stt_settings.describe(settings))
        return RedirectResponse('/settings?saved=1', http.HTTP_303_SEE_OTHER)

    @app.get('/settings/whisper-models', dependencies=protected)
    def whisper_models(conn=Depends(database)):
        """Show the available/loaded models of the configured local API."""
        import urllib.request
        url = (db.get_setting(conn, 'stt') or stt_settings.DEFAULTS)['local_url'].rstrip('/')
        if not url.startswith(('http://', 'https://')):
            return JSONResponse({'ok': False, 'error': 'niepoprawny URL Whisper API'})
        try:
            with urllib.request.urlopen(f'{url}/models', timeout=3) as response:
                data = json.loads(response.read(32768))
            return JSONResponse({'ok': True, 'models': [item['id'] for item in data.get('data', [])],
                                 'loaded_model': data.get('loaded_model')})
        except Exception as exc:
            return JSONResponse({'ok': False, 'error': str(exc)})

    @app.post('/settings/llm', dependencies=protected)
    async def save_llm_settings(request: Request, conn=Depends(database)):
        form = await request.form()
        stored = db.get_setting(conn, 'llm') or {}
        raw = {key: (form.get(key) or '').strip() if form.get(key) is not None else ''
               for key in llm.LLM_DEFAULTS}
        raw = {key: (value or stored.get(key, '')) for key, value in raw.items()}
        raw['api_key'] = ('' if form.get('clear_llm_api_key') else
                          ((form.get('api_key') or '').strip() or stored.get('api_key', '')))
        try:
            settings = llm.validate_llm(raw)
        except llm.LLMConfigError as exc:
            return RedirectResponse('/settings?' + urlencode({'error': f'LLM: {exc}'}),
                                    http.HTTP_303_SEE_OTHER)
        if settings['receiver_mode'] != 'off' and not settings['api_key']:
            return RedirectResponse('/settings?' + urlencode({'error': 'LLM: tryb odbiornika wymaga klucza API'}),
                                    http.HTTP_303_SEE_OTHER)
        db.set_setting(conn, 'llm', settings)
        worker.mark_changed()  # publish _config/llm.json for the receiver
        LOG.info('LLM settings saved (%s / %s, receiver=%s)', settings['url'],
                 settings['model'], settings['receiver_mode'])
        return RedirectResponse('/settings?saved=llm', http.HTTP_303_SEE_OTHER)

    @app.post('/settings/llm-test', dependencies=protected)
    async def test_llm(request: Request, conn=Depends(database)):
        form = await request.form()
        stored = db.get_setting(conn, 'llm') or {}
        raw = {**stored,
               'url': (form.get('url') or '').strip() or stored.get('url', ''),
               'model': (form.get('model') or '').strip() or stored.get('model', ''),
               'api_key': (form.get('api_key') or '').strip() or stored.get('api_key', '')}
        try:
            tester = llm_test_factory(llm.validate_llm(raw))
            message = await run_in_threadpool(tester.test)
        except Exception as exc:
            return JSONResponse({'ok': False, 'error': str(exc)})
        return JSONResponse({'ok': True, 'message': message})

    @app.post('/settings/sonos', dependencies=protected)
    async def save_sonos_settings(request: Request, conn=Depends(database)):
        form = await request.form()
        handlers = db.get_setting(conn, 'handlers') or {}
        sonos_stored = handlers.get('sonos') or {}
        spotify_stored = sonos_stored.get('spotify') or {}
        announce_stored = sonos_stored.get('announce') or {}
        phrases_stored = phrases_from(announce_stored.get('phrases'))
        def field(name: str, previous: dict, key: str | None = None) -> str:
            # A disabled fieldset omits fields; an explicitly emptied field
            # must clear the old value (not silently restore it).
            value = form.get(name)
            return str(previous.get(key or name) or '') if value is None else str(value).strip()

        sonos = {
            'enabled': bool(form.get('enabled')),
            'speaker_ip': field('speaker_ip', sonos_stored),
            'default_favorite': field('default_favorite', sonos_stored),
            'default_radio': field('default_radio', sonos_stored),
            'radio_stations': field('radio_stations', sonos_stored),
            'volume_step': str(form.get('volume_step') or sonos_stored.get('volume_step') or '5').strip(),
            'volume_max': str(form.get('volume_max') or sonos_stored.get('volume_max') or '60').strip(),
            'timeout': str(form.get('timeout') or sonos_stored.get('timeout') or '5').strip(),
            'spotify': {
                'client_id': field('spotify_client_id', spotify_stored, 'client_id'),
                'client_secret': '' if form.get('clear_spotify_secret') else
                                 ((form.get('spotify_client_secret') or '').strip()
                                  or spotify_stored.get('client_secret', '')),
                'market': (form.get('spotify_market') or spotify_stored.get('market') or 'PL').strip(),
            },
            'announce': {
                # A disabled fieldset omits the marker too: keep the stored flag then.
                'enabled': (bool(form.get('announce_enabled')) if form.get('announce_present')
                            else bool(announce_stored.get('enabled'))),
                'base_url': field('announce_base_url', announce_stored, 'base_url').rstrip('/'),
                'model': field('announce_model', announce_stored, 'model') or DEFAULT_TTS_MODEL,
                'voice': field('announce_voice', announce_stored, 'voice') or DEFAULT_TTS_VOICE,
                'phrases': {key: ' '.join(field(f'phrase_{key}', phrases_stored, key).split())
                            for key, _, _ in ANNOUNCE_PHRASES},
            },
        }
        if sonos['enabled'] and not sonos['speaker_ip']:
            return RedirectResponse('/settings?' + urlencode(
                {'error': 'Sonos: podaj adres IP głośnika albo wyłącz handler'}),
                http.HTTP_303_SEE_OTHER)
        if sonos['enabled'] and sonos['announce']['enabled'] and not sonos['announce']['base_url'].startswith(
                ('http://', 'https://')):
            return RedirectResponse('/settings?' + urlencode(
                {'error': 'Komunikaty: podaj adres http://IP-noda:8081, który widzi głośnik'}),
                http.HTTP_303_SEE_OTHER)
        for key in ('volume_step', 'volume_max', 'timeout'):
            if not sonos[key].isdigit():
                return RedirectResponse('/settings?' + urlencode(
                    {'error': f'Sonos: {key} musi być liczbą'}), http.HTTP_303_SEE_OTHER)
        handlers['sonos'] = sonos
        db.set_setting(conn, 'handlers', handlers)
        worker.mark_changed()
        LOG.info('Sonos handler settings saved (enabled=%s, speaker=%s, spotify=%s)',
                 sonos['enabled'], sonos['speaker_ip'] or '-',
                 'yes' if sonos['spotify']['client_id'] else 'no')
        return RedirectResponse('/settings?saved=sonos', http.HTTP_303_SEE_OTHER)

    @app.post('/settings/handlers/{handler_id}', dependencies=protected)
    async def toggle_handler(handler_id: str, request: Request, conn=Depends(database)):
        """Plug/unplug any installed plugin without editing receiver code."""
        if handler_id not in {handler.id for handler in handler_plugins}:
            raise HTTPException(http.HTTP_404_NOT_FOUND, 'handler nie jest zainstalowany')
        form = await request.form()
        handlers = db.get_setting(conn, 'handlers') or {}
        section = handlers.get(handler_id) or {}
        if form.get('enabled') and handler_id == 'sonos' and not section.get('speaker_ip'):
            return RedirectResponse('/settings?' + urlencode({'error': 'Najpierw zapisz IP głośnika Sonos'}),
                                    http.HTTP_303_SEE_OTHER)
        section['enabled'] = bool(form.get('enabled'))
        handlers[handler_id] = section
        db.set_setting(conn, 'handlers', handlers)
        worker.mark_changed()
        return RedirectResponse('/settings?saved=handler', http.HTTP_303_SEE_OTHER)

    @app.post('/settings/sonos-test', dependencies=protected)
    async def test_sonos(request: Request, conn=Depends(database)):
        form = await request.form()
        stored = (db.get_setting(conn, 'handlers') or {}).get('sonos') or {}
        ip = (form.get('speaker_ip') or '').strip() or str(stored.get('speaker_ip') or '')
        if not ip:
            return JSONResponse({'ok': False, 'error': 'brak adresu IP głośnika'})

        def probe() -> dict:
            from open_lexa.handlers.sonos.sonos import SonosSpeaker
            return dict(SonosSpeaker(ip, timeout=5.0).info())

        try:
            info = await run_in_threadpool(probe)
        except Exception as exc:
            return JSONResponse({'ok': False, 'error': str(exc)})
        return JSONResponse({'ok': True, 'info': info})

    @app.post('/settings/sonos-control', dependencies=protected)
    async def sonos_control(request: Request, conn=Depends(database)):
        """Authenticated portal controls using the same plugin as voice commands."""
        from open_lexa.handlers.base import CommandRequest
        from open_lexa.handlers.sonos.handler import SonosHandler

        form = await request.form()
        config = (db.get_setting(conn, 'handlers') or {}).get('sonos') or {}
        if not config.get('enabled'):
            return JSONResponse({'ok': False, 'error': 'handler Sonos jest wyłączony'}, status_code=400)
        action = str(form.get('action') or '')
        allowed = {'play': 'play_music', 'pause': 'pause', 'resume': 'resume',
                   'stop': 'stop', 'next': 'next_track', 'previous': 'previous_track',
                   'volume_up': 'volume_up', 'volume_down': 'volume_down',
                   'volume_set': 'volume_set', 'search': 'play_spotify',
                   'favorite': 'play_spotify'}
        if action not in allowed:
            return JSONResponse({'ok': False, 'error': 'nieznana akcja'}, status_code=400)

        def execute():
            handler = SonosHandler()
            handler.configure(config)
            if action == 'favorite':
                title = handler.speaker.play_favorite(str(form.get('favorite') or '').strip() or None)
                return {'status': 'handled', 'response': f'Włączam {title}', 'error': None}
            entities = {'search_query': str(form.get('query') or '').strip(),
                        'level': str(form.get('level') or '').strip()}
            result = handler.handle(CommandRequest(allowed[action], entities))
            return {'status': result.status, 'response': result.response, 'error': result.error}

        try:
            result = await run_in_threadpool(execute)
        except Exception as exc:
            LOG.warning('Sonos portal control failed: %s', exc)
            return JSONResponse({'ok': False, 'error': str(exc)})
        return JSONResponse({'ok': result['status'] == 'handled', **result})

    @app.get('/record', dependencies=protected, response_class=HTMLResponse)
    def record_page(request: Request, conn=Depends(database)):
        stored = db.get_setting(conn, 'stt')
        return render(request, 'record.html', settings=stored and stt_settings.describe(stored),
                      max_seconds=MAX_UPLOAD_SECONDS)

    @app.post('/record', dependencies=protected)
    async def record(request: Request):
        if int(request.headers.get('content-length') or 0) > MAX_UPLOAD_BYTES + 65536:
            raise HTTPException(http.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 'plik jest za duży')
        form = await request.form()
        upload, origin = form.get('audio'), form.get('origin')
        if upload is None or isinstance(upload, str) or origin not in ORIGINS:
            raise HTTPException(http.HTTP_400_BAD_REQUEST, 'brak pliku audio')
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        try:
            pcm = read_pcm16k(data)
        except ValueError as exc:
            raise HTTPException(http.HTTP_400_BAD_REQUEST, str(exc)) from exc
        # Threadpool: transcription and SMB writes block.
        return JSONResponse(await run_in_threadpool(
            ingest_upload, pcm, origin, request.client.host if request.client else None))

    def ingest_upload(pcm: bytes, origin: str, peer: str | None) -> dict:
        """Store an admin recording exactly like the receiver archives a UDP one."""
        conn = db.connect(db_path)
        try:
            stored = db.get_setting(conn, 'stt')
            session = secrets.token_hex(4)
            stem = f'{time.time_ns()}-{session}'
            created = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
            meta = {'created': created, 'session': session, 'peer': peer, 'origin': origin,
                    'duration_seconds': round(len(pcm) / (RATE * 2), 3), 'text': None}
            if stored is None:
                meta['error'] = 'brak ustawień STT (zakładka Ustawienia)'
            else:
                settings = stt_settings.validate(stored)
                mode = settings['mode']
                meta.update(language=settings['language'], backend=mode,
                            model=settings[f'{mode}_model'] if mode != 'dual' else settings['cloud_model'])
                try:
                    transcriber = transcriber_factory(settings)
                    try:
                        results = transcribe_all(transcriber, pcm, settings['language'])
                    finally:
                        transcriber.close()
                except Exception as exc:
                    # Failed recordings are kept, like the receiver does.
                    LOG.warning('Admin transcription failed: %s', exc)
                    meta['error'] = str(exc)
                else:
                    chosen = preferred_result(results)
                    meta['transcripts'] = transcripts_meta(results)
                    for result in results:
                        if result.error:
                            LOG.warning('Admin STT %s failed: %s', result.role or result.backend, result.error)
                    if chosen is None:
                        # Keep per-backend details in the sidecar, as the receiver does.
                        meta['error'] = format_errors(results) or 'STT returned no transcript'
                    else:
                        # GPT transcribe pre-fills the corrected transcript; the
                        # human only adds the intent and entities.
                        meta['text'] = chosen.text or ''
                        meta['corrected_text'] = meta['text']
                        meta['corrected_source'] = chosen.role or mode
                        meta['model'] = chosen.model or meta.get('model')
            audio_path = f'{created[:10]}/{stem}.wav'
            with worker.lock:  # no ingest between the WAV and its sidecar
                store.write_atomic(audio_path, wav_bytes(pcm))
                store.write_atomic(f'{created[:10]}/{stem}.json',
                                   json.dumps(meta, ensure_ascii=False, indent=2).encode())
                db.insert_recording(conn, stem, audio_path, meta)
            recording_id = conn.execute('SELECT id FROM recordings WHERE stem = ?', (stem,)).fetchone()[0]
        finally:
            conn.close()
        worker.mark_changed()
        return {'id': recording_id, 'url': f'/recordings/{recording_id}', 'text': meta['text'],
                'error': meta.get('error')}

    @app.post('/sync', dependencies=protected)
    def sync():
        added = worker.ingest(full=True)
        worker.export()
        return RedirectResponse(f'/recordings?status=new&synced={added}', http.HTTP_303_SEE_OTHER)

    return app


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logging.getLogger('smbprotocol').setLevel(logging.WARNING)
    logging.getLogger('spnego').setLevel(logging.WARNING)
    password = os.environ.get('ADMIN_PASSWORD', '')
    if len(password) < 12:
        raise SystemExit('ADMIN_PASSWORD must be set (at least 12 characters)')
    db_path = os.environ.get('ADMIN_DB', '/data/labels.sqlite3')
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    app = create_app(db_path=db_path, store=create_store(), username=os.environ.get('ADMIN_USERNAME', 'admin'),
                     password=password, interval=float(os.environ.get('ADMIN_INGEST_SECONDS', '60')))
    # One worker: SQLite + the background thread assume a single process.
    uvicorn.run(app, host=os.environ.get('ADMIN_BIND', '0.0.0.0'), port=int(os.environ.get('ADMIN_PORT', '8000')),
                workers=1, proxy_headers=True, access_log=False)


if __name__ == '__main__':
    main()
