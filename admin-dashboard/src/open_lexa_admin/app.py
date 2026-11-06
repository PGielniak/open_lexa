"""Admin UI for labeling recordings: listen, correct the transcript, tag intent + entities."""
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
from open_lexa.commands import decide
from open_lexa.handlers import CONFIG_PATH as HANDLERS_CONFIG_PATH
from open_lexa.handlers import discover as discover_handlers
from open_lexa.llm import SETTINGS_PATH as LLM_SETTINGS_PATH
from open_lexa.protocol import MAX_BYTES, RATE
from open_lexa.store import Store, create_store, missing
from open_lexa.stt import (ConfigError, Transcriber, format_errors, preferred_result,
                           transcribe_all, transcripts_meta)

from . import db, llm

LOG = logging.getLogger('open_lexa.admin')
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / 'templates'))
LEXICON_PATH = '_lexicon/lexicon.json'
DATASET_PATH = '_dataset/labels.jsonl'
PAGE_SIZE = 50
# Same limit as UDP recordings and the local Whisper API.
MAX_UPLOAD_SECONDS = MAX_BYTES // (RATE * 2)
MAX_UPLOAD_BYTES = MAX_BYTES + 4096
ORIGINS = ('browser', 'upload')
# Flat form fields for the sonos section of _config/handlers.json.
SONOS_DEFAULTS = {'enabled': '', 'speaker_ip': '', 'default_favorite': '', 'volume_step': '5',
                  'volume_max': '60', 'timeout': '5', 'spotify_client_id': '',
                  'spotify_client_secret': '', 'spotify_market': 'PL'}
MANUAL_ANNOTATE_LIMIT = 500


class Worker:
    """Background ingest (every ``interval`` s) and debounced export after label changes.

    A failed export stays pending and is retried every ``interval`` s, so a broken
    share never silently drops labels from the lexicon.
    """

    def __init__(self, db_path: str, store: Store, interval: float = 60, debounce: float = 2,
                 annotator_factory: Callable[[dict], 'llm.Annotator'] = llm.Annotator):
        self.db_path = db_path
        self.store = store
        self.interval = interval
        self.debounce = debounce
        self.annotator_factory = annotator_factory
        self.export_due: float | None = 0.0  # export once at startup
        self.wake = threading.Event()
        self.annotate_wake = threading.Event()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.annotation_gate = threading.Lock()
        self.last_ingest: float | None = None
        self.last_export: float | None = None
        self.last_annotate: float | None = None
        self.annotating = False
        self.manual_annotate_limit: int | None = None
        self.errors: dict[str, str] = {}
        self.thread = threading.Thread(target=self._run, name='admin-worker', daemon=True)
        self.annotation_thread = threading.Thread(target=self._run_annotate,
                                                  name='admin-annotate', daemon=True)

    @property
    def last_error(self) -> str | None:
        return '; '.join(f'{task}: {error}' for task, error in self.errors.items()) or None

    def start(self) -> None:
        self.thread.start()
        self.annotation_thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.wake.set()
        self.annotate_wake.set()
        self.thread.join(timeout=15)
        self.annotation_thread.join(timeout=2)

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
                lexicon, dataset = db.export(conn)
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
            self.store.write_atomic(LEXICON_PATH, json.dumps(lexicon, ensure_ascii=False, indent=1).encode())
            self.store.write_atomic(DATASET_PATH, ''.join(
                json.dumps(row, ensure_ascii=False) + '\n' for row in dataset).encode())
            if llm_settings is not None:
                # The receiver's LLM NLU polls this file; see open_lexa.llm.
                # Published last: a broken LLM section must not block the lexicon.
                self.store.write_atomic(LLM_SETTINGS_PATH, json.dumps(
                    llm.publish_payload(llm_settings), ensure_ascii=False, indent=1).encode())
            self.last_export = time.time()
            LOG.info('Exported lexicon (%d examples) and %d dataset rows', len(lexicon['examples']), len(dataset))

    # --- LLM annotation -------------------------------------------------------------

    def _load_annotation_batch(self, *, auto: bool, limit: int | None,
                               recording_id: int | None = None):
        """Read settings + pending rows under the lock; LLM calls happen outside it."""
        with self.lock:
            conn = db.connect(self.db_path)
            try:
                settings = db.get_setting(conn, 'llm')
                if auto and not (settings or {}).get('auto_label'):
                    return settings, [], {}
                if not llm.llm_enabled(settings):
                    raise llm.LLMConfigError('LLM nie jest skonfigurowany (Ustawienia → LLM: URL, model, klucz API)')
                if recording_id is not None:
                    row = db.get_recording(conn, recording_id)
                    if row is None:
                        raise llm.LLMError(f'nagranie {recording_id} nie istnieje')
                    rows = [row]
                else:
                    rows = db.pending_annotation(conn, limit or int(settings.get('batch_size') or 5))
                pending = [(row['id'], row['corrected_transcript'] or row['original_transcript'],
                            db.annotation_of(row).get('attempts', 0)) for row in rows]
                pending = [(rid, text, attempts) for rid, text, attempts in pending if (text or '').strip()]
                snapshot = llm.dictionary_snapshot(conn) if pending else {}
            finally:
                conn.close()
        return settings, pending, snapshot

    def _record_annotation_error(self, recording_id: int, error: Exception, attempts: int,
                                 model: str = '') -> None:
        try:
            with self.lock:
                conn = db.connect(self.db_path)
                try:
                    row = db.get_recording(conn, recording_id)
                    if row is not None and row['status'] == 'new':
                        db.set_annotation(conn, recording_id, {
                            'source': 'llm', 'model': model, 'prompt_version': llm.PROMPT_VERSION,
                            'at': db.utcnow(), 'attempts': attempts, 'status': 'error',
                            'error': str(error)[:300]})
                finally:
                    conn.close()
        except Exception:
            LOG.exception('Recording the annotation error for %d failed', recording_id)

    def annotate_pending(self, *, auto: bool = False, limit: int | None = None) -> tuple[int, int]:
        """Label pending recordings with the LLM; returns (labeled, failed)."""
        with self.annotation_gate:
            return self._annotate_pending(auto=auto, limit=limit)

    def _annotate_pending(self, *, auto: bool, limit: int | None) -> tuple[int, int]:
        settings, pending, snapshot = self._load_annotation_batch(auto=auto, limit=limit)
        if not pending:
            return 0, 0
        self.annotating = True
        try:
            annotator = self.annotator_factory(settings)
            labeled = failed = 0
            for recording_id, transcript, attempts in pending:
                if self.stopping.is_set():
                    break
                try:
                    answer = annotator.annotate(transcript, snapshot)  # slow; outside the lock
                except Exception as exc:
                    failed += 1
                    LOG.warning('LLM annotation of recording %d failed: %s', recording_id, exc)
                    self._record_annotation_error(recording_id, exc, attempts + 1,
                                                  settings.get('model', ''))
                    continue
                try:
                    with self.lock:
                        conn = db.connect(self.db_path)
                        try:
                            llm.apply_annotation(conn, recording_id, answer,
                                                 model=settings['model'], attempts=attempts + 1,
                                                 expected_transcript=transcript)
                        finally:
                            conn.close()
                except Exception as exc:
                    failed += 1
                    LOG.warning('Applying the LLM label to recording %d failed: %s', recording_id, exc)
                    self._record_annotation_error(recording_id, exc, attempts + 1,
                                                  settings.get('model', ''))
                else:
                    labeled += 1
            if labeled:
                self.mark_changed()
            self.last_annotate = time.time()
            LOG.info('LLM annotation: %d labeled, %d failed', labeled, failed)
            return labeled, failed
        finally:
            self.annotating = False

    def annotate_one(self, recording_id: int) -> dict:
        """Annotate a single recording synchronously; raises llm.LLMError on failure."""
        settings, pending, snapshot = self._load_annotation_batch(auto=False, limit=None,
                                                                  recording_id=recording_id)
        if not pending:
            raise llm.LLMError('to nagranie nie ma transkrypcji do oznaczenia')
        rid, transcript, attempts = pending[0]
        annotator = self.annotator_factory(settings)
        answer = annotator.annotate(transcript, snapshot)
        with self.lock:
            conn = db.connect(self.db_path)
            try:
                annotation = llm.apply_annotation(conn, rid, answer, model=settings['model'],
                                                  attempts=attempts + 1,
                                                  expected_transcript=transcript)
            finally:
                conn.close()
        self.mark_changed()
        return annotation

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

    def _run_annotate(self) -> None:
        # Independent from SMB ingest/export: a slow LLM never blocks either.
        while not self.stopping.is_set():
            self.annotate_wake.wait(max(self.interval, 30.0))
            self.annotate_wake.clear()
            if not self.stopping.is_set():
                limit = self.manual_annotate_limit
                self.manual_annotate_limit = None
                self._task('annotate', lambda: self.annotate_pending(auto=limit is None, limit=limit))


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
               annotator_factory: Callable[[dict], 'llm.Annotator'] = llm.Annotator) -> FastAPI:
    conn = db.open_db(db_path)
    conn.close()
    worker = Worker(db_path, store, interval=interval, annotator_factory=annotator_factory)
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
                'last_annotate': worker.last_annotate, 'annotating': worker.annotating,
                'last_error': worker.last_error}

    protected = [Depends(auth)]

    def render(request: Request, template: str, **context) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, template, {'worker': worker, **context})

    @app.get('/', dependencies=protected)
    def index():
        return RedirectResponse('/recordings?status=new', http.HTTP_303_SEE_OTHER)

    @app.get('/recordings', dependencies=protected, response_class=HTMLResponse)
    def recordings(request: Request, status: str | None = None, intent: str | None = None, page: int = 1,
                   conn=Depends(database)):
        status = status if status in db.STATUSES else None
        page = max(page, 1)
        rows = db.list_recordings(conn, status=status, intent=intent or None,
                                  limit=PAGE_SIZE + 1, offset=(page - 1) * PAGE_SIZE)

        def link(**changes) -> str:
            params = {'status': status, 'intent': intent, 'page': page, **changes}
            return '/recordings?' + urlencode({k: v for k, v in params.items() if v})

        return render(request, 'list.html', rows=rows[:PAGE_SIZE], counts=db.counts(conn), status=status,
                      intent=intent, intents=db.intents(conn), page=page, link=link,
                      has_next=len(rows) > PAGE_SIZE)

    @app.get('/recordings/{recording_id}', dependencies=protected, response_class=HTMLResponse)
    def recording(request: Request, recording_id: int, saved: int = 0, conn=Depends(database)):
        row = db.get_recording(conn, recording_id)
        if row is None:
            raise HTTPException(http.HTTP_404_NOT_FOUND)
        warnings = request.query_params.getlist('warning')
        return render(request, 'detail.html', r=row, entities=db.recording_entities(conn, recording_id),
                       intents=db.intents(conn), entity_types=db.entity_types(conn),
                       transcripts=db.transcript_map(row['transcripts']),
                       allowed=db.intent_entity_map(conn), saved=saved, warnings=warnings,
                       annotation=db.annotation_of(row))

    @app.post('/recordings/{recording_id}', dependencies=protected)
    async def save(request: Request, recording_id: int, conn=Depends(database)):
        if db.get_recording(conn, recording_id) is None:
            raise HTTPException(http.HTTP_404_NOT_FOUND)
        form = await request.form()
        action = form.get('action', 'save')
        entities = [{'type': t, 'value': v, 'original_fragment': f} for t, v, f in zip(
            form.getlist('entity_type'), form.getlist('entity_value'), form.getlist('entity_fragment'))]
        try:
            warnings = db.save_label(
                conn, recording_id, corrected_transcript=form.get('corrected_transcript', ''),
                intent=form.get('intent') or None, entities=entities,
                status='skipped' if action == 'skip' else ('new' if action == 'reset' else 'labeled'),
                notes=form.get('notes'))
        except ValueError as exc:
            return RedirectResponse(f'/recordings/{recording_id}?' + urlencode({'warning': str(exc)}),
                                    http.HTTP_303_SEE_OTHER)
        worker.mark_changed()
        # Preserve provenance but mark a subsequent human correction as reviewed.
        if action != 'reset':
            annotation = db.annotation_of(db.get_recording(conn, recording_id))
            if annotation.get('source') == 'llm':
                annotation['reviewed_at'] = db.utcnow()
                db.set_annotation(conn, recording_id, annotation)
        target = recording_id
        if action in ('next', 'skip') and not warnings:
            target = db.next_unlabeled(conn, recording_id) or recording_id
        query = urlencode([('saved', 1)] + [('warning', w) for w in warnings])
        return RedirectResponse(f'/recordings/{target}?{query}', http.HTTP_303_SEE_OTHER)

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

    @app.get('/recordings/{recording_id}/suggest', dependencies=protected)
    def suggest(recording_id: int, value: str = '', conn=Depends(database)):
        row = db.get_recording(conn, recording_id)
        if row is None:
            raise HTTPException(http.HTTP_404_NOT_FOUND)
        return JSONResponse({'fragment': db.suggest_fragment(row['original_transcript'], value)})

    @app.get('/dictionary', dependencies=protected, response_class=HTMLResponse)
    def dictionary(request: Request, conn=Depends(database)):
        return render(request, 'dictionary.html', intents=db.intents(conn), phrases=db.phrases(conn),
                      entity_types=db.entity_types(conn), values=db.dictionary(conn),
                      allowed=db.intent_entity_map(conn))

    @app.post('/dictionary', dependencies=protected)
    async def edit_dictionary(request: Request, conn=Depends(database)):
        form = await request.form()
        action = form.get('action')
        try:
            with conn:
                if action == 'add_intent':
                    name = db.normalize(form.get('name', '')).replace(' ', '_')
                    if not name:
                        raise ValueError('nazwa intencji jest wymagana')
                    conn.execute('INSERT INTO intents (name, parent_id, description, is_actionable) '
                                 'VALUES (?, (SELECT id FROM intents WHERE name = ?), ?, ?)',
                                 (name, form.get('parent') or None, form.get('description') or None,
                                  0 if form.get('non_actionable') else 1))
                    for entity_type in form.getlist('entity_types'):
                        conn.execute('INSERT OR IGNORE INTO intent_entity_types (intent_id, entity_type_id) '
                                     'SELECT i.id, t.id FROM intents i, entity_types t WHERE i.name = ? '
                                     'AND t.name = ?', (name, entity_type))
                elif action == 'add_phrase':
                    conn.execute('INSERT OR IGNORE INTO intent_phrases (intent_id, phrase) '
                                 'SELECT id, ? FROM intents WHERE name = ?',
                                 (form.get('phrase', '').strip(), form.get('intent')))
                elif action == 'delete_phrase':
                    conn.execute('DELETE FROM intent_phrases WHERE id = ?', (form.get('id'),))
                elif action == 'add_value':
                    value_id = db.ensure_value(conn, form.get('entity_type', ''), form.get('value', ''))
                    for synonym in form.get('synonyms', '').split(','):
                        db.add_synonym(conn, value_id, synonym)
                elif action == 'add_synonym':
                    db.add_synonym(conn, int(form.get('value_id')), form.get('synonym', ''))
                elif action == 'delete_synonym':
                    conn.execute('DELETE FROM entity_synonyms WHERE id = ?', (form.get('id'),))
                elif action == 'delete_value':
                    conn.execute('DELETE FROM entity_values WHERE id = ? AND id NOT IN '
                                 '(SELECT entity_value_id FROM recording_entities '
                                 'WHERE entity_value_id IS NOT NULL)', (form.get('id'),))
                else:
                    raise ValueError(f'nieznana akcja {action!r}')
        except (ValueError, TypeError) as exc:
            raise HTTPException(http.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except db.sqlite3.IntegrityError as exc:
            raise HTTPException(http.HTTP_409_CONFLICT, str(exc)) from exc
        worker.mark_changed()
        return RedirectResponse('/dictionary', http.HTTP_303_SEE_OTHER)

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
        sonos = {**SONOS_DEFAULTS,
                 **{k: v for k, v in sonos_stored.items() if k != 'spotify'},
                 'enabled': '1' if sonos_stored.get('enabled') else '',
                 'spotify_client_id': spotify_stored.get('client_id', ''),
                 'spotify_market': spotify_stored.get('market', 'PL')}
        spotify_secret = str(spotify_stored.get('client_secret') or '')
        return render(request, 'settings.html', s=current, stored=stored is not None,
                      key_hint=f'…{key[-4:]}' if len(key) >= 12 else ('ustawiony' if key else ''),
                      llm=llm_current, llm_stored=llm_stored is not None, llm_key_hint=mask_key(llm_key),
                      sonos=sonos, sonos_stored=bool(handlers_stored),
                      spotify_secret_hint=mask_key(spotify_secret),
                      plugins=[{'id': h.id, 'version': h.version, 'intents': sorted(h.intents),
                                'enabled': bool(handlers_stored.get(h.id, {}).get('enabled'))}
                               for h in handler_plugins],
                      saved=saved, error=error)

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
        raw['auto_label'] = '1' if form.get('auto_label') else ''
        raw['api_key'] = ('' if form.get('clear_llm_api_key') else
                          ((form.get('api_key') or '').strip() or stored.get('api_key', '')))
        try:
            settings = llm.validate_llm(raw)
        except llm.LLMConfigError as exc:
            return RedirectResponse('/settings?' + urlencode({'error': f'LLM: {exc}'}),
                                    http.HTTP_303_SEE_OTHER)
        if settings['auto_label'] and not settings['api_key']:
            return RedirectResponse('/settings?' + urlencode({'error': 'LLM: podaj klucz API dla automatycznego oznaczania'}),
                                    http.HTTP_303_SEE_OTHER)
        if settings['receiver_mode'] != 'off' and not settings['api_key']:
            return RedirectResponse('/settings?' + urlencode({'error': 'LLM: tryb odbiornika wymaga klucza API'}),
                                    http.HTTP_303_SEE_OTHER)
        db.set_setting(conn, 'llm', settings)
        worker.mark_changed()  # publish _config/llm.json for the receiver
        worker.annotate_wake.set()
        LOG.info('LLM settings saved (%s / %s, auto=%s, receiver=%s)', settings['url'],
                 settings['model'], 'on' if settings['auto_label'] else 'off',
                 settings['receiver_mode'])
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
            annotator = annotator_factory(llm.validate_llm(raw))
            message = await run_in_threadpool(annotator.test)
        except Exception as exc:
            return JSONResponse({'ok': False, 'error': str(exc)})
        return JSONResponse({'ok': True, 'message': message})

    @app.post('/settings/sonos', dependencies=protected)
    async def save_sonos_settings(request: Request, conn=Depends(database)):
        form = await request.form()
        handlers = db.get_setting(conn, 'handlers') or {}
        sonos_stored = handlers.get('sonos') or {}
        spotify_stored = sonos_stored.get('spotify') or {}
        def field(name: str, previous: dict, key: str | None = None) -> str:
            # A disabled fieldset omits fields; an explicitly emptied field
            # must clear the old value (not silently restore it).
            value = form.get(name)
            return str(previous.get(key or name) or '') if value is None else str(value).strip()

        sonos = {
            'enabled': bool(form.get('enabled')),
            'speaker_ip': field('speaker_ip', sonos_stored),
            'default_favorite': field('default_favorite', sonos_stored),
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
        }
        if sonos['enabled'] and not sonos['speaker_ip']:
            return RedirectResponse('/settings?' + urlencode(
                {'error': 'Sonos: podaj adres IP głośnika albo wyłącz handler'}),
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

    @app.post('/annotate', dependencies=protected)
    def annotate_all(conn=Depends(database)):
        """Kick off background LLM labeling of every pending recording."""
        if not llm.llm_enabled(db.get_setting(conn, 'llm')):
            return RedirectResponse('/settings?' + urlencode({'error': 'Skonfiguruj klucz API LLM'}),
                                    http.HTTP_303_SEE_OTHER)
        if not background:
            # Tests/offline runs with no worker still need an explicit batch.
            threading.Thread(target=_annotate_all_background, name='annotate', daemon=True).start()
        else:
            worker.manual_annotate_limit = MANUAL_ANNOTATE_LIMIT
            worker.annotate_wake.set()
        return RedirectResponse('/recordings?status=new&annotate=started', http.HTTP_303_SEE_OTHER)

    @app.get('/annotate/status', dependencies=protected)
    def annotation_status(conn=Depends(database)):
        """Progress summary without exposing transcript or API credentials."""
        pending = conn.execute(
            "SELECT count(*) FROM recordings WHERE status = 'new' "
            "AND coalesce(corrected_transcript, original_transcript, '') != '' "
            "AND (annotation IS NULL OR json_extract(annotation, '$.status') = 'error')"
        ).fetchone()[0]
        labeled = conn.execute(
            "SELECT count(*) FROM recordings WHERE status = 'labeled' "
            "AND json_extract(annotation, '$.source') = 'llm'").fetchone()[0]
        failed = conn.execute(
            "SELECT count(*) FROM recordings WHERE status = 'new' "
            "AND json_extract(annotation, '$.status') = 'error'").fetchone()[0]
        return {'running': worker.annotating, 'pending': pending,
                'labeled': labeled, 'failed': failed, 'last_at': worker.last_annotate}

    def _annotate_all_background() -> None:
        try:
            worker.annotate_pending(limit=MANUAL_ANNOTATE_LIMIT)
        except Exception:
            LOG.exception('Background LLM annotation failed')

    @app.post('/recordings/{recording_id}/annotate', dependencies=protected)
    def annotate_recording(recording_id: int):
        try:
            annotation = worker.annotate_one(recording_id)
        except (llm.LLMError, llm.LLMConfigError, ValueError) as exc:
            return RedirectResponse(f'/recordings/{recording_id}?' + urlencode({'warning': str(exc)}),
                                    http.HTTP_303_SEE_OTHER)
        except Exception as exc:  # unexpected: surface it, keep the recording untouched
            LOG.exception('LLM annotation of recording %d failed', recording_id)
            return RedirectResponse(f'/recordings/{recording_id}?' + urlencode({'warning': str(exc)}),
                                    http.HTTP_303_SEE_OTHER)
        return RedirectResponse(f'/recordings/{recording_id}?saved=1&llm={annotation["intent"]}',
                                http.HTTP_303_SEE_OTHER)

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
                        decision = decide(meta['text'])
                        meta.update(command=decision.command, reply=decision.reply)
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
