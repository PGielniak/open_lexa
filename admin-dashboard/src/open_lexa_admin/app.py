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
from open_lexa.protocol import MAX_BYTES, RATE
from open_lexa.store import Store, create_store, missing
from open_lexa.stt import (ConfigError, Transcriber, format_errors, preferred_result,
                           transcribe_all, transcripts_meta)

from . import db

LOG = logging.getLogger('open_lexa.admin')
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / 'templates'))
LEXICON_PATH = '_lexicon/lexicon.json'
DATASET_PATH = '_dataset/labels.jsonl'
PAGE_SIZE = 50
# Same limit as UDP recordings and the local Whisper API.
MAX_UPLOAD_SECONDS = MAX_BYTES // (RATE * 2)
MAX_UPLOAD_BYTES = MAX_BYTES + 4096
ORIGINS = ('browser', 'upload')


class Worker:
    """Background ingest (every ``interval`` s) and debounced export after label changes.

    A failed export stays pending and is retried every ``interval`` s, so a broken
    share never silently drops labels from the lexicon.
    """

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
                lexicon, dataset = db.export(conn)
                settings = db.get_setting(conn, 'stt')
            finally:
                conn.close()
            if settings is not None:
                # The receiver polls this file; see open_lexa.settings.
                self.store.write_atomic(stt_settings.SETTINGS_PATH, json.dumps(
                    stt_settings.validate(settings), ensure_ascii=False, indent=1).encode())
            self.store.write_atomic(LEXICON_PATH, json.dumps(lexicon, ensure_ascii=False, indent=1).encode())
            self.store.write_atomic(DATASET_PATH, ''.join(
                json.dumps(row, ensure_ascii=False) + '\n' for row in dataset).encode())
            self.last_export = time.time()
            LOG.info('Exported lexicon (%d examples) and %d dataset rows', len(lexicon['examples']), len(dataset))

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
            wake_at = min(next_ingest, self.export_due if self.export_due is not None else next_ingest)
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
               transcriber_factory: Callable[[dict], Transcriber] = stt_settings.build_transcriber) -> FastAPI:
    conn = db.open_db(db_path)
    conn.close()
    worker = Worker(db_path, store, interval=interval)
    basic = HTTPBasic(realm='Open Lexa admin')

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
                      allowed=db.intent_entity_map(conn), saved=saved, warnings=warnings)

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

    @app.get('/settings', dependencies=protected, response_class=HTMLResponse)
    def settings_page(request: Request, saved: int = 0, error: str = '', conn=Depends(database)):
        stored = db.get_setting(conn, 'stt')
        current = {**stt_settings.DEFAULTS, **(stored or {})}
        key = current.pop('cloud_api_key')
        return render(request, 'settings.html', s=current, stored=stored is not None,
                      key_hint=f'…{key[-4:]}' if len(key) >= 12 else ('ustawiony' if key else ''),
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
