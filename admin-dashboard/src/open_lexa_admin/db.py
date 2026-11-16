"""SQLite recording index and reviewed-transcript export.

Existing databases are upgraded additively. Old tables and columns remain
untouched so historical labels are not deleted.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sqlite3

LOG = logging.getLogger('open_lexa.admin')
DAY = re.compile(r'^\d{4}-\d{2}-\d{2}$')
STATUSES = ('new', 'labeled', 'skipped')


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute('PRAGMA busy_timeout = 10000')
    if str(path) != ':memory:':
        conn.execute('PRAGMA journal_mode = WAL')
    return conn


def open_db(path: str | Path) -> sqlite3.Connection:
    conn = connect(path)
    with conn:
        conn.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        conn.execute('''CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY, stem TEXT NOT NULL UNIQUE, audio_path TEXT NOT NULL,
            recorded_at TEXT NOT NULL, duration_seconds REAL, session TEXT,
            stt_backend TEXT, stt_model TEXT, language TEXT, original_transcript TEXT,
            stt_error TEXT, predicted_intent TEXT, predicted_entities TEXT,
            predicted_score REAL, predicted_source TEXT, corrected_transcript TEXT,
            status TEXT NOT NULL DEFAULT 'new', labeled_at TEXT, notes TEXT,
            ingested_at TEXT NOT NULL, origin TEXT, transcripts TEXT, corrected_source TEXT,
            annotation TEXT, agent TEXT
        )''')
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(recordings)')}
        for name in ('origin', 'transcripts', 'corrected_source', 'annotation', 'agent'):
            if name not in columns:
                conn.execute(f'ALTER TABLE recordings ADD COLUMN {name} TEXT')
        conn.execute('CREATE INDEX IF NOT EXISTS recordings_status ON recordings (status, recorded_at)')
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        if version < 5:
            conn.execute('PRAGMA user_version = 5')
    return conn


def get_setting(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute('SELECT value FROM settings WHERE key = ?', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def set_setting(conn: sqlite3.Connection, key: str, value: dict) -> None:
    with conn:
        conn.execute('INSERT INTO settings (key, value) VALUES (?, ?) '
                     'ON CONFLICT (key) DO UPDATE SET value = excluded.value',
                     (key, json.dumps(value, ensure_ascii=False)))


def transcript_map(value: str | None) -> dict:
    try:
        result = json.loads(value) if value else {}
        return result if isinstance(result, dict) else {}
    except ValueError:
        return {}


def annotation_of(row) -> dict:
    return transcript_map(row['annotation'])


def set_annotation(conn, recording_id: int, annotation: dict) -> None:
    with conn:
        conn.execute('UPDATE recordings SET annotation = ? WHERE id = ?',
                     (json.dumps(annotation, ensure_ascii=False), recording_id))


def insert_recording(conn: sqlite3.Connection, stem: str, audio_path: str, meta: dict) -> None:
    transcripts = meta.get('transcripts') if isinstance(meta.get('transcripts'), dict) else {}
    local = transcripts.get('local') if isinstance(transcripts.get('local'), dict) else {}
    cloud = transcripts.get('cloud') if isinstance(transcripts.get('cloud'), dict) else {}
    original = local['text'] if 'text' in local else meta.get('text')
    corrected = meta.get('corrected_text')
    if corrected is None:
        corrected = cloud.get('text', meta.get('text'))
    source = meta.get('corrected_source')
    if source is None and transcripts:
        source = 'cloud' if 'text' in cloud else 'local' if 'text' in local else None
    with conn:
        conn.execute('''INSERT OR IGNORE INTO recordings
            (stem, audio_path, recorded_at, duration_seconds, session, stt_backend, stt_model,
             language, original_transcript, stt_error, predicted_intent, predicted_entities,
             predicted_score, predicted_source, corrected_transcript, origin, transcripts,
             corrected_source, agent, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (stem, audio_path, meta.get('created') or utcnow(), meta.get('duration_seconds'),
             meta.get('session'), meta.get('backend'), meta.get('model'), meta.get('language'),
             original, meta.get('error'), meta.get('intent', meta.get('command')),
             json.dumps(meta['entities'], ensure_ascii=False) if meta.get('entities') is not None else None,
             meta.get('score'), meta.get('source'), corrected, meta.get('origin'),
             json.dumps(transcripts, ensure_ascii=False) if transcripts else None, source,
             json.dumps(meta['agent'], ensure_ascii=False) if meta.get('agent') else None, utcnow()))


def ingest(conn, store, *, full: bool = True) -> int:
    days = sorted(day for day in store.list_dir('') if DAY.match(day))
    if not full:
        latest = conn.execute('SELECT max(substr(audio_path, 1, 10)) FROM recordings').fetchone()[0]
        if latest:
            days = [day for day in days if day >= latest]
    known = {row[0] for row in conn.execute('SELECT stem FROM recordings')}
    added = 0
    for day in days:
        names = set(store.list_dir(day))
        for name in sorted(names):
            stem, _, suffix = name.rpartition('.')
            if suffix != 'json' or stem in known or f'{stem}.wav' not in names:
                continue
            try:
                meta = json.loads(store.read(f'{day}/{name}').decode('utf-8'))
                if not isinstance(meta, dict):
                    raise ValueError('sidecar is not an object')
            except (OSError, ValueError) as exc:
                LOG.warning('Skipping unreadable sidecar %s/%s: %s', day, name, exc)
                continue
            insert_recording(conn, stem, f'{day}/{stem}.wav', meta)
            known.add(stem)
            added += 1
    return added


def list_recordings(conn, *, status: str | None = None, limit: int = 50, offset: int = 0):
    where = 'WHERE status = ? ' if status else ''
    params = (status,) if status else ()
    return conn.execute('SELECT * FROM recordings ' + where +
                        'ORDER BY recorded_at DESC, id DESC LIMIT ? OFFSET ?',
                        (*params, limit, offset)).fetchall()


def counts(conn) -> dict[str, int]:
    result = dict.fromkeys(STATUSES, 0)
    result.update(conn.execute('SELECT status, count(*) FROM recordings GROUP BY status').fetchall())
    return result


def get_recording(conn, recording_id: int):
    return conn.execute('SELECT * FROM recordings WHERE id = ?', (recording_id,)).fetchone()


def next_unlabeled(conn, after_id: int | None = None) -> int | None:
    row = conn.execute("SELECT id FROM recordings WHERE status = 'new' AND id != coalesce(?, -1) "
                       'ORDER BY recorded_at DESC, id DESC LIMIT 1', (after_id,)).fetchone()
    return row[0] if row else None


def export_dataset(conn) -> list[dict]:
    dataset = []
    for row in conn.execute("SELECT * FROM recordings WHERE status = 'labeled' ORDER BY recorded_at, id"):
        transcripts = transcript_map(row['transcripts'])
        local = transcripts.get('local') if isinstance(transcripts.get('local'), dict) else {}
        cloud = transcripts.get('cloud') if isinstance(transcripts.get('cloud'), dict) else {}
        dataset.append({'audio_path': row['audio_path'], 'recorded_at': row['recorded_at'],
                        'duration_seconds': row['duration_seconds'], 'stt_model': row['stt_model'],
                        'language': row['language'], 'original_transcript': row['original_transcript'],
                        'local_transcript': local.get('text'), 'cloud_transcript': cloud.get('text'),
                        'corrected_transcript': row['corrected_transcript'],
                        'corrected_source': row['corrected_source'],
                        'annotation': annotation_of(row) or None})
    return dataset
