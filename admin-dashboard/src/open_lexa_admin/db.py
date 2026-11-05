"""SQLite label store: schema, seed dictionary, ingest, labeling and export.

The database lives on the admin's own volume (never on SMB: SQLite locking over
SMB/CIFS is unreliable). Raw recordings stay on SMB; this file only references them.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from rapidfuzz import fuzz

from open_lexa.text import normalize, phonetic

LOG = logging.getLogger('open_lexa.admin')

DAY = re.compile(r'^\d{4}-\d{2}-\d{2}$')
STATUSES = ('new', 'labeled', 'skipped')

SCHEMA_V1 = """
CREATE TABLE intents (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    parent_id     INTEGER REFERENCES intents(id),
    description   TEXT,
    is_actionable INTEGER NOT NULL DEFAULT 1,
    enabled       INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE intent_phrases (
    id        INTEGER PRIMARY KEY,
    intent_id INTEGER NOT NULL REFERENCES intents(id) ON DELETE CASCADE,
    phrase    TEXT NOT NULL,
    UNIQUE (intent_id, phrase)
);
CREATE TABLE entity_types (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL CHECK (kind IN ('text', 'enum', 'number')),
    description TEXT
);
CREATE TABLE intent_entity_types (
    intent_id      INTEGER NOT NULL REFERENCES intents(id) ON DELETE CASCADE,
    entity_type_id INTEGER NOT NULL REFERENCES entity_types(id) ON DELETE CASCADE,
    required       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (intent_id, entity_type_id)
);
CREATE TABLE entity_values (
    id             INTEGER PRIMARY KEY,
    entity_type_id INTEGER NOT NULL REFERENCES entity_types(id) ON DELETE CASCADE,
    value          TEXT NOT NULL,
    UNIQUE (entity_type_id, value)
);
CREATE TABLE recordings (
    id                   INTEGER PRIMARY KEY,
    stem                 TEXT NOT NULL UNIQUE,
    audio_path           TEXT NOT NULL,
    recorded_at          TEXT NOT NULL,
    duration_seconds     REAL,
    session              TEXT,
    stt_backend          TEXT,
    stt_model            TEXT,
    language             TEXT,
    original_transcript  TEXT,
    stt_error            TEXT,
    predicted_intent     TEXT,
    predicted_entities   TEXT,
    predicted_score      REAL,
    predicted_source     TEXT,
    corrected_transcript TEXT,
    intent_id            INTEGER REFERENCES intents(id),
    status               TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'labeled', 'skipped')),
    labeled_at           TEXT,
    notes                TEXT,
    ingested_at          TEXT NOT NULL
);
CREATE INDEX recordings_status ON recordings (status, recorded_at);
CREATE INDEX recordings_intent ON recordings (intent_id);
CREATE TABLE entity_synonyms (
    id              INTEGER PRIMARY KEY,
    entity_value_id INTEGER NOT NULL REFERENCES entity_values(id) ON DELETE CASCADE,
    synonym         TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'manual' CHECK (source IN ('manual', 'learned')),
    recording_id    INTEGER REFERENCES recordings(id) ON DELETE SET NULL,
    UNIQUE (entity_value_id, synonym)
);
CREATE TABLE recording_entities (
    id                INTEGER PRIMARY KEY,
    recording_id      INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    entity_type_id    INTEGER NOT NULL REFERENCES entity_types(id),
    value             TEXT NOT NULL,
    entity_value_id   INTEGER REFERENCES entity_values(id),
    span_start        INTEGER,
    span_end          INTEGER,
    original_fragment TEXT,
    CHECK ((span_start IS NULL) = (span_end IS NULL))
);
"""
SCHEMA_V2 = """
CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
ALTER TABLE recordings ADD COLUMN origin TEXT;
"""
SCHEMA_V3 = """
ALTER TABLE recordings ADD COLUMN transcripts TEXT;
ALTER TABLE recordings ADD COLUMN corrected_source TEXT;
"""
SCHEMA_V4 = """
ALTER TABLE recordings ADD COLUMN annotation TEXT;
"""
MIGRATIONS = (SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4)

# (name, parent, actionable, description, entity types, phrases)
SEED_INTENTS = (
    ('play_music', None, 1, 'Odtwórz muzykę (źródło domyślne lub z encji source)', ('source',),
     ('włącz muzykę', 'puść muzykę', 'zagraj coś', 'play music')),
    ('play_spotify', 'play_music', 1, 'Odtwórz ze Spotify, opcjonalnie wyszukaj', ('search_query',),
     ('włącz spotify', 'puść {search_query} na spotify', 'włącz {search_query} ze spotify',
      'play {search_query} on spotify', 'puść playlistę {search_query}',
      'włącz playlistę {search_query}', 'puść {search_query} ze spotify')),
    ('play_radio', 'play_music', 1, 'Odtwórz stację radiową', ('station',),
     ('włącz radio', 'włącz {station}', 'puść radio {station}')),
    ('stop', None, 1, 'Zatrzymaj odtwarzanie', (), ('stop', 'wyłącz muzykę', 'zatrzymaj')),
    ('pause', None, 1, 'Pauza — zatrzymaj chwilowo', (), ('pauza', 'zatrzymaj chwilowo')),
    ('resume', None, 1, 'Wznów odtwarzanie', (), ('wznów', 'wznów muzykę', 'graj dalej')),
    ('next_track', None, 1, 'Następny utwór', (), ('następny utwór', 'zmień utwór', 'dalej')),
    ('previous_track', None, 1, 'Poprzedni utwór', (),
     ('poprzedni utwór', 'wróć do poprzedniej piosenki', 'cofnij utwór')),
    ('volume_up', None, 1, 'Głośniej', ('level',), ('głośniej', 'podgłośnij')),
    ('volume_down', None, 1, 'Ciszej', ('level',), ('ciszej', 'przycisz')),
    ('volume_set', None, 1, 'Ustaw głośność', ('level',), ('ustaw głośność na {level}', 'głośność {level}')),
    ('lights_on', None, 1, 'Włącz światło', ('room',), ('włącz światło', 'zapal światło w {room}')),
    ('lights_off', None, 1, 'Wyłącz światło', ('room',), ('wyłącz światło', 'zgaś światło w {room}')),
    ('unknown', None, 0, 'Sensowne polecenie, jeszcze nieobsługiwane', (), ()),
    ('noise', None, 0, 'Szum, urwane lub niezrozumiałe nagranie', (), ()),
    ('not_for_me', None, 0, 'Rozmowa w tle, fałszywe wybudzenie', (), ()),
)
SEED_ENTITY_TYPES = (
    ('search_query', 'text', 'Wykonawca, zespół, utwór, gatunek; słownik nazw własnych'),
    ('station', 'enum', 'Stacja radiowa'),
    ('level', 'number', 'Poziom lub krok głośności'),
    ('room', 'enum', 'Pomieszczenie'),
    ('source', 'enum', 'Źródło muzyki'),
)
SEED_VALUES = (
    ('station', 'RMF24', ('rmf', 'rmf24', 'rmf 24', 'erem ef')),
    ('source', 'spotify', ('spotify', 'spotyfaj', 'spotifaj')),
    ('source', 'radio', ('radio',)),
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


@contextmanager
def atomic(conn: sqlite3.Connection):
    """An independently rollbackable savepoint, including inside a transaction."""
    if not conn.in_transaction:
        with conn:
            yield
        return
    conn.execute('SAVEPOINT open_lexa_label')
    try:
        yield
    except BaseException:
        conn.execute('ROLLBACK TO SAVEPOINT open_lexa_label')
        conn.execute('RELEASE SAVEPOINT open_lexa_label')
        raise
    else:
        conn.execute('RELEASE SAVEPOINT open_lexa_label')


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute('PRAGMA busy_timeout = 10000')
    if str(path) != ':memory:':
        conn.execute('PRAGMA journal_mode = WAL')
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute('PRAGMA user_version').fetchone()[0]
    for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
        with conn:
            conn.executescript(f'BEGIN; {script}; PRAGMA user_version = {number}; COMMIT;')
        LOG.info('Database migrated to v%d', number)


def seed(conn: sqlite3.Connection) -> None:
    """Insert the starting dictionary; never overwrites rows edited in the UI."""
    with conn:
        for name, kind, description in SEED_ENTITY_TYPES:
            conn.execute('INSERT OR IGNORE INTO entity_types (name, kind, description) VALUES (?, ?, ?)',
                         (name, kind, description))
        for name, parent, actionable, description, entity_types, phrases in SEED_INTENTS:
            conn.execute(
                'INSERT OR IGNORE INTO intents (name, parent_id, is_actionable, description) '
                'VALUES (?, (SELECT id FROM intents WHERE name = ?), ?, ?)',
                (name, parent, actionable, description))
            for entity_type in entity_types:
                conn.execute(
                    'INSERT OR IGNORE INTO intent_entity_types (intent_id, entity_type_id) '
                    'SELECT i.id, t.id FROM intents i, entity_types t WHERE i.name = ? AND t.name = ?',
                    (name, entity_type))
            for phrase in phrases:
                conn.execute('INSERT OR IGNORE INTO intent_phrases (intent_id, phrase) '
                             'SELECT id, ? FROM intents WHERE name = ?', (phrase, name))
        for entity_type, value, synonyms in SEED_VALUES:
            value_id = ensure_value(conn, entity_type, value)
            for synonym in synonyms:
                add_synonym(conn, value_id, synonym)


def open_db(path: str | Path) -> sqlite3.Connection:
    conn = connect(path)
    migrate(conn)
    seed(conn)
    return conn


# --- Ingest -----------------------------------------------------------------------------

def ingest(conn: sqlite3.Connection, store, *, full: bool = True) -> int:
    """Add recordings whose WAV + JSON pair exists on the store. Idempotent.

    Incremental runs rescan only the newest known day and later ones, which keeps
    the once-a-minute poll cheap as the archive grows.
    """
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
            except (OSError, ValueError) as exc:
                LOG.warning('Skipping unreadable sidecar %s/%s: %s', day, name, exc)
                continue
            insert_recording(conn, stem, f'{day}/{stem}.wav', meta)
            known.add(stem)
            added += 1
    if added:
        LOG.info('Ingested %d new recording(s)', added)
    return added


def insert_recording(conn: sqlite3.Connection, stem: str, audio_path: str, meta: dict) -> None:
    entities = meta.get('entities')
    transcripts = meta.get('transcripts') if isinstance(meta.get('transcripts'), dict) else None
    local = (transcripts or {}).get('local')
    cloud = (transcripts or {}).get('cloud')
    local = local if isinstance(local, dict) else {}
    cloud = cloud if isinstance(cloud, dict) else {}
    # The local Whisper output is the raw baseline; GPT transcribe pre-fills the
    # "corrected" text so a human only has to add intent + entities.
    original = local['text'] if 'text' in local else meta.get('text')
    corrected = meta.get('corrected_text')
    if corrected is None and 'text' in cloud:
        corrected = cloud['text']
    if corrected is None:
        corrected = meta.get('text')
    corrected_source = meta.get('corrected_source')
    if corrected_source is None and transcripts:
        corrected_source = ('cloud' if 'text' in cloud else 'local' if 'text' in local else None)
    with conn:
        conn.execute(
            'INSERT OR IGNORE INTO recordings (stem, audio_path, recorded_at, duration_seconds, session, '
            'stt_backend, stt_model, language, original_transcript, stt_error, predicted_intent, '
            'predicted_entities, predicted_score, predicted_source, corrected_transcript, origin, '
            'transcripts, corrected_source, ingested_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (stem, audio_path, meta.get('created') or utcnow(), meta.get('duration_seconds'),
             meta.get('session'), meta.get('backend'), meta.get('model'), meta.get('language'),
             original, meta.get('error'), meta.get('intent', meta.get('command')),
             json.dumps(entities, ensure_ascii=False) if entities is not None else None,
             meta.get('score'), meta.get('source'), corrected, meta.get('origin'),
             json.dumps(transcripts, ensure_ascii=False) if transcripts else None,
             corrected_source, utcnow()))


# --- Settings ---------------------------------------------------------------------------

def get_setting(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute('SELECT value FROM settings WHERE key = ?', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def set_setting(conn: sqlite3.Connection, key: str, value: dict) -> None:
    with conn:
        conn.execute('INSERT INTO settings (key, value) VALUES (?, ?) '
                     'ON CONFLICT (key) DO UPDATE SET value = excluded.value',
                     (key, json.dumps(value, ensure_ascii=False)))


# --- Queries ----------------------------------------------------------------------------

def list_recordings(conn: sqlite3.Connection, *, status: str | None = None, intent: str | None = None,
                    limit: int = 50, offset: int = 0) -> list[sqlite3.Row]:
    where, params = [], []
    if status:
        where.append('r.status = ?')
        params.append(status)
    if intent:
        where.append('i.name = ?')
        params.append(intent)
    sql = ('SELECT r.*, i.name AS intent FROM recordings r LEFT JOIN intents i ON i.id = r.intent_id '
           + (f'WHERE {" AND ".join(where)} ' if where else '')
           + 'ORDER BY r.recorded_at DESC, r.id DESC LIMIT ? OFFSET ?')
    return conn.execute(sql, (*params, limit, offset)).fetchall()


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    result = dict.fromkeys(STATUSES, 0)
    result.update(conn.execute('SELECT status, count(*) FROM recordings GROUP BY status').fetchall())
    return result


def get_recording(conn: sqlite3.Connection, recording_id: int) -> sqlite3.Row | None:
    return conn.execute('SELECT r.*, i.name AS intent FROM recordings r '
                        'LEFT JOIN intents i ON i.id = r.intent_id WHERE r.id = ?',
                        (recording_id,)).fetchone()


def transcript_map(value: str | None) -> dict:
    """Parse the sidecar's per-backend transcript map stored on a recording."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except ValueError:
        LOG.warning('Ignoring invalid transcripts JSON in recordings')
        return {}
    return parsed if isinstance(parsed, dict) else {}


def recording_entities(conn: sqlite3.Connection, recording_id: int) -> list[sqlite3.Row]:
    return conn.execute('SELECT e.*, t.name AS type, t.kind FROM recording_entities e '
                        'JOIN entity_types t ON t.id = e.entity_type_id WHERE e.recording_id = ? '
                        'ORDER BY coalesce(e.span_start, 1e9), e.id', (recording_id,)).fetchall()


def next_unlabeled(conn: sqlite3.Connection, after_id: int | None = None) -> int | None:
    row = conn.execute("SELECT id FROM recordings WHERE status = 'new' AND id != coalesce(?, -1) "
                       'ORDER BY recorded_at DESC, id DESC LIMIT 1', (after_id,)).fetchone()
    return row[0] if row else None


def intents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Intents in tree order with depth, for indented pickers."""
    rows = conn.execute('SELECT i.*, p.name AS parent FROM intents i '
                        'LEFT JOIN intents p ON p.id = i.parent_id ORDER BY i.id').fetchall()
    children: dict[int | None, list[sqlite3.Row]] = {}
    for row in rows:
        children.setdefault(row['parent_id'], []).append(row)
    ordered: list[dict] = []

    def walk(parent: int | None, depth: int) -> None:
        for row in children.get(parent, []):
            ordered.append({**dict(row), 'depth': depth})
            walk(row['id'], depth + 1)

    walk(None, 0)
    return ordered


def entity_types(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute('SELECT * FROM entity_types ORDER BY id').fetchall()


def intent_entity_map(conn: sqlite3.Connection) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for intent, entity_type in conn.execute(
            'SELECT i.name, t.name FROM intent_entity_types x JOIN intents i ON i.id = x.intent_id '
            'JOIN entity_types t ON t.id = x.entity_type_id ORDER BY t.id'):
        result.setdefault(intent, []).append(entity_type)
    return result


def phrases(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute('SELECT p.*, i.name AS intent FROM intent_phrases p '
                        'JOIN intents i ON i.id = p.intent_id ORDER BY i.id, p.id').fetchall()


def dictionary(conn: sqlite3.Connection) -> list[dict]:
    """Entity values with synonyms, grouped per entity type."""
    values = conn.execute('SELECT v.*, t.name AS type FROM entity_values v '
                          'JOIN entity_types t ON t.id = v.entity_type_id ORDER BY t.id, v.value').fetchall()
    synonyms: dict[int, list[sqlite3.Row]] = {}
    for row in conn.execute('SELECT * FROM entity_synonyms ORDER BY synonym'):
        synonyms.setdefault(row['entity_value_id'], []).append(row)
    return [{**dict(value), 'synonyms': synonyms.get(value['id'], [])} for value in values]


# --- Labeling ---------------------------------------------------------------------------

def ensure_value(conn: sqlite3.Connection, entity_type: str, value: str) -> int:
    value = value.strip()
    row = conn.execute('SELECT v.id FROM entity_values v JOIN entity_types t ON t.id = v.entity_type_id '
                       'WHERE t.name = ? AND lower(v.value) = lower(?)', (entity_type, value)).fetchone()
    if row:
        return row[0]
    cursor = conn.execute('INSERT INTO entity_values (entity_type_id, value) '
                          'SELECT id, ? FROM entity_types WHERE name = ?', (value, entity_type))
    if cursor.rowcount != 1:
        raise ValueError(f'unknown entity type {entity_type!r}')
    return cursor.lastrowid


def add_synonym(conn: sqlite3.Connection, value_id: int, synonym: str, *,
                source: str = 'manual', recording_id: int | None = None) -> bool:
    synonym = normalize(synonym)
    if not synonym:
        return False
    cursor = conn.execute('INSERT OR IGNORE INTO entity_synonyms (entity_value_id, synonym, source, recording_id) '
                          'VALUES (?, ?, ?, ?)', (value_id, synonym, source, recording_id))
    return cursor.rowcount == 1


def find_span(text: str, value: str) -> tuple[int, int] | None:
    start = text.lower().find(value.lower()) if value else -1
    return (start, start + len(value)) if start >= 0 else None


def suggest_fragment(original: str | None, value: str) -> str:
    """The part of the STT transcript that most likely was meant as ``value``."""
    if not original or not value.strip():
        return ''
    lowered = original.lower()
    alignment = fuzz.partial_ratio_alignment(value.lower(), lowered)
    if alignment is None or alignment.score < 40:
        return ''
    start, end = alignment.dest_start, alignment.dest_end
    # Widen to whole words; STT fragments rarely align on word boundaries.
    while start > 0 and lowered[start - 1].isalnum():
        start -= 1
    while end < len(lowered) and lowered[end].isalnum():
        end += 1
    return original[start:end].strip(' ,.!?')


def save_label(conn: sqlite3.Connection, recording_id: int, *, corrected_transcript: str,
               intent: str | None, entities: Iterable[dict], status: str = 'labeled',
               notes: str | None = None) -> list[str]:
    """Store a human label; returns warnings to show next to the form.

    Each entity is ``{'type', 'value', 'original_fragment'}``. Text/enum values join
    the dictionary, and an original fragment that sounds different becomes a learned
    synonym, so the next time STT mishears the same name the receiver can fix it.
    """
    if status not in STATUSES:
        raise ValueError(f'invalid status {status!r}')
    warnings: list[str] = []
    corrected = corrected_transcript.strip()
    with atomic(conn):
        intent_id = None
        if intent:
            row = conn.execute('SELECT id FROM intents WHERE name = ?', (intent,)).fetchone()
            if row is None:
                raise ValueError(f'unknown intent {intent!r}')
            intent_id = row[0]
        elif status == 'labeled':
            raise ValueError('a labeled recording needs an intent')
        conn.execute('UPDATE recordings SET corrected_transcript = ?, intent_id = ?, status = ?, notes = ?, '
                     'labeled_at = ? WHERE id = ?',
                     (corrected or None, intent_id, status, notes or None, utcnow(), recording_id))
        conn.execute('DELETE FROM recording_entities WHERE recording_id = ?', (recording_id,))
        # Re-derived below, so a corrected fragment does not leave a wrong synonym behind.
        conn.execute("DELETE FROM entity_synonyms WHERE source = 'learned' AND recording_id = ?",
                     (recording_id,))
        for entity in entities:
            entity_type, value = entity.get('type', ''), (entity.get('value') or '').strip()
            if not entity_type or not value:
                continue
            kind_row = conn.execute('SELECT id, kind FROM entity_types WHERE name = ?', (entity_type,)).fetchone()
            if kind_row is None:
                raise ValueError(f'unknown entity type {entity_type!r}')
            fragment = (entity.get('original_fragment') or '').strip() or None
            span = find_span(corrected, value)
            if span is None:
                warnings.append(f'„{value}” ({entity_type}) nie występuje w poprawionej transkrypcji; '
                                'zapisano bez pozycji.')
            value_id = None
            if kind_row['kind'] in ('text', 'enum'):
                value_id = ensure_value(conn, entity_type, value)
                if fragment and phonetic(fragment) != phonetic(value):
                    add_synonym(conn, value_id, fragment, source='learned', recording_id=recording_id)
            conn.execute('INSERT INTO recording_entities (recording_id, entity_type_id, value, entity_value_id, '
                         'span_start, span_end, original_fragment) VALUES (?, ?, ?, ?, ?, ?, ?)',
                         (recording_id, kind_row['id'], value, value_id, *(span or (None, None)), fragment))
    return warnings


# --- LLM annotation ---------------------------------------------------------------------

INTENT_NAME = re.compile(r'^[a-z][a-z0-9_]{1,40}$')
MAX_ANNOTATION_ATTEMPTS = 3


def set_annotation(conn: sqlite3.Connection, recording_id: int, annotation: dict) -> None:
    with atomic(conn):
        conn.execute('UPDATE recordings SET annotation = ? WHERE id = ?',
                     (json.dumps(annotation, ensure_ascii=False), recording_id))


def annotation_of(row) -> dict:
    """Parse a recording row's annotation column."""
    try:
        parsed = json.loads(row['annotation']) if row['annotation'] else None
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def pending_annotation(conn: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    """'new' recordings with a transcript that the LLM has not (yet) labeled.

    Failed annotations are retried up to :data:`MAX_ANNOTATION_ATTEMPTS` times.
    """
    return conn.execute(
        'SELECT * FROM recordings WHERE status = \'new\' '
        "AND coalesce(corrected_transcript, original_transcript) IS NOT NULL "
        "AND coalesce(corrected_transcript, original_transcript) != '' "
        'AND (annotation IS NULL '
        "     OR (json_extract(annotation, '$.status') = 'error' "
        f"         AND coalesce(json_extract(annotation, '$.attempts'), 0) < {MAX_ANNOTATION_ATTEMPTS})) "
        'ORDER BY recorded_at, id LIMIT ?', (limit,)).fetchall()


def ensure_intent(conn: sqlite3.Connection, name: str, *, description: str | None = None,
                  phrases: Iterable[str] = (), parent: str | None = None) -> int:
    """Return the intent id, creating a new (actionable) intent when missing."""
    row = conn.execute('SELECT id FROM intents WHERE name = ?', (name,)).fetchone()
    if row:
        return row[0]
    with atomic(conn):
        cursor = conn.execute(
            'INSERT INTO intents (name, parent_id, description, is_actionable) '
            'VALUES (?, (SELECT id FROM intents WHERE name = ?), ?, 1)',
            (name, parent, description or f'Utworzona automatycznie (LLM) {utcnow()}'))
        intent_id = cursor.lastrowid
        for phrase in phrases:
            phrase = str(phrase).strip()
            if phrase:
                conn.execute('INSERT OR IGNORE INTO intent_phrases (intent_id, phrase) VALUES (?, ?)',
                             (intent_id, phrase))
    LOG.info('LLM created intent %r', name)
    return intent_id


def ensure_entity_type(conn: sqlite3.Connection, name: str, kind: str = 'text',
                       description: str | None = None) -> int:
    """Return the entity type id, creating it when missing."""
    row = conn.execute('SELECT id FROM entity_types WHERE name = ?', (name,)).fetchone()
    if row:
        return row[0]
    if kind not in ('text', 'enum', 'number'):
        kind = 'text'
    with atomic(conn):
        cursor = conn.execute('INSERT INTO entity_types (name, kind, description) VALUES (?, ?, ?)',
                              (name, kind, description or f'Utworzony automatycznie (LLM) {utcnow()}'))
    LOG.info('LLM created entity type %r (%s)', name, kind)
    return cursor.lastrowid


# --- Export -----------------------------------------------------------------------------

def export(conn: sqlite3.Connection) -> tuple[dict, list[dict]]:
    """Return (lexicon for the receiver, dataset rows for retraining)."""
    allowed = intent_entity_map(conn)
    intent_rows = conn.execute('SELECT i.*, p.name AS parent FROM intents i '
                               'LEFT JOIN intents p ON p.id = i.parent_id WHERE i.enabled = 1 ORDER BY i.id').fetchall()
    phrase_map: dict[str, list[str]] = {}
    for row in phrases(conn):
        phrase_map.setdefault(row['intent'], []).append(row['phrase'])
    entity_dict: dict[str, dict] = {row['name']: {'kind': row['kind'], 'values': {}}
                                    for row in entity_types(conn)}
    for value in dictionary(conn):
        entity_dict[value['type']]['values'][value['value']] = [s['synonym'] for s in value['synonyms']]

    examples, dataset = [], []
    labeled = conn.execute("SELECT r.*, i.name AS intent, p.name AS parent FROM recordings r "
                           "JOIN intents i ON i.id = r.intent_id LEFT JOIN intents p ON p.id = i.parent_id "
                           "WHERE r.status = 'labeled' ORDER BY r.recorded_at, r.id").fetchall()
    for row in labeled:
        entities = [{'type': e['type'], 'value': e['value'], 'span': [e['span_start'], e['span_end']]
                     if e['span_start'] is not None else None, 'original_fragment': e['original_fragment']}
                    for e in recording_entities(conn, row['id'])]
        transcripts = transcript_map(row['transcripts'])
        local = transcripts.get('local') if isinstance(transcripts.get('local'), dict) else {}
        cloud = transcripts.get('cloud') if isinstance(transcripts.get('cloud'), dict) else {}
        if row['original_transcript']:
            examples.append({'text': row['original_transcript'], 'phonetic': phonetic(row['original_transcript']),
                             'intent': row['intent'],
                             'entities': [{k: e[k] for k in ('type', 'value', 'original_fragment')}
                                          for e in entities]})
        dataset.append({'audio_path': row['audio_path'], 'recorded_at': row['recorded_at'],
                         'duration_seconds': row['duration_seconds'], 'stt_model': row['stt_model'],
                         'language': row['language'], 'original_transcript': row['original_transcript'],
                         'local_transcript': local.get('text'), 'cloud_transcript': cloud.get('text'),
                         'corrected_transcript': row['corrected_transcript'],
                         'corrected_source': row['corrected_source'], 'intent': row['intent'],
                         'parent_intent': row['parent'], 'entities': entities,
                         'annotation': annotation_of(row) or None})
    lexicon = {
        'version': 1,
        'generated_at': utcnow(),
        'intents': [{'name': row['name'], 'parent': row['parent'], 'actionable': bool(row['is_actionable']),
                     'entities': allowed.get(row['name'], []), 'phrases': phrase_map.get(row['name'], [])}
                    for row in intent_rows],
        'entities': entity_dict,
        'examples': examples,
    }
    return lexicon, dataset
