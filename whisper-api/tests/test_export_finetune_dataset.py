import csv
import importlib.util
from pathlib import Path
import sqlite3

SPEC = importlib.util.spec_from_file_location(
    'export_finetune_dataset', Path(__file__).parents[1] / 'tools' / 'export_finetune_dataset.py')
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class DictStore:
    def __init__(self, files):
        self.files = files

    def read(self, rel):
        return self.files[rel]


def make_db():
    conn = sqlite3.connect(':memory:')
    conn.execute('''CREATE TABLE recordings (id INTEGER PRIMARY KEY, stem TEXT, audio_path TEXT,
        recorded_at TEXT, duration_seconds REAL, corrected_transcript TEXT, corrected_source TEXT,
        status TEXT, stt_error TEXT)''')
    rows = [
        ('a', '2026-11-01', 2.0, 'Włącz radio.', 'cloud', 'labeled', None),
        ('b', '2026-11-02', 8.0, 'Ciszej.', 'cloud', 'labeled', None),
        ('c', '2026-11-03', 3.0, 'Stop.', None, 'labeled', None),
        ('d', '2026-11-03', 3.0, 'pominięte', 'cloud', 'skipped', None),
        ('e', '2026-11-03', 3.0, '  ', 'cloud', 'labeled', None),
        ('f', '2026-11-03', 3.0, 'x', 'cloud', 'labeled', 'stt down'),
    ]
    for stem, day, seconds, text, source, status, error in rows:
        conn.execute('INSERT INTO recordings (stem, audio_path, recorded_at, duration_seconds, '
                     'corrected_transcript, corrected_source, status, stt_error) VALUES (?,?,?,?,?,?,?,?)',
                     (stem, f'{day}/{stem}.wav', day + 'T10:00:00', seconds, text, source, status, error))
    store = DictStore({f'{day}/{stem}.wav': stem.encode() for stem, day, *_ in rows})
    return conn, store


def read_split(path):
    with (path / 'metadata.csv').open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def test_export_splits_by_day_and_skips_unusable_rows(tmp_path):
    conn, store = make_db()
    counts = exporter.export(conn, store, tmp_path, test_days=1)
    assert counts == {'train': 2, 'test': 1}
    train, test = read_split(tmp_path / 'train'), read_split(tmp_path / 'test')
    assert [r['transcription'] for r in train] == ['Włącz radio.', 'Ciszej.']
    assert test[0]['file_name'] == 'c.wav' and test[0]['source'] == ''
    assert (tmp_path / 'train' / 'a.wav').read_bytes() == b'a'


def test_export_can_drop_long_clips(tmp_path):
    conn, store = make_db()
    assert exporter.export(conn, store, tmp_path, test_days=0, max_seconds=5) == {'train': 2}
