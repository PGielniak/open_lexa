"""Export labeled recordings as a Hugging Face ``audiofolder`` dataset.

Reads the admin SQLite database (read-only) and the WAVs from the archive
(``SMB_*`` or ``ADMIN_ARCHIVE_DIR``, like the receiver) and writes::

    OUT/train/metadata.csv   file_name,transcription,source,recorded_at,duration_seconds
    OUT/train/<stem>.wav
    OUT/test/...

The test split is chosen by date (``--test-days``, the newest days), never
randomly, so near-duplicate commands from one session do not leak between the
splits. Review the test split by ear: most labels are GPT transcripts, and a WER
against them only measures agreement with GPT.

    uv run python whisper-api/tools/export_finetune_dataset.py \\
        --db labels.sqlite3 --out ~/whisper-data --test-days 7
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sqlite3

QUERY = '''SELECT stem, audio_path, recorded_at, duration_seconds, corrected_transcript,
                  corrected_source FROM recordings
           WHERE status = 'labeled' AND stt_error IS NULL
             AND trim(coalesce(corrected_transcript, '')) != ''
             AND (? = 0 OR duration_seconds <= ?)
           ORDER BY recorded_at, id'''


def select_rows(connection: sqlite3.Connection, *, max_seconds: float = 0) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    return connection.execute(QUERY, (max_seconds, max_seconds)).fetchall()


def split_by_day(rows, test_days: int) -> dict[str, list]:
    days = sorted({row['recorded_at'][:10] for row in rows})
    # Keep at least the oldest day for training.
    test_days = min(test_days, len(days) - 1)
    test = set(days[-test_days:]) if test_days > 0 else set()
    splits: dict[str, list] = {'train': [], 'test': []}
    for row in rows:
        splits['test' if row['recorded_at'][:10] in test else 'train'].append(row)
    return splits


def export(connection: sqlite3.Connection, store, out: Path, *, test_days: int,
           max_seconds: float = 0) -> dict[str, int]:
    splits = split_by_day(select_rows(connection, max_seconds=max_seconds), test_days)
    counts = {}
    for name, rows in splits.items():
        if not rows:
            continue
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'metadata.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['file_name', 'transcription', 'source', 'recorded_at', 'duration_seconds'])
            for row in rows:
                (folder / f'{row["stem"]}.wav').write_bytes(store.read(row['audio_path']))
                writer.writerow([f'{row["stem"]}.wav', row['corrected_transcript'].strip(),
                                 row['corrected_source'] or '', row['recorded_at'],
                                 row['duration_seconds']])
        counts[name] = len(rows)
    return counts


def main() -> None:
    from dotenv import load_dotenv
    from open_lexa.store import create_store

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', type=Path, required=True, help='Admin SQLite label database')
    parser.add_argument('--out', type=Path, required=True, help='Output directory (audiofolder)')
    parser.add_argument('--test-days', type=int, default=7, help='Newest N recording days go to test')
    parser.add_argument('--max-seconds', type=float, default=0,
                        help='Drop longer clips, e.g. 5 for the hailo-whisper base export (0 keeps all)')
    args = parser.parse_args()
    load_dotenv()
    store = create_store()
    connection = sqlite3.connect(f'file:{args.db}?mode=ro', uri=True)
    try:
        counts = export(connection, store, args.out, test_days=args.test_days,
                        max_seconds=args.max_seconds)
    finally:
        connection.close()
        store.close()
    if not counts:
        parser.error('no labeled recordings with a transcript')
    print(', '.join(f'{name}: {count}' for name, count in counts.items()), '->', args.out)


if __name__ == '__main__':
    main()
