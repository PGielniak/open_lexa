"""Compare Base and Small Polish WER/latency on manually labeled recordings.

Run with a live Whisper API and archive credentials (SMB_* or
ADMIN_ARCHIVE_DIR). Never writes to the SQLite database or the archive.

    uv run python whisper-api/tools/compare_models.py --db /data/labels.sqlite3 \
        --url http://open-lexa-whisper:8080/v1 --limit 50
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import time

from dotenv import load_dotenv
from openai import OpenAI

from open_lexa.store import create_store
from open_lexa.text import normalize


def word_error_rate(reference: str, hypothesis: str) -> float:
    a, b = normalize(reference).split(), normalize(hypothesis).split()
    if not a:
        return float(bool(b))
    prev = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, other in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (word != other))
        prev = cur
    return prev[-1] / len(a)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True, help='Admin SQLite label database')
    parser.add_argument('--url', required=True, help='Whisper API /v1 URL')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--api-key', default='not-needed')
    args = parser.parse_args()
    load_dotenv()
    store = create_store()
    client = OpenAI(base_url=args.url, api_key=args.api_key, timeout=120, max_retries=1)
    connection = sqlite3.connect(f'file:{args.db}?mode=ro', uri=True)
    results = {model: [] for model in ('whisper-base', 'whisper-small')}
    try:
        rows = connection.execute(
            "SELECT audio_path, corrected_transcript FROM recordings "
            "WHERE status = 'labeled' AND corrected_transcript IS NOT NULL "
            "AND corrected_transcript != '' AND (annotation IS NULL OR "
            "json_extract(annotation, '$.source') != 'llm' OR "
            "json_extract(annotation, '$.reviewed_at') IS NOT NULL) "
            'ORDER BY recorded_at DESC LIMIT ?', (args.limit,)).fetchall()
        if not rows:
            parser.error('No human-labeled recordings with a corrected transcript')
        for audio_path, reference in rows:
            wav = store.read(audio_path)
            for model in results:
                start = time.monotonic()
                response = client.audio.transcriptions.create(
                    model=model, file=('command.wav', wav, 'audio/wav'), language='pl')
                latency = time.monotonic() - start
                results[model].append((word_error_rate(reference, response.text), latency))
        for model, scores in results.items():
            print(f'{model}: {len(scores)} clips; '
                  f'mean WER {sum(s[0] for s in scores) / len(scores):.3f}; '
                  f'mean latency {sum(s[1] for s in scores) / len(scores):.2f}s')
    finally:
        connection.close()
        client.close()
        store.close()


if __name__ == '__main__':
    main()
