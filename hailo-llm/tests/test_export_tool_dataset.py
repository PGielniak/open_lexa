import importlib.util
import json
from pathlib import Path
import sqlite3

SPEC = importlib.util.spec_from_file_location(
    'export_tool_dataset', Path(__file__).parents[1] / 'tools' / 'export_tool_dataset.py')
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def make_db():
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE recordings (id INTEGER PRIMARY KEY, recorded_at TEXT, status TEXT, '
                 'corrected_transcript TEXT, transcripts TEXT, agent TEXT)')
    radio = {'status': 'handled', 'executions': [
        {'tool': 'play_radio', 'arguments': {'station': 'Radio Zet'}, 'status': 'handled'}]}
    failed_exec = {'status': 'failed', 'executions': [
        {'tool': 'play_spotify', 'arguments': {'search_query': 'drum and bass'}, 'status': 'failed'}]}
    rows = [
        ('2026-11-01', 'labeled', 'Włącz Radio Zet.', {'local': {'text': 'włącz radio zed'}}, radio),
        ('2026-11-01', 'labeled', 'Włącz drum and bass.', None, failed_exec),
        ('2026-11-02', 'labeled', 'Ale pogoda.', None, {'status': 'no_tool', 'executions': []}),
        ('2026-11-02', 'labeled', 'zepsute', None, {'status': 'failed', 'error': 'x', 'executions': []}),
        ('2026-11-02', 'skipped', 'pominięte', None, radio),
        ('2026-11-02', 'labeled', 'Głośność 500', None, {'status': 'handled', 'executions': [
            {'tool': 'volume_set', 'arguments': {'level': 500}}]}),
    ]
    for day, status, text, transcripts, agent in rows:
        conn.execute('INSERT INTO recordings (recorded_at, status, corrected_transcript, transcripts, agent) '
                     'VALUES (?,?,?,?,?)', (day + 'T10:00', status, text,
                                             json.dumps(transcripts) if transcripts else None, json.dumps(agent)))
    return conn


def turns(rows):
    return [(r['messages'][1]['content'], json.loads(r['messages'][2]['content'])) for r in rows]


def test_build_uses_agent_decisions_and_local_transcripts():
    tools = exporter.default_tools()
    splits = exporter.build(make_db(), tools, test_days=1)
    assert turns(splits['train']) == [
        ('Włącz Radio Zet.', {'name': 'play_radio', 'arguments': {'station': 'Radio Zet'}}),
        ('włącz radio zed', {'name': 'play_radio', 'arguments': {'station': 'Radio Zet'}}),
        ('Włącz drum and bass.', {'name': 'play_spotify', 'arguments': {'search_query': 'drum and bass'}}),
    ]
    # LLM failures, skipped rows and schema-invalid arguments are dropped.
    assert turns(splits['test']) == [('Ale pogoda.', {'name': None})]
    system = splits['train'][0]['messages'][0]['content']
    assert '- play_radio(station: string)' in system and 'volume_up(level?: integer)' in system


def test_extra_examples_are_validated_and_train_only():
    extra = [{'text': 'ciszej o pięć', 'name': 'volume_down', 'arguments': {'level': 5}},
             {'text': 'zła', 'name': 'volume_set', 'arguments': {}},
             {'text': 'nie ma', 'name': 'self_destruct', 'arguments': {}}]
    splits = exporter.build(make_db(), exporter.default_tools(), test_days=1, extra=extra)
    assert turns(splits['train'])[-1] == ('ciszej o pięć', {'name': 'volume_down', 'arguments': {'level': 5}})
    assert len(splits['train']) == 4 and len(splits['test']) == 1
