"""LLM annotation, provenance, dictionary growth, and admin controls."""
import base64
import json

import pytest
from fastapi.testclient import TestClient

from open_lexa.store import DirStore
from open_lexa_admin import db, llm
from open_lexa_admin.app import create_app

AUTH = {'Authorization': 'Basic ' + base64.b64encode(b'admin:correct-horse-battery').decode()}


class FakeAnnotator:
    calls = []

    def __init__(self, settings):
        assert settings['api_key'] == 'secret'

    def annotate(self, transcript, dictionary):
        self.calls.append(transcript)
        assert any(item['name'] == 'play_music' for item in dictionary['intents'])
        return llm.parse_answer(json.dumps({
            'intent': 'play_podcast',
            'new_intent': {'name': 'play_podcast', 'description': 'Graj podcast',
                           'phrases': ['puść podcast {search_query}']},
            'entities': [{'type': 'search_query', 'value': 'Technologia',
                          'fragment': 'Technologia'}],
            'new_entity_types': [], 'notes': 'Podcast',
        }))

    def test(self):
        return 'OK'


def test_manual_annotation_creates_intent_and_marks_provenance(tmp_path):
    store = DirStore(tmp_path / 'archive')
    app = create_app(db_path=str(tmp_path / 'labels.sqlite3'), store=store, username='admin',
                     password='correct-horse-battery', background=False,
                     annotator_factory=FakeAnnotator)
    with TestClient(app) as client:
        conn = db.connect(app.state.worker.db_path)
        db.insert_recording(conn, '1', '2026-11-02/1.wav',
                            {'created': '2026-11-02T00:00:00Z', 'text': 'Puść podcast Technologia'})
        conn.close()
        response = client.post('/settings/llm', headers=AUTH, data={
            'url': 'https://llm.example/v1', 'model': 'my-model', 'api_key': 'secret'})
        assert response.status_code == 200
        assert 'value="secret"' not in client.get('/settings', headers=AUTH).text
        result = client.post('/recordings/1/annotate', headers=AUTH, follow_redirects=False)
        assert result.status_code == 303 and 'llm=play_podcast' in result.headers['location']
        conn = db.connect(app.state.worker.db_path)
        row = db.get_recording(conn, 1)
        assert row['status'] == 'labeled' and row['intent'] == 'play_podcast'
        assert db.annotation_of(row)['model'] == 'my-model'
        assert [(e['type'], e['value']) for e in db.recording_entities(conn, 1)] == [
            ('search_query', 'Technologia')]
        lexicon, dataset = db.export(conn)
        assert 'play_podcast' in {i['name'] for i in lexicon['intents']}
        assert 'search_query' in next(i for i in lexicon['intents']
                                      if i['name'] == 'play_podcast')['entities']
        assert dataset[0]['intent'] == 'play_podcast'
        conn.close()
        assert client.post('/recordings/1/annotate', headers=AUTH,
                           follow_redirects=False).status_code == 303
        assert len(FakeAnnotator.calls) >= 1


def test_annotation_is_atomic_when_model_uses_unknown_entity(tmp_path):
    conn = db.open_db(tmp_path / 'labels.sqlite3')
    db.insert_recording(conn, '1', '2026-11-02/1.wav',
                        {'created': '2026-11-02T00:00:00Z', 'text': 'Graj podcast'})
    answer = llm.parse_answer(json.dumps({
        'intent': 'play_podcast',
        'new_intent': {'name': 'play_podcast', 'phrases': ['graj podcast']},
        'entities': [{'type': 'wrong_type', 'value': 'nazwa'}],
        'new_entity_types': [], 'notes': ''}))
    with pytest.raises(llm.LLMError, match='unknown entity type'):
        llm.apply_annotation(conn, 1, answer, model='test')
    assert not conn.execute('SELECT 1 FROM intents WHERE name = ?', ('play_podcast',)).fetchone()
    assert db.get_recording(conn, 1)['status'] == 'new'
    conn.close()


def test_disabled_auto_annotation_is_noop(tmp_path):
    from open_lexa_admin.app import Worker
    conn = db.open_db(tmp_path / 'labels.sqlite3')
    conn.close()
    worker = Worker(str(tmp_path / 'labels.sqlite3'), DirStore(tmp_path / 'archive'))
    assert worker.annotate_pending(auto=True) == (0, 0)
