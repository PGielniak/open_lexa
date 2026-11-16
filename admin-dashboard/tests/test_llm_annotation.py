import base64
import json
from fastapi.testclient import TestClient
from open_lexa.store import DirStore
from open_lexa_admin.app import create_app

AUTH = {'Authorization': 'Basic ' + base64.b64encode(b'admin:correct-horse-battery').decode()}


def test_llm_settings_published_and_old_endpoints_removed(tmp_path):
    archive = tmp_path / 'archive'
    app = create_app(db_path=str(tmp_path / 'labels.sqlite3'), store=DirStore(archive),
                     username='admin', password='correct-horse-battery', background=False)
    with TestClient(app) as client:
        assert client.post('/annotate', headers=AUTH).status_code == 404
        assert client.get('/dictionary', headers=AUTH).status_code == 404
        response = client.post('/settings/llm', headers=AUTH, follow_redirects=False, data={
            'url': 'https://llm.example/v1', 'model': 'my-model', 'api_key': 'secret',
            'receiver_mode': 'always'})
        assert response.status_code == 303
        assert 'value="secret"' not in client.get('/settings', headers=AUTH).text
        app.state.worker.export()
        payload = json.loads((archive / '_config' / 'llm.json').read_text())
        assert payload['mode'] == 'always' and payload['api_key'] == 'secret'
        client.post('/settings/llm', headers=AUTH, data={'receiver_mode': 'off'})
        app.state.worker.export()
        assert 'api_key' not in json.loads((archive / '_config' / 'llm.json').read_text())
