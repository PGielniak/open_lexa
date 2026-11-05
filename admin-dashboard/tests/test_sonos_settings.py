import base64
import json

from fastapi.testclient import TestClient

from open_lexa.store import DirStore
from open_lexa_admin import db
from open_lexa_admin.app import create_app
from open_lexa.handlers.sonos.handler import SonosHandler


AUTH = {'Authorization': 'Basic ' + base64.b64encode(b'admin:correct-horse-battery').decode()}


def test_sonos_config_and_authenticated_controls(tmp_path, monkeypatch):
    called = []

    class Speaker:
        def __init__(self, ip, timeout):
            called.append(('connect', ip))

        def get_volume(self):
            return 20

        def set_volume(self, value):
            called.append(('volume', value))

        def play_favorite(self, name):
            called.append(('favorite', name))
            return name or 'Default'

        def info(self):
            return {'name': 'Living room', 'state': 'PLAYING', 'volume': 20}

    monkeypatch.setattr(SonosHandler, 'speaker_factory', Speaker)
    archive = tmp_path / 'archive'
    app = create_app(db_path=str(tmp_path / 'labels.sqlite3'), store=DirStore(archive),
                     username='admin', password='correct-horse-battery', background=False)
    with TestClient(app) as client:
        assert client.post('/settings/sonos-control', data={'action': 'play'}).status_code == 401
        assert client.post('/settings/handlers/unknown', headers=AUTH).status_code == 404
        response = client.post('/settings/sonos', headers=AUTH, follow_redirects=False, data={
            'enabled': '1', 'speaker_ip': '192.168.1.22', 'default_favorite': 'Chill',
            'volume_step': '5', 'volume_max': '40', 'spotify_client_id': 'client',
            'spotify_client_secret': 'top-secret', 'spotify_market': 'PL'})
        assert response.status_code == 303
        assert 'top-secret' not in client.get('/settings', headers=AUTH).text
        result = client.post('/settings/sonos-control', headers=AUTH,
                             data={'action': 'favorite', 'favorite': 'Chill'}).json()
        assert result['ok'] and result['status'] == 'handled'
        result = client.post('/settings/sonos-control', headers=AUTH,
                             data={'action': 'volume_up'}).json()
        assert result['ok'] and ('volume', 25) in called
        assert client.post('/settings/sonos-control', headers=AUTH,
                           data={'action': 'unknown'}).status_code == 400
        app.state.worker.export()
        published = json.loads((archive / '_config' / 'handlers.json').read_text())
        assert published['sonos']['speaker_ip'] == '192.168.1.22'
        assert published['sonos']['spotify']['client_secret'] == 'top-secret'
        client.post('/settings/sonos', headers=AUTH, data={
            'enabled': '1', 'speaker_ip': '192.168.1.22',
            'default_favorite': '', 'spotify_client_id': ''})
        conn = db.connect(app.state.worker.db_path)
        changed = db.get_setting(conn, 'handlers')['sonos']
        assert changed['default_favorite'] == ''
        assert changed['spotify']['client_id'] == ''
        assert changed['spotify']['client_secret'] == 'top-secret'
        conn.close()
        # Unplugging in the generic plugin list keeps its configuration.
        client.post('/settings/handlers/sonos', headers=AUTH, data={})
        assert not client.post('/settings/sonos-control', headers=AUTH,
                               data={'action': 'play'}).json()['ok']
        conn = db.connect(app.state.worker.db_path)
        assert db.get_setting(conn, 'handlers')['sonos']['speaker_ip'] == '192.168.1.22'
        conn.close()
