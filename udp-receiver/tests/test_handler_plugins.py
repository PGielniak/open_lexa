"""NLU entities and command-plugin isolation, no network/hardware needed."""
import json
from pathlib import Path

from open_lexa.commands import Decision
from open_lexa.handlers import Dispatcher, discover, validate_config
from open_lexa.handlers.base import CommandHandler, HandlerResult
from open_lexa.nlu import NLU
from open_lexa.store import DirStore
from open_lexa.watch import JsonWatcher


def test_nlu_routes_sonos_intents_and_preserves_entity_values():
    from open_lexa_admin import db
    conn = db.open_db(':memory:')
    try:
        lexicon, _ = db.export(conn)
    finally:
        conn.close()
    nlu = NLU()
    nlu.update(lexicon)
    assert nlu.decide('Włącz spotify').intent == 'play_spotify'
    decision = nlu.decide('Puść Chill na spotify')
    assert decision.intent == 'play_spotify'
    assert decision.entities_dict['search_query'] == 'chill'
    decision = nlu.decide('Ustaw głośność na dwadzieścia pięć')
    assert decision.entities_dict['level'] == '25'
    assert nlu.decide('Pauza').intent == 'pause'


class Good(CommandHandler):
    id = 'good'
    intents = frozenset({'play_spotify'})

    def handle(self, request):
        return HandlerResult.handled(f"Gram {request.entities['search_query']}")


class Bad(CommandHandler):
    id = 'bad'
    intents = frozenset({'stop'})

    def handle(self, request):
        raise RuntimeError('device is offline')


def test_dispatcher_plugs_enables_and_isolates_failures():
    dispatcher = Dispatcher([Good(), Bad()])
    decision = Decision('play_spotify', 'play_spotify', intent='play_spotify',
                        entities=(('search_query', 'Chill'),))
    assert dispatcher.dispatch(decision) is None
    dispatcher.apply_config({'good': {'enabled': True}, 'bad': {'enabled': True}})
    assert dispatcher.dispatch(decision)['response'] == 'Gram Chill'
    failure = dispatcher.dispatch(Decision('stop', 'stop', intent='stop'))
    assert failure['status'] == 'failed' and 'offline' in failure['error']
    assert dispatcher.summary()['bad']['failures'] == 1
    dispatcher.apply_config({'bad': {'enabled': False}})
    assert dispatcher.dispatch(Decision('stop', 'stop', intent='stop')) is None
    assert 'sonos' in {handler.id for handler in discover()}
    assert Dispatcher().summary()['sonos']['enabled'] is False


def test_invalid_json_watcher_keeps_previous_configuration(tmp_path: Path):
    store = DirStore(tmp_path)
    path = '_config/handlers.json'
    store.write_atomic(path, json.dumps({'good': {'enabled': True}}).encode())
    watcher = JsonWatcher(store, path, validate=validate_config)
    watcher.poll()
    assert watcher.version == 1
    store.write_atomic(path, b'{broken')
    watcher.poll()
    assert watcher.version == 1 and watcher.data['good']['enabled']
    store.write_atomic(path, b'[]')
    watcher.poll()
    assert watcher.version == 1 and watcher.data['good']['enabled']


def test_sonos_plugin_actions_with_fake_clients():
    from open_lexa.handlers.base import CommandRequest
    from open_lexa.handlers.sonos.handler import SonosHandler

    calls = []

    class Speaker:
        def __init__(self, ip, timeout):
            calls.append(('connect', ip))

        def play_favorite(self, name):
            calls.append(('favorite', name))
            return name

        def play_spotify(self, uri, title):
            calls.append(('spotify', uri))

        def get_volume(self):
            return 10

        def set_volume(self, volume):
            calls.append(('volume', volume))

    class Spotify:
        def __init__(self, client_id, client_secret, market, timeout):
            assert client_id == 'id' and client_secret == 'secret'

        def search_best(self, query, preferred=None):
            return {'uri': 'spotify:track:abc', 'title': query, 'type': 'track'}

    handler = SonosHandler()
    handler.speaker_factory = Speaker
    handler.spotify_factory = Spotify
    handler.configure({'enabled': True, 'speaker_ip': '192.168.0.2',
                       'default_favorite': 'Chill', 'volume_max': 25,
                       'spotify': {'client_id': 'id', 'client_secret': 'secret'}})
    assert handler.handle(CommandRequest('play_music')).status == 'handled'
    assert handler.handle(CommandRequest('play_spotify', {'search_query': 'Pendulum'})).status == 'handled'
    assert handler.handle(CommandRequest('volume_set', {'level': '20'})).status == 'handled'
    assert calls == [('connect', '192.168.0.2'), ('favorite', 'Chill'),
                     ('spotify', 'spotify:track:abc'), ('volume', 20)]
