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


def test_nlu_recognises_sonos_playback_queue_and_volume_commands():
    from open_lexa_admin import db
    conn = db.open_db(':memory:')
    try:
        lexicon, _ = db.export(conn)
    finally:
        conn.close()
    nlu = NLU()
    nlu.update(lexicon)
    cases = {
        'Puść moją playlistę': ('play_music', {}),
        'Włącz muzykę ze Spotify': ('play_spotify', {}),
        'Puść muzykę artysty Pendulum': ('play_spotify', {'search_query': 'pendulum'}),
        'Puść muzykę gatunku synthwave': ('play_spotify', {'search_query': 'synthwave'}),
        'Puść utwór Bohemian Rhapsody na Spotify':
            ('play_spotify', {'search_query': 'bohemian rhapsody'}),
        'Dodaj utwór Bohemian Rhapsody do kolejki':
            ('queue_spotify', {'search_query': 'bohemian rhapsody'}),
        'Dodaj do kolejki Drum and Bass':
            ('queue_spotify', {'search_query': 'drum and bass'}),
        'Dodaj artystę Pendulum do kolejki':
            ('queue_spotify', {'search_query': 'pendulum'}),
        'Dodaj playlistę Drum and Bass do kolejki':
            ('queue_spotify', {'search_query': 'drum and bass'}),
        'Podgłośnij o pięć': ('volume_up', {'level': '5'}),
        'Przycisz o 10': ('volume_down', {'level': '10'}),
        'Ustaw głośność na 20 procent': ('volume_set', {'level': '20'}),
        'Następna piosenka': ('next_track', {}),
        'Poprzednia piosenka': ('previous_track', {}),
    }
    for text, (intent, entities) in cases.items():
        decision = nlu.decide(text)
        assert (decision.intent, decision.entities_dict) == (intent, entities), text


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


def test_bare_genre_plays_playlist_not_track():
    from open_lexa.handlers.base import CommandRequest
    from open_lexa.handlers.sonos.handler import SonosHandler

    calls = []

    class Speaker:
        def __init__(self, ip, timeout):
            pass

        def play_spotify(self, uri, title):
            calls.append(('play', uri))

    class Spotify:
        def __init__(self, client_id, client_secret, market, timeout):
            pass

        def search_genre_playlist(self, genre):
            calls.append(('genre', genre))
            return {'uri': 'spotify:playlist:dnb', 'title': 'DnB mix', 'type': 'playlist'}

        def search_best(self, query, preferred=None):
            calls.append(('search', query, preferred))
            return {'uri': 'spotify:track:song', 'title': query, 'type': 'track'}

        def search_track(self, query):
            calls.append(('track', query))
            return {'uri': 'spotify:track:song', 'title': query, 'type': 'track'}

    handler = SonosHandler()
    handler.speaker_factory = Speaker
    handler.spotify_factory = Spotify
    handler.configure({'enabled': True, 'speaker_ip': '192.168.0.2',
                       'spotify': {'client_id': 'id', 'client_secret': 'secret'}})

    for genre in ('Drum and Bass', 'Dram and Bass', 'DnB'):
        result = handler.handle(CommandRequest('play_spotify', {'search_query': genre},
                                               text=f'Puść {genre} na Spotify'))
        assert result.status == 'handled'
        assert result.detail['type'] == 'playlist'
    assert calls == [('genre', 'drum and bass'), ('play', 'spotify:playlist:dnb')] * 3

    handler.handle(CommandRequest('play_spotify', {'search_query': 'synthwave'},
                                  text='Puść muzykę gatunku synthwave'))
    assert calls[-2:] == [('genre', 'synthwave'), ('play', 'spotify:playlist:dnb')]

    handler.handle(CommandRequest('play_spotify', {'search_query': 'Pendulum'},
                                  text='Puść Pendulum na Spotify'))
    handler.handle(CommandRequest('play_spotify', {'search_query': 'drum and bass'},
                                  text='Puść utwór drum and bass na Spotify'))
    assert calls[-4:] == [('search', 'Pendulum', None), ('play', 'spotify:track:song'),
                          ('track', 'drum and bass'), ('play', 'spotify:track:song')]


def test_genre_playlist_search_picks_only_nonempty_playlists(monkeypatch):
    from open_lexa.handlers.sonos.spotify import SpotifyClient, SpotifyError
    import pytest

    client = SpotifyClient('id', 'secret')
    searches = []

    def search(query, types, limit):
        searches.append((query, types, limit))
        return {'playlists': {'items': [None, {'uri': 'spotify:playlist:empty',
                                               'tracks': {'total': 0}},
                                      {'uri': 'spotify:playlist:a', 'name': 'DnB A'},
                                      {'uri': 'spotify:playlist:b', 'name': 'DnB B'}]}}

    client.search = search
    monkeypatch.setattr('open_lexa.handlers.sonos.spotify.random.choice', lambda items: items[-1])
    assert client.search_genre_playlist('drum and bass') == {
        'uri': 'spotify:playlist:b', 'title': 'DnB B', 'type': 'playlist'}
    assert searches == [('drum and bass', ('playlist',), 10)]

    client.search = lambda *args, **kwargs: {'playlists': {'items': []}}
    with pytest.raises(SpotifyError, match='no playlists found'):
        client.search_genre_playlist('drum and bass')


def test_spotify_artist_playlist_and_direct_track_search():
    from open_lexa.handlers.sonos.spotify import SpotifyClient, SpotifyError
    import pytest

    client = SpotifyClient('id', 'secret')
    calls = []

    def search(query, types=('track', 'album', 'playlist', 'artist'), limit=5):
        calls.append((query, types))
        if types == ('playlist',):
            return {'playlists': {'items': [
                {'uri': 'spotify:playlist:fan', 'name': 'Pendulum hits',
                 'owner': {'id': 'fan'}},
                {'uri': 'spotify:playlist:official', 'name': 'This Is Pendulum',
                 'owner': {'id': 'spotify'}}]}}
        if types == ('track',):
            return {'tracks': {'items': [{'uri': 'spotify:track:song',
                                         'name': 'Bohemian Rhapsody'}]}}
        return {'artists': {'items': [{'name': 'Pendulum'}]},
                'tracks': {'items': [{'uri': 'spotify:track:wrong'}]}}

    client.search = search
    assert client.search_best('Pendulum')['uri'] == 'spotify:playlist:official'
    assert client.search_artist_playlist('Pendulum')['type'] == 'playlist'
    assert client.search_track('Bohemian Rhapsody')['uri'] == 'spotify:track:song'
    assert ('Bohemian Rhapsody', ('track',)) in calls
    client.search = lambda query, **kwargs: {'playlists': {'items': [
        {'uri': 'spotify:playlist:fan', 'name': 'Pendulum hits'}]}}
    assert client.search_artist_playlist('Pendulum')['uri'] == 'spotify:playlist:fan'
    client.search = lambda *args, **kwargs: {'playlists': {'items': []}}
    with pytest.raises(SpotifyError, match='no playlist found'):
        client.search_artist_playlist('Pendulum')

    client.search = lambda *args, **kwargs: {
        'playlists': {'items': []},
        'tracks': {'items': [{'uri': 'spotify:track:unrelated'}]}}
    with pytest.raises(SpotifyError, match='nothing found'):
        client.search_best('missing playlist', preferred='playlist')


def test_sonos_queue_default_and_volume_control():
    from open_lexa.handlers.base import CommandRequest
    from open_lexa.handlers.sonos.handler import SonosHandler

    calls = []

    class Speaker:
        def __init__(self, ip, timeout):
            pass

        def play_favorite(self, name):
            calls.append(('favorite', name))
            return name

        def enqueue_spotify(self, uri, title):
            calls.append(('enqueue', uri))
            return 4

        def play_spotify(self, uri, title):
            calls.append(('play', uri))

        def get_volume(self):
            return 20

        def set_volume(self, value):
            calls.append(('volume', value))

        def next(self):
            calls.append(('next',))

        def previous(self):
            calls.append(('previous',))

    class Spotify:
        def __init__(self, *args, **kwargs):
            pass

        def search_artist_playlist(self, artist):
            calls.append(('artist', artist))
            return {'uri': 'spotify:playlist:pendulum', 'title': 'This Is Pendulum',
                    'type': 'playlist'}

        def search_genre_playlist(self, genre):
            calls.append(('genre', genre))
            return {'uri': 'spotify:playlist:dnb', 'title': 'DnB', 'type': 'playlist'}

    handler = SonosHandler()
    handler.speaker_factory = Speaker
    handler.spotify_factory = Spotify
    handler.configure({'enabled': True, 'speaker_ip': '192.168.0.2',
                       'default_favorite': 'Moja playlista', 'volume_step': 5,
                       'volume_max': 35, 'spotify': {'client_id': 'id', 'client_secret': 'secret'}})
    assert handler.handle(CommandRequest('play_music')).status == 'handled'
    assert handler.handle(CommandRequest('play_spotify')).status == 'handled'
    queued = handler.handle(CommandRequest('queue_spotify', {'search_query': 'Pendulum'},
                                           text='Dodaj artystę Pendulum do kolejki'))
    assert queued.detail['position'] == 4
    assert handler.handle(CommandRequest('queue_spotify', {'search_query': 'drum and bass'},
                                         text='Dodaj do kolejki drum and bass')).status == 'handled'
    assert handler.handle(CommandRequest('queue_spotify')).status == 'skipped'
    handler.handle(CommandRequest('next_track'))
    handler.handle(CommandRequest('previous_track'))
    assert handler.handle(CommandRequest('volume_up', {'level': '12'})).detail['volume_to'] == 32
    assert handler.handle(CommandRequest('volume_down', {'level': '8'})).detail['volume_to'] == 12
    assert handler.handle(CommandRequest('volume_set', {'level': '80'})).detail['volume_to'] == 35
    assert handler.handle(CommandRequest('volume_up', {'level': '0'})).status == 'skipped'
    assert calls == [('favorite', 'Moja playlista')] * 2 + [
        ('artist', 'Pendulum'), ('enqueue', 'spotify:playlist:pendulum'),
        ('genre', 'drum and bass'), ('enqueue', 'spotify:playlist:dnb'),
        ('next',), ('previous',), ('volume', 32), ('volume', 12), ('volume', 35)]

    handler.configure({'enabled': True, 'speaker_ip': '192.168.0.2'})
    assert handler.handle(CommandRequest('play_music')).status == 'skipped'


def test_sonos_queue_does_not_interrupt_and_playback_uses_coordinator(monkeypatch):
    from open_lexa.handlers.sonos.sonos import SonosSpeaker

    calls = []

    class Coordinator:
        def play_from_queue(self, index):
            calls.append(('play_from_queue', index))

    coordinator = Coordinator()

    class Device:
        group = type('Group', (), {'coordinator': coordinator})()

    speaker = SonosSpeaker('192.168.0.2')
    speaker._device = Device()
    monkeypatch.setattr(SonosSpeaker, '_add_share_link', staticmethod(
        lambda target, uri, title: calls.append(('enqueue', target, uri, title)) or 4))
    assert speaker.enqueue_spotify('spotify:playlist:dnb', 'DnB') == 4
    assert calls == [('enqueue', coordinator, 'spotify:playlist:dnb', 'DnB')]
    speaker.play_spotify('spotify:track:song', 'Song')
    assert calls[-2:] == [('enqueue', coordinator, 'spotify:track:song', 'Song'),
                          ('play_from_queue', 3)]
