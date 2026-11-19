"""NLU entities and command-plugin isolation, no network/hardware needed."""
import json
from pathlib import Path

from types import SimpleNamespace
from open_lexa.handlers import Dispatcher, discover, validate_config
from open_lexa.handlers.base import CommandHandler, HandlerResult
from open_lexa.store import DirStore
from open_lexa.watch import JsonWatcher


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


def Decision(command, reply, *, intent, entities=()):
    return SimpleNamespace(command=command, reply=reply, intent=intent,
                           entities=entities, entities_dict=dict(entities))


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


def test_sonos_error_includes_traceback_and_result_reason(caplog):
    import logging
    from open_lexa.handlers.sonos.handler import SonosHandler
    from open_lexa.handlers.sonos.sonos import SonosError

    class Speaker:
        def __init__(self, ip, timeout):
            pass

        def play_favorite(self, name):
            try:
                raise ValueError('UPnP Error 800')
            except ValueError as exc:
                raise SonosError('playing favorite failed') from exc

    handler = SonosHandler()
    handler.speaker_factory = Speaker
    dispatcher = Dispatcher([handler])
    dispatcher.apply_config({'sonos': {'enabled': True, 'speaker_ip': '192.168.1.2',
                                       'default_favorite': 'test'}})
    caplog.set_level(logging.INFO)
    result = dispatcher.dispatch(Decision('play_spotify', '', intent='play_spotify'), session='abc123')
    assert result['status'] == 'failed' and result['error'] == 'playing favorite failed'
    record = next(r for r in caplog.records if 'sonos_tool_error' in r.message)
    assert record.exc_info is not None
    assert 'ValueError: UPnP Error 800' in caplog.text
    assert 'SonosError: playing favorite failed' in caplog.text
    assert 'mcp_result session=abc123 server=sonos tool=play_spotify status=failed' in caplog.text
    assert 'error="playing favorite failed"' in caplog.text


def test_skipped_result_logs_reason_without_traceback(caplog):
    import logging

    class Skipped(CommandHandler):
        id = 'skipped'
        intents = frozenset({'play'})

        def handle(self, request):
            return HandlerResult.skipped('no default favorite configured')

    dispatcher = Dispatcher([Skipped()])
    dispatcher.apply_config({'skipped': {'enabled': True}})
    caplog.set_level(logging.INFO)
    dispatcher.dispatch(Decision('play', '', intent='play'), session='abc')
    record = next(r for r in caplog.records if 'mcp_result' in r.message)
    assert record.exc_info is None
    assert 'status=skipped' in record.message and 'no default favorite configured' in record.message


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
    playlist_limits = []

    def search(query, types=('track', 'album', 'playlist', 'artist'), limit=5):
        calls.append((query, types))
        if types == ('playlist',):
            playlist_limits.append(limit)
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
    assert playlist_limits and max(playlist_limits) <= 10
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


def _radio_handler(favorites=(), channel='', stations=None):
    from open_lexa.handlers.sonos.handler import SonosHandler
    from open_lexa.handlers.sonos.sonos import SonosSpeaker
    from open_lexa.handlers.sonos.tunein import TuneInClient

    calls = []

    class Speaker(SonosSpeaker):
        def favorites(self):
            return [dict(item) for item in favorites]

        def play_favorite_item(self, favorite):
            calls.append(('favorite', favorite['title']))

        def play_tunein(self, station_id, title):
            calls.append(('tunein', station_id, title))

        def current_channel(self):
            return channel

    class TuneIn(TuneInClient):
        def search(self, query):
            calls.append(('search', query))
            return [item for item in (stations or {}).get(query, [])]

    handler = SonosHandler()
    handler.speaker_factory = Speaker
    handler.tunein_factory = TuneIn
    handler.configure({'enabled': True, 'speaker_ip': '192.168.0.2',
                       'radio_stations': 'RMF FM, Radio 357\nrmf fm'})
    return handler, calls


def test_radio_tools_are_listed():
    from open_lexa.handlers.sonos.handler import SonosHandler

    names = {tool.name for tool in SonosHandler.tools}
    assert {'play_radio', 'play_default_radio', 'play_random_radio'} <= names


def test_default_and_named_radio_use_tunein():
    from open_lexa.handlers.base import CommandRequest

    handler, calls = _radio_handler(stations={
        'Radio Zet': [{'id': 's125091', 'title': 'ZET Hits!'},
                      {'id': 's15990', 'title': 'Radio Zet'}],
        'RMF FM': [{'id': 's1217', 'title': 'RMF FM'}],
    })
    assert handler.radio_stations == ['RMF FM', 'Radio 357']
    result = handler.handle(CommandRequest('play_default_radio'))
    assert result.status == 'handled' and result.response == 'Włączam Radio Zet'
    assert handler.handle(CommandRequest('play_radio', {'station': 'RMF FM'})).status == 'handled'
    assert handler.handle(CommandRequest('play_radio', {'station': 'Radio'})).status == 'handled'
    assert [call for call in calls if call[0] == 'tunein'] == [
        ('tunein', 's15990', 'Radio Zet'), ('tunein', 's1217', 'RMF FM'),
        ('tunein', 's15990', 'Radio Zet')]


def test_named_radio_prefers_favorites_and_rejects_unrelated_matches():
    from open_lexa.handlers.base import CommandRequest

    favorites = [{'title': 'Chill', 'uri': 'x-rincon-cpcontainer:abc', 'meta': ''},
                 {'title': 'Trójka Polskie Radio', 'uri': 'x-rincon-mp3radio:http://x', 'meta': ''}]
    handler, calls = _radio_handler(favorites=favorites, stations={
        'Jazz': [{'id': 's1', 'title': 'Dark Edge Radio'}]})
    assert handler.handle(CommandRequest('play_radio', {'station': 'Trójka'})).status == 'handled'
    assert calls == [('favorite', 'Trójka Polskie Radio')]
    # A playlist favorite is not a fuzzy radio match; TuneIn's loose hit is refused.
    result = handler.handle(CommandRequest('play_radio', {'station': 'Jazz'}))
    assert result.status == 'failed' and 'no TuneIn station matches' in result.error


def test_random_radio_skips_current_and_unknown_stations(monkeypatch):
    from open_lexa.handlers.base import CommandRequest

    favorites = [{'title': 'Chillizet', 'uri': 'x-sonosapi-stream:s87828?sid=254', 'meta': ''},
                 {'title': 'Chill', 'uri': 'x-rincon-cpcontainer:abc', 'meta': ''}]
    handler, calls = _radio_handler(favorites=favorites, channel='Radio 357', stations={
        'RMF FM': []})
    monkeypatch.setattr('open_lexa.handlers.sonos.handler.random.shuffle',
                        lambda items: items.sort(key=lambda name: name != 'RMF FM'))
    result = handler.handle(CommandRequest('play_random_radio'))
    assert result.status == 'handled' and result.detail['station'] == 'Chillizet'
    assert ('search', 'Radio 357') not in calls
    assert calls[-1] == ('favorite', 'Chillizet')


def test_tunein_matching_and_play_uri(monkeypatch):
    from open_lexa.handlers.sonos.sonos import SonosSpeaker
    from open_lexa.handlers.sonos.tunein import TuneInClient, TuneInError

    client = TuneInClient()
    monkeypatch.setattr(client, 'search', lambda query: [
        {'id': 's1', 'title': 'Dark Edge Radio'}, {'id': 's9608', 'title': 'Anty Radio'},
        {'id': 's174766', 'title': 'Radio Nowy Świat'}])
    assert client.find_station('Antyradio')['id'] == 's9608'
    assert client.find_station('radio nowy swiat')['id'] == 's174766'
    try:
        client.find_station('Trójka')
    except TuneInError as exc:
        assert 'Dark Edge Radio' in str(exc)
    else:
        raise AssertionError('loose match accepted')

    played = []

    class Coordinator:
        def __init__(self):
            self.avTransport = SimpleNamespace(AddURIToQueue=lambda args: played.append(('enqueue', dict(args))))

        def play_uri(self, uri, meta='', **kwargs):
            played.append((uri, meta, kwargs))

        def clear_queue(self):
            played.append(('clear',))

        def play_from_queue(self, index):
            played.append(('play_from_queue', index))

    speaker = SonosSpeaker('192.168.0.2')
    monkeypatch.setattr(SonosSpeaker, 'coordinator', property(lambda self: Coordinator()))
    speaker.play_tunein('s15990', 'Radio <Zet>')
    speaker.play_favorite_item({'uri': 'x-sonosapi-stream:s1', 'meta': '<DIDL/>'})
    speaker.play_favorite_item({'uri': 'x-rincon-cpcontainer:abc', 'meta': '<DIDL/>'})
    assert played[0][0] == 'x-sonosapi-stream:s15990?sid=254&flags=8224&sn=0'
    assert 'F00092020s15990' in played[0][1] and 'Radio &lt;Zet&gt;' in played[0][1]
    assert played[0][2] == {'timeout': 20.0}
    assert played[1] == ('x-sonosapi-stream:s1', '<DIDL/>', {'timeout': 20.0})
    # A playlist favorite cannot be set as the transport URI (UPnP 714): queue it.
    assert played[2] == ('clear',)
    assert played[3][0] == 'enqueue' and played[3][1]['EnqueuedURI'] == 'x-rincon-cpcontainer:abc'
    assert played[3][1]['EnqueuedURIMetaData'] == '<DIDL/>'
    assert played[4] == ('play_from_queue', 0)
    assert len(played) == 5
