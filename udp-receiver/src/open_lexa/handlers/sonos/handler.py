"""Sonos playback handler: Spotify search, TuneIn radio, favorites, transport, volume.

Configuration section in ``_config/handlers.json`` (edited in the admin
portal under Ustawienia → Sonos / Spotify)::

    {"sonos": {
        "enabled": true,
        "speaker_ip": "192.168.1.42",
        "default_favorite": "Chill",        # Sonos favorite for bare "włącz muzykę"
        "default_radio": "Radio Zet",       # station for bare "włącz radio"
        "radio_stations": "Radio Zet, RMF FM, Radio 357",  # "włącz losowe radio" pool
        "volume_step": 5,
        "volume_max": 60,
        "timeout": 5,
        "spotify": {"client_id": "...", "client_secret": "...", "market": "PL"}
    }}
"""
from __future__ import annotations

import logging
import random
import re

from ..base import CommandHandler, HandlerConfigError, HandlerResult, MCPTool
from .sonos import SonosError, SonosSpeaker
from .spotify import SpotifyClient, SpotifyError
from .tunein import TuneInClient, TuneInError, compact, is_generic_radio

LOG = logging.getLogger('open_lexa.handlers.sonos')

INTENTS = frozenset({
    'play_music', 'play_spotify', 'queue_spotify',
    'play_radio', 'play_default_radio', 'play_random_radio',
    'stop', 'pause', 'resume', 'next_track', 'previous_track',
    'volume_up', 'volume_down', 'volume_set',
})

# Only a genre name (not an artist/title) should start a playlist, not the
# first unrelated track returned by Spotify's multi-type search.
GENRES = {
    'drum and bass': 'drum and bass', 'dram and bass': 'drum and bass',
    'drum & bass': 'drum and bass', 'drum bass': 'drum and bass',
    'dnb': 'drum and bass',
    'ambient': 'ambient', 'blues': 'blues', 'country': 'country',
    'disco': 'disco', 'dubstep': 'dubstep', 'funk': 'funk',
    'hip hop': 'hip hop', 'hip-hop': 'hip hop', 'hip hopu': 'hip hop',
    'house': 'house', 'jazz': 'jazz', 'jazzu': 'jazz',
    'klasyczna': 'classical', 'classical': 'classical',
    'metal': 'metal', 'metalu': 'metal', 'pop': 'pop', 'popu': 'pop',
    'punk': 'punk', 'rap': 'rap', 'rapu': 'rap',
    'reggae': 'reggae', 'rock': 'rock', 'rocka': 'rock', 'soul': 'soul',
    'techno': 'techno', 'trance': 'trance',
}


DEFAULT_RADIO = 'Radio Zet'
# Stations verified on TuneIn; extended by radio favorites at runtime.
DEFAULT_RADIO_STATIONS = ('Radio Zet', 'RMF FM', 'Radio 357', 'Radio Nowy Świat',
                          'Antyradio', 'TOK FM', 'Chillizet', 'ESKA')
RANDOM_RADIO_ATTEMPTS = 3


def _station_list(value) -> list[str]:
    """Accept a list or a comma/newline separated string from the admin form."""
    items = value if isinstance(value, (list, tuple)) else re.split(r'[,\n;]', str(value or ''))
    stations, seen = [], set()
    for item in items:
        name = str(item).strip()
        if name and compact(name) not in seen:
            seen.add(compact(name))
            stations.append(name)
    return stations


def _clamp(value, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


class SonosHandler(CommandHandler):
    """Plays Spotify and Sonos favorites through SoCo on one speaker."""

    id = 'sonos'
    version = '1'
    intents = INTENTS
    priority = 50
    tools = tuple(
        MCPTool(name, description, {
            'type': 'object', 'properties': properties,
            'required': required, 'additionalProperties': False,
        }) for name, description, properties, required in (
            ('play_music', 'Włącz domyślną playlistę Sonos, np. „włącz muzykę”.', {}, []),
            ('play_spotify', 'Włącz Spotify lub wyszukaj wskazaną muzykę. '
             'Nie używaj dla stacji radiowych.',
             {'search_query': {'type': 'string', 'description': 'Tytuł, artysta lub gatunek; pomiń dla domyślnej playlisty.'}}, []),
            ('queue_spotify', 'Dodaj muzykę ze Spotify do kolejki.',
             {'search_query': {'type': 'string', 'minLength': 1}}, ['search_query']),
            ('play_radio', 'Włącz wskazaną stację radiową (TuneIn lub ulubione Sonos), '
             'np. „włącz RMF FM”, „puść Radio 357”.',
             {'station': {'type': 'string', 'minLength': 1,
                          'description': 'Nazwa stacji w mianowniku, np. „RMF FM”, „Radio Zet”.'}},
             ['station']),
            ('play_default_radio', 'Włącz domyślne radio, gdy nie podano stacji, '
             'np. „włącz radio”.', {}, []),
            ('play_random_radio', 'Włącz losową stację radiową, np. „włącz losowe radio”.',
             {}, []),
            ('stop', 'Zatrzymaj muzykę.', {}, []),
            ('pause', 'Wstrzymaj muzykę.', {}, []),
            ('resume', 'Wznów odtwarzanie.', {}, []),
            ('next_track', 'Odtwórz następny utwór.', {}, []),
            ('previous_track', 'Odtwórz poprzedni utwór.', {}, []),
            ('volume_up', 'Zwiększ głośność o opcjonalny krok.',
             {'level': {'type': 'integer', 'minimum': 1, 'maximum': 100}}, []),
            ('volume_down', 'Zmniejsz głośność o opcjonalny krok.',
             {'level': {'type': 'integer', 'minimum': 1, 'maximum': 100}}, []),
            ('volume_set', 'Ustaw głośność na podany poziom.',
             {'level': {'type': 'integer', 'minimum': 0, 'maximum': 100}}, ['level']),
        )
    )

    # Overridable in tests; production builds the real clients lazily.
    speaker_factory = SonosSpeaker
    spotify_factory = SpotifyClient
    tunein_factory = TuneInClient
    _speaker = None
    _spotify = None
    _tunein = None

    def __init__(self) -> None:
        super().__init__()
        self.configure({})

    def configure(self, config) -> None:
        super().configure(config)
        self.speaker_ip = str(self.config.get('speaker_ip') or '').strip()
        if self.enabled and not self.speaker_ip:
            raise HandlerConfigError('speaker_ip is required when the handler is enabled')
        self.default_favorite = str(self.config.get('default_favorite') or '').strip()
        self.default_radio = str(self.config.get('default_radio') or '').strip() or DEFAULT_RADIO
        self.radio_stations = (_station_list(self.config.get('radio_stations'))
                               or list(DEFAULT_RADIO_STATIONS))
        self.volume_step = _clamp(self.config.get('volume_step'), 5, 1, 25)
        self.volume_max = _clamp(self.config.get('volume_max'), 60, 0, 100)
        self.timeout = _clamp(self.config.get('timeout'), 5, 1, 60)
        spotify = self.config.get('spotify') if isinstance(self.config.get('spotify'), dict) else {}
        self.spotify_id = str(spotify.get('client_id') or '').strip()
        self.spotify_secret = str(spotify.get('client_secret') or '').strip()
        self.spotify_market = str(spotify.get('market') or 'PL').strip()
        # Rebuild lazily-created clients so new settings take effect.
        self._speaker = None
        self._spotify = None
        self._tunein = None

    @property
    def speaker(self) -> SonosSpeaker:
        if self._speaker is None:
            self._speaker = self.speaker_factory(self.speaker_ip, timeout=self.timeout)
        return self._speaker

    @property
    def spotify(self) -> SpotifyClient:
        if self._spotify is None:
            self._spotify = self.spotify_factory(self.spotify_id, self.spotify_secret,
                                                 market=self.spotify_market,
                                                 timeout=float(self.timeout) * 2)
        return self._spotify

    @property
    def tunein(self) -> TuneInClient:
        if self._tunein is None:
            self._tunein = self.tunein_factory(timeout=float(self.timeout) * 2)
        return self._tunein

    # --- introspection ----------------------------------------------------------------

    def status(self) -> dict:
        return {'enabled': self.enabled, 'speaker_ip': self.speaker_ip or None,
                'default_favorite': self.default_favorite or None,
                'default_radio': self.default_radio,
                'radio_stations': self.radio_stations,
                'spotify_configured': bool(self.spotify_id and self.spotify_secret)}

    def test(self) -> dict:
        """Reach the speaker and report what it plays (admin "Test" button)."""
        return self.speaker.info()

    # --- execution --------------------------------------------------------------------

    def handle(self, request) -> HandlerResult:
        if not self.enabled:
            return HandlerResult.skipped('sonos handler is disabled')
        try:
            return self._handle(request)
        except (SonosError, SpotifyError, TuneInError) as exc:
            LOG.exception('sonos_tool_error session=%s tool=%s',
                          request.session or '-', request.tool)
            return HandlerResult.failed(str(exc))

    def _handle(self, request) -> HandlerResult:
        intent = request.intent
        if intent == 'play_spotify':
            return self._play_spotify(request)
        if intent == 'queue_spotify':
            return self._queue_spotify(request)
        if intent == 'play_music':
            source = (request.entities.get('source') or '').casefold()
            if source == 'spotify':
                if request.entities.get('search_query'):
                    return self._play_spotify(request)
                return self._play_default(source_label='Spotify')
            return self._play_default()
        if intent == 'play_radio':
            station = str(request.entities.get('station') or '').strip()
            if is_generic_radio(station):  # "włącz radio" sent as station="radio"
                return self._play_station(self.default_radio)
            return self._play_station(station)
        if intent == 'play_default_radio':
            return self._play_station(self.default_radio)
        if intent == 'play_random_radio':
            return self._play_random_radio()
        if intent == 'stop':
            self.speaker.stop()
            return HandlerResult.handled('Zatrzymane')
        if intent == 'pause':
            self.speaker.pause()
            return HandlerResult.handled('Pauza')
        if intent == 'resume':
            self.speaker.play()
            return HandlerResult.handled('Wznawiam')
        if intent == 'next_track':
            self.speaker.next()
            return HandlerResult.handled('Następny utwór')
        if intent == 'previous_track':
            self.speaker.previous()
            return HandlerResult.handled('Poprzedni utwór')
        if intent in ('volume_up', 'volume_down', 'volume_set'):
            return self._volume(intent, request)
        return HandlerResult.skipped(f'intent {intent!r} is not executed by this handler')

    def _play_spotify(self, request) -> HandlerResult:
        query = (request.entities.get('search_query') or '').strip()
        if not query:
            return self._play_default(source_label='Spotify')
        result = self._resolve_spotify(query, request.text)
        self.speaker.play_spotify(result['uri'], result['title'])
        return HandlerResult.handled(f"Gram {result['title']} ze Spotify", **result)

    def _queue_spotify(self, request) -> HandlerResult:
        query = (request.entities.get('search_query') or '').strip()
        if not query:
            return HandlerResult.skipped('no search query to add to the queue')
        result = self._resolve_spotify(query, request.text)
        position = self.speaker.enqueue_spotify(result['uri'], result['title'])
        return HandlerResult.handled(f"Dodano {result['title']} do kolejki", position=position,
                                     **result)

    def _resolve_spotify(self, query: str, text: str | None) -> dict:
        """Route an utterance to a track, genre playlist or artist playlist."""
        utterance = (text or '').casefold()
        preferred = ('playlist' if 'playlist' in utterance else
                      'album' if 'album' in utterance else
                      'artist' if any(word in utterance for word in ('artyst', 'wykonawc', 'zespoł', 'zespol'))
                      else None)
        if preferred == 'artist':
            return self.spotify.search_artist_playlist(query)
        if any(word in utterance for word in ('utwór', 'utwor', 'piosenk')):
            return self.spotify.search_track(query)
        genre = (GENRES.get(' '.join(query.casefold().split()))
                 if preferred in (None, 'playlist') else None)
        if 'gatunk' in utterance:
            genre = genre or query
        if genre and preferred in (None, 'playlist'):
            return self.spotify.search_genre_playlist(genre)
        return self.spotify.search_best(query, preferred=preferred)

    def _play_station(self, name: str) -> HandlerResult:
        """Sonos favorite (exact, then radio substring) first, then TuneIn search."""
        favorite = self.speaker.find_favorite(name, radio_only_fuzzy=True)
        if favorite is not None:
            self.speaker.play_favorite_item(favorite)
            return HandlerResult.handled(f"Włączam {favorite['title']}",
                                         station=favorite['title'], source='favorite')
        station = self.tunein.find_station(name)
        self.speaker.play_tunein(station['id'], station['title'])
        return HandlerResult.handled(f"Włączam {station['title']}", station=station['title'],
                                     station_id=station['id'], source='tunein')

    def _play_random_radio(self) -> HandlerResult:
        """Random pick from configured stations plus radio favorites, not the current one."""
        pool = list(self.radio_stations)
        try:
            pool += [favorite['title'] for favorite in self.speaker.radio_favorites()]
        except SonosError:
            LOG.warning('sonos_radio_favorites_unavailable', exc_info=True)
        pool = _station_list(pool)
        playing = compact(self.speaker.current_channel())
        if playing and len(pool) > 1:
            pool = [name for name in pool if compact(name) != playing] or pool
        random.shuffle(pool)
        errors = []
        for name in pool[:RANDOM_RADIO_ATTEMPTS]:
            try:
                return self._play_station(name)
            except TuneInError as exc:
                LOG.warning('sonos_random_radio_skip station=%r error=%s', name, exc)
                errors.append(str(exc))
        return HandlerResult.failed('no random station could be played: ' + '; '.join(errors))

    def _play_default(self, source_label: str = '') -> HandlerResult:
        """Bare music command: play the configured personal Sonos favorite."""
        if self.default_favorite:
            title = self.speaker.play_favorite(self.default_favorite)
            label = f' ({source_label})' if source_label else ''
            return HandlerResult.handled(f'Włączam {title}{label}', favorite=title)
        return HandlerResult.skipped('set default_favorite in Sonos settings to your playlist')

    def _volume(self, intent: str, request) -> HandlerResult:
        current = self.speaker.get_volume()
        if intent == 'volume_set':
            level = str(request.entities.get('level') if request.entities.get('level') is not None else '').strip()
            try:
                target = max(0, min(self.volume_max, int(level)))
            except ValueError:
                return HandlerResult.skipped(f'no usable volume level in {level!r}')
        elif intent == 'volume_up':
            step = self._volume_step(request)
            if step is None:
                return HandlerResult.skipped('invalid volume step')
            if current >= self.volume_max:
                return HandlerResult.skipped('volume is already at or above the configured maximum')
            target = min(self.volume_max, current + step)
        else:
            step = self._volume_step(request)
            if step is None:
                return HandlerResult.skipped('invalid volume step')
            target = max(0, current - step)
        self.speaker.set_volume(target)
        return HandlerResult.handled(f'Głośność {target}', volume_from=current, volume_to=target)

    def _volume_step(self, request) -> int | None:
        raw = request.entities.get('level')
        if raw is None or str(raw).strip() == '':
            return self.volume_step
        try:
            step = int(raw)
        except (TypeError, ValueError):
            return None
        return step if step > 0 else None


def create() -> SonosHandler:
    return SonosHandler()
