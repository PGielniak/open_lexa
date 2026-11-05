"""Sonos playback handler: Spotify search, Sonos favorites, transport, volume.

Configuration section in ``_config/handlers.json`` (edited in the admin
portal under Ustawienia → Sonos / Spotify)::

    {"sonos": {
        "enabled": true,
        "speaker_ip": "192.168.1.42",
        "default_favorite": "Chill",        # Sonos favorite for bare "włącz muzykę"
        "volume_step": 5,
        "volume_max": 60,
        "timeout": 5,
        "spotify": {"client_id": "...", "client_secret": "...", "market": "PL"}
    }}
"""
from __future__ import annotations

import logging

from ..base import CommandHandler, HandlerConfigError, HandlerResult
from .sonos import SonosError, SonosSpeaker
from .spotify import SpotifyClient, SpotifyError

LOG = logging.getLogger('open_lexa.handlers.sonos')

INTENTS = frozenset({
    'play_music', 'play_spotify', 'play_radio',
    'stop', 'pause', 'resume', 'next_track', 'previous_track',
    'volume_up', 'volume_down', 'volume_set',
})


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

    # Overridable in tests; production builds the real clients lazily.
    speaker_factory = SonosSpeaker
    spotify_factory = SpotifyClient
    _speaker = None
    _spotify = None

    def __init__(self) -> None:
        super().__init__()
        self.configure({})

    def configure(self, config) -> None:
        super().configure(config)
        self.speaker_ip = str(self.config.get('speaker_ip') or '').strip()
        if self.enabled and not self.speaker_ip:
            raise HandlerConfigError('speaker_ip is required when the handler is enabled')
        self.default_favorite = str(self.config.get('default_favorite') or '').strip()
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

    # --- introspection ----------------------------------------------------------------

    def status(self) -> dict:
        return {'enabled': self.enabled, 'speaker_ip': self.speaker_ip or None,
                'default_favorite': self.default_favorite or None,
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
        except (SonosError, SpotifyError) as exc:
            return HandlerResult.failed(str(exc))

    def _handle(self, request) -> HandlerResult:
        intent = request.intent
        if intent == 'play_spotify':
            return self._play_spotify(request)
        if intent == 'play_music':
            source = (request.entities.get('source') or '').casefold()
            if source == 'spotify':
                if request.entities.get('search_query'):
                    return self._play_spotify(request)
                return self._play_default(source_label='Spotify')
            return self._play_default()
        if intent == 'play_radio':
            station = request.entities.get('station')
            if not station:
                return HandlerResult.skipped('no station recognised')
            title = self.speaker.play_favorite(station)
            return HandlerResult.handled(f'Włączam {title}', favorite=title)
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
        utterance = (request.text or '').casefold()
        preferred = ('playlist' if 'playlist' in utterance else
                     'album' if 'album' in utterance else
                     'artist' if any(word in utterance for word in ('artyst', 'wykonawc', 'zespoł', 'zespol'))
                     else None)
        result = self.spotify.search_best(query, preferred=preferred)
        self.speaker.play_spotify(result['uri'], result['title'])
        return HandlerResult.handled(f"Gram {result['title']} ze Spotify", **result)

    def _play_default(self, source_label: str = '') -> HandlerResult:
        """Bare "włącz muzykę": the configured Sonos favorite, else resume."""
        if self.default_favorite:
            title = self.speaker.play_favorite(self.default_favorite)
            label = f' ({source_label})' if source_label else ''
            return HandlerResult.handled(f'Włączam {title}{label}', favorite=title)
        if not source_label:
            self.speaker.play()
            return HandlerResult.handled('Wznawiam odtwarzanie')
        # Spotify was requested but there is nothing configured to play.
        return HandlerResult.skipped('no default favorite and no search query')

    def _volume(self, intent: str, request) -> HandlerResult:
        current = self.speaker.get_volume()
        if intent == 'volume_set':
            level = (request.entities.get('level') or '').strip()
            try:
                target = max(0, min(self.volume_max, int(level)))
            except ValueError:
                return HandlerResult.skipped(f'no usable volume level in {level!r}')
        elif intent == 'volume_up':
            target = min(self.volume_max, current + self.volume_step)
        else:
            target = max(0, current - self.volume_step)
        self.speaker.set_volume(target)
        return HandlerResult.handled(f'Głośność {target}', volume_from=current, volume_to=target)


def create() -> SonosHandler:
    return SonosHandler()
