"""Minimal SoCo wrapper for one Sonos speaker (or group).

All SoCo imports are lazy so the receiver (and its tests) run without the
dependency until the handler is actually enabled. Transport commands target
the group coordinator when the speaker is grouped, which is what Sonos
requires for play/queue operations.
"""
from __future__ import annotations

import logging
import re
from xml.sax.saxutils import escape

LOG = logging.getLogger('open_lexa.handlers.sonos')


# Sonos favorites with these URIs are radio stations (TuneIn, plain streams).
RADIO_URI_PREFIXES = ('x-sonosapi-stream:', 'x-rincon-mp3radio:', 'x-sonosapi-radio:',
                      'aac:', 'hls-radio:')
TUNEIN_SID = 254
# Play blocks until a radio stream has buffered, often longer than the 5 s
# request timeout used for other commands.
STREAM_START_TIMEOUT = 20.0
TUNEIN_DESC = 'SA_RINCON65031_'
TUNEIN_DIDL = (
    '<DIDL-Lite xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/" '
    'xmlns:r="urn:schemas-rinconnetworks-com:metadata-1-0/" '
    'xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/">'
    '<item id="F00092020{id}" parentID="L" restricted="true">'
    '<dc:title>{title}</dc:title>'
    '<upnp:class>object.item.audioItem.audioBroadcast</upnp:class>'
    '<desc id="cdudn" nameSpace="urn:schemas-rinconnetworks-com:metadata-1-0/">{desc}</desc>'
    '</item></DIDL-Lite>')


class SonosError(RuntimeError):
    """Speaker unreachable, or Sonos refused the operation."""


class SonosSpeaker:
    def __init__(self, ip: str, timeout: float = 5.0):
        if not ip:
            raise SonosError('no speaker IP configured')
        self.ip = ip
        self.timeout = timeout
        self._device = None

    def _soco(self):
        if self._device is None:
            try:
                from soco import SoCo
                import soco.config
            except ImportError as exc:
                raise SonosError('soco is not installed') from exc
            soco.config.REQUEST_TIMEOUT = self.timeout
            device = SoCo(self.ip)
            # Fail fast on an unreachable speaker instead of inside an action.
            try:
                _ = device.player_name
            except Exception as exc:
                raise SonosError(f'speaker {self.ip} is not reachable: {exc}') from exc
            self._device = device
        return self._device

    @property
    def coordinator(self):
        device = self._soco()
        try:
            return device.group.coordinator or device
        except Exception:
            return device

    def _call(self, label, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except SonosError:
            raise
        except Exception as exc:
            self._device = None  # force a reconnect on the next command
            raise SonosError(f'{label} failed: {exc}') from exc

    # --- status -------------------------------------------------------------------

    def info(self) -> dict:
        device = self._soco()
        track = {}
        try:
            current = device.get_current_track_info() or {}
            track = {'title': current.get('title') or None, 'artist': current.get('artist') or None}
        except Exception:
            pass
        return {'name': self._call('reading the speaker name', lambda: device.player_name),
                'ip': self.ip,
                'state': self._call('reading the transport state',
                                    lambda: device.get_current_transport_info().get('current_transport_state')),
                'volume': self._call('reading the volume', lambda: device.volume),
                'track': track}

    # --- transport ----------------------------------------------------------------

    def play(self) -> None:
        self._call('play', lambda: self.coordinator.play())

    def pause(self) -> None:
        self._call('pause', lambda: self.coordinator.pause())

    def stop(self) -> None:
        self._call('stop', lambda: self.coordinator.stop())

    def next(self) -> None:
        self._call('next', lambda: self.coordinator.next())

    def previous(self) -> None:
        self._call('previous', lambda: self.coordinator.previous())

    # --- volume -------------------------------------------------------------------

    def get_volume(self) -> int:
        return int(self._call('reading the volume', lambda: self._soco().volume))

    def set_volume(self, volume: int) -> int:
        volume = max(0, min(100, int(volume)))
        self._call('setting the volume', self._set_volume, volume)
        return volume

    def _set_volume(self, volume: int) -> None:
        self._soco().volume = volume

    # --- sources ------------------------------------------------------------------

    def favorites(self) -> list[dict]:
        result = self._call('listing Sonos favorites',
                            lambda: self._soco().get_sonos_favorites())
        return [{'title': item.get('title') or '', 'uri': item.get('uri') or '',
                 'meta': item.get('meta') or ''}
                for item in (result or {}).get('favorites', []) if item.get('uri')]

    @staticmethod
    def is_radio(favorite: dict) -> bool:
        uri = favorite.get('uri') or ''
        return (uri.startswith(RADIO_URI_PREFIXES)
                or 'audioItem.audioBroadcast' in (favorite.get('meta') or ''))

    def radio_favorites(self) -> list[dict]:
        return [favorite for favorite in self.favorites() if self.is_radio(favorite)]

    def find_favorite(self, name: str, *, radio_only_fuzzy: bool = False) -> dict | None:
        """Exact title match first, then a substring match (optionally radio only)."""
        favorites = self.favorites()
        needle = (name or '').casefold().strip()
        if not needle:
            return None
        for favorite in favorites:
            if favorite['title'].casefold().strip() == needle:
                return favorite
        for favorite in favorites:
            if radio_only_fuzzy and not self.is_radio(favorite):
                continue
            title = favorite['title'].casefold()
            if needle in title or title in needle:
                return favorite
        return None

    def play_favorite(self, name: str | None = None) -> str:
        """Play a Sonos favorite by (fuzzy) title; None plays the first one."""
        if name:
            chosen = self.find_favorite(name)
            if chosen is None:
                favorites = self.favorites()
                if not favorites:
                    raise SonosError('the Sonos household has no favorites')
                raise SonosError(f'favorite {name!r} not found; available: '
                                 + ', '.join(f['title'] for f in favorites[:20]))
        else:
            favorites = self.favorites()
            if not favorites:
                raise SonosError('the Sonos household has no favorites')
            chosen = favorites[0]
        self.play_favorite_item(chosen)
        return chosen['title']

    def play_favorite_item(self, favorite: dict) -> None:
        timeout = self._start_timeout() if self.is_radio(favorite) else None
        extra = {'timeout': timeout} if timeout else {}
        self._call('playing the favorite',
                   lambda: self.coordinator.play_uri(favorite['uri'], meta=favorite['meta'],
                                                     **extra))

    def _start_timeout(self) -> float:
        return max(float(self.timeout), STREAM_START_TIMEOUT)

    def play_tunein(self, station_id: str, title: str) -> None:
        """Play a TuneIn station by guide id (``s15990``).

        The classic TuneIn service id is accepted by current Sonos firmware
        and is redirected onto the household's linked TuneIn account.
        """
        if not re.fullmatch(r's\d+', station_id or ''):
            raise SonosError(f'invalid TuneIn station id {station_id!r}')
        uri = f'x-sonosapi-stream:{station_id}?sid={TUNEIN_SID}&flags=8224&sn=0'
        meta = TUNEIN_DIDL.format(id=station_id, title=escape(title or station_id),
                                  desc=TUNEIN_DESC)
        self._call(f'playing TuneIn station {station_id}',
                   lambda: self.coordinator.play_uri(uri, meta=meta,
                                                     timeout=self._start_timeout()))

    def current_channel(self) -> str:
        """Name of the playing radio station, '' for queue playback or on error."""
        try:
            info = self.coordinator.get_current_media_info() or {}
        except Exception:
            return ''
        return str(info.get('channel') or '')

    def play_spotify(self, uri: str, title: str = '') -> None:
        """Queue and play a spotify:track/album/playlist URI through ShareLink."""
        coordinator = self.coordinator

        index = self._call(f'adding {uri} to the queue',
                           self._add_share_link, coordinator, uri, title)
        self._call(f'playing {uri} from queue position {index}',
                   coordinator.play_from_queue, index - 1)

    @staticmethod
    def _add_share_link(coordinator, uri: str, title: str) -> int:
        from soco.plugins.sharelink import ShareLinkPlugin
        return ShareLinkPlugin(coordinator).add_share_link_to_queue(uri, dc_title=title)

    def enqueue_spotify(self, uri: str, title: str = '') -> int:
        """Append a Spotify item without changing the current transport state."""
        return self._call(f'adding {uri} to the queue', self._add_share_link,
                          self.coordinator, uri, title)
