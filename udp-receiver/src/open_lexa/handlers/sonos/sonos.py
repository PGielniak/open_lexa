"""Minimal SoCo wrapper for one Sonos speaker (or group).

All SoCo imports are lazy so the receiver (and its tests) run without the
dependency until the handler is actually enabled. Transport commands target
the group coordinator when the speaker is grouped, which is what Sonos
requires for play/queue operations.
"""
from __future__ import annotations

import logging

LOG = logging.getLogger('open_lexa.handlers.sonos')


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

    def play_favorite(self, name: str | None = None) -> str:
        """Play a Sonos favorite by (fuzzy) title; None plays the first one."""
        favorites = self.favorites()
        if not favorites:
            raise SonosError('the Sonos household has no favorites')
        chosen = None
        if name:
            needle = name.casefold().strip()
            for favorite in favorites:
                if favorite['title'].casefold().strip() == needle:
                    chosen = favorite
                    break
            if chosen is None:
                for favorite in favorites:
                    title = favorite['title'].casefold()
                    if needle in title or title in needle:
                        chosen = favorite
                        break
            if chosen is None:
                raise SonosError(f'favorite {name!r} not found; available: '
                                 + ', '.join(f['title'] for f in favorites[:20]))
        else:
            chosen = favorites[0]
        self._call('playing the favorite',
                   lambda: self.coordinator.play_uri(chosen['uri'], meta_data=chosen['meta']))
        return chosen['title']

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
