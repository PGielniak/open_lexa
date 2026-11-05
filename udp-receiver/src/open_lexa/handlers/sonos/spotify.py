"""Spotify Web API search with client-credentials auth (stdlib only).

Used to resolve "puść X na Spotify" into a spotify:track/album/playlist URI,
which the SoCo ShareLink plugin can enqueue on a speaker whose household is
linked to Spotify. Private user playlists do not need this client: they are
played as Sonos favorites instead.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

LOG = logging.getLogger('open_lexa.handlers.sonos')

TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_URL = 'https://api.spotify.com/v1'
# Generic queries usually mean a track. Explicit playlist/album/artist
# requests override this ordering in search_best.
SEARCH_TYPES = ('track', 'album', 'playlist', 'artist')


class SpotifyError(RuntimeError):
    """Spotify refused the request or found nothing playable."""


class SpotifyClient:
    def __init__(self, client_id: str, client_secret: str, market: str = 'PL',
                 timeout: float = 10.0):
        if not client_id or not client_secret:
            raise SpotifyError('Spotify client credentials are not configured')
        self.client_id = client_id
        self.client_secret = client_secret
        self.market = market or 'PL'
        self.timeout = timeout
        self._token: str | None = None
        self._token_expires = 0.0

    # --- auth ----------------------------------------------------------------------

    def _request(self, url: str, *, data: bytes | None = None,
                 headers: dict[str, str] | None = None) -> dict:
        request = urllib.request.Request(url, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode('utf-8') or '{}')
        except urllib.error.HTTPError as exc:
            body = exc.read()[:200].decode('utf-8', errors='replace')
            raise SpotifyError(f'Spotify API returned {exc.code}: {body}') from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SpotifyError(f'Spotify API request failed: {exc}') from exc

    def token(self) -> str:
        if self._token and time.monotonic() < self._token_expires - 60:
            return self._token
        credentials = base64.b64encode(
            f'{self.client_id}:{self.client_secret}'.encode()).decode('ascii')
        body = urllib.parse.urlencode({'grant_type': 'client_credentials'}).encode()
        result = self._request(TOKEN_URL, data=body, headers={
            'Authorization': f'Basic {credentials}',
            'Content-Type': 'application/x-www-form-urlencoded'})
        token = result.get('access_token')
        if not token:
            raise SpotifyError('Spotify token response has no access_token')
        self._token = token
        self._token_expires = time.monotonic() + float(result.get('expires_in') or 3600)
        return token

    # --- search --------------------------------------------------------------------

    def search(self, query: str, types: tuple[str, ...] = SEARCH_TYPES, limit: int = 5) -> dict:
        params = urllib.parse.urlencode({'q': query, 'type': ','.join(types),
                                         'limit': limit, 'market': self.market})
        try:
            return self._request(f'{API_URL}/search?{params}',
                                 headers={'Authorization': f'Bearer {self.token()}'})
        except SpotifyError as exc:
            if '401' in str(exc):  # token revoked/expired mid-flight: retry once
                self._token = None
                return self._request(f'{API_URL}/search?{params}',
                                     headers={'Authorization': f'Bearer {self.token()}'})
            raise

    def search_best(self, query: str, preferred: str | None = None) -> dict:
        """Best playable match: {'uri', 'title', 'type'}.

        When the utterance explicitly requested a playlist/artist/album,
        prefer that category; otherwise a track takes precedence.
        """
        query = (query or '').strip()
        if not query:
            raise SpotifyError('empty Spotify search query')
        result = self.search(query)
        categories = ((preferred,) + tuple(t for t in SEARCH_TYPES if t != preferred)
                      if preferred in SEARCH_TYPES else SEARCH_TYPES)
        for category in categories:
            items = ((result.get(f'{category}s') or {}).get('items')) or []
            for item in items:
                if item.get('uri'):
                    if category == 'artist':
                        # A share link cannot enqueue an artist URI. Resolve a
                        # representative playable track by that artist instead.
                        artist = item.get('name') or query
                        tracks = self.search(f'artist:"{artist}"', types=('track',))
                        playable = ((tracks.get('tracks') or {}).get('items')) or []
                        for track in playable:
                            if track.get('uri') and track.get('is_playable', True):
                                return {'uri': track['uri'],
                                        'title': f"{track.get('name') or artist} — {artist}",
                                        'type': 'track'}
                        continue
                    title = item.get('name') or query
                    if category == 'album':
                        artists = ', '.join(a.get('name', '') for a in item.get('artists', []))
                        title = f'{title} — {artists}' if artists else title
                    return {'uri': item['uri'], 'title': title, 'type': category}
        raise SpotifyError(f'nothing found on Spotify for {query!r}')
