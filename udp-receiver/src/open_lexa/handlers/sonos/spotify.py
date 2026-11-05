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
import random
import time
import urllib.error
import urllib.parse
import urllib.request

LOG = logging.getLogger('open_lexa.handlers.sonos')

TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_URL = 'https://api.spotify.com/v1'
# Explicit track/playlist/album requests override this ordering in search_best.
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

        Exact artist-name matches use a playlist for that artist. Explicit
        track requests bypass this rule; otherwise tracks take precedence.
        """
        query = (query or '').strip()
        if not query:
            raise SpotifyError('empty Spotify search query')
        result = self.search(query)
        if preferred is None:
            artists = ((result.get('artists') or {}).get('items')) or []
            if any(item and (item.get('name') or '').casefold() == query.casefold()
                   for item in artists):
                return self.search_artist_playlist(query)
        categories = ((preferred,) if preferred in ('playlist', 'album') else SEARCH_TYPES)
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

    def search_track(self, query: str) -> dict:
        """Search tracks only; do not fall back to an album or playlist."""
        items = ((self.search(query, types=('track',)).get('tracks') or {}).get('items')) or []
        for item in items:
            if item and item.get('uri') and item.get('is_playable', True):
                return {'uri': item['uri'], 'title': item.get('name') or query,
                        'type': 'track'}
        raise SpotifyError(f'no playable tracks found on Spotify for {query!r}')

    @staticmethod
    def _playlists(result: dict) -> list[dict]:
        items = ((result.get('playlists') or {}).get('items')) or []
        return [item for item in items if item and item.get('uri') and
                (item.get('tracks') or {}).get('total', 1) != 0]

    def search_artist_playlist(self, artist: str) -> dict:
        """Find an artist's 'This Is' playlist, then a named artist playlist."""
        artist = (artist or '').strip()
        if not artist:
            raise SpotifyError('empty artist name')
        for query in (f'This Is {artist}', artist):
            playlists = self._playlists(self.search(query, types=('playlist',), limit=20))
            if query == artist:
                named = [item for item in playlists
                         if artist.casefold() in (item.get('name') or '').casefold()]
            else:
                named = [item for item in playlists
                         if (item.get('name') or '').casefold() == query.casefold()]
            if named:
                # Prefer the editorial Spotify playlist when it exists.
                named.sort(key=lambda item: (item.get('owner') or {}).get('id') != 'spotify')
                chosen = named[0]
                return {'uri': chosen['uri'], 'title': chosen.get('name') or artist,
                        'type': 'playlist'}
        raise SpotifyError(f'no playlist found for artist {artist!r}')

    def search_genre_playlist(self, genre: str) -> dict:
        """Pick a playable playlist from Spotify's most relevant genre results."""
        result = self.search(genre, types=('playlist',), limit=10)
        playlists = self._playlists(result)
        if not playlists:
            raise SpotifyError(f'no playlists found on Spotify for {genre!r}')
        chosen = random.choice(playlists)
        return {'uri': chosen['uri'], 'title': chosen.get('name') or genre,
                'type': 'playlist'}
