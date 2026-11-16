"""TuneIn station search (stdlib only, no account needed).

Resolves a spoken station name ("RMF FM", "Radio Zet") to a TuneIn guide id
such as ``s15990``. :meth:`SonosSpeaker.play_tunein` turns that id into a
``x-sonosapi-stream`` URI; Sonos maps it onto the household's TuneIn account.
The public profiles endpoint is used because the older OPML search misses
major stations (e.g. Radio Zet).
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

LOG = logging.getLogger('open_lexa.handlers.sonos')

SEARCH_URL = 'https://api.tunein.com/profiles'
# Without ``formats`` the search silently drops major stations (Radio Zet).
SEARCH_FORMATS = 'mp3,aac,ogg,hls'


class TuneInError(RuntimeError):
    """TuneIn is unreachable or no station matches the name."""


def compact(name: str) -> str:
    """Casefolded, accent-free alphanumerics: 'Anty Radio' == 'Antyradio'."""
    text = unicodedata.normalize('NFKD', (name or '').replace('ł', 'l').replace('Ł', 'L'))
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r'[^0-9a-z]+', '', text.casefold())


def _without_radio(name: str) -> str:
    return compact(name).replace('radio', '')


def is_generic_radio(name: str) -> bool:
    """'', 'radio', 'radia', 'radyjko': no station named, so it would match anything."""
    return re.sub(r'^radi?y?[oaeu]?(jko)?$', '', _without_radio(name)) == ''


class TuneInClient:
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def search(self, query: str) -> list[dict]:
        """Playable stations for ``query``: [{'id', 'title', 'subtitle'}]."""
        params = urllib.parse.urlencode({'query': query, 'fullTextSearch': 'true',
                                         'formats': SEARCH_FORMATS})
        request = urllib.request.Request(f'{SEARCH_URL}?{params}',
                                         headers={'Accept': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8') or '{}')
        except urllib.error.HTTPError as exc:
            raise TuneInError(f'TuneIn search returned {exc.code}') from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise TuneInError(f'TuneIn search failed: {exc}') from exc
        stations = []
        for container in data.get('Items') or []:
            if container.get('ContainerType') != 'Stations':
                continue
            for item in container.get('Children') or []:
                guide_id = str(item.get('GuideId') or '')
                playable = ((item.get('Actions') or {}).get('Play') or {}).get('CanPlay', True)
                if guide_id.startswith('s') and item.get('Title') and playable:
                    stations.append({'id': guide_id, 'title': item['Title'],
                                     'subtitle': item.get('Subtitle') or ''})
        return stations

    def find_station(self, name: str) -> dict:
        """Best station for a spoken name; refuses loose, unrelated matches.

        Full-text search returns *something* for any query ("Trójka" yields
        "Dark Edge Radio"), so only exact or containing name matches count.
        """
        name = (name or '').strip()
        if not name:
            raise TuneInError('empty station name')
        stations = self.search(name)
        wanted, bare = compact(name), _without_radio(name)
        for matches in (
            lambda title: compact(title) == wanted,
            lambda title: bool(bare) and _without_radio(title) == bare,
            lambda title: wanted in compact(title),
        ):
            for station in stations:
                if matches(station['title']):
                    return station
        suggestions = ', '.join(station['title'] for station in stations[:5])
        raise TuneInError(f'no TuneIn station matches {name!r}'
                          + (f'; similar: {suggestions}' if suggestions else ''))
