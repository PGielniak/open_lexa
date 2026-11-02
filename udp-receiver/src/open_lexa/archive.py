"""Keep every recording and its transcript on an SMB share for later retraining.

Each recording becomes ``<SMB_PATH>/<YYYY-MM-DD>/<stem>.wav`` plus a ``<stem>.json``
sidecar. The sidecar is written last, so its presence marks a complete pair.
The SMB client is pure Python (smbprotocol); no kernel CIFS mount is needed.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Mapping, Protocol

from .stt import ConfigError

LOG = logging.getLogger('open_lexa.archive')


class Archive(Protocol):
    def save(self, stem: str, wav: bytes, meta: dict) -> None:
        """Store one recording; ``stem`` is unique and contains no path separators."""

    def close(self) -> None:
        """Release connections. May be called more than once."""


class SmbArchive:
    def __init__(self, server: str, share: str, path: str = 'open-lexa', *,
                 username: str | None = None, password: str | None = None,
                 port: int = 445, timeout: int = 10, client=None):
        if client is None:
            import smbclient as client
        self.client = client
        self.server = server
        self.username = username
        self.password = password
        self.port = port
        self.timeout = timeout
        self.root = '\\'.join(part for part in (rf'\\{server}', share, *path.replace('/', '\\').split('\\'))
                              if part)
        self.cache: dict = {}
        self.connected = False
        self.dirs: set[str] = set()

    def __repr__(self) -> str:
        return f'SmbArchive({self.root!r})'

    def _connect(self) -> None:
        if not self.connected:
            self.client.register_session(
                self.server, username=self.username, password=self.password, port=self.port,
                connection_timeout=self.timeout, connection_cache=self.cache)
            self.connected = True

    def _session(self) -> dict:
        # Every path operation must name the same port/cache, otherwise smbclient
        # silently opens a separate unauthenticated session on 445.
        return {'port': self.port, 'connection_cache': self.cache}

    def save(self, stem: str, wav: bytes, meta: dict) -> None:
        for attempt in (1, 2):
            try:
                self._save(stem, wav, meta)
                return
            except Exception:
                # A dropped connection or a day directory deleted on the server leaves
                # stale client state; start from scratch once before giving up.
                self.close()
                if attempt == 2:
                    raise
                LOG.warning('SMB write failed; reconnecting and retrying once', exc_info=True)

    def _save(self, stem: str, wav: bytes, meta: dict) -> None:
        directory = f'{self.root}\\{meta["created"][:10]}'
        self._connect()
        if directory not in self.dirs:
            self.client.makedirs(directory, exist_ok=True, **self._session())
            self.dirs.add(directory)
        for suffix, data in (('wav', wav),
                             ('json', json.dumps(meta, ensure_ascii=False, indent=2).encode())):
            with self.client.open_file(f'{directory}\\{stem}.{suffix}', mode='wb',
                                       **self._session()) as out:
                out.write(data)

    def close(self) -> None:
        self.connected = False
        self.dirs.clear()
        try:
            self.client.reset_connection_cache(fail_on_error=False, connection_cache=self.cache)
        except Exception:
            LOG.debug('SMB connection cache reset failed', exc_info=True)


def create_archive(env: Mapping[str, str] | None = None) -> SmbArchive | None:
    """Build the SMB archive from ``SMB_*`` variables; ``None`` when SMB_SERVER is unset."""
    env = os.environ if env is None else env
    server = env.get('SMB_SERVER', '').strip()
    if not server:
        return None
    share = env.get('SMB_SHARE', '').strip()
    if not share:
        raise ConfigError('SMB_SHARE is required when SMB_SERVER is set')
    try:
        port = int(env.get('SMB_PORT', '445'))
        timeout = int(env.get('SMB_TIMEOUT', '10'))
    except ValueError as exc:
        raise ConfigError('SMB_PORT and SMB_TIMEOUT must be integers') from exc
    return SmbArchive(server, share, env.get('SMB_PATH', 'open-lexa'),
                      username=env.get('SMB_USERNAME') or None,
                      password=env.get('SMB_PASSWORD') or None,
                      port=port, timeout=timeout)
