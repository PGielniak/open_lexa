"""Read recordings from and publish exports to the archive (SMB share or local dir).

Paths are relative to the archive root and use '/' (``2026-11-02/<stem>.wav``).
They come from the database or from directory listings, never from HTTP input.
"""
from __future__ import annotations

import errno
import logging
import os
from pathlib import Path
from typing import Mapping, Protocol

from .stt import ConfigError

LOG = logging.getLogger('open_lexa.store')


def missing(exc: BaseException) -> bool:
    """True for 'no such file' from both pathlib and smbclient (SMBOSError is not FileNotFoundError)."""
    return isinstance(exc, OSError) and exc.errno == errno.ENOENT


class Store(Protocol):
    def list_dir(self, rel: str) -> list[str]: ...
    def read(self, rel: str) -> bytes: ...
    def write_atomic(self, rel: str, data: bytes) -> None: ...
    def close(self) -> None: ...


class DirStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __repr__(self) -> str:
        return f'DirStore({str(self.root)!r})'

    def _path(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f'path escapes the archive: {rel!r}')
        return path

    def list_dir(self, rel: str) -> list[str]:
        path = self._path(rel)
        return sorted(p.name for p in path.iterdir()) if path.is_dir() else []

    def read(self, rel: str) -> bytes:
        return self._path(rel).read_bytes()

    def write_atomic(self, rel: str, data: bytes) -> None:
        path = self._path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def close(self) -> None:
        pass


def smb_rename_over(src: str, dst_in_share: str, **kwargs) -> None:
    """Atomically rename ``src`` onto ``dst_in_share`` (share-relative), replacing it.

    ``smbclient.replace()`` first opens the destination with FILE_EXECUTE to resolve
    DFS paths; Samba denies that on files without the x bit (this share forces 0664),
    so every overwrite failed with STATUS_ACCESS_DENIED. The share is not DFS, so the
    rename is issued directly. Deleting first is no alternative: vfs_recycle would
    keep a copy of every export in .deleted.
    """
    from smbclient._io import SMBFileTransaction, SMBRawIO, set_info
    from smbprotocol.file_info import FileRenameInformation
    from smbprotocol.open import CreateOptions, FilePipePrinterAccessMask

    with SMBRawIO(src, mode='r', share_access='rwd', desired_access=FilePipePrinterAccessMask.DELETE,
                  create_options=CreateOptions.FILE_OPEN_REPARSE_POINT, **kwargs) as raw:
        with SMBFileTransaction(raw) as transaction:
            info = FileRenameInformation()
            info['replace_if_exists'] = True
            info['file_name'] = dst_in_share
            set_info(transaction, info)


class SmbStore:
    def __init__(self, server: str, share: str, path: str = '', *, username: str | None = None,
                 password: str | None = None, port: int = 445, timeout: int = 10, client=None):
        if client is None:
            import smbclient as client
            self.rename_over = smb_rename_over
        else:
            self.rename_over = client.rename_over
        self.client = client
        self.server = server
        self.username = username
        self.password = password
        self.port = port
        self.timeout = timeout
        self.share_root = rf'\\{server}\{share}'
        self.root = '\\'.join(part for part in (self.share_root, *path.replace('/', '\\').split('\\'))
                              if part)
        self.cache: dict = {}
        self.connected = False

    def __repr__(self) -> str:
        return f'SmbStore({self.root!r})'

    def _path(self, rel: str) -> str:
        parts = [part for part in rel.split('/') if part]
        if any(part in ('.', '..') or '\\' in part for part in parts):
            raise ValueError(f'invalid archive path: {rel!r}')
        return '\\'.join((self.root, *parts))

    def _kwargs(self) -> dict:
        # Same port/cache on every call, otherwise smbclient opens an unauthenticated 445 session.
        return {'port': self.port, 'connection_cache': self.cache}

    def _call(self, operation):
        for attempt in (1, 2):
            try:
                if not self.connected:
                    self.client.register_session(
                        self.server, username=self.username, password=self.password, port=self.port,
                        connection_timeout=self.timeout, connection_cache=self.cache)
                    self.connected = True
                return operation()
            except Exception as exc:
                if missing(exc):
                    raise
                self.close()
                if attempt == 2:
                    raise
                LOG.warning('SMB call failed; reconnecting and retrying once', exc_info=True)

    def list_dir(self, rel: str) -> list[str]:
        try:
            return sorted(self._call(lambda: self.client.listdir(self._path(rel), **self._kwargs())))
        except OSError as exc:
            if missing(exc):
                return []
            raise

    def read(self, rel: str) -> bytes:
        def operation():
            with self.client.open_file(self._path(rel), mode='rb', **self._kwargs()) as source:
                return source.read()
        return self._call(operation)

    def write_atomic(self, rel: str, data: bytes) -> None:
        target = self._path(rel)
        directory, _, name = target.rpartition('\\')
        tmp = f'{directory}\\.{name}.tmp'

        def operation():
            self.client.makedirs(directory, exist_ok=True, **self._kwargs())
            with self.client.open_file(tmp, mode='wb', **self._kwargs()) as out:
                out.write(data)
            self.rename_over(tmp, target[len(self.share_root) + 1:], **self._kwargs())
        self._call(operation)

    def close(self) -> None:
        self.connected = False
        try:
            self.client.reset_connection_cache(fail_on_error=False, connection_cache=self.cache)
        except Exception:
            LOG.debug('SMB connection cache reset failed', exc_info=True)


def create_store(env: Mapping[str, str] | None = None) -> Store:
    """SMB from the receiver's ``SMB_*`` variables, else ``ADMIN_ARCHIVE_DIR`` for local runs."""
    env = os.environ if env is None else env
    server = env.get('SMB_SERVER', '').strip()
    if server:
        share = env.get('SMB_SHARE', '').strip()
        if not share:
            raise ConfigError('SMB_SHARE is required when SMB_SERVER is set')
        try:
            port, timeout = int(env.get('SMB_PORT', '445')), int(env.get('SMB_TIMEOUT', '10'))
        except ValueError as exc:
            raise ConfigError('SMB_PORT and SMB_TIMEOUT must be integers') from exc
        return SmbStore(server, share, env.get('SMB_PATH', 'open-lexa'),
                        username=env.get('SMB_USERNAME') or None, password=env.get('SMB_PASSWORD') or None,
                        port=port, timeout=timeout)
    local = env.get('ADMIN_ARCHIVE_DIR', '').strip()
    if local:
        return DirStore(local)
    raise ConfigError('set SMB_SERVER/SMB_SHARE or ADMIN_ARCHIVE_DIR')
