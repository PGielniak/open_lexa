"""Poll a JSON configuration file on the archive store.

Used for the lexicon (``_lexicon/lexicon.json``) and the handler
configuration (``_config/handlers.json``). Polling runs on its own thread
because an SMB read can stall; consumers only compare :attr:`version` between
recordings and re-read :attr:`data`. A file that fails to parse keeps the last
good value, exactly like the STT settings watcher.
"""
from __future__ import annotations

import json
import logging
import threading

from .store import missing

LOG = logging.getLogger('open_lexa.watch')


class JsonWatcher:
    """Watch one JSON file on a store; ``version`` changes whenever it does."""

    def __init__(self, store, path: str, interval: float = 60, validate=None):
        self.store = store
        self.path = path
        self.interval = interval
        self.validate = validate
        self.raw: bytes | None = None
        self.data = None
        self.version = 0
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f'watch:{path}', daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2)
        if not self.thread.is_alive():
            self.store.close()

    def poll(self) -> None:
        try:
            data = self.store.read(self.path)
        except OSError as exc:
            if not missing(exc):
                raise
            data = None
        if data == self.raw:
            return
        self.raw = data
        if data is None:
            self.data = None
            self.version += 1
            LOG.info('%s removed', self.path)
            return
        try:
            parsed = json.loads(data.decode('utf-8'))
            if self.validate is not None:
                parsed = self.validate(parsed)
        except ValueError as exc:
            LOG.warning('Ignoring invalid %s: %s', self.path, exc)
            return
        self.data = parsed
        self.version += 1
        LOG.info('%s changed', self.path)

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                self.poll()
            except Exception:
                LOG.warning('Reading %s failed; keeping the current value', self.path, exc_info=True)
            self.stopping.wait(self.interval)
