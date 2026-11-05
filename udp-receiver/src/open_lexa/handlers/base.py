"""The contract every command handler plugin implements."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


class HandlerConfigError(ValueError):
    """Raised by :meth:`CommandHandler.configure` for invalid configuration."""


STATUSES = ('handled', 'skipped', 'failed')


@dataclass(frozen=True)
class CommandRequest:
    """One recognised command for a handler.

    ``entities`` are the NLU-extracted parameters keyed by entity type
    (``{'search_query': 'drum and bass', 'level': '20'}``).
    """

    intent: str
    entities: Mapping[str, str] = field(default_factory=dict)
    text: str | None = None
    score: float | None = None
    source: str | None = None
    stem: str | None = None
    session: str | None = None


@dataclass(frozen=True)
class HandlerResult:
    """Execution outcome; ``status`` is one of :data:`STATUSES`."""

    status: str
    response: str | None = None
    error: str | None = None
    detail: Mapping | None = None

    @classmethod
    def handled(cls, response: str | None = None, **detail) -> 'HandlerResult':
        return cls('handled', response=response, detail=detail or None)

    @classmethod
    def skipped(cls, reason: str) -> 'HandlerResult':
        return cls('skipped', error=reason)

    @classmethod
    def failed(cls, error: str) -> 'HandlerResult':
        return cls('failed', error=error)


class CommandHandler:
    """Base class for handler plugins.

    A handler declares the intents it executes; the :class:`~open_lexa.handlers
    .Dispatcher` routes recognised intents to enabled handlers only. Handlers
    must be safe to call from the receiver's single worker thread and must
    never raise out of :meth:`handle` for expected failures — return
    :meth:`HandlerResult.failed` instead.
    """

    id: str = ''
    version: str = '1'
    intents: frozenset[str] = frozenset()
    priority: int = 100  # lower wins when two handlers claim one intent

    def __init__(self) -> None:
        self.config: dict = {}
        self.enabled: bool = False

    def configure(self, config: Mapping) -> None:
        """Store and validate one section of ``_config/handlers.json``.

        Called with ``{}`` when the handler is disabled. Raise
        :class:`HandlerConfigError` to stay disabled with a logged reason.
        """
        self.config = dict(config or {})
        self.enabled = bool(self.config.get('enabled'))

    def handle(self, request: CommandRequest) -> HandlerResult:
        raise NotImplementedError

    def status(self) -> Mapping:
        """Short, secret-free state for the health endpoint / admin UI."""
        return {'enabled': self.enabled}

    def test(self) -> Mapping:
        """Probe the external system; raises with a human-readable message."""
        return self.status()

    def close(self) -> None:
        """Release resources; may be called more than once."""
