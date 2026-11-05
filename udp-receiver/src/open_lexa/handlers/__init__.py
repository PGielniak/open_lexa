"""Command handler plugin framework.

Each handler lives in its own subpackage of ``open_lexa/handlers/`` and exposes
a ``create()`` factory in its ``handler`` module. Adding a feature means adding
a folder; removing it means deleting (or disabling) the folder. Handlers never
see raw transcripts only — they receive a structured :class:`CommandRequest`
with the recognised intent and entity parameters from :mod:`open_lexa.nlu`.

Runtime configuration comes from ``_config/handlers.json`` on the archive
(published by the admin portal), keyed by handler id::

    {"sonos": {"enabled": true, "speaker_ip": "192.168.1.42", ...}}

A handler without a section, or with ``"enabled": false``, claims no intents.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import time

from .base import (CommandHandler, CommandRequest, HandlerConfigError, HandlerResult,
                   STATUSES)

LOG = logging.getLogger('open_lexa.handlers')

CONFIG_PATH = '_config/handlers.json'

__all__ = ['CONFIG_PATH', 'CommandHandler', 'CommandRequest', 'Dispatcher',
           'HandlerConfigError', 'HandlerResult', 'STATUSES', 'discover', 'validate_config']


def validate_config(config: object) -> dict:
    """Reject malformed exports rather than disabling every live handler."""
    if not isinstance(config, dict) or any(
            not isinstance(name, str) or not isinstance(section, dict)
            for name, section in config.items()):
        raise ValueError('handler configuration must map IDs to objects')
    return config


def discover() -> list[CommandHandler]:
    """Instantiate every handler subpackage that exposes ``handler.create()``."""
    handlers: list[CommandHandler] = []
    for module_info in pkgutil.iter_modules(__path__):
        if not module_info.ispkg:
            continue
        name = f'{__name__}.{module_info.name}.handler'
        try:
            module = importlib.import_module(name)
        except Exception:
            LOG.exception('Handler %s could not be imported; skipping', module_info.name)
            continue
        create = getattr(module, 'create', None)
        if create is None:
            continue
        try:
            handler = create()
        except Exception:
            LOG.exception('Handler %s failed to initialise; skipping', module_info.name)
            continue
        if not handler.id or not handler.intents:
            LOG.warning('Handler %s declares no id/intents; skipping', module_info.name)
            continue
        handlers.append(handler)
    return handlers


class Dispatcher:
    """Route recognised intents to enabled handlers.

    Handlers are disabled until a configuration is applied, so a receiver
    without ``_config/handlers.json`` behaves exactly like the old log-only
    pipeline. One handler owns each intent; on overlap the lower ``priority``
    (then the id) wins deterministically.
    """

    def __init__(self, handlers: list[CommandHandler] | None = None):
        self.handlers: dict[str, CommandHandler] = {h.id: h for h in
                                                    (discover() if handlers is None else handlers)}
        self.enabled: set[str] = set()
        self.routes: dict[str, CommandHandler] = {}
        self.failures: dict[str, int] = {handler_id: 0 for handler_id in self.handlers}

    def apply_config(self, config: dict | None) -> None:
        """(Re)configure handlers from the published JSON; rebuild the routes."""
        config = config if isinstance(config, dict) else {}
        self.enabled = set()
        for handler_id, handler in self.handlers.items():
            section = config.get(handler_id)
            if not isinstance(section, dict) or not section.get('enabled'):
                try:
                    handler.configure({})
                except Exception:
                    LOG.exception('Disabling handler %s failed', handler_id)
                continue
            try:
                handler.configure(section)
            except (HandlerConfigError, ValueError) as exc:
                LOG.error('Handler %s disabled: %s', handler_id, exc)
                handler.enabled = False
                continue
            except Exception:
                LOG.exception('Handler %s disabled after a configuration failure', handler_id)
                handler.enabled = False
                continue
            self.enabled.add(handler_id)
        routes: dict[str, CommandHandler] = {}
        for handler in sorted(self.handlers.values(), key=lambda h: (h.priority, h.id)):
            if handler.id not in self.enabled:
                continue
            for intent in handler.intents:
                if intent in routes:
                    LOG.warning('Intent %s claimed by %s; ignoring %s', intent,
                                routes[intent].id, handler.id)
                    continue
                routes[intent] = handler
        self.routes = routes
        LOG.info('Handlers enabled: %s; intents routed: %d',
                 ', '.join(sorted(self.enabled)) or '(none)', len(routes))

    def dispatch(self, decision, *, text: str | None = None, stem: str | None = None,
                 session: str | None = None) -> dict | None:
        """Execute a decision; returns archive-ready metadata or None.

        ``None`` means nothing was routed (no intent, or no enabled handler
        claims it) — recognition and archiving are never affected by handlers.
        """
        intent = getattr(decision, 'intent', None)
        if not intent or not getattr(decision, 'command', None):
            return None
        handler = self.routes.get(intent)
        if handler is None:
            return None
        request = CommandRequest(intent=intent,
                                 entities=dict(getattr(decision, 'entities_dict', {}) or
                                               dict(getattr(decision, 'entities', {}) or {})),
                                 text=text, score=getattr(decision, 'score', None),
                                 source=getattr(decision, 'source', None),
                                 stem=stem, session=session)
        started = time.monotonic()
        try:
            result = handler.handle(request)
            if not isinstance(result, HandlerResult):
                raise TypeError(f'handler {handler.id} returned {type(result).__name__}, '
                                'expected HandlerResult')
            if result.status not in STATUSES:
                raise ValueError(f'invalid handler status {result.status!r}')
        except Exception as exc:
            self.failures[handler.id] += 1
            LOG.exception('Handler %s failed for intent %s', handler.id, intent)
            result = HandlerResult.failed(str(exc))
        else:
            if result.status == 'failed':
                self.failures[handler.id] += 1
        meta: dict = {'handler': handler.id, 'intent': intent, 'status': result.status,
                      'response': result.response, 'error': result.error,
                      'duration_ms': round((time.monotonic() - started) * 1000)}
        if result.detail:
            meta['detail'] = dict(result.detail)
        LOG.info('Handler %s: %s -> %s%s', handler.id, intent, result.status,
                 f' ({result.error})' if result.error else '')
        return meta

    def summary(self) -> dict:
        return {handler_id: {'enabled': handler_id in self.enabled,
                             'intents': sorted(handler.intents),
                             'failures': self.failures.get(handler_id, 0),
                             'status': self._safe_status(handler)}
                for handler_id, handler in self.handlers.items()}

    @staticmethod
    def _safe_status(handler: CommandHandler) -> dict:
        try:
            return dict(handler.status())
        except Exception:
            return {}

    def close(self) -> None:
        for handler in self.handlers.values():
            try:
                handler.close()
            except Exception:
                LOG.exception('Closing handler %s failed', handler.id)
