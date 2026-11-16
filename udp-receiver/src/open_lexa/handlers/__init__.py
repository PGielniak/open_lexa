"""MCP server handler plugin framework and tool dispatcher.

Each built-in handler lives in its own subpackage of ``open_lexa/handlers/``
and exposes a ``create()`` factory in its ``handler`` module returning an
:class:`MCPServer`. External HTTP MCP servers can also be configured under
``mcp_servers`` (or as entries with a ``url`` field) in ``_config/handlers.json``::

    {
        "sonos": {"enabled": true, "speaker_ip": "192.168.1.42", ...},
        "mcp_servers": {
            "home_assistant": {"enabled": true, "url": "http://ha.lan:8123/mcp"}
        }
    }

Enabled MCP servers expose their tools and descriptions (:meth:`Dispatcher.list_tools`)
to the LLM after every STT transcript; the LLM's tool call is then executed by
:meth:`Dispatcher.dispatch`.
"""
from __future__ import annotations

import importlib
import json
import logging
import pkgutil
import time
from typing import Any, Mapping

from .base import (CommandHandler, CommandRequest, HandlerConfigError, HandlerResult,
                   MCPServer, MCPTool, RemoteMCPServer, STATUSES, serve_mcp_http)

LOG = logging.getLogger('open_lexa.handlers')


def _log_arguments(value):
    """Mask credentials recursively before logging tool arguments."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace('-', '_')
            if any(part in normalized for part in ('password', 'secret', 'token', 'api_key', 'authorization', 'credential')):
                result[key] = '[REDACTED]'
            else:
                result[key] = _log_arguments(item)
        return result
    if isinstance(value, list):
        return [_log_arguments(item) for item in value]
    return value

CONFIG_PATH = '_config/handlers.json'

__all__ = ['CONFIG_PATH', 'CommandHandler', 'CommandRequest', 'Dispatcher',
           'HandlerConfigError', 'HandlerResult', 'MCPServer', 'MCPTool',
           'RemoteMCPServer', 'STATUSES', 'discover', 'serve_mcp_http', 'validate_config']


def validate_config(config: object) -> dict:
    """Reject malformed exports rather than disabling every live MCP server."""
    if not isinstance(config, dict) or any(
            not isinstance(name, str) or not isinstance(section, dict)
            for name, section in config.items()):
        raise ValueError('handler configuration must map IDs to objects')
    mcp_servers = config.get('mcp_servers')
    if mcp_servers is not None:
        if not isinstance(mcp_servers, dict) or any(
                not isinstance(sid, str) or not isinstance(spec, dict)
                for sid, spec in mcp_servers.items()):
            raise ValueError('mcp_servers must map server IDs to configuration objects')
    return config


def discover() -> list[MCPServer]:
    """Instantiate every local MCP server subpackage that exposes ``handler.create()``."""
    handlers: list[MCPServer] = []
    for module_info in pkgutil.iter_modules(__path__):
        if not module_info.ispkg:
            continue
        name = f'{__name__}.{module_info.name}.handler'
        try:
            module = importlib.import_module(name)
        except Exception:
            LOG.exception('MCP handler %s could not be imported; skipping', module_info.name)
            continue
        create = getattr(module, 'create', None)
        if create is None:
            continue
        try:
            handler = create()
        except Exception:
            LOG.exception('MCP handler %s failed to initialise; skipping', module_info.name)
            continue
        if not handler.id or not handler.list_tools():
            LOG.warning('MCP handler %s declares no id/tools; skipping', module_info.name)
            continue
        handlers.append(handler)
    return handlers


def _extract_remote_specs(config: Mapping[str, Any], local_ids: set[str]) -> dict[str, dict]:
    """Collect remote HTTP MCP server specs from ``mcp_servers`` and top-level URL sections."""
    specs: dict[str, dict] = {}
    nested = config.get('mcp_servers')
    if isinstance(nested, Mapping):
        for server_id, section in nested.items():
            if isinstance(server_id, str) and isinstance(section, Mapping) and server_id not in local_ids:
                specs[server_id] = dict(section)
    for key, section in config.items():
        if key == 'mcp_servers' or key in local_ids:
            continue
        if isinstance(section, Mapping) and section.get('url'):
            specs[key] = dict(section)
    return specs


class Dispatcher:
    """Manage local and remote MCP servers, expose their tools, and execute tool calls.

    MCP servers are disabled until a configuration is applied. One server owns
    each tool name; on overlap the lower ``priority`` (then the id) wins
    deterministically.
    """

    def __init__(self, handlers: list[MCPServer] | None = None):
        local = discover() if handlers is None else handlers
        self.local_handlers: dict[str, MCPServer] = {h.id: h for h in local}
        self.remote_handlers: dict[str, RemoteMCPServer] = {}
        self.handlers: dict[str, MCPServer] = dict(self.local_handlers)
        self.enabled: set[str] = set()
        self.routes: dict[str, MCPServer] = {}
        self.tools: dict[str, MCPTool] = {}
        self.failures: dict[str, int] = {handler_id: 0 for handler_id in self.handlers}

    def apply_config(self, config: dict | None) -> None:
        """(Re)configure local and remote MCP servers and rebuild tool routes."""
        config = config if isinstance(config, dict) else {}
        remote_specs = _extract_remote_specs(config, set(self.local_handlers))
        for stale_id in set(self.remote_handlers) - set(remote_specs):
            removed = self.remote_handlers.pop(stale_id)
            try:
                removed.close()
            except Exception:
                LOG.exception('Closing removed MCP server %s failed', stale_id)
            self.handlers.pop(stale_id, None)
            self.failures.pop(stale_id, None)
        for server_id, spec in remote_specs.items():
            if server_id not in self.remote_handlers:
                remote = RemoteMCPServer(server_id)
                self.remote_handlers[server_id] = remote
                self.handlers[server_id] = remote
                self.failures.setdefault(server_id, 0)

        self.enabled = set()
        for handler_id, handler in self.handlers.items():
            section = (remote_specs.get(handler_id) if handler_id in self.remote_handlers
                       else config.get(handler_id))
            if not isinstance(section, dict) or not section.get('enabled'):
                try:
                    handler.configure({})
                except Exception:
                    LOG.exception('Disabling MCP server %s failed', handler_id)
                continue
            try:
                handler.configure(section)
            except (HandlerConfigError, ValueError) as exc:
                LOG.error('MCP server %s disabled: %s', handler_id, exc)
                handler.enabled = False
                continue
            except Exception:
                LOG.exception('MCP server %s disabled after a configuration failure', handler_id)
                handler.enabled = False
                continue
            self.enabled.add(handler_id)

        self._rebuild_routes()
        LOG.info('MCP servers enabled: %s; tools exposed: %d',
                 ', '.join(sorted(self.enabled)) or '(none)', len(self.routes))

    def _rebuild_routes(self) -> None:
        routes: dict[str, MCPServer] = {}
        tools: dict[str, MCPTool] = {}
        for handler in sorted(self.handlers.values(), key=lambda h: (h.priority, h.id)):
            if handler.id not in self.enabled:
                continue
            for tool in handler.list_tools():
                if tool.name in routes:
                    LOG.warning('MCP tool %s claimed by %s; ignoring %s', tool.name,
                                routes[tool.name].id, handler.id)
                    continue
                routes[tool.name] = handler
                tools[tool.name] = MCPTool(tool.name, tool.description, tool.input_schema,
                                           server_id=handler.id)
        self.routes = routes
        self.tools = tools

    def list_tools(self) -> list[MCPTool]:
        """Return all MCP tools currently available to the LLM."""
        for handler_id in self.enabled:
            handler = self.remote_handlers.get(handler_id)
            if handler is not None and not handler._cached_tools:
                if handler.list_tools():
                    self._rebuild_routes()
        return list(self.tools.values())

    def dispatch(self, decision, *, text: str | None = None, stem: str | None = None,
                 session: str | None = None) -> dict | None:
        """Execute an LLM-selected MCP tool; returns archive-ready metadata or None."""
        tool_name = (getattr(decision, 'tool', None)
                     or getattr(decision, 'command', None)
                     or getattr(decision, 'intent', None))
        if not tool_name or not getattr(decision, 'command', tool_name):
            return None
        handler = self.routes.get(tool_name)
        if handler is None:
            return None
        raw_args = (getattr(decision, 'arguments_dict', None)
                    or getattr(decision, 'arguments', None)
                    or getattr(decision, 'entities_dict', None)
                    or dict(getattr(decision, 'entities', {}) or {}))
        arguments = dict(raw_args) if isinstance(raw_args, Mapping) else {}
        LOG.info('mcp_call session=%s server=%s tool=%s arguments=%s', session or '-',
                 handler.id, tool_name, json.dumps(_log_arguments(arguments), ensure_ascii=False,
                                                    separators=(',', ':'), default=str))
        request = CommandRequest(tool=tool_name, arguments=arguments, text=text,
                                 score=getattr(decision, 'score', None),
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
            self.failures[handler.id] = self.failures.get(handler.id, 0) + 1
            LOG.exception('mcp_exception session=%s server=%s tool=%s',
                          session or '-', handler.id, tool_name)
            result = HandlerResult.failed(str(exc))
        else:
            if result.status == 'failed':
                self.failures[handler.id] = self.failures.get(handler.id, 0) + 1
        meta: dict = {
            'handler': handler.id,
            'server': handler.id,
            'tool': tool_name,
            'intent': tool_name,
            'arguments': arguments,
            'status': result.status,
            'response': result.response,
            'error': result.error,
            'duration_ms': round((time.monotonic() - started) * 1000),
        }
        if result.detail:
            meta['detail'] = dict(result.detail)
        if result.status == 'handled':
            LOG.info('mcp_result session=%s server=%s tool=%s status=%s duration_ms=%d',
                     session or '-', handler.id, tool_name, result.status, meta['duration_ms'])
        else:
            LOG.warning('mcp_result session=%s server=%s tool=%s status=%s duration_ms=%d error=%s',
                        session or '-', handler.id, tool_name, result.status, meta['duration_ms'],
                        json.dumps(result.error or 'no error detail', ensure_ascii=False))
        return meta

    def summary(self) -> dict:
        result: dict[str, dict] = {}
        for handler_id, handler in self.handlers.items():
            tool_names = sorted(t.name for t in handler.list_tools())
            result[handler_id] = {
                'enabled': handler_id in self.enabled,
                'tools': tool_names,
                'intents': tool_names,
                'failures': self.failures.get(handler_id, 0),
                'status': self._safe_status(handler),
            }
        return result

    @staticmethod
    def _safe_status(handler: MCPServer) -> dict:
        try:
            return dict(handler.status())
        except Exception:
            return {}

    def close(self) -> None:
        for handler in self.handlers.values():
            try:
                handler.close()
            except Exception:
                LOG.exception('Closing MCP server %s failed', handler.id)
