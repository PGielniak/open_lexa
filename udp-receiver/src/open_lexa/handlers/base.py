"""MCP (Model Context Protocol) server and client contract for command handlers."""
from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

LOG = logging.getLogger('open_lexa.handlers')

MCP_PROTOCOL_VERSION = '2025-03-26'


class HandlerConfigError(ValueError):
    """Raised by :meth:`MCPServer.configure` for invalid configuration."""


STATUSES = ('handled', 'skipped', 'failed')

EMPTY_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {},
    'additionalProperties': False,
}


@dataclass(frozen=True)
class MCPTool:
    """One tool exposed by an MCP server with its description and JSON Schema."""

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=lambda: dict(EMPTY_SCHEMA))
    server_id: str = ''

    def to_mcp(self) -> dict[str, Any]:
        """Standard MCP ``tools/list`` entry."""
        return {
            'name': self.name,
            'description': self.description,
            'inputSchema': dict(self.input_schema),
        }

    def to_openai(self) -> dict[str, Any]:
        """OpenAI Chat Completions function-calling tool definition."""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': dict(self.input_schema),
            },
        }


@dataclass(frozen=True)
class CommandRequest:
    """One MCP tool call request for a handler.

    ``tool`` is the MCP tool name and ``arguments`` are the LLM-provided
    parameters (e.g. ``{'query': 'drum and bass', 'level': 20}``). ``intent``
    and ``entities`` remain available as aliases.
    """

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    text: str | None = None
    score: float | None = None
    source: str | None = None
    stem: str | None = None
    session: str | None = None

    def __init__(self, tool: str | None = None, arguments: Mapping[str, Any] | None = None, *,
                 intent: str | None = None, entities: Mapping[str, Any] | None = None,
                 text: str | None = None, score: float | None = None,
                 source: str | None = None, stem: str | None = None,
                 session: str | None = None) -> None:
        resolved_tool = tool if tool is not None else (intent or '')
        resolved_args = dict(arguments if arguments is not None else (entities or {}))
        object.__setattr__(self, 'tool', resolved_tool)
        object.__setattr__(self, 'arguments', resolved_args)
        object.__setattr__(self, 'text', text)
        object.__setattr__(self, 'score', score)
        object.__setattr__(self, 'source', source)
        object.__setattr__(self, 'stem', stem)
        object.__setattr__(self, 'session', session)

    @property
    def intent(self) -> str:
        return self.tool

    @property
    def entities(self) -> Mapping[str, Any]:
        return self.arguments


@dataclass(frozen=True)
class HandlerResult:
    """Execution outcome of an MCP tool call; ``status`` is one of :data:`STATUSES`."""

    status: str
    response: str | None = None
    error: str | None = None
    detail: Mapping[str, Any] | None = None

    @classmethod
    def handled(cls, response: str | None = None, **detail: Any) -> 'HandlerResult':
        return cls('handled', response=response, detail=detail or None)

    @classmethod
    def skipped(cls, reason: str) -> 'HandlerResult':
        return cls('skipped', error=reason)

    @classmethod
    def failed(cls, error: str) -> 'HandlerResult':
        return cls('failed', error=error)

    def to_mcp(self) -> dict[str, Any]:
        """Standard MCP ``CallToolResult`` payload."""
        text = self.response or self.error or self.status
        structured: dict[str, Any] = {
            'status': self.status,
            'response': self.response,
            'error': self.error,
        }
        if self.detail:
            structured.update(dict(self.detail))
        return {
            'content': [{'type': 'text', 'text': str(text)}],
            'isError': self.status != 'handled',
            'structuredContent': structured,
        }

    @classmethod
    def from_mcp(cls, payload: Mapping[str, Any]) -> 'HandlerResult':
        """Convert an MCP ``CallToolResult`` back into a :class:`HandlerResult`."""
        structured = payload.get('structuredContent')
        if isinstance(structured, Mapping) and structured.get('status') in STATUSES:
            status = str(structured['status'])
            response = structured.get('response')
            error = structured.get('error')
            detail = {k: v for k, v in structured.items()
                      if k not in ('status', 'response', 'error')}
            return cls(status,
                       response=str(response) if response is not None else None,
                       error=str(error) if error is not None else None,
                       detail=detail or None)
        texts: list[str] = []
        for item in payload.get('content') or []:
            if isinstance(item, Mapping) and item.get('type') == 'text' and item.get('text'):
                texts.append(str(item['text']))
        message = '\n'.join(texts).strip() or None
        if payload.get('isError'):
            return cls.failed(message or 'MCP tool returned an error')
        return cls.handled(message)


class MCPServer:
    """Base class for local and remote MCP servers.

    A handler is an MCP server that exposes ``tools`` (:class:`MCPTool` with
    descriptions and JSON schemas) and executes them via :meth:`call_tool` /
    :meth:`handle`. It also speaks the MCP JSON-RPC 2.0 protocol directly via
    :meth:`handle_mcp` (``initialize``, ``ping``, ``tools/list``, ``tools/call``).
    """

    id: str = ''
    version: str = '1'
    tools: tuple[MCPTool, ...] = ()
    intents: frozenset[str] = frozenset()
    priority: int = 100  # lower wins when two servers expose the same tool name

    def __init__(self) -> None:
        self.config: dict = {}
        self.enabled: bool = False

    def configure(self, config: Mapping) -> None:
        """Store and validate one section of ``_config/handlers.json``."""
        self.config = dict(config or {})
        self.enabled = bool(self.config.get('enabled'))

    def list_tools(self) -> list[MCPTool]:
        """Return MCP tools exposed by this server."""
        if self.tools:
            return [MCPTool(t.name, t.description, t.input_schema, server_id=self.id)
                    for t in self.tools]
        return [MCPTool(name=name, description=f'Execute {name}', server_id=self.id)
                for name in sorted(self.intents)]

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(tool.name for tool in self.list_tools())

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None, *,
                  text: str | None = None, stem: str | None = None,
                  session: str | None = None) -> HandlerResult:
        """Invoke one MCP tool by name and return a :class:`HandlerResult`."""
        request = CommandRequest(tool=name, arguments=arguments or {}, text=text,
                                 source='llm', stem=stem, session=session)
        return self.handle(request)

    def handle(self, request: CommandRequest) -> HandlerResult:
        raise NotImplementedError

    def handle_mcp(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        """Process one MCP JSON-RPC 2.0 message; returns None for notifications."""
        if not isinstance(message, Mapping) or message.get('jsonrpc') != '2.0':
            return {'jsonrpc': '2.0', 'id': None,
                    'error': {'code': -32600, 'message': 'Invalid Request'}}
        method = str(message.get('method') or '')
        req_id = message.get('id')
        if 'id' not in message:
            # Notification (e.g. notifications/initialized).
            return None
        params = message.get('params') if isinstance(message.get('params'), Mapping) else {}
        if method == 'initialize':
            return {
                'jsonrpc': '2.0',
                'id': req_id,
                'result': {
                    'protocolVersion': MCP_PROTOCOL_VERSION,
                    'capabilities': {'tools': {'listChanged': False}},
                    'serverInfo': {'name': self.id or 'open-lexa-handler',
                                   'version': self.version},
                },
            }
        if method == 'ping':
            return {'jsonrpc': '2.0', 'id': req_id, 'result': {}}
        if method == 'tools/list':
            return {
                'jsonrpc': '2.0',
                'id': req_id,
                'result': {'tools': [tool.to_mcp() for tool in self.list_tools()]},
            }
        if method == 'tools/call':
            name = str(params.get('name') or '').strip()
            arguments = params.get('arguments')
            if not isinstance(arguments, Mapping):
                arguments = {}
            known = {tool.name for tool in self.list_tools()}
            if not name or name not in known:
                return {
                    'jsonrpc': '2.0',
                    'id': req_id,
                    'error': {'code': -32602, 'message': f'Unknown tool: {name!r}'},
                }
            try:
                result = self.call_tool(name, arguments)
            except Exception as exc:
                LOG.exception('mcp_server_error server=%s tool=%s', self.id, name)
                result = HandlerResult.failed(str(exc))
            return {'jsonrpc': '2.0', 'id': req_id, 'result': result.to_mcp()}
        return {
            'jsonrpc': '2.0',
            'id': req_id,
            'error': {'code': -32601, 'message': f'Method not found: {method}'},
        }

    def status(self) -> Mapping:
        """Short, secret-free state for the health endpoint / admin UI."""
        return {'enabled': self.enabled, 'tools': [t.name for t in self.list_tools()]}

    def test(self) -> Mapping:
        """Probe the server or underlying device; raises with a human-readable message."""
        return self.status()

    def close(self) -> None:
        """Release resources; may be called more than once."""


# Backwards-compatible alias used by plugins and tests.
CommandHandler = MCPServer


def _parse_sse_json(raw: str) -> dict[str, Any]:
    """Extract the last JSON-RPC message from a ``text/event-stream`` payload."""
    last: dict[str, Any] | None = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith('data:'):
            data = line[5:].strip()
            if data:
                last = json.loads(data)
    if last is None:
        raise ValueError('empty SSE response from MCP server')
    return last


class RemoteMCPServer(MCPServer):
    """Client for an external MCP server reachable over Streamable HTTP (JSON-RPC 2.0)."""

    def __init__(self, server_id: str, config: Mapping | None = None, *,
                 transport: Callable[[str, dict, dict[str, str], float], tuple[dict | None, dict[str, str]]] | None = None) -> None:
        super().__init__()
        self.id = server_id
        self.url: str = ''
        self.api_key: str = ''
        self.timeout: float = 10.0
        self._session_id: str | None = None
        self._rpc_id: int = 0
        self._cached_tools: list[MCPTool] = []
        self._transport = transport
        if config is not None:
            self.configure(config)

    def configure(self, config: Mapping) -> None:
        super().configure(config)
        self.url = str(self.config.get('url') or '').strip()
        self.api_key = str(self.config.get('api_key') or '').strip()
        try:
            self.timeout = max(1.0, min(60.0, float(self.config.get('timeout') or 10.0)))
        except (TypeError, ValueError):
            self.timeout = 10.0
        self._session_id = None
        self._cached_tools = []
        if not self.enabled:
            return
        if not self.url.startswith(('http://', 'https://')):
            raise HandlerConfigError(f'MCP server {self.id!r} requires an http(s):// url')
        try:
            self.refresh_tools()
        except Exception as exc:
            LOG.warning('Could not fetch tools from MCP server %s (%s) during configure: %s',
                        self.id, self.url, exc)

    def _headers(self) -> dict[str, str]:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        if self._session_id:
            headers['Mcp-Session-Id'] = self._session_id
        extra = self.config.get('headers')
        if isinstance(extra, Mapping):
            for k, v in extra.items():
                if k and v is not None:
                    headers[str(k)] = str(v)
        return headers

    def _send(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        headers = self._headers()
        if self._transport is not None:
            response, resp_headers = self._transport(self.url, payload, headers, self.timeout)
        else:
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            req = urllib.request.Request(self.url, data=body, headers=headers, method='POST')
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                    raw = resp.read(1024 * 1024).decode('utf-8')
                    ctype = resp_headers.get('content-type', '')
                    if not raw.strip():
                        response = None
                    elif 'text/event-stream' in ctype:
                        response = _parse_sse_json(raw)
                    else:
                        response = json.loads(raw)
            except urllib.error.HTTPError as exc:
                detail = exc.read(4096).decode('utf-8', errors='replace')
                raise RuntimeError(f'MCP HTTP {exc.code}: {detail or exc.reason}') from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f'MCP connection failed: {exc.reason}') from exc
        sid = (resp_headers or {}).get('mcp-session-id') or (resp_headers or {}).get('Mcp-Session-Id')
        if sid:
            self._session_id = str(sid)
        if response is None:
            return None
        if 'error' in response and response['error']:
            err = response['error']
            msg = err.get('message') if isinstance(err, Mapping) else str(err)
            raise RuntimeError(f'MCP RPC error: {msg}')
        result = response.get('result')
        return result if isinstance(result, dict) else {}

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def initialize(self) -> dict[str, Any]:
        result = self._send({
            'jsonrpc': '2.0',
            'id': self._next_id(),
            'method': 'initialize',
            'params': {
                'protocolVersion': MCP_PROTOCOL_VERSION,
                'capabilities': {},
                'clientInfo': {'name': 'open-lexa-receiver', 'version': '0.1.0'},
            },
        }) or {}
        try:
            self._send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        except Exception:
            pass
        return result

    def refresh_tools(self) -> list[MCPTool]:
        """Connect to the remote MCP server and fetch its ``tools/list``."""
        if not self._session_id:
            self.initialize()
        result = self._send({
            'jsonrpc': '2.0',
            'id': self._next_id(),
            'method': 'tools/list',
            'params': {},
        }) or {}
        tools: list[MCPTool] = []
        for item in result.get('tools') or []:
            if not isinstance(item, Mapping) or not item.get('name'):
                continue
            schema = item.get('inputSchema')
            if not isinstance(schema, Mapping):
                schema = dict(EMPTY_SCHEMA)
            tools.append(MCPTool(
                name=str(item['name']).strip(),
                description=str(item.get('description') or '').strip(),
                input_schema=dict(schema),
                server_id=self.id,
            ))
        self._cached_tools = tools
        return list(tools)

    def list_tools(self) -> list[MCPTool]:
        if self.enabled and not self._cached_tools:
            try:
                self.refresh_tools()
            except Exception as exc:
                LOG.warning('Lazy tools/list failed for MCP server %s: %s', self.id, exc)
        return list(self._cached_tools)

    def handle(self, request: CommandRequest) -> HandlerResult:
        if not self.enabled:
            return HandlerResult.skipped(f'MCP server {self.id} is disabled')
        try:
            if not self._session_id:
                self.initialize()
            result = self._send({
                'jsonrpc': '2.0',
                'id': self._next_id(),
                'method': 'tools/call',
                'params': {'name': request.tool, 'arguments': dict(request.arguments)},
            }) or {}
            return HandlerResult.from_mcp(result)
        except Exception as exc:
            LOG.exception('mcp_remote_error session=%s server=%s tool=%s',
                          request.session or '-', self.id, request.tool)
            return HandlerResult.failed(str(exc))

    def status(self) -> Mapping:
        return {
            'enabled': self.enabled,
            'url': self.url or None,
            'tools': [t.name for t in self._cached_tools],
        }

    def test(self) -> Mapping:
        """Perform a live ``initialize`` + ``tools/list`` round-trip."""
        self._session_id = None
        info = self.initialize()
        tools = self.refresh_tools()
        server_info = info.get('serverInfo') if isinstance(info.get('serverInfo'), Mapping) else {}
        return {
            'server': server_info.get('name') or self.id,
            'version': server_info.get('version') or self.version,
            'tools': [{'name': t.name, 'description': t.description} for t in tools],
        }


def serve_mcp_http(server: MCPServer, host: str = '127.0.0.1', port: int = 0) -> ThreadingHTTPServer:
    """Expose any local :class:`MCPServer` over Streamable HTTP (JSON-RPC 2.0)."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length') or 0)
                message = json.loads(self.rfile.read(length).decode('utf-8'))
            except Exception:
                self.send_error(400, 'Invalid JSON')
                return
            reply = server.handle_mcp(message)
            if reply is None:
                self.send_response(202)
                self.end_headers()
                return
            body = json.dumps(reply, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd
