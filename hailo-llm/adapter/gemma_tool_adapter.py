"""Sidecar that gives hailo-ollama's Gemma4 real OpenAI/Ollama ``tool_calls``.

hailo-ollama 5.5.0 injects ``tools`` into Gemma4's chat template but returns the
model's raw tool-call syntax as plain text::

    <|tool_call>call:set_light{room:<|"|>salon<|"|>,state:<|"|>on<|"|>}<tool_call|><|tool_response>

This proxy sits in front of it. Requests without ``tools`` are streamed through
untouched (except OpenAI ``stream: true``, which hailo-ollama rejects). Requests with ``tools`` to ``/v1/chat/completions`` or ``/api/chat``
are sent upstream non-streaming, the reply text is parsed, and the response is
rewritten with a structured ``tool_calls`` list (re-streamed if the client asked
for a stream). Standard library only.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = logging.getLogger('gemma_tool_adapter')
UPSTREAM = os.environ.get('UPSTREAM', '127.0.0.1:8000')
LISTEN_HOST = os.environ.get('ADAPTER_HOST', '0.0.0.0')
LISTEN_PORT = int(os.environ.get('ADAPTER_PORT', '8001'))
TIMEOUT = float(os.environ.get('UPSTREAM_TIMEOUT_SECONDS', '300'))

QUOTE = '<|"|>'
CALL_RE = re.compile(r'<\|tool_call>\s*call:([A-Za-z_][\w.\-]*)\s*\{', re.S)
END_TAG = '<tool_call|>'
TRAILERS = ('<|tool_response>', '<turn|>')


class ParseError(ValueError):
    pass


class _Parser:
    def __init__(self, text: str, pos: int):
        self.s, self.i = text, pos

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in ' \t\r\n':
            self.i += 1

    def value(self):
        self.ws()
        s = self.s
        if s.startswith(QUOTE, self.i):
            end = s.find(QUOTE, self.i + len(QUOTE))
            if end < 0:
                raise ParseError('unterminated string')
            out = s[self.i + len(QUOTE):end]
            self.i = end + len(QUOTE)
            return out
        if self.i < len(s) and s[self.i] == '{':
            self.i += 1
            return self.members('}')
        if self.i < len(s) and s[self.i] == '[':
            self.i += 1
            items = []
            while True:
                self.ws()
                if self.i >= len(s):
                    raise ParseError('unterminated array')
                if s[self.i] == ']':
                    self.i += 1
                    return items
                items.append(self.value())
                self.ws()
                if self.i < len(s) and s[self.i] == ',':
                    self.i += 1
        m = re.compile(r'[^,}\]\s]+').match(s, self.i)
        if not m:
            raise ParseError(f'unexpected input at {self.i}')
        self.i = m.end()
        token = m.group()
        if token in ('true', 'false'):
            return token == 'true'
        if token in ('null', 'None'):
            return None
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            return token

    def members(self, close: str) -> dict:
        s, out = self.s, {}
        while True:
            self.ws()
            if self.i >= len(s):
                raise ParseError('unterminated object')
            if s[self.i] == close:
                self.i += 1
                return out
            m = re.compile(r'[A-Za-z_][\w.\-]*').match(s, self.i)
            if not m:
                raise ParseError(f'bad key at {self.i}')
            self.i = m.end()
            self.ws()
            if self.i >= len(s) or s[self.i] != ':':
                raise ParseError(f'missing ":" after {m.group()!r}')
            self.i += 1
            out[m.group()] = self.value()
            self.ws()
            if self.i < len(s) and s[self.i] == ',':
                self.i += 1


def _strip(text: str) -> str:
    for trailer in TRAILERS:
        text = text.replace(trailer, '')
    return text.strip()


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Return (remaining text, [{'name': str, 'arguments': dict}]).

    Text without a tool call, or with a malformed one, is returned unchanged.
    """
    if not text:
        return text, []
    if '<|tool_call>' not in text:
        return _strip(text), []
    calls, kept, pos = [], [], 0
    for m in CALL_RE.finditer(text):
        if m.start() < pos:
            continue
        try:
            parser = _Parser(text, m.end())
            args = parser.members('}')
        except ParseError as exc:
            LOG.warning('unparseable tool call (%s): %.200r', exc, text)
            return text, []
        end = text.find(END_TAG, parser.i)
        if end >= 0:
            end += len(END_TAG)
        else:
            end = parser.i
        kept.append(text[pos:m.start()])
        calls.append({'name': m.group(1), 'arguments': args})
        pos = end
    kept.append(text[pos:])
    return _strip(''.join(kept)), calls


def rewrite_openai(body: dict) -> bool:
    """Rewrite a chat.completion body in place; True if tool calls were found."""
    changed = False
    for choice in body.get('choices') or []:
        message = choice.get('message') or {}
        rest, calls = parse_tool_calls(message.get('content') or '')
        if not calls:
            if message.get('content'):
                message['content'] = rest
            continue
        message['content'] = rest or None
        message['tool_calls'] = [{
            'id': 'call_' + uuid.uuid4().hex[:24], 'type': 'function',
            'function': {'name': c['name'],
                         'arguments': json.dumps(c['arguments'], ensure_ascii=False)},
        } for c in calls]
        choice['finish_reason'] = 'tool_calls'
        changed = True
    return changed


def rewrite_ollama(body: dict) -> bool:
    message = body.get('message') or {}
    rest, calls = parse_tool_calls(message.get('content') or '')
    if not calls:
        if message.get('content'):
            message['content'] = rest
        return False
    message['content'] = rest
    message['tool_calls'] = [{'function': {'name': c['name'], 'arguments': c['arguments']}}
                             for c in calls]
    body['message'] = message
    return True


def openai_sse(body: dict) -> bytes:
    """Re-stream a finished chat.completion as chat.completion.chunk events."""
    base = {'id': body.get('id', 'chatcmpl-adapter'), 'object': 'chat.completion.chunk',
            'created': body.get('created', int(time.time())), 'model': body.get('model')}
    events = []
    for choice in body.get('choices') or []:
        message = choice.get('message') or {}
        delta = {'role': 'assistant'}
        if message.get('content'):
            delta['content'] = message['content']
        if message.get('tool_calls'):
            delta['tool_calls'] = [dict(call, index=n)
                                   for n, call in enumerate(message['tool_calls'])]
        events.append(dict(base, choices=[{'index': choice.get('index', 0),
                                           'delta': delta, 'finish_reason': None}]))
        events.append(dict(base, choices=[{'index': choice.get('index', 0), 'delta': {},
                                           'finish_reason': choice.get('finish_reason', 'stop')}]))
    out = ''.join(f'data: {json.dumps(e, ensure_ascii=False)}\n\n' for e in events)
    return (out + 'data: [DONE]\n\n').encode()


def flatten_messages(request: dict) -> dict:
    """hailo-ollama wants string contents; OpenAI clients may send part arrays."""
    for message in request.get('messages') or []:
        content = message.get('content')
        if isinstance(content, list):
            message['content'] = ''.join(p.get('text', '') for p in content
                                         if isinstance(p, dict) and p.get('type') == 'text')
    return request


TOOL_PATHS = {'/v1/chat/completions': 'openai', '/api/chat': 'ollama'}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'  # close-delimited bodies keep streaming simple
    server_version = 'gemma-tool-adapter'

    def log_message(self, fmt, *args):  # route access logs through logging
        LOG.info('%s %s', self.address_string(), fmt % args)

    def _read_body(self) -> bytes:
        return self.rfile.read(int(self.headers.get('Content-Length') or 0))

    def _upstream(self, method: str, path: str, body: bytes | None, headers: dict):
        conn = http.client.HTTPConnection(UPSTREAM, timeout=TIMEOUT)
        conn.request(method, path, body=body, headers=headers)
        return conn, conn.getresponse()

    def _forward_headers(self, body: bytes | None) -> dict:
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() in ('content-type', 'accept', 'authorization')}
        if body is not None:
            headers['Content-Length'] = str(len(body))
        return headers

    def _proxy(self, method: str, body: bytes | None):
        try:
            conn, resp = self._upstream(method, self.path, body, self._forward_headers(body))
        except OSError as exc:
            return self._error(502, f'upstream unavailable: {exc}')
        try:
            self.send_response(resp.status)
            for name in ('Content-Type',):
                if resp.getheader(name):
                    self.send_header(name, resp.getheader(name))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            while True:
                chunk = resp.read1(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    def _error(self, status: int, message: str):
        data = json.dumps({'error': message}).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: bytes, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle_tools(self, kind: str, request: dict, raw: bytes):
        wants_stream = bool(request.get('stream', kind == 'ollama'))
        upstream_request = flatten_messages(dict(request, stream=False))
        body = json.dumps(upstream_request).encode()
        started = time.monotonic()
        try:
            conn, resp = self._upstream('POST', self.path, body, {
                'Content-Type': 'application/json', 'Content-Length': str(len(body))})
        except OSError as exc:
            return self._error(502, f'upstream unavailable: {exc}')
        try:
            payload = resp.read()
            status = resp.status
        finally:
            conn.close()
        try:
            data = json.loads(payload)
        except ValueError:
            return self._send_json(status, payload)
        changed = status == 200 and (rewrite_openai(data) if kind == 'openai'
                                     else rewrite_ollama(data))
        LOG.info('tools_request path=%s tools=%d stream=%s tool_calls=%s status=%d duration_ms=%d',
                 self.path, len(request.get('tools') or []), wants_stream, changed, status,
                 round((time.monotonic() - started) * 1000))
        if not wants_stream or status != 200:
            return self._send_json(status, json.dumps(data, ensure_ascii=False).encode())
        if kind == 'openai':
            return self._send_json(200, openai_sse(data), 'text/event-stream')
        data['done'] = True
        return self._send_json(200, (json.dumps(data, ensure_ascii=False) + '\n').encode(),
                               'application/x-ndjson')

    def do_POST(self):
        raw = self._read_body()
        path = self.path.split('?', 1)[0]
        kind = TOOL_PATHS.get(path)
        if kind:
            try:
                request = json.loads(raw)
            except ValueError:
                request = None
            # hailo-ollama's /v1 endpoint rejects stream=true, so OpenAI streaming
            # requests are served from a buffered reply and re-streamed as SSE.
            if isinstance(request, dict) and (
                    request.get('tools') or (kind == 'openai' and request.get('stream'))):
                return self._handle_tools(kind, request, raw)
        self._proxy('POST', raw)

    def do_GET(self):
        self._proxy('GET', None)

    def do_DELETE(self):
        self._proxy('DELETE', self._read_body())


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    server.daemon_threads = True
    LOG.info('listening on %s:%d -> %s', LISTEN_HOST, LISTEN_PORT, UPSTREAM)
    server.serve_forever()


if __name__ == '__main__':
    main()
