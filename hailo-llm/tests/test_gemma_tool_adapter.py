import importlib.util
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'gemma_tool_adapter', Path(__file__).parents[1] / 'adapter' / 'gemma_tool_adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

RAW = ('<|tool_call>call:set_light{brightness:40,room:<|"|>kuchnia<|"|>,'
       'state:<|"|>on<|"|>}<tool_call|><|tool_response>')


def test_parse_simple_call():
    rest, calls = adapter.parse_tool_calls(RAW)
    assert rest == ''
    assert calls == [{'name': 'set_light',
                      'arguments': {'brightness': 40, 'room': 'kuchnia', 'state': 'on'}}]


def test_parse_types_arrays_and_nesting():
    text = ('<|tool_call>call:set_timer{labels:[<|"|>herbata<|"|>,<|"|>jajka, 2<|"|>], '
            'loud:true, minutes:2.5, opts:{a:null,b:<|"|>x}y<|"|>}}<tool_call|>')
    _, calls = adapter.parse_tool_calls(text)
    assert calls[0]['arguments'] == {'labels': ['herbata', 'jajka, 2'], 'loud': True,
                                     'minutes': 2.5, 'opts': {'a': None, 'b': 'x}y'}}


def test_parse_keeps_text_and_multiple_calls():
    text = 'Dobrze. <|tool_call>call:a{}<tool_call|><|tool_call>call:b{x:1}<tool_call|>'
    rest, calls = adapter.parse_tool_calls(text)
    assert rest == 'Dobrze.'
    assert [c['name'] for c in calls] == ['a', 'b']


def test_strips_turn_token_and_flattens_parts():
    assert adapter.parse_tool_calls('Cześć<turn|>') == ('Cześć', [])
    req = adapter.flatten_messages({'messages': [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}]})
    assert req['messages'][0]['content'] == 'ab'


def test_plain_and_malformed_text_unchanged():
    assert adapter.parse_tool_calls('Stolicą Francji jest Paryż.') == ('Stolicą Francji jest Paryż.', [])
    bad = '<|tool_call>call:x{room:<|"|>oops}<tool_call|>'
    assert adapter.parse_tool_calls(bad) == (bad, [])


class FakeUpstream(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"models":[]}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeUpstream.seen.append(body)
        content = RAW if body.get('tools') else 'hej'
        if self.path == '/api/chat':
            out = {'model': 'm', 'message': {'role': 'assistant', 'content': content}, 'done': True}
        else:
            out = {'id': 'x', 'model': 'm', 'choices': [
                {'index': 0, 'message': {'role': 'assistant', 'content': content},
                 'finish_reason': 'stop'}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _serve(handler):
    srv = adapter.ThreadingHTTPServer(('127.0.0.1', 0), handler) if handler is adapter.Handler \
        else HTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _post(srv, path, payload):
    req = urllib.request.Request(f'http://127.0.0.1:{srv.server_port}{path}',
                                 json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.headers['Content-Type'], r.read().decode()


def test_end_to_end(monkeypatch):
    up = _serve(FakeUpstream)
    monkeypatch.setattr(adapter, 'UPSTREAM', f'127.0.0.1:{up.server_port}')
    srv = _serve(adapter.Handler)
    tools = [{'type': 'function', 'function': {'name': 'set_light', 'parameters': {}}}]
    msgs = [{'role': 'user', 'content': 'x'}]
    try:
        ctype, body = _post(srv, '/v1/chat/completions', {'model': 'm', 'messages': msgs, 'tools': tools})
        choice = json.loads(body)['choices'][0]
        assert choice['finish_reason'] == 'tool_calls' and choice['message']['content'] is None
        call = choice['message']['tool_calls'][0]
        assert call['function']['name'] == 'set_light'
        assert json.loads(call['function']['arguments'])['room'] == 'kuchnia'

        ctype, body = _post(srv, '/v1/chat/completions',
                            {'model': 'm', 'messages': msgs, 'tools': tools, 'stream': True})
        assert ctype == 'text/event-stream' and 'tool_calls' in body and body.endswith('data: [DONE]\n\n')
        assert FakeUpstream.seen[-1]['stream'] is False

        _, body = _post(srv, '/api/chat', {'model': 'm', 'messages': msgs, 'tools': tools, 'stream': False})
        message = json.loads(body)['message']
        assert message['tool_calls'][0]['function']['arguments']['state'] == 'on'

        _, body = _post(srv, '/v1/chat/completions', {'model': 'm', 'messages': msgs})
        assert json.loads(body)['choices'][0]['message']['content'] == 'hej'
        ctype, body = _post(srv, '/v1/chat/completions', {'model': 'm', 'messages': msgs, 'stream': True})
        assert ctype == 'text/event-stream' and '"content": "hej"' in body
        assert FakeUpstream.seen[-1]['stream'] is False
        with urllib.request.urlopen(f'http://127.0.0.1:{srv.server_port}/api/tags') as r:
            assert json.load(r) == {'models': []}
    finally:
        srv.shutdown()
        up.shutdown()
