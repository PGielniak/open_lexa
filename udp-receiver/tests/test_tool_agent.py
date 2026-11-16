import json
from types import SimpleNamespace as NS

from open_lexa.handlers import Dispatcher
from open_lexa.handlers.base import MCPServer, MCPTool, HandlerResult, RemoteMCPServer
from open_lexa.tool_agent import process


class Server(MCPServer):
    id = 'test'
    tools = (MCPTool('play', 'Włącz Spotify', {
        'type': 'object', 'properties': {'level': {'type': 'integer'}},
        'additionalProperties': False}),)

    def __init__(self):
        super().__init__()
        self.calls = []

    def handle(self, request):
        self.calls.append(request.arguments)
        return HandlerResult.handled('OK')


def llm(name='play', arguments='{}'):
    call = NS(function=NS(name=name, arguments=arguments))
    client = NS(chat=NS(completions=NS(create=lambda **kwargs: NS(
        choices=[NS(message=NS(tool_calls=[call], content=None))]))))
    return NS(model='test', client=client, stats={'calls': 0, 'failures': 0})


def test_tool_selection_and_schema_validation():
    server = Server()
    dispatcher = Dispatcher([server])
    dispatcher.apply_config({'test': {'enabled': True}})
    result = process('włącz spotify', llm(arguments='{"level": 5}'), dispatcher)
    assert result['executions'][0]['tool'] == 'play'
    assert server.calls == [{'level': 5}]
    for name, arguments in [('play', '{"level":"bad"}'), ('unknown', '{}'), ('play', 'bad')]:
        assert process('włącz', llm(name, arguments), dispatcher)['status'] == 'failed'
    assert server.calls == [{'level': 5}]


def test_remote_mcp_protocol_discovery_and_call():
    local = Server()
    local.configure({'enabled': True})
    methods = []

    def transport(url, message, headers, timeout):
        methods.append(message['method'])
        return local.handle_mcp(message), {'mcp-session-id': 'test-session'}

    remote = RemoteMCPServer('remote', transport=transport)
    remote.configure({'enabled': True, 'url': 'http://localhost/mcp'})
    assert remote.list_tools()[0].description == 'Włącz Spotify'
    assert remote.call_tool('play', {'level': 3}).status == 'handled'
    assert local.calls == [{'level': 3}]
    assert methods == ['initialize', 'notifications/initialized', 'tools/list', 'tools/call']


def test_worker_archives_tool_result_without_nlu(caplog):
    import logging
    from test_commands_archive import MemoryArchive, run_worker
    from test_server import FakeTranscriber

    server = Server()
    dispatcher = Dispatcher([server])
    dispatcher.apply_config({'test': {'enabled': True}})
    archive = MemoryArchive()
    model = llm()
    caplog.set_level(logging.INFO)
    state = run_worker(FakeTranscriber(['włącz spotify']), archive=archive,
                       dispatcher=dispatcher, llm_client=model)
    assert state.archived == 1
    assert archive.saved[0][2]['agent']['executions'][0]['tool'] == 'play'
    assert server.calls == [{}]
    assert 'transcript session=00000001 backend=fake role=- text="włącz spotify"' in caplog.text
    assert 'stt_total session=00000001' in caplog.text
    assert 'stt_backend session=00000001' in caplog.text
    assert 'llm_request session=00000001 model=test status=ok duration_ms=' in caplog.text
    assert 'mcp_call session=00000001 server=test tool=play arguments={}' in caplog.text
    assert 'mcp_result session=00000001 server=test tool=play status=handled duration_ms=' in caplog.text
    assert 'agent_total session=00000001 duration_ms=' in caplog.text
    assert 'processing_total session=00000001 status=handled duration_ms=' in caplog.text


def test_tool_arguments_are_logged_but_secrets_are_masked(caplog):
    import logging
    server = Server()
    server.tools = (MCPTool('play', 'test', {'type': 'object'}),)
    dispatcher = Dispatcher([server])
    dispatcher.apply_config({'test': {'enabled': True}})
    caplog.set_level(logging.INFO)
    result = process('test', llm(arguments='{"level":5,"nested":{"api_key":"private",'
                                           '"list":[{"password":"hidden"}]}}'),
                     dispatcher, session='abc')
    assert result['status'] == 'handled'
    assert '"level":5' in caplog.text and '[REDACTED]' in caplog.text
    assert 'private' not in caplog.text and 'hidden' not in caplog.text


def test_llm_no_tool_and_failure_are_logged(caplog):
    import logging
    caplog.set_level(logging.INFO)
    model = llm()
    model.client.chat.completions.create = lambda **_: NS(choices=[NS(message=NS(tool_calls=[], content=''))])
    assert process('test', model, None, session='abc')['status'] == 'no_tool'
    assert 'agent_result session=abc status=no_tool' in caplog.text
    def fail(**_):
        raise RuntimeError('network error')
    model.client.chat.completions.create = fail
    assert process('test', model, None, session='def')['status'] == 'failed'
    assert 'llm_request session=def model=test status=failed duration_ms=' in caplog.text
    record = next(r for r in caplog.records if 'agent_result session=def status=failed' in r.message)
    assert record.exc_info is not None and 'RuntimeError: network error' in caplog.text


def test_remote_exception_has_traceback(caplog):
    import logging
    from open_lexa.handlers.base import CommandRequest

    remote = RemoteMCPServer('remote')
    remote.enabled = True
    remote.initialize = lambda: (_ for _ in ()).throw(ConnectionError('MCP unreachable'))
    caplog.set_level(logging.ERROR)
    result = remote.handle(CommandRequest('play', session='abc'))
    assert result.status == 'failed' and result.error == 'MCP unreachable'
    record = next(r for r in caplog.records if 'mcp_remote_error session=abc' in r.message)
    assert record.exc_info is not None and 'ConnectionError: MCP unreachable' in caplog.text


def test_worker_reloads_llm_settings_and_keeps_previous_on_invalid(monkeypatch):
    import queue
    import threading
    from open_lexa import server as receiver
    from open_lexa.protocol import Recording
    from test_server import FakeTranscriber

    first = llm()
    first.close = lambda: None
    replacement = llm()
    replacement.close = lambda: None
    replacement.mode = 'always'
    watcher = NS(version=1, data={'model': 'configured'})
    monkeypatch.setattr(receiver, 'create_llm_from_file', lambda _: replacement)
    jobs = queue.Queue()
    state = receiver.HealthState('fake')
    dispatcher = Dispatcher([Server()])
    dispatcher.apply_config({'test': {'enabled': True}})
    thread = threading.Thread(target=receiver.worker,
                              args=(jobs, FakeTranscriber(['one', 'two']), 'pl', None, state),
                              kwargs={'llm_client': first, 'llm_config': watcher,
                                      'dispatcher': dispatcher})
    thread.start()
    jobs.put(Recording(('127.0.0.1', 1), 1, bytes(640)))
    jobs.join()
    assert replacement.stats['calls'] == 1

    def bad(_):
        raise ValueError('invalid config')

    monkeypatch.setattr(receiver, 'create_llm_from_file', bad)
    watcher.version = 2
    jobs.put(Recording(('127.0.0.1', 1), 2, bytes(640)))
    jobs.join()
    jobs.put(None)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert replacement.stats['calls'] == 2
