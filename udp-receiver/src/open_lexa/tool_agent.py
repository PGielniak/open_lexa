"""Route transcripts through Chat Completions tool calling and MCP servers."""
from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

from jsonschema import Draft202012Validator

LOG = logging.getLogger('open_lexa.tool_agent')
SYSTEM = ('Jesteś polskim asystentem głosowym. Użyj dostępnego narzędzia tylko, '
          'gdy wypowiedź jednoznacznie o to prosi. Dla szumu, rozmowy w tle lub '
          'nieobsługiwanej prośby nie wywołuj narzędzia. Nie wymyślaj parametrów.')


def process(text, llm, dispatcher, *, session=None, stem=None):
    """Make one bounded tool-calling request. Never fall back to regex or NLU."""
    if not text or not text.strip():
        LOG.info('agent_result session=%s status=empty', session or '-')
        return {'status': 'empty', 'executions': []}
    if llm is None:
        LOG.warning('agent_result session=%s status=unavailable', session or '-')
        return {'status': 'unavailable', 'error': 'LLM is not configured', 'executions': []}
    discovery_started = time.monotonic()
    tools = dispatcher.list_tools() if dispatcher else []
    LOG.info('mcp_discovery session=%s tools=%d duration_ms=%d', session or '-',
             len(tools), round((time.monotonic() - discovery_started) * 1000))
    by_name = {tool.name: tool for tool in tools}
    llm.stats['calls'] += 1
    try:
        params = {'model': llm.model, 'messages': [
            {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]}
        if tools:
            params.update(tools=[tool.to_openai() for tool in tools], tool_choice='auto',
                          parallel_tool_calls=False)
        started = time.monotonic()
        request_status = 'failed'
        try:
            result = llm.client.chat.completions.create(**params)
            request_status = 'ok'
        finally:
            llm.stats['latency_ms'] = round((time.monotonic() - started) * 1000)
            LOG.info('llm_request session=%s model=%s status=%s duration_ms=%d',
                     session or '-', llm.model, request_status, llm.stats['latency_ms'])
        if not result.choices:
            raise ValueError('LLM returned no choices')
        if getattr(result.choices[0], 'finish_reason', None) in ('length', 'content_filter'):
            raise ValueError('LLM response was truncated or filtered')
        message = result.choices[0].message
        calls = getattr(message, 'tool_calls', None) or []
        if len(calls) > 1:
            raise ValueError('LLM returned more than one tool call')
        if not calls:
            LOG.info('agent_result session=%s status=no_tool', session or '-')
            return {'status': 'no_tool', 'reply': getattr(message, 'content', None), 'executions': []}
        call = calls[0]
        name = call.function.name
        if name not in by_name:
            raise ValueError(f'unknown MCP tool {name!r}')
        arguments = json.loads(call.function.arguments)
        if not isinstance(arguments, dict):
            raise ValueError('tool arguments must be an object')
        Draft202012Validator.check_schema(dict(by_name[name].input_schema))
        Draft202012Validator(dict(by_name[name].input_schema)).validate(arguments)
        decision = SimpleNamespace(tool=name, command=name, arguments=arguments, source='llm')
        execution = dispatcher.dispatch(decision, text=text, session=session, stem=stem)
        if execution is None:
            raise ValueError(f'MCP tool {name!r} is no longer available')
        return {'status': execution['status'], 'executions': [execution]}
    except Exception as exc:
        llm.stats['failures'] += 1
        LOG.exception('agent_result session=%s status=failed', session or '-')
        return {'status': 'failed', 'error': str(exc)[:500], 'executions': []}
