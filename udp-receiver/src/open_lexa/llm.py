"""Configuration and client lifecycle for LLM tool calling."""
from __future__ import annotations

import os
from typing import Mapping

SETTINGS_PATH = '_config/llm.json'
# Accept legacy fallback settings, but all enabled modes call the LLM directly.
MODES = ('off', 'fallback', 'always')
DEFAULTS = {'mode': '', 'url': 'https://api.openai.com/v1', 'model': '',
            'api_key': '', 'timeout': '15'}


class LLMConfigError(ValueError):
    """Invalid LLM settings."""


def settings_from_env(env: Mapping[str, str] | None = None, mode_override: str | None = None) -> dict:
    env = os.environ if env is None else env
    return {'mode': (mode_override or env.get('LLM_INTENT_MODE') or '').strip().lower(),
            'url': (env.get('LLM_URL') or DEFAULTS['url']).strip(),
            'model': (env.get('LLM_MODEL') or '').strip(),
            'api_key': (env.get('LLM_API_KEY') or '').strip(),
            'timeout': (env.get('LLM_TIMEOUT') or '15').strip()}


def validate_settings(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        raise LLMConfigError('LLM settings must be a JSON object')
    settings = {key: str(raw.get(key) if raw.get(key) is not None else default).strip()
                for key, default in DEFAULTS.items()}
    if not settings['mode']:
        settings['mode'] = 'always' if settings['api_key'] and settings['model'] else 'off'
    if settings['mode'] not in MODES:
        raise LLMConfigError(f'LLM_INTENT_MODE must be one of: {", ".join(MODES)}')
    if settings['mode'] == 'off':
        return None
    for key in ('model', 'api_key'):
        if not settings[key]:
            raise LLMConfigError(f'LLM_{key.upper()} is required when LLM is on')
    if not settings['url'].startswith(('http://', 'https://')):
        raise LLMConfigError('LLM_URL must start with http:// or https://')
    settings['url'] = settings['url'].rstrip('/').removesuffix('/chat/completions')
    try:
        settings['timeout'] = str(int(max(2.0, min(120.0, float(settings['timeout'] or '15')))))
    except ValueError as exc:
        raise LLMConfigError('LLM_TIMEOUT must be a number') from exc
    return settings


class ToolLLM:
    def __init__(self, settings: dict, client=None):
        settings = validate_settings(settings)
        if settings is None:
            raise LLMConfigError('LLM is disabled')
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=settings['api_key'], base_url=settings['url'],
                            timeout=float(settings['timeout']), max_retries=1)
        self.settings, self.client = settings, client
        self.mode, self.model = settings['mode'], settings['model']
        self.stats = {'mode': self.mode, 'model': self.model, 'enabled': True,
                      'calls': 0, 'failures': 0, 'latency_ms': None}

    def close(self):
        close = getattr(self.client, 'close', None)
        if close is not None:
            close()


def create_llm(env: Mapping[str, str] | None = None, *, mode: str | None = None, client=None):
    settings = validate_settings(settings_from_env(env, mode))
    return ToolLLM(settings, client=client) if settings is not None else None


def create_llm_from_file(data: dict | None, env: Mapping[str, str] | None = None, *, client=None):
    if data is None:
        return create_llm(env, client=client)
    settings = validate_settings(data)
    return ToolLLM(settings, client=client) if settings is not None else None
