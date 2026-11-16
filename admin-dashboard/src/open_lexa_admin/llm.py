"""LLM connection settings published to the receiver."""
from __future__ import annotations

LLM_DEFAULTS = {'url': 'https://api.openai.com/v1', 'model': 'gpt-4o-mini',
                'api_key': '', 'timeout': '60', 'receiver_mode': 'off'}
RECEIVER_MODES = ('off', 'fallback', 'always')


class LLMConfigError(ValueError):
    pass


def validate_llm(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise LLMConfigError('settings must be a JSON object')
    settings = {key: str(raw.get(key) if raw.get(key) is not None else default).strip()
                for key, default in LLM_DEFAULTS.items()}
    if not settings['url'].startswith(('http://', 'https://')):
        raise LLMConfigError('LLM URL musi zaczynać się od http:// lub https://')
    settings['url'] = settings['url'].rstrip('/')
    if not settings['model']:
        raise LLMConfigError('model LLM jest wymagany')
    settings['receiver_mode'] = settings['receiver_mode'] or 'off'
    if settings['receiver_mode'] not in RECEIVER_MODES:
        raise LLMConfigError('niepoprawny tryb odbiornika LLM')
    try:
        settings['timeout'] = str(int(max(5.0, min(600.0, float(settings['timeout'] or 60)))))
    except ValueError as exc:
        raise LLMConfigError('timeout musi być liczbą') from exc
    return settings


def publish_payload(settings: dict) -> dict:
    settings = validate_llm(settings)
    payload = {'mode': settings['receiver_mode'], 'url': settings['url'],
               'model': settings['model'], 'timeout': settings['timeout']}
    if settings['receiver_mode'] != 'off':
        payload['api_key'] = settings['api_key']
    return payload


class ConnectionTest:
    def __init__(self, settings: dict, client=None):
        self.settings = validate_llm(settings)
        if not self.settings['api_key']:
            raise LLMConfigError('brak klucza API LLM')
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=self.settings['api_key'], base_url=self.settings['url'],
                            timeout=float(self.settings['timeout']), max_retries=1)
        self.client = client

    def test(self) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.settings['model'],
                messages=[{'role': 'user', 'content': 'Respond with OK.'}], max_tokens=16)
            if not response.choices:
                raise LLMConfigError('model nie zwrócił odpowiedzi')
            return f'OK: model {self.settings["model"]} odpowiada'
        finally:
            close = getattr(self.client, 'close', None)
            if close:
                close()
