"""LLM intent/entity recognition right after transcription.

When configured, the receiver sends every transcript to an OpenAI-compatible
Chat Completions endpoint so the model picks the intent and extracts the
entities immediately after STT. Unlike the admin annotation flow
(:mod:`open_lexa_admin.llm`), the receiver never invents new intents or entity
types: the prompt embeds the published lexicon and the answer is validated
against it. Every failure (network, JSON, unknown label) returns ``None`` and
the lexicon/rules decision stands, so an unreachable model never breaks the
voice pipeline.

Configuration via ``.env`` (or :option:`--llm-mode` on the command line)::

    LLM_INTENT_MODE=always|fallback|off   # default: always when key+model set
    LLM_MODEL=gpt-4o-mini
    LLM_API_KEY=
    LLM_URL=https://api.openai.com/v1     # any OpenAI-compatible /v1
    LLM_TIMEOUT=15                        # seconds, clamped to 2..120

The admin UI publishes the same settings to ``_config/llm.json`` on the
archive; when that file exists the receiver prefers it over its environment,
so the LLM API key only has to be entered once (Ustawienia → LLM).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Mapping

from .commands import RULES, Decision
from .nlu import Lexicon, _extract_number

LOG = logging.getLogger('open_lexa.llm')

PROMPT_VERSION = 'receiver-1'

MODES = ('off', 'fallback', 'always')

# Settings published by the admin (single source for the LLM key and model).
SETTINGS_PATH = '_config/llm.json'

# Recognised but never executed or answered, matching the admin's seeded intents.
NON_ACTIONABLE = ('noise', 'not_for_me')

DEFAULTS = {
    'mode': '',
    'url': 'https://api.openai.com/v1',
    'model': '',
    'api_key': '',
    'timeout': '15',
}

MAX_PHRASES = 6
MAX_VALUES = 40

SYSTEM_PROMPT = (
    'Jesteś modułem NLU polskiego asystenta głosowego Open Lexa. Dostajesz transkrypcję '
    'polecenia głosowego (polski, możliwa zniekształcona przez STT) oraz słownik intencji '
    'i encji. Wybierz intencję i encje WYŁĄCZNIE ze słownika.\n'
    'Zasady:\n'
    '- Pole "intent" musi być dokładną nazwą jednej intencji z listy "intents".\n'
    '- NIE wymyślaj nowych intencji ani typów encji. Jeśli żadna intencja nie pasuje, '
    'zwróć "unknown".\n'
    '- Szum, urwane lub niezrozumiałe nagranie → "noise". Rozmowa w tle / fałszywe '
    'wybudzenie → "not_for_me".\n'
    '- Encje tylko z listy "entities" wybranej intencji, w polu "entities" jako mapa '
    'typ -> wartość. Dla typu "enum" podaj kanoniczną wartość ze słownika, dla "number" '
    'liczbę, dla "text" krótką frazę z transkrypcji.\n'
    '- "confidence" to pewność rozpoznania od 0 do 1.\n'
    '- Odpowiedz WYŁĄCZNIE poprawnym JSON-em, bez komentarzy, w formacie:\n'
    '{"intent": "<nazwa lub unknown>", "entities": {"<typ>": "<wartość>"}, '
    '"confidence": 0.0}'
)


class LLMConfigError(ValueError):
    """Invalid LLM settings."""


def settings_from_env(env: Mapping[str, str] | None = None, mode_override: str | None = None) -> dict:
    """Collect raw LLM settings from the environment; ``mode_override`` wins."""
    env = os.environ if env is None else env
    return {
        'mode': (mode_override or env.get('LLM_INTENT_MODE') or '').strip().lower(),
        'url': (env.get('LLM_URL') or DEFAULTS['url']).strip(),
        'model': (env.get('LLM_MODEL') or DEFAULTS['model']).strip(),
        'api_key': (env.get('LLM_API_KEY') or '').strip(),
        'timeout': (env.get('LLM_TIMEOUT') or DEFAULTS['timeout']).strip(),
    }


def validate_settings(raw: dict) -> dict | None:
    """Normalise settings or raise :class:`LLMConfigError`; None disables the LLM.

    An empty mode means: ``always`` when both key and model are configured
    (the user asked for LLM recognition on every transcript), otherwise ``off``.
    """
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
    if not settings['model']:
        raise LLMConfigError('LLM_MODEL is required when the LLM NLU is on')
    if not settings['api_key']:
        raise LLMConfigError('LLM_API_KEY is required when the LLM NLU is on')
    if not settings['url'].startswith(('http://', 'https://')):
        raise LLMConfigError('LLM_URL must start with http:// or https://')
    settings['url'] = settings['url'].rstrip('/').removesuffix('/chat/completions')
    try:
        timeout = max(2.0, min(120.0, float(settings['timeout'] or '15')))
    except ValueError as exc:
        raise LLMConfigError('LLM_TIMEOUT must be a number') from exc
    settings['timeout'] = str(int(timeout))
    return settings


def build_dictionary(lexicon: Lexicon | None, *, max_phrases: int = MAX_PHRASES,
                     max_values: int = MAX_VALUES) -> dict:
    """Compact intent/entity dictionary for the prompt.

    With a lexicon the model may only use its labels; without one (no SMB
    archive) the regex-rule commands are offered so the LLM path still works
    standalone. ``noise``/``not_for_me`` are always available.
    """
    if lexicon is None or lexicon.empty:
        intents = [{'name': name, 'actionable': True, 'entities': [], 'phrases': []}
                   for name in sorted({command for command, _ in RULES})]
        entity_types: list[dict] = []
    else:
        intents = [{'name': name,
                    'actionable': bool(info.get('actionable', True)),
                    'entities': list(info.get('entities') or []),
                    'phrases': list(lexicon.phrases.get(name, ()))[:max_phrases]}
                   for name, info in sorted(lexicon.intents.items())]
        entity_types = [{'name': name, 'kind': spec.get('kind', 'text'),
                         'values': sorted(spec.get('values') or {})[:max_values]}
                        for name, spec in sorted(lexicon.entities.items())]
    known = {intent['name'] for intent in intents}
    for name in NON_ACTIONABLE:
        if name not in known:
            intents.append({'name': name, 'actionable': False, 'entities': [], 'phrases': []})
    return {'intents': intents, 'entity_types': entity_types}


def _clean_entities(answer: dict, allowed: list[str], kinds: dict[str, str]) -> dict[str, str]:
    """Keep only the intent's declared entity types; numbers become digits."""
    raw = answer.get('entities')
    pairs: list[tuple[str, str]] = []
    if isinstance(raw, dict):
        pairs = [(str(key), str(value)) for key, value in raw.items()]
    elif isinstance(raw, list):  # tolerate the admin's list-of-objects shape
        for item in raw:
            if isinstance(item, dict) and item.get('type'):
                pairs.append((str(item['type']), str(item.get('value') or '')))
    result: dict[str, str] = {}
    for name, value in pairs:
        key = re.sub(r'[^a-z0-9_]', '_', name.strip().lower())
        value = value.strip()[:200]
        if not key or not value or key not in allowed:
            continue
        if kinds.get(key) == 'number':
            value = _extract_number(value) or ''
            if not value:
                continue
        result[key] = value
    return result


def parse_answer(content: str, dictionary: dict) -> Decision | None:
    """Validate one model answer against the dictionary; None keeps the base decision."""
    text = (content or '').strip()
    fence = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        answer = json.loads(text)
    except ValueError as exc:
        LOG.warning('LLM returned invalid JSON: %s', exc)
        return None
    if not isinstance(answer, dict):
        LOG.warning('LLM answer is not a JSON object')
        return None
    intent = str(answer.get('intent') or '').strip().lower()
    info = next((item for item in dictionary['intents'] if item['name'] == intent), None)
    if info is None:
        LOG.info('LLM intent %r is not in the dictionary; ignoring', intent)
        return None
    kinds = {spec['name']: spec['kind'] for spec in dictionary['entity_types']}
    entities = _clean_entities(answer, info['entities'], kinds)
    score: float | None = None
    try:
        confidence = float(answer.get('confidence'))
        score = round(max(0.0, min(1.0, confidence)) * 100, 1)
    except (TypeError, ValueError):
        pass
    if not info['actionable'] or intent in NON_ACTIONABLE:
        return Decision(None, '', intent=intent, entities=(), score=score, source='llm')
    return Decision(intent, intent, intent=intent, entities=tuple(sorted(entities.items())),
                    score=score, source='llm')


class IntentLLM:
    """Chat-completions client that only ever returns dictionary-valid decisions."""

    def __init__(self, settings: dict, client=None):
        settings = validate_settings(settings)
        if settings is None:
            raise LLMConfigError('LLM NLU is disabled')
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=settings['api_key'], base_url=settings['url'],
                            timeout=float(settings['timeout']), max_retries=1)
        self.settings = settings
        self.mode = settings['mode']
        self.model = settings['model']
        self.client = client
        # Shared with HealthState so /healthz sees the live counters.
        self.stats: dict = {'mode': self.mode, 'model': self.model, 'enabled': True,
                            'calls': 0, 'failures': 0, 'latency_ms': None}
        self._dictionary_cache: tuple[object, dict] | None = None

    def dictionary(self, lexicon: Lexicon | None) -> dict:
        """Prompt dictionary for the current lexicon (rebuilt only when it changes)."""
        key = lexicon.version if lexicon is not None and not lexicon.empty else None
        if self._dictionary_cache is None or self._dictionary_cache[0] != key:
            self._dictionary_cache = (key, build_dictionary(lexicon))
        return self._dictionary_cache[1]

    def recognize(self, text: str, lexicon: Lexicon | None) -> Decision | None:
        """One transcript -> a validated Decision, or None on any failure."""
        dictionary = self.dictionary(lexicon)
        self.stats['calls'] += 1
        started = time.monotonic()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={'type': 'json_object'},
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': json.dumps(
                        {'dictionary': dictionary, 'transcript': text}, ensure_ascii=False)},
                ])
            content = completion.choices[0].message.content if completion.choices else None
        except Exception as exc:
            self.stats['failures'] += 1
            LOG.warning('LLM request failed: %s', exc)
            return None
        self.stats['latency_ms'] = round((time.monotonic() - started) * 1000)
        decision = parse_answer(content or '', dictionary)
        if decision is None:
            self.stats['failures'] += 1
        return decision

    def close(self) -> None:
        close = getattr(self.client, 'close', None)
        if close is not None:
            close()


def create_llm(env: Mapping[str, str] | None = None, *, mode: str | None = None,
               client=None) -> IntentLLM | None:
    """Build the LLM client from the environment; None when the feature is off."""
    settings = validate_settings(settings_from_env(env, mode))
    if settings is None:
        return None
    return IntentLLM(settings, client=client)


def create_llm_from_file(data: dict | None, env: Mapping[str, str] | None = None,
                         *, client=None) -> IntentLLM | None:
    """Client for the admin-published ``_config/llm.json``; the file wins over env.

    ``data`` is the parsed file (None when it is absent → environment fallback)
    and uses the same keys as :data:`DEFAULTS`. Raises :class:`LLMConfigError`
    for an invalid file, so the caller can keep the running client, exactly
    like the STT settings watcher.
    """
    if data is None:
        return create_llm(env, client=client)
    if not isinstance(data, dict):
        raise LLMConfigError(f'{SETTINGS_PATH} must contain a JSON object')
    settings = validate_settings(data)
    if settings is None:
        return None
    return IntentLLM(settings, client=client)
