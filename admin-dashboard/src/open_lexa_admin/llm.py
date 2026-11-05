"""LLM auto-annotation: assign intents and entities to unlabeled recordings.

Uses any OpenAI-compatible Chat Completions API (extra key, configured in the
admin portal under Ustawienia → LLM). The prompt embeds the current
dictionary (intents with phrases, entity types with values, recent labeled
examples) so the model can reuse existing labels or propose new ones; the
answer is validated and applied through :mod:`open_lexa_admin.db`, which
creates missing intents/entity types and stores provenance in
``recordings.annotation``.
"""
from __future__ import annotations

import json
import logging
import re

from . import db

LOG = logging.getLogger('open_lexa.admin')

PROMPT_VERSION = '1'

LLM_DEFAULTS = {
    'url': 'https://api.openai.com/v1',
    'model': 'gpt-4o-mini',
    'api_key': '',
    'auto_label': '',        # checkbox: annotate new recordings in the background
    'batch_size': '5',
    'timeout': '60',
}

SYSTEM_PROMPT = (
    'Jesteś systemem NLU polskiego asystenta głosowego Open Lexa. Dostajesz transkrypcję '
    'polecenia głosowego (polski, możliwa zniekształcona przez STT) oraz słownik intencji, '
    'typów encji i przykładów. Wybierz najlepiej pasującą intencję i wypisz encje.\n'
    'Zasady:\n'
    '- Preferuj intencje ze słownika. Jeśli żadna nie pasuje, a transkrypcja jest sensownym '
    'poleceniem, zaproponuj NOWĄ intencję: nazwa snake_case po angielsku (np. play_podcast), '
    'krótki opis i 2-4 przykładowe frazy po polsku z {slot}.\n'
    '- Szum, urwane lub niezrozumiałe nagranie → "noise". Rozmowa w tle / fałszywe wybudzenie '
    '→ "not_for_me".\n'
    '- Encje: używaj typów ze słownika; wartość kanoniczna (np. nazwa własna), a "fragment" to '
    'dokładny fragment transkrypcji, który jej odpowiada (może być zniekształcony przez STT).\n'
    '- Nowe typy encji tylko gdy naprawdę potrzebne (kind: text, enum lub number).\n'
    '- Odpowiedz WYŁĄCZNIE poprawnym JSON-em, bez komentarzy, w formacie:\n'
    '{"intent": "<nazwa>", "new_intent": {"name": "<snake_case>", "description": "<opis>", '
    '"phrases": ["..."]} | null, "entities": [{"type": "<typ>", "value": "<wartość>", '
    '"fragment": "<fragment transkrypcji>"}], "new_entity_types": [{"name": "<typ>", '
    '"kind": "text|enum|number"}], "notes": "<krótka uwaga lub pusty string>"}'
)


class LLMConfigError(ValueError):
    """Invalid LLM settings."""


class LLMError(RuntimeError):
    """The LLM call or its answer could not be used."""


def validate_llm(raw: dict) -> dict:
    """Normalise the settings form into stored values; raises LLMConfigError."""
    if not isinstance(raw, dict):
        raise LLMConfigError('settings must be a JSON object')
    settings = {key: str(raw.get(key) if raw.get(key) is not None else default).strip()
                for key, default in LLM_DEFAULTS.items()}
    if not settings['url'].startswith(('http://', 'https://')):
        raise LLMConfigError('LLM URL musi zaczynać się od http:// lub https://')
    settings['url'] = settings['url'].rstrip('/')
    if not settings['model']:
        raise LLMConfigError('model LLM jest wymagany')
    settings['auto_label'] = '1' if settings['auto_label'] in ('1', 'true', 'on', 'yes') else ''
    try:
        batch = max(1, min(50, int(settings['batch_size'] or 5)))
        timeout = max(5.0, min(600.0, float(settings['timeout'] or 60)))
    except ValueError as exc:
        raise LLMConfigError('batch_size i timeout muszą być liczbami') from exc
    settings['batch_size'] = str(batch)
    settings['timeout'] = str(int(timeout))
    return settings


def llm_enabled(settings: dict | None) -> bool:
    return bool(settings and settings.get('api_key') and settings.get('model'))


def dictionary_snapshot(conn, *, max_values: int = 60, max_examples: int = 40) -> dict:
    """Compact dictionary for the prompt: intents, entity types, examples."""
    allowed = db.intent_entity_map(conn)
    phrase_map: dict[str, list[str]] = {}
    for row in db.phrases(conn):
        phrase_map.setdefault(row['intent'], []).append(row['phrase'])
    intents = [{'name': row['name'], 'description': row['description'] or '',
                'actionable': bool(row['is_actionable']),
                'entities': allowed.get(row['name'], []),
                'phrases': phrase_map.get(row['name'], [])[:8]}
               for row in db.intents(conn)]
    values_by_type: dict[str, list[str]] = {}
    for value in db.dictionary(conn):
        values_by_type.setdefault(value['type'], []).append(value['value'])
    entity_types = [{'name': row['name'], 'kind': row['kind'], 'description': row['description'] or '',
                     'values': values_by_type.get(row['name'], [])[:max_values]}
                    for row in db.entity_types(conn)]
    examples = []
    for row in conn.execute(
            "SELECT r.original_transcript, r.corrected_transcript, i.name AS intent "
            "FROM recordings r JOIN intents i ON i.id = r.intent_id "
            "WHERE r.status = 'labeled' AND r.original_transcript IS NOT NULL "
            "ORDER BY r.recorded_at DESC LIMIT ?", (max_examples,)):
        examples.append({'text': row['original_transcript'], 'intent': row['intent']})
    return {'intents': intents, 'entity_types': entity_types, 'examples': examples}


def parse_answer(content: str) -> dict:
    """Parse and validate the model's JSON answer; raises LLMError."""
    text = (content or '').strip()
    fence = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        answer = json.loads(text)
    except ValueError as exc:
        raise LLMError(f'model zwrócił niepoprawny JSON: {exc}') from exc
    if not isinstance(answer, dict):
        raise LLMError('odpowiedź modelu nie jest obiektem JSON')
    intent = str(answer.get('intent') or '').strip()
    if not db.INTENT_NAME.match(intent):
        raise LLMError(f'niepoprawna nazwa intencji: {intent!r}')
    new_intent = answer.get('new_intent')
    if isinstance(new_intent, dict) and str(new_intent.get('name') or '').strip() == intent:
        answer['new_intent'] = {'name': intent,
                                'description': str(new_intent.get('description') or ''),
                                'phrases': [str(p) for p in (new_intent.get('phrases') or [])
                                            if str(p).strip()][:6]}
    else:
        answer['new_intent'] = None
    entities = []
    for entity in answer.get('entities') or []:
        if not isinstance(entity, dict):
            continue
        name = re.sub(r'[^a-z0-9_]', '_', str(entity.get('type') or '').strip().lower())
        value = str(entity.get('value') or '').strip()
        if not name or not value:
            continue
        entities.append({'type': name, 'value': value[:200],
                         'fragment': str(entity.get('fragment') or '').strip()[:200]})
    answer['entities'] = entities[:10]
    new_types = []
    for item in answer.get('new_entity_types') or []:
        if not isinstance(item, dict):
            continue
        name = re.sub(r'[^a-z0-9_]', '_', str(item.get('name') or '').strip().lower())
        kind = str(item.get('kind') or 'text').strip().lower()
        if name and kind in ('text', 'enum', 'number'):
            new_types.append({'name': name, 'kind': kind})
    answer['new_entity_types'] = new_types[:5]
    answer['notes'] = str(answer.get('notes') or '').strip()[:300]
    return answer


def apply_annotation(conn, recording_id: int, answer: dict, *, model: str,
                     attempts: int = 1, expected_transcript: str | None = None) -> dict:
    """Create missing dictionary rows and label the recording; returns annotation meta.

    Refuses to touch recordings that are no longer 'new', so a manual edit made
    while the LLM call was in flight always wins.
    """
    with conn:
        conn.execute('BEGIN IMMEDIATE')
        row = db.get_recording(conn, recording_id)
        if row is None:
            raise LLMError(f'recording {recording_id} does not exist')
        if row['status'] != 'new':
            raise LLMError(f'recording {recording_id} is already {row["status"]}')
        transcript = row['corrected_transcript'] or row['original_transcript'] or ''
        if expected_transcript is not None and expected_transcript != transcript:
            raise LLMError('transcript changed while the LLM was running')
        new_intent = answer.get('new_intent')
        if new_intent and not conn.execute('SELECT 1 FROM intents WHERE name = ?',
                                           (answer['intent'],)).fetchone():
            db.ensure_intent(conn, answer['intent'], description=new_intent.get('description'),
                             phrases=new_intent.get('phrases') or [])
        for item in answer.get('new_entity_types') or []:
            db.ensure_entity_type(conn, item['name'], item['kind'])
        if not conn.execute('SELECT 1 FROM intents WHERE name = ?', (answer['intent'],)).fetchone():
            raise LLMError(f'unknown intent without definition: {answer["intent"]}')
        entities = []
        for entity in answer['entities']:
            if not conn.execute('SELECT 1 FROM entity_types WHERE name = ?', (entity['type'],)).fetchone():
                raise LLMError(f'unknown entity type without definition: {entity["type"]}')
            fragment = entity['fragment'] if entity['fragment'] in (row['original_transcript'] or '') else ''
            entities.append({'type': entity['type'], 'value': entity['value'],
                             'original_fragment': fragment})
            conn.execute('INSERT OR IGNORE INTO intent_entity_types (intent_id, entity_type_id) '
                         'SELECT i.id, t.id FROM intents i, entity_types t WHERE i.name = ? AND t.name = ?',
                         (answer['intent'], entity['type']))
        notes = f'LLM {model}' + (f': {answer["notes"]}' if answer.get('notes') else '')
        warnings = db.save_label(conn, recording_id, corrected_transcript=transcript,
                                 intent=answer['intent'], entities=entities,
                                 status='labeled', notes=notes[:500])
        annotation = {'source': 'llm', 'model': model, 'prompt_version': PROMPT_VERSION,
                      'at': db.utcnow(), 'attempts': attempts, 'status': 'ok',
                      'intent': answer['intent'], 'entities': answer['entities']}
        if warnings:
            annotation['warnings'] = warnings
        db.set_annotation(conn, recording_id, annotation)
    return annotation


class Annotator:
    """Chat-completions client for one configured LLM endpoint."""

    def __init__(self, settings: dict, client=None):
        self.settings = validate_llm(settings)
        if not self.settings['api_key']:
            raise LLMConfigError('brak klucza API LLM')
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=self.settings['api_key'], base_url=self.settings['url'],
                            timeout=float(self.settings['timeout']), max_retries=1)
        self.client = client

    def annotate(self, transcript: str, dictionary: dict) -> dict:
        """One transcript -> validated answer dict (see :func:`parse_answer`)."""
        try:
            completion = self.client.chat.completions.create(
                model=self.settings['model'],
                temperature=0,
                response_format={'type': 'json_object'},
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': json.dumps(
                        {'dictionary': dictionary, 'transcript': transcript},
                        ensure_ascii=False)},
                ])
        except Exception as exc:
            raise LLMError(f'LLM request failed: {exc}') from exc
        content = completion.choices[0].message.content if completion.choices else None
        return parse_answer(content or '')

    def test(self) -> str:
        """Small Chat Completions round-trip used by the admin Test button."""
        try:
            response = self.client.chat.completions.create(
                model=self.settings['model'],
                messages=[{'role': 'user', 'content': 'Respond with OK.'}],
                max_tokens=16)
        except Exception as exc:
            raise LLMError(f'połączenie z LLM nie działa: {exc}') from exc
        if not response.choices:
            raise LLMError('model nie zwrócił odpowiedzi')
        return f'OK: model {self.settings["model"]} odpowiada'
