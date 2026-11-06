"""Lexicon-based intent/entity recognition for the receiver.

The admin exports ``_lexicon/lexicon.json`` (intents with canonical phrase
templates, entity dictionary with synonyms, and labeled examples). Matching
follows docs/LABELING.md: Polish phonetic folding + rapidfuzz against examples
and literal phrases, regex matching of phrase templates with ``{slot}``
placeholders, and dictionary/number entity extraction. When the lexicon is
missing or nothing matches confidently, the regex rules in
:mod:`open_lexa.commands` remain the fallback. An optional LLM cascade
(:mod:`open_lexa.llm`) can decide immediately after transcription or only
when both the lexicon and the rules recognise nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re

from rapidfuzz import fuzz

from .commands import DEFAULT_REPLY, Decision, decide as rules_decide
from .text import compact, normalize, phonetic

LOG = logging.getLogger('open_lexa.nlu')

LEXICON_PATH = '_lexicon/lexicon.json'

# Acceptance: best score must reach ACCEPT_SCORE and lead the runner-up by
# MARGIN, so ambiguous utterances fall back to the rules instead of guessing.
ACCEPT_SCORE = 78.0
MARGIN = 8.0
TEMPLATE_SCORE = 92.0
ENUM_MATCH_SCORE = 88.0

SLOT = re.compile(r'\{(\w+)\}')

# Basic Polish number words; composites ("dwadzieścia pięć") are summed.
UNITS = {'zero': 0, 'jeden': 1, 'jedną': 1, 'jedna': 1, 'dwa': 2, 'dwie': 2, 'trzy': 3,
         'cztery': 4, 'pięć': 5, 'piec': 5, 'sześć': 6, 'szesc': 6, 'siedem': 7,
         'osiem': 8, 'dziewięć': 9, 'dziewiec': 9}
TEENS = {'dziesięć': 10, 'dziesiec': 10, 'jedenaście': 11, 'jedenascie': 11,
         'dwanaście': 12, 'dwanascie': 12, 'trzynaście': 13, 'trzynascie': 13,
         'czternaście': 14, 'czternascie': 14, 'piętnaście': 15, 'pietnascie': 15,
         'szesnaście': 16, 'szesnascie': 16, 'siedemnaście': 17, 'siedemnascie': 17,
         'osiemnaście': 18, 'osiemnascie': 18, 'dziewiętnaście': 19, 'dziewietnascie': 19}
TENS = {'dwadzieścia': 20, 'dwadziescia': 20, 'trzydzieści': 30, 'trzydziesci': 30,
        'czterdzieści': 40, 'czterdziesci': 40, 'pięćdziesiąt': 50, 'piecdziesiat': 50,
        'sześćdziesiąt': 60, 'szescdziesiat': 60, 'siedemdziesiąt': 70, 'siedemdziesiat': 70,
        'osiemdziesiąt': 80, 'osiemdziesiat': 80, 'dziewięćdziesiąt': 90, 'dziewiecdziesiat': 90}
HUNDREDS = {'sto': 100, 'dwieście': 200, 'dwiescie': 200, 'trzysta': 300,
            'czterysta': 400, 'pięćset': 500, 'piecset': 500, 'sześćset': 600,
            'szescset': 600, 'siedemset': 700, 'osiemset': 800, 'dziewięćset': 900,
            'dziewiecset': 900}
NUMBER_WORDS = {**UNITS, **TEENS, **TENS, **HUNDREDS}


@dataclass(frozen=True)
class _Example:
    phonetic: str
    compact: str
    intent: str


@dataclass(frozen=True)
class _Template:
    regex: re.Pattern[str]
    slots: tuple[str, ...]
    intent: str
    score: float
    prefix_words: int
    suffix_words: int


@dataclass(frozen=True)
class Lexicon:
    """Parsed, pre-compiled form of the admin-exported lexicon."""

    intents: dict[str, dict] = field(default_factory=dict)          # name -> info
    examples: tuple[_Example, ...] = ()
    literals: tuple[tuple[str, str, str], ...] = ()                 # (phonetic, compact, intent)
    templates: tuple[_Template, ...] = ()
    phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)  # intent -> original phrases
    entities: dict[str, dict] = field(default_factory=dict)         # type -> {'kind','values'}
    version: int = 0

    @property
    def empty(self) -> bool:
        return not self.intents


def _template_regex(phrase: str) -> tuple[re.Pattern[str], tuple[str, ...]] | None:
    """Compile a canonical phrase with {slot} placeholders into a regex.

    Built on the phonetic form so STT spelling noise ("rozpotify") still
    matches; slot values are therefore diacritic-free, which is fine for
    search queries and gets re-resolved against the dictionary for enums.
    """
    if '{' not in phrase:
        return None
    slots: list[str] = []
    parts: list[str] = []
    position = 0
    for match in SLOT.finditer(phrase):
        literal = phonetic(phrase[position:match.start()])
        if literal:
            parts.append(re.escape(literal).replace(r'\ ', r'\s+'))
        slots.append(match.group(1))
        parts.append(f'(?P<slot{len(slots) - 1}>.+?)')
        position = match.end()
    tail = phonetic(phrase[position:])
    if tail:
        parts.append(re.escape(tail).replace(r'\ ', r'\s+'))
    if not slots:
        return None
    return re.compile(r'\s*'.join(parts) + r'\s*'), tuple(slots)


def parse_lexicon(data: dict) -> Lexicon:
    """Validate and pre-compile an exported lexicon; raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError('lexicon must be a JSON object')
    intents = data.get('intents')
    if not isinstance(intents, list):
        raise ValueError('lexicon.intents must be a list')
    info: dict[str, dict] = {}
    literals: list[tuple[str, str, str]] = []
    templates: list[_Template] = []
    phrases: dict[str, list[str]] = {}
    for intent in intents:
        if not isinstance(intent, dict) or not intent.get('name'):
            raise ValueError('every intent needs a name')
        name = str(intent['name'])
        info[name] = {
            'parent': intent.get('parent'),
            'actionable': bool(intent.get('actionable', True)),
            'entities': [str(e) for e in intent.get('entities') or []],
        }
        for phrase in intent.get('phrases') or []:
            phrase = str(phrase)
            phrases.setdefault(name, []).append(phrase)
            compiled = _template_regex(phrase)
            if compiled is not None:
                prefix = phonetic(phrase.split('{', 1)[0])
                suffix = phonetic(phrase.rsplit('}', 1)[-1])
                specificity = 115.0 if suffix or len(prefix.split()) >= 2 else 85.0
                templates.append(_Template(compiled[0], compiled[1], name, specificity,
                                           len(prefix.split()), len(suffix.split())))
            else:
                literals.append((phonetic(phrase), compact(phrase), name))
    examples = []
    for example in data.get('examples') or []:
        if not isinstance(example, dict) or not example.get('text') or not example.get('intent'):
            continue
        text = str(example['text'])
        examples.append(_Example(example.get('phonetic') or phonetic(text), compact(text),
                                 str(example['intent'])))
    entities = {}
    for name, spec in (data.get('entities') or {}).items():
        if not isinstance(spec, dict):
            continue
        values = {}
        for value, synonyms in (spec.get('values') or {}).items():
            values[str(value)] = [str(s) for s in synonyms or []]
        entities[str(name)] = {'kind': spec.get('kind', 'text'), 'values': values}
    return Lexicon(intents=info, examples=tuple(examples), literals=tuple(literals),
                   templates=tuple(templates), phrases={k: tuple(v) for k, v in phrases.items()},
                   entities=entities, version=int(data.get('version') or 1))


def _extract_number(text: str) -> str | None:
    """Digits or a sum of Polish number words found in ``text``."""
    digits = re.search(r'\d+', text)
    if digits:
        return digits.group(0)
    total = 0
    found = False
    for token in normalize(text).split():
        value = NUMBER_WORDS.get(token)
        if value is not None:
            total += value
            found = True
    return str(total) if found else None


def _extract_entities(text: str, allowed: list[str], dictionary: dict[str, dict],
                      have: dict[str, str]) -> dict[str, str]:
    """Fill entity types the templates did not already capture."""
    folded = phonetic(text)
    result: dict[str, str] = {}
    for entity_type in allowed:
        if entity_type in have:
            continue
        spec = dictionary.get(entity_type)
        if spec is None:
            continue
        if spec['kind'] == 'number':
            number = _extract_number(text)
            if number is not None:
                result[entity_type] = number
            continue
        # enum/text: best fuzzy dictionary hit over the phonetic form.
        best_value, best_score = None, 0.0
        for value, synonyms in spec['values'].items():
            for candidate in (value, *synonyms):
                score = fuzz.partial_ratio(phonetic(candidate), folded)
                if score > best_score:
                    best_value, best_score = value, score
        if best_value is not None and best_score >= ENUM_MATCH_SCORE:
            result[entity_type] = best_value
    return result


def match(text: str | None, lexicon: Lexicon) -> Decision | None:
    """Best lexicon intent for ``text``, or None when not confident enough."""
    if not text or not text.strip() or lexicon.empty:
        return None
    folded = phonetic(text)
    squeezed = compact(text)
    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    exact_literals: set[str] = set()

    def track(intent: str, score: float, source: str) -> None:
        if score > scores.get(intent, 0.0):
            scores[intent] = score
            sources[intent] = source

    for example in lexicon.examples:
        score = max(fuzz.token_set_ratio(folded, example.phonetic),
                    fuzz.ratio(squeezed, example.compact))
        if squeezed == example.compact:
            score = 125.0
        track(example.intent, float(score), 'lexicon')
    for phrase_phonetic, phrase_compact, intent in lexicon.literals:
        score = max(fuzz.token_set_ratio(folded, phrase_phonetic),
                    fuzz.ratio(squeezed, phrase_compact))
        if squeezed == phrase_compact:
            score = 125.0
            exact_literals.add(intent)
        track(intent, float(score), 'canonical')

    slot_values: dict[str, dict[str, str]] = {}
    slot_specificity: dict[str, tuple[float, int, int]] = {}
    for template in lexicon.templates:
        found = template.regex.fullmatch(folded)
        if found is None:
            continue
        # A generic "włącz {station}" must not beat an exact "włącz
        # spotify", or a more specific "włącz playlistę {search_query}".
        score = template.score
        track(template.intent, score, 'canonical')
        captured = {slot: (found.group(f'slot{index}') or '').strip()
                    for index, slot in enumerate(template.slots)}
        if len(template.slots) == 1:
            # Preserve the original spelling of names ("Chill", not the
            # phonetic "hill") when extracting a template's single slot.
            words = normalize(text).split()
            end = len(words) - template.suffix_words if template.suffix_words else len(words)
            if template.prefix_words < end:
                captured[template.slots[0]] = ' '.join(words[template.prefix_words:end])
        for slot, value in captured.items():
            if lexicon.entities.get(slot, {}).get('kind') == 'number':
                captured[slot] = _extract_number(value) or value
        specificity = (score, template.prefix_words + template.suffix_words,
                       template.prefix_words)
        if specificity > slot_specificity.get(template.intent, (0.0, 0, 0)):
            slot_values[template.intent] = {key: value for key, value in captured.items() if value}
            slot_specificity[template.intent] = specificity

    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    intent, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best < ACCEPT_SCORE or (best - runner_up) < MARGIN:
        return None
    info = lexicon.intents.get(intent, {})
    # An exact command without slots (e.g. "włącz muzykę ze Spotify") must
    # not inherit a slot from a more generic template for the same intent.
    entities = {} if intent in exact_literals else dict(slot_values.get(intent, {}))
    entities.update(_extract_entities(text or '', info.get('entities', []),
                                      lexicon.entities, entities))
    if not info.get('actionable', True):
        # noise/not_for_me: recognised, but nothing to execute or answer.
        return Decision(None, '', intent=intent, entities=tuple(sorted(entities.items())),
                        score=round(best, 1), source=sources.get(intent, 'lexicon'))
    return Decision(intent, intent, intent=intent, entities=tuple(sorted(entities.items())),
                    score=round(best, 1), source=sources.get(intent, 'lexicon'))


class NLU:
    """Lexicon matcher with the regex-rule fallback and an optional LLM cascade."""

    def __init__(self, lexicon: Lexicon | None = None, llm=None, mode: str = 'off'):
        self.lexicon = lexicon or Lexicon()
        self.llm = llm
        self.mode = mode if llm is not None else 'off'

    def update(self, data: dict | None) -> bool:
        """Swap in a new lexicon (None clears it). True when it changed."""
        if data is None:
            replaced, self.lexicon = self.lexicon, Lexicon()
            return not replaced.empty
        try:
            lexicon = parse_lexicon(data)
        except ValueError as exc:
            LOG.warning('Invalid lexicon ignored: %s', exc)
            return False
        changed = lexicon.intents != self.lexicon.intents or lexicon.version != self.lexicon.version
        self.lexicon = lexicon
        LOG.info('Lexicon loaded: %d intents, %d examples', len(lexicon.intents), len(lexicon.examples))
        return changed

    def decide(self, text: str | None) -> Decision:
        """Lexicon first, then regex rules, then the LLM cascade if configured."""
        decision = match(text, self.lexicon)
        if decision is not None:
            LOG.info('NLU: %s (%s, score %.0f) for %r', decision.intent, decision.source,
                     decision.score or 0, text)
            base = decision
        else:
            base = rules_decide(text) if (text or '').strip() else Decision(None, DEFAULT_REPLY)
        if self.llm is None or self.mode == 'off' or not (text or '').strip():
            return base
        if self.mode == 'fallback' and base.intent is not None:
            return base
        llm_decision = self.llm.recognize(text, self.lexicon)
        if llm_decision is not None:
            LOG.info('LLM NLU: %s (score %s) for %r', llm_decision.intent,
                     llm_decision.score, text)
            return llm_decision
        return base
