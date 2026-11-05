"""Map a transcript to a voice command.

``decide()`` keeps the original regex rules and acts as the fallback when no
lexicon is available or nothing matches confidently; the lexicon-based NLU
lives in :mod:`open_lexa.nlu`. A match is passed to the handler plugins in
:mod:`open_lexa.handlers` for execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import re

LOG = logging.getLogger('open_lexa.commands')

DEFAULT_REPLY = 'sorry ale nie rozumiem'

# (command, pattern) checked in order against the normalised transcript.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ('lights_on', re.compile(r'\bwłącz\b.*\bświatło\b')),
    ('lights_off', re.compile(r'\bwyłącz\b.*\bświatło\b')),
    ('play_radio', re.compile(r'\b(włącz|graj|puść)\b.*\bradio\b')),
    ('volume_up', re.compile(r'\bgłośniej\b')),
    ('volume_down', re.compile(r'\bciszej\b')),
)


@dataclass(frozen=True)
class Decision:
    """Recognition outcome for one transcript.

    ``command``/``reply`` keep the original receiver contract (and the archive
    sidecar keys); ``intent``/``entities``/``score``/``source`` carry the NLU
    detail the handler plugins and the admin UI use.
    """

    command: str | None
    reply: str
    intent: str | None = None
    entities: tuple[tuple[str, str], ...] = ()
    score: float | None = None
    source: str | None = None

    @property
    def entities_dict(self) -> dict[str, str]:
        return dict(self.entities)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace; keeps Polish letters."""
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.lower()).split())


def decide(text: str | None) -> Decision:
    """Choose the command for a transcript using the built-in regex rules."""
    normalized = normalize(text or '')
    for command, pattern in RULES:
        if pattern.search(normalized):
            LOG.info('Command: %s (%r)', command, text)
            return Decision(command, command, intent=command, score=100.0, source='rules')
    LOG.info('No command for %r: %s', text, DEFAULT_REPLY)
    return Decision(None, DEFAULT_REPLY)
