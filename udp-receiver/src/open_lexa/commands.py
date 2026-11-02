"""Map a transcript to a voice command.

Commands are not executed yet: a match is only logged. Add a rule to
:data:`RULES` to teach the receiver a new phrase.
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
    command: str | None
    reply: str


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace; keeps Polish letters."""
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.lower()).split())


def decide(text: str | None) -> Decision:
    """Choose the command for a transcript and "execute" it (log only for now)."""
    normalized = normalize(text or '')
    for command, pattern in RULES:
        if pattern.search(normalized):
            LOG.info('Command: %s (%r)', command, text)
            return Decision(command, command)
    LOG.info('No command for %r: %s', text, DEFAULT_REPLY)
    return Decision(None, DEFAULT_REPLY)
