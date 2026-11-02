"""Polish text normalisation shared by labeling and (later) the receiver's NLU.

STT errors are mostly phonetic, so matching happens on a folded form where
spellings that sound alike collapse: diacritics are dropped, ó/u, rz/ż and ch/h
are merged. :func:`compact` also removes spaces because Whisper often moves word
boundaries ("rozpotify" for "ze spotify").
"""
from __future__ import annotations

import re

_FOLD = str.maketrans('ąćęłńóśźż', 'acelnuszz')


def normalize(text: str | None) -> str:
    """Lowercase, drop punctuation and collapse whitespace; keeps Polish letters."""
    return ' '.join(re.sub(r'[^\w\s]', ' ', (text or '').lower()).split())


def phonetic(text: str | None) -> str:
    """Folded form for fuzzy matching: 'Włącz muzykę ze Spotify' -> 'wlacz muzyke ze spotify'."""
    return normalize(text).replace('rz', 'z').replace('ch', 'h').translate(_FOLD)


def compact(text: str | None) -> str:
    """:func:`phonetic` without spaces."""
    return phonetic(text).replace(' ', '')
