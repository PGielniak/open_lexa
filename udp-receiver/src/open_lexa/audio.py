"""Audio helpers shared by the receiver and the STT backends."""
from __future__ import annotations

import io
import wave

from .protocol import RATE


def wav_bytes(pcm: bytes) -> bytes:
    """Wrap mono PCM16LE samples in a canonical 16 kHz WAV container."""
    result = io.BytesIO()
    with wave.open(result, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    return result.getvalue()
