"""Whisper on a local Hailo-10H through HailoRT's GenAI Speech2Text API."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Mapping

from . import RATE

DEFAULT_HAILO_HEF = '/models/Whisper-Base.hef'


class ConfigError(ValueError):
    """Raised when the model is configured incorrectly."""


class TranscriptionError(RuntimeError):
    """Raised when HailoRT cannot initialise or run."""


class HailoWhisperTranscriber:
    """Whisper running on a local Hailo-10H through HailoRT's GenAI API.

    HailoRT ties device handles to the thread that created them, so the model
    is initialised lazily inside its single inference worker thread. The lock
    also serialises access to the single-tenant NPU.
    """

    name = 'hailo'

    def __init__(self, hef_path: str | Path = DEFAULT_HAILO_HEF, *,
                  timeout_ms: int = 15_000, chunk_seconds: float = 10.0,
                  group_id: str | None = None, normalize_peak: float = 0.9):
        self.hef_path = Path(hef_path)
        self.timeout_ms = timeout_ms
        self.chunk_seconds = chunk_seconds
        self.group_id = group_id
        self.normalize_peak = normalize_peak
        self._lock = threading.Lock()
        self._device = None
        self._speech = None
        self._task = None
        self._np = None
        if not self.hef_path.is_file():
            raise ConfigError(f'Hailo Whisper model not found: {self.hef_path}')

    def warmup(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._speech is not None:
            return
        try:
            import numpy as np
            from hailo_platform import VDevice
            from hailo_platform.genai import Speech2Text, Speech2TextTask
        except ImportError as exc:
            raise TranscriptionError(
                'Hailo backend requires HailoRT Python bindings (hailo_platform); '
                'build the image from whisper-api/build/Dockerfile'
            ) from exc
        params = VDevice.create_params()
        if self.group_id:
            params.group_id = self.group_id
        device = VDevice(params)
        try:
            speech = Speech2Text(device, str(self.hef_path))
        except Exception:
            device.release()
            raise
        self._device = device
        self._speech = speech
        self._task = Speech2TextTask
        self._np = np

    def transcribe(self, pcm: bytes, language: str) -> str:
        with self._lock:
            self._ensure_loaded()
            np = self._np
            audio = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
            # The Hailo Speech2Text API expects normalized float32 PCM. The
            # ESP32's fixed mic gain can leave quieter utterances far below
            # the model's useful range. Boost only those utterances to a
            # conservative peak below full scale; do not attenuate already
            # loud/clipped recordings, whose decoding can change. A value of
            # 0 disables this behaviour for diagnosis/tuning.
            if self.normalize_peak:
                peak = float(np.max(np.abs(audio)))
                if 0 < peak < self.normalize_peak:
                    audio *= self.normalize_peak / peak
            if self.chunk_seconds > 0:
                size = max(1, int(RATE * self.chunk_seconds))
                chunks = [audio[offset:offset + size]
                          for offset in range(0, len(audio), size)]
            else:
                chunks = [audio]
            parts: list[str] = []
            for chunk in chunks:
                kwargs = {
                    'audio_data': chunk,
                    'task': self._task.TRANSCRIBE,
                    'timeout_ms': self.timeout_ms,
                }
                if language and language != 'auto':
                    kwargs['language'] = language
                segments = self._speech.generate_all_segments(**kwargs)
                parts.append(''.join(segment.text for segment in segments))
            return ' '.join(part.strip() for part in parts if part.strip()).strip()

    def close(self) -> None:
        with self._lock:
            speech, device = self._speech, self._device
            self._speech = self._device = None
            if speech is not None:
                try:
                    speech.release()
                except Exception:
                    pass
            if device is not None:
                try:
                    device.release()
                except Exception:
                    pass


def create_transcriber(env: Mapping[str, str] | None = None) -> HailoWhisperTranscriber:
    """Build the transcriber from the ``HAILO_WHISPER_*`` environment variables."""
    env = os.environ if env is None else env
    try:
        timeout_ms = int(env.get('HAILO_WHISPER_TIMEOUT_MS', '15000'))
        chunk_seconds = float(env.get('HAILO_WHISPER_CHUNK_SECONDS', '10'))
        normalize_peak = float(env.get('HAILO_WHISPER_NORMALIZE_PEAK', '0.9'))
    except ValueError as exc:
        raise ConfigError(
            'HAILO_WHISPER_TIMEOUT_MS/HAILO_WHISPER_CHUNK_SECONDS/'
            'HAILO_WHISPER_NORMALIZE_PEAK must be numeric') from exc
    if not 0 <= normalize_peak <= 1:
        raise ConfigError('HAILO_WHISPER_NORMALIZE_PEAK must be between 0 and 1')
    return HailoWhisperTranscriber(
        env.get('HAILO_WHISPER_HEF', DEFAULT_HAILO_HEF),
        timeout_ms=timeout_ms,
        chunk_seconds=chunk_seconds,
        group_id=env.get('HAILO_VDEVICE_GROUP_ID') or None,
        normalize_peak=normalize_peak,
    )
