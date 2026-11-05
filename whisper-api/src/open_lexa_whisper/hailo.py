"""Whisper on a local Hailo-10H through HailoRT's GenAI Speech2Text API.

Several Whisper HEFs (e.g. Base and Small) can be bundled in one image. Only
one model is loaded on the device at a time; ``select_model`` schedules a
switch that happens on the inference worker thread, because HailoRT device
handles are thread-affine.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Mapping

from . import RATE

LOG = logging.getLogger('open_lexa.whisper_api')
DEFAULT_HAILO_HEF = '/models/Whisper-Base.hef'
DEFAULT_MODEL_NAME = 'whisper-base'
# Canonical names for the bundled Hailo HEFs, keyed by the model id the
# receiver/admin uses. WHISPER_API_MODELS may override or extend this map.
WELL_KNOWN_HEFS = {
    'whisper-base': '/models/Whisper-Base.hef',
    'whisper-small': '/models/Whisper-Small.hef',
}


class ConfigError(ValueError):
    """Raised when the model is configured incorrectly."""


class TranscriptionError(RuntimeError):
    """Raised when HailoRT cannot initialise or run."""


def parse_models(env: Mapping[str, str]) -> tuple[dict[str, Path], str]:
    """Return ({model name: HEF path}, default model) from the environment.

    ``HAILO_WHISPER_MODELS`` is a comma-separated ``name=/path.hef`` list.
    Without it, ``HAILO_WHISPER_HEF`` (single model) or the well-known bundled
    HEFs that exist on disk are used. The default model comes from
    ``WHISPER_API_DEFAULT_MODEL``, else the single/first configured model.
    """
    models: dict[str, Path] = {}
    spec = (env.get('HAILO_WHISPER_MODELS') or '').strip()
    if spec:
        for item in spec.split(','):
            item = item.strip()
            if not item:
                continue
            name, _, path = item.partition('=')
            if not name.strip() or not path.strip():
                raise ConfigError(f'HAILO_WHISPER_MODELS entries must be name=/path.hef, got {item!r}')
            models[name.strip().lower()] = Path(path.strip())
    elif env.get('HAILO_WHISPER_HEF'):
        # 'Whisper-Base.hef' -> 'whisper-base' keeps the usual model id shape.
        name = (env.get('WHISPER_API_DEFAULT_MODEL') or '').strip().lower() \
            or Path(env['HAILO_WHISPER_HEF']).stem.lower()
        models[name] = Path(env['HAILO_WHISPER_HEF'])
    else:
        for name, path in WELL_KNOWN_HEFS.items():
            if Path(path).is_file():
                models[name] = Path(path)
        if not models:
            models[DEFAULT_MODEL_NAME] = Path(DEFAULT_HAILO_HEF)
    default = (env.get('WHISPER_API_DEFAULT_MODEL') or '').strip().lower() or next(iter(models))
    if default not in models:
        raise ConfigError(f'default model {default!r} is not among {", ".join(models)}')
    missing = [str(path) for path in models.values() if not path.is_file()]
    if missing:
        raise ConfigError(f'Hailo Whisper model not found: {", ".join(missing)}')
    return models, default


class HailoWhisperTranscriber:
    """Whisper running on a local Hailo-10H through HailoRT's GenAI API.

    HailoRT ties device handles to the thread that created them, so the model
    is initialised lazily inside its single inference worker thread. The lock
    also serialises access to the single-tenant NPU. Model switches requested
    through :meth:`select_model` are applied on that same thread by the next
    :meth:`warmup`/:meth:`transcribe` call.
    """

    name = 'hailo'

    def __init__(self, models: Mapping[str, str | Path], default: str, *,
                  timeout_ms: int = 15_000, chunk_seconds: float = 10.0,
                  group_id: str | None = None, normalize_peak: float = 0.9):
        self._models = {name: Path(path) for name, path in models.items()}
        if not self._models:
            raise ConfigError('at least one Whisper model must be configured')
        if default not in self._models:
            raise ConfigError(f'default model {default!r} is not among {", ".join(self._models)}')
        self.default_model = default
        self.timeout_ms = timeout_ms
        self.chunk_seconds = chunk_seconds
        self.group_id = group_id
        self.normalize_peak = normalize_peak
        self._lock = threading.Lock()
        self._selected = default
        self._loaded: str | None = None
        self._device = None
        self._speech = None
        self._task = None
        self._np = None
        for path in self._models.values():
            if not path.is_file():
                raise ConfigError(f'Hailo Whisper model not found: {path}')

    @property
    def models(self) -> tuple[str, ...]:
        """Model ids this transcriber can switch between."""
        return tuple(self._models)

    @property
    def model(self) -> str:
        """The model selected for the next transcription."""
        with self._lock:
            return self._selected

    @property
    def loaded_model(self) -> str | None:
        """The model currently loaded on the device (None before warmup)."""
        with self._lock:
            return self._loaded

    def select_model(self, model: str) -> None:
        """Choose the model for subsequent transcriptions.

        Only records the selection; the device work happens on the inference
        thread inside :meth:`_ensure_loaded` so HailoRT thread affinity holds.
        """
        with self._lock:
            if model not in self._models:
                raise ValueError(f'unknown model {model!r}; available: {", ".join(self._models)}')
            self._selected = model

    def warmup(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def _release_locked(self) -> None:
        speech, device = self._speech, self._device
        self._speech = self._device = None
        self._loaded = None
        if speech is not None:
            try:
                speech.release()
            except Exception:
                LOG.exception('Releasing the Hailo Speech2Text model failed')
        if device is not None:
            try:
                device.release()
            except Exception:
                LOG.exception('Releasing the Hailo device failed')

    def _ensure_loaded(self) -> None:
        if self._speech is not None and self._loaded == self._selected:
            return
        previous = self._loaded
        if self._speech is not None:
            LOG.info('Switching Hailo Whisper model %s -> %s', self._loaded, self._selected)
            self._release_locked()
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
        try:
            device = VDevice(params)
            speech = Speech2Text(device, str(self._models[self._selected]))
        except Exception:
            if 'device' in locals():
                device.release()
            if previous and previous != self._selected:
                self._selected = previous
                try:
                    self._ensure_loaded()
                except Exception:
                    LOG.exception('Restoring Hailo Whisper model %s failed', previous)
            raise
        self._device = device
        self._speech = speech
        self._task = Speech2TextTask
        self._np = np
        self._loaded = self._selected

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
            self._release_locked()


def create_transcriber(env: Mapping[str, str] | None = None) -> HailoWhisperTranscriber:
    """Build the transcriber from the ``HAILO_WHISPER_*``/``WHISPER_API_*`` environment."""
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
    models, default = parse_models(env)
    return HailoWhisperTranscriber(
        models,
        default,
        timeout_ms=timeout_ms,
        chunk_seconds=chunk_seconds,
        group_id=env.get('HAILO_VDEVICE_GROUP_ID') or None,
        normalize_peak=normalize_peak,
    )
