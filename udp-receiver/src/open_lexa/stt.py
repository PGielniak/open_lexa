"""Pluggable speech-to-text backends.

The UDP receiver only depends on the small :class:`Transcriber` protocol below.
The backend is selected at runtime so the UDP receiver can run against OpenAI
in the cloud, any OpenAI-compatible HTTP transcription API (such as the
in-cluster Hailo Whisper API in ``whisper-api/``), or both at once.

``DualTranscriber`` always calls a local Whisper endpoint and a cloud
``gpt-transcribe`` endpoint. Both transcripts are returned to the archive; the
cloud one is preferred for command handling and is saved as the automatically
"correct" transcript in the admin UI.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
import time
from typing import Mapping, Protocol

from .audio import wav_bytes

LOG = logging.getLogger('open_lexa.stt')

BACKENDS = ('dual', 'openai', 'http', 'local', 'openai-compatible')


class ConfigError(ValueError):
    """Raised when a backend is selected without its required configuration."""


@dataclass(slots=True)
class TranscriptResult:
    """One backend's attempt at a recording.

    ``role`` is ``local`` or ``cloud`` for the dual backend and ``None`` for a
    single backend. ``text`` is ``None`` only when the backend failed; a
    successful empty transcription is stored as an empty string.
    """

    text: str | None
    backend: str
    model: str | None = None
    role: str | None = None
    error: str | None = None
    latency_ms: int | None = None

    def as_dict(self) -> dict:
        data: dict = {'backend': self.backend, 'text': self.text}
        for key in ('model', 'error', 'latency_ms'):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


class Transcriber(Protocol):
    """Minimal interface the receiver's worker thread needs."""

    name: str

    def warmup(self) -> None:
        """Load models or validate the remote endpoint before the first job."""

    def transcribe(self, pcm: bytes, language: str) -> str:
        """Return the transcript for mono PCM16LE/16 kHz audio."""

    def close(self) -> None:
        """Release any model/device resources. May be called more than once."""


class OpenAITranscriber:
    """OpenAI Audio Transcriptions (or any OpenAI-compatible endpoint)."""

    name = 'openai'

    def __init__(self, model: str, *, api_key: str | None = None,
                 base_url: str | None = None, client=None):
        self.model = model
        if client is None:
            from openai import OpenAI

            kwargs: dict = {'timeout': 45.0, 'max_retries': 2}
            if api_key:
                kwargs['api_key'] = api_key
            if base_url:
                kwargs['base_url'] = base_url
            client = OpenAI(**kwargs)
        self._client = client

    def warmup(self) -> None:
        # The OpenAI client is lazy; a real request happens with the first job.
        pass

    def transcribe(self, pcm: bytes, language: str) -> str:
        kwargs = {
            'model': self.model,
            'file': ('command.wav', wav_bytes(pcm), 'audio/wav'),
            'response_format': 'json',
        }
        if language and language != 'auto':
            # gpt-transcribe uses the newer `languages` list; classic Whisper
            # models on the same API still use the singular `language` field.
            if self.model.startswith('gpt-transcribe'):
                kwargs['extra_body'] = {'languages': [language]}
            else:
                kwargs['language'] = language
        return self._client.audio.transcriptions.create(**kwargs).text.strip()

    def transcribe_all(self, pcm: bytes, language: str) -> list[TranscriptResult]:
        """Call the backend and keep a failure as data instead of an exception."""
        started = time.monotonic()
        try:
            text = self.transcribe(pcm, language)
        except Exception as exc:
            return [TranscriptResult(None, self.name, self.model, error=str(exc),
                                     latency_ms=round((time.monotonic() - started) * 1000))]
        return [TranscriptResult(text, self.name, self.model,
                                 latency_ms=round((time.monotonic() - started) * 1000))]

    def close(self) -> None:
        close = getattr(self._client, 'close', None)
        if close is not None:
            close()


class OpenAICompatibleTranscriber(OpenAITranscriber):
    """An OpenAI-compatible transcription server, local or cloud-hosted."""

    name = 'http'

    def __init__(self, base_url: str, model: str, api_key: str = 'not-needed'):
        super().__init__(model, api_key=api_key, base_url=base_url)


# Kept as a public alias for existing callers and configuration documentation.
LocalOpenAITranscriber = OpenAICompatibleTranscriber


class DualTranscriber:
    """Run the local Whisper API and the cloud GPT transcription in parallel.

    The cloud transcript wins when it is non-empty; the local one is used as a
    fallback. One failing backend never discards the other one's transcript.
    """

    name = 'dual'

    def __init__(self, local: Transcriber, cloud: Transcriber):
        self.local = local
        self.cloud = cloud
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='stt-dual')

    def warmup(self) -> None:
        failures = []
        for transcriber in (self.local, self.cloud):
            try:
                transcriber.warmup()
            except Exception as exc:  # keep the working backend
                failures.append(f'{transcriber.name}: {exc}')
                LOG.warning('Dual STT backend %s failed to warm up: %s', transcriber.name, exc)
        if len(failures) == 2:
            raise RuntimeError('; '.join(failures))

    def _run(self, role: str, transcriber: Transcriber, pcm: bytes, language: str) -> TranscriptResult:
        started = time.monotonic()
        try:
            text = transcriber.transcribe(pcm, language)
        except Exception as exc:
            return TranscriptResult(None, transcriber.name, getattr(transcriber, 'model', None), role,
                                    str(exc), round((time.monotonic() - started) * 1000))
        return TranscriptResult(text, transcriber.name, getattr(transcriber, 'model', None), role,
                                latency_ms=round((time.monotonic() - started) * 1000))

    def transcribe_all(self, pcm: bytes, language: str) -> list[TranscriptResult]:
        futures = [
            (role, self._executor.submit(self._run, role, transcriber, pcm, language))
            for role, transcriber in (('local', self.local), ('cloud', self.cloud))
        ]
        results = []
        for role, future in futures:
            try:
                results.append(future.result())
            except Exception as exc:  # executor failure, not a backend failure
                results.append(TranscriptResult(None, 'dual', role=role, error=str(exc)))
        return results

    def transcribe(self, pcm: bytes, language: str) -> str:
        results = self.transcribe_all(pcm, language)
        chosen = preferred_result(results)
        if chosen is None:
            raise RuntimeError(format_errors(results) or 'STT returned no transcript')
        return chosen.text or ''

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        for transcriber in (self.local, self.cloud):
            try:
                transcriber.close()
            except Exception:
                LOG.exception('Closing %s failed', transcriber.name)


def transcribe_all(transcriber: Transcriber, pcm: bytes, language: str) -> list[TranscriptResult]:
    """Return all backend results, also for simple test/legacy transcribers."""
    method = getattr(transcriber, 'transcribe_all', None)
    if method is not None:
        return method(pcm, language)
    started = time.monotonic()
    try:
        text = transcriber.transcribe(pcm, language)
    except Exception as exc:
        return [TranscriptResult(None, transcriber.name, getattr(transcriber, 'model', None),
                                 error=str(exc), latency_ms=round((time.monotonic() - started) * 1000))]
    return [TranscriptResult(text, transcriber.name, getattr(transcriber, 'model', None),
                             latency_ms=round((time.monotonic() - started) * 1000))]


def preferred_result(results: list[TranscriptResult]) -> TranscriptResult | None:
    """Cloud first, then local, then any non-empty text.

    An empty transcript is still a successful result (it triggers the receiver's
    default reply), so a successful empty result beats a failed one.
    """
    for role in ('cloud', 'local'):
        for result in results:
            if result.role == role and result.text:
                return result
    for role in ('cloud', 'local'):
        for result in results:
            if result.role == role and result.text is not None and result.error is None:
                return result
    for result in results:
        if result.text:
            return result
    for result in results:
        if result.text is not None and result.error is None:
            return result
    return None


def format_errors(results: list[TranscriptResult]) -> str:
    errors = [result for result in results if result.error]
    if not errors:
        return ''
    if len(errors) == 1:
        return errors[0].error or ''
    return '; '.join(f'{result.role or result.backend}: {result.error}' for result in errors)


def transcripts_meta(results: list[TranscriptResult]) -> dict:
    """JSON-ready map used in the archive sidecar (``local``/``cloud`` keys)."""
    return {(result.role or result.backend): result.as_dict() for result in results}


def default_backend(env: Mapping[str, str]) -> str:
    """Dual when both endpoints are configured, otherwise the legacy cloud mode."""
    explicit = (env.get('STT_BACKEND') or '').strip()
    if explicit:
        return explicit
    if (env.get('STT_HTTP_URL') or env.get('STT_LOCAL_URL')) and env.get('OPENAI_API_KEY'):
        return 'dual'
    return 'openai'


def create_transcriber(backend: str | None = None, *, model: str | None = None,
                       env: Mapping[str, str] | None = None) -> Transcriber:
    """Build a backend from ``STT_BACKEND`` and the matching environment vars.

    Supported values: ``dual``, ``openai`` and ``http`` (also ``local`` or
    ``openai-compatible``). Without an explicit ``STT_BACKEND``, ``dual`` is
    selected when both a local URL and an OpenAI key are present; otherwise the
    legacy ``openai`` backend is kept. ``dual`` reads the local endpoint from
    ``STT_HTTP_URL``/``STT_HTTP_MODEL`` and the cloud one from
    ``OPENAI_API_KEY``/``OPENAI_STT_MODEL`` (optionally ``OPENAI_BASE_URL``);
    ``--model`` overrides the cloud model.
    """
    env = os.environ if env is None else env
    selected = (backend or default_backend(env)).strip().lower()

    if selected == 'dual':
        local_url = env.get('STT_HTTP_URL') or env.get('STT_LOCAL_URL')
        if not local_url:
            raise ConfigError('STT_HTTP_URL is required for STT_BACKEND=dual')
        api_key = env.get('OPENAI_API_KEY')
        if not api_key:
            raise ConfigError('OPENAI_API_KEY is required for STT_BACKEND=dual')
        local = OpenAICompatibleTranscriber(
            local_url,
            env.get('STT_HTTP_MODEL') or env.get('STT_LOCAL_MODEL', 'whisper-1'),
            api_key=env.get('STT_HTTP_API_KEY') or 'not-needed',
        )
        cloud = OpenAITranscriber(
            model or env.get('OPENAI_STT_MODEL', 'gpt-transcribe'),
            api_key=api_key,
            base_url=env.get('OPENAI_BASE_URL') or None,
        )
        return DualTranscriber(local, cloud)

    if selected == 'openai':
        api_key = env.get('OPENAI_API_KEY')
        if not api_key:
            raise ConfigError('OPENAI_API_KEY is required for STT_BACKEND=openai')
        return OpenAITranscriber(
            model or env.get('OPENAI_STT_MODEL', 'gpt-transcribe'),
            api_key=api_key,
            base_url=env.get('OPENAI_BASE_URL') or None,
        )

    if selected in ('http', 'local', 'openai-compatible'):
        base_url = env.get('STT_HTTP_URL') or env.get('OPENAI_BASE_URL')
        if not base_url:
            raise ConfigError('STT_HTTP_URL is required for STT_BACKEND=http')
        return OpenAICompatibleTranscriber(
            base_url,
            model or env.get('STT_HTTP_MODEL', 'whisper-1'),
            api_key=env.get('STT_HTTP_API_KEY') or 'not-needed',
        )

    raise ConfigError(f'Unknown STT_BACKEND {selected!r}; choose one of: {", ".join(BACKENDS)}')
