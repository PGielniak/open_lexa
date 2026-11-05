"""STT settings edited in the admin UI and published to the archive for the receiver.

The admin keeps the settings in SQLite and writes them to ``_config/stt.json`` on
the archive. The receiver polls that file and rebuilds its transcriber between
recordings; while the file is missing it keeps using its environment (``STT_*``).
A file that fails validation never replaces a working transcriber.

The default mode is ``dual``: every recording goes to the local Whisper API and
to cloud ``gpt-transcribe`` in parallel. The cloud transcript is stored as the
automatically corrected transcript in the admin UI. ``local`` and ``cloud``
remain available for offline or single-provider operation.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Callable

from .store import missing
from .stt import ConfigError, Transcriber, create_transcriber

LOG = logging.getLogger('open_lexa.settings')

SETTINGS_PATH = '_config/stt.json'
MODES = ('dual', 'local', 'cloud')
DEFAULTS = {
    'mode': 'dual',
    'local_url': 'http://open-lexa-whisper:8080/v1',
    # Base is the default for comparison on Polish recordings; the portal can
    # switch to 'whisper-small' (or 'whisper-1' = the
    # Whisper API's own default). See whisper-api/deploy/configmap.yaml.
    'local_model': 'whisper-base',
    'cloud_url': 'https://api.openai.com/v1',
    'cloud_model': 'gpt-transcribe',
    'cloud_api_key': '',
    'language': 'pl',
}


def validate(raw: dict) -> dict:
    """Return complete, normalised settings or raise :class:`ConfigError`."""
    if not isinstance(raw, dict):
        raise ConfigError('settings must be a JSON object')
    settings = {key: str(raw.get(key) or default).strip() for key, default in DEFAULTS.items()}
    if settings['mode'] not in MODES:
        raise ConfigError(f'mode must be one of: {", ".join(MODES)}')
    for key in ('local_url', 'cloud_url'):
        # The client appends /audio/transcriptions itself; a pasted endpoint URL would 404.
        settings[key] = settings[key].rstrip('/').removesuffix('/audio/transcriptions')
    modes = ('local', 'cloud') if settings['mode'] == 'dual' else (settings['mode'],)
    for prefix in modes:
        if not settings[f'{prefix}_url'].startswith(('http://', 'https://')):
            raise ConfigError(f'{prefix}_url must start with http:// or https://')
    if 'cloud' in modes and not settings['cloud_api_key']:
        raise ConfigError('the cloud service needs an API key')
    return settings


def build_transcriber(settings: dict) -> Transcriber:
    """Build the selected backend; dual uses both endpoints concurrently."""
    settings = validate(settings)
    mode = settings['mode']
    if mode == 'dual':
        return create_transcriber('dual', env={
            'STT_HTTP_URL': settings['local_url'],
            'STT_HTTP_MODEL': settings['local_model'],
            'OPENAI_BASE_URL': settings['cloud_url'],
            'OPENAI_STT_MODEL': settings['cloud_model'],
            'OPENAI_API_KEY': settings['cloud_api_key'],
        })
    return create_transcriber('http', env={
        'STT_HTTP_URL': settings[f'{mode}_url'],
        'STT_HTTP_MODEL': settings[f'{mode}_model'],
        'STT_HTTP_API_KEY': settings['cloud_api_key'] if mode == 'cloud' else '',
    })


def describe(settings: dict) -> str:
    mode = settings['mode']
    if mode == 'dual':
        return (f'dual: {settings["local_url"]} / {settings["local_model"]}'
                f' + {settings["cloud_url"]} / {settings["cloud_model"]}')
    return f'{mode}: {settings[f"{mode}_url"]} / {settings[f"{mode}_model"]}'


class SettingsWatcher:
    """Poll ``SETTINGS_PATH`` on a store; ``version`` changes whenever the file does.

    Polling runs on its own thread because an SMB read can stall; the receiver's
    worker only compares ``version`` and calls :meth:`build` between recordings.
    """

    def __init__(self, store, fallback: Callable[[], Transcriber], interval: float = 30):
        self.store = store
        self.fallback = fallback
        self.interval = interval
        self.raw: bytes | None = None
        self.version = 0
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name='settings', daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()

    def poll(self) -> None:
        try:
            data = self.store.read(SETTINGS_PATH)
        except OSError as exc:
            if not missing(exc):
                raise
            data = None
        if data != self.raw:
            self.raw = data
            self.version += 1
            LOG.info('STT settings %s', 'removed; using environment' if data is None else 'changed')

    def build(self) -> tuple[Transcriber, str | None]:
        """Transcriber and language override for the current file (env fallback when absent)."""
        raw = self.raw
        if raw is None:
            return self.fallback(), None
        try:
            settings = validate(json.loads(raw.decode('utf-8')))
        except ValueError as exc:  # JSONDecodeError and ConfigError
            raise ConfigError(f'invalid {SETTINGS_PATH}: {exc}') from exc
        return build_transcriber(settings), settings['language']

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                self.poll()
            except Exception:
                LOG.warning('Reading %s failed; keeping current STT settings', SETTINGS_PATH, exc_info=True)
            self.stopping.wait(self.interval)
