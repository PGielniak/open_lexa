"""Spoken confirmations on the Sonos speaker before an MCP tool runs.

After the LLM picks a tool (and before the dispatcher executes it) the receiver
plays a short phrase such as "Okej, uruchamiam Spotify"; when the LLM picks no
tool it plays "Sorry, nie zrozumiałam", and when the request or the tool
fails it plays "Przepraszam, coś poszło nie tak" (the ``failed`` phrase may
contain ``{error}``, replaced by the error message). Phrases are synthesised once with an
OpenAI-compatible TTS endpoint, cached in memory and on the archive under
``_clips/`` and served over HTTP, because Sonos only plays URLs it fetches
itself. Configuration lives in the ``sonos.announce`` section of
``_config/handlers.json``::

    {"sonos": {"enabled": true, "speaker_ip": "192.168.1.42",
               "announce": {"enabled": true,
                            "base_url": "http://192.168.1.10:8081",
                            "model": "gpt-4o-mini-tts", "voice": "nova",
                            "phrases": {"play_spotify": "Okej, uruchamiam Spotify"}}}}

An empty phrase disables the announcement for that tool. Announcements never
raise: a missing clip or an unreachable speaker only logs a warning and the
tool still runs.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Mapping

from .store import missing

LOG = logging.getLogger('open_lexa.announce')

NOT_UNDERSTOOD = 'not_understood'
FAILED = 'failed'
ERROR_PLACEHOLDER = '{error}'
MAX_ERROR_CHARS = 160
# Error phrases differ per error: keep only the most recent ones in memory.
MAX_EPHEMERAL_CLIPS = 16
CLIPS_DIR = '_clips'
DEFAULT_MODEL = 'gpt-4o-mini-tts'
DEFAULT_VOICE = 'nova'
DEFAULT_INSTRUCTIONS = 'Mów po polsku, naturalnie i krótko, przyjaznym głosem asystentki.'
MAX_CLIP_SECONDS = 10.0
# TTS runs inline on a cache miss, right before the action: keep it short, and
# do not retry a failing phrase on every command.
TTS_TIMEOUT = 5.0
FAILURE_BACKOFF = 300.0

# (key, admin label, default phrase); keys are MCP tool names plus NOT_UNDERSTOOD
# and FAILED.
PHRASES: tuple[tuple[str, str, str], ...] = (
    ('play_music', 'Domyślna playlista', 'Okej, włączam muzykę.'),
    ('play_spotify', 'Spotify', 'Okej, uruchamiam Spotify.'),
    ('queue_spotify', 'Dodanie do kolejki', 'Okej, dodaję do kolejki.'),
    ('play_radio', 'Wybrane radio', 'Okej, włączam radio.'),
    ('play_default_radio', 'Domyślne radio', 'Okej, włączam radio.'),
    ('play_random_radio', 'Losowe radio', 'Okej, losuję stację.'),
    ('stop', 'Stop', 'Zatrzymuję.'),
    ('pause', 'Pauza', 'Pauza.'),
    ('resume', 'Wznowienie', 'Wznawiam.'),
    ('next_track', 'Następny utwór', 'Następny utwór.'),
    ('previous_track', 'Poprzedni utwór', 'Poprzedni utwór.'),
    ('volume_up', 'Głośniej', 'Głośniej.'),
    ('volume_down', 'Ciszej', 'Ciszej.'),
    ('volume_set', 'Ustaw głośność', 'Ustawiam głośność.'),
    (NOT_UNDERSTOOD, 'Nie zrozumiano polecenia', 'Sorry, nie zrozumiałam.'),
    (FAILED, 'Błąd (może zawierać {error})', 'Przepraszam, coś poszło nie tak, spróbuj ponownie.'),
)
DEFAULT_PHRASES = {key: text for key, _, text in PHRASES}

# The tool starts new playback anyway: restoring the old source first would
# only add seconds and a blip of the previous music.
REPLACES_PLAYBACK = frozenset({'play_music', 'play_spotify', 'play_radio',
                               'play_default_radio', 'play_random_radio'})
# Restore the source and position without resuming it.
STOPS_PLAYBACK = frozenset({'stop', 'pause'})

CLIP_NAME = re.compile(r'[0-9a-f]{24}\.mp3')


def restore_mode(key: str) -> str:
    """'none', 'paused' or 'full': what to do with the previous playback after the clip."""
    if key in REPLACES_PLAYBACK:
        return 'none'
    if key in STOPS_PLAYBACK:
        return 'paused'
    return 'full'


def error_phrase(template: str, error: str | None) -> str:
    """Fill ``{error}`` with a shortened, single-line error message."""
    if ERROR_PLACEHOLDER not in template:
        return template
    detail = ' '.join(str(error or '').split())
    if len(detail) > MAX_ERROR_CHARS:
        detail = detail[:MAX_ERROR_CHARS].rsplit(' ', 1)[0] + '…'
    return ' '.join(template.replace(ERROR_PLACEHOLDER, detail).split())


def phrases_from(raw) -> dict[str, str]:
    """Defaults overlaid with stored phrases; an explicit empty string disables one."""
    phrases = dict(DEFAULT_PHRASES)
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(key, str):
                phrases[key] = ' '.join(str(value if value is not None else '').split())
    return phrases


class OpenAITTS:
    """``audio/speech`` on an OpenAI-compatible endpoint; returns MP3 bytes."""

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = TTS_TIMEOUT,
                 client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout,
                            max_retries=0)
        self.client = client

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> 'OpenAITTS | None':
        env = os.environ if env is None else env
        key = (env.get('TTS_API_KEY') or env.get('OPENAI_API_KEY') or '').strip()
        if not key:
            return None
        url = (env.get('TTS_URL') or env.get('OPENAI_BASE_URL') or '').strip() or None
        return cls(key, url)

    def __call__(self, text: str, model: str, voice: str) -> bytes:
        params = {'model': model, 'voice': voice, 'input': text, 'response_format': 'mp3'}
        if not model.startswith('tts-'):  # tts-1/tts-1-hd reject instructions
            params['instructions'] = DEFAULT_INSTRUCTIONS
        response = self.client.audio.speech.create(**params)
        data = response.content if hasattr(response, 'content') else response.read()
        if not data:
            raise RuntimeError('TTS returned an empty clip')
        return bytes(data)


class ClipCache:
    """Synthesised phrases by content hash: memory first, then archive, then TTS."""

    def __init__(self, store=None, tts: Callable[[str, str, str], bytes] | None = None):
        self.store = store
        self.tts = tts
        self.memory: dict[str, bytes] = {}
        self.failed: dict[str, float] = {}  # clip name -> monotonic time of the last TTS error
        self.ephemeral: list[str] = []  # names of persist=False clips, oldest first
        self.lock = threading.Lock()

    @staticmethod
    def name(text: str, model: str, voice: str) -> str:
        digest = hashlib.sha256(f'{model}\0{voice}\0{text}'.encode('utf-8')).hexdigest()
        return f'{digest[:24]}.mp3'

    def data(self, name: str) -> bytes | None:
        with self.lock:
            return self.memory.get(name)

    def get(self, text: str, model: str, voice: str, *, persist: bool = True) -> str:
        """Clip name for ``text``, synthesising and persisting it when needed.

        ``persist=False`` (one-off texts such as error messages) skips the
        archive and keeps only the latest few such clips in memory.
        """
        name = self.name(text, model, voice)
        with self.lock:
            if name in self.memory:
                return name
            failed = self.failed.get(name)
        if failed is not None and time.monotonic() - failed < FAILURE_BACKOFF:
            raise RuntimeError('TTS failed recently for this phrase; skipping')
        data = self._read(name) if persist else None
        if data is None:
            if self.tts is None:
                raise RuntimeError('no TTS configured (set OPENAI_API_KEY or TTS_API_KEY)')
            try:
                data = self.tts(text, model, voice)
            except Exception:
                with self.lock:
                    self.failed[name] = time.monotonic()
                raise
            LOG.info('announce_tts clip=%s bytes=%d text=%r', name, len(data), text)
            if persist:
                self._write(name, data)
        with self.lock:
            self.memory[name] = data
            self.failed.pop(name, None)
            if not persist:
                if name in self.ephemeral:
                    self.ephemeral.remove(name)
                self.ephemeral.append(name)
                while len(self.ephemeral) > MAX_EPHEMERAL_CLIPS:
                    self.memory.pop(self.ephemeral.pop(0), None)
        return name

    def _read(self, name: str) -> bytes | None:
        if self.store is None:
            return None
        try:
            return self.store.read(f'{CLIPS_DIR}/{name}')
        except OSError as exc:
            if not missing(exc):
                LOG.warning('Reading clip %s from the archive failed: %s', name, exc)
        except Exception as exc:
            LOG.warning('Reading clip %s from the archive failed: %s', name, exc)
        return None

    def _write(self, name: str, data: bytes) -> None:
        if self.store is None:
            return
        try:
            self.store.write_atomic(f'{CLIPS_DIR}/{name}', data)
        except Exception as exc:
            LOG.warning('Saving clip %s to the archive failed: %s', name, exc)

    def prefetch(self, texts, model: str, voice: str) -> threading.Thread:
        """Warm the cache on a background thread so the first command is not delayed."""
        def run():
            for text in dict.fromkeys(texts):
                try:
                    self.get(text, model, voice)
                except Exception as exc:
                    LOG.warning('Preparing announcement %r failed: %s', text, exc)
        thread = threading.Thread(target=run, name='announce-prefetch', daemon=True)
        thread.start()
        return thread


def start_clip_server(cache: ClipCache, port: int, host: str = '0.0.0.0') -> ThreadingHTTPServer:
    """Serve ``/clips/<hash>.mp3`` from memory to the Sonos speaker (GET, HEAD, Range)."""

    class Handler(BaseHTTPRequestHandler):
        def _clip(self) -> bytes | None:
            path = self.path.split('?', 1)[0]
            prefix = '/clips/'
            if not path.startswith(prefix) or not CLIP_NAME.fullmatch(path[len(prefix):]):
                return None
            return cache.data(path[len(prefix):])

        def _respond(self, body: bool) -> None:
            data = self._clip()
            if data is None:
                self.send_error(404)
                return
            start, end, status = 0, len(data) - 1, 200
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range') or '')
            if match and (match.group(1) or match.group(2)):
                if match.group(1):
                    start = int(match.group(1))
                    end = min(int(match.group(2)), end) if match.group(2) else end
                else:  # suffix range: last N bytes
                    start = max(0, len(data) - int(match.group(2)))
                if start > end:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{len(data)}')
                    self.end_headers()
                    return
                status = 206
            self.send_response(status)
            self.send_header('Content-Type', 'audio/mpeg')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(data)}')
            self.end_headers()
            if body:
                self.wfile.write(data[start:end + 1])

        def do_GET(self):
            self._respond(True)

        def do_HEAD(self):
            self._respond(False)

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            LOG.debug('clip_http %s ' + format, self.client_address[0], *args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name='clips', daemon=True).start()
    return server


class Announcer:
    """Play the phrase for a tool (or NOT_UNDERSTOOD/FAILED) on the configured Sonos speaker."""

    def __init__(self, clips: ClipCache, speaker_factory=None):
        self.clips = clips
        if speaker_factory is None:
            from .handlers.sonos.sonos import SonosSpeaker
            speaker_factory = SonosSpeaker
        self.speaker_factory = speaker_factory
        self.enabled = False
        self.speaker = None
        self.base_url = ''
        self.model = DEFAULT_MODEL
        self.voice = DEFAULT_VOICE
        self.phrases: dict[str, str] = dict(DEFAULT_PHRASES)

    def configure(self, sonos: Mapping | None) -> None:
        """Apply the ``sonos`` section of ``_config/handlers.json``."""
        sonos = sonos if isinstance(sonos, Mapping) else {}
        section = sonos.get('announce') if isinstance(sonos.get('announce'), Mapping) else {}
        ip = str(sonos.get('speaker_ip') or '').strip()
        self.base_url = str(section.get('base_url') or '').strip().rstrip('/')
        self.model = str(section.get('model') or '').strip() or DEFAULT_MODEL
        self.voice = str(section.get('voice') or '').strip() or DEFAULT_VOICE
        self.phrases = phrases_from(section.get('phrases'))
        self.enabled = bool(sonos.get('enabled') and section.get('enabled') and ip)
        if self.enabled and not self.base_url.startswith(('http://', 'https://')):
            LOG.error('Announcements disabled: sonos.announce.base_url must be the '
                      'http:// address the speaker can reach')
            self.enabled = False
        self.speaker = None
        if not self.enabled:
            LOG.info('Announcements disabled')
            return
        try:
            timeout = max(1.0, min(60.0, float(sonos.get('timeout') or 5)))
        except (TypeError, ValueError):
            timeout = 5.0
        self.speaker = self.speaker_factory(ip, timeout=timeout)
        LOG.info('Announcements enabled: speaker=%s base_url=%s model=%s voice=%s',
                 ip, self.base_url, self.model, self.voice)
        self.clips.prefetch([text for text in self.phrases.values()
                             if text and ERROR_PLACEHOLDER not in text],
                            self.model, self.voice)

    def __call__(self, key: str, *, session: str | None = None,
                 error: str | None = None) -> bool:
        """Play the phrase for ``key``; True when a clip played. Never raises.

        ``error`` fills an ``{error}`` placeholder in the phrase (FAILED).
        """
        if not self.enabled or self.speaker is None:
            return False
        template = self.phrases.get(key) or ''
        text = error_phrase(template, error)
        if not text:
            return False
        try:
            name = self.clips.get(text, self.model, self.voice,
                                  persist=ERROR_PLACEHOLDER not in template)
            played = self.speaker.play_clip(f'{self.base_url}/clips/{name}', text,
                                            after=restore_mode(key),
                                            max_seconds=MAX_CLIP_SECONDS)
        except Exception as exc:
            LOG.warning('announce_failed session=%s key=%s error=%s', session or '-', key, exc)
            return False
        LOG.info('announce session=%s key=%s played=%s', session or '-', key, played)
        return bool(played)
