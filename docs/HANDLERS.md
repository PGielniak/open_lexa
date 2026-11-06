# Handler plugins, LLM labels and Sonos playback

The receiver decides *what* an utterance means before executing any action:

1. It polls `_lexicon/lexicon.json` from the archive and matches the Polish
   transcript against phrases and labeled examples. Intents, entity values,
   match score and source go into each recording sidecar. Without the lexicon
   it falls back to the original regex rules (which still do not operate
   devices).
2. It polls `_config/handlers.json` separately. Only enabled, installed
   handlers receive matching actionable intents. Execution results go into
   `execution` in the sidecar, independently of STT and archival failures.
3. Admin **Ustawienia → Sonos / Spotify** publishes the Sonos plugin settings;
   **Zainstalowane handlery** lets you disable/re-enable any installed plugin.
   The receiver needs no admin HTTP connection; changes take effect after
   the next settings poll (default 30 s).

## Add or remove a handler

Each plugin is a Python package in `udp-receiver/src/open_lexa/handlers/<name>/`
with a `handler.py` exporting `create()`. The factory returns a
`CommandHandler` subclass. `id` and `intents` declare its route; `configure`
receives the plugin's `_config/handlers.json` section; `handle` receives a
`CommandRequest(intent, entities, text, score, source, stem, session)` and
returns a `HandlerResult`. A plugin may return `handled`, `skipped`, or
`failed`. To add an independently configurable command, create another
folder; the dispatcher discovers it automatically. To unplug it, toggle it in
admin Settings. Removing a folder requires rebuilding the receiver image.

```python
# handlers/my_device/handler.py
from open_lexa.handlers.base import CommandHandler, HandlerResult

class MyDevice(CommandHandler):
    id = 'my_device'
    intents = frozenset({'my_intent'})

    def handle(self, request):
        value = request.entities.get('my_parameter')
        if not value:
            return HandlerResult.skipped('missing parameter')
        # Send a bounded-time request to your device here.
        return HandlerResult.handled('Done')

def create():
    return MyDevice()
```

One handler claims each intent; lowest `priority`, then alphabetic id wins on
conflicts. A new intent created in the LLM dictionary is *not* automatically
executable: install a handler claiming it. Non-actionable intents like `noise`
and `not_for_me` are never dispatched.

## LLM annotation

**Ustawienia → LLM** accepts a separate OpenAI-compatible Chat Completions
URL, model and API key. Click **Oznacz nieoznaczone nagrania przez LLM** on
the recordings list, **Oznacz to nagranie przez LLM** on a detail page, or
enable automatic labeling. The LLM receives the corrected transcript (or
local STT fallback), existing dictionary and examples. It can create intents,
entity types and values and assign entities to the recording. It does not
receive WAV audio. Annotations remain editable; per-recording provenance,
model, attempt count and errors live in SQLite and the dataset export. Failed
items retry up to three times. Only `status=new` rows are auto-annotated, and
editing a row while a request runs prevents the LLM result from overwriting it.
LLM annotation has its own background thread so a slow model does not delay
SMB ingest/export. The LLM key stays in the admin's SQLite volume; it reaches
the archive only as part of `_config/llm.json` while the receiver mode is on
(see below). Protect that volume, make backups, and restrict admin access;
keep the archive SMB share private as it stores STT, Spotify and (when the
receiver LLM is enabled) LLM keys.

## LLM recognition on the receiver

The receiver can also ask an LLM right after each transcription, before
dispatching to handlers. The single source of truth is **Ustawienia → LLM**:
besides URL, model and key it has a *receiver mode* (off / fallback / always).
Saving publishes `_config/llm.json` (key included only while the mode is on)
and the receiver polls it like the STT settings, so no second key entry in
`.env` is needed. Without an archive the LLM_* environment variables work the
same way: `LLM_MODEL` and `LLM_API_KEY` (Secret), optionally `LLM_URL` and
`LLM_TIMEOUT`. Precedence: `--llm-mode` (reads no file), `_config/llm.json`,
environment. `LLM_INTENT_MODE=always` decides intent and entities for every
transcript (the default once key and model are set), `fallback` asks only when
the lexicon and regex rules recognised nothing, and `off` disables it. An
invalid or missing file never drops a running client (warning in the log),
matching the STT settings behaviour. The prompt embeds the
published lexicon — intents with their phrases and entity types — and the
answer is validated against it: the receiver's model may only reuse existing
labels, never create intents or entity types (that stays the admin annotation
job). Without an SMB archive the prompt falls back to the regex-rule command
names. `noise` and `not_for_me` stay non-actionable, exactly like in the
lexicon. Any failure (network, invalid JSON, unknown label) is logged, counted
in `/healthz` under `llm`, and the lexicon/rules decision stands, so an
unreachable endpoint never breaks the voice pipeline. Recognised entities are
canonical values (`number` entities are normalised to digits), which is what
handlers like Sonos expect.

## Sonos / Spotify

Set the Sonos speaker's **LAN IP** (discovery from k3s pods is unreliable),
then test the connection before enabling playback. Link Spotify in the Sonos
app first. A Sonos favorite/playlist can be played without Spotify Web API
credentials. Named catalog search requires a Spotify developer app's client
ID and secret, used with client-credentials search, a market (default `PL`)
and a linked Sonos Spotify account for actual playback. Spotify private
playlists work best as Sonos favorites. The portal's playback controls and
the voice handler use the same SoCo adapter. A group routes transport/queue
commands to its coordinator. SoCo must be able to reach the speaker's LAN IP
from the receiver pod; verify networking/firewall before debugging NLU.

Add your personal Spotify playlist to **Sonos Favorites**, then enter its
exact name in **Ustawienia → Sonos / Spotify → Moja domyślna playlista**
(`default_favorite`). Client-credentials Spotify search cannot access a user's
private playlists; this favorite is the way to play a personal/private one.
Bare “włącz muzykę”, “włącz Spotify” or “puść moją playlistę” play that favorite.
Without a configured favorite, the command is skipped instead of resuming
unrelated audio. “Wznów” still resumes playback.

| Example voice command | Action |
| --- | --- |
| “Puść Drum and Bass na Spotify” | Random nonempty playlist from the top 10 Spotify genre results |
| “Puść playlistę rock” | Random rock playlist |
| “Puść muzykę gatunku synthwave” | Random playlist for any explicitly named genre |
| “Puść artystę Pendulum” / “Puść Pendulum na Spotify” | Artist's “This Is …” playlist when available, otherwise a named artist playlist; exact artist-name recognition for the latter form |
| “Puść utwór Bohemian Rhapsody na Spotify” | Search tracks only and play the best playable result |
| “Dodaj utwór Bohemian Rhapsody do kolejki” | Append to the Sonos queue without interrupting playback |
| “Dodaj do kolejki Drum and Bass” | Append a random genre playlist to the queue |
| “Następna piosenka”, “poprzednia piosenka”, “pauza”, “wznów”, “stop” | Transport controls |
| “Podgłośnij o pięć”, “przycisz o 10”, “ustaw głośność na 20 procent” | Relative or absolute volume, bounded by `volume_max` |

Genre detection uses an explicit list in `handlers/sonos/handler.py`; other
queries use Spotify catalog search. An explicit playlist/album request does
not silently fall back to a track. Sonos group transport and queue operations
target the coordinator. A failed action is archived as `execution` metadata
without suppressing subsequent recordings. Redeploy both admin (to export the
new lexicon phrases) and receiver (for the handler), and verify Spotify is
linked in the Sonos app. Integration with live Sonos hardware is not covered
by the mocked unit tests.

## Whisper model comparison

The Whisper image bundles the checksummed multilingual HailoRT 5.4.0 Base
and Small HEFs. The API loads one model on its inference thread and switches
between jobs from the admin's STT model selection; `whisper-1` remains a
legacy alias for Base. The admin shows available and last loaded models.
The selected model must be tested against real Polish recordings, not assumed
to be better. From a shell that can access both admin SQLite and the archive:

```sh
uv run python whisper-api/tools/compare_models.py \
  --db /path/to/labels.sqlite3 --url http://open-lexa-whisper:8080/v1
```

The script uses only human-reviewed labels, transcribes the same WAV with
both models, then prints mean word error rate and response latency. Tests
mock Hailo and Sonos; real HEF loading and playback still require hardware.
