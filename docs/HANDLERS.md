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
SMB ingest/export. The LLM key stays in the admin's SQLite volume and is not
published to SMB. Protect that volume, make backups, and restrict admin
access; keep the archive SMB share private as it stores STT and Spotify keys.

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

The plugin maps `play_spotify`, `play_music`, `play_radio`, `stop`, `pause`,
`resume`, `next_track`, `previous_track`, and volume intents to SoCo. Spotify
search returns a URI and SoCo's ShareLink plugin queues it; artist results
are resolved to playable tracks. A failed action is archived as `execution`
metadata without suppressing subsequent recordings.

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
