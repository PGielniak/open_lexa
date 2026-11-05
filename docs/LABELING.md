# Etykietowanie nagrań — pierwotny plan projektu

**Status:** poniższy dokument zachowuje historyczne założenia i szkice etapów.
Aktualny opis działania (Whisper Base/Small, NLU, LLM i handlery Sonos/Spotify)
znajduje się w [HANDLERS.md](HANDLERS.md). Nie traktuj wzmianki o Small/Medium
ani ręcznym etykietowaniu jako bieżących ograniczeń systemu.

Pierwotny cel: transkrypcje z Hailo Whisper-Small wymagały porównania. Odbiornik wysyła
więc **każde** nagranie równolegle do lokalnego Whispera i do chmury
(`gpt-transcribe`). GPT automatycznie wypełnia „poprawną” transkrypcję, a
człowiek lub osobny LLM dodaje **intencję i encje** (LLM może rozszerzać słownik).
System uczy się na nagraniach poprawionych ręcznie w admin UI.

```
ESP32 ─UDP─> receiver ─> STT: local Whisper + GPT (parallel) ─> NLU (intencja + encje) ─> handler / „sorry ale nie rozumiem”
                │                   ▲
                │      _lexicon/lexicon.json (SMB, przeładowanie bez restartu)
                ▼                   │
   SMB \\files\open-lexa  <──── admin UI ── SQLite (PVC)
   RRRR-MM-DD/<stem>.wav/.json    │  ingest → odsłuch → GPT jako „poprawna” → intencja + encje
                                  └─ eksport: lexicon.json + dataset JSONL
```

Zasady:

- **SMB** pozostaje magazynem surowych danych (WAV + JSON od odbiornika, nigdy nie nadpisywany).
- **SQLite** jest źródłem prawdy dla etykiet. Leży na PVC admina, nie na SMB
  (blokady SQLite przez SMB/CIFS są niewiarygodne).
- **Odbiornik nie zależy od admina w runtime**: czyta tylko wyeksportowany
  `lexicon.json` z SMB i trzyma ostatnią poprawną wersję w pamięci.
- Wolimy „nie rozumiem” niż fałszywie wykonaną komendę: progi i margines dobierane
  na danych (etap 2), nie na oko.

## Model intencji i encji

Intencje tworzą **drzewo**. Liść jest konkretną akcją, rodzic ogólną; jeśli
klasyfikator nie jest pewny liścia, ale jest pewny rodzica, wynikiem jest rodzic
i handler wybiera domyślne źródło.

```
play_music            „włącz muzykę”               → handler: domyślne/ostatnie źródło
├── play_spotify      „puść drum and bass na spotify”  + search_query="drum and bass"
└── play_radio        „włącz RMF”                  + station=RMF24
stop                  „stop”, „wyłącz muzykę”
volume_up / volume_down / volume_set              + level=30
lights_on / lights_off                            + room=kuchnia
unknown               sensowne polecenie, którego jeszcze nie obsługujemy (kandydaci na nowe intencje)
noise                 szum, urwane nagranie, „To”      (nieakcyjne)
not_for_me            rozmowa w tle, fałszywe „Alexa” (nieakcyjne)
```

Typy encji (`entity_types.kind`):

| Typ | Rodzaj | Przykład | Normalizacja |
| --- | --- | --- | --- |
| `search_query` | `text` | „dram end bejs” → „drum and bass”, „pendalam” → `Pendulum` | słownik nazw własnych (artyści, zespoły, gatunki); bez trafienia surowy tekst |
| `station` | `enum` | „rmf”, „erem ef” → `RMF24` | słownik synonimów |
| `level` | `number` | „trzydzieści”, „30” → `30` | liczebniki PL → int |
| `room` | `enum` | „w kuchni” → `kuchnia` | słownik synonimów |
| `source` | `enum` | „ze spotify” → `spotify` | słownik synonimów |

## Schemat SQLite

Migracje przez `PRAGMA user_version`; tryb WAL; jedna replika admina.

```sql
PRAGMA foreign_keys = ON;

-- Słownik intencji (drzewo przez parent_id).
CREATE TABLE intents (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,                 -- 'play_spotify'
    parent_id     INTEGER REFERENCES intents(id),       -- → 'play_music'
    description   TEXT,
    is_actionable INTEGER NOT NULL DEFAULT 1,           -- 0: noise, not_for_me, unknown
    enabled       INTEGER NOT NULL DEFAULT 1
);

-- Frazy kanoniczne i szablony; {nazwa_encji} oznacza slot.
CREATE TABLE intent_phrases (
    id        INTEGER PRIMARY KEY,
    intent_id INTEGER NOT NULL REFERENCES intents(id) ON DELETE CASCADE,
    phrase    TEXT NOT NULL,                            -- 'puść {search_query} na spotify'
    UNIQUE (intent_id, phrase)
);

CREATE TABLE entity_types (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,                   -- 'search_query'
    kind        TEXT NOT NULL CHECK (kind IN ('text', 'enum', 'number')),
    description TEXT
);

-- Jakie encje przyjmuje intencja (lista w UI + walidacja).
CREATE TABLE intent_entity_types (
    intent_id      INTEGER NOT NULL REFERENCES intents(id) ON DELETE CASCADE,
    entity_type_id INTEGER NOT NULL REFERENCES entity_types(id) ON DELETE CASCADE,
    required       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (intent_id, entity_type_id)
);

-- Wartości kanoniczne encji i ich synonimy (także typowe przekręcenia STT).
-- Dla 'enum' to zamknięta lista; dla 'text' (search_query) słownik nazw własnych:
-- trafienie ≥ próg zwraca wartość kanoniczną, inaczej zostaje surowy tekst.
CREATE TABLE entity_values (
    id             INTEGER PRIMARY KEY,
    entity_type_id INTEGER NOT NULL REFERENCES entity_types(id) ON DELETE CASCADE,
    value          TEXT NOT NULL,                       -- 'RMF24'
    UNIQUE (entity_type_id, value)
);
CREATE TABLE entity_synonyms (
    id              INTEGER PRIMARY KEY,
    entity_value_id INTEGER NOT NULL REFERENCES entity_values(id) ON DELETE CASCADE,
    synonym         TEXT NOT NULL,                      -- znormalizowany: 'erem ef'
    source          TEXT NOT NULL DEFAULT 'manual' CHECK (source IN ('manual', 'learned')),
    recording_id    INTEGER REFERENCES recordings(id) ON DELETE SET NULL,  -- skąd 'learned'
    UNIQUE (entity_value_id, synonym)
);

-- Jedno nagranie = jeden wiersz.
CREATE TABLE recordings (
    id                     INTEGER PRIMARY KEY,
    stem                   TEXT NOT NULL UNIQUE,        -- '1793628774338692713-9db3213c'
    audio_path             TEXT NOT NULL,               -- '2026-11-02/<stem>.wav' względem udziału;
                                                        -- UI linkuje /recordings/{id}/audio (stream z SMB)
    recorded_at            TEXT NOT NULL,               -- ISO 8601 UTC z sidecara
    duration_seconds       REAL,
    session                TEXT,
    stt_backend            TEXT,                        -- 'hailo'
    stt_model              TEXT,                        -- 'Whisper-Small.hef' (do porównania Small/Medium)
    language               TEXT,
    original_transcript    TEXT,                        -- NULL, gdy STT zawiodło
    stt_error              TEXT,
    -- Tryb dual: oba wyniki i informacja, który jest automatyczną poprawną.
    transcripts            TEXT,                        -- JSON: {local: {...}, cloud: {...}}
    corrected_source       TEXT,                        -- 'cloud' | 'local' | ...
    -- Co odbiornik zrobił w chwili nagrania (do oceny skuteczności, nie edytowalne):
    predicted_intent       TEXT,
    predicted_entities     TEXT,                        -- JSON
    predicted_score        REAL,
    predicted_source       TEXT,                        -- 'rules' | 'lexicon' | 'canonical'
    -- Etykieta człowieka (edytowalne w admin UI):
    corrected_transcript   TEXT,
    intent_id              INTEGER REFERENCES intents(id),
    status                 TEXT NOT NULL DEFAULT 'new'
                           CHECK (status IN ('new', 'labeled', 'skipped')),
    labeled_at             TEXT,
    notes                  TEXT,
    ingested_at            TEXT NOT NULL
);
CREATE INDEX recordings_status ON recordings (status, recorded_at);
CREATE INDEX recordings_intent ON recordings (intent_id);

-- Encje oznaczone w poprawionej transkrypcji.
CREATE TABLE recording_entities (
    id              INTEGER PRIMARY KEY,
    recording_id    INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    entity_type_id  INTEGER NOT NULL REFERENCES entity_types(id),
    value           TEXT NOT NULL,                      -- 'drum and bass' / '30' / 'RMF24'
    entity_value_id INTEGER REFERENCES entity_values(id),   -- dla enum
    span_start      INTEGER,                            -- offsety znaków w corrected_transcript
    span_end        INTEGER,
    original_fragment TEXT,                             -- jak to usłyszało STT: 'dram end bejs'
    CHECK ((span_start IS NULL) = (span_end IS NULL))
);
```

Przykład etykiety:

| Pole | Wartość |
| --- | --- |
| `original_transcript` | „Plej dram end bejs on spotyfaj” |
| `corrected_transcript` | „Play drum and bass on spotify” |
| `intent` | `play_spotify` (rodzic `play_music`) |
| `recording_entities` | `search_query`=„drum and bass”, span 5–18 |

Słownik `search_query` uczy się z etykiet: przy zapisie encji z `original_fragment`
(UI podpowiada go dopasowaniem `rapidfuzz` do oryginalnej transkrypcji) i wartością
kanoniczną admin dodaje brakujący synonim `learned` („dram end bejs” → „drum and bass”).
Słownik można też zasilić ręcznie listą ulubionych wykonawców (później: import ze Spotify).

Spany są opcjonalne, ale cenne: z nich powstają szablony („play {search_query} on spotify”)
i dane do przyszłego modelu NER. Po edycji `corrected_transcript` admin przelicza spany
wyszukując `value` w nowym tekście; jeśli go nie znajdzie, pokazuje ostrzeżenie zamiast
zapisywać zepsuty offset.

## Komponenty

### Admin UI (`open-lexa-admin`, nowy Deployment)

- FastAPI + Jinja2 + htmx, stdlib `sqlite3`; pakiet `admin-dashboard/` (zależy od pakietu receivera).
- **Ingest**: co 60 s (i przyciskiem) skanuje `RRRR-MM-DD/*.json` na SMB, dodaje nowe
  `stem` (idempotentnie, `INSERT OR IGNORE`). Odbiornik nic nie wie o adminie.
- **Lista**: filtr `status`, intencja, data, model STT; sortowanie od najnowszych.
- **Szczegóły**: `<audio src="/recordings/{id}/audio">` (stream z SMB), oryginalna
  transkrypcja, pole poprawnej transkrypcji (domyślnie wypełnione oryginałem),
  wybór intencji z drzewa, encje dozwolone dla intencji (zaznaczenie fragmentu tekstu
  lub wpisanie wartości), przyciski „zapisz i następne” / „pomiń”. Skróty klawiszowe.
- **Słowniki**: edycja intencji, fraz/szablonów, wartości i synonimów encji.
- **Nagraj**: nagranie z mikrofonu w przeglądarce (wymaga HTTPS albo localhost) lub
  wgranie pliku audio. Przeglądarka konwertuje dźwięk do WAV mono 16 bit 16 kHz (maks.
  12 s, jak z urządzenia); serwer przyjmuje tylko ten format. Admin transkrybuje nagranie
  według *Ustawień*, zapisuje parę `.wav` + `.json` w archiwum jak odbiornik (z polem
  `origin`: `browser`/`upload`) i od razu otwiera je do oznaczenia.
- **Ustawienia**: domyślnie tryb **dual** — lokalne Whisper API **i** chmura
  (URL + klucz API) są wywoływane równolegle; dostępne są też tryby pojedyncze
  (`local`, `cloud`) jako awaryjne. Model, język, klucz API. Eksport zapisuje je
  do `_config/stt.json`; odbiornik czyta ten plik i przełącza backend bez
  restartu. Klucz API nie jest nigdy wyświetlany w UI.
- **Automatyczna poprawna transkrypcja**: w trybie dual sidecar zawiera oba
  wyniki (`transcripts.local`, `transcripts.cloud`). Admin ustawia
  `corrected_transcript` na tekst GPT i pokazuje oba do porównania; użytkownik
  zatwierdza intencję/encje, a tekst poprawia tylko w razie potrzeby.
- **Eksport** (po każdym zapisie, z debounce):
  - `_lexicon/lexicon.json` — intencje, szablony, przykłady (znormalizowany
    `original_transcript` → intencja + encje), synonimy; pole `version`;
    zapis do pliku tymczasowego + rename, żeby odbiornik nie przeczytał połowy.
  - `_dataset/labels.jsonl` — `audio_path`, oryginał (lokalny Whisper),
    `local_transcript`, `cloud_transcript`, poprawka, `corrected_source`,
    intencja, encje (format pod fine-tuning Whispera / ewaluację WER).
- k3s: PVC Longhorn 1 Gi na SQLite, `replicas: 1`, `strategy: Recreate`, konto SMB
  `open-lexa` (Secret `open-lexa-smb`), dostęp tylko z Tailscale/LAN + basic auth.
  Backup: SQLite jest mały, cron `sqlite3 .backup` na SMB obok datasetu.

### NLU w odbiorniku (zastępuje regexy w `commands.py`)

1. **Normalizacja PL**: małe litery, bez interpunkcji, zdjęte ogonki, ó→u, rz→ż→z,
   ch→h; dodatkowo wariant bez spacji (Whisper myli granice słów: „rozpotify”).
2. **Intencja (kNN)**: `rapidfuzz` — max z `token_set_ratio` i `ratio` bez spacji —
   względem przykładów z lexiconu i fraz kanonicznych (sloty zastąpione wildcardem).
   Akceptacja, gdy wynik ≥ próg **i** przewaga nad najlepszą inną intencją ≥ margines
   **i** tekst ma minimalną długość. Gdy liść niepewny, a suma dla rodzica pewna →
   rodzic. Wygrana `noise`/`not_for_me` → nic nie robimy.
3. **Encje**:
   - `enum`: rozmyte wyszukiwanie synonimów w n-gramach tekstu (`partial_ratio`);
   - `number`: liczebniki PL + cyfry;
   - `text`: najlepiej pasujący szablon intencji, kotwice przed/po slocie dopasowane
     rozmyto, slot = tekst pomiędzy (z surowej transkrypcji).
4. **Wynik**: `Decision(intent, parent, entities, score, source)`; `command` zostaje
   jako alias `intent`, więc archiwum i testy działają dalej. Archiwum dostaje
   `entities`, `score`, `source`, `nearest_example`.
5. **Przeładowanie**: co 60 s sprawdza mtime `_lexicon/lexicon.json`; błąd odczytu
   lub walidacji → zostaje poprzednia wersja + `WARNING`. Bez lexiconu działa na
   frazach kanonicznych wbudowanych w obraz.

### Ewaluacja (`admin-dashboard/tools/evaluate_nlu.py`)

Na etykietach z SQLite (lub `labels.jsonl`), leave-one-out: trafność intencji,
odsetek fałszywych wyzwoleń (akcja dla `noise`/`not_for_me`), F1 encji, WER STT per
model. Wynik: rekomendowany próg i margines. Ten sam zbiór posłuży do porównania
Whisper-Small z Medium po aktualizacji HailoRT 5.5.0.

## Etapy

| # | Zakres | Gotowe, gdy |
| --- | --- | --- |
| 0 | Schemat + migracje, seed intencji/encji z tego dokumentu, normalizacja PL | testy schematu i normalizacji |
| 1 | Admin UI: ingest, lista, szczegóły z odsłuchem, edycja etykiet i słowników, eksport; Deployment na k3s | 4 obecne nagrania oznaczone w UI, `lexicon.json` i `labels.jsonl` na SMB |
| 2 | `admin-dashboard/tools/evaluate_nlu.py` | raport dla ≥ 50 oznaczonych nagrań |
| 3 | NLU w odbiorniku z przeładowaniem lexiconu, rozszerzone archiwum | „Wąc muzyk, rozpotify” → `play_spotify` po oznaczeniu; `noise` nie wyzwala akcji |
| 4 | Handlery (Sonos/Spotify/radio — `sonos_control/`), opcjonalny LLM dla niepewnych, Whisper-Medium | poza tym planem |

## Decyzje

- Dostęp do admina: basic auth za Ingressem Tailscale (2026-11-02).
- Ustawienia STT z admina trafiają do odbiornika przez plik na SMB (jak lexicon), nie przez API admina ani Secret K8s: odbiornik nie zależy od admina w runtime (2026-11-02).
- `search_query` przechodzi przez słownik nazw własnych z synonimami uczonymi
  z etykiet (2026-11-02).
- Początkowa lista intencji i encji jest propozycją; edytowalna w admin UI.
- Domyślny STT to **dual**: lokalny Whisper i cloudowy GPT transcribe na każdym
  nagraniu. GPT jest automatyczną poprawną transkrypcją, więc etykietowanie
  ogranicza się do intencji i encji (2026-11-02).
