# LLM i narzędzia MCP

Receiver wysyła każdą niepustą transkrypcję do modelu Chat Completions
zgodnego z OpenAI, razem z nazwami, opisami i schematami narzędzi włączonych
serwerów MCP. Model może wybrać narzędzie; odbiornik waliduje argumenty i
wykonuje najwyżej jedno wywołanie dla nagrania. Brak LLM, niepoprawna odpowiedź
lub błąd serwera nie uruchamiają żadnego awaryjnego NLU. Wynik znajduje się w
polu `agent` sidecara JSON i w szczegółach nagrania w panelu.

## Konfiguracja

W panelu **Ustawienia → LLM** podaj URL, model z obsługą tool calling i klucz,
następnie włącz LLM w odbiorniku. Panel publikuje ustawienia w
`_config/llm.json` na SMB. Bez SMB można użyć `LLM_MODEL`, `LLM_API_KEY`,
`LLM_URL` i `LLM_TIMEOUT`. `--llm-mode` przypina konfigurację środowiskową;
historyczny tryb `fallback` jest traktowany jak `always`.

Sekcja **Sonos / Spotify** konfiguruje wbudowany serwer narzędzi. W sekcji
**Zewnętrzne serwery MCP** można dodać URL Streamable HTTP i opcjonalny token
Bearer. Konfiguracja jest publikowana w `_config/handlers.json`, np.:

```json
{"mcp_servers": {"home": {"enabled": true, "url": "http://home:8000/mcp"}}}
```

Podłączaj tylko zaufane serwery: ich narzędzia mogą wykonywać działania, a ich
opisy trafiają do modelu. Zabezpiecz dostęp do panelu i udziału SMB — pliki
konfiguracyjne mogą zawierać klucze. Klient HTTP obsługuje podstawowe metody
`initialize`, `tools/list` i `tools/call`; zgodność z konkretnym zewnętrznym
serwerem należy sprawdzić w danym wdrożeniu.

## Sonos i Spotify

Podaj adres IP głośnika i ustaw `default_favorite` na nazwę osobistej playlisty
w Ulubionych Sonos. Polecenie „włącz Spotify” lub „włącz muzykę” odtworzy ją;
bez niej jest pomijane. Ulubione będące playlistą lub albumem
(`x-rincon-cpcontainer:`) zastępują kolejkę i grają od pierwszego utworu —
Sonos odrzuca je jako bezpośrednie źródło (UPnP 714 „Illegal MIME-Type”).
Spotify musi być połączone w aplikacji Sonos.
Wyszukiwanie katalogu po nazwie wymaga dodatkowo Spotify Client ID i Secret.
Obsługiwane są odtwarzanie i kolejka, pauza/wznowienie, następny/poprzedni
utwór oraz głośność ograniczona przez `volume_max`.

Radio (TuneIn) ma trzy narzędzia MCP:

| Narzędzie | Przykład | Działanie |
|---|---|---|
| `play_default_radio` | „włącz radio” | stacja z `default_radio` (domyślnie Radio Zet) |
| `play_radio` | „włącz RMF FM” | najpierw Ulubione Sonos (dokładna nazwa, potem fragment nazwy stacji radiowej), potem wyszukiwarka TuneIn |
| `play_random_radio` | „włącz losowe radio” | losuje z `radio_stations` i stacji radiowych w Ulubionych, pomijając aktualnie graną |

Wyszukiwanie TuneIn nie wymaga klucza. Akceptowane jest tylko dokładne lub
zawierające się dopasowanie nazwy, więc stację, której TuneIn nie zna (np.
Trójkę), dodaj do Ulubionych Sonos pod nazwą, którą wypowiadasz. Sonos
przekierowuje odtwarzanie na konto TuneIn połączone w aplikacji Sonos.

## Komunikaty głosowe na Sonos

Gdy LLM wybierze narzędzie, receiver najpierw odtwarza na głośniku krótką frazę
(np. „Okej, uruchamiam Spotify.”), a dopiero potem wykonuje akcję MCP. Gdy LLM
nie wybierze narzędzia, odtwarza „Sorry, nie zrozumiałam.”. Gdy STT, zapytanie do
LLM, walidacja wywołania albo samo narzędzie się nie powiedzie (status `failed`
lub `skipped`), odtwarza frazę `failed` — domyślnie „Przepraszam, coś poszło
nie tak, spróbuj ponownie.”. Fraza może zawierać `{error}`, np. „Przepraszam,
błąd: {error}” — zostanie on zastąpiony treścią błędu (max 160 znaków); takie
klipy są syntetyzowane przy każdym nowym błędzie i trzymane tylko w pamięci
(ostatnie 16), bez zapisu na SMB. Pusta transkrypcja (szum, fałszywe wybudzenie) nie wywołuje
komunikatu. Błąd TTS albo głośnika jest logowany (`announce_failed`) i nigdy nie
blokuje samej akcji.

Konfiguracja: **Ustawienia → Sonos / Spotify → Komunikaty głosowe**, zapisywana
w `_config/handlers.json` jako `sonos.announce`:

```json
{"sonos": {"enabled": true, "speaker_ip": "192.168.1.42",
           "announce": {"enabled": true, "base_url": "http://192.168.1.10:8081",
                        "model": "gpt-4o-mini-tts", "voice": "nova",
                        "phrases": {"play_spotify": "Okej, uruchamiam Spotify.",
                                    "pause": "", "not_understood": "Sorry, nie zrozumiałam."}}}}
```

- Frazy są syntetyzowane raz przez `audio/speech` (klucz `OPENAI_API_KEY`
  receivera, opcjonalnie `TTS_API_KEY`/`TTS_URL`), cache'owane w pamięci i w
  `_clips/<hash>.mp3` na SMB. Hash obejmuje tekst, model i głos, więc zmiana
  frazy generuje nowy plik.
- Sonos pobiera klip sam, przez HTTP. Receiver serwuje `/clips/` na porcie
  `OPEN_LEXA_CLIP_PORT` (8081); Service `open-lexa-clips` (ServiceLB) wystawia
  go na każdym nodzie. `base_url` to `http://<LAN IP dowolnego noda>:8081` —
  ten sam adres, którego używa ESP32.
- Przed klipem zapisywany jest snapshot (SoCo). Po klipie: narzędzia
  uruchamiające odtwarzanie (`play_*`) nie przywracają poprzedniego źródła;
  `pause`/`stop` przywracają źródło i pozycję bez wznawiania; pozostałe
  (głośność, następny/poprzedni, kolejka, „nie zrozumiałam”) przywracają
  wszystko. Źródeł, których snapshot nie umie wznowić (Spotify Connect,
  AirPlay, TV, line-in), komunikat nie przerywa — jest wtedy pomijany.
- Komunikat zatrzymuje muzykę na czas trwania frazy (~1–2 s).

## Dane i migracja

Panel zapisuje poprawione transkrypcje do `_dataset/transcripts.jsonl`. Nie
nadpisuje dawnego `_dataset/labels.jsonl` ani `_lexicon/lexicon.json`; nie są
one już używane przez receiver. Historyczne tabele i etykiety w istniejącej
bazie SQLite pozostają nienaruszone. Pliki na SMB można usunąć ręcznie po
wykonaniu kopii zapasowej i potwierdzeniu, że nie są potrzebne.

Porównanie modeli Whispera na ręcznie sprawdzonych transkrypcjach:

```sh
uv run python whisper-api/tools/compare_models.py \
  --db /path/to/labels.sqlite3 --url http://open-lexa-whisper:8080/v1
```
