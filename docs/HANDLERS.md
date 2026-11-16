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
bez niej jest pomijane. Spotify musi być połączone w aplikacji Sonos.
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
