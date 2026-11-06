# Open Lexa — Arduino Nano ESP32 + Atomic Echo Base

**„Alexa” lokalnie na ESP32 → cichy sygnał → nagranie polecenia → UDP → odbiornik (host lub k3s) → API STT (lokalny Whisper + chmurowy GPT transcribe równolegle) → tekst w konsoli.**

Repozytorium to monorepo z czterema komponentami. Każdy ma `src/` (kod), `build/` (Dockerfile + pipeline Tekton; dla firmware skrypty Arduino) i `deploy/` (manifesty Kubernetes):

| Komponent | Kod | Obraz / wynik | Wdrożenie |
| --- | --- | --- | --- |
| [`firmware/`](firmware) | szkic Arduino Nano ESP32 | `firmware/build/arduino.sh` → `.bin` | wgranie przez USB |
| [`udp-receiver/`](udp-receiver) | pakiet `open_lexa` | `ghcr.io/pgielniak/open-lexa` | `kubectl apply -k udp-receiver/deploy` |
| [`whisper-api/`](whisper-api) | pakiet `open_lexa_whisper` (Hailo-10H) | `ghcr.io/pgielniak/open-lexa-whisper-hailo` | `kubectl apply -k whisper-api/deploy` |
| [`admin-dashboard/`](admin-dashboard) | pakiet `open_lexa_admin` | `ghcr.io/pgielniak/open-lexa-admin` | `kubectl apply -k admin-dashboard/deploy` |

Pakiety Pythona tworzą uv workspace (`pyproject.toml` w root). Admin korzysta z pakietu receivera (archiwum SMB, ustawienia STT, komendy); Whisper API jest niezależny. Budowanie obrazów: [`docs/BUILD.md`](docs/BUILD.md), wdrożenie: [`docs/KUBERNETES.md`](docs/KUBERNETES.md).

Główny firmware to szkic Arduino: [`firmware/src/OpenLexaNano/OpenLexaNano.ino`](firmware/src/OpenLexaNano/OpenLexaNano.ino). Można go kompilować i wgrywać w Arduino IDE lub Arduino CLI. Nie wymaga ESPHome ani Home Assistanta. WakeNet9 z modelem Alexa jest osadzony w firmware, bez osobnego wgrywania modelu/partycji. Klucze API i modele STT pozostają po stronie odbiornika.

## Połączenia

Zweryfikowana konfiguracja z 2026-07-13. Firmware zakłada dokładnie to przypisanie pinów.

| Sygnał Atomic Echo Base | Oznaczenie na Nano | GPIO ESP32-S3 |
| --- | --- | --- |
| 3V3 | 3V3 | — |
| GND | GND | — |
| SDA | A4 | 11 |
| SCL | A5 | 12 |
| SCK / BCLK | D2 | 5 |
| LRCK / WS | D3 | 6 |
| DOUT z bazy: mikrofon → ESP32 | D4 | 7 |
| DIN bazy: ESP32 → głośnik | D5 | 8 |

`Wire.begin(A4,A5)` używa oznaczeń Arduino. Sterownik I2S ESP-IDF dostaje **surowe GPIO**, w kolejności mikrofon=7, WS=6, głośnik=8, BCLK=5. W IDE pozostaw **Pin Numbering → By Arduino pin (default)**.

Zachowany tor audio z działającego szkicu: **44,1 kHz, stereo 16-bit**, inicjalizacja kodeka ES8311 jak w M5Atomic-EchoBase. Mikrofon trafia następnie przez filtr dolnoprzepustowy i resampling do **mono PCM16, 16 kHz**, którego używają WakeNet i odbiornik STT. Także podczas nasłuchu podtrzymywane są zegary I2S.

**Ważne dla tej bazy:** firmware nie używa `M5EchoBase::setMute(true)`. Ekspander pozostaje w stanie zasilającym mikrofon, a wyciszany jest wyłącznie DAC przez rejestr kodeka. Zapobiega to opisanej w Twojej dokumentacji sytuacji z próbkami `0xFFFF` po wyłączeniu mikrofonu.

## 1. Konfiguracja

W tej kopii projektu są już `.venv`, `.env` oraz `firmware/src/OpenLexaNano/Config.h`. Konfiguracja Arduino została utworzona z wcześniejszego `secrets.yaml` i ma token zgodny z `.env`.

Uzupełnij lokalnie:

- `firmware/src/OpenLexaNano/Config.h`: `WIFI_SSID`, `WIFI_PASSWORD`, `RECEIVER_IP`.
- `.env`: `OPENAI_API_KEY`, `STT_HTTP_URL` (local Whisper API) i ewentualnie `STT_BACKEND` (domyślnie `dual`, gdy oba endpointy są ustawione; inaczej `openai`).
- `RECEIVER_TOKEN` w `Config.h` musi być identyczny z `OPEN_LEXA_TOKEN` w `.env` / Secret k3s.

`RECEIVER_IP` to **lokalny adres IPv4 odbiornika**, nie `127.0.0.1` i nie adres Tailscale. Przy wdrożeniu w k3s wpisz LAN-owy adres dowolnego węzła (np. `192.168.1.50`); usługa UDP jest publikowana na każdym węźle. Szczegóły w [`docs/KUBERNETES.md`](docs/KUBERNETES.md).

Nie wpisuj kluczy OpenAI/Hailo do szkicu. `Config.h`, `.env` i pliki z kompilacji są ignorowane przez Git. Stary `firmware/secrets.yaml` służy tylko do jednorazowego importu: późniejsze zmiany dla Nano wprowadzaj w `Config.h`.

Na świeżej kopii:

```sh
cd ~/src/open_lexa
uv sync
.venv/bin/python firmware/build/configure.py --receiver-ip 192.168.1.50
.venv/bin/python firmware/build/configure_arduino.py
```

Oba skrypty odmawiają nadpisania istniejącej konfiguracji. Alternatywnie skopiuj `Config.example.h` do `Config.h` i ustaw wartości ręcznie. Python wymaga wersji 3.12–3.14.

## 2. Arduino IDE

Potrzebny jest pakiet **esp32 by Espressif Systems 3.3.8**, zawierający biblioteki WakeNet. Pakiet **Arduino ESP32 Boards 2.x** nie zawiera wymaganych bibliotek. Oba pakiety mogą być zainstalowane równocześnie; wybierz płytkę z właściwego pakietu.

1. W Preferences → Additional Boards Manager URLs dodaj:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
2. W Boards Manager zainstaluj **esp32 by Espressif Systems**, wersja **3.3.8**.
3. Otwórz `firmware/src/OpenLexaNano/OpenLexaNano.ino`.
4. Wybierz **Tools → Board → esp32 → Arduino Nano ESP32**.
5. Pozostaw **Pin Numbering: By Arduino pin**, **Partition Scheme: With FAT partition (default)**, **USB Mode: Normal mode (TinyUSB)**. Pamięć PSRAM jest skonfigurowana w definicji tej płytki.
6. Wybierz port Nano, kliknij Verify, następnie Upload.

Nie trzeba instalować M5Unified/M5Atomic-EchoBase przez Library Manager. Potrzebny sterownik ES8311 i model są dołączone do szkicu. Biblioteki WakeNet pochodzą z pakietu Espressif. Nie trzeba dodatkowo wybierać partycji ESP-SR.

Monitor portu szeregowego: **115200 baud**. Program startuje też bez monitora. Wyślij literę **`r`**, aby ręcznie uruchomić sygnał i nagranie — przydatne do sprawdzenia audio niezależnie od Alexa.

## 3. Arduino CLI

Skrypt znajduje `arduino-cli` w PATH albo CLI dołączone do Arduino IDE Flatpak. Możesz też podać `ARDUINO_CLI=/pelna/sciezka/arduino-cli`.

```sh
cd ~/src/open_lexa
./firmware/build/arduino.sh install
./firmware/build/arduino.sh compile
./firmware/build/arduino.sh boards
./firmware/build/arduino.sh upload /dev/ttyACM0
```

`install` instaluje dokładnie pakiet 3.3.8. `compile` niczego nie wgrywa. `upload` najpierw kompiluje aktualną konfigurację, potem wgrywa. Wskaż rzeczywisty port Nano. Pliki wynikowe są w `firmware/.build/` (ignorowane przez Git).

Równoważna komenda bez skryptu:

```sh
arduino-cli compile \
  --fqbn 'esp32:esp32:nano_nora:PinNumbers=default,PartitionScheme=default,USBMode=default' \
  --build-path firmware/.build firmware/src/OpenLexaNano

arduino-cli upload \
  --fqbn 'esp32:esp32:nano_nora:PinNumbers=default,PartitionScheme=default,USBMode=default' \
  --input-dir firmware/.build --port /dev/ttyACM0 firmware/src/OpenLexaNano
```

Jeśli Nano nie pojawia się przy wgrywaniu, dwukrotnie naciśnij RESET, wybierz ponownie port i spróbuj Upload. Twoja notatka opisuje również błąd `LIBUSB_ERROR_OTHER` z Flatpakiem oraz ręczną ścieżkę DFU. Skrypt nie resetuje ani nie kasuje całego flasha na własną rękę.

## 4. Odbiornik UDP

Odbiornik jest osobną usługą: ten sam pakiet działa lokalnie i w k3s; adres `0.0.0.0:12345/UDP` jest domyślny. Firmware i odbiornik muszą być w tej samej sieci LAN (albo w sieci, której firewall przepuszcza UDP); ESP32 łączy się z Wi-Fi 2,4 GHz. Port domyślny: **12345/UDP**.

### Backendy STT

Domyślny tryb to **`dual`**: każde nagranie jest równolegle wysyłane do lokalnego
Whispera i do chmury (`gpt-transcribe`). Oba wyniki trafiają do archiwum, a tekst
GPT jest preferowany dla komend i automatycznie staje się „poprawną
transkrypcją” w adminie. Wybór przez `STT_BACKEND`, `--backend` lub zakładkę
*Ustawienia*:

| Backend | Zastosowanie | Wymagane zmienne |
| --- | --- | --- |
| `dual` | lokalny Whisper **i** chmura równolegle (zalecane) | `STT_HTTP_URL`, `STT_HTTP_MODEL`, `OPENAI_API_KEY`, `OPENAI_STT_MODEL`, opcjonalnie `OPENAI_BASE_URL` |
| `openai` | tylko OpenAI Audio Transcriptions | `OPENAI_API_KEY`, `OPENAI_STT_MODEL`, opcjonalnie `OPENAI_BASE_URL` |
| `http` | tylko dowolny endpoint zgodny z OpenAI — lokalny lub cloud | `STT_HTTP_URL`, `STT_HTTP_MODEL`, opcjonalnie `STT_HTTP_API_KEY` |
| `local` | zgodny wstecz alias `http` | jak wyżej |

Bez jawnego `STT_BACKEND` odbiornik wybiera `dual`, gdy ustawione są
`STT_HTTP_URL` i `OPENAI_API_KEY`; w przeciwnym razie zostaje przy `openai`.
Odbiornik nie korzysta bezpośrednio z Hailo: model Whisper na Hailo-10H działa w
osobnej usłudze [`whisper-api/`](whisper-api), do której receiver łączy się przez
`http`.

Wspólne: `STT_LANGUAGE` (domyślnie `pl`; `auto` wyłącza podpowiedź języka). `--model` nadpisuje model chmurowy w trybie `dual` albo model wybranego backendu.

**Ustawienia z admin UI.** Zakładka *Ustawienia* w `open-lexa-admin` domyślnie
wybiera **dual** (lokalne Whisper API + chmura: URL, model i klucz API) oraz
język. Admin zapisuje je w SQLite i publikuje jako `_config/stt.json` na udziale
SMB; odbiornik z włączonym archiwum SMB czyta ten plik co
`OPEN_LEXA_SETTINGS_SECONDS` (domyślnie 30 s, `0` wyłącza) i przełącza backend
przed kolejnym nagraniem. Kolejność: `--backend`/`--model` (wyłączają odczyt
pliku) > `_config/stt.json` > zmienne `STT_*`/`OPENAI_*`. Brak pliku oznacza
powrót do zmiennych; niepoprawny plik zostawia działający backend (`WARNING` w
logu). Klucz API leży w pliku na udziale otwartym tekstem, więc udział musi być
dostępny tylko dla Open Lexy.

**Automatyczna korekta.** W trybie `dual` sidecar archiwum zawiera osobno
`transcripts.local` i `transcripts.cloud` (tekst, model, błąd, czas),
`corrected_text` oraz `corrected_source`. Admin wczytuje tekst GPT jako
`corrected_transcript`, pokazuje obie wersje do porównania i pozwala jednym
kliknięciem wstawić lokalną wersję. W praktyce wystarczy więc dodać intencję i
encje — tekstu nie trzeba przepisywać. Jeśli wcześniej zapisano tryb `local`
albo `cloud`, otwórz *Ustawienia*, wybierz `dual` i zapisz.


Najpierw jakość audio, bez API:

```sh
cd ~/src/open_lexa
.venv/bin/open-lexa --dry-run --save-dir recordings
```

Powiedz „Alexa” albo wyślij `r` w monitorze szeregowym (115200 baud), poczekaj na sygnał i wypowiedz polecenie. Odsłuchaj wynikowy WAV w `recordings/`.

Diagnostyka w monitorze: `s` pokazuje stan Wi-Fi, powód ostatniego resetu, etap audio i wolną pamięć; `c` odczytuje ostatni zapisany zrzut awarii. Zrzut może pochodzić ze starszego firmware. Zamknij monitor IDE przed korzystaniem z innego programu obsługującego port. Błąd Wi-Fi `reason=201` oznacza, że płytka nie znalazła skonfigurowanej sieci.

Następnie STT:

```sh
.venv/bin/open-lexa                               # dual (oba endpointy), język pl
.venv/bin/open-lexa --backend dual                # wymuś lokalny Whisper + GPT transcribe
.venv/bin/open-lexa --backend http                # tylko lokalny (np. Whisper API) lub cloudowy serwer OpenAI-compatible
```

Po zakończeniu wypowiedzi program wypisuje rozpoznany tekst. `--save-dir recordings` można stosować również z STT; WAV jest zapisywany przed wywołaniem backendu. Bez tej opcji audio pozostaje w pamięci. Logi trafiają na stderr, transkrypcja na stdout.

Transkrypcja trafia do NLU opartego o eksportowany słownik (`_lexicon/lexicon.json`); przy jego braku funkcja `decide()` z [`commands.py`](udp-receiver/src/open_lexa/commands.py) zachowuje reguły regex jako awaryjne. Odbiornik przekazuje rozpoznaną intencję i encje do włączonych handlerów. Sonos/Spotify jest pierwszym pluginem (odtwarzanie, pauza, głośność); pozostałe intencje bez handlera są tylko logowane i archiwizowane. Ustawienia STT, oznaczanie LLM, model Base/Small i konfiguracja Sonos są dostępne w panelu admina. Architektura oraz instrukcje uruchomienia: [`docs/HANDLERS.md`](docs/HANDLERS.md).

**Rozpoznawanie intencji przez LLM (odbiornik).** Po każdej transkrypcji tekst może
trafić bezpośrednio do LLM (dowolny endpoint Chat Completions zgodny z OpenAI).
Wystarczy skonfigurować model w panelu admina (**Ustawienia → LLM**) i wybrać tryb
odbiornika — URL, model, timeout i klucz API trafiają wtedy do `_config/llm.json`
na udziale SMB (klucz tylko gdy tryb odbiornika jest włączony), więc nie trzeba go
wpisywać drugi raz w `.env`. Alternatywnie, bez archiwum SMB: zmienne `LLM_MODEL`,
`LLM_API_KEY` (Secret), `LLM_URL`, `LLM_TIMEOUT`. Tryb przez `LLM_INTENT_MODE` lub
`--llm-mode`: `always` — model rozstrzyga intencję i encje dla każdego nagrania
(domyślnie, gdy klucz i model są ustawione), `fallback` — LLM działa tylko, gdy
słownik i reguły nic nie rozpoznały, `off` — wyłączone. Kolejność źródeł:
`--llm-mode` (wyłącza odczyt pliku) > `_config/llm.json` > zmienne `LLM_*`. Model
wybiera wyłącznie intencje i typy encji z opublikowanego słownika (bez słownika
dostaje nazwy komend z reguł regex); nigdy nie tworzy nowych etykiet — to rola
oznaczania w adminie. Błędna lub niedostępna odpowiedź LLM nie przerywa potoku:
zostaje decyzja leksykalna/regułowa; niepoprawny plik zostawia działający klient
(`WARNING` w logu). Rozpoznanie trafia do handlerów i do sidecara archiwum z
`source: "llm"`, a liczniki wywołań widać w `/healthz`.

#### Archiwum SMB (do retrainingu)

Po ustawieniu `SMB_SERVER` i `SMB_SHARE` każde nagranie jest zapisywane na udziale SMB jako `\\SMB_SERVER\SMB_SHARE\SMB_PATH\RRRR-MM-DD\<ns>-<sesja>.wav` wraz z plikiem `.json` (tekst, język, backend, komenda, odpowiedź, czas trwania; przy błędzie STT `text: null` i `error`). W trybie `dual` sidecar zawiera dodatkowo `transcripts.local` i `transcripts.cloud` (tekst, model, błąd, czas odpowiedzi), `corrected_text` (tekst GPT) i `corrected_source`. JSON jest zapisywany po WAV, więc jego obecność oznacza kompletną parę. Klient SMB działa w Pythonie (`smbprotocol`), bez montowania CIFS, więc działa też w podzie non-root z read-only rootfs.

| Zmienna | Domyślnie | Opis |
| --- | --- | --- |
| `SMB_SERVER` | — | host serwera; puste wyłącza archiwum |
| `SMB_SHARE` | — | nazwa udziału (wymagana z `SMB_SERVER`) |
| `SMB_PATH` | `open-lexa` | katalog w udziale |
| `SMB_USERNAME`, `SMB_PASSWORD` | — | poświadczenia (w k3s: Secret) |
| `SMB_PORT`, `SMB_TIMEOUT` | `445`, `10` | port i timeout połączenia [s] |

Zapis odbywa się w osobnym wątku z kolejką na 16 nagrań, już po obsłudze komendy, więc wolny lub niedostępny udział nie blokuje transkrypcji ani komend. Błąd zapisu lub przepełniona kolejka są logowane i liczone w `archive_failures` (`/healthz`); kolejne nagranie nawiązuje połączenie od nowa. W `--dry-run` archiwum jest wyłączone.

Odbiornik wystawia też `GET /healthz` i `GET /readyz` na porcie 8080 (`OPEN_LEXA_HEALTH_PORT`, `0` wyłącza). `/readyz` zwraca 503, dopóki backend STT nie został poprawnie zainicjalizowany.

#### Osobne API Whisper w Kubernetes

Odbiornik UDP nie posiada HailoRT ani urządzenia Hailo. Wysyła WAV do
konfigurowalnego endpointu OpenAI-compatible:

```text
ESP32 -- UDP --> open-lexa receiver -- HTTP --> open-lexa-whisper Service --> Hailo-10H
```

Domyślny wariant `dual` używa `STT_BACKEND=dual`,
`STT_HTTP_URL=http://open-lexa-whisper:8080/v1` oraz `OPENAI_API_KEY` i
`OPENAI_STT_MODEL=gpt-transcribe`. Ten sam receiver można przełączyć na sam
lokalny Whisper (`STT_BACKEND=http`) albo sam cloud (`STT_BACKEND=openai` lub
`http`) bez zmiany firmware. Lokalna usługa obsługuje
`POST /v1/audio/transcriptions`, `GET /healthz` i `GET /readyz`; akceptuje
mono WAV PCM16/16 kHz z limitu UDP (maks. 12 s).

### k3s

Weryfikacja UDP przez ServiceLB, sekrety, SMB i instrukcja Hailo-10H: [`docs/KUBERNETES.md`](docs/KUBERNETES.md). Budowanie obrazów (Docker lub Tekton): [`docs/BUILD.md`](docs/BUILD.md).

- `udp-receiver/build/Dockerfile` — receiver (dual / OpenAI / serwer OpenAI-compatible),
- `whisper-api/build/Dockerfile` — osobny obraz API z HailoRT i Whisper,
- `admin-dashboard/build/Dockerfile` — panel admina,
- `<komponent>/deploy` — manifesty kustomize z przypiętymi digestami obrazów.

W skrócie: zbuduj obrazy arm64 (Tekton), utwórz Secret z `OPEN_LEXA_TOKEN`, zastosuj `kubectl apply -k udp-receiver/deploy` (tworzy też namespace), potem `whisper-api/deploy` i `admin-dashboard/deploy`. Usługa `type: LoadBalancer` z UDP jest obsługiwana przez k3s ServiceLB i nasłuchuje na **każdym** węźle, więc ESP32 może wysyłać na LAN-owy adres dowolnego węzła. Tailscale nie jest potrzebny na Arduino.

Test odbiornika bez Nano, istniejącym WAV mono PCM16/16 kHz o długości 0,2–12 s:

```sh
.venv/bin/python udp-receiver/tools/send_wav.py sciezka/do/polecenia.wav --host 127.0.0.1
# w k3s:
.venv/bin/python udp-receiver/tools/send_wav.py sciezka/do/polecenia.wav --host 192.168.1.50
```

Po wdrożeniu wariantu z API Whisper można przetestować wszystkie poprawne
nagrania z `recordings/`. Skrypt odczytuje adres z lokalnego `Config.h`,
wysyła każde WAV przez UDP i sprawdza liczniki `/readyz` receivera oraz stan
osobnego API, więc kończy się błędem, jeśli nagranie nie zostanie przetworzone:

```sh
.venv/bin/python udp-receiver/tools/test_hailo_recordings.py \
  --expected-backend dual --whisper-deployment open-lexa-whisper
kubectl -n open-lexa logs -f deployment/open-lexa -c receiver
```

Hailo Whisper podbija tylko ciche float PCM do szczytu `0.9`; pomaga to
szczególnie przy cichych nagraniach z ESP32, bez zmiany głośnych sygnałów. W `whisper-api/deploy/configmap.yaml` wartość kontroluje
`HAILO_WHISPER_NORMALIZE_PEAK` (`0` wyłącza normalizację do diagnostyki).

## Zachowanie i strojenie

- Model **WakeNet9 Alexa** działa na urządzeniu. Przed aktywacją audio nie jest wysyłane przez UDP ani do STT.
- Sygnał: 880 Hz przez 100 ms, łagodne zbocza. Mów po sygnale; audio wypowiedziane w jego trakcie jest odrzucane.
- Limit: 12 s od rozpoczęcia nagrania po sygnale. Zakończenie po 1 s ciszy. Brak co najmniej 200 ms aktywności w pierwszych 4 s powoduje pominięcie nagrania.
- Koniec wypowiedzi wykrywany progiem energii RMS w oknach 20 ms. To prosty detektor, który wymaga dostrojenia do hałasu i odległości. `SPEECH_RMS=500`: zmniejsz dla cichego głosu, zwiększ, jeśli szum podtrzymuje nagranie. `SILENCE_MS=1000`: zwiększ, jeśli chcesz dłuższe pauzy.
- `MIC_GAIN_DB` obsługuje 0 lub 6 dB. Zacznij od 0 dB, jeśli słyszysz przesterowanie. `BEEP_VOLUME` reguluje głośność sygnału.
- Zachowywane jest 250 ms przed pierwszą aktywnością oraz 300 ms po ostatniej. PCM zajmuje najwyżej 384 kB PSRAM.
- Wysyłka zaczyna się po zakończeniu nagrania, z ACK i ponawianiem pakietów. Niekompletne nagranie nie trafia do STT. Szczegóły: [protokół UDP](docs/UDP.md).
- Podczas nagrywania/wysyłania kolejne aktywacje są ignorowane. Po przesłaniu wraca nasłuch Alexa, podczas gdy odbiornik transkrybuje w osobnym wątku. Brak odpowiedzi odbiornika kończy się komunikatem i powrotem do nasłuchu.

## Struktura i weryfikacja

- `firmware/src/OpenLexaNano/` — główny szkic i kod C++.
- `AudioBoard.cpp` — pinout, inicjalizacja ES8311, I2S, bezpieczne dla mikrofonu wyciszanie DAC.
- `App.cpp` — WakeNet, nagrywanie, Wi-Fi i UDP.
- `src/audio/` — resampling i granice nagrania, używane również w testach C++.
- `src/model/` — osadzony model Alexa i manifest z SHA-256.
- `firmware/build/` — `arduino.sh` (install/compile/upload), konfiguracja i odtwarzanie modelu.
- `udp-receiver/src/open_lexa/` — odbiornik UDP, protokół, backendy STT (`stt.py`), NLU słownikowe i LLM (`nlu.py`, `llm.py`), archiwum i store SMB.
- `whisper-api/src/open_lexa_whisper/` — API OpenAI-compatible (`api.py`) i Whisper na HailoRT (`hailo.py`).
- `admin-dashboard/src/open_lexa_admin/` — panel do oznaczania nagrań i ustawień STT.
- `<komponent>/tests/` — testy; uruchamiane razem z roota.

```sh
.venv/bin/pytest -q
./firmware/build/arduino.sh compile
```

Testy obejmują protokół UDP, STT z mockiem, wybór backendu, tryb dual (równoległe wywołanie lokalny + chmura, preferencja GPT i fallback), faktyczny lokalny transfer 12 s audio, filtr antyaliasingowy/tempo resamplingu oraz limit długości, ciszę i przycinanie nagrania. Test C++ wymaga `g++`. Sama kompilacja nie potwierdza jakości mikrofonu ani czułości Alexa. 2026-10-06: firmware skompilowano i wgrano przez DFU na `/dev/ttyACM0`; 18 testów przeszło. Na fizycznym Nano usunięto reset przy przejściu od sygnału do nagrywania: `WakeNet9.clean()` w tym zestawie biblioteki/modelu powodował `LoadProhibited` w `dl_convq_queue_bzero`. Stan modelu jest teraz odtwarzany przez `destroy()` / `create()`. Potwierdzono start nagrania, zakończenie przy braku mowy i powrót do nasłuchu bez resetu. Test akustyczny, transfer z płytki i rzeczywista transkrypcja pozostają do potwierdzenia; skonfigurowana sieć Wi-Fi nie była widoczna, a klucz OpenAI w `.env` trzeba uzupełnić.

Model można odtworzyć skryptem `firmware/build/prepare_alexa_model.py`; wskazuje on stały commit ESP-SR v2.0.0. Zależności i licencje opisane są w [THIRD_PARTY.md](firmware/THIRD_PARTY.md).

Źródła: [pinout Nano](https://docs.arduino.cc/resources/pinouts/ABX00083-full-pinout.pdf), [ESP-SR](https://github.com/espressif/esp-sr), [pakiet Arduino Espressif](https://github.com/espressif/arduino-esp32/tree/3.3.8), [M5Atomic-EchoBase](https://github.com/m5stack/M5Atomic-EchoBase), [OpenAI STT](https://developers.openai.com/api/docs/guides/speech-to-text), [HailoRT GenAI Speech2Text](https://github.com/hailo-ai/hailo-apps).

## Licencja

MIT — zobacz [LICENSE](LICENSE). Komponenty zewnętrzne (sterownik ES8311, model WakeNet) zachowują własne licencje: [THIRD_PARTY.md](firmware/THIRD_PARTY.md).
