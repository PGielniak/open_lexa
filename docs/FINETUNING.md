# Fine-tuning Whispera na danych z panelu admina

Jak z oznaczonych nagrań w SQLite (`labels.sqlite3`) zrobić zbiór danych, dotrenować
Whispera i porównać go z obecnym modelem. Najpierw ograniczenia, bo to one decydują,
czy w ogóle warto zaczynać.

## 0. Najważniejsze ograniczenie: Hailo

Produkcyjny `whisper-api` używa `hailo_platform.genai.Speech2Text` z **gotowym
plikiem HEF zbudowanym przez Hailo** (obraz zawiera tylko `Whisper-Medium.hef`,
zob. `whisper-api/build/Dockerfile`). Nie ma udokumentowanej ścieżki, żeby
wkompilować własne wagi do tego formatu GenAI (z KV-cache). Przewodnik GenAI
Hailo mówi wprost, że wgrywanie własnych modeli nie jest wspierane, a pytania na
forum o to pozostają bez odpowiedzi.

**Dotrenowanego modelu nie da się więc po prostu podmienić w miejsce obecnego.**
Realne ścieżki:

| | Ścieżka | Gdzie działa | Zmiany w kodzie | Uwagi |
| --- | --- | --- | --- | --- |
| **A** | dotrenowany `whisper-small`/`base` → CTranslate2 (faster-whisper) za serwerem zgodnym z OpenAI | CPU (ARM64 Pi albo inna maszyna) | **brak w odbiorniku** — backend `http`/`dual` potrzebuje tylko `STT_HTTP_URL` | trzeba zmierzyć opóźnienie na Pi; Small na CPU Pi może być za wolny |
| **B** | dotrenowany `tiny`/`base` → [hailo-whisper](https://github.com/hailocs/hailo-whisper) → osobne HEF enkodera i dekodera | Hailo-10H | **nowy backend w `whisper-api`** (własna pętla dekodowania, embeddingi na CPU) | tylko tiny/base; base ma wejście **5 s**; brak KV-cache; konwersja tylko na x86 z DFC 5.x; Small według zgłoszeń się nie kompiluje |
| **C** | zostaje fabryczny Whisper-Medium na Hailo | Hailo-10H | brak | dane służą do **ewaluacji** (WER) i do słowników/normalizacji NLU, a nie do treningu |

Rekomendacja: **C teraz, A gdy będzie dość danych**. B ma sens tylko wtedy, gdy
koniecznie chcemy NPU i godzimy się na model base (słabszy od obecnego Medium) oraz
na sporą pracę integracyjną.

## 1. Ile jest danych

Stan lokalnej kopii bazy (17.11.2026): **85 oznaczonych nagrań, ok. 7 minut audio**.
25 z nich trwa dłużej niż 5 s (najdłuższe 12 s).

To rozmiar **zbioru testowego**, nie treningowego. Fine-tuning na kilku minutach
nauczy model najwyżej powtarzania tych samych komend i najpewniej się przeuczy.
Orientacyjne progi:

- kilkadziesiąt minut — ma sens tylko lekkie dostrojenie (zamrożony enkoder, 1–3
  epoki) pod słownictwo domowe: nazwy stacji, pokoi, „Sonos”, „Spotify”;
- **kilka godzin** — dopiero wtedy można oczekiwać wyraźnego spadku WER.

Od czasu automatycznego etykietowania w trybie dual każde nagranie trafia od razu do
`labeled`, więc zbiór rośnie sam. Postęp sprawdzisz tak:

```sh
python3 -c "import sqlite3; c=sqlite3.connect('labels.sqlite3'); print(c.execute(
  \"select count(*), round(sum(duration_seconds)/3600, 2) from recordings where status='labeled'\").fetchone())"
```

## 2. Skąd biorą się etykiety (ważne dla ewaluacji)

Większość `corrected_transcript` to **tekst GPT** (`corrected_source = 'cloud'`),
a od dziś w trybie dual wszystkie nagrania są oznaczane automatycznie. Wynika z tego:

- **trening** na nich to destylacja GPT → Whisper; to w porządku, model nauczy się
  także błędów GPT, ale jest ich mało;
- **WER liczony względem tych etykiet mierzy tylko zgodność z GPT**, a nie
  poprawność. Zbiór testowy trzeba **przesłuchać i poprawić ręcznie** w panelu
  (szczegóły nagrania → popraw tekst → zapisz).

## 3. Eksport danych

Panel już eksportuje `_dataset/transcripts.jsonl` na SMB, ale do treningu
wygodniejszy jest format `audiofolder` z Hugging Face (WAV + `metadata.csv`).
Robi go `whisper-api/tools/export_finetune_dataset.py`. Narzędzie czyta bazę tylko
do odczytu i pobiera WAV-y z archiwum tak jak odbiornik (`SMB_*` albo
`ADMIN_ARCHIVE_DIR` dla lokalnego katalogu ułożonego po dniach).

```sh
# kopia bazy z poda (PVC Longhorn)
kubectl --context default -n open-lexa cp <pod-admina>:/data/labels.sqlite3 ./labels.sqlite3

uv run python whisper-api/tools/export_finetune_dataset.py \
  --db labels.sqlite3 --out ~/whisper-data --test-days 7
```

- Brane są tylko nagrania `labeled`, bez `stt_error`, z niepustym transkryptem.
  Pominięte (`skipped`) nie trafiają do zbioru.
- Podział na train/test jest **według dni** (`--test-days` najnowszych dni idzie do
  testu), a nie losowy, żeby powtórzenia z jednej sesji nie przeciekały między
  zbiorami. **Zbiór `test/` przesłuchaj i popraw w panelu przed ewaluacją**, potem
  wyeksportuj go ponownie.
- `--max-seconds 5` odrzuca dłuższe nagrania (potrzebne dla ścieżki B z modelem base).
- Kolumna `source` w `metadata.csv` mówi, skąd jest etykieta (`cloud` = GPT).

## 4. Trening (maszyna z GPU)

Klaster to ARM64 Pi bez GPU, więc trenujemy gdzie indziej: na własnym PC z kartą
NVIDIA, Colabie albo wynajętym GPU. Zwykle wystarczy 8–12 GB VRAM dla `small`.

```sh
uv run --with 'transformers>=4.44,<5' --with 'datasets[audio]>=2.20,<4' \
  --with accelerate --with evaluate --with jiwer --with torch \
  python whisper-api/tools/finetune_whisper.py \
  --data ~/whisper-data --model openai/whisper-small --out ~/whisper-small-pl-lexa
```

Skrypt (`whisper-api/tools/finetune_whisper.py`) używa `Seq2SeqTrainer` z językiem
`polish` i zadaniem `transcribe`. Domyślnie:

- **enkoder jest zamrożony** (`--train-encoder` go odmraża; ma sens dopiero przy
  godzinach danych),
- `lr = 1e-5`, 5 epok, ewaluacja WER na `test/` po każdej epoce i zapis najlepszego
  checkpointu,
- przed i po treningu wypisuje WER (`baseline` / `fine-tuned`), żeby było widać,
  czy w ogóle coś się poprawiło.

Wskazówki:

- Patrz na **WER testu**, nie na loss treningu. Jeśli WER rośnie po 1–2 epokach,
  model się przeucza: zmniejsz `--epochs` albo `--lr`.
- Przy bardzo małym zbiorze alternatywą jest LoRA (biblioteka `peft`), ale przy
  zamrożonym enkoderze i niskim LR różnica jest niewielka.
- Model bazowy wybierz pod ścieżkę wdrożenia: A → `whisper-small` (albo `base`,
  jeśli Small na Pi będzie za wolny), B → `whisper-base` lub `whisper-tiny`.

Skrypt treningowy nie był uruchamiany w tym repo (brak GPU), więc pierwsze
uruchomienie traktuj jako test.

## 5. Ewaluacja względem obecnego modelu

Zanim cokolwiek wdrożysz, porównaj WER na **ręcznie sprawdzonym** zbiorze testowym:

- obecny Hailo Whisper-Medium: `whisper-api/tools/compare_models.py`
  (`--url http://open-lexa-whisper:8080/v1`; listę modeli w skrypcie ustaw na
  `whisper-medium`, bo Base i Small nie ma już w obrazie),
- dotrenowany model: WER z końca `finetune_whisper.py`, albo ten sam
  `compare_models.py` skierowany na serwer ze ścieżki A.

Wdrażaj tylko wtedy, gdy model jest **wyraźnie lepszy od Medium** przy akceptowalnym
opóźnieniu. Dla komend głosowych liczy się też trafność intencji, nie tylko WER.

## 6. Wdrożenie

### Ścieżka A — CTranslate2 + serwer zgodny z OpenAI

```sh
uv run --with ctranslate2 --with 'transformers[torch]' \
  ct2-transformers-converter --model ~/whisper-small-pl-lexa \
  --output_dir ~/whisper-small-pl-lexa-ct2 --quantization int8 \
  --copy_files tokenizer.json preprocessor_config.json
```

Model w formacie CTranslate2 serwuje dowolny serwer z endpointem
`/v1/audio/transcriptions` zgodnym z OpenAI, opartym na faster-whisper (np.
speaches, dawniej faster-whisper-server). Potrzebny jest obraz ARM64. W panelu
admina (*Ustawienia*) wskaż jego URL i model jako „lokalny”. Tryb dual nadal
woła GPT równolegle, więc nowe nagrania dalej będą się oznaczać automatycznie.
Odbiornik nie wymaga zmian.

### Ścieżka B — hailo-whisper na Hailo-10H

Według [Conversion.md](https://github.com/hailocs/hailo-whisper/blob/main/docs/Conversion.md):

```sh
# x86, Ubuntu 22, Python 3.10/3.11, DFC 5.x z Hailo Developer Zone
git clone https://github.com/hailocs/hailo-whisper.git && cd hailo-whisper
python3 setup.py && source whisper_env/bin/activate
python3 -m export.export_whisper_model --variant base
python3 -m conversion.convert_whisper_encoder <encoder.onnx> --variant base --hw-arch hailo10h
python3 -m conversion.create_decoder_calib_set --encoder <encoder.onnx> --decoder <decoder.onnx> --variant base
python3 -m conversion.convert_whisper_decoder <decoder.onnx> --variant base --hw-arch hailo10h
```

Czego to wymaga:

- Eksport bierze oficjalny checkpoint; **wczytanie własnych wag trzeba dopisać**
  (dotrenowany model HF przekonwertować do formatu `openai-whisper` albo podmienić
  `state_dict` w skrypcie eksportu).
- Kalibracja domyślnie używa LibriSpeech (angielski). Lepiej dopisać parser dla
  własnych nagrań (wzór: `create_calibration_set_librispeech` w
  `convert_whisper_encoder.py`).
- Wynik to dwa pliki HEF (enkoder i dekoder) plus pliki `.npy` z embeddingami.
  `whisper-api` potrzebuje **nowego backendu** z własną pętlą dekodowania na HailoRT
  (bez `Speech2Text`, bez KV-cache, więc wolniej przy dłuższych zdaniach).
- Base przyjmuje 5 s audio. Dłuższe komendy trzeba ciąć; `whisper-api` ma już
  `chunk_seconds`, ale cięcie w środku słowa pogarsza wynik.
- Społeczność zgłasza, że fine-tuned **Small/Tiny nie kompilują się** na DFC 5.x
  (błąd parsera enkodera), a działający przykład dotyczy polskiego **base** z WER
  ok. 37,6% na zbiorze autora, co jest prawdopodobnie gorsze od obecnego Medium.

## Źródła

- [hailocs/hailo-whisper](https://github.com/hailocs/hailo-whisper) i
  [docs/Conversion.md](https://github.com/hailocs/hailo-whisper/blob/main/docs/Conversion.md)
- [Compiling a fine-tuned Whisper-Small for Hailo-10H](https://community.hailo.ai/t/compiling-a-fine-tuned-whisper-small-for-hailo-10h/20009)
- [KV-cache / low-latency decoding for custom fine-tuned Whisper on Hailo-10H?](https://community.hailo.ai/t/kv-cache-low-latency-decoding-for-custom-fine-tuned-whisper-on-hailo-10h/19528)
- [Whisper-base.en fine tuning with LoRA](https://community.hailo.ai/t/whisper-base-en-fine-tuning-with-lora/19640)
