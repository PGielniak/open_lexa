# Lokalny model do tool callingu na Hailo-10H (zamiast GPT)

Jak dotrenować mały LLM tak, żeby wybierał narzędzie Sonos i jego argumenty zamiast
GPT, i uruchomić go jako HEF na Hailo-10H. Zbiór danych do speech-to-text opisuje
[FINETUNING.md](FINETUNING.md); ten dokument dotyczy kroku **tekst → tool call**
(`udp-receiver/src/open_lexa/tool_agent.py`).

## TL;DR

| Decyzja | Wybór | Dlaczego |
| --- | --- | --- |
| Model bazowy | **Qwen2-1.5B-Instruct + LoRA** | jedyny model, dla którego Hailo publikuje gotowy do LoRA plik HAR dla naszej wersji (`v5.5.0/blob/qwen2_1.5b_instruct.q.har`) |
| Metoda | **LoRA (SFT)** na MLP (`gate/up/down_proj`), r=16 | pełny fine-tuning nie przejdzie przez kompilator Hailo; LoRA jest oficjalnie wspierane w DFC |
| Format odpowiedzi | **jedna linia JSON**: `{"name": ..., "arguments": {...}}` / `{"name": null}` | ~5 tok/s na Hailo: krótka odpowiedź = mniejsze opóźnienie; łatwo parsować i walidować |
| Prompt | **stały, zwięzły system prompt** (lista narzędzi jako sygnatury, bez JSON Schema) | kontekst 2048 tokenów, długi prefill na NPU; model uczy się narzędzi z danych |
| Dane | decyzje GPT z kolumny `agent` + transkrypty lokalnego Whispera + dane syntetyczne | destylacja GPT → mały model, z prawdziwymi błędami lokalnego STT |
| Trening | GPU poza klastrem | Pi nie mają GPU |
| Kompilacja | DFC **5.5.0** (zgodny z HailoRT 5.5.0), x86 Linux, kilka godzin | HEF musi pasować do runtime w obrazie |

**Największe ryzyko**: jakość po kwantyzacji na urządzeniu. Na forum Hailo opisano
adapter LoRA na Qwen2-1.5B, który na GPU miał 100%, a na Hailo-10H **39%** (gorzej niż
model bez adaptera). Dlatego GPT wyłączamy dopiero po ewaluacji **na urządzeniu**
(krok 6), a nie na GPU.

## 0. Dlaczego nie inne modele

- **Gemma4-E2B** (obecny `open-lexa-llm`): działa, ale Hailo nie publikuje dla niego
  HAR-a pod LoRA, więc nie da się go dotrenować i skompilować.
- **Własny HEF z dowolnego checkpointu HF**
  ([hailo-10h-llm-compiler](https://github.com/l-nmch/hailo-10h-llm-compiler)):
  projekt eksperymentalny, spójny tekst tylko dla modeli ~25M parametrów; od
  szerokości 576 (np. Qwen2.5-0.5B) jakość się sypie, kontekst 128 tokenów, DFC 5.3.0.
  Do tool callingu po polsku się nie nadaje.
- **Qwen2.5 / Qwen3 / Llama 3.2** są w hailo-ollama jako gotowe HEF-y, ale nie
  znalazłem dla nich publicznego `*.q.har` z obsługą LoRA (`qwen2.5_1.5b_instruct.q.har`
  zwraca 404). Sprawdź listę plików w Developer Zone dla DFC 5.5.0. Jeśli pojawi się
  HAR dla Qwen2.5-1.5B-Instruct, wybierz go: ma lepszy tool calling i lepszy polski.
- **Bez fine-tuningu**: Gemma4 + `tool-adapter` już działa jako backend
  „lokalny”. Warto zmierzyć jego trafność tym samym skryptem (krok 4). Jeśli wystarczy,
  cała reszta jest zbędna.

## 1. Dane

Każde nagranie przetworzone przez odbiornik ma w SQLite kolumnę `agent`: narzędzie i
argumenty wybrane przez GPT (już zwalidowane względem JSON Schema) albo `no_tool`. To
jest etykieta. Eksporter:

```sh
kubectl --context default -n open-lexa cp <pod-admina>:/data/labels.sqlite3 ./labels.sqlite3

uv run python hailo-llm/tools/export_tool_dataset.py \
  --db labels.sqlite3 --out ~/tool-data --test-days 2
```

- Wejście to transkrypt GPT **oraz** transkrypt lokalnego Whispera z tą samą
  etykietą. Model uczy się więc, że „i wąz radii zep” to
  `play_radio(station="Radio Zet")`.
- Brane są `handled`, `no_tool` (przykłady negatywne) i `failed` z poprawnie wybranym
  narzędziem (np. błąd UPnP Sonosa). Odrzucane: błędy samego LLM, `skipped` i argumenty
  niezgodne ze schematem.
- Podział według dni (`--test-days`, najnowsze dni), przy czym zawsze zostaje co
  najmniej jeden dzień na trening.
- `system_prompt.txt` to prompt, z którym model będzie **trenowany i serwowany**.
  Musi być identyczny w obu miejscach.

Stan na 17.11.2026: 48 decyzji → 65 przykładów treningowych + 12 testowych. Na LoRA
potrzeba **~1–3 tys.** przykładów (w działającym przykładzie z forum ok. 2000).
Brakujące dane uzupełnij syntetycznie.

### Dane syntetyczne (GPT jako nauczyciel)

Poproś GPT (np. `gpt-4o`) o warianty wypowiedzi dla każdego narzędzia i zapisz je
jako JSONL `{"text": ..., "name": ..., "arguments": {...}}`. Eksporter waliduje
każdy wiersz względem schematów narzędzi i dokłada go **tylko do train**:

```sh
uv run python hailo-llm/tools/export_tool_dataset.py --db labels.sqlite3 \
  --out ~/tool-data --extra synthetic.jsonl
```

Co generować, żeby model nie nauczył się skrótów:

- **różne formy**: „ścisz”, „ciszej o pięć”, „daj głośniej”, „głośność na
  trzydzieści” (liczebniki słownie i cyframi), tryb rozkazujący i prośby;
- **nazwy własne**: stacje z ulubionych Sonos / TuneIn, artyści i gatunki (także
  angielskie nazwy w polskiej wymowie: „czejs end statas”);
- **zakłócenia STT**: zgubione pierwsze słowo, „i” na początku, zlepione słowa
  („rozpotify”), brak polskich znaków; najlepiej przepuść część tekstów przez TTS
  i lokalny Whisper;
- **negatywy, ok. 20–30%**: rozmowa w tle, pytania o pogodę, „Alexa, która
  godzina”, pojedyncze słowa, szum → `{"name": null}`;
- **pułapki**: „włącz radio” → `play_default_radio`, a nie `play_radio`; stacja na
  Spotify → `play_radio`; „następna” bez kontekstu → `next_track`.

Nie wrzucaj do danych syntetycznych zdań z `test.jsonl`, bo zawyżysz wynik.

## 2. Trening LoRA (GPU)

Wystarczy GPU z ~12 GB VRAM: na Colabie, wynajętej maszynie albo własnym PC.

```sh
uv run --with torch --with 'transformers>=4.51,<5' --with 'peft>=0.15,<1' \
  --with 'trl>=0.20,<1' --with 'datasets>=3,<5' --with accelerate \
  python hailo-llm/tools/train_tool_lora.py --data ~/tool-data --out ~/qwen2-lexa-lora
```

`hailo-llm/tools/train_tool_lora.py`:

- bazuje na `Qwen/Qwen2-1.5B-Instruct` i liczy loss tylko na odpowiedzi
  (prompt–completion, `completion_only_loss`),
- LoRA r=16, alpha=32, na `gate_proj`/`up_proj`/`down_proj`, czyli w układzie
  z działającego przykładu Hailo. Zanim dodasz `q/k/v/o_proj`, sprawdź w
  tutorialu LoRA DFC, co jest wspierane,
- przed i po treningu wypisuje `exact` (narzędzie + argumenty), `tool` (samo
  narzędzie), `invalid` (nieparsowalny JSON) i `false_trigger` (akcja przy
  negatywie).

Na co patrzeć:

- `false_trigger` ma być bliski zera. Fałszywe „włącz muzykę” w środku rozmowy jest
  gorsze niż „nie zrozumiałam”.
- Jeśli `exact` na teście spada po kolejnych epokach, model się przeucza. Zmniejsz
  `--epochs` albo dodaj danych.
- Skrypt nie był uruchamiany w tym repo (brak GPU), więc pierwszy przebieg traktuj
  jako test.

## 3. Kompilacja do HEF (x86, DFC 5.5.0)

Potrzebne z [Hailo Developer Zone](https://hailo.ai/developer-zone/), z darmowym
kontem:

- **Dataflow Compiler 5.5.0** w tej samej wersji co HailoRT w obrazie
  (`ghcr.io/pgielniak/hailort:5.5.0`),
- bazowy HAR:
  `https://dev-public.hailo.ai/v5.5.0/blob/qwen2_1.5b_instruct.q.har`, wraz z
  plikami `.alls` dla tego modelu z tej samej strony,
- **tutorial LoRA z dokumentacji DFC** (notatnik `DFC_7_LoRA_Tutorial`). To on
  jest źródłem prawdy o API: wczytanie wag adaptera (`load_lora_weights`),
  kwantyzacja adaptera i kompilacja.

Przebieg według tutoriala:

1. Wczytaj `qwen2_1.5b_instruct.q.har` w `ClientRunner`.
2. Dodaj adapter z `~/qwen2-lexa-lora` (format PEFT `adapter_model.safetensors`).
   Każdy adapter jest kwantyzowany osobno, a jeden HEF może mieć ich kilka.
3. Kalibracja: **użyj własnych promptów** (64–128 przykładów z `train.jsonl`, z tym
   samym system promptem), a nie tekstów ogólnych. Na forum utrata jakości na
   urządzeniu wyglądała na problem kwantyzacji, więc kalibracja na danych docelowych
   to najtańsza obrona.
4. Skompiluj do HEF. Na zwykłym PC trwa to kilka godzin; zarezerwuj dużo RAM.

## 4. Ewaluacja na urządzeniu

Zanim cokolwiek przełączysz, zmierz HEF na Hailo-10H tymi samymi danymi
(`test.jsonl`) i tymi samymi metrykami co w kroku 2, a następnie porównaj:

| Model | exact | false_trigger | p50 opóźnienia |
| --- | --- | --- | --- |
| GPT (obecny) | odniesienie | | |
| Gemma4-E2B + tool-adapter (bez treningu) | | | |
| Qwen2-1.5B bez adaptera (HEF Hailo) | | | |
| Qwen2-1.5B + LoRA (GPU, fp) | | | |
| **Qwen2-1.5B + LoRA (Hailo-10H)** | | | |

Jeśli wynik na urządzeniu wyraźnie odstaje od GPU (jak 100% → 39% na forum):
popraw kalibrację, zmniejsz rank, skróć prompt, sprawdź zgodność wersji DFC i HailoRT.
GPT zostaje w użyciu, dopóki lokalny model nie będzie wystarczająco dobry.

## 5. Serwowanie

1. **Rejestracja w hailo-ollama.** Skopiuj manifest `qwen2/1.5b`
   (`hailo-llm/vendor/share/hailo-ollama/models/manifests/qwen2/1.5b/manifest.json`)
   jako np. `qwen2/lexa`, wskaż w nim własny HEF i umieść HEF w cache modeli na
   raspberry02 (`/home/patryk/.local/share/hailo-ollama`), obok pobranych modeli.
   Moderator Hailo pisał, że wybór adaptera LoRA w hailo-ollama jest dopiero
   planowany. Jeśli hailo-ollama zignoruje adapter, uruchom mały serwer na
   `hailo_platform.genai.LLM(..., lora_name=...)`, tak jak `whisper-api` korzysta
   z `Speech2Text`.
2. **Adapter.** Obecny `hailo-llm/adapter/gemma_tool_adapter.py` parsuje składnię
   Gemmy. Dla dotrenowanego modelu potrzebny jest wariant, który:
   - zamiast systemu i `tools` od odbiornika wstawia **dokładnie**
     `system_prompt.txt` z treningu (np. z ConfigMap),
   - parsuje jedną linię JSON odpowiedzi do OpenAI `tool_calls` (`{"name": null}` →
     brak `tool_calls`, czyli „nie zrozumiałam”).

   Odbiornik nadal waliduje argumenty JSON Schema, więc błędny wynik modelu
   kończy się komunikatem „nie zrozumiałam”, a nie złą akcją.
3. **Przełączenie.** W panelu admina (*Ustawienia → LLM*) ustaw URL
   `http://open-lexa-llm.open-lexa.svc:8000/v1` i model `qwen2:lexa`. Odbiornik nie
   wymaga zmian. Odbiornik używa jednego LLM naraz, więc powrót do GPT to
   zmiana URL-a z powrotem w tym samym miejscu.

## 6. Utrzymanie

- Każde nowe narzędzie MCP wymaga nowej wersji `system_prompt.txt`, nowych danych i
  ponownego treningu z kompilacją. Przy częstych zmianach narzędzi lepiej zostać przy
  modelu ogólnym z promptem (GPT albo Gemma4).
- Etykiety w kolumnie `agent` pochodzą od LLM, który akurat obsługiwał nagranie. Po
  przełączeniu na model lokalny nowe nagrania przestają być destylacją GPT. Do
  dalszego zbierania danych zbieraj je przy GPT albo przeglądaj ręcznie.

## Źródła

- [LoRA HEF (Qwen2-1.5B) on Hailo-10H: 100% on GPU, 39% on device](https://community.hailo.ai/t/lora-hef-qwen2-1-5b-runs-on-hailo-10h-but-scores-below-the-base-model-100-on-gpu-39-on-device/19798)
- [Hailo blog: Bringing Generative AI to the Edge – LLM on Hailo-10H](https://hailo.ai/blog/bringing-generative-ai-to-the-edge-llm-on-hailo-10h/) (LoRA w DFC, wiele adapterów w jednym HEF)
- [Hailo-Ollama Server with bigger model support?](https://community.hailo.ai/t/hailo-ollama-server-with-bigger-model-support/18644) (LoRA w hailo-ollama planowane)
- [l-nmch/hailo-10h-llm-compiler](https://github.com/l-nmch/hailo-10h-llm-compiler) (eksperymentalna kompilacja dowolnego LLM)
- [Macnica: Qwen na Hailo-10H](https://www.macnica.co.jp/en/business/semiconductor/articles/hailo/148158/)
