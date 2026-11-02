# Third-party components

## ES8311 driver

`src/OpenLexaNano/src/codec/es8311.cpp`, `es8311.h`, `es8311_reg.h` are vendored unchanged from the M5Atomic-EchoBase Arduino library (`src/`) on 2026-10-05. Upstream: https://github.com/m5stack/M5Atomic-EchoBase . Original Espressif copyright and Apache-2.0 SPDX notices are retained. See `licenses/Apache-2.0.txt`.

The application deliberately uses the legacy ESP-IDF I2S interface to preserve the verified raw-GPIO interpretation and 44.1 kHz/stereo/16-bit framing. It does not use the M5 library's version-dependent I2S wrapper. Package 3.3.8 still provides this legacy interface; its deprecation warning is expected.

## Alexa wake word

The generated `src/OpenLexaNano/src/model/AlexaModel.cpp` contains `wn9_alexa` from Espressif ESP-SR v2.0.0, commit `1c0503e7e2dc5e1ecb2fd3da0a5d7e51e3574842`. Exact input hashes and packed hash are in `manifest.json`. The model is for use on Espressif products; see `licenses/ESP-SR-LICENSE.txt` for the upstream Espressif MIT license. Source: https://github.com/espressif/esp-sr/tree/v2.0.0/model/wakenet_model/wn9_alexa . The binary layout follows upstream `model/pack_model.py` and is generated with explicit little-endian fields by `build/prepare_alexa_model.py`.

WakeNet inference binaries/headers are supplied by Arduino `esp32:esp32@3.3.8`, not copied here. Their upstream licenses remain in that installed package. The model is linked in read-only flash and loaded using `srmodel_load`, so no filesystem/model partition or custom upload recipe is needed.
