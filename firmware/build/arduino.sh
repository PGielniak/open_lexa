#!/usr/bin/env bash
set -euo pipefail
FIRMWARE="$(cd "$(dirname "$0")/.." && pwd)"
SKETCH="$FIRMWARE/src/OpenLexaNano"
# Compiler output; firmware/build/ holds only these scripts.
BUILD="$FIRMWARE/.build"
FQBN="esp32:esp32:nano_nora:PinNumbers=default,PartitionScheme=default,USBMode=default"
URL="https://espressif.github.io/arduino-esp32/package_esp32_index.json"
CLI="${ARDUINO_CLI:-}"
if [[ -z "$CLI" ]]; then
  if command -v arduino-cli >/dev/null; then
    CLI="$(command -v arduino-cli)"
  else
    for candidate in "$HOME"/.local/share/flatpak/app/cc.arduino.IDE2/current/active/files/arduino-ide/resources/app/lib/backend/resources/arduino-cli /var/lib/flatpak/app/cc.arduino.IDE2/current/active/files/arduino-ide/resources/app/lib/backend/resources/arduino-cli; do
      if [[ -x "$candidate" ]]; then CLI="$candidate"; break; fi
    done
  fi
fi
if [[ -z "$CLI" ]]; then echo 'Install Arduino CLI or set ARDUINO_CLI to its executable.' >&2; exit 1; fi
case "${1:-}" in
  install)
    "$CLI" core update-index --additional-urls "$URL"
    "$CLI" core install esp32:esp32@3.3.8 --additional-urls "$URL"
    ;;
  compile)
    "$CLI" compile --fqbn "$FQBN" --build-path "$BUILD" "$SKETCH"
    ;;
  upload)
    if [[ -z "${2:-}" ]]; then echo 'Usage: firmware/build/arduino.sh upload /dev/ttyACM0' >&2; exit 1; fi
    "$CLI" compile --fqbn "$FQBN" --build-path "$BUILD" "$SKETCH"
    "$CLI" upload --fqbn "$FQBN" --input-dir "$BUILD" --port "$2" "$SKETCH"
    ;;
  boards) "$CLI" board list ;;
  *) echo 'Usage: firmware/build/arduino.sh {install|compile|boards|upload PORT}' >&2; exit 1 ;;
esac
