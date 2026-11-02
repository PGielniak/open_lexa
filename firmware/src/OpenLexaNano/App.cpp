#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_arduino_version.h>
#if ESP_ARDUINO_VERSION_MAJOR < 3
#error "Select Arduino Nano ESP32 from esp32 by Espressif Systems 3.3.8 (the Arduino 2.x package lacks WakeNet)."
#endif
#include <esp_heap_caps.h>
#include <esp_random.h>
#include <esp_system.h>
#include <esp_core_dump.h>
#include <esp_wn_iface.h>
#include <esp_wn_models.h>
#include <model_path.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <new>

#include "Config.h"
#include "AudioBoard.h"
#include "src/audio/Resampler.h"
#include "src/audio/CommandCapture.h"
#include "src/model/AlexaModel.h"

namespace {
uint32_t previous_audio_stage = 0;
std::atomic<unsigned> audio_stack_free{0};
std::atomic<bool> requested{false}, busy{false};
const esp_wn_iface_t *wake = nullptr;
model_iface_data_t *wake_data = nullptr;
const char *wake_name = nullptr;
int16_t *wake_frame = nullptr, *recording = nullptr;
size_t wake_size = 0, wake_used = 0;
Resampler *resampler = nullptr;
IPAddress receiver;
uint8_t token[16];
float previous_input = 0, previous_output = 0;

void fatal(const char *reason) {
  Serial.printf("FATAL: %s\n", reason);
  AudioBoard::muteDac(true);
  while (true) { delay(3000); Serial.printf("FATAL: %s\n", reason); }
}
int hexDigit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}
void put32(uint8_t *p, uint32_t n) {
  p[0] = n >> 24; p[1] = n >> 16; p[2] = n >> 8; p[3] = n;
}
uint32_t get32(const uint8_t *p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}

bool upload(const int16_t *samples, size_t count) {
  if (WiFi.status() != WL_CONNECTED) { Serial.println("No Wi-Fi; recording discarded."); return false; }
  WiFiUDP udp;
  if (!udp.begin(0)) return false;
  constexpr size_t chunk = 1024;
  const uint32_t session = esp_random(), total = count * 2;
  const uint32_t started = millis();
  uint8_t packet[34 + chunk];
  memcpy(packet, "OLX1", 4); memcpy(packet + 4, token, 16);
  put32(packet + 20, session); put32(packet + 24, total);
  for (uint32_t seq = 0; seq * chunk < total; ++seq) {
    const size_t offset = seq * chunk;
    const size_t length = std::min(chunk, size_t(total - offset));
    put32(packet + 28, seq); packet[32] = length >> 8; packet[33] = length;
    memcpy(packet + 34, reinterpret_cast<const uint8_t *>(samples) + offset, length);
    bool acknowledged = false;
    for (unsigned attempt = 0; attempt < 10 && !acknowledged; ++attempt) {
      if (millis() - started >= 30000 || WiFi.status() != WL_CONNECTED) break;
      if (udp.beginPacket(receiver, UDP_PORT)) {
        udp.write(packet, 34 + length);
        udp.endPacket();
      }
      const uint32_t sent = millis();
      while (millis() - sent < 100) {
        const int size = udp.parsePacket();
        if (size) {
          uint8_t ack[29];
          const bool peer = udp.remoteIP() == receiver && udp.remotePort() == UDP_PORT;
          const int received = udp.read(ack, sizeof(ack));
          if (size == 28 && received == 28 && peer && memcmp(ack, "OLA1", 4) == 0 &&
              memcmp(ack + 4, token, 16) == 0 && get32(ack + 20) == session && get32(ack + 24) == seq) {
            acknowledged = true;
            break;
          }
        }
        delay(1);
      }
    }
    if (!acknowledged) {
      Serial.printf("UDP: no ACK for chunk %lu; check receiver/IP/token/firewall.\n", (unsigned long) seq);
      udp.stop();
      return false;
    }
  }
  udp.stop();
  Serial.printf("Delivered %.2f s to receiver. Listening for Alexa.\n", count / 16000.0);
  return true;
}

void resetAudioHistory() {
  AudioBoard::diagnostic_stage = 200;
  AudioBoard::discardInput();
  AudioBoard::diagnostic_stage = 201;
  resampler->reset();
  previous_input = previous_output = 0;
  wake_used = 0;
  AudioBoard::diagnostic_stage = 202;
  // The bundled WakeNet9 clean() crashes in dl_convq_queue_bzero on this
  // core/model combination. Recreate via the public lifecycle API instead.
  wake->destroy(wake_data);
  wake_data = wake->create(wake_name, DET_MODE_90);
  if (!wake_data) fatal("WakeNet recreation failed.");
  AudioBoard::diagnostic_stage = 203;
}

int16_t removeDC(int16_t sample) {
  const float y = sample - previous_input + 0.995f * previous_output;
  previous_input = sample; previous_output = y;
  return static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, y)));
}

void command() {
  busy = true;
  requested = false;
  Serial.println("Alexa / serial trigger -> beep");
  if (!AudioBoard::beep()) {
    Serial.println("Beep failed; command cancelled.");
    resetAudioHistory(); busy = false; return;
  }
  resetAudioHistory();
  CommandCapture capture(recording, SPEECH_RMS, SILENCE_MS);
  Serial.println("Listening for command (max 12 s)...");
  const uint32_t started = millis();
  bool done = false, valid = true;
  int16_t stereo[256 * 2];
  while (!done && millis() - started < 14000) {
    size_t frames = 0;
    if (!AudioBoard::read(stereo, 256, frames)) { valid = false; break; }
    for (size_t i = 0; i < frames; ++i) {
      int16_t mono;
      if (resampler->push(stereo[i * 2], mono) && capture.push(removeDC(mono))) {
        done = true; break;
      }
    }
  }
  if (!valid || !done) Serial.println("Audio read/recording timeout; recording discarded.");
  else if (!capture.hasSpeech()) Serial.println("No speech detected; no API request.");
  else {
    const size_t start = capture.start(), count = capture.end() - start;
    Serial.printf("Sending %u PCM bytes...\n", unsigned(count * 2));
    upload(recording + start, count);
  }
  resetAudioHistory();
  busy = false;
}

void audioTask(void *) {
  /*
  Background FreeRTOS task pinned to Core 1 (priority 3) that runs continuously
  alongside appLoop(). In each iteration it:
  1. Checks `requested` to see if the user manually triggered a command over
     serial ('r').
  2. Reads up to 256 stereo I2S frames (left + right channels) from the ES8311
     codec, aborting via fatal() after 10 consecutive read failures.
  3. Extracts the left channel (stereo[i * 2]), downsamples 44.1 kHz -> 16 kHz,
     removes DC offset, and fills the 1-frame internal-RAM `wake_frame` buffer.
  4. Runs WakeNet9 inference whenever `wake_frame` is full; if the wake word is
     detected (`detected == true`), invokes command() to beep, capture up to
     12 s of speech, and upload it over UDP.
  5. Records remaining task stack headroom (`uxTaskGetStackHighWaterMark`) and
     yields Core 1 for 1 ms (`delay(1)`) so appLoop() and the FreeRTOS idle
     watchdog task can run.
  */
  int16_t stereo[256 * 2];
  unsigned read_errors = 0;
  while (true) {
    if (requested.exchange(false)) { command(); continue; }
    size_t frames = 0;
    if (!AudioBoard::read(stereo, 256, frames)) {
      if (++read_errors >= 10) fatal("I2S produced no audio. Check wiring and power, then reset.");
      delay(10); continue;
    }
    read_errors = 0;
    bool detected = false;
    for (size_t i = 0; i < frames; ++i) {
      int16_t mono;
      if (!resampler->push(stereo[i * 2], mono)) continue;
      wake_frame[wake_used++] = removeDC(mono);
      if (wake_used == wake_size) {
        wake_used = 0;
        if (wake->detect(wake_data, wake_frame) == WAKENET_DETECTED) { detected = true; break; }
      }
    }
    if (detected) command();
    // Allow the idle task to run even if inference temporarily leaves an I2S backlog.
    audio_stack_free = uxTaskGetStackHighWaterMark(nullptr);
    delay(1);
  }
}

void record_previous_audio_stage_for_diagnostics() {
  /*
  This function is used to record the previous stage when the device work was interrupted (reset, crash, power on).
  This is used to diagnose the cause of the interruption.
  esp_reset_reason() comes from ESP-IDF API, Espressif's official SDK for ESP32 chips. 
  #include <esp_heap_caps.h>
  #include <esp_random.h>
  #include <esp_system.h>
  #include <esp_core_dump.h>
  #include <esp_wn_iface.h>
  #include <esp_wn_models.h>

  documentation - docs.espressif.com 

  */
    previous_audio_stage = esp_reset_reason() == ESP_RST_POWERON ? 0 : AudioBoard::diagnostic_stage;
    AudioBoard::diagnostic_stage = 0;
} 

bool validate_receiver_token(const char *receiver_token)
{
  /*
  This function is used to validate the receiver token - a 32 character hex string.
  This function validates the length and whether it's realy a hex string.
  */
  if (strlen(receiver_token) != 32) 
  {
    fatal("Config.h: RECEIVER_TOKEN must contain 32 hex characters.");
    return false;
  }
  for (unsigned i = 0; i < 16; ++i) {
    const int a = hexDigit(receiver_token[i * 2]), b = hexDigit(receiver_token[i * 2 + 1]);
    if (a < 0 || b < 0) 
    {
      fatal("Invalid receiver token.");
      return false;
    } 
    token[i] = (a << 4) | b;
  }
  return true;

}

void init_serial() {
  /*
  Opens the serial console used for diagnostics and for the 'r' serial trigger.
  A short delay gives an attached serial monitor time to connect before the
  banner is printed, but we never wait indefinitely (no monitor == no block).
  */
  Serial.begin(115200);
  delay(500);
  Serial.println("Open Lexa / Arduino Nano ESP32 / Atomic Echo Base");
}

void parse_receiver_ip() {
  /*
  Parses RECEIVER_IP from Config.h into the receiver IPAddress used by upload().
  Aborts with fatal() if the configured value is not a valid IPv4 literal.
  */
  if (!receiver.fromString(RECEIVER_IP)) fatal("Config.h: invalid receiver IPv4.");
}

void validate_psram() {
  /*
  Confirms OPI PSRAM is present. The recording buffer and Resampler are
  allocated there, so the firmware cannot run without it. Select the Nano
  ESP32 board with its default OPI PSRAM setting in the Arduino IDE.
  */
  if (!psramFound()) fatal("PSRAM unavailable: select the Nano ESP32 board with its default OPI PSRAM.");
}

void validate_audio_config() {
  /*
  Validates the audio-related Config.h values that are easy to mis-set:
  MIC_GAIN_DB (0 or 6) and the speech/silence/beep thresholds. Aborts on the
  first bad value so a misconfiguration is obvious on reset rather than
  surfacing later as a silent capture failure.
  */
  if (MIC_GAIN_DB != 0 && MIC_GAIN_DB != 6) fatal("MIC_GAIN_DB must be 0 or 6.");
  if (SPEECH_RMS < 50 || SPEECH_RMS > 16000 || SILENCE_MS < 300 || SILENCE_MS > 3000 || BEEP_VOLUME > 100)
    fatal("Invalid audio thresholds/volume in Config.h.");
}

void allocate_recording_buffers() {
  /*
  Allocates the command recording buffer and the Resampler in PSRAM. The
  recording buffer holds a full command (CommandCapture::MAX_SAMPLES) and the
  Resampler is placement-new'd into its own PSRAM block so its state lives in
  PSRAM rather than scarce internal RAM.
  */
  recording = static_cast<int16_t *>(heap_caps_malloc(CommandCapture::MAX_SAMPLES * 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  void *resampler_memory = heap_caps_malloc(sizeof(Resampler), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!recording || !resampler_memory) fatal("Not enough PSRAM for recording/resampler.");
  resampler = new (resampler_memory) Resampler();
  resampler->begin();
}

void load_wake_word_model() {
  /*
  Loads the embedded Alexa wake-word model and brings up WakeNet9. The model
  travels inside the .ino firmware (no data partition upload needed). After
  creating the detector the input format (16 kHz, mono) and frame size are
  validated, and the inference frame buffer is allocated in internal RAM so
  the WakeNet hot path never touches PSRAM.
  */
  auto *models = srmodel_load(kAlexaModel);
  if (!models) fatal("Cannot load embedded Alexa model.");
  char *name = esp_srmodel_filter(models, "wn9", "alexa");
  if (!name) fatal("Alexa model not found.");
  wake = esp_wn_handle_from_name(name);
  if (!wake) fatal("Core lacks WakeNet9; install esp32 by Espressif Systems 3.3.8.");
  wake_name = name;
  wake_data = wake->create(name, DET_MODE_90);
  if (!wake_data) fatal("WakeNet allocation failed.");
  if (wake->get_samp_rate(wake_data) != 16000 || wake->get_channel_num(wake_data) != 1)
    fatal("Unexpected WakeNet input format.");
  const int frame_size = wake->get_samp_chunksize(wake_data);
  if (frame_size <= 0 || frame_size > 2048) fatal("Unexpected WakeNet frame size.");
  wake_size = frame_size;
  wake_frame = static_cast<int16_t *>(heap_caps_malloc(wake_size * 2, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (!wake_frame) fatal("Cannot allocate inference frame.");
}

void init_audio_board() {
  /*
  Starts the I2S audio path. Microphone gain and beep volume come from
  Config.h and were validated earlier in validate_audio_config().
  */
  if (!AudioBoard::begin(MIC_GAIN_DB, BEEP_VOLUME)) fatal("Audio initialization failed.");
}

void init_wifi() {
  /*
  Brings up Wi-Fi in station mode with power save off (for low-latency wake
  inference) and auto-reconnect on. A disconnect handler logs the reason code
  so transient drops are visible on the serial console.
  */
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.onEvent([](WiFiEvent_t event, WiFiEventInfo_t info) {
    if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED)
      Serial.printf("Wi-Fi disconnected: reason=%u\n", unsigned(info.wifi_sta_disconnected.reason));
  });
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void start_audio_task() {
  /*
  Pins the audioTask (wake-word inference + command capture) to core 1 so it
  never contends with Wi-Fi on core 0. 16 KB of stack covers the WakeNet
  inference path with headroom.
  */
  if (xTaskCreatePinnedToCore(audioTask, "lexa_audio", 16384, nullptr, 3, nullptr, 1) != pdPASS)
    fatal("Cannot start audio task.");
}
}// namespace

void appSetup() {
  // Setup proceeds in named stages so each step can be read and debugged in
  // isolation. Any failure aborts via fatal() with a specific message.
  record_previous_audio_stage_for_diagnostics();
  init_serial();
  validate_receiver_token(RECEIVER_TOKEN);
  parse_receiver_ip();
  validate_psram();
  validate_audio_config();
  allocate_recording_buffers();
  load_wake_word_model();
  init_audio_board();
  init_wifi();
  start_audio_task();
  Serial.printf("WakeNet model: %s; frame: %u samples. Say Alexa, or send r over serial.\n",
                wake_name, unsigned(wake_size));
}

void appLoop() {
  static uint32_t retry_at = 0;
  static bool was_connected = false;
  const bool connected = WiFi.status() == WL_CONNECTED;
  if (connected && !was_connected) Serial.printf("Wi-Fi connected. IP: %s\n", WiFi.localIP().toString().c_str());
  if (!connected && millis() - retry_at > 10000) {
    retry_at = millis();
    Serial.println("Waiting for Wi-Fi; check Config.h. Retrying...");
    WiFi.reconnect();
  }
  was_connected = connected;
  while (Serial.available()) {
    const char c = Serial.read();
    if ((c == 'r' || c == 'R') && !busy.load()) requested = true;
    if (c == 'c' || c == 'C') {
      esp_core_dump_summary_t summary{};
      if (esp_core_dump_get_summary(&summary) == ESP_OK) {
        Serial.printf("PANIC task=%s pc=%08lx cause=%lu address=%08lx\n", summary.exc_task,
          (unsigned long)summary.exc_pc, (unsigned long)summary.ex_info.exc_cause,
          (unsigned long)summary.ex_info.exc_vaddr);
        for (unsigned i = 0; i < summary.exc_bt_info.depth; ++i)
          Serial.printf("BT %08lx\n", (unsigned long)summary.exc_bt_info.bt[i]);
      } else Serial.println("No readable core dump.");
    }
    if (c == 's' || c == 'S') Serial.printf("STATUS reset=%d previous_stage=%lu current_stage=%lu busy=%d wifi=%d ip=%s stack_free=%u heap_free=%u psram_free=%u\n",
      int(esp_reset_reason()), (unsigned long)previous_audio_stage, (unsigned long)AudioBoard::diagnostic_stage,
      int(busy.load()), int(connected), WiFi.localIP().toString().c_str(), audio_stack_free.load(),
      unsigned(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)), unsigned(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
  }
  delay(10);
}
