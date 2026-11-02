#include "AudioBoard.h"
#include <cmath>
#include <algorithm>

namespace AudioBoard {
RTC_NOINIT_ATTR uint32_t diagnostic_stage;
static es8311_handle_t codec = nullptr;
static bool writeReg(uint8_t device, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(device);
  Wire.write(reg); Wire.write(value);
  return Wire.endTransmission() == 0;
}
static bool present(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}
bool muteDac(bool mute) { return codec && es8311_voice_mute(codec, mute) == ESP_OK; }

bool begin(unsigned mic_gain, unsigned volume) {
  if (!Wire.begin(A4, A5, 100000) || !present(0x18) || !present(0x43)) {
    Serial.println("I2C: expected ES8311 0x18 and PI4IOE 0x43. Check A4/A5 and power.");
    return false;
  }
  i2s_config_t config{};
  config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_RX);
  config.sample_rate = RATE;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  config.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
  config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  config.dma_buf_count = 8;
  config.dma_buf_len = 256;
  config.use_apll = false;
  config.tx_desc_auto_clear = true;  // Transmit zeros after the beep, never repeat old DMA audio.
  i2s_pin_config_t pins{};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = BCLK; pins.ws_io_num = WS;
  pins.data_out_num = DOUT; pins.data_in_num = DIN;
  if (i2s_driver_install(PORT, &config, 0, nullptr) != ESP_OK || i2s_set_pin(PORT, &pins) != ESP_OK) return false;
  i2s_zero_dma_buffer(PORT);
  es8311_set_twowire(&Wire);
  codec = es8311_create(I2C_NUM_0, 0x18);
  if (!codec) return false;
  es8311_clock_config_t clock{};
  clock.mclk_from_mclk_pin = false;
  clock.sample_frequency = RATE;
  // Deliberately retain the verified M5 configuration: codec 32-bit, bus stereo 16-bit at 44.1 kHz.
  if (es8311_init(codec, &clock, ES8311_RESOLUTION_32, ES8311_RESOLUTION_32) != ESP_OK ||
      es8311_microphone_config(codec, false) != ESP_OK ||
      es8311_microphone_gain_set(codec, mic_gain == 0 ? ES8311_MIC_GAIN_0DB : ES8311_MIC_GAIN_6DB) != ESP_OK ||
      es8311_voice_volume_set(codec, volume, nullptr) != ESP_OK) return false;
  // Same expander setup as the working M5 driver. Keep ALL powered rails enabled.
  // M5EchoBase::setMute(true) would write 0 to reg 0x05 and kill the microphone.
  if (!writeReg(0x43, 0x07, 0x00) || !writeReg(0x43, 0x0d, 0xff) ||
      !writeReg(0x43, 0x03, 0x6f) || !writeReg(0x43, 0x05, 0xff)) return false;
  return muteDac(true);
}

bool read(int16_t *stereo, size_t frames, size_t &received) {
  size_t bytes = 0;
  const auto result = i2s_read(PORT, stereo, frames * 4, &bytes, pdMS_TO_TICKS(100));
  received = bytes / 4;
  return result == ESP_OK && bytes % 4 == 0 && received > 0;
}

void discardInput() {
  int16_t scratch[256 * 2];
  size_t bytes;
  // Bounded flush: enough for all eight DMA buffers, even with audio arriving meanwhile.
  for (unsigned i = 0; i < 12; ++i) {
    bytes = 0;
    i2s_read(PORT, scratch, sizeof(scratch), &bytes, 0);
    if (!bytes) break;
  }
}

bool beep() {
  diagnostic_stage = 100;
  if (!muteDac(false)) return false;
  diagnostic_stage = 101;
  int16_t data[256 * 2];
  constexpr size_t count = RATE / 10;
  bool ok = true;
  for (size_t offset = 0; offset < count && ok; offset += 256) {
    const size_t n = std::min(size_t(256), count - offset);
    for (size_t i = 0; i < n; ++i) {
      const size_t t = offset + i;
      const float env = std::min(1.0f, std::min(float(t), float(count - 1 - t)) / 441.0f);
      const int16_t sample = static_cast<int16_t>(std::sin(2 * 3.141592653589793 * 880 * t / RATE) * 2000 * env);
      data[2*i] = data[2*i+1] = sample;
    }
    diagnostic_stage = 102;
    size_t written = 0;
    ok = i2s_write(PORT, data, n * 4, &written, pdMS_TO_TICKS(500)) == ESP_OK && written == n * 4;
  }
  // i2s_write can return with up to 8*256 frames pending in DMA (~47 ms).
  diagnostic_stage = 103;
  delay(70);
  diagnostic_stage = 104;
  const bool muted = muteDac(true);
  diagnostic_stage = 105;
  i2s_zero_dma_buffer(PORT);
  delay(100);  // Let the speaker/ringing settle before capturing a command.
  diagnostic_stage = 106;
  discardInput();
  diagnostic_stage = 107;
  return ok && muted;
}
}
