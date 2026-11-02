#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s.h>
#include "src/codec/es8311.h"

// Verified local wiring from AtomicEchoBase_NanoESP32_Wiring.md.
// The legacy ESP-IDF I2S API takes RAW GPIOs, while Wire takes Arduino labels.
namespace AudioBoard {
extern uint32_t diagnostic_stage;
constexpr int RATE = 44100;
constexpr i2s_port_t PORT = I2S_NUM_0;
constexpr int BCLK = 5;  // Nano D2
constexpr int WS = 6;    // Nano D3
constexpr int DIN = 7;   // Nano D4 <- base DOUT, microphone
constexpr int DOUT = 8;  // Nano D5 -> base DIN, speaker
bool begin(unsigned mic_gain, unsigned volume);
bool muteDac(bool mute);  // Never disables microphone power.
bool read(int16_t *stereo, size_t frames, size_t &received);
bool beep();
void discardInput();
}
