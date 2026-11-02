#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

// Streaming rational resampler: 44100 -> 16000 Hz, 160 fractional phases.
// A 96-tap Blackman-windowed low-pass prevents frequencies above 8 kHz from aliasing.
// Coefficients are computed once at boot, not in the audio callback.
class Resampler {
 public:
  static constexpr unsigned TAPS = 96;
  void begin() {
    constexpr double pi = 3.14159265358979323846;
    constexpr double cutoff = 6800.0 / 44100.0;
    for (unsigned phase = 0; phase < 160; ++phase) {
      double sum = 0;
      for (unsigned k = 0; k < TAPS; ++k) {
        const double x = double(k) - (TAPS - 1) / 2.0 - double(phase) / 160.0;
        const double window = 0.42 - 0.5 * std::cos(2 * pi * k / (TAPS - 1))
                             + 0.08 * std::cos(4 * pi * k / (TAPS - 1));
        const double sinc = std::abs(x) < 1e-12 ? 2 * cutoff : std::sin(2 * pi * cutoff * x) / (pi * x);
        taps_[phase][k] = static_cast<float>(sinc * window);
        sum += taps_[phase][k];
      }
      for (unsigned k = 0; k < TAPS; ++k) taps_[phase][k] /= sum;
    }
    reset();
  }
  void reset() { std::memset(history_, 0, sizeof(history_)); pos_ = phase_ = 0; }
  bool push(int16_t sample, int16_t &out) {
    history_[pos_] = sample;
    const unsigned newest = pos_;
    pos_ = (pos_ + 1) % TAPS;
    phase_ += 160;
    if (phase_ < 441) return false;
    phase_ -= 441;  // remainder in [0,159] identifies sub-sample delay
    float value = 0;
    unsigned index = newest;
    for (unsigned k = 0; k < TAPS; ++k) {
      value += history_[index] * taps_[phase_][k];
      index = index == 0 ? TAPS - 1 : index - 1;
    }
    value = std::fmax(-32768.0f, std::fmin(32767.0f, value));
    out = static_cast<int16_t>(std::lrint(value));
    return true;
  }
 private:
  float taps_[160][TAPS]{};
  int16_t history_[TAPS]{};
  unsigned pos_{0}, phase_{0};
};
