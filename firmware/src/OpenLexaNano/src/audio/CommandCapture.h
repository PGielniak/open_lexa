#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

class CommandCapture {
 public:
  static constexpr size_t MAX_SAMPLES = 16000 * 12;
  CommandCapture(int16_t *buffer, unsigned rms, unsigned silence_ms)
      : buffer_(buffer), threshold_(uint64_t(rms) * rms), silence_samples_(silence_ms * 16) {}
  // 20 ms windows; includes a bounded initial wait and uses sample time, not CPU time.
  bool push(int16_t sample) {
    if (done_) return true;
    buffer_[count_++] = sample;
    energy_ += int64_t(sample) * sample;
    if (count_ % 320 == 0) {
      if (energy_ / 320 >= threshold_) {
        first_voice_ = std::min(first_voice_, count_ - 320);
        last_voice_ = count_;
        voiced_ += 320;
      }
      energy_ = 0;
      done_ = count_ >= MAX_SAMPLES || (!hasSpeech() && count_ >= 64000) ||
              (hasSpeech() && count_ - last_voice_ >= silence_samples_);
    }
    return done_;
  }
  bool hasSpeech() const { return voiced_ >= 3200; }
  size_t start() const { return first_voice_ > 4000 ? first_voice_ - 4000 : 0; }
  size_t end() const { return std::min(count_, last_voice_ + 4800); }
  size_t count() const { return count_; }
 private:
  int16_t *buffer_;
  uint64_t threshold_, energy_{0};
  size_t silence_samples_;
  size_t count_{0}, first_voice_{MAX_SAMPLES}, last_voice_{0}, voiced_{0};
  bool done_{false};
};
