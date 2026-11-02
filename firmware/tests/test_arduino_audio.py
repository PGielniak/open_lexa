"""Compile and exercise the exact portable audio code used by the Arduino sketch."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_resampler_and_recording_boundaries(tmp_path):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('g++ needed for native firmware DSP tests')
    source = tmp_path / 'audio_test.cpp'
    source.write_text(r'''
#include "Resampler.h"
#include "CommandCapture.h"
#include <cassert>
#include <iostream>
#include <vector>
static constexpr double PI = 3.14159265358979323846;
std::vector<int16_t> tone(double frequency) {
  Resampler r; r.begin();
  std::vector<int16_t> result;
  for (int n=0; n<44100; ++n) {
    int16_t out;
    if (r.push(int16_t(12000 * std::sin(2*PI*frequency*n/44100)), out)) result.push_back(out);
  }
  assert(result.size() == 16000);
  return result;
}
int main() {
  Resampler r; r.begin();
  int16_t out = 0;
  unsigned produced = 0;
  for (int n=0; n<441000; ++n) {
    if (r.push(10000, out)) { ++produced; if (n>300) assert(std::abs(int(out)-10000)<=1); }
  }
  assert(produced == 160000); // No accumulated timing drift over ten seconds.
  r.reset();
  for (int n=0; n<441; ++n) if(r.push(0,out)) assert(out == 0);
  auto low = tone(1000), high = tone(10000);
  double s=0,c=0,power=0,alias=0;
  for (size_t i=1000;i<low.size();++i) {
    const double theta = 2*PI*1000*i/16000;
    s += low[i]*std::sin(theta); c += low[i]*std::cos(theta);
    power += double(low[i])*low[i]; alias += double(high[i])*high[i];
  }
  const double n=low.size()-1000;
  const double amplitude = 2*std::sqrt(s*s+c*c)/n;
  assert(amplitude>11800 && amplitude<12200);
  const double residual = std::sqrt(std::max(0.0, power/n-amplitude*amplitude/2));
  assert(residual < 12);  // Detects wrong fractional-phase sign/jitter.
  assert(std::sqrt(alias/n) < 20); // More than 50 dB rejection above Nyquist.
  std::vector<int16_t> buffer(CommandCapture::MAX_SAMPLES);
  CommandCapture silence(buffer.data(),500,1000);
  for (int i=0;i<64000;++i) assert(silence.push(0) == (i==63999));
  assert(!silence.hasSpeech());
  CommandCapture speech(buffer.data(),500,1000);
  for(int i=0;i<16000;++i) assert(!speech.push(0));
  for(int i=0;i<16000;++i) assert(!speech.push(i%2?3000:-3000));
  for(int i=0;i<16000;++i) assert(speech.push(0) == (i==15999));
  assert(speech.hasSpeech());
  assert(speech.start()==12000 && speech.end()==36800);
  CommandCapture continuous(buffer.data(),500,1000);
  for(size_t i=0;i<buffer.size();++i) assert(continuous.push(3000) == (i==buffer.size()-1));
  assert(continuous.push(3000)); // Already full: must not overrun.
  assert(continuous.count()==buffer.size());
  std::cout << "DSP OK: amplitude=" << amplitude << " residual=" << residual << " alias=" << std::sqrt(alias/n) << '\n';
}
''')
    exe = tmp_path / 'audio_test'
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'src/OpenLexaNano/src/audio'),
                    str(source), '-o', str(exe)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert 'DSP OK' in result.stdout
