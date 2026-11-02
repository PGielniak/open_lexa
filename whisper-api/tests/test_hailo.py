from pathlib import Path

import pytest

from open_lexa_whisper.hailo import ConfigError, HailoWhisperTranscriber, create_transcriber


def test_hailo_backend_requires_an_existing_hef(tmp_path: Path):
    with pytest.raises(ConfigError, match='not found'):
        create_transcriber(env={'HAILO_WHISPER_HEF': str(tmp_path / 'missing.hef')})
    hef = tmp_path / 'Whisper-Base.hef'
    hef.write_bytes(b'not a real hef')
    transcriber = create_transcriber(env={'HAILO_WHISPER_HEF': str(hef)})
    assert isinstance(transcriber, HailoWhisperTranscriber)
    assert transcriber.normalize_peak == 0.9


def test_hailo_normalize_peak_must_be_a_fraction(tmp_path: Path):
    hef = tmp_path / 'Whisper-Base.hef'
    hef.write_bytes(b'not a real hef')
    env = {'HAILO_WHISPER_HEF': str(hef)}
    assert create_transcriber(env={**env, 'HAILO_WHISPER_NORMALIZE_PEAK': '0'}).normalize_peak == 0
    with pytest.raises(ConfigError, match='between 0 and 1'):
        create_transcriber(env={**env, 'HAILO_WHISPER_NORMALIZE_PEAK': '1.01'})
    with pytest.raises(ConfigError, match='must be numeric'):
        create_transcriber(env={**env, 'HAILO_WHISPER_NORMALIZE_PEAK': 'loud'})
