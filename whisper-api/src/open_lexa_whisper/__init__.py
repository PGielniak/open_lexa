"""Hailo Whisper exposed as an OpenAI-compatible transcription API for Open Lexa."""

# Audio contract of the UDP receiver (open_lexa.protocol): mono PCM16LE at 16 kHz,
# at most 12 s. Repeated here so this image needs neither openai nor SMB clients.
RATE = 16000
MAX_BYTES = RATE * 2 * 12
