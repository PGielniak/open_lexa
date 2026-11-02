"""Exercise the real CLI/socket/worker/WAV path, without microphone or OpenAI calls."""
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time
import wave

from open_lexa.protocol import ACK, CHUNK, MAX_BYTES, packet


def test_cli_receives_full_length_recording_and_deduplicates_final_ack(tmp_path):
    token = bytes(range(16))
    pcm = bytes(range(256)) * (MAX_BYTES // 256)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    env = dict(os.environ, OPEN_LEXA_TOKEN=token.hex(), PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    proc = subprocess.Popen([sys.executable, '-m', 'open_lexa.server', '--dry-run', '--bind', '127.0.0.1',
                             '--port', str(port), '--health-port', '0', '--save-dir', str(tmp_path),
                             '--env-file', str(tmp_path / 'absent')],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert select.select([proc.stderr], [], [], 10)[0], 'CLI did not start'
        assert 'Listening on' in proc.stderr.readline()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.connect(('127.0.0.1', port))
            client.settimeout(2)
            for seq in range((len(pcm) + CHUNK - 1) // CHUNK):
                data = packet(token, 123, pcm, seq)
                client.send(data)
                assert client.recv(2048) == ACK.pack(b'OLA1', token, 123, seq)
            # Simulate loss of the last acknowledgement and retransmission.
            client.send(data)
            assert client.recv(2048) == ACK.pack(b'OLA1', token, 123, seq)
        assert select.select([proc.stdout], [], [], 5)[0], 'Worker did not receive recording'
        assert '12.00 s' in proc.stdout.readline()
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode == 0, stderr
        assert not stdout  # Duplicate final packet must not create another job.
        files = list(tmp_path.glob('*.wav'))
        assert len(files) == 1
        with wave.open(str(files[0]), 'rb') as result:
            assert result.readframes(result.getnframes()) == pcm
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
