#!/usr/bin/env python3
"""Send an existing PCM16 mono/16k WAV using the same UDP protocol as the firmware."""
import argparse
import os
import secrets
import socket
import wave

from dotenv import load_dotenv
from open_lexa.protocol import ACK, CHUNK, MAX_BYTES, packet, parse_token

parser = argparse.ArgumentParser()
parser.add_argument('wav')
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', type=int, default=12345)
parser.add_argument('--ack-timeout', type=float, default=0.1,
                    help='seconds to wait for each ACK attempt (default: 0.1)')
parser.add_argument('--retries', type=int, default=10,
                    help='ACK attempts per UDP chunk (default: 10)')
args = parser.parse_args()
if args.ack_timeout <= 0:
    parser.error('--ack-timeout must be positive')
if args.retries <= 0:
    parser.error('--retries must be positive')
load_dotenv('.env')
token = parse_token(os.environ.get('OPEN_LEXA_TOKEN', ''))
with wave.open(args.wav, 'rb') as source:
    if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, 16000, 'NONE'):
        parser.error('WAV must be uncompressed mono PCM16 at 16000 Hz')
    pcm = source.readframes(16000 * 12 + 1)
if not 6400 <= len(pcm) <= MAX_BYTES:
    parser.error('Recording must last 0.2–12 seconds')
session = secrets.randbits(32)
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.connect((args.host, args.port))
    sock.settimeout(args.ack_timeout)
    for seq in range((len(pcm) + CHUNK - 1) // CHUNK):
        expected = ACK.pack(b'OLA1', token, session, seq)
        for attempt in range(args.retries):
            sock.send(packet(token, session, pcm, seq))
            try:
                if sock.recv(1024) == expected:
                    break
            except socket.timeout:
                pass
        else:
            raise SystemExit(f'No ACK for chunk {seq}; check receiver, token and port')
print('Recording delivered')
