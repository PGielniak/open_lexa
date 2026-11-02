#!/usr/bin/env python3
"""Exercise every local WAV recording through the deployed receiver/STT path.

The UDP protocol acknowledges delivery only; it does not return transcripts.
This runner therefore verifies that the deployed receiver's health counters
observe and transcribe each recording without adding a backend failure. It can
also verify the separate Hailo Whisper API used by the split architecture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import wave

from open_lexa.protocol import MAX_BYTES


ROOT = Path(__file__).resolve().parents[2]
SEND_WAV = Path(__file__).resolve().parent / 'send_wav.py'
CONFIG = ROOT / 'firmware' / 'src' / 'OpenLexaNano' / 'Config.h'
MIN_BYTES = 6_400


class TestFailure(RuntimeError):
    """The deployed receiver did not process a recording as expected."""


def run(command: list[str], *, timeout: float = 60) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise TestFailure(f'{command[0]!r} is not installed or is not on PATH') from exc
    except subprocess.TimeoutExpired as exc:
        raise TestFailure(f'command timed out after {timeout:g}s: {command[0]}') from exc
    if result.returncode:
        detail = '\n'.join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        raise TestFailure(f'command failed ({result.returncode}): {" ".join(command)}\n{detail}')
    return result.stdout


def firmware_endpoint() -> tuple[str, int]:
    try:
        text = CONFIG.read_text()
    except FileNotFoundError as exc:
        raise TestFailure(f'firmware config not found: {CONFIG}') from exc
    host = re.search(r'RECEIVER_IP\[\]\s*=\s*"([^"]+)"', text)
    port = re.search(r'UDP_PORT\s*=\s*(\d+)', text)
    if host is None or port is None:
        raise TestFailure(f'could not read RECEIVER_IP/UDP_PORT from {CONFIG}')
    return host.group(1), int(port.group(1))


def valid_recordings(directory: Path) -> list[tuple[Path, float]]:
    if not directory.is_dir():
        raise TestFailure(f'recordings directory does not exist: {directory}')
    recordings: list[tuple[Path, float]] = []
    for path in sorted(directory.glob('*.wav')):
        try:
            with wave.open(str(path), 'rb') as source:
                format_ = (source.getnchannels(), source.getsampwidth(),
                           source.getframerate(), source.getcomptype())
                byte_count = source.getnframes() * source.getnchannels() * source.getsampwidth()
        except (wave.Error, OSError) as exc:
            raise TestFailure(f'{path}: cannot read WAV: {exc}') from exc
        if format_ != (1, 2, 16_000, 'NONE'):
            raise TestFailure(
                f'{path}: requires uncompressed mono PCM16/16 kHz, got {format_}')
        if not MIN_BYTES <= byte_count <= MAX_BYTES:
            raise TestFailure(
                f'{path}: requires 0.2–12 s of PCM, got {byte_count / 32_000:.2f} s')
        recordings.append((path, byte_count / 32_000))
    if not recordings:
        raise TestFailure(f'no .wav recordings found in {directory}')
    return recordings


def receiver_health(namespace: str, deployment: str) -> dict:
    code = (
        "import urllib.request; "
        "print(urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=5).read().decode())"
    )
    output = run([
        'kubectl', '-n', namespace, 'exec', f'deployment/{deployment}', '--',
        'python3', '-c', code,
    ], timeout=20)
    try:
        state = json.loads(output)
    except json.JSONDecodeError as exc:
        raise TestFailure(f'receiver returned invalid health JSON: {output!r}') from exc
    if not isinstance(state, dict):
        raise TestFailure(f'receiver returned unexpected health JSON: {state!r}')
    return state


def metric(state: dict, name: str) -> int:
    value = state.get(name)
    if not isinstance(value, int):
        raise TestFailure(f'receiver health is missing integer {name!r}: {state!r}')
    return value


def wait_for_transcription(*, namespace: str, deployment: str,
                           received: int, transcriptions: int, failures: int,
                           timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        state = receiver_health(namespace, deployment)
        current_failures = metric(state, 'failures')
        if current_failures > failures:
            raise TestFailure(
                f'receiver failure count rose from {failures} to {current_failures}: '
                f'{state.get("last_error") or "unknown error"}')
        if (metric(state, 'recordings_received') >= received
                and metric(state, 'transcriptions') >= transcriptions):
            return state
        if time.monotonic() >= deadline:
            raise TestFailure(
                f'timed out after {timeout:g}s waiting for one recording/transcription; '
                f'last health: {state}')
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings-dir', type=Path, default=ROOT / 'recordings')
    parser.add_argument('--host', help='UDP receiver host (default: firmware Config.h)')
    parser.add_argument('--port', type=int, help='UDP receiver port (default: firmware Config.h)')
    parser.add_argument('--namespace', default='open-lexa')
    parser.add_argument('--deployment', default='open-lexa')
    parser.add_argument('--timeout', type=float, default=45,
                        help='seconds to wait for each transcription (default: 45)')
    parser.add_argument('--ack-timeout', type=float, default=0.25,
                        help='seconds to wait per UDP ACK attempt (default: 0.25)')
    parser.add_argument('--ack-retries', type=int, default=40,
                        help='UDP ACK attempts per chunk (default: 40)')
    parser.add_argument('--limit', type=int, help='test only the first N recordings')
    parser.add_argument('--expected-backend',
                        help='require this receiver backend, e.g. http or openai')
    parser.add_argument('--whisper-deployment',
                        help='also require this separate Whisper API Deployment to be ready')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.ack_timeout <= 0:
        parser.error('--ack-timeout must be positive')
    if args.ack_retries <= 0:
        parser.error('--ack-retries must be positive')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')

    firmware_host, firmware_port = firmware_endpoint()
    host, port = args.host or firmware_host, args.port or firmware_port
    recordings = valid_recordings(args.recordings_dir.expanduser().resolve())
    if args.limit is not None:
        recordings = recordings[:args.limit]

    state = receiver_health(args.namespace, args.deployment)
    if state.get('ready') is not True:
        raise TestFailure(f'receiver is not ready: {state}')
    if args.expected_backend and state.get('backend') != args.expected_backend:
        raise TestFailure(
            f'receiver backend is {state.get("backend")!r}, expected {args.expected_backend!r}: {state}')
    if args.whisper_deployment:
        whisper = receiver_health(args.namespace, args.whisper_deployment)
        if whisper.get('ready') is not True:
            raise TestFailure(f'Whisper API is not ready: {whisper}')
    received = metric(state, 'recordings_received')
    transcriptions = metric(state, 'transcriptions')
    failures = metric(state, 'failures')
    backend = state.get('backend', 'unknown')
    print(f'Testing {len(recordings)} recording(s) through {backend} at {host}:{port}')

    for index, (path, seconds) in enumerate(recordings, start=1):
        run([
            sys.executable, str(SEND_WAV), str(path), '--host', host, '--port', str(port),
            '--ack-timeout', str(args.ack_timeout), '--retries', str(args.ack_retries),
        ])
        received += 1
        transcriptions += 1
        state = wait_for_transcription(
            namespace=args.namespace,
            deployment=args.deployment,
            received=received,
            transcriptions=transcriptions,
            failures=failures,
            timeout=args.timeout,
        )
        print(f'[{index}/{len(recordings)}] {path.name} ({seconds:.2f}s): transcribed')

    print(
        f'PASS: {len(recordings)} recording(s) were transcribed by the {backend} backend '
        f'with {metric(state, "failures")} receiver failures.')


if __name__ == '__main__':
    try:
        main()
    except TestFailure as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        raise SystemExit(1)
