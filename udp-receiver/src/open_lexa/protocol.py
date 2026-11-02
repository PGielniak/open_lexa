"""Bounded, retransmittable UDP transport for mono 16 kHz signed PCM16LE.

A shared token filters unrelated senders. It does not encrypt/authenticate LAN traffic.
"""
from dataclasses import dataclass, field
import hmac
import struct
from typing import Callable

RATE = 16000
CHUNK = 1024
MAX_BYTES = RATE * 2 * 12
HEADER = struct.Struct('!4s16sIIIH')
ACK = struct.Struct('!4s16sII')
Peer = tuple[str, int]


def parse_token(value: str) -> bytes:
    try:
        token = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError('OPEN_LEXA_TOKEN must be 32 hex characters') from exc
    if len(value) != 32 or len(token) != 16:
        raise ValueError('OPEN_LEXA_TOKEN must be 32 hex characters')
    return token


def packet(token: bytes, session: int, pcm: bytes, sequence: int) -> bytes:
    payload = pcm[sequence * CHUNK:(sequence + 1) * CHUNK]
    return HEADER.pack(b'OLX1', token, session, len(pcm), sequence, len(payload)) + payload


@dataclass(frozen=True)
class Recording:
    peer: Peer
    session: int
    pcm: bytes


@dataclass
class Partial:
    total: int
    created: float
    updated: float
    chunks: dict[int, bytes] = field(default_factory=dict)


class Receiver:
    def __init__(self, token: bytes, *, max_sessions: int = 4):
        if len(token) != 16:
            raise ValueError('token must be 16 bytes')
        self.token = token
        self.max_sessions = max_sessions
        self.pending: dict[tuple[Peer, int], Partial] = {}
        self.completed: dict[tuple[Peer, int], tuple[float, int]] = {}

    def expire(self, now: float) -> int:
        stale = [k for k, v in self.pending.items()
                 if now - v.updated > 5 or now - v.created > 35]
        for key in stale:
            del self.pending[key]
        self.completed = {k: v for k, v in self.completed.items() if now - v[0] < 120}
        return len(stale)

    def feed(self, data: bytes, peer: Peer, now: float,
             deliver: Callable[[Recording], bool]) -> bytes | None:
        """Return ACK only for accepted data; deliver once, after every chunk arrived.

        deliver must enqueue without blocking and return False on backpressure.
        A lost final ACK is answered from completed without repeating STT.
        """
        self.expire(now)
        if len(data) < HEADER.size:
            return None
        magic, token, session, total, seq, size = HEADER.unpack_from(data)
        if magic != b'OLX1' or not hmac.compare_digest(token, self.token):
            return None
        if not 6400 <= total <= MAX_BYTES or total % 2:
            return None
        count = (total + CHUNK - 1) // CHUNK
        if seq >= count or size != min(CHUNK, total - seq * CHUNK):
            return None
        if len(data) != HEADER.size + size:
            return None
        key = (peer, session)
        ack = ACK.pack(b'OLA1', self.token, session, seq)
        if key in self.completed:
            return ack if self.completed[key][1] == total else None
        if key not in self.pending:
            if len(self.pending) >= self.max_sessions:
                return None
            self.pending[key] = Partial(total, now, now)
        partial = self.pending[key]
        payload = data[HEADER.size:]
        if partial.total != total or (seq in partial.chunks and partial.chunks[seq] != payload):
            return None
        partial.updated = now
        partial.chunks[seq] = payload
        if len(partial.chunks) == count:
            # Keep the partial when the STT queue is full; sender retries its final packet.
            recording = Recording(peer, session, b''.join(partial.chunks[i] for i in range(count)))
            if not deliver(recording):
                return None
            del self.pending[key]
            self.completed[key] = (now, total)
            # Bounded replay cache. Normal device throughput is far below this limit.
            if len(self.completed) > 1024:
                del self.completed[next(iter(self.completed))]
        return ack
