import random

import pytest

from open_lexa.protocol import ACK, CHUNK, HEADER, MAX_BYTES, Receiver, packet

TOKEN = bytes(range(16))
PEER = ('127.0.0.1', 1234)
PCM = bytes(range(256)) * 49  # Last chunk is partial.


def packets(pcm=PCM, session=7):
    return [packet(TOKEN, session, pcm, n) for n in range((len(pcm) + CHUNK - 1) // CHUNK)]


def test_reorder_duplicates_and_lost_final_ack_deliver_exactly_once():
    receiver = Receiver(TOKEN)
    delivered = []
    send = packets()
    random.Random(42).shuffle(send)
    for data in send + send:
        ack = receiver.feed(data, PEER, 0, lambda r: delivered.append(r) or True)
        assert ack is not None
        assert ACK.unpack(ack)[2:] == (7, HEADER.unpack_from(data)[4])
    assert len(delivered) == 1
    assert delivered[0].pcm == PCM


def test_missing_packet_never_submits_partial_audio_and_expires():
    receiver = Receiver(TOKEN)
    for data in packets()[1:]:
        receiver.feed(data, PEER, 0, lambda _: pytest.fail('Incomplete recording delivered'))
    assert receiver.expire(6) == 1
    assert not receiver.pending


@pytest.mark.parametrize('corrupt', [
    lambda p: p[:12],
    lambda p: b'BAD!' + p[4:],
    lambda p: p[:4] + b'X' * 16 + p[20:],
    lambda p: p[:-1],
    lambda p: p + b'X',
    lambda p: HEADER.pack(b'OLX1', TOKEN, 7, MAX_BYTES + 2, 0, 1024) + p[34:],
    lambda p: HEADER.pack(b'OLX1', TOKEN, 7, len(PCM), 999999, 1024) + p[34:],
    lambda p: HEADER.pack(b'OLX1', TOKEN, 7, len(PCM) + 1, 0, 1024) + p[34:],
])
def test_malformed_packets_do_not_allocate_or_deliver(corrupt):
    receiver = Receiver(TOKEN)
    assert receiver.feed(corrupt(packets()[0]), PEER, 0, lambda _: pytest.fail()) is None
    assert not receiver.pending


def test_queue_backpressure_retries_final_packet():
    receiver = Receiver(TOKEN)
    send = packets()
    for data in send[:-1]:
        assert receiver.feed(data, PEER, 0, lambda _: False)
    assert receiver.feed(send[-1], PEER, 0, lambda _: False) is None
    delivered = []
    assert receiver.feed(send[-1], PEER, 0.1, lambda r: delivered.append(r) or True)
    assert len(delivered) == 1


def test_session_isolation_and_memory_bound():
    receiver = Receiver(TOKEN, max_sessions=1)
    assert receiver.feed(packets()[0], PEER, 0, lambda _: True)
    assert receiver.feed(packets(session=8)[0], PEER, 0, lambda _: True) is None
    assert receiver.feed(packets()[0], ('127.0.0.2', 1234), 0, lambda _: True) is None
    assert len(receiver.pending) == 1


def test_conflicting_duplicate_and_changed_total_are_rejected():
    receiver = Receiver(TOKEN)
    first = packets()[0]
    assert receiver.feed(first, PEER, 0, lambda _: True)
    assert receiver.feed(first[:-1] + b'X', PEER, 0, lambda _: True) is None
    assert receiver.feed(packets(pcm=PCM + b'XX')[0], PEER, 0, lambda _: True) is None
