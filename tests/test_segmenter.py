import numpy as np

from kenku_stt.segmenter import BufferConfig, BurstBuffer, SAMPLE_RATE


def make_samples(duration_ms: float, value: float = 0.1) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    return np.full(n, value, dtype=np.float32)


def test_push_buffers_without_finalizing_below_max_length():
    buf = BurstBuffer(BufferConfig(max_segment_ms=10_000))
    assert buf.push(make_samples(500)) is None
    assert buf.push(make_samples(500)) is None


def test_finalize_returns_concatenated_buffer_and_resets():
    buf = BurstBuffer()
    buf.push(make_samples(300))
    buf.push(make_samples(300))

    segment = buf.finalize()

    assert segment is not None
    assert len(segment) == int(SAMPLE_RATE * 0.6)
    # buffer is empty after finalize
    assert buf.finalize() is None


def test_max_segment_forces_cutoff_and_keeps_buffering_afterward():
    buf = BurstBuffer(BufferConfig(max_segment_ms=1000))

    assert buf.push(make_samples(600)) is None
    forced = buf.push(make_samples(600))

    assert forced is not None
    assert len(forced) == int(SAMPLE_RATE * 1.2)

    # a fresh burst accumulates independently after the forced cutoff
    assert buf.push(make_samples(200)) is None
    tail = buf.finalize()
    assert tail is not None
    assert len(tail) == int(SAMPLE_RATE * 0.2)


def test_finalize_with_nothing_buffered_returns_none():
    buf = BurstBuffer()
    assert buf.finalize() is None
