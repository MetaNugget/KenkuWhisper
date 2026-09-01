from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class BufferConfig:
    max_segment_ms: int = 30000


class BurstBuffer:
    """Accumulates raw audio for one WebSocket connection between explicit
    finalize() calls.

    Utterance boundaries are decided externally, by a message-arrival
    inactivity timeout in session.py -- the KenkuMimic client's Discord
    capture pipeline only forwards audio while a user is actively speaking
    and sends nothing at all during a pause, so a gap shows up as an
    absence of incoming WebSocket messages, not as silence encoded in the
    audio itself. This class only tracks buffered duration for the
    max-length safety cutoff (an unusually long, uninterrupted burst still
    gets cut and emitted rather than growing unbounded).
    """

    def __init__(self, config: BufferConfig | None = None) -> None:
        self._config = config or BufferConfig()
        self._chunks: list[np.ndarray] = []
        self._buffered_ms = 0.0

    def push(self, samples: np.ndarray) -> np.ndarray | None:
        self._chunks.append(samples)
        self._buffered_ms += 1000 * len(samples) / SAMPLE_RATE
        if self._buffered_ms >= self._config.max_segment_ms:
            return self.finalize()
        return None

    def finalize(self) -> np.ndarray | None:
        if not self._chunks:
            return None
        audio = np.concatenate(self._chunks)
        self._chunks = []
        self._buffered_ms = 0.0
        return audio
