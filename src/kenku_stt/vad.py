import asyncio
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
CONTEXT_SAMPLES = 64  # required trailing-context window size for Silero VAD @16kHz
WINDOW_SAMPLES = 512  # Silero VAD's required fixed input window size @16kHz
WINDOW_MS = 1000 * WINDOW_SAMPLES / SAMPLE_RATE  # 32ms

_MODEL_URL = "https://github.com/snakers4/silero-vad/raw/v5.0/files/silero_vad.onnx"
_DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "SILERO_VAD_MODEL_PATH",
        Path.home() / ".cache" / "kenku-stt" / "silero_vad.onnx",
    )
)


def _ensure_model(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


class SileroVADModel:
    """Loads the Silero VAD ONNX weights once and shares the session across
    every connection. The session call is stateless from onnxruntime's
    perspective -- the recurrent hidden state and trailing audio context
    are passed in/out explicitly as tensors, so per-connection state lives
    in SileroVAD below, not here.
    """

    def __init__(self, model_path: Path | None = None, max_workers: int = 2) -> None:
        resolved = _ensure_model(model_path or _DEFAULT_MODEL_PATH)
        self._session = ort.InferenceSession(
            str(resolved), providers=["CPUExecutionProvider"]
        )
        # Separate from WhisperEngine's GPU pool: the worst case for
        # contains_speech() below is a full MAX_SEGMENT_MS burst of pure
        # noise that never trips the speech threshold (~937 window
        # inferences), and that shouldn't be able to starve transcription
        # or any other connected speaker's receive loop/inactivity timer.
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vad")

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)

    def run(
        self, audio_input: np.ndarray, state: np.ndarray, sr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        prob, new_state = self._session.run(
            None, {"input": audio_input, "state": state, "sr": sr}
        )
        return prob, new_state


class SileroVAD:
    """Per-connection VAD state (recurrent hidden state + audio context)
    bound to a shared SileroVADModel. Each WebSocket connection must own
    its own instance -- state must not be shared across independent
    speaker streams.
    """

    def __init__(self, model: SileroVADModel, threshold: float = 0.5) -> None:
        self._model = model
        self._threshold = threshold
        self.reset_states()

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def speech_probability(self, window: np.ndarray) -> float:
        """window must be exactly 512 float32 samples @16kHz."""
        audio_input = np.concatenate([self._context, window])[np.newaxis, :].astype(
            np.float32
        )
        sr = np.array(SAMPLE_RATE, dtype=np.int64)

        prob, new_state = self._model.run(audio_input, self._state, sr)

        self._state = new_state
        self._context = window[-CONTEXT_SAMPLES:]
        return float(prob.squeeze())

    def is_speech(self, window: np.ndarray) -> bool:
        return self.speech_probability(window) >= self._threshold


def contains_speech(vad: SileroVAD, audio: np.ndarray, min_speech_ms: float) -> bool:
    """One-shot content check over a fully-buffered burst: does it contain
    at least min_speech_ms of VAD-classified speech? Used to discard noise
    blips before they reach the GPU -- utterance *boundaries* are decided
    upstream by message-arrival gaps (see session.py), not by this.
    """
    vad.reset_states()
    speech_ms = 0.0
    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        window = audio[start : start + WINDOW_SAMPLES]
        if vad.is_speech(window):
            speech_ms += WINDOW_MS
        if speech_ms >= min_speech_ms:
            return True
    return False


async def contains_speech_async(vad: SileroVAD, audio: np.ndarray, min_speech_ms: float) -> bool:
    """Event-loop-safe wrapper around contains_speech() -- runs it on
    vad's model's own thread pool instead of blocking the asyncio loop
    thread. See SileroVADModel's pool for why that pool is separate from
    WhisperEngine's.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(vad._model.pool, contains_speech, vad, audio, min_speech_ms)
