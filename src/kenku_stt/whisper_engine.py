import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from faster_whisper import WhisperModel

from .config import Settings


class WhisperEngine:
    """Shared GPU model + bounded thread pool for offloading transcribe()
    calls off the asyncio event loop. One instance is shared across every
    WebSocket connection; the model itself and CTranslate2's own worker
    threads handle concurrent decode calls.
    """

    def __init__(self, settings: Settings) -> None:
        self._language = settings.whisper_language
        self._initial_prompt = settings.whisper_initial_prompt or None
        self._model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            num_workers=settings.whisper_num_workers,
        )
        self._pool = ThreadPoolExecutor(
            max_workers=settings.whisper_num_workers, thread_name_prefix="whisper-gpu"
        )

    async def transcribe(self, audio: np.ndarray) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,
            condition_on_previous_text=False,
            word_timestamps=False,
            initial_prompt=self._initial_prompt,
            # session.py's contains_speech() has already gated this audio on
            # Silero VAD. A second, independent VAD pass here is wasted work,
            # and faster-whisper's own min_speech_duration_ms default can
            # swallow exactly the short utterances ("Yes.", "I attack.") that
            # matter most in a TTRPG transcript.
            vad_filter=False,
        )
        return "".join(segment.text for segment in segments).strip()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
