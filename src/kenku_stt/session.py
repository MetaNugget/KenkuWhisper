import asyncio
import logging

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .audio import pcm16le_to_float32
from .protocol import TranscriptMessage
from .segmenter import BufferConfig, BurstBuffer
from .vad import SileroVAD, SileroVADModel, contains_speech_async
from .whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)


class TranscriptionSession:
    """One instance per WebSocket connection. All state here (VAD buffer
    state, audio buffer) is local to this connection -- nothing is shared
    with any other concurrently-connected speaker.
    """

    def __init__(
        self,
        websocket: WebSocket,
        engine: WhisperEngine,
        vad_model: SileroVADModel,
        vad_threshold: float,
        min_speech_ms: int,
        inactivity_timeout_ms: int,
        buffer_config: BufferConfig,
    ) -> None:
        self._ws = websocket
        self._engine = engine
        self._vad = SileroVAD(vad_model, threshold=vad_threshold)
        self._min_speech_ms = min_speech_ms
        self._inactivity_timeout_s = inactivity_timeout_ms / 1000
        self._buffer = BurstBuffer(buffer_config)

    async def run(self) -> None:
        await self._ws.accept()
        try:
            while True:
                try:
                    data = await asyncio.wait_for(
                        self._ws.receive_bytes(), timeout=self._inactivity_timeout_s
                    )
                except asyncio.TimeoutError:
                    await self._finalize()
                    continue

                segment = self._buffer.push(pcm16le_to_float32(data))
                if segment is not None:
                    await self._process(segment)
        except WebSocketDisconnect:
            pass
        finally:
            await self._finalize(best_effort=True)

    async def _finalize(self, best_effort: bool = False) -> None:
        segment = self._buffer.finalize()
        if segment is None:
            return
        try:
            await self._process(segment)
        except Exception:
            if not best_effort:
                raise
            # socket may already be unwritable if the client closed abruptly

    async def _process(self, audio: np.ndarray) -> None:
        if not await contains_speech_async(self._vad, audio, self._min_speech_ms):
            return
        transcript = await self._engine.transcribe(audio)
        if not transcript:
            return
        message = TranscriptMessage(transcript, final=True).to_json()
        try:
            await self._ws.send_text(message)
        except RuntimeError:
            # send_text on a socket whose peer already disconnected raises
            # RuntimeError, not WebSocketDisconnect (that's only raised on
            # receive) -- so it isn't caught by run()'s except clause and
            # would otherwise escape as an unhandled traceback, silently
            # dropping the transcript. Sending after the peer closed is a
            # protocol violation, not a transient failure, so don't retry.
            logger.warning("Dropped transcript, peer already disconnected: %r", transcript)
