from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from .config import settings
from .segmenter import BufferConfig
from .session import TranscriptionSession
from .vad import SileroVADModel
from .whisper_engine import WhisperEngine

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["engine"] = WhisperEngine(settings)
    _state["vad_model"] = SileroVADModel()
    yield
    _state["engine"].shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/transcribe")
async def transcribe(websocket: WebSocket) -> None:
    session = TranscriptionSession(
        websocket=websocket,
        engine=_state["engine"],
        vad_model=_state["vad_model"],
        vad_threshold=settings.vad_threshold,
        min_speech_ms=settings.min_speech_ms,
        inactivity_timeout_ms=settings.inactivity_timeout_ms,
        buffer_config=BufferConfig(max_segment_ms=settings.max_segment_ms),
    )
    await session.run()


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    run()
