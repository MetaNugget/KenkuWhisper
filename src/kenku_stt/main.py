import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from .config import settings
from .segmenter import BufferConfig
from .session import TranscriptionSession
from .vad import SileroVADModel
from .whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.auth_token:
        logger.warning(
            "AUTH_TOKEN is unset -- /transcribe is open to anyone who can reach this "
            "host. Fine for local development; do not run a RunPod deployment this way."
        )
    _state["engine"] = WhisperEngine(settings)
    _state["vad_model"] = SileroVADModel()
    yield
    _state["engine"].shutdown()
    _state["vad_model"].shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/transcribe")
async def transcribe(websocket: WebSocket) -> None:
    if settings.auth_token:
        token = websocket.query_params.get("token", "")
        if not hmac.compare_digest(token, settings.auth_token):
            await websocket.close(code=1008)
            return

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
