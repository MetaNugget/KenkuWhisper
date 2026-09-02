import asyncio
import logging

import numpy as np
import pytest
from fastapi import WebSocketDisconnect

import kenku_stt.session as session_module
from kenku_stt.segmenter import BufferConfig
from kenku_stt.session import TranscriptionSession

_DISCONNECT = object()


class FakeWebSocket:
    """Minimal stand-in for fastapi.WebSocket: a queue-backed receive_bytes()
    (raising WebSocketDisconnect once disconnect() is called, matching a real
    client closing the socket) and a send_text() that records what went out,
    or raises if configured to. This is the entire surface run() touches.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self.sent: list[str] = []
        self.send_text_error: Exception | None = None

    async def accept(self) -> None:
        pass

    async def receive_bytes(self) -> bytes:
        item = await self._queue.get()
        if item is _DISCONNECT:
            raise WebSocketDisconnect()
        return item

    async def send_text(self, text: str) -> None:
        if self.send_text_error is not None:
            raise self.send_text_error
        self.sent.append(text)

    def push_audio(self, duration_ms: float) -> None:
        n = int(16000 * duration_ms / 1000)
        self._queue.put_nowait(np.zeros(n, dtype="<i2").tobytes())

    def disconnect(self) -> None:
        self._queue.put_nowait(_DISCONNECT)


class StubEngine:
    """Stands in for WhisperEngine -- returns a canned transcript instead of
    running real GPU inference, so these tests exercise session.py's own
    finalize/timeout/error-handling logic, not whisper's accuracy.
    """

    def __init__(self, transcript: str = "hello world") -> None:
        self.transcript = transcript
        self.call_count = 0

    async def transcribe(self, audio: np.ndarray) -> str:
        self.call_count += 1
        return self.transcript


def make_session(
    ws: FakeWebSocket,
    *,
    engine: StubEngine | None = None,
    inactivity_timeout_ms: int = 30,
    min_speech_ms: int = 250,
    max_segment_ms: int = 30_000,
) -> TranscriptionSession:
    return TranscriptionSession(
        websocket=ws,
        engine=engine or StubEngine(),
        # Never touched directly: contains_speech_async is monkeypatched
        # below instead of running real Silero inference over dummy zero
        # samples (which the real VAD would just call silence anyway).
        # SileroVAD's constructor only stores this and zeroes its own
        # arrays, so a dummy object is safe to pass through.
        vad_model=object(),
        vad_threshold=0.5,
        min_speech_ms=min_speech_ms,
        inactivity_timeout_ms=inactivity_timeout_ms,
        buffer_config=BufferConfig(max_segment_ms=max_segment_ms),
    )


@pytest.fixture(autouse=True)
def _speech_always_present(monkeypatch):
    """Default every test to "yes, this buffer contains speech" so
    finalize/timeout behavior can be tested without real VAD inference;
    individual tests override this to exercise the opposite path.
    """

    async def _true(vad, audio, min_speech_ms):
        return True

    monkeypatch.setattr(session_module, "contains_speech_async", _true)


async def test_inactivity_gap_finalizes_and_sends_exactly_one_message():
    ws = FakeWebSocket()
    engine = StubEngine("hello world")
    session = make_session(ws, engine=engine, inactivity_timeout_ms=100)
    task = asyncio.create_task(session.run())

    ws.push_audio(300)
    await asyncio.sleep(0.5)  # generous margin over the 100ms inactivity timeout

    assert engine.call_count == 1
    assert ws.sent == ['{"transcript": "hello world", "final": true}']

    ws.disconnect()
    await task

    # buffer was already empty by the time the connection closed -- the
    # best-effort finalize in run()'s finally block must not send again
    assert ws.sent == ['{"transcript": "hello world", "final": true}']


async def test_continuous_audio_does_not_finalize_early():
    ws = FakeWebSocket()
    engine = StubEngine()
    session = make_session(ws, engine=engine, inactivity_timeout_ms=300)
    task = asyncio.create_task(session.run())

    # Cumulative sleep here (~100ms) must stay comfortably under the 300ms
    # inactivity timeout, with enough headroom to absorb scheduling jitter
    # under load -- a 5x10ms version of this flaked under load (effectively
    # racing the timeout, not "well under" it).
    for _ in range(5):
        ws.push_audio(100)
        await asyncio.sleep(0.02)

    assert engine.call_count == 0
    assert ws.sent == []

    # now go quiet and let the inactivity timeout finalize the accumulated burst
    await asyncio.sleep(0.6)
    assert engine.call_count == 1
    assert len(ws.sent) == 1

    ws.disconnect()
    await task


async def test_max_segment_forces_cutoff_mid_burst():
    ws = FakeWebSocket()
    engine = StubEngine()
    # inactivity_timeout_ms kept large relative to the test's real-time
    # sleeps so only the max_segment_ms cutoff -- not the timeout -- can
    # explain a finalize here.
    session = make_session(ws, engine=engine, inactivity_timeout_ms=5000, max_segment_ms=200)
    task = asyncio.create_task(session.run())

    ws.push_audio(150)
    await asyncio.sleep(0.02)
    assert engine.call_count == 0  # still under the 200ms cap

    ws.push_audio(100)  # 250ms buffered total > 200ms max_segment_ms
    await asyncio.sleep(0.02)
    assert engine.call_count == 1
    assert len(ws.sent) == 1

    ws.disconnect()
    await task


async def test_burst_failing_vad_check_sends_nothing(monkeypatch):
    async def _no_speech(vad, audio, min_speech_ms):
        return False

    monkeypatch.setattr(session_module, "contains_speech_async", _no_speech)

    ws = FakeWebSocket()
    engine = StubEngine()
    session = make_session(ws, engine=engine, inactivity_timeout_ms=100)
    task = asyncio.create_task(session.run())

    ws.push_audio(300)
    await asyncio.sleep(0.5)

    assert engine.call_count == 0
    assert ws.sent == []

    ws.disconnect()
    await task


async def test_send_failure_does_not_escape_run(caplog):
    ws = FakeWebSocket()
    ws.send_text_error = RuntimeError('Cannot call "send" once a close message has been sent.')
    engine = StubEngine("dropped transcript")
    session = make_session(ws, engine=engine, inactivity_timeout_ms=100)

    with caplog.at_level(logging.WARNING, logger="kenku_stt.session"):
        task = asyncio.create_task(session.run())
        ws.push_audio(300)
        await asyncio.sleep(0.5)

        assert engine.call_count == 1  # transcription itself still ran
        assert ws.sent == []  # but the send failed

        ws.disconnect()
        await task  # must not raise -- this is the regression this test guards

    assert "Dropped transcript" in caplog.text
