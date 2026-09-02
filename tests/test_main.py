import kenku_stt.main as main_module


class FakeAuthWebSocket:
    """Only the surface transcribe()'s auth check touches before it either
    closes or falls through into building a TranscriptionSession -- no need
    to boot the real app/lifespan (which would load the actual Whisper/VAD
    models) just to exercise this check.
    """

    def __init__(self, token: str) -> None:
        self.query_params = {"token": token}
        self.closed_with: int | None = None

    async def close(self, code: int | None = None) -> None:
        self.closed_with = code


async def test_rejects_mismatched_token(monkeypatch):
    monkeypatch.setattr(main_module.settings, "auth_token", "correct-token")
    ws = FakeAuthWebSocket("wrong-token")

    await main_module.transcribe(ws)

    assert ws.closed_with == 1008


async def test_rejects_non_ascii_token_without_raising(monkeypatch):
    # hmac.compare_digest(str, str) raises TypeError unless BOTH arguments
    # are pure ASCII. The token comes straight off the query string and is
    # fully attacker-controlled, so a request like ?token=café must still
    # close cleanly with 1008 instead of raising out of the handler.
    monkeypatch.setattr(main_module.settings, "auth_token", "correct-token")
    ws = FakeAuthWebSocket("café")

    await main_module.transcribe(ws)

    assert ws.closed_with == 1008


async def test_rejects_when_configured_token_itself_is_non_ascii(monkeypatch):
    # Same TypeError risk applies if the operator sets a non-ASCII
    # AUTH_TOKEN -- an incoming ASCII token must still be cleanly rejected,
    # not raise.
    monkeypatch.setattr(main_module.settings, "auth_token", "café")
    ws = FakeAuthWebSocket("wrong-token")

    await main_module.transcribe(ws)

    assert ws.closed_with == 1008
