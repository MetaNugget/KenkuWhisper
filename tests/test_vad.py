import urllib.error

import numpy as np
import pytest

from kenku_stt.vad import CONTEXT_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES, SileroVAD, SileroVADModel, contains_speech


@pytest.fixture(scope="module")
def vad_model() -> SileroVADModel:
    try:
        return SileroVADModel()
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"Silero VAD model unavailable (no network?): {exc}")


def test_silence_is_not_classified_as_speech(vad_model: SileroVADModel):
    vad = SileroVAD(vad_model, threshold=0.5)
    silence = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    for _ in range(20):
        prob = vad.speech_probability(silence)
        assert 0.0 <= prob <= 1.0
        assert not vad.is_speech(silence)


def test_reset_states_clears_context_between_utterances(vad_model: SileroVADModel):
    vad = SileroVAD(vad_model, threshold=0.5)
    silence = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    for _ in range(5):
        vad.speech_probability(silence)

    vad.reset_states()
    assert np.all(vad._context == 0)
    assert np.all(vad._state == 0)


def test_independent_instances_do_not_share_state(vad_model: SileroVADModel):
    vad_a = SileroVAD(vad_model, threshold=0.5)
    vad_b = SileroVAD(vad_model, threshold=0.5)
    speech_like = np.full(WINDOW_SAMPLES, 0.8, dtype=np.float32)
    silence = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    vad_a.speech_probability(speech_like)
    vad_b.speech_probability(silence)

    # Each instance's context must reflect only its own calls, not the
    # other's -- a shared SileroVADModel session is stateless per the call,
    # so this only holds if SileroVAD itself isn't leaking state sideways.
    assert np.array_equal(vad_a._context, speech_like[-CONTEXT_SAMPLES:])
    assert np.array_equal(vad_b._context, silence[-CONTEXT_SAMPLES:])
    assert not np.array_equal(vad_a._context, vad_b._context)


def test_contains_speech_false_for_pure_silence(vad_model: SileroVADModel):
    vad = SileroVAD(vad_model, threshold=0.5)
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s of silence

    assert not contains_speech(vad, silence, min_speech_ms=250)


def test_contains_speech_resets_state_between_calls(vad_model: SileroVADModel):
    vad = SileroVAD(vad_model, threshold=0.5)
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)

    contains_speech(vad, silence, min_speech_ms=250)
    # a second call on more silence must not be affected by leftover state
    assert not contains_speech(vad, silence, min_speech_ms=250)
