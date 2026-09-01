import json

from kenku_stt.protocol import TranscriptMessage


def test_to_json_round_trip():
    msg = TranscriptMessage(transcript="hello world", final=True)
    decoded = json.loads(msg.to_json())
    assert decoded == {"transcript": "hello world", "final": True}


def test_final_false():
    msg = TranscriptMessage(transcript="", final=False)
    decoded = json.loads(msg.to_json())
    assert decoded == {"transcript": "", "final": False}
