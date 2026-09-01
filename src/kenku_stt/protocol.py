import json
from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptMessage:
    transcript: str
    final: bool

    def to_json(self) -> str:
        return json.dumps({"transcript": self.transcript, "final": self.final})
