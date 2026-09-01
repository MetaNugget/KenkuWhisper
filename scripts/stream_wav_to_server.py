#!/usr/bin/env python3
"""Streams a 16kHz mono 16-bit PCM WAV file to a running kenku-stt server at
real-time pace and prints incoming transcript messages.

Usage:
    python scripts/stream_wav_to_server.py path/to/audio.wav [ws://localhost:8000/transcribe?sample_rate=16000]

The input WAV must already be 16kHz mono 16-bit PCM (matching exactly what
the KenkuMimic bot sends in production). Convert with ffmpeg if needed:
    ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 audio.wav
"""

import asyncio
import sys
import wave

import websockets

DEFAULT_URL = "ws://localhost:8000/transcribe?sample_rate=16000"
CHUNK_MS = 100


async def stream(wav_path: str, url: str) -> None:
    with wave.open(wav_path, "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise SystemExit(
                f"{wav_path} must be 16kHz mono 16-bit PCM "
                f"(got {wav.getframerate()}Hz, {wav.getnchannels()}ch, "
                f"{wav.getsampwidth() * 8}-bit). Convert with:\n"
                f"  ffmpeg -i {wav_path} -ar 16000 -ac 1 -sample_fmt s16 fixed.wav"
            )

        frames_per_chunk = int(16000 * CHUNK_MS / 1000)
        chunk_bytes = frames_per_chunk * 2  # 16-bit samples

        async with websockets.connect(url) as ws:
            print(f"Connected to {url}")

            async def receiver() -> None:
                async for message in ws:
                    print(f"<- {message}")

            recv_task = asyncio.create_task(receiver())

            data = wav.readframes(frames_per_chunk)
            while data:
                await ws.send(data)
                await asyncio.sleep(CHUNK_MS / 1000)
                if len(data) < chunk_bytes:
                    break
                data = wav.readframes(frames_per_chunk)

            print("Done sending audio, waiting 2s for trailing finals...")
            await asyncio.sleep(2)
            recv_task.cancel()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    wav_path = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_URL
    asyncio.run(stream(wav_path, url))


if __name__ == "__main__":
    main()
