# kenku-stt

GPU-side real-time speech-to-text server for [KenkuMimic](../KenkuMimic). Runs
inside a RunPod pod, gets started/stopped by the bot's existing GPU
orchestration client (`TRANSCRIPTION_PROVIDER=selfhosted` in the bot's
`.env`).

## Protocol

This server implements a protocol already frozen by the bot's client
(`KenkuMimic/lib/transcription/selfhosted.js`) — do not change it here.

- `ws://<host>:8000/transcribe?sample_rate=16000`
- One WebSocket connection per speaker. Client sends raw 16kHz mono PCM16LE
  binary frames, no envelope.
- Server sends JSON text messages: `{"transcript": "...", "final": true}` once
  per detected utterance. There are no partial/interim messages in v1.
- No auth/TLS — RunPod's private networking is the trust boundary.

### Utterance boundaries: message gaps, not in-audio silence

The KenkuMimic client's Discord capture pipeline only forwards PCM while a
speaker is actively talking — during a pause it sends nothing at all (no
silence-padded audio), so this server can't detect "silence" by listening
for quiet samples the way a continuous-stream transcriber would. Instead,
`src/kenku_stt/session.py` treats a gap in **WebSocket message arrival**
(`INACTIVITY_TIMEOUT_MS`, default 400ms) as the utterance boundary, buffering
raw audio in `src/kenku_stt/segmenter.py`'s `BurstBuffer` until either that
timeout fires or `MAX_SEGMENT_MS` forces a cutoff on an unusually long
uninterrupted burst. This timeout has to stay comfortably under two timers
on the client side — its Discord capture pipeline closes an utterance's
opus stream after 1000ms of no packets, and `speakerPool.js` closes the
WebSocket entirely after 1500ms of no `send()` calls — so the finalized
transcript can be sent back over the connection while it's still open;
trying to send after the client has closed the socket is a hard WebSocket
protocol violation, not something retryable.

Silero VAD is still used, but only as a one-shot content filter
(`contains_speech()` in `src/kenku_stt/vad.py`) run once per finalized
burst to discard noise blips (mic clicks, breath sounds) before they reach
the GPU — it does not drive real-time boundary detection.

## Local development

Requires an NVIDIA GPU + drivers + the NVIDIA Container Toolkit for a
production-like run; a plain CPU run also works for iterating on the
non-model logic (slow, but fine for testing segmentation/VAD/protocol code).

### With Docker (recommended, matches production)

```bash
docker build -t kenku-stt .
docker run --rm --gpus all -p 8000:8000 kenku-stt
```

### Without Docker

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # edit WHISPER_DEVICE=cpu here if you have no local GPU
python -m kenku_stt.main
```

The Silero VAD ONNX model and the faster-whisper model weights are both
downloaded on first run if not already cached (baked into the image at
`docker build` time; downloaded to `~/.cache`-equivalent paths otherwise).

## Testing

Run the unit tests (segmenter state machine, VAD wrapper, protocol schema):

```bash
pip install -e ".[dev]"
pytest
```

To manually exercise the whole pipeline against a running server, stream a
WAV file at it and watch the `final: true` messages come back:

```bash
python scripts/stream_wav_to_server.py path/to/audio.wav
```

The WAV must be 16kHz mono 16-bit PCM (matching what the bot actually sends).
Convert an arbitrary audio file with ffmpeg first if needed:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 audio.wav
```

Run it twice in parallel against two different WAVs to sanity-check that
concurrent connections don't cross-contaminate transcripts.

## Deploying as a RunPod Template

1. Build and push the image to a registry RunPod can pull from:
   ```bash
   docker build -t <your-registry>/kenku-stt:latest .
   docker push <your-registry>/kenku-stt:latest
   ```
2. In the RunPod dashboard: **Templates -> New Template**
   - **Container Image**: `<your-registry>/kenku-stt:latest`
   - **Container Disk**: a few GB is enough (model weights are baked into
     the image, not downloaded at runtime)
   - **Expose HTTP/TCP Ports**: `8000` — the bot's orchestration client
     (`KenkuMimic/lib/transcription/selfhosted.js`) looks for a pod port
     with `private === 8000` and maps to whatever public port RunPod
     assigns; nothing in this server needs to know about that mapping.
   - **Environment Variables**: only needed if you want non-default values
     from `.env.example` (e.g. a different `WHISPER_MODEL_SIZE`) — every
     setting has a working default.
3. Note the resulting **Template ID** and set it as `GPU_PROVIDER_TEMPLATE_ID`
   in the bot's `.env`.
4. Pick a GPU type in the bot's `.env` as `GPU_PROVIDER_GPU_TYPE_ID` (RunPod's
   type identifier string, e.g. `"NVIDIA GeForce RTX 4090"` — see RunPod's
   `GET /v2/catalog/gpus` for valid values). Any GPU with enough VRAM for the
   configured `WHISPER_MODEL_SIZE` at `WHISPER_COMPUTE_TYPE=float16` works;
   `small` is comfortable on a T4-class GPU and up.
5. The bot polls the pod's health via port 8000 for up to 5 minutes on
   startup (`HEALTH_POLL_TIMEOUT_MS` in `selfhosted.js`) — this server's
   `/health` endpoint responds as soon as the process is up, which happens
   quickly since model weights are pre-baked into the image rather than
   downloaded at cold start.

## Tuning

`INACTIVITY_TIMEOUT_MS` (default 400ms) is a reasoned default, not measured
against real Discord session audio — if utterances feel like they're
splitting mid-sentence too often (e.g. from network jitter between packets),
raise it, but keep real margin under the client's 1000ms/1500ms teardown
timers described above or finals will stop being deliverable. `MAX_SEGMENT_MS`
(default 30s) forces a `final: true` on an unusually long, uninterrupted
burst even without a message gap, since the client only ever consumes
finals. See `.env.example` for the full set of tunables.
