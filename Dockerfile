# This tag's unversioned "-cudnn-" suffix bundles cuDNN 9.x, which is what
# the pinned faster-whisper/ctranslate2 versions in pyproject.toml require
# (ctranslate2 4.5+ needs cuDNN 9; earlier releases needed cuDNN 8 and a
# "-cudnnN-" tagged base image instead). If either pin set changes, re-check
# this coupling before assuming a plain version bump is safe -- a mismatch
# fails at first inference, as a libcudnn_ops_infer.so load error, well
# after the pod has already started billing.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install .

# Bake the Silero VAD ONNX model into the image so pod cold-start doesn't
# depend on outbound network access.
ENV SILERO_VAD_MODEL_PATH=/app/models/silero_vad.onnx
RUN mkdir -p /app/models && \
    curl -fL -o "$SILERO_VAD_MODEL_PATH" \
        https://github.com/snakers4/silero-vad/raw/v5.0/files/silero_vad.onnx

# Pre-download the faster-whisper model weights so pod cold-start doesn't
# pay for a Hugging Face download on first request. Must match
# WHISPER_MODEL_SIZE at runtime -- override with --build-arg if you change
# the default. CPU/int8 here only to trigger the download; no GPU needed
# at build time.
ARG WHISPER_MODEL_SIZE=small
RUN python3.11 -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL_SIZE}', device='cpu', compute_type='int8')"

EXPOSE 8000

CMD ["python3.11", "-m", "kenku_stt.main"]
