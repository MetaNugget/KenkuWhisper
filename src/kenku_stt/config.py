from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    whisper_model_size: str = "small"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: str = "en"
    whisper_num_workers: int = 2
    whisper_initial_prompt: str = ""

    vad_threshold: float = 0.5
    inactivity_timeout_ms: int = 400
    min_speech_ms: int = 250
    max_segment_ms: int = 30000

    port: int = 8000

    # Shared secret required on the /transcribe query string. The server is
    # reachable on the public internet via RunPod's public port mapping (the
    # bot runs on a home Raspberry Pi outside RunPod's private network), so
    # without this an unauthenticated client gets free GPU inference and can
    # inject arbitrary text into a live session's transcript. Empty disables
    # the check, for local development only -- see main.py's startup warning.
    auth_token: str = ""


settings = Settings()
