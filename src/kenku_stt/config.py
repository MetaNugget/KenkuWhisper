from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    whisper_model_size: str = "small"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: str = "en"
    whisper_num_workers: int = 2

    vad_threshold: float = 0.5
    inactivity_timeout_ms: int = 400
    min_speech_ms: int = 250
    max_segment_ms: int = 30000

    port: int = 8000


settings = Settings()
