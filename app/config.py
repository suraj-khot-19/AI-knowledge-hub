"""Application settings loaded from environment variables or the local .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Knowledge Hub"
    data_dir: str = "./data"
    embedding_model: str =ransformers/all-MiniLM-L6-v2"
    chunk_size: int =  "sentence-t1000
    chunk_overlap: int = 150
    retrieval_k: int = 4
    max_upload_mb: int = 25
    # Optional Hugging Face text-generation model. Leave empty for grounded extractive answers.
    hf_generation_model: str = ""
    hf_token: str = ""


@lru_cache
def get_settings() -> Settings:
    """Reuse one settings instance so every API component shares the same config."""
    return Settings()
