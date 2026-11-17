"""AegisRAG configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    CHUNK_SIZE: int = 500
    TOP_K: int = 3
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    QA_MODEL: str = "deepset/roberta-base-squad2"


settings = Settings()
