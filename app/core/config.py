"""Environment-driven application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_VERSION = "1.0.0"
RANDOM_SEED = 42


class Settings(BaseSettings):
    """All runtime configuration. Every field can be set via an environment variable
    of the same name (case-insensitive) or via a local ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    log_level: str = "INFO"

    # --- LLM provider (optional). "demo" needs no key and makes no network calls. ---
    llm_provider: Literal["demo", "anthropic", "openai"] = "demo"
    llm_api_key: SecretStr | None = None
    llm_model: str = ""
    llm_base_url: str = ""
    llm_timeout_seconds: float = Field(20.0, gt=0, le=120)

    # --- Retrieval ---
    embedding_backend: Literal["auto", "hybrid", "sentence-transformers"] = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_top_k: int = Field(3, ge=1, le=10)
    retrieval_min_score: float = Field(0.35, ge=0, le=1)

    # --- Classification ---
    classifier_min_confidence: float = Field(0.30, ge=0, le=1)

    # --- Limits ---
    max_request_bytes: int = Field(32_768, ge=1_024)

    # --- Paths ---
    kb_dir: Path = ROOT_DIR / "data" / "knowledge_base"
    training_data_path: Path = ROOT_DIR / "data" / "sample_tickets.csv"
    model_path: Path = ROOT_DIR / "models" / "classifier.joblib"

    @property
    def llm_configured(self) -> bool:
        return (
            self.llm_provider != "demo"
            and self.llm_api_key is not None
            and bool(self.llm_api_key.get_secret_value())
            and bool(self.llm_model)
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
