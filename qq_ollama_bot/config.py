from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    onebot_access_token: SecretStr = Field(min_length=16)
    allowed_group_ids: str = Field(min_length=1)

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:9b"
    model_provider: Literal["ollama", "deepseek"] = "ollama"
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = Field(default="deepseek-flash", min_length=1)
    deepseek_max_tokens: int = Field(default=800, ge=1, le=393216)
    database_path: Path = Path("data/chat_history.db")
    log_level: str = "INFO"

    bot_host: str = "127.0.0.1"
    bot_port: int = Field(default=8080, ge=1, le=65535)

    request_timeout_seconds: float = Field(default=120.0, gt=0)
    image_model_timeout_seconds: float = Field(default=600.0, gt=0)
    busy_wait_seconds: float = Field(default=5.0, ge=0)
    user_cooldown_seconds: float = Field(default=15.0, ge=0)
    history_turns: int = Field(default=8, ge=1, le=50)
    history_character_budget: int = Field(default=12_000, ge=1_000)
    retention_days: int = Field(default=30, ge=1)
    profile_message_limit: int = Field(default=50, ge=1, le=200)
    profile_character_budget: int = Field(default=6_000, ge=500, le=24_000)
    profile_target_limit: int = Field(default=5, ge=1, le=20)
    forward_max_depth: int = Field(default=3, ge=1, le=10)
    forward_message_limit: int = Field(default=100, ge=1, le=500)
    forward_character_budget: int = Field(default=6_000, ge=500, le=24_000)
    forward_timeout_seconds: float = Field(default=30.0, gt=0)
    forward_model_timeout_seconds: float = Field(default=600.0, gt=0)
    forward_queue_size: int = Field(default=20, ge=1, le=100)
    reply_chunk_size: int = Field(default=1_500, ge=100, le=4_000)
    image_max_count: int = Field(default=4, ge=1, le=10)
    image_max_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)
    image_total_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if normalized not in {"http://127.0.0.1:11434", "http://localhost:11434"}:
            raise ValueError("OLLAMA_BASE_URL must use the local Ollama endpoint")
        return normalized

    @field_validator("deepseek_base_url")
    @classmethod
    def validate_deepseek_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        parts = urlsplit(normalized)
        if (parts.scheme != "https" or not parts.hostname or parts.username
                or parts.password or parts.query or parts.fragment):
            raise ValueError("DEEPSEEK_BASE_URL must be an HTTPS base URL without credentials, query or fragment")
        return normalized

    @field_validator("bot_host")
    @classmethod
    def validate_bot_host(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost"}:
            raise ValueError("BOT_HOST must stay on loopback")
        return value

    @model_validator(mode="after")
    def validate_group_ids(self) -> "Settings":
        if self.model_provider == "deepseek" and (
            self.deepseek_api_key is None or not self.deepseek_api_key.get_secret_value().strip()
        ):
            raise ValueError("DEEPSEEK_API_KEY is required for the DeepSeek provider")
        if not self.allowed_group_id_set:
            raise ValueError("ALLOWED_GROUP_IDS must contain at least one QQ group number")
        if self.image_total_max_bytes < self.image_max_bytes:
            raise ValueError("image_total_max_bytes cannot be smaller than image_max_bytes")
        return self

    @property
    def allowed_group_id_set(self) -> frozenset[int]:
        try:
            return frozenset(
                int(item.strip())
                for item in self.allowed_group_ids.split(",")
                if item.strip()
            )
        except ValueError as exc:
            raise ValueError("ALLOWED_GROUP_IDS must be comma-separated integers") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
