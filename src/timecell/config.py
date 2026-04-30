"""Runtime config from env vars / .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str | None = None
    timecell_model: str = "gemini-2.5-flash"
    timecell_http_timeout: float = 10.0
    timecell_cache_ttl: int = 60


def get_settings() -> Settings:
    return Settings()
