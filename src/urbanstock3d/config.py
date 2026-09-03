"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with safe, portable defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="URBANSTOCK_",
        extra="ignore",
    )

    output_dir: Path = Path("outputs")
    keep_temporary: bool = False
    http_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    http_read_timeout_seconds: float = Field(default=30.0, gt=0)
