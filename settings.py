"""Application settings using pydantic_settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI API settings
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for authentication",
    )

    # Model settings
    transcription_model: str = Field(
        default="whisper-1",
        description="Model to use for speech-to-text transcription",
    )
    chat_model: str = Field(
        default="gpt-4.1",
        description="Model to use for chat completions",
    )
    tts_model: str = Field(
        default="gpt-4o-mini-tts",
        description="Model to use for text-to-speech synthesis",
    )
    tts_voice: Literal[
        "alloy", "echo", "fable", "onyx", "nova", "shimmer", "sage", "coral"
    ] = Field(
        default="sage",
        description="Voice to use for text-to-speech synthesis",
    )
    tts_instructions: str = Field(
        default="",
        description="Instructions for text-to-speech synthesis (e.g. speaking style, emotion)",
    )
    tts_streaming: bool = Field(
        default=True,
        description="Whether to use streaming mode for text-to-speech synthesis",
    )

    # Chat settings
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature for response generation",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens to generate",
        ge=1,
    )

    # Application settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # Paths
    base_dir: Path = Path(__file__).parent
    prompts_dir: Path = Field(
        default=Path(__file__).parent / "prompts",
        description="Directory containing prompt templates",
    )
    assets_dir: Path = Field(
        default=Path(__file__).parent / "assets",
        description="Directory containing application assets",
    )
    sounds_dir: Path = Field(
        default=Path(__file__).parent / "assets" / "sounds",
        description="Directory containing sound effects",
    )

    # Server settings
    host: str = Field(
        default="127.0.0.1",
        description="Host to bind the server to",
    )
    port: int = Field(
        default=7860,
        description="Port to bind the server to",
        ge=1024,
        le=65535,
    )

    @classmethod
    @field_validator("openai_api_key")
    def validate_api_key(cls, v: str) -> str:
        """Validate that the API key is not empty and has a reasonable format."""
        empty_key_error = "OpenAI API key cannot be empty"
        format_error = "OpenAI API key should start with 'sk-'"

        if not v:
            raise ValueError(empty_key_error)
        if not v.startswith("sk-") and not v.startswith("OPENAI_API_KEY"):
            raise ValueError(format_error)
        return v

    @classmethod
    @field_validator("prompts_dir", "assets_dir", "sounds_dir")
    def validate_directory_exists(cls, v: Path) -> Path:
        """Validate that the directory exists or create a warning."""
        from loguru import logger

        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Directory {v} did not exist and was created.")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
        env_nested_delimiter="__",
        protected_namespaces=("model_",),
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance for better performance."""
    return Settings()


# Create a global instance
settings = get_settings()
