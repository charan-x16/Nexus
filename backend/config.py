from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENROUTER_API_KEY: SecretStr | None = None
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        min_length=1,
    )
    OPENROUTER_MODEL: str = Field(default="anthropic/claude-sonnet-4", min_length=1)
    OPENROUTER_APP_NAME: str = Field(default="Nexus", min_length=1)
    OPENROUTER_SITE_URL: str = Field(default="http://localhost:8501", min_length=1)
    OPENAI_API_KEY: SecretStr | None = None
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", min_length=1)
    TAVILY_API_KEY: SecretStr | None = None
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/nexus",
        min_length=1,
    )
    LANGSMITH_API_KEY: SecretStr | None = None
    LANGSMITH_PROJECT: str = Field(default="nexus-phase1", min_length=1)
    ENVIRONMENT: str = Field(default="development", min_length=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
