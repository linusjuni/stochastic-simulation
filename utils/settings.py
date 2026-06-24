from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    SEED: int = Field(default=67)
    PLOT_STYLE: str = Field(default="ggplot")


settings = Settings()
