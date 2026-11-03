"""Настройки из окружения и локального .env без вывода секретов."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default="", repr=False)
    openai_model: str = "gpt-5-mini"


def load_settings() -> Settings:
    values = dotenv_values(Path(__file__).with_name(".env"))
    # Переменная окружения имеет приоритет. Повторное чтение позволяет
    # подхватить изменение .env при следующем обновлении интерфейса.
    key = os.environ.get("OPENAI_API_KEY", values.get("OPENAI_API_KEY") or "")
    model = os.environ.get("OPENAI_MODEL", values.get("OPENAI_MODEL") or "gpt-5-mini")
    return Settings(openai_api_key=key.strip(), openai_model=model.strip() or "gpt-5-mini")
