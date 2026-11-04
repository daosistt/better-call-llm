"""Настройки из окружения и локального .env без вывода секретов."""

import os
import math
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default="", repr=False)
    openai_model: str = "gpt-5-mini"
    extractor_pause: float = 3
    extractor_maximum: float = 15
    task_stability: float = 3
    audio_buffer_seconds: float = 5
    asr_queue_seconds: float = 48
    speech_fragment_seconds: float = 12


def load_settings() -> Settings:
    values = dotenv_values(Path(__file__).with_name(".env"))
    # Переменная окружения имеет приоритет. Повторное чтение позволяет
    # подхватить изменение .env при следующем обновлении интерфейса.
    key = os.environ.get("OPENAI_API_KEY", values.get("OPENAI_API_KEY") or "")
    model = os.environ.get("OPENAI_MODEL", values.get("OPENAI_MODEL") or "gpt-5-mini")
    def seconds(name: str, default: float) -> float:
        try:
            value = float(os.environ.get(name, values.get(name) or default))
            if not math.isfinite(value) or not 0 < value <= 300:
                raise ValueError
            return value
        except (TypeError, ValueError):
            raise ValueError(f"{name}: укажите число секунд больше 0 и не больше 300.") from None

    pause = seconds("EXTRACTOR_PAUSE_SECONDS", 3)
    maximum = seconds("EXTRACTOR_MAX_SECONDS", 15)
    if maximum < pause:
        raise ValueError("EXTRACTOR_MAX_SECONDS должен быть не меньше EXTRACTOR_PAUSE_SECONDS.")
    fragment = seconds("SPEECH_FRAGMENT_SECONDS", 12)
    queue = seconds("ASR_QUEUE_SECONDS", 48)
    if fragment < 1 or queue < fragment:
        raise ValueError("SPEECH_FRAGMENT_SECONDS должен быть не меньше 1, ASR_QUEUE_SECONDS — не меньше длины фрагмента.")
    return Settings(
        openai_api_key=key.strip(), openai_model=model.strip() or "gpt-5-mini",
        extractor_pause=pause, extractor_maximum=maximum,
        task_stability=seconds("TASK_STABILITY_SECONDS", 3),
        audio_buffer_seconds=seconds("AUDIO_BUFFER_SECONDS", 5),
        asr_queue_seconds=queue, speech_fragment_seconds=fragment,
    )
