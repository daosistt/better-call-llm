"""Чтение промптов перед запросом, без кеширования."""

from pathlib import Path
from typing import Literal

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: Literal["solver", "extractor"]) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Промпт {name}.md пуст.")
    return text
