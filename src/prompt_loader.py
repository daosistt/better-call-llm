"""Чтение промптов перед запросом, без кеширования."""

from pathlib import Path
from typing import Literal

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
SOLVER_PROMPTS = {"analytics": "Аналитика", "physics": "Физика", "ml": "ML и Python"}


def _read(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Промпт {path.name} пуст.")
    return text


def load_prompt(name: Literal["extractor", "screen-extractor"]) -> str:
    return _read(PROMPTS_DIR / f"{name}.md")


def load_solver_prompt(name: str = "analytics") -> str:
    if name not in SOLVER_PROMPTS:
        raise ValueError("Неизвестный промпт решателя.")
    return _read(PROMPTS_DIR / "solver" / f"{name}.md")
