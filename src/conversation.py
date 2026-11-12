"""Ограниченный контекст и debounce извлечения по новым репликам."""

import asyncio
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from time import monotonic

from src.codex_backend import invoke
from src.prompt_loader import load_prompt


@dataclass(frozen=True)
class ExtractedTask:
    condition: str = ""
    expressions: tuple[str, ...] = ()
    complete: bool = False
    ambiguities: tuple[str, ...] = ()

    @property
    def solver_condition(self) -> str:
        if not self.ambiguities:
            return self.condition
        return self.condition + "\n\nНеоднозначности распознавания и возможные трактовки (не подтверждённые данные):\n" + "\n".join(self.ambiguities)


async def extract(context: tuple[str, ...], previous: ExtractedTask, *, image: bytes | None = None) -> ExtractedTask:
    options = {"image": image} if image is not None else {}
    answer = await invoke(
        load_prompt("screen-extractor" if image is not None else "extractor") + "\n" + json.dumps({
            "previous_condition": previous.condition,
            "previous_expressions": previous.expressions,
            "previous_ambiguities": previous.ambiguities,
            "replicas": context,
        }, ensure_ascii=False), structured=True, **options,
    )
    data = json.loads(answer)
    if not isinstance(data, dict) or type(data.get("complete")) is not bool:
        raise ValueError("Некорректный ответ экстрактора")
    if not isinstance(data.get("condition"), str) or len(data["condition"]) > 8000:
        raise ValueError("Некорректное условие")
    for field in ("expressions", "ambiguities"):
        if not isinstance(data.get(field), list) or not all(isinstance(x, str) for x in data[field]):
            raise ValueError("Некорректное поле экстрактора")
    return ExtractedTask(data["condition"].strip(), tuple(data["expressions"]), data["complete"], tuple(data["ambiguities"]))


@dataclass(frozen=True)
class ConversationState:
    replicas: tuple[str, ...] = ()
    task: ExtractedTask = ExtractedTask()
    status: str = "waiting"
    checks: int = 0
    error: str = ""


class Conversation:
    """add/close вызываются в asyncio loop; snapshot безопасен для UI."""

    def __init__(self, on_result: Callable[[ExtractedTask], None], *, pause: float = 3, maximum: float = 15) -> None:
        if not 0 < pause <= maximum:
            raise ValueError("Нужно 0 < pause <= maximum")
        self.pause, self.maximum = pause, maximum
        self._on_result = on_result
        self._lock = Lock()
        self._state = ConversationState()
        self._context: deque[str] = deque()
        self._revision = 0
        self._dirty_since: float | None = None
        self._last = 0.0
        self._changed = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._image: bytes | None = None
        self._image_waiter: asyncio.Future[None] | None = None

    def snapshot(self) -> ConversationState:
        with self._lock:
            return self._state

    def _update(self, **changes) -> None:
        with self._lock:
            self._state = replace(self._state, **changes)

    def add(self, text: str, speaker: str) -> None:
        text = text.strip()
        if not text:
            return
        self._context.append(f"{speaker}: {text[:2000]}")
        while sum(map(len, self._context)) > 8000 or len(self._context) > 40:
            self._context.popleft()
        now = monotonic()
        self._revision += 1
        if self._dirty_since is None:
            self._dirty_since = now
        self._last = now
        self._changed.set()
        self._update(replicas=tuple(self._context), status="pending", error="")
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._run())

    async def analyze_image(self, image: bytes) -> None:
        """Снимок проходит через тот же последовательный экстрактор, что и речь."""
        if self._image_waiter is not None:
            raise ValueError("Разбор снимка уже выполняется")
        waiter = asyncio.get_running_loop().create_future()
        self._image, self._image_waiter = image, waiter
        self._revision += 1
        self._dirty_since = monotonic() - self.maximum
        self._changed.set()
        self._update(status="pending", error="")
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._run())
        try:
            await waiter
        finally:
            if self._image_waiter is waiter:
                # Отмена владельца снимка делает его ответ устаревшим.
                self._image = self._image_waiter = None
                self._revision += 1

    def _finish_image(self, *, error: bool = False) -> None:
        waiter = self._image_waiter
        self._image = self._image_waiter = None
        if waiter is not None and not waiter.done():
            if error:
                waiter.set_exception(ValueError("Разбор снимка не удался. Проверьте Codex CLI и повторите запрос."))
            else:
                waiter.set_result(None)

    async def _run(self) -> None:
        while self._dirty_since is not None:
            due = min(self._last + self.pause, self._dirty_since + self.maximum)
            self._changed.clear()
            remaining = due - monotonic()
            if remaining > 0:
                try:
                    await asyncio.wait_for(self._changed.wait(), remaining)
                    continue
                except TimeoutError:
                    pass
            revision = self._revision
            context = tuple(self._context)
            self._dirty_since = None
            state = self.snapshot()
            self._update(status="extracting", checks=state.checks + 1)
            try:
                if self._image is not None:
                    result = await extract(context, state.task, image=self._image)
                else:
                    result = await extract(context, state.task)
            except asyncio.CancelledError:
                raise
            except Exception:
                if revision == self._revision:
                    self._update(status="error", error="Проверка условия не удалась. Проверьте Codex CLI и добавьте новую реплику для повтора.")
                    self._finish_image(error=True)
                continue
            if revision != self._revision:
                # Новые реплики уже накоплены; не принимаем устаревшее условие.
                continue
            self._update(task=result, status="ready" if result.complete and result.condition else "incomplete")
            self._finish_image()
            self._on_result(result)

    def close(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
        if self._image_waiter is not None:
            self._image_waiter.cancel()
        self._image = self._image_waiter = None

    def reset(self) -> None:
        self.close()
        self._runner = None
        self._revision += 1
        self._dirty_since = None
        self._context.clear()
        with self._lock:
            self._state = ConversationState()

    def suspend(self) -> None:
        self.close()
        self._revision += 1
        self._dirty_since = None
        self._update(status="waiting")

    def pause_for_gap(self) -> None:
        self.close()
        self._revision += 1
        self._dirty_since = None
        task = replace(self.snapshot().task, complete=False,
                       ambiguities=("Часть аудио пропущена. Начните новый контекст и повторите условие.",))
        self._update(task=task, status="incomplete")
