"""Один asyncio loop на сессию; UI читает только неизменяемые снимки."""

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass, replace
from functools import lru_cache
from threading import Lock, Thread
from time import perf_counter
from typing import Literal

from config import Settings
from solver import SolverError, solve
from speech import SpeechRecognizer


@lru_cache(maxsize=1)
def get_recognizer() -> SpeechRecognizer:
    # Одна ленивая модель на процесс, доступ сериализован внутри recognizer.
    return SpeechRecognizer(offline=True)


@dataclass(frozen=True)
class Transcription:
    status: Literal["waiting", "recognizing", "done", "error"] = "waiting"
    filename: str = ""
    text: str = ""
    seconds: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class Solution:
    status: Literal["waiting", "solving", "done", "error"] = "waiting"
    problem: str = ""
    text: str = ""
    error: str = ""


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


class SolverWorker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._solution = Solution()
        self._generation = 0
        self._future: Future[None] | None = None
        self._closed = False
        self._transcription = Transcription()
        self._speech_future: Future[None] | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_run_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def snapshot(self) -> Solution:
        with self._lock:
            return self._solution

    def speech_snapshot(self) -> Transcription:
        with self._lock:
            return self._transcription

    def start_transcription(self, audio: bytes, filename: str) -> bool:
        with self._lock:
            if self._closed or self._transcription.status == "recognizing":
                return False
            if not audio or len(audio) > 20 * 1024 * 1024:
                self._transcription = Transcription(
                    status="error", error="Выберите непустой аудиофайл размером до 20 МБ.",
                )
                return False
            self._transcription = Transcription(status="recognizing", filename=filename)
            self._speech_future = asyncio.run_coroutine_threadsafe(
                self._recognize(audio, filename), self._loop,
            )
            return True

    async def _recognize(self, audio: bytes, filename: str) -> None:
        started = perf_counter()
        try:
            segments = await asyncio.to_thread(get_recognizer().transcribe_bytes, audio)
            result = Transcription(
                status="done", filename=filename,
                text=" ".join(segment.text for segment in segments),
                seconds=perf_counter() - started,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            result = Transcription(
                status="error", filename=filename,
                error="Не удалось распознать файл. Проверьте формат аудио и наличие модели base в локальном кеше.",
            )
        with self._lock:
            if not self._closed:
                self._transcription = result

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1
            self._solution = replace(self._solution, status="waiting", error="")
            if self._future is not None:
                self._future.cancel()

    def start(self, problem: str, settings: Settings) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Фоновый обработчик закрыт")
            self._generation += 1
            if self._future is not None:
                self._future.cancel()
            self._solution = Solution(status="solving", problem=problem)
            self._future = asyncio.run_coroutine_threadsafe(
                self._solve(problem, settings, self._generation), self._loop,
            )

    def _update(self, generation: int, **changes: str) -> None:
        with self._lock:
            if generation == self._generation:
                self._solution = replace(self._solution, **changes)

    async def _solve(self, problem: str, settings: Settings, generation: int) -> None:
        text = ""
        try:
            async with asyncio.timeout(300):
                async for chunk in solve(problem, settings):
                    text += chunk
                    self._update(generation, text=text)
            self._update(generation, status="done")
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._update(generation, status="error", error="Превышено время решения. Попробуйте ещё раз.")
        except SolverError as error:
            self._update(generation, status="error", error=str(error))
        except Exception:
            # Не показывать исходное исключение: оно может содержать данные запроса.
            self._update(generation, status="error", error="Не удалось получить решение. Попробуйте ещё раз.")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._speech_future is not None:
                self._speech_future.cancel()
        self.cancel()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
