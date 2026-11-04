"""Один asyncio loop на сессию; UI читает только неизменяемые снимки."""

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass, replace
from functools import lru_cache
from threading import Lock, Thread
from time import perf_counter
from typing import Literal

from src.config import Settings, load_settings
from src.solver import SolverError, solve
from src.speech import SpeechRecognizer
from src.audio_capture import AudioCapture
from src.conversation import Conversation, ExtractedTask
from src.codex_backend import solve_with_codex
from src.live_speech import LiveSpeech, LiveState, Replica


@lru_cache(maxsize=1)
def get_recognizer() -> SpeechRecognizer:
    # Одна ленивая модель на процесс, доступ сериализован внутри recognizer.
    return SpeechRecognizer(offline=True, model_size=load_settings().asr_model)


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
        self.conversation = Conversation(self._on_extracted)
        self.task_stability = 3.0
        self.audio_buffer_seconds = 5.0
        self.asr_queue_seconds = 48.0
        self.speech_fragment_seconds = 12.0
        self._auto_enabled = False
        self._auto_solution = False
        self._last_auto = ""
        self._auto_timer: asyncio.TimerHandle | None = None
        self.capture = AudioCapture()
        self._live: LiveSpeech | None = None
        self._live_generation = 0
        self._context_broken = False
        self._capture_future: Future[None] | None = None
        self._transcription = Transcription()
        self._speech_future: Future[None] | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_run_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def snapshot(self) -> Solution:
        with self._lock:
            return self._solution

    @property
    def auto_enabled(self) -> bool:
        with self._lock:
            return self._auto_enabled

    def add_replica(self, text: str, speaker: str) -> None:
        self._loop.call_soon_threadsafe(self._add_replica, text, speaker)

    def _add_replica(self, text: str, speaker: str) -> None:
        if self._auto_timer is not None:
            self._auto_timer.cancel()
        self.conversation.add(text, speaker)

    def set_auto(self, enabled: bool) -> None:
        with self._lock:
            if enabled and self._context_broken:
                return
            self._auto_enabled = enabled
        if not enabled:
            self.cancel()
        else:
            self._loop.call_soon_threadsafe(self._on_extracted, self.conversation.snapshot().task)

    def _on_extracted(self, task: ExtractedTask) -> None:
        if self._auto_timer is not None:
            self._auto_timer.cancel()
        fingerprint = " ".join(task.solver_condition.split())
        if self._auto_solution and (fingerprint != " ".join(self.snapshot().problem.split()) or not task.complete):
            # Уточнение или неполное новое условие делает старый ответ неактуальным.
            with self._lock:
                self._generation += 1
                if self._future is not None:
                    self._future.cancel()
                self._solution = replace(self._solution, status="waiting")
                self._last_auto = ""
        if self.auto_enabled and self.conversation.snapshot().status == "ready" and task.complete and fingerprint and fingerprint != self._last_auto:
            self._auto_timer = self._loop.call_later(self.task_stability, self._start_auto, task.solver_condition, fingerprint)

    def _start_auto(self, condition: str, fingerprint: str) -> None:
        state = self.conversation.snapshot()
        if not self.auto_enabled or self._closed or state.status != "ready" or " ".join(state.task.solver_condition.split()) != fingerprint:
            return
        self._last_auto = fingerprint
        self.start(condition, Settings(), use_codex=True, automatic=True)

    def start_capture(self, microphone: str, monitor: str, *, consent: bool) -> bool:
        with self._lock:
            if self._closed or not consent or not microphone or not monitor or microphone == monitor:
                return False
            if self.capture.snapshot().running:
                return False
            generation = self.capture.prepare()
            self._live_generation += 1
            token = self._live_generation
            live = LiveSpeech(get_recognizer(),
                              lambda replica: self._live_replica(replica, token),
                              lambda: self._audio_gap(token),
                              audio_buffer_seconds=self.audio_buffer_seconds,
                              asr_queue_seconds=self.asr_queue_seconds,
                              fragment_seconds=self.speech_fragment_seconds)
            self._live = live
            self._capture_future = asyncio.run_coroutine_threadsafe(
                self._capture_with_asr(microphone, monitor, generation, live, token), self._loop,
            )
            return True

    async def _capture_with_asr(self, microphone: str, monitor: str, generation: int,
                                live: LiveSpeech, token: int) -> None:
        live.start()
        try:
            await self.capture.run(microphone, monitor, generation, live.feed)
            if self.capture.snapshot().error:
                self._audio_gap(token)
        finally:
            await live.close()

    def _live_replica(self, replica: Replica, token: int) -> None:
        if token == self._live_generation and not self._closed and not self.context_broken:
            speaker = f"{replica.speaker} [{replica.start:.1f}–{replica.end:.1f} с]"
            self._add_replica(replica.text, speaker)

    @property
    def context_broken(self) -> bool:
        with self._lock:
            return self._context_broken

    def _audio_gap(self, token: int) -> None:
        if token != self._live_generation or self._closed or self.context_broken:
            return
        with self._lock:
            self._context_broken = True
        self.cancel()
        self.conversation.pause_for_gap()

    def reset_context(self) -> None:
        self.cancel()
        self._loop.call_soon_threadsafe(self._reset_context)

    def _reset_context(self) -> None:
        self.conversation.reset()
        if self._live is not None:
            self._live.reset_context()
        with self._lock:
            self._context_broken = False
            self._last_auto = ""

    def live_snapshot(self) -> LiveState:
        return self._live.snapshot() if self._live is not None else LiveState()

    def stop_capture(self) -> None:
        with self._lock:
            self._live_generation += 1
            self.capture.stop()
            if self._capture_future is not None:
                self._capture_future.cancel()
        # После остановки не запускаем решение по последним накопленным репликам.
        self.cancel()
        self._loop.call_soon_threadsafe(self.conversation.suspend)

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
                error="Не удалось распознать файл. Проверьте формат аудио и наличие выбранной ASR-модели в кеше.",
            )
        with self._lock:
            if not self._closed:
                self._transcription = result

    def cancel(self) -> None:
        with self._lock:
            self._auto_enabled = False
            self._generation += 1
            self._solution = replace(self._solution, status="waiting", error="")
            if self._future is not None:
                self._future.cancel()

    def start(self, problem: str, settings: Settings, *, use_codex: bool = False, automatic: bool = False) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Фоновый обработчик закрыт")
            self._generation += 1
            if self._future is not None:
                self._future.cancel()
            self._solution = Solution(status="solving", problem=problem)
            self._auto_solution = automatic
            self._future = asyncio.run_coroutine_threadsafe(
                self._solve(problem, settings, self._generation, use_codex), self._loop,
            )

    def _update(self, generation: int, **changes: str) -> None:
        with self._lock:
            if generation == self._generation:
                self._solution = replace(self._solution, **changes)

    async def _solve(self, problem: str, settings: Settings, generation: int, use_codex: bool = False) -> None:
        text = ""
        try:
            async with asyncio.timeout(300):
                if use_codex:
                    text = await solve_with_codex(problem)
                    self._update(generation, text=text)
                else:
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
        self.stop_capture()
        self._loop.call_soon_threadsafe(self.conversation.close)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
