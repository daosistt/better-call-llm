import asyncio
import time
import unittest
from threading import Event
from unittest.mock import patch

from src.config import Settings
from src.solver import SolverError
from src.worker import SolverWorker
from src.speech import SpeechSegment
from unittest.mock import MagicMock
from src.conversation import ExtractedTask


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.worker = SolverWorker()
        self.addCleanup(self.worker.close)

    def wait_done(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = self.worker.snapshot()
            if state.status != "solving":
                return state
            time.sleep(0.01)
        self.fail("Обработчик не завершил запрос")

    def test_replacement_ignores_late_old_answer(self):
        started = Event()
        old_finished = Event()

        async def fake_solve(problem, settings):
            if problem == "old":
                started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    # Имитируем поздний ответ уже отменённого запроса.
                    yield "Устаревший ответ"
                    old_finished.set()
            else:
                yield "Новый ответ"

        with patch("src.worker.solve", side_effect=fake_solve):
            self.worker.start("old", Settings("key"))
            self.assertTrue(started.wait(2))
            self.worker.start("new", Settings("key"))
            state = self.wait_done()
            self.assertTrue(old_finished.wait(2))
            self.assertEqual(state.status, "done")
            self.assertEqual(state.problem, "new")
            self.assertEqual(self.worker.snapshot().text, "Новый ответ")

    def test_partial_answer_survives_error(self):
        async def fake_solve(problem, settings):
            yield "Часть решения"
            raise SolverError("Сеть недоступна")

        with patch("src.worker.solve", side_effect=fake_solve):
            self.worker.start("Задача", Settings("key"))
            state = self.wait_done()
            self.assertEqual(state.status, "error")
            self.assertEqual(state.text, "Часть решения")
            self.assertEqual(state.error, "Сеть недоступна")

    def test_auto_solves_once_and_manual_edit_disables_it(self):
        self.worker.conversation.pause = 0.01
        self.worker.conversation.maximum = 0.02
        self.worker.task_stability = 0.01
        task = ExtractedTask("Найти производную x²", complete=True)
        from unittest.mock import AsyncMock
        with (
            patch("src.conversation.extract", new_callable=AsyncMock, return_value=task),
            patch("src.worker.solve_with_codex", new_callable=AsyncMock, return_value="2x") as solve,
        ):
            self.worker.set_auto(True)
            self.worker.add_replica("Найти производную x²", "Ты")
            deadline = time.monotonic() + 2
            while self.worker.snapshot().status != "done" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(self.worker.snapshot().text, "2x")
            self.worker.add_replica("да, x²", "Ты")
            time.sleep(0.08)
            solve.assert_awaited_once()
            self.worker.cancel()  # Тот же обработчик вызывается ручной правкой.
            self.assertFalse(self.worker.auto_enabled)
            self.worker.add_replica("Найти производную x³", "Ты")
            time.sleep(0.08)
            solve.assert_awaited_once()

    def test_auto_passes_ambiguities_and_reacts_to_changed_interpretation(self):
        from unittest.mock import AsyncMock

        self.worker.conversation.pause = 0.01
        self.worker.conversation.maximum = 0.02
        self.worker.task_stability = 0.01
        task = ExtractedTask("Найти производную икс плюс один в квадрате", complete=True,
                             ambiguities=("Вариант 1: (x+1)^2; вариант 2: x+1^2",))
        with (
            patch("src.conversation.extract", new_callable=AsyncMock, return_value=task) as extract,
            patch("src.worker.solve_with_codex", new_callable=AsyncMock, return_value="Два ответа") as solve,
        ):
            self.worker.set_auto(True)
            self.worker.add_replica(task.condition, "Ты")
            deadline = time.monotonic() + 2
            while solve.await_count < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            solve.assert_awaited_once()
            self.assertIn(task.ambiguities[0], solve.call_args.args[0])
            self.worker.add_replica("Обсуждаем эти варианты", "Ты")
            time.sleep(0.08)
            solve.assert_awaited_once()
            extract.return_value = ExtractedTask(task.condition, complete=True,
                                                ambiguities=("Уточнение: имелось в виду (x+1)^2",))
            self.worker.add_replica("Квадрат всей суммы", "Ты")
            deadline = time.monotonic() + 2
            while solve.await_count < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(solve.await_count, 2)
            self.assertIn("Уточнение", solve.call_args.args[0])

    def test_recognition_does_not_block_solver_or_queue_more_files(self):
        started = Event()
        release = Event()

        def recognize(audio):
            started.set()
            release.wait(3)
            return (SpeechSegment(0, 1, "Распознанная задача"),)

        async def fake_solve(problem, settings):
            yield "Ответ"

        recognizer = MagicMock()
        recognizer.transcribe_bytes.side_effect = recognize
        with patch("src.worker.get_recognizer", return_value=recognizer), patch("src.worker.solve", side_effect=fake_solve):
            try:
                self.assertTrue(self.worker.start_transcription(b"audio", "test.wav"))
                self.assertTrue(started.wait(2))
                self.assertFalse(self.worker.start_transcription(b"another", "other.wav"))
                self.worker.start("Ручная задача", Settings("key"))
                self.assertEqual(self.wait_done().text, "Ответ")
            finally:
                release.set()
            deadline = time.monotonic() + 3
            while self.worker.speech_snapshot().status == "recognizing" and time.monotonic() < deadline:
                time.sleep(0.01)
            state = self.worker.speech_snapshot()
            self.assertEqual(state.status, "done")
            self.assertEqual(state.text, "Распознанная задача")
            self.assertEqual(self.worker.snapshot().problem, "Ручная задача")

    def test_bad_audio_is_reported_and_can_be_retried(self):
        recognizer = MagicMock()
        recognizer.transcribe_bytes.side_effect = ValueError("private decoder details")
        with patch("src.worker.get_recognizer", return_value=recognizer):
            self.assertFalse(self.worker.start_transcription(b"", "empty.wav"))
            self.assertEqual(self.worker.speech_snapshot().status, "error")
            self.assertTrue(self.worker.start_transcription(b"broken", "bad.wav"))
            deadline = time.monotonic() + 3
            while self.worker.speech_snapshot().status == "recognizing" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(self.worker.speech_snapshot().status, "error")
            self.assertNotIn("private", self.worker.speech_snapshot().error)

    def test_audio_gap_requires_explicit_context_reset(self):
        self.worker.set_auto(True)
        self.worker._loop.call_soon_threadsafe(self.worker._audio_gap, self.worker._live_generation)
        deadline = time.monotonic() + 2
        while not self.worker.context_broken and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.worker.context_broken)
        self.assertFalse(self.worker.auto_enabled)
        self.worker.set_auto(True)
        self.assertFalse(self.worker.auto_enabled)
        self.worker.reset_context()
        deadline = time.monotonic() + 2
        while self.worker.context_broken and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.worker.context_broken)
        self.assertFalse(self.worker.auto_enabled)

    def test_extractor_does_not_cancel_manual_codex_solution(self):
        from unittest.mock import AsyncMock
        with patch("src.worker.solve_with_codex", new_callable=AsyncMock, return_value="Ручной ответ"):
            self.worker.start("Моя задача", Settings(), use_codex=True)
            self.assertEqual(self.wait_done().text, "Ручной ответ")
            self.worker._loop.call_soon_threadsafe(
                self.worker._on_extracted, ExtractedTask("Другая задача", complete=True),
            )
            time.sleep(0.03)
            self.assertEqual(self.worker.snapshot().status, "done")
            self.assertEqual(self.worker.snapshot().problem, "Моя задача")

    def test_shutdown_cancels_request_and_stops_thread(self):
        started = Event()
        stopped = Event()

        async def fake_solve(problem, settings):
            started.set()
            try:
                await asyncio.sleep(60)
                yield "Ответ"
            finally:
                stopped.set()

        with patch("src.worker.solve", side_effect=fake_solve):
            self.worker.start("Задача", Settings("key"))
            self.assertTrue(started.wait(2))
            self.worker.close()
            self.assertTrue(stopped.wait(2))
            self.assertFalse(self.worker._thread.is_alive())
