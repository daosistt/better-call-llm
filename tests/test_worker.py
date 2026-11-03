import asyncio
import time
import unittest
from threading import Event
from unittest.mock import patch

from config import Settings
from solver import SolverError
from worker import SolverWorker
from speech import SpeechSegment
from unittest.mock import MagicMock


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

        with patch("worker.solve", side_effect=fake_solve):
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

        with patch("worker.solve", side_effect=fake_solve):
            self.worker.start("Задача", Settings("key"))
            state = self.wait_done()
            self.assertEqual(state.status, "error")
            self.assertEqual(state.text, "Часть решения")
            self.assertEqual(state.error, "Сеть недоступна")

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
        with patch("worker.get_recognizer", return_value=recognizer), patch("worker.solve", side_effect=fake_solve):
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
        with patch("worker.get_recognizer", return_value=recognizer):
            self.assertFalse(self.worker.start_transcription(b"", "empty.wav"))
            self.assertEqual(self.worker.speech_snapshot().status, "error")
            self.assertTrue(self.worker.start_transcription(b"broken", "bad.wav"))
            deadline = time.monotonic() + 3
            while self.worker.speech_snapshot().status == "recognizing" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(self.worker.speech_snapshot().status, "error")
            self.assertNotIn("private", self.worker.speech_snapshot().error)

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

        with patch("worker.solve", side_effect=fake_solve):
            self.worker.start("Задача", Settings("key"))
            self.assertTrue(started.wait(2))
            self.worker.close()
            self.assertTrue(stopped.wait(2))
            self.assertFalse(self.worker._thread.is_alive())
