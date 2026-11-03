import time
import unittest
import streamlit as st
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from config import Settings
from worker import SolverWorker, Transcription


class AppTests(unittest.TestCase):
    def setUp(self):
        # AppTest переиспользует тестовый ID сессии между экземплярами.
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

    def test_recognized_text_requires_explicit_transfer(self):
        worker = SolverWorker()
        transcript = Transcription(status="done", filename="test.wav", text="Задача из аудио", seconds=0.8)
        with (
            patch("config.load_settings", return_value=Settings()),
            patch("worker.SolverWorker", return_value=worker),
            patch.object(worker, "speech_snapshot", return_value=transcript),
        ):
            try:
                app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
                self.assertFalse(app.exception)
                app.text_area[0].input("Моя ручная правка").run()
                self.assertEqual(app.text_area[0].value, "Моя ручная правка")
                app.run()
                self.assertEqual(app.text_area[0].value, "Моя ручная правка")
                app.button(key="use_transcript").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.text_area[0].value, "Задача из аудио")
                self.assertEqual(worker.snapshot().status, "waiting")
            finally:
                worker.close()

    def test_solve_edit_and_retry_without_network(self):
        workers = []

        def create_worker():
            worker = SolverWorker()
            workers.append(worker)
            return worker

        async def fake_solve(problem, settings):
            yield "Производная: $2x$."

        with (
            patch("config.load_settings", return_value=Settings("test-key")),
            patch("worker.SolverWorker", side_effect=create_worker),
            patch("worker.solve", side_effect=fake_solve) as solve,
        ):
            try:
                app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
                self.assertFalse(app.exception)
                app.button[0].click().run()
                solve.assert_not_called()
                app.text_area[0].input("Найти производную x²").run()
                app.button[0].click().run()
                deadline = time.monotonic() + 3
                while workers[0].snapshot().status == "solving" and time.monotonic() < deadline:
                    time.sleep(0.01)
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(workers[0].snapshot().status, "done")
                self.assertEqual(app.button[0].label, "Решить заново")
                self.assertIn("Производная", "".join(item.value for item in app.markdown))
                app.text_area[0].input("Найти производную x³").run()
                self.assertEqual(workers[0].snapshot().status, "waiting")
                self.assertEqual(app.text_area[0].value, "Найти производную x³")
                self.assertEqual(solve.call_count, 1)
                app.button[0].click().run()
                deadline = time.monotonic() + 3
                while workers[0].snapshot().status == "solving" and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(workers[0].snapshot().problem, "Найти производную x³")
                self.assertEqual(solve.call_count, 2)
                self.assertEqual(len(workers), 1)
            finally:
                for worker in workers:
                    worker.close()
