import time
import unittest
import streamlit as st
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from config import Settings
from worker import SolverWorker, Transcription
from audio_capture import AudioSource


class AppTests(unittest.TestCase):
    def setUp(self):
        # AppTest переиспользует тестовый ID сессии между экземплярами.
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        st.cache_data.clear()
        devices = patch("audio_capture.list_sources", return_value=(
            (AudioSource("mic", "Микрофон ноутбука", False), AudioSource("headset.monitor", "Наушники", True)),
            "mic", "headset.monitor",
        ))
        devices.start()
        self.addCleanup(devices.stop)
        self.addCleanup(st.cache_data.clear)

    def test_capture_requires_consent_and_stops(self):
        from unittest.mock import AsyncMock

        worker = SolverWorker()
        with patch("worker.SolverWorker", return_value=worker), patch.object(worker.capture, "run", new_callable=AsyncMock):
            try:
                app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
                self.assertTrue(app.button(key="start_capture").disabled)
                app.checkbox(key="capture_consent").check().run()
                app.button(key="start_capture").click().run()
                self.assertTrue(worker.capture.snapshot().running)
                app.run()
                self.assertTrue(app.selectbox(key="capture_mic").disabled)
                app.button(key="stop_capture").click().run()
                self.assertFalse(worker.capture.snapshot().running)
            finally:
                worker.close()

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
