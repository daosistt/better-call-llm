import time
import unittest
import streamlit as st
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.config import Settings
from src.worker import SolverWorker, Transcription
from src.audio_capture import AudioSource
from src.conversation import ConversationState, ExtractedTask


class AppTests(unittest.TestCase):
    def test_phone_button_shares_laptop_screenshot(self):
        from unittest.mock import AsyncMock
        from test_screen_capture import PNG

        worker = SolverWorker()
        self.addCleanup(worker.close)
        with (
            patch("src.ui.SolverWorker", return_value=worker),
            patch("src.worker.capture_screen", new_callable=AsyncMock, return_value=PNG) as capture,
            patch("src.conversation.extract", new_callable=AsyncMock,
                  return_value=ExtractedTask("Условие с экрана: найти производную x²", complete=True)) as extract,
        ):
            path = Path(__file__).resolve().parents[1] / "app.py"
            phone = AppTest.from_file(path).run()
            capture.assert_not_called()
            phone.button(key="capture_screen").click().run()
            deadline = time.monotonic() + 2
            while worker.screen_snapshot().status == "capturing" and time.monotonic() < deadline:
                time.sleep(0.01)
            phone.run()
            laptop = AppTest.from_file(path).run()
            self.assertFalse(phone.exception)
            self.assertFalse(laptop.exception)
            self.assertEqual(worker.screen_snapshot().image, PNG)
            self.assertTrue(phone.get("image"))
            self.assertTrue(laptop.get("image"))
            capture.assert_awaited_once()
            extract.assert_not_called()
            phone.button(key="analyze_screen").click().run()
            deadline = time.monotonic() + 2
            while worker.screen_snapshot().status == "analyzing" and time.monotonic() < deadline:
                time.sleep(0.01)
            phone.run()
            laptop.run()
            extract.assert_awaited_once()
            self.assertEqual(extract.call_args.kwargs, {"image": PNG})
            for view in (phone, laptop):
                self.assertFalse(view.exception)
                self.assertIn("Условие с экрана", " ".join(item.value for item in view.markdown))
            phone.button(key="clear_screen").click().run()
            self.assertFalse(phone.exception)
            self.assertEqual(worker.screen_snapshot().image, b"")
            laptop.run()
            self.assertFalse(laptop.get("image"))

    def test_second_view_shares_result_and_does_not_stop_capture(self):
        worker = SolverWorker()
        worker.capture.prepare()  # Синтетическое состояние, без микрофона.

        async def solve(problem, settings):
            yield "Общий ответ для ноутбука и телефона"

        with patch("src.ui.SolverWorker", return_value=worker) as factory, patch("src.worker.solve", side_effect=solve):
            try:
                worker.start("Общая задача", Settings("test-key"))
                deadline = time.monotonic() + 2
                while worker.snapshot().status == "solving" and time.monotonic() < deadline:
                    time.sleep(0.01)
                path = Path(__file__).resolve().parents[1] / "app.py"
                laptop = AppTest.from_file(path).run()
                phone = AppTest.from_file(path).run()
                for view in (laptop, phone):
                    self.assertFalse(view.exception)
                    content = " ".join(item.value for item in view.markdown)
                    self.assertIn("Общая задача", content)
                    self.assertIn("Общий ответ", content)
                factory.assert_called_once()
                self.assertTrue(worker.capture.snapshot().running)
                phone.checkbox(key="capture_consent").uncheck().run()
                self.assertFalse(worker.capture.snapshot().running)
                laptop.run()
                self.assertFalse(laptop.exception)
                self.assertTrue(laptop.button(key="start_capture"))
            finally:
                worker.close()

    def test_edit_extracted_condition_preserves_manual_control(self):
        worker = SolverWorker()
        worker.set_auto(True)
        state = ConversationState(task=ExtractedTask("Найти производную x²", complete=True), status="ready")
        with patch("src.ui.SolverWorker", return_value=worker), patch.object(worker.conversation, "snapshot", return_value=state):
            try:
                app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
                self.assertFalse(app.exception)
                app.button(key="edit_extracted").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.text_area[0].value, "Найти производную x²")
                self.assertFalse(worker.auto_enabled)
                self.assertTrue(app.session_state["editor_open"])
                app.text_area[0].input("Найти производную x³").run()
                app.run()
                self.assertEqual(app.text_area[0].value, "Найти производную x³")
            finally:
                worker.close()

    def setUp(self):
        # AppTest переиспользует тестовый ID сессии между экземплярами.
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        st.cache_data.clear()
        devices = patch("src.ui.list_sources", return_value=(
            (AudioSource("mic", "Микрофон ноутбука", False), AudioSource("headset.monitor", "Наушники", True)),
            "mic", "headset.monitor",
        ))
        devices.start()
        self.addCleanup(devices.stop)
        self.addCleanup(st.cache_data.clear)

    def test_capture_requires_consent_and_stops(self):
        from unittest.mock import AsyncMock

        worker = SolverWorker()
        with patch("src.ui.SolverWorker", return_value=worker), patch.object(worker.capture, "run", new_callable=AsyncMock):
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
            patch("src.ui.load_settings", return_value=Settings()),
            patch("src.ui.SolverWorker", return_value=worker),
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
            patch("src.ui.load_settings", return_value=Settings("test-key")),
            patch("src.ui.SolverWorker", side_effect=create_worker),
            patch("src.worker.solve", side_effect=fake_solve) as solve,
        ):
            try:
                app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
                app.selectbox(key="solver_backend").select("OpenAI API").run()
                self.assertFalse(app.exception)
                app.button(key="solve").click().run()
                solve.assert_not_called()
                app.text_area[0].input("Найти производную x²").run()
                app.button(key="solve").click().run()
                deadline = time.monotonic() + 3
                while workers[0].snapshot().status == "solving" and time.monotonic() < deadline:
                    time.sleep(0.01)
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(workers[0].snapshot().status, "done")
                self.assertEqual(app.button(key="solve").label, "Решить заново")
                self.assertIn("Производная", "".join(item.value for item in app.markdown))
                app.text_area[0].input("Найти производную x³").run()
                self.assertEqual(workers[0].snapshot().status, "waiting")
                self.assertEqual(app.text_area[0].value, "Найти производную x³")
                self.assertEqual(solve.call_count, 1)
                app.button(key="solve").click().run()
                deadline = time.monotonic() + 3
                while workers[0].snapshot().status == "solving" and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(workers[0].snapshot().problem, "Найти производную x³")
                self.assertEqual(solve.call_count, 2)
                self.assertEqual(len(workers), 1)
            finally:
                for worker in workers:
                    worker.close()
