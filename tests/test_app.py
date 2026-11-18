import time
import unittest
import streamlit as st
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.config import Settings
from src.worker import SolverWorker
from src.audio_capture import AudioSource
from src.conversation import ConversationState, ExtractedTask


class AppTests(unittest.TestCase):
    def test_compact_ui_keeps_transcript_and_current_condition_retry(self):
        worker = SolverWorker()
        self.addCleanup(worker.close)
        task = ExtractedTask("Найти производную x²", complete=True)

        async def solve(problem, settings):
            yield "Производная: $2x$."

        with (
            patch("src.ui.SolverWorker", return_value=worker),
            patch("src.ui.load_settings", return_value=Settings("test-key")),
            patch.object(worker.conversation, "snapshot", return_value=ConversationState(task=task, status="ready")),
            patch("src.worker.solve", side_effect=solve) as solve_mock,
        ):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            self.assertFalse(app.exception)
            labels = [item.label for item in app.expander]
            self.assertIn("Расшифровка разговора", labels)
            self.assertNotIn("Дополнительно", labels)
            self.assertNotIn("Ручной ввод и повторное решение", labels)
            self.assertFalse(app.text_area)
            self.assertFalse(app.get("file_uploader"))
            self.assertFalse([b for b in app.button if b.key == "edit_extracted"])
            app.selectbox(key="solver_backend").select("OpenAI API").run()
            app.button(key="solve").click().run()
            deadline = time.monotonic() + 2
            while worker.snapshot().status == "solving" and time.monotonic() < deadline:
                time.sleep(0.01)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(worker.snapshot().text, "Производная: $2x$.")
            self.assertEqual(app.button(key="solve").label, "Решить заново")
            self.assertEqual(solve_mock.call_args.args[0], task.condition)
            app.button(key="solve").click().run()
            deadline = time.monotonic() + 2
            while worker.snapshot().status == "solving" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(solve_mock.call_count, 2)

    def test_prompt_choice_is_shared_without_stale_view_overwriting_it(self):
        worker = SolverWorker()
        self.addCleanup(worker.close)
        with patch("src.ui.SolverWorker", return_value=worker):
            path = Path(__file__).resolve().parents[1] / "app.py"
            laptop = AppTest.from_file(path).run()
            phone = AppTest.from_file(path).run()
            laptop.selectbox(key="solver_prompt").select("ml").run()
            self.assertFalse(laptop.exception)
            self.assertEqual(worker.solver_prompt, "ml")
            phone.run()
            self.assertFalse(phone.exception)
            self.assertEqual(phone.selectbox(key="solver_prompt").value, "ml")
            self.assertEqual(worker.solver_prompt, "ml")
            phone.selectbox(key="solver_prompt").select("physics").run()
            laptop.run()
            self.assertFalse(laptop.exception)
            self.assertEqual(laptop.selectbox(key="solver_prompt").value, "physics")

    def test_recovery_button_restores_previous_auto_mode(self):
        worker = SolverWorker()
        self.addCleanup(worker.close)
        worker.capture.prepare()
        worker.set_auto(True)
        worker._loop.call_soon_threadsafe(worker._audio_gap, worker._live_generation)
        deadline = time.monotonic() + 2
        while not worker.context_broken and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(worker.context_broken)
        with patch("src.ui.SolverWorker", return_value=worker):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
            self.assertTrue(app.button(key="toggle_auto").disabled)
            app.button(key="reset_audio_context").click().run()
            deadline = time.monotonic() + 2
            while worker.context_broken and time.monotonic() < deadline:
                time.sleep(0.01)
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(worker.context_broken)
            self.assertEqual(app.button(key="toggle_auto").label, "Авторешение: включено")

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
