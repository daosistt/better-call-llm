import asyncio
from base64 import b64decode
from threading import Event
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.screen_capture import ScreenCaptureError, capture_screen
from src.worker import SolverWorker
from src.conversation import ExtractedTask

# Синтетический PNG 1×1, без снимка настоящего экрана.
PNG = b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aB1sAAAAASUVORK5CYII=")


class ScreenCaptureTests(unittest.IsolatedAsyncioTestCase):
    def process(self, data=PNG, returncode=0):
        process = MagicMock()
        process.returncode = returncode
        process.stdout.read = AsyncMock(side_effect=[data, b""])
        process.wait = AsyncMock()
        return process

    async def test_image_returned_in_memory(self):
        process = self.process()
        with patch("src.screen_capture.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process) as spawn:
            self.assertEqual(await capture_screen(), PNG)
        command = spawn.call_args.args
        self.assertEqual(command[0], "grim")
        self.assertEqual(command[-1], "-")
        self.assertEqual(spawn.call_args.kwargs["stdout"], asyncio.subprocess.PIPE)

    async def test_missing_grim_has_actionable_error(self):
        with patch("src.screen_capture.asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ScreenCaptureError, "Не найден grim"):
                await capture_screen()

    async def test_failure_and_invalid_output(self):
        for process in (self.process(returncode=1), self.process(data=b""), self.process(data=b"not an image")):
            with self.subTest(returncode=process.returncode):
                with patch("src.screen_capture.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process):
                    with self.assertRaises(ScreenCaptureError):
                        await capture_screen()

    async def test_oversize_output_kills_process(self):
        process = self.process(data=b"x" * 17, returncode=None)
        with (
            patch("src.screen_capture.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process),
            patch("src.screen_capture.MAX_IMAGE_BYTES", 16),
        ):
            with self.assertRaisesRegex(ScreenCaptureError, "16 МБ"):
                await capture_screen()
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()

    async def test_timeout_and_cancellation_kill_process(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                started = asyncio.Event()
                process = self.process(returncode=None)

                async def read(size):
                    started.set()
                    await asyncio.Event().wait()

                process.stdout.read.side_effect = read
                with (
                    patch("src.screen_capture.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process),
                    patch("src.screen_capture.CAPTURE_TIMEOUT", 1 if cancel else 0.01),
                ):
                    task = asyncio.create_task(capture_screen())
                    await asyncio.wait_for(started.wait(), 1)
                    if cancel:
                        task.cancel()
                    with self.assertRaises(asyncio.CancelledError if cancel else ScreenCaptureError):
                        await task
                process.kill.assert_called_once()
                process.wait.assert_awaited_once()


class ScreenWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def wait_done(self, worker):
        async with asyncio.timeout(2):
            while worker.screen_snapshot().status in {"capturing", "analyzing"}:
                await asyncio.sleep(0.01)

    async def test_screen_task_auto_solves_and_audio_can_correct_it(self):
        worker = SolverWorker()
        self.addCleanup(worker.close)
        worker.task_stability = 0.01
        worker.conversation.pause = 0.01
        worker.conversation.maximum = 0.03
        worker.set_auto(True)
        first = ExtractedTask("Производная x²", complete=True)
        corrected = ExtractedTask("Производная x³", complete=True)
        with (
            patch("src.worker.capture_screen", new_callable=AsyncMock, return_value=PNG),
            patch("src.conversation.extract", new_callable=AsyncMock, side_effect=[first, corrected]) as extract,
            patch("src.worker.solve_with_codex", new_callable=AsyncMock, return_value="Решение") as solve,
        ):
            self.assertFalse(worker.start_screen_analysis())
            worker.start_screenshot()
            await self.wait_done(worker)
            self.assertTrue(worker.start_screen_analysis())
            await self.wait_done(worker)
            async with asyncio.timeout(2):
                while solve.await_count < 1:
                    await asyncio.sleep(0.01)
            self.assertTrue(worker.screen_snapshot().analyzed)
            self.assertEqual(solve.call_args.args, (first.condition,))
            self.assertEqual(extract.call_args.kwargs, {"image": PNG})
            worker.add_replica("Поправка: степень три", "Ты")
            async with asyncio.timeout(2):
                while solve.await_count < 2:
                    await asyncio.sleep(0.01)
            self.assertEqual(solve.call_args.args, (corrected.condition,))
            self.assertEqual(extract.call_args.args[1], first)
            self.assertEqual(extract.call_args.kwargs, {})

    async def test_analysis_blocks_recapture_and_can_retry_after_error(self):
        worker = SolverWorker()
        self.addCleanup(worker.close)
        started, release = Event(), Event()

        async def extract(*args, **kwargs):
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            raise ValueError("private-detail")

        with (
            patch("src.worker.capture_screen", new_callable=AsyncMock, return_value=PNG),
            patch("src.conversation.extract", side_effect=extract),
        ):
            worker.start_screenshot()
            await self.wait_done(worker)
            worker.start_screen_analysis()
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            self.assertFalse(worker.start_screen_analysis())
            self.assertFalse(worker.start_screenshot())
            worker.clear_screenshot()
            self.assertEqual(worker.screen_snapshot().status, "analyzing")
            release.set()
            await self.wait_done(worker)
        self.assertEqual(worker.screen_snapshot().image, PNG)
        self.assertNotIn("private-detail", worker.screen_snapshot().error)
        with patch("src.conversation.extract", new_callable=AsyncMock, return_value=ExtractedTask("Новая задача")):
            self.assertTrue(worker.start_screen_analysis())
            await self.wait_done(worker)
        self.assertTrue(worker.screen_snapshot().analyzed)

    async def test_one_capture_at_a_time_and_clear(self):
        worker = SolverWorker()
        self.addCleanup(worker.close)
        started, release = Event(), Event()

        async def capture():
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            return PNG

        with patch("src.worker.capture_screen", side_effect=capture) as capture_mock:
            self.assertTrue(worker.start_screenshot())
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            self.assertFalse(worker.start_screenshot())
            worker.clear_screenshot()
            self.assertEqual(worker.screen_snapshot().status, "capturing")
            release.set()
            await self.wait_done(worker)
        capture_mock.assert_called_once()
        self.assertEqual(worker.screen_snapshot().image, PNG)
        self.assertTrue(worker.screen_snapshot().captured_at)
        worker.clear_screenshot()
        self.assertEqual(worker.screen_snapshot().image, b"")

    async def test_failure_can_be_retried(self):
        worker = SolverWorker()
        self.addCleanup(worker.close)
        with patch("src.worker.capture_screen", new_callable=AsyncMock, side_effect=[ScreenCaptureError("Ошибка захвата"), PNG]):
            self.assertTrue(worker.start_screenshot())
            await self.wait_done(worker)
            self.assertEqual(worker.screen_snapshot().error, "Ошибка захвата")
            self.assertTrue(worker.start_screenshot())
            await self.wait_done(worker)
            self.assertEqual(worker.screen_snapshot().status, "done")

    async def test_shutdown_cancels_capture(self):
        worker = SolverWorker()
        self.addCleanup(worker.close)
        started, stopped = Event(), Event()

        async def capture():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with patch("src.worker.capture_screen", side_effect=capture):
            worker.start_screenshot()
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            worker.close()
            self.assertTrue(stopped.is_set())
        self.assertFalse(worker.start_screenshot())
