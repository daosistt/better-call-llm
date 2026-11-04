import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from audio_capture import AudioCapture, level


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    def test_meter_silence_and_signal(self):
        self.assertEqual(level(b"\0\0" * 1600), 0)
        self.assertGreater(level(np.full(1600, 16000, dtype="<i2").tobytes()), 0.8)

    async def test_two_streams_drained_and_terminated_on_stop(self):
        capture = AudioCapture()
        processes = []
        ready = asyncio.Event()

        async def spawn(*args, **kwargs):
            process = MagicMock()
            process.returncode = None
            process.wait = AsyncMock(return_value=0)
            count = 0

            async def readexactly(size):
                nonlocal count
                count += 1
                if count == 1:
                    return b"\0\0" * (size // 2)
                ready.set()
                await asyncio.Event().wait()

            process.stdout.readexactly = readexactly
            processes.append(process)
            return process

        with patch("audio_capture.asyncio.create_subprocess_exec", side_effect=spawn) as spawn_mock:
            generation = capture.prepare()
            task = asyncio.create_task(capture.run("mic", "headset.monitor", generation))
            await asyncio.wait_for(ready.wait(), 1)
            # Оба reader task успевают обработать первый блок в этом цикле.
            await asyncio.sleep(0)
            state = capture.snapshot()
            self.assertEqual(state.microphone_blocks, 1)
            self.assertEqual(state.system_blocks, 1)
            self.assertEqual(spawn_mock.call_count, 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            capture.stop()
            self.assertFalse(capture.snapshot().running)
            for process in processes:
                process.terminate.assert_called_once()
                process.wait.assert_awaited()

    async def test_process_failure_is_reported(self):
        capture = AudioCapture()
        with patch("audio_capture.asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            await capture.run("mic", "monitor", capture.prepare())
        self.assertFalse(capture.snapshot().running)
        self.assertTrue(capture.snapshot().error)
