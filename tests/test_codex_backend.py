import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.codex_backend import CodexError, invoke
from test_screen_capture import PNG


class CodexImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_attachment_is_temporary_and_schema_is_available(self):
        attached = []
        process = MagicMock(returncode=0)
        process.stdin.drain = AsyncMock()
        process.stdout.read = AsyncMock(side_effect=[json.dumps({
            "type": "item.completed", "item": {"type": "agent_message", "text": "Результат"},
        }).encode() + b"\n", b""])
        process.wait = AsyncMock()

        async def spawn(*command, **options):
            path = Path(command[command.index("--image") + 1])
            attached.append(path)
            self.assertEqual(path.read_bytes(), PNG)
            self.assertEqual(list(path.parent.iterdir()), [path])
            self.assertEqual(Path(options["cwd"]), path.parent)
            self.assertTrue(Path(command[command.index("--output-schema") + 1]).is_file())
            self.assertEqual(command[-2:], ("--", "-"))
            return process

        with TemporaryDirectory(dir="/tmp") as root:
            def temporary(**kwargs):
                self.assertEqual(kwargs["dir"], "/dev/shm")
                return TemporaryDirectory(dir=root)

            with (
                patch("src.codex_backend.TemporaryDirectory", side_effect=temporary),
                patch("src.codex_backend.asyncio.create_subprocess_exec", side_effect=spawn),
            ):
                self.assertEqual(await invoke("Условие из изображения", structured=True, image=PNG), "Результат")
            self.assertFalse(attached[0].exists())
            self.assertEqual(list(Path(root).iterdir()), [])

    async def test_failed_or_cancelled_request_removes_image(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), TemporaryDirectory(dir="/tmp") as root:
                process = MagicMock(returncode=None if cancel else 1)
                process.stdin.drain = AsyncMock()
                process.wait = AsyncMock()
                started = asyncio.Event()

                async def read(size):
                    started.set()
                    if cancel:
                        await asyncio.Event().wait()
                    return b""

                process.stdout.read = AsyncMock(side_effect=read)
                with (
                    patch("src.codex_backend.TemporaryDirectory", side_effect=lambda **kw: TemporaryDirectory(dir=root)),
                    patch("src.codex_backend.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process),
                    patch("src.codex_backend.os.killpg") as kill,
                ):
                    task = asyncio.create_task(invoke("Условие", image=PNG))
                    await asyncio.wait_for(started.wait(), 1)
                    if cancel:
                        task.cancel()
                    with self.assertRaises(asyncio.CancelledError if cancel else CodexError):
                        await task
                    self.assertEqual(kill.call_count, int(cancel))
                self.assertEqual(list(Path(root).iterdir()), [])

    async def test_invalid_image_never_starts_cli(self):
        with patch("src.codex_backend.asyncio.create_subprocess_exec", new_callable=AsyncMock) as spawn:
            with self.assertRaises(CodexError):
                await invoke("Условие", image=b"not PNG")
            spawn.assert_not_called()
