import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
from openai import AuthenticationError, RateLimitError

from src.config import Settings
from src.solver import SolverError, solve
from src.codex_backend import solve_with_codex


class SolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_edits_reach_both_backends_without_restart(self):
        with TemporaryDirectory() as directory:
            resources = Path(directory)
            (resources / "prompts").mkdir()
            prompt = resources / "prompts" / "solver.md"
            with (
                patch("src.prompt_loader.PROMPTS_DIR", resources / "prompts"),
                patch("src.codex_backend.invoke", new_callable=AsyncMock, return_value="Ответ") as invoke,
            ):
                for instructions in ("Объясняй методы анализа.", "Помогай с вопросами программирования."):
                    prompt.write_text(instructions, encoding="utf-8")
                    client = self.stream_client([SimpleNamespace(type="response.output_text.delta", delta="Ответ")])
                    with patch("src.solver.AsyncOpenAI", return_value=client):
                        await self.collect()
                    self.assertEqual(client.responses.stream.call_args.kwargs["instructions"], instructions)
                    await solve_with_codex("Как устроен asyncio?")
                    sent = invoke.call_args.args[0]
                    self.assertTrue(sent.startswith(instructions + "\n\n"))
                    self.assertEqual(json.loads(sent.splitlines()[-1]), {"condition": "Как устроен asyncio?"})

    async def collect(self) -> str:
        return "".join([chunk async for chunk in solve("Найти производную x²", Settings("test-key"))])

    def stream_client(self, events: list[SimpleNamespace], status: str = "completed"):
        async def iterate():
            for event in events:
                yield event

        stream = MagicMock()
        stream.__aenter__ = AsyncMock(return_value=stream)
        stream.__aexit__ = AsyncMock(return_value=False)
        stream.__aiter__.side_effect = iterate
        stream.get_final_response = AsyncMock(return_value=SimpleNamespace(status=status))
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.responses.stream.return_value = stream
        return client

    async def test_stream_and_request_privacy(self):
        client = self.stream_client([
            SimpleNamespace(type="response.output_text.delta", delta="Ответ: "),
            SimpleNamespace(type="response.output_text.delta", delta="$2x$"),
        ])
        with patch("src.solver.AsyncOpenAI", return_value=client):
            self.assertEqual(await self.collect(), "Ответ: $2x$")
        request = client.responses.stream.call_args.kwargs
        self.assertFalse(request["store"])
        self.assertEqual(request["model"], "gpt-5-mini")
        client.__aexit__.assert_awaited_once()

    async def test_incomplete_response_is_not_success(self):
        client = self.stream_client([
            SimpleNamespace(type="response.output_text.delta", delta="Часть решения"),
        ], "incomplete")
        received = []
        with patch("src.solver.AsyncOpenAI", return_value=client):
            with self.assertRaisesRegex(SolverError, "не завершён"):
                async for chunk in solve("Задача", Settings("test-key")):
                    received.append(chunk)
        self.assertEqual(received, ["Часть решения"])

    async def test_stream_error_and_empty_response(self):
        for events in ([SimpleNamespace(type="error")], []):
            with self.subTest(events=events):
                with patch("src.solver.AsyncOpenAI", return_value=self.stream_client(events)):
                    with self.assertRaises(SolverError):
                        await self.collect()

    async def test_api_errors_hide_raw_details(self):
        response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com"))
        for error, expected in (
            (AuthenticationError("secret-details", response=response, body=None), "ключ"),
            (RateLimitError("secret-details", response=response, body=None), "баланс"),
        ):
            with self.subTest(error=type(error).__name__):
                with patch("src.solver.AsyncOpenAI", side_effect=error):
                    with self.assertRaises(SolverError) as caught:
                        await self.collect()
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn("secret-details", str(caught.exception))

    async def test_missing_input_does_not_call_api(self):
        with patch("src.solver.AsyncOpenAI") as client:
            for problem, settings in ((" ", Settings("key")), ("Задача", Settings())):
                with self.assertRaises(SolverError):
                    _ = [chunk async for chunk in solve(problem, settings)]
            client.assert_not_called()
