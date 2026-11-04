import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from src.conversation import Conversation, ExtractedTask


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    def make(self, **kwargs):
        results = []
        manager = Conversation(results.append, pause=0.03, maximum=0.09, **kwargs)
        self.addCleanup(manager.close)
        return manager, results

    async def settled(self, manager):
        async with asyncio.timeout(2):
            while manager._runner is not None and not manager._runner.done():
                await asyncio.sleep(0.005)

    async def test_replicas_combined_and_idle_has_no_calls(self):
        manager, results = self.make()
        task = ExtractedTask("Найти производную x²", complete=True)
        with patch("src.conversation.extract", new_callable=AsyncMock, return_value=task) as extract:
            manager.add("Найти производную", "Ты")
            await asyncio.sleep(0.01)
            manager.add("икс в квадрате", "Собеседник")
            await self.settled(manager)
            await asyncio.sleep(0.1)
            extract.assert_awaited_once()
            self.assertEqual(len(extract.call_args.args[0]), 2)
            self.assertEqual(results, [task])

    async def test_continuous_replicas_hit_maximum(self):
        manager, _ = self.make()
        called = asyncio.Event()

        async def extract(context, previous):
            called.set()
            return ExtractedTask()

        with patch("src.conversation.extract", side_effect=extract):
            for _ in range(15):
                manager.add("продолжаем говорить", "Ты")
                await asyncio.sleep(0.01)
                if called.is_set():
                    break
            self.assertTrue(called.is_set())
            await self.settled(manager)

    async def test_stale_result_discarded_and_single_request(self):
        manager, results = self.make()
        started, release = asyncio.Event(), asyncio.Event()
        count = 0

        async def extract(context, previous):
            nonlocal count
            count += 1
            if count == 1:
                started.set()
                await release.wait()
                return ExtractedTask("x²", complete=True)
            return ExtractedTask("x³", complete=True)

        with patch("src.conversation.extract", side_effect=extract):
            manager.add("x²", "Ты")
            await asyncio.wait_for(started.wait(), 1)
            manager.add("нет, x³", "Ты")
            await asyncio.sleep(0.05)
            self.assertEqual(count, 1)
            release.set()
            await self.settled(manager)
            self.assertEqual([task.condition for task in results], ["x³"])

    async def test_errors_do_not_retry_without_new_text(self):
        manager, _ = self.make()
        with patch("src.conversation.extract", new_callable=AsyncMock, side_effect=ValueError) as extract:
            manager.add("Задача", "Ты")
            await self.settled(manager)
            await asyncio.sleep(0.1)
            extract.assert_awaited_once()
            self.assertEqual(manager.snapshot().status, "error")

    async def test_context_bounded_and_alternative_interpretations_ready(self):
        manager, _ = self.make()
        result = ExtractedTask("Найти производную x² или x³", complete=True, ambiguities=("Степень 2 либо 3",))
        with patch("src.conversation.extract", new_callable=AsyncMock, return_value=result):
            for _ in range(20):
                manager.add("x" * 2000, "Ты")
            await self.settled(manager)
            self.assertLessEqual(sum(map(len, manager.snapshot().replicas)), 8000)
            self.assertEqual(manager.snapshot().status, "ready")

    async def test_unfinished_question_still_waits(self):
        manager, _ = self.make()
        result = ExtractedTask("Есть функция", complete=False, ambiguities=("Нет функции и вопроса",))
        with patch("src.conversation.extract", new_callable=AsyncMock, return_value=result):
            manager.add("Есть функция", "Ты")
            await self.settled(manager)
            self.assertEqual(manager.snapshot().status, "incomplete")
