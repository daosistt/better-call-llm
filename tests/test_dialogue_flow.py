"""Сценарии целой цепочки с управляемыми ответами LLM, без сети и аудио."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from src.conversation import ExtractedTask
from src.worker import SolverWorker


class DialogueFlowTests(unittest.IsolatedAsyncioTestCase):
    async def until(self, predicate):
        async with asyncio.timeout(3):
            while not predicate():
                await asyncio.sleep(0.005)

    async def test_partial_discussion_correction_late_answer_and_new_task(self):
        worker = SolverWorker()
        worker.conversation.pause = 0.01
        worker.conversation.maximum = 0.02
        worker.task_stability = 0.01
        first = ExtractedTask("Найти производную f(x)=3*x^2", complete=True)
        corrected = ExtractedTask("Найти производную f(x)=4*x^2", complete=True)
        new_task = ExtractedTask("Как проверить эффективность промоакции?", complete=True)
        released = False
        cancelled = False
        finished_old = False
        calls = []

        async def solve(problem):
            nonlocal cancelled, finished_old
            calls.append(problem)
            if problem == first.condition:
                try:
                    while not released:
                        await asyncio.sleep(0.005)
                except asyncio.CancelledError:
                    cancelled = True
                    # Старый запрос может прислать поздний результат после отмены.
                    while not released:
                        await asyncio.sleep(0.005)
                finished_old = True
                return "Устаревший ответ: 6x"
            return "Актуальный ответ: " + problem

        with (
            patch("src.conversation.extract", new_callable=AsyncMock, side_effect=[
                ExtractedTask("Дана функция", complete=False), first, first, corrected, new_task,
            ]),
            patch("src.worker.solve_with_codex", side_effect=solve),
        ):
            try:
                worker.set_auto(True)
                worker.add_replica("Дана функция. Сейчас скажу какая.", "Ты")
                await self.until(lambda: worker.conversation.snapshot().status == "incomplete")
                self.assertEqual(calls, [])
                worker.add_replica("f(x)=3*x^2. Найти производную.", "Ты")
                await self.until(lambda: len(calls) == 1)
                worker.add_replica("Давай используем правило степени.", "Собеседник")
                await self.until(lambda: worker.conversation.snapshot().checks == 3 and worker.conversation.snapshot().status == "ready")
                await asyncio.sleep(0.04)
                self.assertEqual(calls, [first.condition])
                self.assertFalse(cancelled)
                worker.add_replica("Нет, коэффициент четыре.", "Ты")
                await self.until(lambda: len(calls) == 2 and worker.snapshot().status == "done")
                self.assertTrue(cancelled)
                self.assertEqual(worker.snapshot().problem, corrected.condition)
                released = True
                await self.until(lambda: finished_old)
                self.assertNotIn("Устаревший", worker.snapshot().text)
                worker.add_replica("Теперь другая задача: как проверить промоакцию?", "Собеседник")
                await self.until(lambda: len(calls) == 3 and worker.snapshot().status == "done")
                self.assertEqual(worker.snapshot().problem, new_task.condition)
                self.assertNotIn("производную", worker.snapshot().text)
            finally:
                released = True
                worker.close()
