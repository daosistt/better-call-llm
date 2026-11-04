"""Явная интеграционная проверка экстрактора: python -m checks.check_dialogue.

Пять настоящих вызовов Codex на синтетических репликах. Не включается
микрофон, не сохраняется разговор, решатель не запускается.
"""

import asyncio

from conversation import ExtractedTask, extract


async def main() -> None:
    context: list[str] = []
    previous = ExtractedTask()
    failures = []
    cases = [
        ("Неполное условие", "Дана функция. Сейчас скажу какая.", "incomplete"),
        ("Полное условие", "f(x)=3*x^2. Найдите её производную по x.", "complete"),
        ("Обсуждение решения", "Давай применим правило производной степени. Да, так и сделаем.", "unchanged"),
        ("Исправление коэффициента", "Стоп, коэффициент не три, а четыре: f(x)=4*x^2. Вопрос тот же.", "correction"),
        ("Новая задача", "Теперь другая задача. В интернет-магазине предлагают запустить промоакцию со скидкой. Как проверить, стоит ли её вводить?", "new"),
    ]
    for title, text, expected in cases:
        context.append("Ты: " + text)
        task = await extract(tuple(context), previous)
        if expected == "incomplete":
            valid = not task.complete
        elif expected == "unchanged":
            valid = task.complete and " ".join(task.solver_condition.split()) == " ".join(previous.solver_condition.split())
        elif expected == "correction":
            valid = task.complete and "4" in task.condition and "3" not in task.condition
        elif expected == "new":
            valid = task.complete and "акци" in task.condition.lower() and "производн" not in task.condition.lower()
        else:
            valid = task.complete and bool(task.condition)
        print(f'{"OK" if valid else "FAIL"}: {title}', flush=True)
        print(f'  complete={task.complete}; {task.solver_condition}', flush=True)
        if not valid:
            failures.append(title)
        previous = task
    if failures:
        raise RuntimeError("Проверки не прошли: " + ", ".join(failures))
    print("Все пять проверок прошли. Это один прогон, а не гарантия качества LLM.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
