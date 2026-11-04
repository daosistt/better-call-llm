"""Асинхронный текстовый решатель. Не обращается к Streamlit."""

from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from src.config import Settings
from src.prompt_loader import load_prompt


class SolverError(Exception):
    """Сообщение, которое можно показать пользователю без секретов."""


async def solve(problem: str, settings: Settings) -> AsyncIterator[str]:
    if not problem.strip():
        raise SolverError("Введите условие задачи.")
    if not settings.openai_api_key:
        raise SolverError("Добавьте OPENAI_API_KEY в .env.")
    try:
        # Без автоматических повторов: пользователь явно решает, повторять ли запрос.
        async with AsyncOpenAI(
            api_key=settings.openai_api_key, timeout=180.0, max_retries=0,
        ) as client:
            async with client.responses.stream(
                model=settings.openai_model,
                instructions=load_prompt("solver"),
                input=[{"role": "user", "content": problem.strip()}],
                reasoning={"effort": "medium"},
                max_output_tokens=16000,
                store=False,
            ) as stream:
                has_text = False
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        has_text = True
                        yield event.delta
                    elif event.type == "response.refusal.delta":
                        has_text = True
                        yield event.delta
                    elif event.type in {"error", "response.failed"}:
                        raise SolverError("API не смог завершить ответ. Попробуйте ещё раз.")
                    elif event.type == "response.incomplete":
                        raise SolverError("Ответ не завершён. Показанный текст может быть неполным.")
                response = await stream.get_final_response()
                if response.status != "completed":
                    raise SolverError("Ответ не завершён. Показанный текст может быть неполным.")
                if not has_text:
                    raise SolverError("API вернул пустой ответ. Попробуйте ещё раз.")
    except AuthenticationError:
        raise SolverError("API отклонил ключ. Проверьте OPENAI_API_KEY в .env.") from None
    except RateLimitError:
        raise SolverError("API ограничил запрос: проверьте баланс кредитов и лимиты API.") from None
    except APITimeoutError:
        raise SolverError("Превышено время ожидания API. Попробуйте ещё раз.") from None
    except APIConnectionError:
        raise SolverError("Не удалось связаться с API. Проверьте подключение к сети.") from None
    except APIStatusError as error:
        if error.status_code in {400, 403, 404}:
            message = "Запрос отклонён. Проверьте OPENAI_MODEL и доступ к модели в API."
        else:
            message = "Ошибка сервиса API. Попробуйте повторить запрос позже."
        raise SolverError(message) from None
