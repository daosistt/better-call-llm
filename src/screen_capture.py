"""Разовый снимок экрана Wayland через grim, только в оперативной памяти."""

import asyncio
from dataclasses import dataclass
from typing import Literal

MAX_IMAGE_BYTES = 16 * 1024 * 1024
CAPTURE_TIMEOUT = 10
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ScreenCaptureError(Exception):
    """Сообщение для интерфейса без вывода внутренних ошибок процесса."""


@dataclass(frozen=True)
class Screenshot:
    status: Literal["waiting", "capturing", "done", "error"] = "waiting"
    image: bytes = b""
    captured_at: str = ""
    error: str = ""


async def capture_screen() -> bytes:
    """Захватывает все мониторы ноутбука; '-' направляет PNG в stdout."""
    try:
        process = await asyncio.create_subprocess_exec(
            "grim", "-t", "png", "-l", "1", "-s", "1", "-",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise ScreenCaptureError("Не найден grim. Установите его на ноутбуке для захвата экрана Wayland.") from None
    except OSError:
        raise ScreenCaptureError("Не удалось запустить захват экрана на ноутбуке.") from None

    try:
        async with asyncio.timeout(CAPTURE_TIMEOUT):
            assert process.stdout is not None
            image = bytearray()
            while chunk := await process.stdout.read(65536):
                if len(image) + len(chunk) > MAX_IMAGE_BYTES:
                    raise ScreenCaptureError("Снимок превышает 16 МБ. Уменьшите разрешение или число активных мониторов.")
                image.extend(chunk)
            await process.wait()
        if process.returncode != 0:
            raise ScreenCaptureError(
                "grim не смог получить снимок. Запустите сервер из графической сессии ноутбука. "
                "Этот способ рассчитан на Wayland с поддержкой grim, например Hyprland или Sway."
            )
        if not image.startswith(PNG_SIGNATURE):
            raise ScreenCaptureError("Захват экрана не вернул изображение PNG. Попробуйте ещё раз.")
        return bytes(image)
    except TimeoutError:
        raise ScreenCaptureError("Захват экрана занял больше 10 секунд. Попробуйте ещё раз.") from None
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
