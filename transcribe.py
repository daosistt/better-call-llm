"""Ручная проверка ASR на коротком файле: python transcribe.py задача.wav."""

import argparse
import sys
from pathlib import Path

from speech import SpeechRecognizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Распознавание русской речи локально: Whisper base/small, CPU INT8, Silero VAD.")
    parser.add_argument("audio", type=Path, help="Короткий аудиофайл WAV, MP3, M4A или другой формат PyAV")
    parser.add_argument("--offline", action="store_true", help="Использовать только уже скачанную модель, без доступа к сети")
    parser.add_argument("--model", choices=["base", "small"], default="base")
    args = parser.parse_args()
    if not args.audio.is_file():
        print("Ошибка: аудиофайл не найден.", file=sys.stderr)
        return 1
    print(f"Распознавание: {args.model}, CPU INT8, язык ru, Silero VAD.", file=sys.stderr)
    if not args.offline:
        print("При первом запуске веса модели загрузятся в локальный кеш. Аудио никуда не отправляется.", file=sys.stderr)
    try:
        segments = SpeechRecognizer(offline=args.offline, model_size=args.model).transcribe_file(args.audio)
    except KeyboardInterrupt:
        print("Распознавание остановлено.", file=sys.stderr)
        return 130
    except Exception:
        print(
            "Ошибка распознавания. Проверьте аудиофайл и наличие выбранной модели в кеше. "
            "Для первой загрузки требуется интернет; --offline запрещает загрузку.",
            file=sys.stderr,
        )
        return 1
    if not segments:
        print("Речь не обнаружена.")
    for segment in segments:
        print(f"[{segment.start:.2f}–{segment.end:.2f}] {segment.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
