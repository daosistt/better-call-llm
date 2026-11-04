"""Одну модель за запуск: сравнивать base и small в отдельных процессах."""

import argparse
import json
import resource
from pathlib import Path
from time import perf_counter

from src.speech import SpeechRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["base", "small"], required=True)
    parser.add_argument("--download", action="store_true", help="Разрешить загрузку весов")
    parser.add_argument("audio", type=Path, nargs="+", help="Предоставленные тестовые записи")
    args = parser.parse_args()
    recognizer = SpeechRecognizer(offline=not args.download, model_size=args.model)
    for path in args.audio:
        started = perf_counter()
        segments = recognizer.transcribe_file(path)
        cold = perf_counter() - started
        times = []
        for _ in range(3):
            started = perf_counter()
            recognizer.transcribe_file(path)
            times.append(perf_counter() - started)
        print(json.dumps({
            "model": args.model, "file": path.name,
            "first_call_seconds": round(cold, 3), "warm_seconds": [round(t, 3) for t in times],
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "text": " ".join(segment.text for segment in segments),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
