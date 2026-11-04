"""Два записанных примера вместо parec, настоящий VAD/ASR, LLM без сети.

python -m checks.check_stability --seconds 600 --report /tmp/math-stability.json
Сохраняет только численные метрики, не аудио, транскрипты или решения.
"""

import argparse
import asyncio
import json
import os
import resource
from collections import deque
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter
from unittest.mock import patch

import numpy as np
from faster_whisper.audio import decode_audio

from src.config import Settings
from src.conversation import ExtractedTask
from src.speech import SpeechRecognizer
from src.worker import SolverWorker


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS недоступен")


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


class TimedRecognizer(SpeechRecognizer):
    def __init__(self, model_size: str = "base") -> None:
        super().__init__(offline=True, model_size=model_size)
        self.times: deque[float] = deque(maxlen=2048)
        self.lock = Lock()

    def transcribe_pcm(self, pcm):
        started = perf_counter()
        result = super().transcribe_pcm(pcm)
        with self.lock:
            self.times.append(perf_counter() - started)
        return result


class ReplayProcess:
    """Минимальный контракт asyncio subprocess для двух reader-задач захвата."""

    def __init__(self, pcm: bytes, origin: float) -> None:
        self.pcm, self.origin = pcm, origin
        self.offset = self.blocks = 0
        self.returncode = None
        self.stdout = self
        self.stopped = asyncio.Event()

    async def readexactly(self, size: int) -> bytes:
        self.blocks += 1
        due = self.origin + self.blocks * 0.1
        await asyncio.sleep(max(0, due - monotonic()))
        result = self.pcm[self.offset:self.offset + size]
        self.offset += size
        if self.offset >= len(self.pcm):
            self.offset = 0
        return result.ljust(size, b"\0")

    def terminate(self) -> None:
        self.returncode = 0
        self.stopped.set()

    kill = terminate

    async def wait(self) -> int:
        await self.stopped.wait()
        return 0


async def main(seconds: float, paths: list[Path], report: Path | None, model_size: str = "base") -> None:
    if len(paths) != 2 or not all(path.is_file() for path in paths):
        raise ValueError("Нужны два существующих аудиопримера")
    clips = []
    for path in paths:
        audio = decode_audio(str(path))
        if len(audio) > 60 * 16000:
            raise ValueError("Для воспроизводимого теста используйте примеры до минуты")
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        pcm += b"\0\0" * 16000  # Пауза между повторениями.
        clips.append(pcm + b"\0" * (-len(pcm) % 3200))

    recognizer = TimedRecognizer(model_size)
    # Прогрев отдельно: загрузка модели не маскирует последующий рост памяти.
    recognizer.transcribe_pcm(clips[0])
    with recognizer.lock:
        recognizer.times.clear()
    worker = SolverWorker()
    processes = []
    samples = []
    latencies: deque[float] = deque(maxlen=2048)
    metrics_lock = Lock()
    errors: set[str] = set()
    observations = {"raw_max": 0, "fragments_max": 0, "replicas": 0, "extract_calls": 0, "solve_calls": 0}
    origin = monotonic()

    async def spawn(*command, **kwargs):
        clip = clips[0 if "--device=fixture-microphone" in command else 1]
        process = ReplayProcess(clip, origin)
        processes.append(process)
        return process

    async def fake_extract(context, previous):
        observations["extract_calls"] += 1
        await asyncio.sleep(5)
        return ExtractedTask("Синтетическая задача нагрузки", complete=True)

    async def fake_solve(problem):
        observations["solve_calls"] += 1
        await asyncio.sleep(20)
        return "Синтетический ответ для проверки неблокирующего ожидания"

    original_replica = worker._live_replica

    def receive(replica, token):
        with metrics_lock:
            observations["replicas"] += 1
            latencies.append(max(0, monotonic() - origin - replica.end))
        original_replica(replica, token)

    baseline_cpu = previous_cpu = cpu_seconds()
    previous_time = origin
    next_print = origin
    try:
        with (
            # new= не накапливает историю вызовов mock и не искажает рост RSS.
            patch("src.audio_capture.asyncio.create_subprocess_exec", new=spawn),
            patch("src.worker.get_recognizer", new=lambda: recognizer),
            patch("src.conversation.extract", new=fake_extract),
            patch("src.worker.solve_with_codex", new=fake_solve),
            patch.object(worker, "_live_replica", new=receive),
        ):
            worker.start_capture("fixture-microphone", "fixture-output.monitor", consent=True)
            worker.start("Синтетическая задача нагрузки", Settings(), use_codex=True)
            while monotonic() - origin < seconds:
                await asyncio.sleep(0.1)
                now = monotonic()
                state = worker.live_snapshot()
                observations["raw_max"] = max(observations["raw_max"], state.raw_blocks)
                observations["fragments_max"] = max(observations["fragments_max"], state.fragments)
                for error in (state.error, worker.capture.snapshot().error):
                    if error:
                        errors.add(error)
                if now >= next_print:
                    current_cpu = cpu_seconds()
                    sample = {
                        "elapsed_seconds": round(now - origin, 1),
                        "rss_mib": round(rss_mib(), 1),
                        "cpu_percent_one_core": round(100 * (current_cpu - previous_cpu) / (now - previous_time), 1),
                        "replicas": observations["replicas"],
                        "raw_blocks": state.raw_blocks, "fragments": state.fragments,
                        "dropped_blocks": state.dropped_blocks, "dropped_fragments": state.dropped_fragments,
                    }
                    samples.append(sample)
                    print(json.dumps(sample), flush=True)
                    previous_cpu, previous_time = current_cpu, now
                    next_print = now + 30
            final_state = worker.live_snapshot()
            capture = worker.capture.snapshot()
            warm_rss = rss_mib()
            elapsed = monotonic() - origin
            with metrics_lock:
                latency_sample = tuple(latencies)
            with recognizer.lock:
                asr_sample = tuple(recognizer.times)
            result = {
                "mode": "real-time dual fixture replay; real VAD/ASR; fake extractor 5s, fake solver 20s; no physical audio devices or network",
                "seconds": round(elapsed, 1), "logical_cpus": os.cpu_count(),
                "asr_model": model_size,
                "cpu_average_one_core_percent": round(100 * (cpu_seconds() - baseline_cpu) / elapsed, 1),
                "rss_initial_warm_mib": samples[0]["rss_mib"], "rss_final_mib": round(warm_rss, 1),
                "rss_peak_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "blocks": {"microphone": capture.microphone_blocks, "system": capture.system_blocks},
                **observations,
                "dropped_blocks": final_state.dropped_blocks, "dropped_fragments": final_state.dropped_fragments,
                "latency_p95_seconds": round(float(np.percentile(latency_sample, 95)), 3) if latency_sample else None,
                "asr_p95_seconds": round(float(np.percentile(asr_sample, 95)), 3) if asr_sample else None,
                "errors": sorted(errors), "samples": samples,
            }
            worker.stop_capture()
            async with asyncio.timeout(10):
                while any(process.returncode is None for process in processes):
                    await asyncio.sleep(0.05)
            result["subprocesses_stopped"] = all(process.returncode is not None for process in processes)
    finally:
        worker.close()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if report is not None:
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if result["dropped_blocks"] or result["dropped_fragments"] or result["errors"] or not result["replicas"]:
        raise RuntimeError("В прогоне обнаружены пропуски, ошибки или отсутствие распознавания")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--audio", nargs=2, type=Path, default=[Path("example.mp3"), Path("example1.mp3")])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", choices=["base", "small"], default="base")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 7200:
        parser.error("--seconds должен быть от 1 до 7200")
    asyncio.run(main(args.seconds, args.audio, args.report, args.model))
