"""Первый шаг живого захвата: два PCM-потока и измерение уровня без хранения."""

import asyncio
import json
import subprocess
from dataclasses import dataclass, replace
from threading import Lock
from collections.abc import Callable

import numpy as np


@dataclass(frozen=True)
class AudioSource:
    name: str
    description: str
    monitor: bool


def list_sources() -> tuple[tuple[AudioSource, ...], str, str]:
    def pactl(*args: str) -> str:
        return subprocess.run(
            ["pactl", *args], check=True, capture_output=True, text=True, timeout=3,
        ).stdout.strip()

    sources = tuple(
        AudioSource(item["name"], item.get("description") or item["name"], item["name"].endswith(".monitor"))
        for item in json.loads(pactl("--format=json", "list", "sources"))
    )
    return sources, pactl("get-default-source"), pactl("get-default-sink") + ".monitor"


def level(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
    return max(0.0, min(1.0, (20 * np.log10(max(rms, 1e-6)) + 60) / 60))


@dataclass(frozen=True)
class CaptureState:
    running: bool = False
    microphone_level: float = 0.0
    system_level: float = 0.0
    microphone_blocks: int = 0
    system_blocks: int = 0
    error: str = ""


class AudioCapture:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state = CaptureState()
        self._generation = 0

    def snapshot(self) -> CaptureState:
        with self._lock:
            return self._state

    def prepare(self) -> int:
        with self._lock:
            self._generation += 1
            self._state = CaptureState(running=True)
            return self._generation

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            self._state = replace(self._state, running=False, microphone_level=0, system_level=0)

    async def run(self, microphone: str, monitor: str, generation: int,
                  on_block: Callable[[str, bytes], None] | None = None) -> None:
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._read(microphone, "microphone", generation, on_block))
                group.create_task(self._read(monitor, "system", generation, on_block))
        except asyncio.CancelledError:
            raise
        except Exception:
            with self._lock:
                if generation == self._generation:
                    self._state = replace(
                        self._state, running=False, microphone_level=0, system_level=0,
                        error="Захват прерван. Проверьте подключение устройств и доступ к PulseAudio/PipeWire, затем обновите список.",
                    )

    async def _read(self, source: str, channel: str, generation: int,
                    on_block: Callable[[str, bytes], None] | None = None) -> None:
        process = await asyncio.create_subprocess_exec(
            "parec", "--raw", f"--device={source}", "--rate=16000",
            "--channels=1", "--format=s16le", "--latency-msec=100",
            "--client-name=Better call LLM", stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            assert process.stdout is not None
            while True:
                # 100 мс, 16 кГц, mono s16le; память не растёт со временем.
                block = await process.stdout.readexactly(3200)
                value = level(block)
                with self._lock:
                    if generation == self._generation:
                        count = getattr(self._state, f"{channel}_blocks") + 1
                        self._state = replace(self._state, **{
                            f"{channel}_level": value, f"{channel}_blocks": count,
                        })
                if on_block is not None:
                    on_block(channel, block)
        finally:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.kill()
                await process.wait()
