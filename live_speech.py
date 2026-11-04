"""Ограниченные очереди PCM → Silero VAD → последовательный Whisper."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from time import monotonic

import numpy as np

from speech import SpeechRecognizer

RATE = 16000


@dataclass(frozen=True)
class Utterance:
    channel: str
    start: float
    end: float
    pcm: bytes


@dataclass(frozen=True)
class Replica:
    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class LiveState:
    status: str = "waiting"
    replicas: tuple[Replica, ...] = ()
    dropped_blocks: int = 0
    dropped_fragments: int = 0
    backlog_seconds: float = 0
    raw_blocks: int = 0
    fragments: int = 0
    error: str = ""


class Segmenter:
    """VAD пересматривает окно не более 12 с, раз в 0,5 с, вне event loop."""

    def __init__(self, channel: str, maximum_seconds: float = 12) -> None:
        self.channel = channel
        self.maximum_seconds = maximum_seconds
        self.buffer = bytearray()
        self.origin = 0.0
        self.pending = 0

    def reset(self) -> None:
        self.buffer.clear()
        self.pending = 0

    def push(self, pcm: bytes, start: float) -> list[Utterance]:
        from faster_whisper.vad import get_speech_timestamps

        if not self.buffer:
            self.origin = start
        self.buffer.extend(pcm)
        self.pending += len(pcm)
        if self.pending < RATE:  # 0,5 с × 2 байта.
            return []
        self.pending = 0
        samples = np.frombuffer(bytes(self.buffer), dtype="<i2").astype(np.float32) / 32768
        ranges = get_speech_timestamps(
            samples, sampling_rate=RATE, min_silence_duration_ms=600,
            speech_pad_ms=150, min_speech_duration_ms=250,
        )
        result = []
        consumed = 0
        for speech in ranges:
            end = speech["end"]
            if len(samples) - end >= int(0.4 * RATE) or len(samples) >= self.maximum_seconds * RATE:
                begin = speech["start"]
                result.append(Utterance(
                    self.channel, self.origin + begin / RATE, self.origin + end / RATE,
                    bytes(self.buffer[begin * 2:end * 2]),
                ))
                consumed = end
            else:
                break
        if not ranges:
            consumed = max(0, len(samples) - RATE // 2)
        elif len(samples) >= self.maximum_seconds * RATE and not consumed:
            # Жёсткая граница памяти даже при необычном результате VAD.
            consumed = len(samples)
        if consumed:
            del self.buffer[:consumed * 2]
            self.origin += consumed / RATE
        return result


class LiveSpeech:
    def __init__(self, recognizer: SpeechRecognizer, on_replica: Callable[[Replica], None], on_gap: Callable[[], None],
                 *, audio_buffer_seconds: float = 5, asr_queue_seconds: float = 48, fragment_seconds: float = 12) -> None:
        self.recognizer = recognizer
        self.on_replica, self.on_gap = on_replica, on_gap
        self.origin = monotonic()
        self.fragment_seconds = fragment_seconds
        self.raw = {channel: asyncio.Queue(maxsize=max(1, int(audio_buffer_seconds * 10))) for channel in ("microphone", "system")}
        self.sequence = {channel: 0 for channel in self.raw}
        self.fragments: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max(1, int(asr_queue_seconds / fragment_seconds)))
        self._counter = 0
        self._lock = Lock()
        self._state = LiveState(status="listening")
        self._tasks: list[asyncio.Task] = []
        self._active = True
        self._epoch = 0

    def snapshot(self) -> LiveState:
        with self._lock:
            return self._state

    def _update(self, **changes) -> None:
        with self._lock:
            self._state = replace(self._state, **changes)

    def _gap(self, *, blocks: int = 0, fragments: int = 0) -> None:
        state = self.snapshot()
        self._update(dropped_blocks=state.dropped_blocks + blocks,
                     dropped_fragments=state.dropped_fragments + fragments)
        self.on_gap()

    def feed(self, channel: str, pcm: bytes) -> None:
        if not self._active:
            return
        queue = self.raw[channel]
        self.sequence[channel] += 1
        if queue.full():
            queue.get_nowait()
            self._gap(blocks=1)
        queue.put_nowait((self.sequence[channel], pcm, max(0, monotonic() - self.origin - 0.1)))
        self._update(raw_blocks=sum(item.qsize() for item in self.raw.values()))

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._vad(channel)) for channel in self.raw]
        self._tasks.append(asyncio.create_task(self._asr()))

    async def close(self) -> None:
        self._active = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for queue in [*self.raw.values(), self.fragments]:
            while not queue.empty():
                queue.get_nowait()
        self._update(status="stopped", raw_blocks=0, fragments=0, backlog_seconds=0)

    def reset_context(self) -> None:
        self._epoch += 1
        for queue in [*self.raw.values(), self.fragments]:
            while not queue.empty():
                queue.get_nowait()

    async def _vad(self, channel: str) -> None:
        segmenter = Segmenter(channel, self.fragment_seconds)
        previous = 0
        previous_epoch = self._epoch
        try:
            while True:
                sequence, pcm, start = await self.raw[channel].get()
                epoch = self._epoch
                if sequence != previous + 1 or epoch != previous_epoch:
                    segmenter.reset()
                previous = sequence
                previous_epoch = epoch
                utterances = await asyncio.to_thread(segmenter.push, pcm, start)
                if epoch != self._epoch:
                    continue
                for utterance in utterances:
                    if self.fragments.full():
                        self.fragments.get_nowait()
                        self._gap(fragments=1)
                    self._counter += 1
                    self.fragments.put_nowait((utterance.start, self._counter, epoch, utterance))
                self._update(raw_blocks=sum(item.qsize() for item in self.raw.values()), fragments=self.fragments.qsize())
        except asyncio.CancelledError:
            raise
        except Exception:
            self._update(error="Ошибка VAD. Остановите прослушивание и запустите его снова.")
            self._gap()

    async def _asr(self) -> None:
        while True:
            _, _, epoch, utterance = await self.fragments.get()
            self._update(status="recognizing", fragments=self.fragments.qsize(),
                         backlog_seconds=max(0, monotonic() - self.origin - utterance.end))
            try:
                segments = await asyncio.to_thread(self.recognizer.transcribe_pcm, utterance.pcm)
                if not self._active:
                    return
                if epoch != self._epoch:
                    self._update(status="listening", backlog_seconds=0)
                    continue
                text = " ".join(segment.text for segment in segments).strip()
                if text:
                    replica = Replica("Ты" if utterance.channel == "microphone" else "Собеседник",
                                      utterance.start, utterance.end, text)
                    lines = sorted((*self.snapshot().replicas, replica), key=lambda item: item.start)[-50:]
                    self._update(replicas=tuple(lines))
                    self.on_replica(replica)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._update(error="Ошибка распознавания. Проверьте локальную модель base и перезапустите прослушивание.")
                self._gap()
            self._update(status="listening", backlog_seconds=0)
