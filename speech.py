"""Локальная ASR для законченного аудиофрагмента, без API и записи истории."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, BinaryIO
import numpy as np

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float
    text: str


class SpeechRecognizer:
    """Один экземпляр модели и не более одного инференса одновременно.

    Синхронный CPU-bound метод: при подключении к приложению вызывать
    через asyncio.to_thread, не из event loop. Создание объекта не скачивает
    модель; загрузка происходит при первом распознавании.
    """

    def __init__(self, *, offline: bool = False) -> None:
        self._offline = offline
        self._model: WhisperModel | None = None
        self._lock = Lock()

    def transcribe_file(self, path: Path) -> tuple[SpeechSegment, ...]:
        if not path.is_file():
            raise FileNotFoundError("Аудиофайл не найден.")
        return self._transcribe(str(path))

    def transcribe_bytes(self, audio: bytes) -> tuple[SpeechSegment, ...]:
        """Декодирование загруженного файла в памяти, без временного файла."""
        if not audio:
            raise ValueError("Аудиофайл пуст.")
        return self._transcribe(BytesIO(audio))

    def transcribe_pcm(self, pcm: bytes) -> tuple[SpeechSegment, ...]:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        return self._transcribe(samples)

    def _transcribe(self, source: str | BinaryIO | np.ndarray) -> tuple[SpeechSegment, ...]:
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    "base",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4,
                    num_workers=1,
                    local_files_only=self._offline,
                )
            segments, _ = self._model.transcribe(
                source,
                language="ru",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
            )
            # faster-whisper возвращает ленивый генератор: весь инференс
            # и потребление генератора должны оставаться внутри блокировки.
            return tuple(
                SpeechSegment(segment.start, segment.end, segment.text.strip())
                for segment in segments
                if segment.text.strip()
            )
