import tempfile
import unittest
import io
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from speech import SpeechRecognizer, SpeechSegment


class SpeechTests(unittest.TestCase):
    def test_real_audio_decoder_accepts_wav(self):
        # Проверка совместимости PyAV без ASR-модели и чужих аудиозаписей.
        from faster_whisper.audio import decode_audio

        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 1600)
        audio.seek(0)
        decoded = decode_audio(audio)
        self.assertEqual(decoded.shape, (1600,))
        self.assertTrue((decoded == 0).all())

    def test_missing_file_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("faster_whisper.WhisperModel") as model:
                with self.assertRaises(FileNotFoundError):
                    SpeechRecognizer().transcribe_file(Path(directory) / "absent.wav")
                model.assert_not_called()

    def test_local_model_reused_and_lazy_segments_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "placeholder.wav"
            path.touch()  # Декодирование заменено заглушкой; это не запись разговора.
            model = MagicMock()

            def transcribe(*args, **kwargs):
                return iter([
                    SimpleNamespace(start=0.5, end=2.0, text=" Найдите производную "),
                    SimpleNamespace(start=2.0, end=2.5, text="  "),
                ]), SimpleNamespace()

            model.transcribe.side_effect = transcribe
            with patch("faster_whisper.WhisperModel", return_value=model) as constructor:
                recognizer = SpeechRecognizer(offline=True, model_size="small")
                for _ in range(2):
                    self.assertEqual(recognizer.transcribe_file(path), (
                        SpeechSegment(0.5, 2.0, "Найдите производную"),
                    ))
                constructor.assert_called_once()
                self.assertEqual(constructor.call_args.args, ("small",))
                options = constructor.call_args.kwargs
                self.assertEqual(options["device"], "cpu")
                self.assertEqual(options["compute_type"], "int8")
                self.assertTrue(options["local_files_only"])
                self.assertEqual(model.transcribe.call_args.kwargs["language"], "ru")
                self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])

    def test_silence_has_no_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "placeholder.wav"
            path.touch()
            model = MagicMock()
            model.transcribe.return_value = (iter([]), SimpleNamespace())
            with patch("faster_whisper.WhisperModel", return_value=model):
                self.assertEqual(SpeechRecognizer().transcribe_file(path), ())
