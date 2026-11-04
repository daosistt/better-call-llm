import asyncio
import unittest
from unittest.mock import MagicMock, patch
from threading import Event

from live_speech import LiveSpeech, Segmenter, Utterance
from speech import SpeechSegment


class SegmenterTests(unittest.TestCase):
    def test_phrase_emitted_after_silence(self):
        segmenter = Segmenter("microphone")
        with patch("faster_whisper.vad.get_speech_timestamps", return_value=[{"start": 0, "end": 8000}]):
            self.assertEqual(segmenter.push(b"\0\0" * 8000, 0), [])
            result = segmenter.push(b"\0\0" * 8000, 0.5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].channel, "microphone")
        self.assertEqual(result[0].start, 0)
        self.assertEqual(result[0].end, 0.5)

    def test_silence_and_continuous_speech_have_bounded_buffers(self):
        segmenter = Segmenter("system")
        with patch("faster_whisper.vad.get_speech_timestamps", return_value=[]):
            for index in range(300):
                self.assertEqual(segmenter.push(b"\0\0" * 1600, index / 10), [])
                self.assertLessEqual(len(segmenter.buffer), 32000)
        segmenter.reset()
        with patch("faster_whisper.vad.get_speech_timestamps", side_effect=lambda samples, **kw: [{"start": 0, "end": len(samples)}]):
            output = []
            for index in range(150):
                output.extend(segmenter.push(b"\0\0" * 1600, index / 10))
                self.assertLessEqual(len(segmenter.buffer), 12 * 32000)
        self.assertTrue(output)
        self.assertLessEqual(output[0].end - output[0].start, 12)


class LiveSpeechTests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_overflow_is_bounded_and_reported(self):
        gaps = []
        live = LiveSpeech(MagicMock(), lambda replica: None, lambda: gaps.append(True))
        for _ in range(70):
            live.feed("microphone", b"\0\0" * 1600)
        self.assertEqual(live.raw["microphone"].qsize(), 50)
        self.assertEqual(live.snapshot().dropped_blocks, 20)
        self.assertEqual(len(gaps), 20)
        await live.close()
        self.assertTrue(live.raw["microphone"].empty())

    async def test_both_speakers_reach_callback(self):
        replicas = []
        recognizer = MagicMock()
        recognizer.transcribe_pcm.return_value = (SpeechSegment(0, 1, "Реплика"),)
        live = LiveSpeech(recognizer, replicas.append, lambda: None)

        def push(segmenter, pcm, start):
            return [Utterance(segmenter.channel, start, start + 0.1, pcm)]

        with patch.object(Segmenter, "push", new=push):
            live.start()
            try:
                live.feed("microphone", b"\0\0" * 1600)
                live.feed("system", b"\0\0" * 1600)
                async with asyncio.timeout(2):
                    while len(replicas) < 2:
                        await asyncio.sleep(0.01)
            finally:
                await live.close()
        self.assertEqual({replica.speaker for replica in replicas}, {"Ты", "Собеседник"})
        self.assertEqual(recognizer.transcribe_pcm.call_count, 2)
        self.assertEqual(live.snapshot().status, "stopped")

    async def test_reset_discards_inflight_old_transcription(self):
        started, release = Event(), Event()
        replicas = []
        calls = 0

        def recognize(pcm):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                release.wait(2)
                return (SpeechSegment(0, 1, "OLD"),)
            return (SpeechSegment(0, 1, "NEW"),)

        recognizer = MagicMock()
        recognizer.transcribe_pcm.side_effect = recognize
        live = LiveSpeech(recognizer, replicas.append, lambda: None)

        def push(segmenter, pcm, start):
            return [Utterance(segmenter.channel, start, start + 0.1, pcm)]

        with patch.object(Segmenter, "push", new=push):
            live.start()
            try:
                live.feed("microphone", b"\0\0" * 1600)
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                live.reset_context()
                live.feed("microphone", b"\0\0" * 1600)
                release.set()
                async with asyncio.timeout(2):
                    while not replicas:
                        await asyncio.sleep(0.01)
            finally:
                release.set()
                await live.close()
        self.assertEqual([replica.text for replica in replicas], ["NEW"])
