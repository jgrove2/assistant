import io
import logging
import os
import threading
import wave
from collections.abc import Iterator

import httpx
import pyaudio
import pytest

from src.config import get_int_env

logger = logging.getLogger(__name__)

PIPER_URL = os.environ.get("PIPER_URL", "http://piper:5000")
SENTENCE_ENDINGS = {".", "?", "!"}
MAX_SPOKEN_SENTENCES = get_int_env("MAX_SPOKEN_SENTENCES", 5)
MAX_BUFFER_SIZE = 5000


def split_sentences(text_stream: Iterator[str]) -> Iterator[str]:
    buffer = ""
    for chunk in text_stream:
        buffer += chunk
        if len(buffer) > MAX_BUFFER_SIZE:
            truncated = buffer[:MAX_BUFFER_SIZE]
            last_boundary = -1
            for i, char in enumerate(truncated):
                if char in SENTENCE_ENDINGS:
                    last_boundary = i
            if last_boundary >= 0:
                logger.warning("Buffer exceeded MAX_BUFFER_SIZE, truncating at sentence boundary")
                sentence = truncated[:last_boundary + 1].strip()
                buffer = buffer[last_boundary + 1:]
                if sentence:
                    yield sentence
            else:
                logger.warning("Buffer exceeded MAX_BUFFER_SIZE, no sentence boundary found, truncating")
                sentence = truncated.strip()
                buffer = buffer[MAX_BUFFER_SIZE:]
                if sentence:
                    yield sentence
        while True:
            for i, char in enumerate(buffer):
                if char in SENTENCE_ENDINGS:
                    sentence = buffer[: i + 1].strip()
                    buffer = buffer[i + 1 :]
                    if sentence:
                        yield sentence
                    break
            else:
                break
    if buffer.strip():
        yield buffer.strip()


class TTSClient:
    def __init__(self, piper_url: str = PIPER_URL) -> None:
        self._piper_url = piper_url.rstrip("/")

    def _fetch_audio(self, sentence: str) -> bytes:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(self._piper_url + "/", json={"text": sentence})
            response.raise_for_status()
            return response.content

    def _play_wav(self, wav_bytes: bytes) -> None:
        with io.BytesIO(wav_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

        audio = pyaudio.PyAudio()
        try:
            fmt = audio.get_format_from_width(sample_width)
            stream = audio.open(
                format=fmt,
                channels=channels,
                rate=sample_rate,
                output=True,
            )
            try:
                stream.write(frames)
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            audio.terminate()

    def speak(self, text: str) -> None:
        wav_bytes = self._fetch_audio(text)
        self._play_wav(wav_bytes)

    def speak_stream(self, text_stream: Iterator[str], stop_event: threading.Event | None = None) -> None:
        logger.info("Speaking response...")
        sentence_count = 0
        for sentence in split_sentences(text_stream):
            logger.debug("TTS sentence: %s", sentence)
            wav_bytes = self._fetch_audio(sentence)
            self._play_wav(wav_bytes)
            sentence_count += 1
            if stop_event is not None and stop_event.is_set():
                logger.info("Barge-in detected, stopping speech")
                break
            if sentence_count >= MAX_SPOKEN_SENTENCES:
                logger.info("Max sentences reached, stopping speech")
                break


def test_split_sentences_basic() -> None:
    chunks = iter(["Hello world. How are", " you? I am fine!"])
    sentences = list(split_sentences(chunks))
    assert sentences == ["Hello world.", "How are you?", "I am fine!"]


def test_split_sentences_trailing_buffer() -> None:
    chunks = iter(["Hello world"])
    sentences = list(split_sentences(chunks))
    assert sentences == ["Hello world"]


def test_split_sentences_empty() -> None:
    chunks = iter([""])
    sentences = list(split_sentences(chunks))
    assert sentences == []


def test_speak_stream_calls_fetch_and_play() -> None:
    from unittest.mock import MagicMock, patch
    import struct
    import wave
    import io

    def make_wav() -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(struct.pack("<h", 0) * 100)
        return buf.getvalue()

    wav_data = make_wav()

    mock_response = MagicMock()
    mock_response.content = wav_data

    mock_http_client = MagicMock()
    mock_http_client.__enter__ = lambda s: s
    mock_http_client.__exit__ = MagicMock(return_value=False)
    mock_http_client.post.return_value = mock_response

    mock_pa_stream = MagicMock()
    mock_pa = MagicMock()
    mock_pa.open.return_value = mock_pa_stream
    mock_pa.get_format_from_width.return_value = pyaudio.paInt16

    with patch("src.tts_client.httpx.Client", return_value=mock_http_client), \
         patch("src.tts_client.pyaudio.PyAudio", return_value=mock_pa):
        client = TTSClient(piper_url="http://test:5000")
        client.speak_stream(iter(["Hello. World."]), stop_event=None)

    assert mock_http_client.post.call_count == 2


def test_speak_stream_stress(benchmark) -> None:
    pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
    from unittest.mock import MagicMock, patch
    import struct
    import wave
    import io

    def make_wav() -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(struct.pack("<h", 0) * 10)
        return buf.getvalue()

    wav_data = make_wav()

    mock_response = MagicMock()
    mock_response.content = wav_data

    mock_http_client = MagicMock()
    mock_http_client.__enter__ = lambda s: s
    mock_http_client.__exit__ = MagicMock(return_value=False)
    mock_http_client.post.return_value = mock_response

    mock_pa_stream = MagicMock()
    mock_pa = MagicMock()
    mock_pa.open.return_value = mock_pa_stream
    mock_pa.get_format_from_width.return_value = pyaudio.paInt16

    sentences = " ".join(f"Sentence number {i}." for i in range(100))

    with patch("src.tts_client.httpx.Client", return_value=mock_http_client), \
         patch("src.tts_client.pyaudio.PyAudio", return_value=mock_pa):
        tts = TTSClient(piper_url="http://test:5000")

        def run() -> None:
            tts.speak_stream(iter([sentences]))

        benchmark(run)


def test_speak_http_error_propagates() -> None:
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=MagicMock()
    )

    mock_http_client = MagicMock()
    mock_http_client.__enter__ = lambda s: s
    mock_http_client.__exit__ = MagicMock(return_value=False)
    mock_http_client.post.return_value = mock_response

    with patch("src.tts_client.httpx.Client", return_value=mock_http_client):
        client = TTSClient(piper_url="http://test:5000")
        try:
            client.speak("Hello.")
            assert False, "Expected HTTPStatusError"
        except httpx.HTTPStatusError:
            pass


def test_play_wav_invalid_bytes_cleans_up() -> None:
    from unittest.mock import MagicMock, patch

    mock_pa = MagicMock()

    with patch("src.tts_client.pyaudio.PyAudio", return_value=mock_pa):
        client = TTSClient(piper_url="http://test:5000")
        try:
            client._play_wav(b"not-a-wav-file")
            assert False, "Expected wave.Error"
        except wave.Error:
            pass

    mock_pa.terminate.assert_called_once()


def test_speak_stream_stops_on_barge_in() -> None:
    from unittest.mock import MagicMock, patch
    import struct
    import wave
    import io

    def make_wav() -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(struct.pack("<h", 0) * 100)
        return buf.getvalue()

    wav_data = make_wav()

    mock_response = MagicMock()
    mock_response.content = wav_data

    mock_http_client = MagicMock()
    mock_http_client.__enter__ = lambda s: s
    mock_http_client.__exit__ = MagicMock(return_value=False)
    mock_http_client.post.return_value = mock_response

    mock_pa_stream = MagicMock()
    mock_pa = MagicMock()
    mock_pa.open.return_value = mock_pa_stream
    mock_pa.get_format_from_width.return_value = pyaudio.paInt16

    stop_event = threading.Event()
    stop_event.set()

    with patch("src.tts_client.httpx.Client", return_value=mock_http_client), \
         patch("src.tts_client.pyaudio.PyAudio", return_value=mock_pa):
        client = TTSClient(piper_url="http://test:5000")
        client.speak_stream(iter(["Hello. World. Goodbye."]), stop_event=stop_event)

    assert mock_http_client.post.call_count == 1


def test_speak_stream_stops_at_max_sentences() -> None:
    from unittest.mock import MagicMock, patch
    import struct
    import wave
    import io

    def make_wav() -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(struct.pack("<h", 0) * 100)
        return buf.getvalue()

    wav_data = make_wav()

    mock_response = MagicMock()
    mock_response.content = wav_data

    mock_http_client = MagicMock()
    mock_http_client.__enter__ = lambda s: s
    mock_http_client.__exit__ = MagicMock(return_value=False)
    mock_http_client.post.return_value = mock_response

    mock_pa_stream = MagicMock()
    mock_pa = MagicMock()
    mock_pa.open.return_value = mock_pa_stream
    mock_pa.get_format_from_width.return_value = pyaudio.paInt16

    sentences = " ".join(f"Sentence {i}." for i in range(10))

    with patch("src.tts_client.httpx.Client", return_value=mock_http_client), \
         patch("src.tts_client.pyaudio.PyAudio", return_value=mock_pa), \
         patch("src.tts_client.MAX_SPOKEN_SENTENCES", 2):
        client = TTSClient(piper_url="http://test:5000")
        client.speak_stream(iter([sentences]), stop_event=None)

    assert mock_http_client.post.call_count == 2
