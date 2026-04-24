import logging
import os
import threading
import time
import numpy as np
import pyaudio

from src.config import get_int_env

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = get_int_env("AUDIO_SAMPLE_RATE", 16000)
AUDIO_CHANNELS = get_int_env("AUDIO_CHANNELS", 1)
SPEECH_TIMEOUT = get_int_env("SPEECH_TIMEOUT", 10)
FOLLOWUP_TIMEOUT = get_int_env("FOLLOWUP_TIMEOUT", 5)
CHUNK_SIZE = 1024
ENERGY_THRESHOLD = 300
SILENCE_DURATION = 1.5
BARGE_IN_CHUNKS = 3


class ListenTimeout(Exception):
    pass


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        timeout: int = SPEECH_TIMEOUT,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._timeout = timeout

    def _compute_energy(self, chunk: bytes) -> float:
        pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(pcm ** 2)))

    def record(self, timeout: int | None = None) -> bytes:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            rate=self._sample_rate,
            channels=self._channels,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        frames: list[bytes] = []
        speech_started = False
        silence_start: float | None = None
        start_time = time.monotonic()
        logger.info("Recording... speak now")

        try:
            while True:
                chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                elapsed = time.monotonic() - start_time
                if not speech_started and elapsed >= (timeout if timeout is not None else self._timeout):
                    logger.info("Listen timeout, returning to idle")
                    raise ListenTimeout("No speech detected within timeout")

                energy = self._compute_energy(chunk)

                if energy >= ENERGY_THRESHOLD:
                    if not speech_started:
                        logger.info("Speech detected")
                    speech_started = True
                    silence_start = None
                    frames.append(chunk)
                elif speech_started:
                    frames.append(chunk)
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start >= SILENCE_DURATION:
                        logger.info("Speech ended, processing")
                        break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        return b"".join(frames)

    def monitor_for_barge_in(self, stop_event: threading.Event) -> None:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            rate=self._sample_rate,
            channels=self._channels,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        consecutive_loud = 0
        try:
            while not stop_event.is_set():
                chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                energy = self._compute_energy(chunk)
                if energy >= ENERGY_THRESHOLD:
                    consecutive_loud += 1
                    if consecutive_loud >= BARGE_IN_CHUNKS:
                        stop_event.set()
                        return
                else:
                    consecutive_loud = 0
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


def test_listen_timeout_raises() -> None:
    from unittest.mock import MagicMock, patch
    import numpy as np

    silent_chunk = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

    mock_stream = MagicMock()
    mock_stream.read.return_value = silent_chunk

    mock_audio = MagicMock()
    mock_audio.open.return_value = mock_stream

    recorder = AudioRecorder(timeout=0)

    with patch("src.recorder.pyaudio.PyAudio", return_value=mock_audio), \
         patch("src.recorder.time.monotonic", side_effect=[0.0, 1.0, 1.0, 1.0]):
        try:
            recorder.record()
            assert False, "Expected ListenTimeout"
        except ListenTimeout:
            pass


def test_default_timeout_raises_listen_timeout() -> None:
    from unittest.mock import MagicMock, patch
    import numpy as np

    silent_chunk = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

    mock_stream = MagicMock()
    mock_stream.read.return_value = silent_chunk

    mock_audio = MagicMock()
    mock_audio.open.return_value = mock_stream

    recorder = AudioRecorder()

    monotonic_values = [0.0] + [float(SPEECH_TIMEOUT + 1)] * 10

    with patch("src.recorder.pyaudio.PyAudio", return_value=mock_audio), \
         patch("src.recorder.time.monotonic", side_effect=monotonic_values):
        try:
            recorder.record()
            assert False, "Expected ListenTimeout"
        except ListenTimeout:
            pass


def test_vad_detects_speech_and_stops_on_silence() -> None:
    from unittest.mock import MagicMock, patch
    import numpy as np

    loud_chunk = (np.ones(CHUNK_SIZE, dtype=np.int16) * 1000).tobytes()
    silent_chunk = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

    chunks = [loud_chunk] * 5 + [silent_chunk] * 50

    mock_stream = MagicMock()
    mock_stream.read.side_effect = chunks

    mock_audio = MagicMock()
    mock_audio.open.return_value = mock_stream

    times = [0.0] + [float(i) * 0.1 for i in range(100)]

    with patch("src.recorder.pyaudio.PyAudio", return_value=mock_audio), \
         patch("src.recorder.time.monotonic", side_effect=times):
        recorder = AudioRecorder(timeout=10)
        result = recorder.record()

    assert len(result) > 0


def test_monitor_for_barge_in_sets_event() -> None:
    from unittest.mock import MagicMock, patch
    import numpy as np

    loud_chunk = (np.ones(CHUNK_SIZE, dtype=np.int16) * 1000).tobytes()

    mock_stream = MagicMock()
    mock_stream.read.return_value = loud_chunk

    mock_audio = MagicMock()
    mock_audio.open.return_value = mock_stream

    recorder = AudioRecorder()
    stop_event = threading.Event()

    with patch("src.recorder.pyaudio.PyAudio", return_value=mock_audio):
        recorder.monitor_for_barge_in(stop_event)

    assert stop_event.is_set()
