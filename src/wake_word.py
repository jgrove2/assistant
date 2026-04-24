import logging
import os
import numpy as np
import pyaudio
from openwakeword.model import Model

from src.config import get_int_env

logger = logging.getLogger(__name__)

WAKE_WORD_MODEL = os.environ.get("WAKE_WORD_MODEL", "alexa")
AUDIO_SAMPLE_RATE = get_int_env("AUDIO_SAMPLE_RATE", 16000)
CHUNK_SIZE = 1280


class WakeWordDetector:
    def __init__(self, model_name: str = WAKE_WORD_MODEL) -> None:
        self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
        self._model_name = model_name

    def listen(self) -> None:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            rate=AUDIO_SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        logger.info("Listening for wake word...")
        try:
            while True:
                chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                pcm = np.frombuffer(chunk, dtype=np.int16)
                predictions = self._model.predict(pcm)
                score = max(predictions.values(), default=0.0)
                if score >= 0.5:
                    logger.info("Wake word detected (score=%.2f)", score)
                    return
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


def test_wake_word_detection_returns_on_high_score() -> None:
    from unittest.mock import MagicMock, patch

    mock_model = MagicMock()
    mock_model.predict.return_value = {"alexa": 0.9}

    mock_stream = MagicMock()
    mock_stream.read.return_value = bytes(CHUNK_SIZE * 2)

    mock_audio = MagicMock()
    mock_audio.open.return_value = mock_stream

    with patch("src.wake_word.Model", return_value=mock_model), \
         patch("src.wake_word.pyaudio.PyAudio", return_value=mock_audio):
        detector = WakeWordDetector()
        detector.listen()

    mock_model.predict.assert_called()


def test_wake_word_does_not_trigger_on_low_score() -> None:
    from unittest.mock import MagicMock, patch
    import threading

    call_count = 0
    max_calls = 5

    def fake_predict(pcm: np.ndarray) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count >= max_calls:
            return {"alexa": 0.9}
        return {"alexa": 0.1}

    mock_model = MagicMock()
    mock_model.predict.side_effect = fake_predict

    mock_stream = MagicMock()
    mock_stream.read.return_value = bytes(CHUNK_SIZE * 2)

    mock_audio = MagicMock()
    mock_audio.open.return_value = mock_stream

    with patch("src.wake_word.Model", return_value=mock_model), \
         patch("src.wake_word.pyaudio.PyAudio", return_value=mock_audio):
        detector = WakeWordDetector()
        detector.listen()

    assert call_count == max_calls
