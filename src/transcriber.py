import io
import logging
import os
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
AUDIO_SAMPLE_RATE = int(os.environ.get("AUDIO_SAMPLE_RATE", "16000"))


class Transcriber:
    def __init__(self, model_name: str = WHISPER_MODEL) -> None:
        self._model = WhisperModel(model_name, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    def transcribe(self, audio_bytes: bytes) -> str:
        logger.info("Transcribing audio...")
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(pcm, language="en")
        result = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info("Transcribed: %s", result)
        return result


def test_transcribe_returns_text() -> None:
    from unittest.mock import MagicMock, patch

    mock_segment = MagicMock()
    mock_segment.text = "hello world"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())

    with patch("src.transcriber.WhisperModel", return_value=mock_model):
        transcriber = Transcriber()
        result = transcriber.transcribe(bytes(3200))

    assert result == "hello world"


def test_transcribe_joins_multiple_segments() -> None:
    from unittest.mock import MagicMock, patch

    seg1 = MagicMock()
    seg1.text = "hello"
    seg2 = MagicMock()
    seg2.text = "world"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())

    with patch("src.transcriber.WhisperModel", return_value=mock_model):
        transcriber = Transcriber()
        result = transcriber.transcribe(bytes(3200))

    assert result == "hello world"
