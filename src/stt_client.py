import logging

import httpx

logger = logging.getLogger(__name__)


class STTError(Exception):
    pass


class STTClient:
    def __init__(self, url: str) -> None:
        self._url = url.rstrip("/")

    def transcribe(self, audio_bytes: bytes) -> str:
        logger.info("Sending audio to STT service (%d bytes)...", len(audio_bytes))
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self._url}/transcribe",
                    content=audio_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                )
                response.raise_for_status()
                try:
                    text = response.json().get("text", "")
                except ValueError as e:
                    logger.warning("STT response JSON decode error: %s", e)
                    raise STTError("Invalid JSON in STT response") from e
                logger.info("STT result: %s", text)
                return text
        except httpx.HTTPStatusError as e:
            logger.error("STT service error: %s", e)
            raise STTError(str(e)) from e


def test_transcribe_returns_text() -> None:
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"text": "hello"}

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.stt_client.httpx.Client", return_value=mock_client_instance):
        client = STTClient("http://localhost:5002")
        result = client.transcribe(b"audio")
    assert result == "hello"


def test_transcribe_missing_key_returns_empty() -> None:
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.stt_client.httpx.Client", return_value=mock_client_instance):
        client = STTClient("http://localhost:5002")
        result = client.transcribe(b"audio")
    assert result == ""


def test_transcribe_http_error_raises_stt_error() -> None:
    import pytest
    from unittest.mock import MagicMock, patch

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=MagicMock()
    )
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.stt_client.httpx.Client", return_value=mock_client_instance):
        client = STTClient("http://localhost:5002")
        with pytest.raises(STTError):
            client.transcribe(b"audio")

