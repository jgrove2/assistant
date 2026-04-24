import logging
from typing import Optional

import httpx

from src.recorder import ListenTimeout

logger = logging.getLogger(__name__)

RECORD_TIMEOUT_SECONDS = 30.0


class AudioError(Exception):
    pass


class AudioClient:
    def __init__(self, url: str) -> None:
        self._url = url.rstrip("/")

    def record(self, timeout: Optional[int] = None) -> bytes:
        logger.info("Requesting audio recording...")
        payload = {"timeout": timeout} if timeout is not None else {}
        try:
            with httpx.Client(timeout=RECORD_TIMEOUT_SECONDS) as client:
                response = client.post(f"{self._url}/record", json=payload)
                if response.status_code == 408:
                    raise ListenTimeout("No speech detected within timeout")
                response.raise_for_status()
                return response.content
        except httpx.HTTPStatusError as e:
            logger.error("Audio service error: %s", e)
            raise AudioError(str(e)) from e

    def start_barge_in(self) -> None:
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(f"{self._url}/barge_in/start")
        except Exception as e:
            logger.warning("Failed to start barge-in monitor: %s", e)

    def stop_barge_in(self) -> None:
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(f"{self._url}/barge_in/stop")
        except Exception as e:
            logger.warning("Failed to stop barge-in monitor: %s", e)


def test_record_returns_bytes() -> None:
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"audio"

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.audio_client.httpx.Client", return_value=mock_client_instance):
        client = AudioClient("http://localhost:5001")
        result = client.record()
    assert result == b"audio"


def test_record_listen_timeout() -> None:
    import pytest
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 408

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.audio_client.httpx.Client", return_value=mock_client_instance):
        client = AudioClient("http://localhost:5001")
        with pytest.raises(ListenTimeout):
            client.record()


def test_record_http_error() -> None:
    import pytest
    from unittest.mock import MagicMock, patch

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=MagicMock()
    )
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.audio_client.httpx.Client", return_value=mock_client_instance):
        client = AudioClient("http://localhost:5001")
        with pytest.raises(AudioError):
            client.record()


def test_start_barge_in_swallows_exception() -> None:
    from unittest.mock import MagicMock, patch

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = Exception("fail")
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.audio_client.httpx.Client", return_value=mock_client_instance):
        client = AudioClient("http://localhost:5001")
        client.start_barge_in()


def test_stop_barge_in_swallows_exception() -> None:
    from unittest.mock import MagicMock, patch

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = Exception("fail")
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.audio_client.httpx.Client", return_value=mock_client_instance):
        client = AudioClient("http://localhost:5001")
        client.stop_barge_in()

