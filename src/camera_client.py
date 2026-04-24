import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class CameraError(Exception):
    pass


class CameraClient:
    def __init__(self, url: str) -> None:
        self._url = url.rstrip("/")

    def identify(self) -> tuple[list[Optional[str]], str]:
        """Returns (names_list, context_string). names_list contains None for unknown faces."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self._url}/identify")
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as e:
                    logger.warning("Camera identify JSON decode error: %s", e)
                    return [], ""
                names = data.get("names", [])
                context = data.get("context", "")
                logger.info("Camera identify: names=%s", names)
                return names, context
        except httpx.HTTPStatusError as e:
            logger.error("Camera service error: %s", e)
            return [], ""
        except Exception as e:
            logger.warning("Camera service unavailable: %s", e)
            return [], ""

    def register(self, name: str) -> bool:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(f"{self._url}/register", json={"name": name})
                response.raise_for_status()
                return response.json().get("ok", False)
        except Exception as e:
            logger.warning("Failed to register face: %s", e)
            return False


def test_identify_returns_names_and_context() -> None:
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"names": ["Alice"], "context": "People visible: Alice"}

    mock_client_instance = MagicMock()
    mock_client_instance.get.return_value = mock_response
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.camera_client.httpx.Client", return_value=mock_client_instance):
        client = CameraClient("http://localhost:5003")
        result = client.identify()
    assert result == (["Alice"], "People visible: Alice")


def test_identify_http_error_returns_empty() -> None:
    from unittest.mock import MagicMock, patch

    mock_client_instance = MagicMock()
    mock_client_instance.get.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=MagicMock()
    )
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.camera_client.httpx.Client", return_value=mock_client_instance):
        client = CameraClient("http://localhost:5003")
        result = client.identify()
    assert result == ([], "")


def test_identify_connection_error_returns_empty() -> None:
    from unittest.mock import MagicMock, patch

    mock_client_instance = MagicMock()
    mock_client_instance.get.side_effect = Exception("connection refused")
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.camera_client.httpx.Client", return_value=mock_client_instance):
        client = CameraClient("http://localhost:5003")
        result = client.identify()
    assert result == ([], "")


def test_register_returns_true() -> None:
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True}

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.camera_client.httpx.Client", return_value=mock_client_instance):
        client = CameraClient("http://localhost:5003")
        result = client.register("Alice")
    assert result is True


def test_register_exception_returns_false() -> None:
    from unittest.mock import MagicMock, patch

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = Exception("fail")
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.camera_client.httpx.Client", return_value=mock_client_instance):
        client = CameraClient("http://localhost:5003")
        result = client.register("Alice")
    assert result is False


def test_register_returns_false_when_ok_is_false() -> None:
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": False}

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = lambda s: mock_client_instance
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("src.camera_client.httpx.Client", return_value=mock_client_instance):
        client = CameraClient("http://localhost:5003")
        result = client.register("Alice")
    assert result is False

