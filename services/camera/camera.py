import logging
import numpy as np
import cv2

from config import get_int_env

logger = logging.getLogger(__name__)

CAMERA_INDEX_MAX = get_int_env("CAMERA_INDEX_MAX", 3)


class CameraError(Exception):
    pass


class Camera:
    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._index: int = self._find_camera()

    def _find_camera(self) -> int:
        for i in range(CAMERA_INDEX_MAX):
            logger.debug("Trying camera index %d", i)
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                logger.info("Camera found at index %d", i)
                return i
            logger.debug("Camera index %d not available", i)
        raise CameraError("No camera found")

    def capture_frame(self) -> np.ndarray:
        cap = cv2.VideoCapture(self._index)
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("cap.read() returned ret=%s frame_is_none=%s", ret, frame is None)
                raise CameraError("Failed to capture frame")
            logger.debug("Captured frame: shape=%s dtype=%s", frame.shape, frame.dtype)
            return frame
        finally:
            cap.release()


def test_camera_finds_first_working_index() -> None:
    from unittest.mock import MagicMock, patch

    def mock_video_capture(index: int) -> MagicMock:
        cap = MagicMock()
        cap.isOpened.return_value = index == 1
        return cap

    with patch("camera.cv2.VideoCapture", side_effect=mock_video_capture):
        camera = Camera()
        assert camera._index == 1


def test_camera_raises_on_no_camera() -> None:
    import pytest
    from unittest.mock import MagicMock, patch

    def mock_video_capture(index: int) -> MagicMock:
        cap = MagicMock()
        cap.isOpened.return_value = False
        return cap

    with patch("camera.cv2.VideoCapture", side_effect=mock_video_capture):
        with pytest.raises(CameraError):
            Camera()


def test_capture_frame_returns_ndarray() -> None:
    from unittest.mock import MagicMock, patch

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def mock_find_camera(self: Camera) -> int:
        return 0

    def mock_video_capture(index: int) -> MagicMock:
        cap = MagicMock()
        cap.read.return_value = (True, fake_frame)
        return cap

    with patch.object(Camera, "_find_camera", mock_find_camera):
        with patch("camera.cv2.VideoCapture", side_effect=mock_video_capture):
            camera = Camera()
            result = camera.capture_frame()
            assert isinstance(result, np.ndarray)


def test_capture_frame_raises_on_read_failure() -> None:
    import pytest
    from unittest.mock import MagicMock, patch

    def mock_find_camera(self: Camera) -> int:
        return 0

    def mock_video_capture(index: int) -> MagicMock:
        cap = MagicMock()
        cap.read.return_value = (False, None)
        return cap

    with patch.object(Camera, "_find_camera", mock_find_camera):
        with patch("camera.cv2.VideoCapture", side_effect=mock_video_capture):
            camera = Camera()
            with pytest.raises(CameraError):
                camera.capture_frame()
