import json
import logging
import os
import stat
import uuid
from pathlib import Path
import numpy as np
import face_recognition

from config import get_int_env

logger = logging.getLogger(__name__)

FACES_DIR = Path(os.environ.get("FACES_DIR", "faces"))
FACE_RECOGNITION_TOLERANCE = float(os.environ.get("FACE_RECOGNITION_TOLERANCE", "0.5"))
KNOWN_FACES_FILE = FACES_DIR / "known_faces.json"
FACES_IMAGES_DIR = FACES_DIR / "images"


class FaceRecognizer:
    def __init__(self) -> None:
        self._known_names: list[str] = []
        self._known_encodings: list[list[float]] = []
        self._load_known_faces()

    def _load_known_faces(self) -> None:
        logger.debug("Loading known faces from %s", KNOWN_FACES_FILE)
        if not KNOWN_FACES_FILE.exists():
            self._known_names = []
            self._known_encodings = []
            return
        try:
            with open(KNOWN_FACES_FILE) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load known faces: %s", e)
            self._known_names = []
            self._known_encodings = []
            return
        self._known_names = []
        self._known_encodings = []
        for person in data.get("people", []):
            for encoding in person.get("encodings", []):
                if isinstance(encoding, list) and len(encoding) == 128:
                    self._known_names.append(person["name"])
                    self._known_encodings.append(encoding)
                else:
                    logger.warning("Skipping invalid encoding for %s", person.get("name"))
        logger.debug("Loaded %d face encodings for %d unique people",
                     len(self._known_encodings), len(set(self._known_names)))

    def identify_faces(self, frame: np.ndarray) -> list[str | None]:
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        logger.debug("identify_faces: frame shape=%s dtype=%s", rgb_frame.shape, rgb_frame.dtype)
        face_locs = face_recognition.face_locations(rgb_frame)
        logger.debug("Face locations found: %d", len(face_locs))
        if not face_locs:
            return []
        encodings = face_recognition.face_encodings(rgb_frame)
        results: list[str | None] = []
        known_encodings = [np.array(e) for e in self._known_encodings]
        for encoding in encodings:
            if not known_encodings:
                results.append(None)
                continue
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_index = int(np.argmin(distances))
            logger.debug("Face %d distances: min=%.4f best_index=%d match=%s",
                         encodings.index(encoding), float(np.min(distances)) if len(distances) else -1.0,
                         best_index, matches[best_index] if known_encodings else False)
            if matches[best_index]:
                results.append(self._known_names[best_index])
            else:
                results.append(None)
        logger.info("Identified %d known and %d unknown faces",
                    sum(1 for r in results if r is not None),
                    sum(1 for r in results if r is None))
        return results

    def register_face(self, name: str, frame: np.ndarray) -> bool:
        sanitized_name = "".join(c for c in name if c.isalnum() or c in "_-").strip()
        if not sanitized_name:
            logger.warning("Registration rejected: name contains no valid characters")
            return False
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        encodings = face_recognition.face_encodings(rgb_frame)
        if not encodings:
            logger.warning("No face found in frame for registration")
            return False
        encoding = encodings[0]
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_dir = FACES_IMAGES_DIR / sanitized_name
        image_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        image_path = image_dir / f"{sanitized_name}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        import cv2
        cv2.imwrite(str(image_path), frame)
        os.chmod(image_path, stat.S_IRUSR | stat.S_IWUSR)
        self._known_names.append(sanitized_name)
        self._known_encodings.append(encoding.tolist())
        self._save_known_faces()
        logger.info("Registered face for %s", name)
        return True

    def _save_known_faces(self) -> None:
        KNOWN_FACES_FILE.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        people: dict[str, list[list[float]]] = {}
        for name, encoding in zip(self._known_names, self._known_encodings):
            if name not in people:
                people[name] = []
            people[name].append(encoding)
        data = {
            "people": [
                {"name": name, "encodings": encs}
                for name, encs in people.items()
            ]
        }
        with open(KNOWN_FACES_FILE, "w") as f:
            json.dump(data, f)
        os.chmod(KNOWN_FACES_FILE, stat.S_IRUSR | stat.S_IWUSR)


def test_identify_faces_returns_known_name() -> None:
    from unittest.mock import patch, MagicMock
    import numpy as np

    known_encoding = np.array([0.1] * 128)
    face_enc = np.array([0.1] * 128)

    recognizer = FaceRecognizer.__new__(FaceRecognizer)
    recognizer._known_names = ["Justin"]
    recognizer._known_encodings = [known_encoding.tolist()]

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("face_recognizer.face_recognition.face_locations", return_value=[(0, 100, 100, 0)]):
        with patch("face_recognizer.face_recognition.face_encodings", return_value=[face_enc]):
            with patch("face_recognizer.face_recognition.compare_faces", return_value=[True]):
                with patch("face_recognizer.face_recognition.face_distance", return_value=np.array([0.0])):
                    result = recognizer.identify_faces(dummy_frame)
                    assert result == ["Justin"]


def test_identify_faces_returns_none_for_unknown() -> None:
    from unittest.mock import patch
    import numpy as np

    known_encoding = np.array([0.1] * 128)
    face_enc = np.array([0.9] * 128)

    recognizer = FaceRecognizer.__new__(FaceRecognizer)
    recognizer._known_names = ["Justin"]
    recognizer._known_encodings = [known_encoding.tolist()]

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("face_recognizer.face_recognition.face_locations", return_value=[(0, 100, 100, 0)]):
        with patch("face_recognizer.face_recognition.face_encodings", return_value=[face_enc]):
            with patch("face_recognizer.face_recognition.compare_faces", return_value=[False]):
                with patch("face_recognizer.face_recognition.face_distance", return_value=np.array([0.9])):
                    result = recognizer.identify_faces(dummy_frame)
                    assert result == [None]


def test_identify_faces_returns_empty_for_no_faces() -> None:
    from unittest.mock import patch
    import numpy as np

    recognizer = FaceRecognizer.__new__(FaceRecognizer)
    recognizer._known_names = []
    recognizer._known_encodings = []

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("face_recognizer.face_recognition.face_locations", return_value=[]):
        result = recognizer.identify_faces(dummy_frame)
        assert result == []


def test_register_face_saves_and_updates(tmp_path: Path) -> None:
    from unittest.mock import patch
    import numpy as np

    encoding = np.array([0.1] * 128)

    recognizer = FaceRecognizer.__new__(FaceRecognizer)
    recognizer._known_names = []
    recognizer._known_encodings = []

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("face_recognizer.FACES_DIR", tmp_path):
        with patch("face_recognizer.KNOWN_FACES_FILE", tmp_path / "known_faces.json"):
            with patch("face_recognizer.FACES_IMAGES_DIR", tmp_path / "images"):
                with patch("face_recognizer.face_recognition.face_encodings", return_value=[encoding]):
                    with patch("face_recognizer.cv2.imwrite"):
                        result = recognizer.register_face("Justin", dummy_frame)
                        assert result is True
                        assert "Justin" in recognizer._known_names


def test_register_face_returns_false_on_no_face() -> None:
    from unittest.mock import patch
    import numpy as np

    recognizer = FaceRecognizer.__new__(FaceRecognizer)
    recognizer._known_names = []
    recognizer._known_encodings = []

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("face_recognizer.face_recognition.face_encodings", return_value=[]):
        result = recognizer.register_face("Justin", dummy_frame)
        assert result is False


def test_encoding_roundtrip() -> None:
    import tempfile, shutil
    from unittest.mock import patch
    tmp = tempfile.mkdtemp()
    try:
        tmp_path = Path(tmp)
        tmp_faces_file = tmp_path / "known_faces.json"
        original = [float(i) * 0.01 for i in range(128)]
        data = {"people": [{"name": "Alice", "encodings": [original]}]}
        with open(tmp_faces_file, "w") as f:
            json.dump(data, f)
        with patch("face_recognizer.KNOWN_FACES_FILE", tmp_faces_file), \
             patch("face_recognizer.FACES_DIR", tmp_path):
            fr = FaceRecognizer()
        assert fr._known_names == ["Alice"]
        assert len(fr._known_encodings[0]) == 128
        assert np.allclose(fr._known_encodings[0], original)
    finally:
        shutil.rmtree(tmp)


def test_load_known_faces_handles_corrupt_json() -> None:
    import tempfile, shutil
    from unittest.mock import patch
    tmp = tempfile.mkdtemp()
    try:
        tmp_path = Path(tmp)
        tmp_faces_file = tmp_path / "known_faces.json"
        tmp_faces_file.write_text("not valid json {{{")
        with patch("face_recognizer.KNOWN_FACES_FILE", tmp_faces_file), \
             patch("face_recognizer.FACES_DIR", tmp_path):
            fr = FaceRecognizer()
        assert fr._known_names == []
        assert fr._known_encodings == []
    finally:
        shutil.rmtree(tmp)


def test_register_face_rejects_path_traversal() -> None:
    from unittest.mock import MagicMock, patch
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        tmp_path = Path(tmp)
        with patch("face_recognizer.FACES_DIR", tmp_path), \
             patch("face_recognizer.FACES_IMAGES_DIR", tmp_path / "images"), \
             patch("face_recognizer.KNOWN_FACES_FILE", tmp_path / "known_faces.json"):
            fr = FaceRecognizer.__new__(FaceRecognizer)
            fr._known_names = []
            fr._known_encodings = []
            result = fr.register_face("../../../etc/evil", MagicMock())
        assert result is False
    finally:
        shutil.rmtree(tmp)
