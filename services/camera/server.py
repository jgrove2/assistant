import logging
import os
from typing import Optional

import numpy as np
from flask import Flask, jsonify, request

from camera import Camera, CameraError
from face_recognizer import FaceRecognizer

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

app = Flask(__name__)

_camera: Optional[Camera] = None
_face_recognizer: Optional[FaceRecognizer] = None
_last_frame: Optional[np.ndarray] = None


def _init() -> None:
    global _camera, _face_recognizer
    try:
        _camera = Camera()
        _face_recognizer = FaceRecognizer()
        logger.info("Camera and face recognizer initialised")
    except CameraError as e:
        logger.warning("Camera not available: %s", e)


@app.route("/identify", methods=["GET"])
def identify():
    global _last_frame
    if _camera is None or _face_recognizer is None:
        return jsonify({"names": [], "context": ""})
    try:
        frame = _camera.capture_frame()
    except CameraError as e:
        logger.warning("Failed to capture frame: %s", e)
        return jsonify({"names": [], "context": ""})
    _last_frame = frame
    names = _face_recognizer.identify_faces(frame)
    logger.info("Face scan: %s", ", ".join(str(n) for n in names) if names else "no faces")
    known = [n for n in names if n is not None]
    context = ("People visible: " + ", ".join(known)) if known else ""
    return jsonify({"names": names, "context": context})


@app.route("/register", methods=["POST"])
def register():
    global _last_frame
    if _face_recognizer is None:
        return jsonify({"ok": False, "error": "face recognizer not available"}), 503
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "name required"}), 400
    if _last_frame is None:
        return jsonify({"ok": False, "error": "no frame cached, call /identify first"}), 400
    ok = _face_recognizer.register_face(name, _last_frame)
    return jsonify({"ok": ok})


if __name__ == "__main__":
    _init()
    app.run(host="0.0.0.0", port=5003)
