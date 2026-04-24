import json
import os
import subprocess
import tempfile
from flask import Flask, request, Response

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

PIPER_BINARY = "/app/piper/piper"
VOICE_MODEL = "/app/voice.onnx"
MAX_TEXT_LENGTH = 10000


@app.route("/", methods=["POST"])
def synthesize() -> Response:
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return Response(json.dumps({"error": "text is required"}), status=400, mimetype="application/json")
    if len(text) > MAX_TEXT_LENGTH:
        return Response(json.dumps({"error": "text too long"}), status=400, mimetype="application/json")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)

    try:
        try:
            subprocess.run(
                [PIPER_BINARY, "--model", VOICE_MODEL, "--output_file", tmp_path],
                input=text.encode(),
                check=True,
                capture_output=True,
                timeout=10,
            )
        except subprocess.CalledProcessError:
            return Response(json.dumps({"error": "synthesis failed"}), status=500, mimetype="application/json")
        except subprocess.TimeoutExpired:
            return Response(json.dumps({"error": "synthesis timed out"}), status=504, mimetype="application/json")

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    return Response(wav_bytes, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
