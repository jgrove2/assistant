import logging
import os
import threading
import time

import paho.mqtt.client as mqtt
from flask import Flask, Response, jsonify, request

from config import get_int_env
from recorder import AudioRecorder, ListenTimeout
from wake_word import WakeWordDetector

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

MQTT_HOST = os.environ.get("MQTT_HOST", "mosquitto")
MQTT_PORT = get_int_env("MQTT_PORT", 1883)
WAKE_WORD_MODEL = os.environ.get("WAKE_WORD_MODEL", "alexa")
AUDIO_SAMPLE_RATE = get_int_env("AUDIO_SAMPLE_RATE", 16000)
AUDIO_CHANNELS = get_int_env("AUDIO_CHANNELS", 1)
SPEECH_TIMEOUT = get_int_env("SPEECH_TIMEOUT", 10)

app = Flask(__name__)

_recorder = AudioRecorder(
    sample_rate=AUDIO_SAMPLE_RATE,
    channels=AUDIO_CHANNELS,
    timeout=SPEECH_TIMEOUT,
)

_mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
_mqtt.connect(MQTT_HOST, MQTT_PORT)
_mqtt.loop_start()

_recording_active = threading.Event()

_barge_in_stop = threading.Event()


def _wake_word_loop() -> None:
    detector = WakeWordDetector(model_name=WAKE_WORD_MODEL)
    while True:
        if _recording_active.is_set():
            time.sleep(0.05)
            continue
        try:
            detector.listen()
            logger.info("Wake word detected, publishing bot/wake")
            _recording_active.set()
            try:
                _mqtt.publish("bot/wake", "")
            finally:
                _recording_active.clear()
        except Exception as e:
            logger.error("Wake word loop error: %s", e)


@app.route("/record", methods=["POST"])
def record() -> tuple[bytes, int, dict] | tuple[Response, int]:
    data = request.get_json(silent=True) or {}
    timeout = data.get("timeout")
    if timeout is not None and (not isinstance(timeout, int) or not (1 <= timeout <= 300)):
        return jsonify({"error": "invalid timeout"}), 400
    _recording_active.set()
    try:
        audio = _recorder.record(timeout=timeout)
        return audio, 200, {"Content-Type": "application/octet-stream"}
    except ListenTimeout:
        return jsonify({"error": "timeout"}), 408
    finally:
        _recording_active.clear()


def _run_barge_in_monitor() -> None:
    _barge_in_stop.clear()
    triggered = threading.Event()

    def monitor() -> None:
        _recorder.monitor_for_barge_in(triggered)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()

    while not triggered.is_set() and not _barge_in_stop.is_set():
        time.sleep(0.05)

    if triggered.is_set() and not _barge_in_stop.is_set():
        logger.info("Barge-in detected, publishing bot/barge_in")
        _mqtt.publish("bot/barge_in", "")

    triggered.set()
    t.join(timeout=1.0)


@app.route("/barge_in/start", methods=["POST"])
def barge_in_start() -> tuple[Response, int]:
    threading.Thread(target=_run_barge_in_monitor, daemon=True).start()
    return jsonify({"ok": True}), 200


@app.route("/barge_in/stop", methods=["POST"])
def barge_in_stop() -> tuple[Response, int]:
    _barge_in_stop.set()
    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    threading.Thread(target=_wake_word_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5001)
