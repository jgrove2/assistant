import json
import logging
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Generator
from typing import Any

import docker
import paho.mqtt.client as mqtt
from flask import Flask, Response, render_template

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

MQTT_HOST = os.environ.get("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
DB_PATH = os.environ.get("DB_PATH", "/app/data/bot.db")
FACES_DIR = os.environ.get("FACES_DIR", "/app/faces/images")
STATS_INTERVAL = 5

app = Flask(__name__)

_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                topic TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                user_text TEXT NOT NULL,
                bot_text TEXT NOT NULL
            )
        """)
        conn.commit()


def _broadcast(data: dict[str, Any]) -> None:
    with _sse_lock:
        dead: list[queue.Queue] = []
        for q in _sse_clients:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


def _on_message(client: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage) -> None:
    topic = msg.topic
    payload = msg.payload.decode("utf-8", errors="replace")
    logger.debug("MQTT %s: %s", topic, payload)
    ts = time.time()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO events (ts, topic, payload) VALUES (?, ?, ?)", (ts, topic, payload))
        if topic == "bot/conversation":
            try:
                data = json.loads(payload)
                if isinstance(data, dict):
                    user_text = str(data.get("user", ""))
                    bot_text = str(data.get("bot", ""))
                    conn.execute(
                        "INSERT INTO conversations (ts, user_text, bot_text) VALUES (?, ?, ?)",
                        (ts, user_text, bot_text),
                    )
                else:
                    logger.warning("Unexpected conversation payload shape: %s", payload)
            except json.JSONDecodeError:
                logger.warning("Failed to parse conversation payload: %s", payload)
        conn.commit()
    _broadcast({"topic": topic, "payload": payload, "ts": ts})


def start_mqtt() -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_message = _on_message
    try:
        client.connect(MQTT_HOST, MQTT_PORT)
    except Exception as e:
        logger.error("Failed to connect to MQTT broker at %s:%d: %s", MQTT_HOST, MQTT_PORT, e)
        raise
    client.subscribe("bot/state")
    client.subscribe("bot/wake")
    client.subscribe("bot/conversation")
    client.loop_start()
    logger.info("MQTT subscribed to bot/state, bot/wake, bot/conversation")


def _get_container_stats() -> list[dict[str, Any]]:
    try:
        with docker.from_env() as docker_client:
            containers = docker_client.containers.list()
            results: list[dict[str, Any]] = []
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
                    system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
                    num_cpus = stats["cpu_stats"].get("online_cpus") or len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1]))
                    cpu_pct = (cpu_delta / system_delta) * num_cpus * 100.0 if system_delta > 0 else 0.0
                    mem_usage = stats["memory_stats"].get("usage", 0)
                    mem_limit = stats["memory_stats"].get("limit", 1)
                    mem_pct = (mem_usage / mem_limit) * 100.0
                    results.append({
                        "name": container.name,
                        "cpu_pct": round(cpu_pct, 1),
                        "mem_mb": round(mem_usage / 1024 / 1024, 1),
                        "mem_limit_mb": round(mem_limit / 1024 / 1024, 1),
                        "mem_pct": round(mem_pct, 1),
                    })
                except Exception as e:
                    logger.warning("Failed to get stats for %s: %s", container.name, e)
            return results
    except Exception as e:
        logger.warning("Docker stats unavailable: %s", e)
        return []


_container_stats: list[dict[str, Any]] = []
_stats_lock = threading.Lock()


def _stats_poller() -> None:
    while True:
        stats = _get_container_stats()
        with _stats_lock:
            global _container_stats
            _container_stats = stats
        _broadcast({"topic": "stats", "payload": json.dumps(stats), "ts": time.time()})
        time.sleep(STATS_INTERVAL)


def _count_faces() -> int:
    try:
        files = [f for f in os.listdir(FACES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        return len(files)
    except FileNotFoundError:
        return 0


@app.route("/")
def index() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conversations = [
            dict(row)
            for row in conn.execute(
                "SELECT ts, user_text, bot_text FROM conversations ORDER BY ts DESC LIMIT 50"
            ).fetchall()
        ]
        events = [
            dict(row)
            for row in conn.execute(
                "SELECT ts, topic, payload FROM events ORDER BY ts DESC LIMIT 100"
            ).fetchall()
        ]
    with _stats_lock:
        container_stats = list(_container_stats)
    face_count = _count_faces()
    return render_template(
        "index.html",
        conversations=conversations,
        events=events,
        face_count=face_count,
        container_stats=container_stats,
    )


@app.route("/stream")
def stream() -> Response:
    q: queue.Queue = queue.Queue(maxsize=50)
    with _sse_lock:
        _sse_clients.append(q)

    def generate() -> Generator[str, None, None]:
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    init_db()
    threading.Thread(target=start_mqtt, daemon=True).start()
    threading.Thread(target=_stats_poller, daemon=True).start()
    app.run(host="0.0.0.0", port=5004, threaded=True)
