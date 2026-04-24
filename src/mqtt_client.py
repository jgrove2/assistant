import json
import logging
import threading
from typing import Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTClient:
    def __init__(self, host: str, port: int = 1883) -> None:
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self._client.on_message = self._on_message
        self._client.connect(host, port)
        self._client.subscribe("bot/wake")
        self._client.subscribe("bot/barge_in")
        self._client.loop_start()
        self._wake_event = threading.Event()
        self._barge_in_event: Optional[threading.Event] = None
        self._barge_in_lock = threading.Lock()
        logger.info("MQTT client connected to %s:%d", host, port)

    def _on_message(self, client: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage) -> None:
        logger.debug("MQTT message received: %s", msg.topic)
        if msg.topic == "bot/wake":
            self._wake_event.set()
        elif msg.topic == "bot/barge_in":
            with self._barge_in_lock:
                if self._barge_in_event is not None:
                    self._barge_in_event.set()

    def wait_for_wake(self) -> None:
        logger.info("Waiting for wake word...")
        self._wake_event.clear()
        self._wake_event.wait()

    def publish_state(self, state: str) -> None:
        self._client.publish("bot/state", state)
        logger.debug("Published bot/state: %s", state)

    def set_barge_in_event(self, event: Optional[threading.Event]) -> None:
        with self._barge_in_lock:
            self._barge_in_event = event

    def publish_conversation(self, user_text: str, bot_text: str) -> None:
        try:
            payload = json.dumps({"user": user_text, "bot": bot_text})
            self._client.publish("bot/conversation", payload)
            logger.debug("Published bot/conversation")
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize conversation: %s", e)


def _make_mqtt_client() -> MQTTClient:
    from unittest.mock import MagicMock, patch

    with patch("src.mqtt_client.mqtt.Client", return_value=MagicMock()) as mock_mqtt:
        mock_mqtt.return_value.CallbackAPIVersion = MagicMock()
        client = MQTTClient("localhost", 1883)
    return client


def test_on_message_wake_sets_event() -> None:
    from unittest.mock import MagicMock, patch

    with patch("src.mqtt_client.mqtt.Client", return_value=MagicMock()):
        client = MQTTClient("localhost", 1883)

    msg = MagicMock()
    msg.topic = "bot/wake"
    client._on_message(MagicMock(), None, msg)
    assert client._wake_event.is_set()


def test_on_message_barge_in_sets_event() -> None:
    from unittest.mock import MagicMock, patch

    with patch("src.mqtt_client.mqtt.Client", return_value=MagicMock()):
        client = MQTTClient("localhost", 1883)

    barge_in_event = threading.Event()
    client.set_barge_in_event(barge_in_event)

    msg = MagicMock()
    msg.topic = "bot/barge_in"
    client._on_message(MagicMock(), None, msg)
    assert barge_in_event.is_set()


def test_on_message_barge_in_no_event_no_error() -> None:
    from unittest.mock import MagicMock, patch

    with patch("src.mqtt_client.mqtt.Client", return_value=MagicMock()):
        client = MQTTClient("localhost", 1883)

    msg = MagicMock()
    msg.topic = "bot/barge_in"
    client._on_message(MagicMock(), None, msg)


def test_publish_state_called() -> None:
    from unittest.mock import MagicMock, patch

    mock_mqtt_instance = MagicMock()
    with patch("src.mqtt_client.mqtt.Client", return_value=mock_mqtt_instance):
        client = MQTTClient("localhost", 1883)

    client.publish_state("idle")
    mock_mqtt_instance.publish.assert_called_with("bot/state", "idle")


def test_set_barge_in_event_thread_safety() -> None:
    from unittest.mock import MagicMock, patch

    with patch("src.mqtt_client.mqtt.Client", return_value=MagicMock()):
        client = MQTTClient("localhost", 1883)

    errors: list[Exception] = []

    def setter() -> None:
        for _ in range(100):
            try:
                client.set_barge_in_event(threading.Event())
                client.set_barge_in_event(None)
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=setter) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


def test_init_connects_and_subscribes() -> None:
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock()
    with patch("src.mqtt_client.mqtt.Client", return_value=mock_client):
        MQTTClient("localhost", 1883)

    mock_client.connect.assert_called_once_with("localhost", 1883)
    mock_client.subscribe.assert_any_call("bot/wake")
    mock_client.subscribe.assert_any_call("bot/barge_in")
    mock_client.loop_start.assert_called_once()


def test_publish_conversation_called() -> None:
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock()
    with patch("src.mqtt_client.mqtt.Client", return_value=mock_client):
        client = MQTTClient("localhost", 1883)

    client.publish_conversation("hello", "hi there")
    raw_payload = mock_client.publish.call_args[0][1]
    assert json.loads(raw_payload) == {"user": "hello", "bot": "hi there"}

