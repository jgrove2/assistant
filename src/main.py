import logging
import os

from src.audio_client import AudioClient
from src.camera_client import CameraClient
from src.config import get_int_env
from src.llm_client import LLMClient
from src.mqtt_client import MQTTClient
from src.stt_client import STTClient
from src.state_machine import VoiceBot
from src.tts_client import TTSClient

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
PIPER_URL = os.environ.get("PIPER_URL", "http://piper:5000")
MQTT_HOST = os.environ.get("MQTT_HOST", "mosquitto")
MQTT_PORT = get_int_env("MQTT_PORT", 1883)
AUDIO_URL = os.environ.get("AUDIO_URL", "http://audio:5001")
STT_URL = os.environ.get("STT_URL", "http://stt:5002")
CAMERA_URL = os.environ.get("CAMERA_URL", "http://camera:5003")


def main() -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    logger.info("Starting orchestrator")

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable must be set")

    mqtt_client = MQTTClient(host=MQTT_HOST, port=MQTT_PORT)
    audio_client = AudioClient(url=AUDIO_URL)
    stt_client = STTClient(url=STT_URL)
    camera_client = CameraClient(url=CAMERA_URL)
    llm_client = LLMClient(api_key=OPENROUTER_API_KEY, model=OPENROUTER_MODEL)
    tts_client = TTSClient(piper_url=PIPER_URL)

    bot = VoiceBot(
        mqtt_client=mqtt_client,
        audio_client=audio_client,
        stt_client=stt_client,
        camera_client=camera_client,
        llm_client=llm_client,
        tts_client=tts_client,
    )
    bot.run()


if __name__ == "__main__":
    main()
