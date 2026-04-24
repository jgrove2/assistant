import logging
import os

from src.camera import Camera, CameraError
from src.config import get_int_env
from src.face_recognizer import FaceRecognizer
from src.wake_word import WakeWordDetector
from src.recorder import AudioRecorder
from src.transcriber import Transcriber
from src.llm_client import LLMClient
from src.tts_client import TTSClient
from src.state_machine import VoiceBot

logger = logging.getLogger(__name__)

WAKE_WORD_MODEL = os.environ.get("WAKE_WORD_MODEL", "alexa")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
PIPER_URL = os.environ.get("PIPER_URL", "http://piper:5000")
AUDIO_SAMPLE_RATE = get_int_env("AUDIO_SAMPLE_RATE", 16000)
AUDIO_CHANNELS = get_int_env("AUDIO_CHANNELS", 1)
SPEECH_TIMEOUT = get_int_env("SPEECH_TIMEOUT", 10)


def main() -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    logger.info("Starting voice bot")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable must be set")

    wake_word_detector = WakeWordDetector(model_name=WAKE_WORD_MODEL)
    recorder = AudioRecorder(
        sample_rate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        timeout=SPEECH_TIMEOUT,
    )
    transcriber = Transcriber(model_name=WHISPER_MODEL)
    llm_client = LLMClient(api_key=OPENROUTER_API_KEY, model=OPENROUTER_MODEL)
    tts_client = TTSClient(piper_url=PIPER_URL)

    camera: Camera | None = None
    try:
        camera = Camera()
    except CameraError:
        logger.warning("No camera found, face recognition disabled")

    face_recognizer = FaceRecognizer() if camera is not None else None

    bot = VoiceBot(
        wake_word_detector=wake_word_detector,
        recorder=recorder,
        transcriber=transcriber,
        llm_client=llm_client,
        tts_client=tts_client,
        camera=camera,
        face_recognizer=face_recognizer,
    )
    bot.run()


if __name__ == "__main__":
    main()
