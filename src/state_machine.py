import enum
import logging
import threading

from src.camera import Camera, CameraError
from src.face_recognizer import FaceRecognizer
from src.recorder import AudioRecorder, ListenTimeout, FOLLOWUP_TIMEOUT
from src.transcriber import Transcriber
from src.llm_client import LLMClient
from src.tts_client import TTSClient
from src.wake_word import WakeWordDetector

logger = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class VoiceBot:
    def __init__(
        self,
        wake_word_detector: WakeWordDetector,
        recorder: AudioRecorder,
        transcriber: Transcriber,
        llm_client: LLMClient,
        tts_client: TTSClient,
        camera: Camera | None = None,
        face_recognizer: FaceRecognizer | None = None,
    ) -> None:
        self._wake_word_detector = wake_word_detector
        self._recorder = recorder
        self._transcriber = transcriber
        self._llm_client = llm_client
        self._tts_client = tts_client
        self._camera = camera
        self._face_recognizer = face_recognizer
        self._state = State.IDLE

    def _handle_camera(self) -> str:
        logger.debug("_handle_camera called: camera=%s recognizer=%s", self._camera, self._face_recognizer)
        if self._camera is None or self._face_recognizer is None:
            return ""
        try:
            frame = self._camera.capture_frame()
        except CameraError:
            logger.warning("Failed to capture camera frame")
            return ""
        results = self._face_recognizer.identify_faces(frame)
        for i, name in enumerate(results):
            if name is None:
                self._tts_client.speak("I don't recognise everyone. What is your name?")
                try:
                    audio = self._recorder.record(timeout=FOLLOWUP_TIMEOUT)
                except ListenTimeout:
                    logger.info("Registration timed out for unknown face %d", i)
                    continue
                transcribed_name = self._transcriber.transcribe(audio).strip()
                if len(transcribed_name) > 50:
                    transcribed_name = ""
                if transcribed_name:
                    self._face_recognizer.register_face(transcribed_name, frame)
                    results[i] = transcribed_name
        names = [n for n in results if n is not None]
        if not names:
            return ""
        face_context = "People visible: " + ", ".join(names)
        logger.debug("Face context: %r", face_context)
        logger.info("Face scan: %s", ", ".join(names) if names else "no faces recognised")
        return face_context

    def _transition(self, new_state: State) -> None:
        logger.info("State: %s -> %s", self._state.value, new_state.value)
        self._state = new_state

    def _run_once(self) -> None:
        self._transition(State.IDLE)
        self._wake_word_detector.listen()

        listen_timeout: int | None = None
        while True:
            self._transition(State.LISTENING)
            try:
                audio = self._recorder.record(timeout=listen_timeout)
            except ListenTimeout:
                return

            self._transition(State.PROCESSING)
            text = self._transcriber.transcribe(audio)
            if not text:
                return

            face_context = self._handle_camera()
            logger.info("Visible people: %s", ", ".join(n for n in [face_context.replace("People visible: ", "")] if face_context) or "none")

            token_stream = self._llm_client.stream(text, context=face_context)

            self._transition(State.SPEAKING)
            barge_in_event = threading.Event()
            monitor_thread = threading.Thread(
                target=self._recorder.monitor_for_barge_in,
                args=(barge_in_event,),
                daemon=True,
            )
            monitor_thread.start()
            self._tts_client.speak_stream(token_stream, stop_event=barge_in_event)
            barge_in_event.set()
            monitor_thread.join(timeout=1.0)
            if monitor_thread.is_alive():
                logger.warning("Monitor thread did not exit within timeout")

            listen_timeout = FOLLOWUP_TIMEOUT

    def run(self) -> None:
        try:
            while True:
                self._run_once()
        except KeyboardInterrupt:
            pass


def test_full_state_transition() -> None:
    from unittest.mock import MagicMock

    wake = MagicMock()
    recorder = MagicMock()
    recorder.record.side_effect = [b"audio", ListenTimeout("timeout")]
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "hello"
    llm = MagicMock()
    llm.stream.return_value = iter(["Hi there."])
    tts = MagicMock()

    bot = VoiceBot(wake, recorder, transcriber, llm, tts)

    assert bot._state == State.IDLE
    wake.listen.side_effect = lambda: None
    original_transition = bot._transition
    states: list[State] = []

    def tracking_transition(s: State) -> None:
        original_transition(s)
        states.append(s)

    bot._transition = tracking_transition
    bot._run_once()

    assert states == [State.IDLE, State.LISTENING, State.PROCESSING, State.SPEAKING, State.LISTENING]
    assert bot._state == State.LISTENING
    wake.listen.assert_called_once()
    recorder.record.assert_called()
    transcriber.transcribe.assert_called_once_with(b"audio")
    llm.stream.assert_called_once_with("hello", context="")
    tts.speak_stream.assert_called_once()


def test_listen_timeout_returns_to_idle() -> None:
    from unittest.mock import MagicMock

    wake = MagicMock()
    recorder = MagicMock()
    recorder.record.side_effect = ListenTimeout("timeout")
    transcriber = MagicMock()
    llm = MagicMock()
    tts = MagicMock()

    bot = VoiceBot(wake, recorder, transcriber, llm, tts)
    bot._run_once()

    assert bot._state == State.LISTENING
    transcriber.transcribe.assert_not_called()
    llm.stream.assert_not_called()
    tts.speak_stream.assert_not_called()


def test_empty_transcription_skips_llm() -> None:
    from unittest.mock import MagicMock

    wake = MagicMock()
    recorder = MagicMock()
    recorder.record.return_value = b"audio"
    transcriber = MagicMock()
    transcriber.transcribe.return_value = ""
    llm = MagicMock()
    tts = MagicMock()

    bot = VoiceBot(wake, recorder, transcriber, llm, tts)
    bot._run_once()

    llm.stream.assert_not_called()
    tts.speak_stream.assert_not_called()


def test_run_stops_on_keyboard_interrupt() -> None:
    from unittest.mock import MagicMock

    wake = MagicMock()
    wake.listen.side_effect = KeyboardInterrupt

    bot = VoiceBot(wake, MagicMock(), MagicMock(), MagicMock(), MagicMock())
    bot.run()


def test_camera_context_prepended_to_prompt() -> None:
    import numpy as np
    from unittest.mock import MagicMock

    wake = MagicMock()
    recorder = MagicMock()
    recorder.record.side_effect = [b"audio", ListenTimeout("timeout")]
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "hello"
    llm = MagicMock()
    llm.stream.return_value = iter(["Hi there."])
    tts = MagicMock()

    camera = MagicMock()
    camera.capture_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    face_recognizer = MagicMock()
    face_recognizer.identify_faces.return_value = ["Justin"]

    bot = VoiceBot(wake, recorder, transcriber, llm, tts, camera=camera, face_recognizer=face_recognizer)
    bot._run_once()

    call_kwargs = llm.stream.call_args
    assert "Justin" in call_kwargs.kwargs.get("context", "")


def test_handle_camera_returns_empty_when_camera_none() -> None:
    from unittest.mock import MagicMock
    bot = VoiceBot(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                   camera=None, face_recognizer=None)
    result = bot._handle_camera()
    assert result == ""


def test_handle_camera_returns_empty_on_camera_error() -> None:
    from unittest.mock import MagicMock
    from src.camera import CameraError
    camera = MagicMock()
    camera.capture_frame.side_effect = CameraError("no device")
    bot = VoiceBot(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                   camera=camera, face_recognizer=MagicMock())
    result = bot._handle_camera()
    assert result == ""


def test_handle_camera_skips_on_registration_timeout() -> None:
    from unittest.mock import MagicMock
    camera = MagicMock()
    face_recognizer = MagicMock()
    face_recognizer.identify_faces.return_value = [None]
    recorder = MagicMock()
    recorder.record.side_effect = ListenTimeout("timeout")
    bot = VoiceBot(MagicMock(), recorder, MagicMock(), MagicMock(), MagicMock(),
                   camera=camera, face_recognizer=face_recognizer)
    result = bot._handle_camera()
    face_recognizer.register_face.assert_not_called()
    assert result == ""


def test_handle_camera_registers_multiple_unknown_faces() -> None:
    from unittest.mock import MagicMock, call
    camera = MagicMock()
    frame = MagicMock()
    camera.capture_frame.return_value = frame
    face_recognizer = MagicMock()
    face_recognizer.identify_faces.return_value = [None, None]
    face_recognizer.register_face.return_value = True
    recorder = MagicMock()
    recorder.record.side_effect = [b"audio1", b"audio2"]
    transcriber = MagicMock()
    transcriber.transcribe.side_effect = ["Alice", "Bob"]
    tts = MagicMock()
    bot = VoiceBot(MagicMock(), recorder, transcriber, MagicMock(), tts,
                   camera=camera, face_recognizer=face_recognizer)
    result = bot._handle_camera()
    assert face_recognizer.register_face.call_count == 2
    assert "Alice" in result
    assert "Bob" in result
