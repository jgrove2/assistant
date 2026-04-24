import enum
import json
import logging
import threading
from typing import Optional

from src.audio_client import AudioClient
from src.camera_client import CameraClient
from src.config import get_int_env
from src.llm_client import LLMClient, ToolCall
from src.mqtt_client import MQTTClient
from src.recorder import ListenTimeout
from src.stt_client import STTClient, STTError
from src.tools import TOOL_SCHEMAS, dispatch, get_filler, ALLOWED_TOOL_NAMES
from src.tts_client import TTSClient

logger = logging.getLogger(__name__)

FOLLOWUP_TIMEOUT = get_int_env("FOLLOWUP_TIMEOUT", 5)


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceBot:
    def __init__(
        self,
        mqtt_client: MQTTClient,
        audio_client: AudioClient,
        stt_client: STTClient,
        camera_client: CameraClient,
        llm_client: LLMClient,
        tts_client: TTSClient,
    ) -> None:
        self._mqtt_client = mqtt_client
        self._audio_client = audio_client
        self._stt_client = stt_client
        self._camera_client = camera_client
        self._llm_client = llm_client
        self._tts_client = tts_client
        self._state = State.IDLE

    def _transition(self, new_state: State) -> None:
        logger.info("State: %s -> %s", self._state.value, new_state.value)
        self._state = new_state
        self._mqtt_client.publish_state(new_state.value)

    def _handle_camera(self) -> str:
        logger.debug("_handle_camera called")
        names, context = self._camera_client.identify()
        logger.info("Visible people: %s", ", ".join(str(n) for n in names) if names else "none")
        for i, name in enumerate(names):
            if name is None:
                self._tts_client.speak("I don't recognise everyone. What is your name?")
                try:
                    audio = self._audio_client.record(timeout=FOLLOWUP_TIMEOUT)
                except ListenTimeout:
                    logger.info("Registration timed out for unknown face %d", i)
                    continue
                transcribed_name = self._stt_client.transcribe(audio).strip()
                if len(transcribed_name) > 50:
                    transcribed_name = ""
                if transcribed_name:
                    ok = self._camera_client.register(transcribed_name)
                    if ok:
                        names[i] = transcribed_name
        known = [n for n in names if n is not None]
        context = ("People visible: " + ", ".join(known)) if known else ""
        logger.debug("Face context: %r", context)
        return context

    def _run_once(self) -> None:
        self._transition(State.IDLE)
        self._mqtt_client.wait_for_wake()

        listen_timeout: Optional[int] = None
        while True:
            self._transition(State.LISTENING)
            try:
                audio = self._audio_client.record(timeout=listen_timeout)
            except ListenTimeout:
                return

            self._transition(State.THINKING)
            try:
                text = self._stt_client.transcribe(audio)
            except STTError as e:
                logger.error("STT error: %s", e)
                return
            if not text:
                return

            face_context = self._handle_camera()
            logger.debug("LLM system prompt context: %r", face_context)

            messages = self._llm_client.build_messages(text, context=face_context)
            bot_text = ""
            MAX_TOOL_ITERATIONS = 3

            for _ in range(MAX_TOOL_ITERATIONS):
                result = self._llm_client.stream(messages, tools=TOOL_SCHEMAS)

                if isinstance(result, ToolCall):
                    if result.name not in ALLOWED_TOOL_NAMES:
                        logger.warning("Rejected unknown tool call: %s", result.name)
                        break
                    filler = get_filler(result.name)
                    self._tts_client.speak(filler)
                    try:
                        tool_output = dispatch(result.name, result.arguments)
                    except Exception as e:
                        logger.error("Tool dispatch failed: %s", e)
                        tool_output = "The tool encountered an error."
                    messages.append({"role": "assistant", "content": None, "tool_calls": [
                        {
                            "id": result.id,
                            "type": "function",
                            "function": {"name": result.name, "arguments": json.dumps(result.arguments)},
                        }
                    ]})
                    messages.append({"role": "tool", "content": tool_output, "tool_call_id": result.id})
                    continue

                self._transition(State.SPEAKING)
                barge_in_event = threading.Event()
                self._mqtt_client.set_barge_in_event(barge_in_event)
                self._audio_client.start_barge_in()
                try:
                    bot_text = self._tts_client.speak_stream(result, stop_event=barge_in_event)
                finally:
                    self._audio_client.stop_barge_in()
                    barge_in_event.set()
                    self._mqtt_client.set_barge_in_event(None)
                break
            else:
                logger.warning("Max tool iterations reached without text response")
                self._tts_client.speak("I wasn't able to find a good answer for that.")

            if bot_text:
                self._mqtt_client.publish_conversation(text, bot_text)

            listen_timeout = FOLLOWUP_TIMEOUT

    def run(self) -> None:
        try:
            while True:
                self._run_once()
        except KeyboardInterrupt:
            pass


def _make_bot(**kwargs) -> "VoiceBot":
    from unittest.mock import MagicMock
    defaults = dict(
        mqtt_client=MagicMock(),
        audio_client=MagicMock(),
        stt_client=MagicMock(),
        camera_client=MagicMock(),
        llm_client=MagicMock(),
        tts_client=MagicMock(),
    )
    defaults.update(kwargs)
    if "camera_client" not in kwargs:
        defaults["camera_client"].identify.return_value = ([], "")
    if "llm_client" not in kwargs:
        defaults["llm_client"].build_messages.return_value = [{"role": "user", "content": "hello"}]
    return VoiceBot(**defaults)


def test_full_state_transition() -> None:
    from unittest.mock import MagicMock

    mqtt_client = MagicMock()
    audio_client = MagicMock()
    audio_client.record.side_effect = [b"audio", ListenTimeout("timeout")]
    stt_client = MagicMock()
    stt_client.transcribe.return_value = "hello"
    camera_client = MagicMock()
    camera_client.identify.return_value = ([], "")
    llm = MagicMock()
    llm.stream.return_value = iter(["Hi there."])
    tts = MagicMock()
    tts.speak_stream.return_value = "Hi there."

    bot = VoiceBot(
        mqtt_client=mqtt_client,
        audio_client=audio_client,
        stt_client=stt_client,
        camera_client=camera_client,
        llm_client=llm,
        tts_client=tts,
    )

    assert bot._state == State.IDLE
    mqtt_client.wait_for_wake.side_effect = lambda: None
    original_transition = bot._transition
    states: list[State] = []

    def tracking_transition(s: State) -> None:
        original_transition(s)
        states.append(s)

    bot._transition = tracking_transition
    bot._run_once()

    assert states == [State.IDLE, State.LISTENING, State.THINKING, State.SPEAKING, State.LISTENING]
    assert bot._state == State.LISTENING
    mqtt_client.wait_for_wake.assert_called_once()
    audio_client.record.assert_called()
    stt_client.transcribe.assert_called_once_with(b"audio")
    llm.stream.assert_called_once()
    tts.speak_stream.assert_called_once()
    mqtt_client.publish_conversation.assert_called_once()
    assert mqtt_client.publish_conversation.call_args[0][0] == "hello"
    call_args = mqtt_client.publish_conversation.call_args[0]
    assert call_args[0] == "hello"
    assert call_args[1] == "Hi there."


def test_listen_timeout_returns_to_idle() -> None:
    from unittest.mock import MagicMock

    mqtt_client = MagicMock()
    audio_client = MagicMock()
    audio_client.record.side_effect = ListenTimeout("timeout")
    stt_client = MagicMock()
    camera_client = MagicMock()
    camera_client.identify.return_value = ([], "")
    llm = MagicMock()
    tts = MagicMock()

    bot = VoiceBot(
        mqtt_client=mqtt_client,
        audio_client=audio_client,
        stt_client=stt_client,
        camera_client=camera_client,
        llm_client=llm,
        tts_client=tts,
    )
    bot._run_once()

    assert bot._state == State.LISTENING
    stt_client.transcribe.assert_not_called()
    llm.stream.assert_not_called()
    tts.speak_stream.assert_not_called()


def test_empty_transcription_skips_llm() -> None:
    from unittest.mock import MagicMock

    mqtt_client = MagicMock()
    audio_client = MagicMock()
    audio_client.record.return_value = b"audio"
    stt_client = MagicMock()
    stt_client.transcribe.return_value = ""
    camera_client = MagicMock()
    camera_client.identify.return_value = ([], "")
    llm = MagicMock()
    tts = MagicMock()

    bot = VoiceBot(
        mqtt_client=mqtt_client,
        audio_client=audio_client,
        stt_client=stt_client,
        camera_client=camera_client,
        llm_client=llm,
        tts_client=tts,
    )
    bot._run_once()

    llm.stream.assert_not_called()
    tts.speak_stream.assert_not_called()


def test_run_stops_on_keyboard_interrupt() -> None:
    from unittest.mock import MagicMock

    mqtt_client = MagicMock()
    mqtt_client.wait_for_wake.side_effect = KeyboardInterrupt

    bot = VoiceBot(
        mqtt_client=mqtt_client,
        audio_client=MagicMock(),
        stt_client=MagicMock(),
        camera_client=MagicMock(),
        llm_client=MagicMock(),
        tts_client=MagicMock(),
    )
    bot.run()


def test_state_publishes_to_mqtt() -> None:
    from unittest.mock import MagicMock, call

    mqtt_client = MagicMock()
    audio_client = MagicMock()
    audio_client.record.side_effect = ListenTimeout("timeout")
    stt_client = MagicMock()
    camera_client = MagicMock()
    camera_client.identify.return_value = ([], "")
    llm = MagicMock()
    tts = MagicMock()

    bot = VoiceBot(
        mqtt_client=mqtt_client,
        audio_client=audio_client,
        stt_client=stt_client,
        camera_client=camera_client,
        llm_client=llm,
        tts_client=tts,
    )
    bot._run_once()

    publish_calls = [c.args[0] for c in mqtt_client.publish_state.call_args_list]
    assert "idle" in publish_calls
    assert "listening" in publish_calls


def test_handle_camera_registers_unknown_face() -> None:
    from unittest.mock import MagicMock

    camera_client = MagicMock()
    camera_client.identify.return_value = ([None], "")
    audio_client = MagicMock()
    audio_client.record.return_value = b"audio"
    stt_client = MagicMock()
    stt_client.transcribe.return_value = "Alice"
    camera_client.register.return_value = True

    bot = _make_bot(
        camera_client=camera_client,
        audio_client=audio_client,
        stt_client=stt_client,
    )
    context = bot._handle_camera()

    camera_client.register.assert_called_with("Alice")
    assert "Alice" in context


def test_speak_stream_exception_still_cleans_up() -> None:
    import pytest
    from unittest.mock import MagicMock, call

    bot = _make_bot()
    bot._audio_client.record.return_value = b"audio"
    bot._stt_client.transcribe.return_value = "hello"
    bot._llm_client.stream.return_value = iter(["Hi."])
    bot._tts_client.speak_stream.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        bot._run_once()

    bot._audio_client.stop_barge_in.assert_called()
    assert any(c == call(None) for c in bot._mqtt_client.set_barge_in_event.call_args_list)


def test_followup_timeout_passed_to_record() -> None:
    from unittest.mock import call

    bot = _make_bot()
    bot._audio_client.record.side_effect = [b"audio", ListenTimeout("timeout")]
    bot._stt_client.transcribe.return_value = "hello"
    bot._llm_client.stream.return_value = iter(["Hi."])

    bot._run_once()

    assert bot._audio_client.record.call_args_list[1] == call(timeout=FOLLOWUP_TIMEOUT)


def test_tool_call_loop_exhaustion_speaks_fallback() -> None:
    from unittest.mock import patch

    bot = _make_bot()
    bot._audio_client.record.side_effect = [b"audio", ListenTimeout("timeout")]
    bot._stt_client.transcribe.return_value = "search forever"
    bot._llm_client.build_messages.return_value = [{"role": "user", "content": "search forever"}]
    bot._llm_client.stream.return_value = ToolCall(id="c1", name="web_search", arguments={"query": "x"})

    with patch("src.state_machine.dispatch", return_value="result"), \
         patch("src.state_machine.get_filler", return_value="thinking"):
        bot._run_once()

    assert any("wasn't able" in str(c) for c in bot._tts_client.speak.call_args_list)
    bot._tts_client.speak_stream.assert_not_called()


def test_tool_call_message_construction() -> None:
    from unittest.mock import patch

    bot = _make_bot()
    bot._audio_client.record.side_effect = [b"audio", ListenTimeout("timeout")]
    bot._stt_client.transcribe.return_value = "what is the capital of France"
    bot._llm_client.build_messages.return_value = [{"role": "user", "content": "what is the capital of France"}]
    bot._tts_client.speak_stream.return_value = "Paris is the capital."

    captured_messages: list[list[dict]] = []

    def stream_side_effect(messages: list, tools: list | None = None) -> object:
        captured_messages.append(list(messages))
        if len(captured_messages) == 1:
            return ToolCall(id="call_1", name="web_search", arguments={"query": "Paris"})
        return iter(["Paris is the capital."])

    bot._llm_client.stream.side_effect = stream_side_effect

    with patch("src.state_machine.dispatch", return_value="Paris is the capital of France"), \
         patch("src.state_machine.get_filler", return_value="Let me look that up."):
        bot._run_once()

    second_call_messages = captured_messages[1]
    assistant_msg = second_call_messages[-2]
    tool_msg = second_call_messages[-1]

    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "web_search"

    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "Paris is the capital of France"


def test_tool_call_speaks_filler_and_continues() -> None:
    from unittest.mock import MagicMock, patch

    bot = _make_bot()
    bot._audio_client.record.side_effect = [b"audio", ListenTimeout("timeout")]
    bot._stt_client.transcribe.return_value = "what time is it"
    bot._llm_client.build_messages.return_value = [{"role": "user", "content": "what time is it"}]
    bot._llm_client.stream.side_effect = [
        ToolCall(id="c1", name="web_search", arguments={"query": "test"}),
        iter(["The answer is 42."]),
    ]
    bot._tts_client.speak_stream.return_value = "The answer is 42."

    with patch("src.state_machine.dispatch", return_value="42 is the answer") as mock_dispatch, \
         patch("src.state_machine.get_filler", return_value="Let me look that up.") as mock_filler:
        bot._run_once()

    bot._tts_client.speak.assert_called_with("Let me look that up.")
    bot._tts_client.speak_stream.assert_called_once()
    bot._mqtt_client.publish_conversation.assert_called()


def test_tool_call_max_iterations_does_not_hang() -> None:
    from unittest.mock import patch

    bot = _make_bot()
    bot._audio_client.record.side_effect = [b"audio", ListenTimeout("timeout")]
    bot._stt_client.transcribe.return_value = "search forever"
    bot._llm_client.build_messages.return_value = [{"role": "user", "content": "search forever"}]
    bot._llm_client.stream.return_value = ToolCall(id="c1", name="web_search", arguments={"query": "loop"})

    with patch("src.state_machine.dispatch", return_value="result"), \
         patch("src.state_machine.get_filler", return_value="thinking"):
        bot._run_once()

    bot._tts_client.speak_stream.assert_not_called()
