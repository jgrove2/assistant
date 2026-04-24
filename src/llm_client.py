import json
import logging
import os
from collections.abc import Iterator

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a friendly voice assistant. Keep responses concise and conversational, "
    "using at most 1 to 3 sentences. Never use lists, markdown, or headers. "
    "When you know the names of the people you are talking to, address them by name "
    "naturally in conversation, the way a person would."
)


class LLMClient:
    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model: str = OPENROUTER_MODEL,
    ) -> None:
        self._api_key = api_key
        self._model = model

    def stream(self, prompt: str, context: str = "") -> Iterator[str]:
        logger.debug("LLM system prompt: %s", SYSTEM_PROMPT + ("\n\n" + context if context else ""))
        logger.info("LLM user prompt: %s", prompt)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + ("\n\n" + context if context else "")},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", OPENROUTER_URL, headers=headers, json=payload) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    logger.error("HTTP error from LLM: %s", e)
                    raise
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                        content = parsed["choices"][0]["delta"].get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


def test_stream_yields_chunks() -> None:
    from unittest.mock import MagicMock, patch

    sse_lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": " world"}}]}',
        "data: [DONE]",
    ]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("src.llm_client.httpx.Client", return_value=mock_client):
        client = LLMClient(api_key="test", model="test-model")
        chunks = list(client.stream("Hi"))

    assert chunks == ["Hello", " world"]


def test_stream_handles_malformed_json() -> None:
    from unittest.mock import MagicMock, patch

    sse_lines = [
        "data: not-json",
        'data: {"choices": [{"delta": {"content": "ok"}}]}',
        "data: [DONE]",
    ]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("src.llm_client.httpx.Client", return_value=mock_client):
        client = LLMClient(api_key="test", model="test-model")
        chunks = list(client.stream("Hi"))

    assert chunks == ["ok"]


def test_stream_raises_on_http_error() -> None:
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=MagicMock()
    )
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("src.llm_client.httpx.Client", return_value=mock_client):
        client = LLMClient(api_key="test", model="test-model")
        try:
            list(client.stream("Hi"))
            assert False, "Expected HTTPStatusError"
        except httpx.HTTPStatusError:
            pass


def test_stream_skips_missing_choices_key() -> None:
    from unittest.mock import MagicMock, patch

    sse_lines = [
        'data: {"no_choices": true}',
        'data: {"choices": [{"delta": {"content": "ok"}}]}',
        "data: [DONE]",
    ]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("src.llm_client.httpx.Client", return_value=mock_client):
        client = LLMClient(api_key="test", model="test-model")
        chunks = list(client.stream("Hi"))

    assert chunks == ["ok"]


def test_stream_done_ends_iteration() -> None:
    from unittest.mock import MagicMock, patch

    sse_lines = [
        "data: [DONE]",
        'data: {"choices": [{"delta": {"content": "should not appear"}}]}',
    ]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("src.llm_client.httpx.Client", return_value=mock_client):
        client = LLMClient(api_key="test", model="test-model")
        chunks = list(client.stream("Hi"))

    assert chunks == []
