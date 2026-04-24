import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

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


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


class LLMClient:
    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model: str = OPENROUTER_MODEL,
    ) -> None:
        self._api_key = api_key
        self._model = model

    def build_messages(self, prompt: str, context: str = "") -> list[dict[str, Any]]:
        system = SYSTEM_PROMPT + ("\n\n" + context if context else "")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str] | ToolCall:
        logger.info("LLM request with %d messages", len(messages))
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        chunks: list[str] = []
        tool_call_id = ""
        tool_call_name = ""
        tool_call_args_buf = ""

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
                        delta = parsed["choices"][0]["delta"]
                        content = delta.get("content")
                        if content:
                            chunks.append(content)
                            continue
                        tool_calls = delta.get("tool_calls")
                        if tool_calls:
                            tc = tool_calls[0]
                            if tc.get("id"):
                                tool_call_id = tc["id"]
                            if tc.get("function", {}).get("name"):
                                tool_call_name = tc["function"]["name"]
                            tool_call_args_buf += tc.get("function", {}).get("arguments", "")
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.debug("SSE parse error (%s): %s", type(e).__name__, line)
                        continue

        if tool_call_name:
            try:
                args = json.loads(tool_call_args_buf) if tool_call_args_buf else {}
            except json.JSONDecodeError:
                args = {}
            return ToolCall(id=tool_call_id, name=tool_call_name, arguments=args)

        return iter(chunks)


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
        chunks = list(client.stream(messages=[{"role": "user", "content": "Hi"}]))

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
        chunks = list(client.stream(messages=[{"role": "user", "content": "Hi"}]))

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
            list(client.stream(messages=[{"role": "user", "content": "Hi"}]))
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
        chunks = list(client.stream(messages=[{"role": "user", "content": "Hi"}]))

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
        chunks = list(client.stream(messages=[{"role": "user", "content": "Hi"}]))

    assert chunks == []


def test_stream_returns_tool_call() -> None:
    from unittest.mock import MagicMock, patch

    sse_lines = [
        'data: {"choices": [{"delta": {"tool_calls": [{"id": "call_1", "function": {"name": "web_search", "arguments": ""}}]}}]}',
        'data: {"choices": [{"delta": {"tool_calls": [{"id": "", "function": {"name": "", "arguments": "{\\"query\\": \\"capital of France\\"}"}}]}}]}',
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
        result = client.stream(messages=[{"role": "user", "content": "Hi"}])

    assert isinstance(result, ToolCall)
    assert result.name == "web_search"
    assert result.arguments == {"query": "capital of France"}


def test_stream_partial_tool_call_arguments() -> None:
    from unittest.mock import MagicMock, patch

    sse_lines = [
        'data: {"choices": [{"delta": {"tool_calls": [{"id": "c1", "function": {"name": "web_search", "arguments": "{\\"que"}}]}}]}',
        'data: {"choices": [{"delta": {"tool_calls": [{"id": "", "function": {"name": "", "arguments": "ry\\": \\"Paris\\"}"}}]}}]}',
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
        result = client.stream(messages=[{"role": "user", "content": "Hi"}])

    assert isinstance(result, ToolCall)
    assert result.name == "web_search"
    assert result.arguments == {"query": "Paris"}


def test_stream_tool_call_invalid_json_args_returns_empty_dict() -> None:
    from unittest.mock import MagicMock, patch

    sse_lines = [
        'data: {"choices": [{"delta": {"tool_calls": [{"id": "c1", "function": {"name": "get_datetime", "arguments": "not-json"}}]}}]}',
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
        result = client.stream(messages=[{"role": "user", "content": "Hi"}])

    assert isinstance(result, ToolCall)
    assert result.name == "get_datetime"
    assert result.arguments == {}
