import datetime
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this for news, facts, weather, or anything you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

ALLOWED_TOOL_NAMES: frozenset[str] = frozenset(t["function"]["name"] for t in TOOL_SCHEMAS)

FILLER_PHRASES: dict[str, str] = {
    "web_search": "Let me look that up.",
    "get_datetime": "One moment.",
}

DEFAULT_FILLER = "Let me think about that."


def web_search(query: str) -> str:
    encoded_query = urllib.parse.quote_plus(query)
    instant_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
    try:
        req = urllib.request.Request(instant_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        abstract = data.get("AbstractText", "")
        if abstract:
            return abstract
        answer = data.get("Answer", "")
        if answer:
            return answer
    except Exception as e:
        logger.warning("DuckDuckGo instant answer failed: %s", e)
        return "Search unavailable right now."

    try:
        html_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        req = urllib.request.Request(html_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        snippets: list[str] = []
        pos = 0
        while len(snippets) < 3:
            idx = html.find('<a class="result__snippet"', pos)
            if idx == -1:
                break
            start = html.find(">", idx)
            if start == -1:
                break
            end = html.find("</a>", start)
            if end == -1:
                break
            raw = html[start + 1:end]
            clean = re.sub(r"<[^>]+>", "", raw).strip()
            if clean:
                snippets.append(clean)
            pos = end + 1
        if snippets:
            return " | ".join(snippets)
        return "No results found."
    except Exception as e:
        logger.warning("DuckDuckGo HTML scrape failed: %s", e)
        return "Search unavailable right now."


def get_datetime() -> str:
    return datetime.datetime.now().strftime("It is %A, %B %d %Y, %I:%M %p")


def dispatch(tool_name: str, tool_args: dict[str, Any]) -> str:
    logger.info("Dispatching tool %s with args %s", tool_name, tool_args)
    if tool_name == "web_search":
        query = tool_args.get("query", "").strip()
        if not query:
            return "Please provide a search query."
        return web_search(query)
    if tool_name == "get_datetime":
        return get_datetime()
    return f"Unknown tool: {tool_name}"


def get_filler(tool_name: str) -> str:
    return FILLER_PHRASES.get(tool_name, DEFAULT_FILLER)


def test_get_datetime_format() -> None:
    result = get_datetime()
    assert result.startswith("It is ")
    assert str(datetime.datetime.now().year) in result


def test_dispatch_get_datetime() -> None:
    result = dispatch("get_datetime", {})
    assert result.startswith("It is ")


def test_dispatch_unknown_tool() -> None:
    result = dispatch("nonexistent", {})
    assert "Unknown tool" in result


def test_web_search_instant_answer() -> None:
    from unittest.mock import MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"AbstractText": "Paris is the capital of France.", "Answer": ""}).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = web_search("capital of France")

    assert result == "Paris is the capital of France."


def test_web_search_answer_fallback() -> None:
    from unittest.mock import MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"AbstractText": "", "Answer": "42"}).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = web_search("answer to everything")

    assert result == "42"


def test_web_search_html_fallback() -> None:
    from unittest.mock import MagicMock, patch, call

    instant_resp = MagicMock()
    instant_resp.read.return_value = json.dumps({"AbstractText": "", "Answer": ""}).encode()
    instant_resp.__enter__ = lambda s: s
    instant_resp.__exit__ = MagicMock(return_value=False)

    html_content = '<a class="result__snippet">Some result text</a>'
    html_resp = MagicMock()
    html_resp.read.return_value = html_content.encode()
    html_resp.__enter__ = lambda s: s
    html_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", side_effect=[instant_resp, html_resp]):
        result = web_search("something")

    assert "Some result text" in result


def test_web_search_exception_returns_unavailable() -> None:
    from unittest.mock import patch

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
        result = web_search("anything")

    assert result == "Search unavailable right now."


def test_get_filler_known() -> None:
    assert get_filler("web_search") == "Let me look that up."


def test_get_filler_unknown() -> None:
    assert get_filler("xyz") == DEFAULT_FILLER


def test_dispatch_empty_query() -> None:
    result = dispatch("web_search", {"query": ""})
    assert result == "Please provide a search query."
