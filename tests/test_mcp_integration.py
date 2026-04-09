from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import textwrap
import unittest
from unittest.mock import patch

from planner_service.graph import run_planner_graph
from planner_service.mcp import build_mcp_registry
from planner_service.persistence import load_jsonl


def _base_context() -> dict[str, object]:
    return {
        "calendarStartDate": "2026-04-08",
        "calendarEndDate": "2026-04-12",
        "visibleDays": 5,
        "dayStartHour": 7,
        "dayEndHour": 22,
        "timezone": "America/Chicago",
        "existingEvents": [],
        "relativeDayHints": [],
    }


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _make_http_urlopen(calls: list[dict[str, object]]):
    tools = [
        {
            "name": "weather_forecast",
            "description": "Read-only forecast lookup for weather planning.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "location": {"type": "string"},
                },
            },
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "calendar_write",
            "description": "Write an event to an external calendar.",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": False},
        },
    ]

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        body = json.loads((request.data or b"{}").decode("utf-8"))
        calls.append(body)
        method = body.get("method")
        request_id = body.get("id")
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05"}
        elif method == "tools/list":
            result = {"tools": tools}
        elif method == "tools/call":
            params = body.get("params") if isinstance(body.get("params"), dict) else {}
            tool_name = params.get("name")
            arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            if tool_name == "weather_forecast":
                result = {
                    "date": arguments.get("date", "2026-04-08"),
                    "location": arguments.get("location", "local area"),
                    "condition": "rain",
                    "high_f": 63,
                    "low_f": 54,
                    "humidity_pct": 92,
                    "summary": "rain with a high of 63F and a low of 54F.",
                }
            else:
                result = {"error": "unexpected tool"}
        else:
            result = {"ok": True}
        return _FakeHTTPResponse({"jsonrpc": "2.0", "id": request_id, "result": result})

    return fake_urlopen


def _build_stdio_script() -> str:
    return textwrap.dedent(
        """
import json
import sys

TOOLS = [
    {
        "name": "route_estimate",
        "description": "Read-only route and travel-time lookup.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "mode": {"type": "string"},
            },
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "route_write",
        "description": "Write a route to an external service.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": False},
    },
]


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line == b"\\r\\n":
            break
        key, value = line.decode("utf-8").split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def send_message(message):
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\\r\\n\\r\\n".encode("utf-8")
    sys.stdout.buffer.write(header + body)
    sys.stdout.buffer.flush()


while True:
    request = read_message()
    if request is None:
        break
    method = request.get("method")
    request_id = request.get("id")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05"}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        tool_name = params.get("name")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if tool_name == "route_estimate":
            result = {
                "origin": {"name": arguments.get("origin", "Local area"), "lat": 41.878, "lng": -87.629},
                "destination": {"name": arguments.get("destination", "Museum"), "lat": 41.882, "lng": -87.624},
                "duration_minutes": 47,
                "distance_km": 8.5,
                "summary": "Estimated 47 minute route between local area and museum.",
            }
        else:
            result = {"error": "unexpected tool"}
    else:
        result = {"ok": True}
    send_message({"jsonrpc": "2.0", "id": request_id, "result": result})
"""
    ).lstrip()


class MCPIntegrationTests(unittest.TestCase):
    def test_discovery_filters_write_tools_for_http_and_stdio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-mcp-") as tmp:
            tmp_path = Path(tmp)
            http_calls: list[dict[str, object]] = []

            stdio_script = tmp_path / "stdio_mcp_stub.py"
            stdio_script.write_text(_build_stdio_script(), encoding="utf-8")

            config = [
                {
                    "name": "weather-http",
                    "transport": "http",
                    "url": "http://mcp.example.invalid/mcp",
                },
                {
                    "name": "route-stdio",
                    "transport": "stdio",
                    "command": [sys.executable, str(stdio_script)],
                },
            ]
            (tmp_path / "mcp_servers.json").write_text(json.dumps(config), encoding="utf-8")

            with patch("planner_service.mcp.urlrequest.urlopen", new=_make_http_urlopen(http_calls)):
                registry = build_mcp_registry(data_root=tmp_path)
                self.addCleanup(registry.close)
            summary = registry.discovery_summary()

            tool_names = {tool["name"] for tool in summary["tools"]}
            self.assertEqual(tool_names, {"weather_forecast", "route_estimate"})
            server_statuses = {server["name"]: server["status"] for server in summary["servers"]}
            self.assertEqual(server_statuses["weather-http"], "available")
            self.assertEqual(server_statuses["route-stdio"], "available")
            self.assertGreaterEqual(len(http_calls), 2)

    def test_planner_uses_live_weather_and_route_context(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-mcp-run-") as tmp:
            tmp_path = Path(tmp)
            http_calls: list[dict[str, object]] = []

            stdio_script = tmp_path / "stdio_mcp_stub.py"
            stdio_script.write_text(_build_stdio_script(), encoding="utf-8")

            config = [
                {
                    "name": "weather-http",
                    "transport": "http",
                    "url": "http://mcp.example.invalid/mcp",
                },
                {
                    "name": "route-stdio",
                    "transport": "stdio",
                    "command": [sys.executable, str(stdio_script)],
                },
            ]
            (tmp_path / "mcp_servers.json").write_text(json.dumps(config), encoding="utf-8")

            with patch("planner_service.mcp.urlrequest.urlopen", new=_make_http_urlopen(http_calls)):
                registry = build_mcp_registry(data_root=tmp_path)
                self.addCleanup(registry.close)
                payload = {
                    "calendarId": "cal-mcp-live",
                    "messages": [{"role": "user", "text": "Plan an outdoor walk before lunch."}],
                    "context": _base_context(),
                }
                state = run_planner_graph(
                    payload,
                    ollama_url="http://127.0.0.1:11434",
                    model="gemma4:e2b",
                    timeout_seconds=1.0,
                    data_root=tmp_path,
                    mcp_registry=registry,
                )

            validation = state.get("validation_report") or {}
            self.assertTrue(validation.get("weather"))
            self.assertTrue(validation.get("routes"))
            self.assertEqual(validation["weather"][0]["source"], "mcp")
            self.assertEqual(validation["routes"][0]["source"], "mcp")

            session_log = tmp_path / "sessions" / "cal-mcp-live.jsonl"
            events = load_jsonl(session_log)
            event_names = [str(event.get("event")) for event in events]
            self.assertIn("mcp_discovery_summary", event_names)
            self.assertIn("mcp_tool_call", event_names)
            self.assertIn("mcp_tool_result", event_names)
            mcp_calls = [event for event in events if event.get("event") == "mcp_tool_call"]
            capabilities = {str(event.get("payload", {}).get("capability")) for event in mcp_calls}
            self.assertIn("weather", capabilities)
            self.assertIn("route", capabilities)
            self.assertGreaterEqual(len(http_calls), 3)

    def test_planner_falls_back_to_local_helpers_when_mcp_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-mcp-fallback-") as tmp:
            tmp_path = Path(tmp)
            payload = {
                "calendarId": "cal-mcp-fallback",
                "messages": [{"role": "user", "text": "Plan an outdoor walk before lunch."}],
                "context": _base_context(),
            }
            state = run_planner_graph(
                payload,
                ollama_url="http://127.0.0.1:11434",
                model="gemma4:e2b",
                timeout_seconds=1.0,
                data_root=tmp_path,
            )

            validation = state.get("validation_report") or {}
            self.assertTrue(validation.get("weather"))
            self.assertTrue(validation.get("routes"))
            self.assertEqual(validation["weather"][0]["source"], "mock")
            self.assertEqual(validation["routes"][0]["source"], "mock")

            session_log = tmp_path / "sessions" / "cal-mcp-fallback.jsonl"
            events = load_jsonl(session_log)
            discovery = next(event for event in events if event.get("event") == "mcp_discovery_summary")
            self.assertEqual(discovery.get("payload", {}).get("summary", {}).get("available_capabilities"), [])


if __name__ == "__main__":
    unittest.main()
