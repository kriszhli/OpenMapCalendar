from __future__ import annotations

from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from .mcp import MCPRegistry, build_mcp_registry
from .export import export_training_rows
from .graph import run_planner_graph


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:e2b"
DEFAULT_TIMEOUT_SECONDS = 45.0


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


class PlannerHandler(BaseHTTPRequestHandler):
    server_version = "PlannerService/0.1"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _send_json(self, 200, {"ok": True})
            return
        _send_json(self, 404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ai/plan-events":
            _send_json(self, 404, {"error": "not found"})
            return

        try:
            payload = _read_json_body(self)
        except Exception as exc:  # pragma: no cover - defensive
            _send_json(self, 400, {"error": "Invalid JSON body", "detail": str(exc)})
            return

        server = self.server  # type: ignore[assignment]
        assert isinstance(server, PlannerHTTPServer)

        try:
            state = run_planner_graph(
                payload,
                ollama_url=server.ollama_url,
                model=server.model,
                timeout_seconds=server.timeout_seconds,
                data_root=server.data_root,
                mcp_registry=server.mcp_registry,
            )
            response = state.get("response") or {
                "status": "needs_clarification",
                "assistantMessage": "Need a little more detail.",
                "events": [],
                "task_queue": [],
                "schedule_draft": state.get("schedule_draft", {}),
            }
            _send_json(self, 200, response)
        except Exception as exc:
            _send_json(
                self,
                503,
                {
                    "error": "Planner request failed",
                    "detail": str(exc),
                },
            )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


class PlannerHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        ollama_url: str,
        model: str,
        timeout_seconds: float,
        data_root: str,
        mcp_registry: MCPRegistry | None,
    ) -> None:
        super().__init__(address, handler)
        self.ollama_url = ollama_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.data_root = data_root
        self.mcp_registry = mcp_registry


def _default_smoke_payload(prompt: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "text": prompt},
        ],
        "context": {
            "calendarStartDate": "2026-04-08",
            "calendarEndDate": "2026-04-12",
            "visibleDays": 5,
            "dayStartHour": 7,
            "dayEndHour": 22,
            "timezone": "America/Chicago",
            "existingEvents": [],
            "relativeDayHints": [],
        },
    }


def run_smoke(
    prompt: str,
    *,
    ollama_url: str,
    model: str,
    timeout_seconds: float,
    data_root: str,
    verbose: bool = True,
) -> int:
    payload = _default_smoke_payload(prompt)
    payload["calendarId"] = "smoke-calendar"
    mcp_registry = build_mcp_registry(data_root=data_root)
    state = run_planner_graph(
        payload,
        ollama_url=ollama_url,
        model=model,
        timeout_seconds=timeout_seconds,
        data_root=data_root,
        mcp_registry=mcp_registry,
    )

    if verbose:
        print("== raw planner output ==")
        print(state.get("raw_model_output", ""))
        print("== task queue ==")
        print(json.dumps(state.get("task_queue", []), indent=2, sort_keys=True))
        print("== schedule draft ==")
        print(json.dumps(state.get("schedule_draft", {}), indent=2, sort_keys=True))
        print("== response ==")
        print(json.dumps(state.get("response", {}), indent=2, sort_keys=True))
    return 0


def run_server(
    *,
    host: str,
    port: int,
    ollama_url: str,
    model: str,
    timeout_seconds: float,
    data_root: str,
) -> int:
    mcp_registry = build_mcp_registry(data_root=data_root)
    server = PlannerHTTPServer(
        (host, port),
        PlannerHandler,
        ollama_url=ollama_url,
        model=model,
        timeout_seconds=timeout_seconds,
        data_root=data_root,
        mcp_registry=mcp_registry,
    )
    print(f"Planner service running at http://{host}:{port}")
    print(f"Ollama: {ollama_url}")
    print(f"Model: {model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Local planner service")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--data-root", default="./data", help="Local planner data root")
    parser.add_argument("--export", action="store_true", help="Export session logs to JSONL")
    parser.add_argument("--calendar-id", default=None, help="Limit export to one calendar")
    parser.add_argument("--serve", action="store_true", help="Run the HTTP service")
    parser.add_argument("--smoke", metavar="PROMPT", help="Run a one-off smoke test and print graph state")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.smoke:
        return run_smoke(
            args.smoke,
            ollama_url=args.ollama_url,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            data_root=args.data_root,
        )

    if args.export:
        batch = export_training_rows(data_root=args.data_root, calendar_id=args.calendar_id)
        print(f"Exported {batch.row_count} training rows to {batch.path}")
        return 0

    if args.serve or not args.smoke:
        return run_server(
            host=args.host,
            port=args.port,
            ollama_url=args.ollama_url,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            data_root=args.data_root,
        )

    return 0
