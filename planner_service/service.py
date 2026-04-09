from __future__ import annotations

from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from .distillation import distill_session_log
from .mcp import MCPRegistry, build_mcp_registry
from .export import export_training_rows
from .graph import run_planner_graph
from .memory import LongTermMemoryStore, MemoryFact
from .paths import build_planner_paths


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


def _confirmation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    confirmed_schedule = payload.get("confirmed_schedule") if isinstance(payload.get("confirmed_schedule"), dict) else {}
    schedule_draft = payload.get("schedule_draft") if isinstance(payload.get("schedule_draft"), dict) else {}
    correction_diff = payload.get("correction_diff") if isinstance(payload.get("correction_diff"), dict) else {}
    preference_updates = payload.get("preference_updates") if isinstance(payload.get("preference_updates"), list) else []
    user_confirmed = payload.get("user_confirmed") is True
    return {
        "user_confirmed": user_confirmed,
        "run_id": str(payload.get("run_id") or ""),
        "calendar_id": str(payload.get("calendar_id") or payload.get("calendarId") or ""),
        "confirmed_schedule": confirmed_schedule,
        "schedule_draft": schedule_draft,
        "correction_diff": correction_diff,
        "preference_updates": preference_updates,
        "confirmed_at": payload.get("confirmed_at") or payload.get("confirmedAt") or "",
    }


def _manual_correction_facts(
    *,
    calendar_id: str,
    session_id: str,
    run_id: str,
    correction_diff: dict[str, Any],
    preference_updates: list[dict[str, Any]],
) -> list[MemoryFact]:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    facts: list[MemoryFact] = []

    for update in preference_updates:
        category = str(update.get("category") or "manual_correction").strip() or "manual_correction"
        normalized_value = str(update.get("normalized_value") or update.get("value") or "manual correction").strip() or "manual correction"
        summary = str(update.get("summary") or normalized_value).strip() or normalized_value
        facts.append(
            MemoryFact(
                calendar_id=calendar_id,
                category=category,
                normalized_value=normalized_value,
                confidence=1.0,
                summary=summary,
                source_session_ids=[session_id],
                source_run_ids=[run_id],
                created_at=now,
                updated_at=now,
                forced=True,
                priority=100,
            )
        )

    if correction_diff:
        summary = str(correction_diff.get("summary") or correction_diff.get("kind") or "manual correction").strip()
        if summary:
            facts.append(
                MemoryFact(
                    calendar_id=calendar_id,
                    category="manual_correction",
                    normalized_value=summary[:120],
                    confidence=0.85,
                    summary=summary,
                    source_session_ids=[session_id],
                    source_run_ids=[run_id],
                    created_at=now,
                    updated_at=now,
                    forced=False,
                    priority=75,
                )
            )

    return facts


def _append_confirmation_record(
    *,
    data_root: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    confirmation = _confirmation_payload(payload)
    if not confirmation["user_confirmed"]:
        raise ValueError("plan confirmation requires user_confirmed: true")

    calendar_id = confirmation["calendar_id"] or "default-calendar"
    paths = build_planner_paths(data_root, calendar_id)
    from .persistence import SessionEventLogger

    session_logger = SessionEventLogger(paths.session_log_path, calendar_id, calendar_id)
    run_id = confirmation["run_id"]
    session_logger.append("plan_confirmation_received", confirmation, run_id=run_id)
    if confirmation["correction_diff"]:
        session_logger.append(
            "manual_correction_received",
            {
                "run_id": run_id,
                "calendar_id": calendar_id,
                "correction_diff": confirmation["correction_diff"],
                "preference_updates": confirmation["preference_updates"],
            },
            run_id=run_id,
        )

    memory_store = LongTermMemoryStore(paths.data_root / "chromadb")
    distillation = distill_session_log(
        session_log_path=str(paths.session_log_path),
        calendar_id=calendar_id,
        session_id=calendar_id,
    )
    if distillation.facts:
        memory_store.upsert_facts(distillation.facts)
    distilled_rows = [
        {
            "category": fact.category,
            "normalized_value": fact.normalized_value,
            "confidence": fact.confidence,
            "forced": fact.forced,
            "priority": fact.priority,
            "document_id": fact.document_id,
        }
        for fact in distillation.facts
    ]
    session_logger.append(
        "distillation_results",
        {
            "summary": distillation.summary,
            "upserted_facts": distilled_rows,
            "skipped": distillation.skipped,
        },
        run_id=run_id,
    )
    session_logger.append(
        "plan_confirmation_applied",
        {
        "run_id": run_id,
        "calendar_id": calendar_id,
        "upserted_fact_count": len(distilled_rows),
    },
        run_id=run_id,
    )
    return {
        "ok": True,
        "calendar_id": calendar_id,
        "run_id": run_id,
        "confirmed_schedule": confirmation["confirmed_schedule"],
        "correction_diff": confirmation["correction_diff"],
        "distilled_facts": distilled_rows,
    }


def _health_check_seed_memory(data_root: str, calendar_id: str) -> None:
    memory_store = LongTermMemoryStore(build_planner_paths(data_root, calendar_id).data_root / "chromadb")
    existing = memory_store.query(calendar_id=calendar_id, query_text="lunch focus commute morning afternoon", limit=5)
    if existing.get("facts"):
        return

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    memory_store.upsert_facts(
        [
            MemoryFact(
                calendar_id=calendar_id,
                category="lunch_avoidance",
                normalized_value="avoid lunch window",
                confidence=0.96,
                summary="User avoids lunch meetings.",
                source_session_ids=["health-seed"],
                source_run_ids=["health-seed"],
                created_at=now,
                updated_at=now,
                forced=True,
                priority=100,
            ),
            MemoryFact(
                calendar_id=calendar_id,
                category="focus_block",
                normalized_value="2 hour focus block",
                confidence=0.94,
                summary="User prefers protected focus time.",
                source_session_ids=["health-seed"],
                source_run_ids=["health-seed"],
                created_at=now,
                updated_at=now,
                forced=True,
                priority=95,
            ),
        ]
    )


def run_health_check(
    *,
    data_root: str,
    ollama_url: str,
    model: str,
    timeout_seconds: float,
    calendar_id: str = "health-calendar",
) -> int:
    _health_check_seed_memory(data_root, calendar_id)
    mcp_registry = build_mcp_registry(data_root=data_root)
    prompts = [
        {
            "name": "lunch_avoidance",
            "prompt": "Schedule a lunch check-in today.",
            "expect": lambda state: (state.get("response") or {}).get("status") == "ready",
            "check": lambda state: _first_event_time(state) not in {"12:00", "12:30", "13:00"},
        },
        {
            "name": "focus_block",
            "prompt": "Plan a 2 hour focus block this afternoon.",
            "expect": lambda state: (state.get("response") or {}).get("status") == "ready",
            "check": lambda state: any("focus" in str((block or {}).get("title", "")).lower() for block in _draft_blocks(state)),
        },
        {
            "name": "weather_sensitive",
            "prompt": "Plan an outdoor walk.",
            "expect": lambda state: (state.get("response") or {}).get("status") == "ready",
            "check": lambda state: bool((state.get("validation_report") or {}).get("weather")),
        },
        {
            "name": "route_buffer",
            "prompt": "Plan a museum visit before lunch.",
            "expect": lambda state: (state.get("response") or {}).get("status") == "ready",
            "check": lambda state: len(_draft_blocks(state)) >= 1 and bool((state.get("validation_report") or {}).get("routes")),
        },
    ]

    try:
        results: list[dict[str, Any]] = []
        all_ok = True
        for item in prompts:
            state = run_planner_graph(
                {
                    "calendarId": calendar_id,
                    "messages": [{"role": "user", "text": item["prompt"]}],
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
                },
                ollama_url=ollama_url,
                model=model,
                timeout_seconds=timeout_seconds,
                data_root=data_root,
                mcp_registry=mcp_registry,
            )
            passed = bool(item["expect"](state)) and bool(item["check"](state))
            all_ok = all_ok and passed
            results.append(
                {
                    "name": item["name"],
                    "passed": passed,
                    "status": (state.get("response") or {}).get("status"),
                    "assistant_message": (state.get("response") or {}).get("assistantMessage"),
                    "event_count": len((state.get("response") or {}).get("events") or []),
                }
            )

        print(json.dumps({"calendar_id": calendar_id, "results": results}, indent=2, sort_keys=True))
        return 0 if all_ok else 1
    finally:
        mcp_registry.close()


def _draft_blocks(state: dict[str, Any]) -> list[dict[str, Any]]:
    draft = state.get("schedule_draft") or {}
    blocks = draft.get("blocks") if isinstance(draft, dict) else []
    return [block for block in blocks if isinstance(block, dict)]


def _first_event_time(state: dict[str, Any]) -> str:
    blocks = _draft_blocks(state)
    if not blocks:
        return ""
    first = blocks[0]
    return str(first.get("start_time") or "")


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
        try:
            payload = _read_json_body(self)
        except Exception as exc:  # pragma: no cover - defensive
            _send_json(self, 400, {"error": "Invalid JSON body", "detail": str(exc)})
            return

        server = self.server  # type: ignore[assignment]
        assert isinstance(server, PlannerHTTPServer)

        try:
            if self.path == "/api/ai/plan-events":
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
                return

            if self.path == "/api/ai/confirm-plan":
                confirmation = _append_confirmation_record(data_root=server.data_root, payload=payload)
                _send_json(self, 200, confirmation)
                return

            _send_json(self, 404, {"error": "not found"})
        except ValueError as exc:
            _send_json(
                self,
                400,
                {
                    "error": "Invalid confirmation payload",
                    "detail": str(exc),
                },
            )
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
    parser.add_argument("--health-check", action="store_true", help="Run local planning regression checks")
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

    if args.health_check:
        target_calendar = args.calendar_id or "health-calendar"
        return run_health_check(
            data_root=args.data_root,
            ollama_url=args.ollama_url,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            calendar_id=target_calendar,
        )

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
