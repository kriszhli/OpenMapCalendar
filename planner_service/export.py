from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import safe_identifier
from .persistence import dump_json, load_jsonl


@dataclass(frozen=True)
class ExportBatch:
    path: Path
    row_count: int


def _group_runs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    run_order: list[str] = []

    for event in events:
        run_id = str(event.get("runId") or "")
        if not run_id:
            continue
        entry = grouped.setdefault(
            run_id,
            {
                "run_id": run_id,
                "calendar_id": str(event.get("calendarId") or ""),
                "request": None,
                "memory": None,
                "final": None,
                "distillation": [],
                "tool_summaries": [],
                "order": len(run_order),
            },
        )
        if run_id not in run_order:
            run_order.append(run_id)

        if event.get("event") == "planner_request_received":
            entry["request"] = event
        elif event.get("event") == "retrieved_memory_summary":
            entry["memory"] = event
        elif event.get("event") == "final_response":
            entry["final"] = event
        elif event.get("event") == "distillation_results":
            entry["distillation"].append(event)
        elif event.get("event") == "tool_result_summary":
            entry["tool_summaries"].append(event)

    rows: list[dict[str, Any]] = []
    for run_id in run_order:
        entry = grouped[run_id]
        request = entry["request"]
        final = entry["final"]
        if not request or not final:
            continue

        request_payload = request.get("payload") if isinstance(request.get("payload"), dict) else {}
        final_payload = final.get("payload") if isinstance(final.get("payload"), dict) else {}
        memory_payload = entry["memory"].get("payload") if isinstance(entry["memory"], dict) else {}

        rows.append(
            {
                "calendar_id": entry["calendar_id"],
                "run_id": run_id,
                "run_index": request_payload.get("run_index", entry["order"] + 1),
                "input": {
                    "calendar_window": request_payload.get("calendar_window", {}),
                    "messages": request_payload.get("messages", []),
                    "retrieved_memory": memory_payload.get("retrieved_memory", {}),
                },
                "output": {
                    "status": final_payload.get("status"),
                    "assistant_message": final_payload.get("assistant_message"),
                    "events": final_payload.get("events", []),
                    "tool_actions": [event.get("payload") for event in entry["tool_summaries"]],
                },
            }
        )
    return rows


def export_training_rows(*, data_root: str | Path, calendar_id: str | None = None) -> ExportBatch:
    root = Path(data_root)
    sessions_dir = root / "sessions"
    exports_dir = root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    if calendar_id:
        safe_calendar_id = safe_identifier(calendar_id)
        session_paths = [sessions_dir / f"{safe_calendar_id}.jsonl"]
        export_path = exports_dir / f"{safe_calendar_id}.jsonl"
    else:
        session_paths = sorted(sessions_dir.glob("*.jsonl"))
        export_path = exports_dir / "training.jsonl"

    rows: list[dict[str, Any]] = []
    for session_path in session_paths:
        events = load_jsonl(session_path)
        rows.extend(_group_runs(events))

    rows.sort(key=lambda row: (str(row.get("calendar_id") or ""), int(row.get("run_index") or 0), str(row.get("run_id") or "")))

    with export_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dump_json(row))
            handle.write("\n")

    return ExportBatch(path=export_path, row_count=len(rows))
