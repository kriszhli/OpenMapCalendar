from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [canonicalize(item) for item in value]
    return value


def dump_json(value: Any) -> str:
    return json.dumps(canonicalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


@dataclass
class SessionEventLogger:
    path: Path
    calendar_id: str
    session_id: str

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, payload: dict[str, Any], *, run_id: str | None = None, ts: str | None = None) -> dict[str, Any]:
        row = {
            "ts": ts or utc_now_iso(),
            "calendarId": self.calendar_id,
            "sessionId": self.session_id,
            "event": event,
            "payload": canonicalize(payload),
        }
        if run_id is not None:
            row["runId"] = run_id
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(dump_json(row))
            handle.write("\n")
        return row

    def count_events(self, event_name: str) -> int:
        return sum(1 for row in load_jsonl(self.path) if row.get("event") == event_name)


def tail_context_from_events(events: Iterable[dict[str, Any]], *, limit: int = 8) -> str:
    selected = list(events)[-limit:]
    parts: list[str] = []
    for event in selected:
        event_name = str(event.get("event") or "")
        payload = event.get("payload")
        if event_name == "final_response" and isinstance(payload, dict):
            parts.append(f"final:{payload.get('status', '')}:{payload.get('assistant_message', '')}")
            continue
        if event_name == "state_transition" and isinstance(payload, dict):
            parts.append(f"state:{payload.get('state', '')}")
            continue
        if event_name == "retrieved_memory_summary" and isinstance(payload, dict):
            parts.append(f"memory:{payload.get('summary_text', '')}")
            continue
        if event_name == "normalized_user_message_batch" and isinstance(payload, dict):
            messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
            texts = []
            for message in messages[-3:]:
                if isinstance(message, dict):
                    text = str(message.get("content") or message.get("text") or "")
                    if text:
                        texts.append(text[:120])
            if texts:
                parts.append("messages:" + " | ".join(texts))
    return "\n".join(part for part in parts if part).strip()


def read_recent_session_events(path: str | Path, *, limit: int = 8) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    return rows[-limit:]


class SQLitePlannerCheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS planner_checkpoints (
                    calendar_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    PRIMARY KEY (calendar_id, run_id, step_name)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS planner_runs (
                    calendar_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    latest_step TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    PRIMARY KEY (calendar_id, run_id)
                )
                """
            )

    def save(self, *, calendar_id: str, run_id: str, step_name: str, state: dict[str, Any], created_at: str) -> None:
        state_json = dump_json(state)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO planner_checkpoints (
                    calendar_id, run_id, step_name, created_at, state_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (calendar_id, run_id, step_name, created_at, state_json),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO planner_runs (
                    calendar_id, run_id, created_at, latest_step, state_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (calendar_id, run_id, created_at, step_name, state_json),
            )

    def latest_run(self, calendar_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, created_at, latest_step, state_json
                FROM planner_runs
                WHERE calendar_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (calendar_id,),
            ).fetchone()
        if not row:
            return None
        run_id, created_at, latest_step, state_json = row
        try:
            state = json.loads(state_json)
        except Exception:
            state = {}
        return {
            "calendar_id": calendar_id,
            "run_id": run_id,
            "created_at": created_at,
            "latest_step": latest_step,
            "state": state if isinstance(state, dict) else {},
        }
