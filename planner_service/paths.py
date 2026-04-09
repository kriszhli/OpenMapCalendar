from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


def safe_identifier(value: str | None, *, default: str = "default") -> str:
    text = (value or "").strip()
    if not text:
        return default
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or default


@dataclass(frozen=True)
class PlannerPaths:
    data_root: Path
    checkpoints_dir: Path
    checkpoint_db_path: Path
    sessions_dir: Path
    session_log_path: Path
    exports_dir: Path


def build_planner_paths(data_root: str | Path, calendar_id: str) -> PlannerPaths:
    root = Path(data_root)
    safe_calendar_id = safe_identifier(calendar_id)
    checkpoints_dir = root / "checkpoints"
    sessions_dir = root / "sessions"
    exports_dir = root / "exports"
    return PlannerPaths(
        data_root=root,
        checkpoints_dir=checkpoints_dir,
        checkpoint_db_path=checkpoints_dir / "langgraph.sqlite",
        sessions_dir=sessions_dir,
        session_log_path=sessions_dir / f"{safe_calendar_id}.jsonl",
        exports_dir=exports_dir,
    )
