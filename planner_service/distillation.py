from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any

from .memory import MemoryFact
from .persistence import load_jsonl


@dataclass(frozen=True)
class DistillationResult:
    facts: list[MemoryFact]
    skipped: list[dict[str, Any]]
    summary: dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_time_token(token: str) -> str | None:
    text = token.strip().lower().replace(" ", "")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or "00")
    suffix = match.group(3)
    if minute > 59 or hour > 24:
        return None
    if suffix:
        if hour == 12:
            hour = 0
        if suffix == "pm":
            hour += 12
    if hour > 23:
        return None
    return f"{hour:02d}:{minute:02d}"


def _extract_time_range(text: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:-|to|until|through)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    start = _normalize_time_token(match.group(1))
    end = _normalize_time_token(match.group(2))
    if not start or not end:
        return None
    return start, end


def _message_texts(events: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    texts: list[tuple[str, str | None]] = []
    for event in events:
        if event.get("event") != "normalized_user_message_batch":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        run_id = str(event.get("runId")) if event.get("runId") else None
        messages = payload.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "") != "user":
                continue
            text = str(message.get("content") or message.get("text") or "").strip()
            if text:
                texts.append((text, run_id))
    return texts


def _add_candidate(
    candidates: dict[tuple[str, str], dict[str, Any]],
    *,
    calendar_id: str,
    category: str,
    normalized_value: str,
    summary: str,
    confidence: float,
    session_id: str,
    run_id: str | None,
    evidence: str,
) -> None:
    key = (category, normalized_value)
    entry = candidates.setdefault(
        key,
        {
            "calendar_id": calendar_id,
            "category": category,
            "normalized_value": normalized_value,
            "summary": summary,
            "confidence": 0.0,
            "session_ids": set(),
            "run_ids": set(),
            "evidence": [],
        },
    )
    entry["confidence"] = max(float(entry["confidence"]), confidence)
    entry["session_ids"].add(session_id)
    if run_id:
        entry["run_ids"].add(run_id)
    entry["evidence"].append(evidence)


def _build_fact(
    calendar_id: str,
    category: str,
    normalized_value: str,
    summary: str,
    confidence: float,
    *,
    session_ids: set[str],
    run_ids: set[str],
) -> MemoryFact:
    now = _now_iso()
    return MemoryFact(
        calendar_id=calendar_id,
        category=category,
        normalized_value=normalized_value,
        confidence=round(float(confidence), 3),
        summary=summary,
        source_session_ids=sorted(session_ids),
        source_run_ids=sorted(run_ids),
        created_at=now,
        updated_at=now,
    )


def _extract_facts_from_text(
    text: str,
    *,
    calendar_id: str,
    session_id: str,
    run_id: str | None,
    candidates: dict[tuple[str, str], dict[str, Any]],
) -> None:
    lower = text.lower()
    time_range = _extract_time_range(text)

    if any(phrase in lower for phrase in ["work hours", "i work", "work from", "my schedule is"]):
        if time_range:
            start, end = time_range
            _add_candidate(
                candidates,
                calendar_id=calendar_id,
                category="work_hours",
                normalized_value=f"{start}-{end}",
                summary=f"Work hours are {start}-{end}.",
                confidence=0.94,
                session_id=session_id,
                run_id=run_id,
                evidence=text,
            )

    if any(phrase in lower for phrase in ["focus block", "deep work", "heads-down", "protected time"]):
        duration_match = re.search(r"(\d{1,2})\s*(?:minute|minutes|hour|hours|hr|hrs)", lower)
        duration = duration_match.group(1) if duration_match else ""
        normalized = f"{duration}m focus block".strip()
        confidence = 0.9 if "focus block" in lower or "deep work" in lower else 0.82
        _add_candidate(
            candidates,
            calendar_id=calendar_id,
            category="focus_block",
            normalized_value=normalized or "focus block",
            summary="User prefers protected focus time.",
            confidence=confidence,
            session_id=session_id,
            run_id=run_id,
            evidence=text,
        )

    if any(phrase in lower for phrase in ["before lunch", "after lunch", "avoid lunch", "lunch window", "not at lunch"]):
        normalized = "avoid lunch window"
        if "before lunch" in lower:
            normalized = "prefer before lunch"
        elif "after lunch" in lower:
            normalized = "prefer after lunch"
        _add_candidate(
            candidates,
            calendar_id=calendar_id,
            category="lunch_avoidance",
            normalized_value=normalized,
            summary="User avoids scheduling around lunch.",
            confidence=0.91,
            session_id=session_id,
            run_id=run_id,
            evidence=text,
        )

    if any(phrase in lower for phrase in ["commute", "traffic", "travel buffer", "buffer", "drive time", "leave extra time"]):
        minutes_match = re.search(r"(\d{1,3})\s*(?:minute|minutes|min)\b", lower)
        if minutes_match:
            buffer_minutes = int(minutes_match.group(1))
            normalized = f"{buffer_minutes} minute commute buffer"
            _add_candidate(
                candidates,
                calendar_id=calendar_id,
                category="commute_sensitivity",
                normalized_value=normalized,
                summary="User wants commute/travel buffer.",
                confidence=0.92,
                session_id=session_id,
                run_id=run_id,
                evidence=text,
            )

    if any(phrase in lower for phrase in ["prefer mornings", "prefer afternoon", "prefer afternoons", "prefer evenings", "always", "every", "usually", "first thing", "last thing"]):
        if "morning" in lower:
            normalized = "prefer mornings"
        elif "afternoon" in lower:
            normalized = "prefer afternoons"
        elif "evening" in lower:
            normalized = "prefer evenings"
        else:
            normalized = "recurring time preference"
        confidence = 0.88 if any(phrase in lower for phrase in ["prefer", "always", "every"]) else 0.76
        _add_candidate(
            candidates,
            calendar_id=calendar_id,
            category="recurring_habit",
            normalized_value=normalized,
            summary="User shows a recurring scheduling habit.",
            confidence=confidence,
            session_id=session_id,
            run_id=run_id,
            evidence=text,
        )


def distill_session_log(
    *,
    session_log_path: str,
    calendar_id: str,
    session_id: str,
) -> DistillationResult:
    events = load_jsonl(session_log_path)
    texts = _message_texts(events)
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    run_ids = [str(event.get("runId")) for event in events if event.get("runId")]
    active_run_id = run_ids[-1] if run_ids else None

    for text, run_id in texts:
        _extract_facts_from_text(
            text,
            calendar_id=calendar_id,
            session_id=session_id,
            run_id=run_id or active_run_id,
            candidates=candidates,
        )

    facts: list[MemoryFact] = []
    skipped: list[dict[str, Any]] = []
    for key in sorted(candidates):
        entry = candidates[key]
        source_sessions = entry["session_ids"]
        source_runs = entry["run_ids"]
        evidence_count = len(entry["evidence"])
        confidence = float(entry["confidence"])
        if evidence_count >= 2:
            confidence = max(confidence, 0.85)
        if confidence < 0.75:
            skipped.append(
                {
                    "category": entry["category"],
                    "normalized_value": entry["normalized_value"],
                    "reason": "below_confidence_threshold",
                    "confidence": round(confidence, 3),
                }
            )
            continue
        facts.append(
            _build_fact(
                calendar_id,
                entry["category"],
                entry["normalized_value"],
                entry["summary"],
                confidence,
                session_ids=source_sessions,
                run_ids=source_runs,
            )
        )

    summary = {
        "calendar_id": calendar_id,
        "fact_count": len(facts),
        "skipped_count": len(skipped),
        "categories": [fact.category for fact in facts],
    }
    return DistillationResult(facts=facts, skipped=skipped, summary=summary)
