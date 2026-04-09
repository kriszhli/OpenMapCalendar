from __future__ import annotations

from copy import deepcopy
from datetime import date as date_cls, timedelta
import sqlite3
from typing import Any, Callable, Literal, NotRequired, TypedDict

from .distillation import distill_session_log
from .memory import LongTermMemoryStore, sanitize_query_text
from .paths import build_planner_paths
from .persistence import (
    SessionEventLogger,
    SQLitePlannerCheckpointStore,
    read_recent_session_events,
    tail_context_from_events,
    utc_now_iso,
)
from .tools import MockCalendarTool, MockWeatherTool

try:
    from langgraph.graph import END, StateGraph  # type: ignore
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    END = "__end__"
    StateGraph = None
    SqliteSaver = None


Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class TaskItem(TypedDict, total=False):
    id: str
    tool: Literal["calendar_mock", "weather_mock"]
    action: str
    date: str
    start_time: NotRequired[str]
    end_time: NotRequired[str]
    title: NotRequired[str]
    location: NotRequired[str]
    origin: NotRequired[str]
    destination: NotRequired[str]
    color: NotRequired[str]
    description: NotRequired[str]


class ScheduleDraft(TypedDict, total=False):
    status: Literal["ready", "needs_clarification"]
    assistantMessage: str
    tasks: list[TaskItem]
    events: list[dict[str, Any]]
    calendar_checks: list[dict[str, Any]]
    weather: list[dict[str, Any]]
    notes: list[str]
    blocks: list[dict[str, Any]]
    validation: dict[str, Any]
    clarification_options: list[dict[str, Any]]
    resolution_path: list[str]


class AgentState(TypedDict, total=False):
    calendar_id: str
    session_id: str
    run_id: str
    event_log_path: str
    retrieved_memory: dict[str, Any]
    distilled_facts: list[dict[str, Any]]
    messages: list[ChatMessage]
    task_queue: list[TaskItem]
    schedule_draft: ScheduleDraft
    decomposition: dict[str, Any]
    validation_report: dict[str, Any]
    conflict_history: list[dict[str, Any]]
    replan_attempts: int
    latest_resolution_path: list[str]
    clarification_options: list[dict[str, Any]]
    raw_model_output: str
    planner_payload: dict[str, Any]
    response: dict[str, Any]


DEFAULT_REPLAN_LIMIT = 2
GOOD_WEATHER_CONDITIONS = {"sunny", "partly cloudy"}
DEFAULT_BUFFER_MINUTES = 30
LUNCH_START_MINUTES = 12 * 60
LUNCH_END_MINUTES = 13 * 60
BAD_WEATHER_KEYWORDS = {"rain", "windy", "storm", "snow", "sleet", "hail"}


def _clean_text(value: Any, max_len: int = 400) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:max_len]


def _iso_date(value: Any) -> str:
    text = _clean_text(value, 20)
    try:
        parsed = date_cls.fromisoformat(text)
    except Exception:
        return ""
    return parsed.isoformat()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : idx + 1]
                try:
                    import json

                    parsed = json.loads(snippet)
                except Exception:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    think_start = lower.find("<|think|>")
    if think_start >= 0:
        after_think = text[think_start + len("<|think|>") :]
        end_token_candidates = ["</think>", "<|endthink|>", "<|end_of_think|>"]
        for token in end_token_candidates:
            token_index = after_think.lower().find(token)
            if token_index >= 0:
                return after_think[token_index + len(token) :].strip()
        brace_index = after_think.find("{")
        return after_think[brace_index:].strip() if brace_index >= 0 else after_think.strip()
    return text.strip()


def _normalize_messages(messages: list[dict[str, Any]] | None) -> list[ChatMessage]:
    normalized: list[ChatMessage] = []
    for item in messages or []:
        role = "assistant" if item.get("role") == "assistant" else "user"
        content = _clean_text(item.get("content") or item.get("text"), 4000)
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _compact_message_batch(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [
        {
            "role": message.get("role", "user"),
            "content": _clean_text(message.get("content"), 240),
        }
        for message in messages
    ]


def _latest_user_message(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _extract_date_hint(text: str) -> str:
    import re

    lower = text.lower()
    if "second day" in lower:
        return "day_2"
    if "first day" in lower:
        return "day_1"
    match = re.search(r"\bday\s+(\d{1,2})\b", lower)
    if match:
        return f"day_{match.group(1)}"
    return ""


def _time_to_minutes(value: str) -> int | None:
    text = _clean_text(value, 5)
    if not text or ":" not in text:
        return None
    try:
        hour_text, minute_text = text.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except Exception:
        return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour * 60 + minute


def _minutes_to_time(value: int) -> str:
    value = max(0, min(23 * 60 + 59, int(value)))
    return f"{value // 60:02d}:{value % 60:02d}"


def _clamp_minutes(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _parse_time_window(text: str) -> tuple[str, str] | None:
    import re

    lower = text.lower()
    match = re.search(
        r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:-|to|until|through)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
        lower,
    )
    if not match:
        return None

    def parse_token(token: str) -> str | None:
        token = token.strip().replace(" ", "")
        token_match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", token)
        if not token_match:
            return None
        hour = int(token_match.group(1))
        minute = int(token_match.group(2) or "00")
        suffix = token_match.group(3)
        if hour == 12:
            hour = 0
        if suffix == "pm":
            hour += 12
        if hour > 23 or minute > 59:
            return None
        return f"{hour:02d}:{minute:02d}"

    start = parse_token(match.group(1))
    end = parse_token(match.group(2))
    if not start or not end:
        return None
    return start, end


def _has_explicit_time(text: str) -> bool:
    return _parse_time_window(text) is not None or bool(
        __import__("re").search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", text, flags=__import__("re").IGNORECASE)
    )


def _split_intent_segments(text: str) -> list[str]:
    import re

    raw = _clean_text(text, 400)
    if not raw:
        return []

    splitters = [r"\band then\b", r"\bthen\b", r"\bafter that\b", r"\bnext\b", r"[;,]"]
    segments = [raw]
    for splitter in splitters:
        next_segments: list[str] = []
        for segment in segments:
            pieces = [piece.strip() for piece in re.split(splitter, segment, flags=re.IGNORECASE) if piece.strip()]
            if len(pieces) > 1:
                next_segments.extend(pieces)
            else:
                next_segments.append(segment.strip())
        segments = next_segments

    if len(segments) == 1:
        text_lower = raw.lower()
        if " and " in text_lower:
            keywords = [
                "museum",
                "dinner",
                "coffee",
                "lunch",
                "meeting",
                "call",
                "walk",
                "hike",
                "visit",
                "work",
                "focus",
                "travel",
                "commute",
                "run",
                "gym",
                "doctor",
                "shopping",
            ]
            if sum(1 for keyword in keywords if keyword in text_lower) >= 2:
                pieces = [piece.strip() for piece in re.split(r"\band\b", raw, flags=re.IGNORECASE) if piece.strip()]
                if len(pieces) > 1:
                    segments = pieces

    normalized: list[str] = []
    for segment in segments:
        cleaned = segment.strip(" .")
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)

    expanded: list[str] = []
    for segment in normalized:
        lower = segment.lower()
        if " and " in lower:
            keyword_count = sum(
                1
                for keyword in [
                    "museum",
                    "dinner",
                    "coffee",
                    "lunch",
                    "meeting",
                    "call",
                    "walk",
                    "hike",
                    "visit",
                    "work",
                    "focus",
                    "travel",
                    "commute",
                    "run",
                    "gym",
                    "doctor",
                    "shopping",
                ]
                if keyword in lower
            )
            pieces = [piece.strip() for piece in re.split(r"\band\b", segment, flags=re.IGNORECASE) if piece.strip()]
            if keyword_count >= 2 and len(pieces) > 1:
                expanded.extend(pieces)
                continue
        expanded.append(segment)
    return expanded


def _extract_target_date(text: str, context: dict[str, Any], *, segment_index: int = 0) -> str:
    import re

    lower = text.lower()
    explicit = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", lower)
    if explicit:
        return explicit.group(1)

    calendar_start = _iso_date(context.get("calendarStartDate")) or _iso_date(context.get("today"))
    if not calendar_start:
        calendar_start = date_cls.today().isoformat()

    day_hint = _extract_date_hint(text)
    if day_hint.startswith("day_"):
        try:
            day_index = max(1, int(day_hint.split("_", 1)[1]))
            return (date_cls.fromisoformat(calendar_start) + timedelta(days=day_index - 1)).isoformat()
        except Exception:
            pass

    if "tomorrow" in lower:
        return (date_cls.fromisoformat(calendar_start) + timedelta(days=1)).isoformat()
    if "today" in lower:
        return calendar_start
    if "next week" in lower:
        return (date_cls.fromisoformat(calendar_start) + timedelta(days=7)).isoformat()

    return (date_cls.fromisoformat(calendar_start) + timedelta(days=segment_index)).isoformat()


def _infer_activity_title(segment: str) -> str:
    lower = segment.lower()
    for pattern, title in [
        ("museum", "Museum visit"),
        ("dinner", "Dinner"),
        ("lunch", "Lunch"),
        ("coffee", "Coffee meeting"),
        ("meeting", "Meeting"),
        ("call", "Call"),
        ("walk", "Walk"),
        ("hike", "Hike"),
        ("travel", "Travel block"),
        ("commute", "Commute buffer"),
        ("focus", "Focus block"),
        ("work", "Work block"),
        ("shopping", "Shopping"),
        ("doctor", "Doctor visit"),
    ]:
        if pattern in lower:
            return title
    return _clean_text(segment.title(), 80) or "Planned item"


def _estimate_duration_minutes(segment: str) -> int:
    lower = segment.lower()
    if any(keyword in lower for keyword in ["focus", "deep work", "work block", "meeting", "call"]):
        return 60
    if any(keyword in lower for keyword in ["museum", "visit", "hike", "walk", "travel", "commute", "shopping"]):
        return 90
    if any(keyword in lower for keyword in ["dinner", "lunch", "coffee", "breakfast"]):
        return 60
    return 60


def _infer_weather_sensitive(segment: str) -> bool:
    lower = segment.lower()
    return any(keyword in lower for keyword in ["outdoor", "outside", "walk", "hike", "beach", "park", "picnic", "bike", "run"])


def _infer_flexible(segment: str) -> bool:
    if _has_explicit_time(segment):
        return False
    return True


def _infer_window_from_text(
    segment: str,
    context: dict[str, Any],
    *,
    previous_end: int | None = None,
) -> tuple[str, str, bool]:
    lower = segment.lower()
    if "before lunch" in lower:
        return "10:00", "12:00", False
    if "after lunch" in lower:
        return "13:00", "16:00", False
    if "around lunch" in lower or (("lunch" in lower) and "before lunch" not in lower and "after lunch" not in lower):
        return "12:00", "13:00", True
    if "morning" in lower:
        return "09:00", "12:00", False
    if "afternoon" in lower:
        return "13:00", "17:00", False
    if "evening" in lower:
        return "18:00", "21:00", False

    parsed_window = _parse_time_window(segment)
    if parsed_window:
        return parsed_window[0], parsed_window[1], False

    day_start = int(context.get("dayStartHour") or 7)
    day_end = int(context.get("dayEndHour") or 22)
    start_minutes = day_start * 60 + 120
    if previous_end is not None:
        start_minutes = previous_end + DEFAULT_BUFFER_MINUTES
    start_minutes = _clamp_minutes(start_minutes, day_start * 60, max(day_start * 60, (day_end * 60) - 60))
    duration = _estimate_duration_minutes(segment)
    end_minutes = _clamp_minutes(start_minutes + duration, start_minutes + 30, day_end * 60)
    return _minutes_to_time(start_minutes), _minutes_to_time(end_minutes), _infer_flexible(segment)


def _extract_memory_preferences(retrieved_memory: dict[str, Any]) -> dict[str, Any]:
    facts = retrieved_memory.get("facts") if isinstance(retrieved_memory, dict) else []
    parsed_facts = facts if isinstance(facts, list) else []
    summary_text = str(retrieved_memory.get("summary_text") or "") if isinstance(retrieved_memory, dict) else ""
    preference_flags = {
        "avoid_lunch": "lunch_avoidance" in summary_text or any(
            isinstance(fact, dict) and fact.get("category") == "lunch_avoidance" for fact in parsed_facts
        ),
        "focus_block": "focus_block" in summary_text or any(
            isinstance(fact, dict) and fact.get("category") == "focus_block" for fact in parsed_facts
        ),
        "prefer_mornings": "prefer mornings" in summary_text.lower() or any(
            isinstance(fact, dict) and fact.get("normalized_value") == "prefer mornings" for fact in parsed_facts
        ),
        "prefer_afternoons": "prefer afternoons" in summary_text.lower() or any(
            isinstance(fact, dict) and fact.get("normalized_value") == "prefer afternoons" for fact in parsed_facts
        ),
        "commute_buffer": any(
            isinstance(fact, dict) and fact.get("category") == "commute_sensitivity" for fact in parsed_facts
        ),
        "recurring_habit": any(
            isinstance(fact, dict) and fact.get("category") == "recurring_habit" for fact in parsed_facts
        ),
    }
    return {"summary_text": summary_text, "facts": parsed_facts, "flags": preference_flags}


def _priority_for_block(block: dict[str, Any], memory_preferences: dict[str, Any]) -> int:
    priority = 40
    if block.get("fixed_time"):
        priority += 40
    if block.get("weather_sensitive"):
        priority += 10
    if block.get("kind") in {"focus", "work", "habit"}:
        priority += 15
    if block.get("kind") in {"calendar_check", "write"}:
        priority -= 5
    if memory_preferences.get("flags", {}).get("recurring_habit") and block.get("kind") == "habit":
        priority += 20
    if memory_preferences.get("flags", {}).get("focus_block") and block.get("kind") == "focus":
        priority += 15
    return max(0, min(100, priority))


def _build_decomposition_blocks(state: AgentState, context: dict[str, Any]) -> list[dict[str, Any]]:
    user_text = _latest_user_message(state.get("messages", []))
    planner_payload = state.get("planner_payload") or {}
    memory_preferences = _extract_memory_preferences(context.get("retrievedMemory") or {})
    segments = _split_intent_segments(user_text)
    if not segments:
        segments = [user_text.strip() or "Planned item"]

    blocks: list[dict[str, Any]] = []
    base_target_date = _extract_target_date(user_text, context)

    for index, segment in enumerate(segments):
        lower = segment.lower()
        action_keywords = [
            "schedule",
            "plan",
            "visit",
            "dinner",
            "lunch",
            "coffee",
            "meeting",
            "call",
            "walk",
            "hike",
            "travel",
            "commute",
            "focus",
            "work",
            "shopping",
            "doctor",
            "gym",
            "run",
        ]
        if not any(keyword in lower for keyword in action_keywords):
            continue

        target_date = base_target_date
        if __import__("re").search(r"\b20\d{2}-\d{2}-\d{2}\b", lower) or any(
            word in lower for word in ["tomorrow", "today", "next week", "first day", "second day", "third day"]
        ):
            target_date = _extract_target_date(segment, context, segment_index=index)
        previous_end = None
        if blocks:
            previous_end = _time_to_minutes(str(blocks[-1].get("end_time") or "")) or None
        start_time, end_time, flexible = _infer_window_from_text(segment, context, previous_end=previous_end)
        fixed_time = _has_explicit_time(segment)
        kind = "generic"
        if any(keyword in lower for keyword in ["focus", "deep work"]):
            kind = "focus"
        elif any(keyword in lower for keyword in ["meeting", "call"]):
            kind = "meeting"
        elif any(keyword in lower for keyword in ["lunch", "dinner", "coffee", "meal"]):
            kind = "meal"
        elif any(keyword in lower for keyword in ["walk", "hike", "travel", "commute", "visit", "museum"]):
            kind = "outing"
        elif any(keyword in lower for keyword in ["habit", "routine", "daily", "weekly", "recurring"]):
            kind = "habit"
        priority = 40 + (index * 5)
        if fixed_time:
            priority += 30
        if memory_preferences.get("flags", {}).get("recurring_habit") and kind == "habit":
            priority += 20
        if memory_preferences.get("flags", {}).get("focus_block") and kind == "focus":
            priority += 15

        block = {
            "id": f"block-{index + 1}",
            "segment": segment,
            "title": _infer_activity_title(segment),
            "kind": kind,
            "date": target_date,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": max(30, (_time_to_minutes(end_time) or 0) - (_time_to_minutes(start_time) or 0)),
            "buffer_minutes": DEFAULT_BUFFER_MINUTES,
            "travel_minutes": 0,
            "location": "local area",
            "origin": "",
            "destination": _infer_activity_title(segment),
            "weather_sensitive": _infer_weather_sensitive(segment),
            "flexible": flexible,
            "fixed_time": fixed_time,
            "priority": priority,
            "depends_on": [f"block-{index}"] if index > 0 else [],
            "weather_goal": "good_weather" if _infer_weather_sensitive(segment) else "",
            "notes": [],
        }
        blocks.append(block)

    for index, block in enumerate(blocks):
        if index == 0:
            continue
        prev = blocks[index - 1]
        block["depends_on"] = [prev["id"]]
        block_start = _time_to_minutes(str(block.get("start_time") or "")) or 0
        prev_end = _time_to_minutes(str(prev.get("end_time") or "")) or 0
        if block_start < prev_end + DEFAULT_BUFFER_MINUTES:
            suggested_start = prev_end + DEFAULT_BUFFER_MINUTES
            block["start_time"] = _minutes_to_time(suggested_start)
            block["end_time"] = _minutes_to_time(suggested_start + max(30, block.get("duration_minutes") or 60))
            block["notes"].append("sequentially-adjusted")

    for block in blocks:
        block["priority"] = _priority_for_block(block, memory_preferences)

    return blocks


def _build_task_queue_from_blocks(blocks: list[dict[str, Any]]) -> list[TaskItem]:
    task_queue: list[TaskItem] = []
    for block in blocks:
        if block.get("weather_sensitive") or block.get("weather_goal"):
            task_queue.append(
                {
                    "id": f"weather-{block['id']}",
                    "tool": "weather_mock",
                    "action": "forecast",
                    "date": _clean_text(block.get("date"), 20),
                    "location": _clean_text(block.get("location"), 120) or "local area",
                    "description": f"Check weather for {block.get('title', 'planned item')}",
                }
            )
        task_queue.append(
            {
                "id": f"calendar-check-{block['id']}",
                "tool": "calendar_mock",
                "action": "check_availability",
                "date": _clean_text(block.get("date"), 20),
                "start_time": _clean_text(block.get("start_time"), 5),
                "end_time": _clean_text(block.get("end_time"), 5),
                "title": _clean_text(block.get("title"), 120),
            }
        )
        task_queue.append(
            {
                "id": f"calendar-write-{block['id']}",
                "tool": "calendar_mock",
                "action": "write_event",
                "date": _clean_text(block.get("date"), 20),
                "start_time": _clean_text(block.get("start_time"), 5),
                "end_time": _clean_text(block.get("end_time"), 5),
                "title": _clean_text(block.get("title"), 120),
                "description": _clean_text(block.get("segment"), 240),
                "origin": _clean_text(block.get("origin"), 120),
                "destination": _clean_text(block.get("destination"), 120),
                "color": "#5B7FBF",
            }
        )
    return task_queue


def _collect_existing_events(context: dict[str, Any], date: str) -> list[dict[str, Any]]:
    events = context.get("existingEvents")
    if not isinstance(events, list):
        return []
    matched: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if _clean_text(event.get("date"), 20) != date:
            continue
        matched.append(
            {
                "date": date,
                "start_time": _clean_text(event.get("startTime") or event.get("start_time"), 5),
                "end_time": _clean_text(event.get("endTime") or event.get("end_time"), 5),
                "title": _clean_text(event.get("title"), 120),
            }
        )
    return matched


def _time_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


def _find_conflicting_events(
    *,
    date: str,
    start_time: str,
    end_time: str,
    title: str,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    start_minutes = _time_to_minutes(start_time)
    end_minutes = _time_to_minutes(end_time)
    if start_minutes is None or end_minutes is None:
        return []

    conflicts: list[dict[str, Any]] = []
    for source in (MockCalendarTool.seed_events, _collect_existing_events(context, date)):
        for event in source:
            if not isinstance(event, dict):
                continue
            if _clean_text(event.get("date"), 20) != date:
                continue
            event_start = _time_to_minutes(_clean_text(event.get("start_time"), 5))
            event_end = _time_to_minutes(_clean_text(event.get("end_time"), 5))
            if event_start is None or event_end is None:
                continue
            if _time_overlaps(start_minutes, end_minutes, event_start, event_end):
                conflicts.append(
                    {
                        "date": date,
                        "start_time": _clean_text(event.get("start_time"), 5),
                        "end_time": _clean_text(event.get("end_time"), 5),
                        "title": _clean_text(event.get("title"), 120),
                    }
                )
    if "busy" in title.lower():
        conflicts.append({"date": date, "title": "busy rule"})
    return conflicts


def _memory_conflicts_for_block(block: dict[str, Any], memory_preferences: dict[str, Any]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    start_minutes = _time_to_minutes(str(block.get("start_time") or ""))
    end_minutes = _time_to_minutes(str(block.get("end_time") or ""))
    if start_minutes is None or end_minutes is None:
        return conflicts

    flags = memory_preferences.get("flags") if isinstance(memory_preferences, dict) else {}
    if not isinstance(flags, dict):
        flags = {}

    if flags.get("avoid_lunch") and _time_overlaps(start_minutes, end_minutes, LUNCH_START_MINUTES, LUNCH_END_MINUTES):
        conflicts.append(
            {
                "kind": "memory",
                "code": "lunch_avoidance",
                "message": "Planned time falls inside the remembered lunch-avoidance window.",
            }
        )

    if flags.get("prefer_mornings") and start_minutes >= 12 * 60:
        conflicts.append(
            {
                "kind": "memory",
                "code": "prefer_mornings",
                "message": "Planned time lands after the remembered morning preference.",
            }
        )

    if flags.get("prefer_afternoons") and start_minutes < 12 * 60:
        conflicts.append(
            {
                "kind": "memory",
                "code": "prefer_afternoons",
                "message": "Planned time lands before the remembered afternoon preference.",
            }
        )

    return conflicts


def _weather_conflicts_for_block(block: dict[str, Any], context: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not block.get("weather_sensitive"):
        return [], {}
    report = MockWeatherTool.forecast(
        date=_clean_text(block.get("date"), 20),
        location=_clean_text(block.get("location"), 120) or "local area",
    )
    condition = str(report.get("condition", "")).lower()
    if condition in GOOD_WEATHER_CONDITIONS:
        return [], report
    if any(keyword in condition for keyword in BAD_WEATHER_KEYWORDS) or condition not in GOOD_WEATHER_CONDITIONS:
        return (
            [
                {
                    "kind": "weather",
                    "code": "bad_weather",
                    "message": f"Weather is {report.get('condition')} for the requested outdoor block.",
                    "report": report,
                }
            ],
            report,
        )
    return [], report


def _score_candidate_start(
    *,
    candidate_start: int,
    block: dict[str, Any],
    memory_preferences: dict[str, Any],
) -> int:
    score = candidate_start
    flags = memory_preferences.get("flags") if isinstance(memory_preferences, dict) else {}
    if not isinstance(flags, dict):
        flags = {}
    if flags.get("avoid_lunch") and _time_overlaps(candidate_start, candidate_start + (block.get("duration_minutes") or 60), LUNCH_START_MINUTES, LUNCH_END_MINUTES):
        score += 500
    if flags.get("prefer_mornings"):
        score += max(0, candidate_start - (10 * 60))
    if flags.get("prefer_afternoons"):
        score += max(0, (12 * 60) - candidate_start)
    if block.get("weather_sensitive") and candidate_start >= 18 * 60:
        score += 100
    return score


def _find_best_slot(
    block: dict[str, Any],
    *,
    context: dict[str, Any],
    memory_preferences: dict[str, Any],
    allow_date_shift: bool = False,
) -> dict[str, Any] | None:
    day_start = int(context.get("dayStartHour") or 7) * 60
    day_end = int(context.get("dayEndHour") or 22) * 60
    duration = max(30, int(block.get("duration_minutes") or 60))
    current_date = _clean_text(block.get("date"), 20)
    visible_days = int(context.get("visibleDays") or 1)
    start_candidates: list[tuple[str, int]] = []

    base_start = _time_to_minutes(str(block.get("start_time") or "")) or (day_start + 120)
    for offset in range(0, max(1, visible_days)):
        candidate_date = current_date
        if offset > 0:
            try:
                candidate_date = (date_cls.fromisoformat(current_date) + timedelta(days=offset)).isoformat()
            except Exception:
                candidate_date = current_date
        if offset > 0 and not allow_date_shift:
            continue
        for candidate_start in range(day_start, max(day_start, day_end - duration) + 1, 30):
            if offset == 0 and candidate_start < base_start and block.get("fixed_time"):
                continue
            start_candidates.append((candidate_date, candidate_start))

    best: dict[str, Any] | None = None
    for candidate_date, candidate_start in start_candidates:
        candidate_end = candidate_start + duration
        if candidate_end > day_end:
            continue
        if _find_conflicting_events(
            date=candidate_date,
            start_time=_minutes_to_time(candidate_start),
            end_time=_minutes_to_time(candidate_end),
                title=str(block.get("title") or ""),
                context=context,
        ):
            continue
        if block.get("weather_sensitive"):
            candidate_weather_issues, _ = _weather_conflicts_for_block(
                {
                    **block,
                    "date": candidate_date,
                    "start_time": _minutes_to_time(candidate_start),
                    "end_time": _minutes_to_time(candidate_end),
                },
                context,
            )
            if candidate_weather_issues:
                continue
        if _memory_conflicts_for_block(
            {
                **block,
                "date": candidate_date,
                "start_time": _minutes_to_time(candidate_start),
                "end_time": _minutes_to_time(candidate_end),
            },
            memory_preferences,
        ):
            continue
        candidate = {
            **block,
            "date": candidate_date,
            "start_time": _minutes_to_time(candidate_start),
            "end_time": _minutes_to_time(candidate_end),
        }
        if best is None or _score_candidate_start(candidate_start=candidate_start, block=block, memory_preferences=memory_preferences) < _score_candidate_start(candidate_start=_time_to_minutes(str(best["start_time"])) or candidate_start, block=block, memory_preferences=memory_preferences):
            best = candidate
    return best


def _clarification_options_from_blocks(blocks: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    if not blocks:
        return [
            {
                "label": "Keep current request",
                "date": _iso_date(context.get("calendarStartDate")) or date_cls.today().isoformat(),
                "start_time": "10:00",
                "end_time": "11:00",
                "description": "I need a little more detail to schedule this safely.",
            },
            {
                "label": "Move to the next open slot",
                "date": _iso_date(context.get("calendarStartDate")) or date_cls.today().isoformat(),
                "start_time": "14:00",
                "end_time": "15:00",
                "description": "I can also shift this later if that works better.",
            },
        ]

    first = blocks[0]
    last = blocks[-1]
    return [
        {
            "label": "Preserve the earliest feasible block",
            "date": _clean_text(first.get("date"), 20),
            "start_time": _clean_text(first.get("start_time"), 5),
            "end_time": _clean_text(first.get("end_time"), 5),
            "description": f"Keep {first.get('title', 'the first item')} at its current slot and defer lower-priority work.",
        },
        {
            "label": "Shift the plan later",
            "date": _clean_text(last.get("date"), 20),
            "start_time": _clean_text(last.get("start_time"), 5),
            "end_time": _clean_text(last.get("end_time"), 5),
            "description": f"Move {last.get('title', 'the remaining item')} to the next feasible window.",
        },
    ]


def _compose_clarification_message(blocks: list[dict[str, Any]], options: list[dict[str, Any]]) -> str:
    if not options:
        return "I need a little more detail to schedule this safely."
    lines = ["I found a conflict that I cannot resolve automatically. Choose one:"]
    for index, option in enumerate(options[:2], start=1):
        lines.append(
            f"{index}. {option.get('label', 'Option')} on {option.get('date', '')} from {option.get('start_time', '')} to {option.get('end_time', '')}: {option.get('description', '')}"
        )
    return "\n".join(lines)


def _build_system_prompt(context: dict[str, Any]) -> str:
    existing_events = context.get("existingEvents") or []
    relative_day_hints = context.get("relativeDayHints") or []
    memory_summary = context.get("retrievedMemory") or {}
    memory_text = memory_summary.get("summary_text") or "none"

    return "\n".join(
        [
            "<|think|>",
            "You are the local planning core for a calendar assistant.",
            "Think privately, then emit one valid JSON object only.",
            "Do not reveal your reasoning outside the <|think|> block.",
            "The JSON object must have these keys:",
            '{"status":"ready|needs_clarification","assistantMessage":"string","task_queue":[...],"schedule_draft":{...},"decomposition":{...}}',
            "Use task_queue entries with tool names calendar_mock or weather_mock.",
            "If the request includes multiple intents, break them into atomic scheduling blocks with explicit dates, windows, durations, priorities, and dependencies.",
            "When the user request is incomplete, set status to needs_clarification and keep task_queue empty.",
            f"Calendar window: {context.get('calendarStartDate')} to {context.get('calendarEndDate')}",
            f"Visible days: {context.get('visibleDays')}",
            f"Hours: {context.get('dayStartHour')}:00-{context.get('dayEndHour')}:00",
            f"Timezone: {context.get('timezone')}",
            f"Stable user context from prior runs: {memory_text}",
            f"Existing events snapshot: {existing_events}",
            f"Relative day hints: {relative_day_hints}",
            "Use this memory only as additive context. Never let it override explicit instructions in the current request.",
            "Prefer this task order when ready: calendar availability check, weather check, calendar write.",
            "Use exact ISO dates and 24-hour times.",
        ]
    )


def _log_event(context: dict[str, Any], event: str, payload: dict[str, Any], *, run_id: str) -> None:
    logger = context.get("event_logger")
    if isinstance(logger, SessionEventLogger):
        logger.append(event, payload, run_id=run_id)


def _build_fallback_plan(state: AgentState, context: dict[str, Any]) -> dict[str, Any]:
    user_text = _latest_user_message(state.get("messages", []))
    day_hint = _extract_date_hint(user_text)
    calendar_start = _iso_date(context.get("calendarStartDate")) or _iso_date(context.get("today"))
    if not calendar_start:
        calendar_start = date_cls.today().isoformat()

    import re

    date_index = 1
    if day_hint.startswith("day_"):
        try:
            date_index = max(1, int(day_hint.split("_", 1)[1]))
        except Exception:
            date_index = 1

    target_date = calendar_start
    if date_index > 1:
        target_date = (date_cls.fromisoformat(calendar_start) + timedelta(days=date_index - 1)).isoformat()

    wants_weather = "weather" in user_text.lower()
    wants_place = re.search(r"\b(museum|park|restaurant|library|airport|station)\b", user_text.lower())
    title = "Planned outing"
    if wants_place:
        title = f"{wants_place.group(1).title()} visit"

    start_time = "11:00"
    if "before lunch" in user_text.lower():
        start_time = "10:00"
    elif "afternoon" in user_text.lower():
        start_time = "14:00"

    end_time = "12:00" if start_time == "11:00" else ("11:00" if start_time == "10:00" else "15:00")
    assistant = "I need a little more detail to schedule this safely."
    task_queue: list[TaskItem] = []

    if "?" in user_text and not day_hint:
        return {
            "status": "needs_clarification",
            "assistantMessage": assistant,
            "task_queue": [],
            "schedule_draft": {
                "status": "needs_clarification",
                "assistantMessage": assistant,
                "tasks": [],
                "events": [],
                "calendar_checks": [],
                "weather": [],
                "notes": ["fallback clarification"],
            },
        }

    if wants_weather:
        task_queue.append(
            {
                "id": "weather-1",
                "tool": "weather_mock",
                "action": "forecast",
                "date": target_date,
                "location": "local area",
            }
        )

    task_queue.append(
        {
            "id": "calendar-check-1",
            "tool": "calendar_mock",
            "action": "check_availability",
            "date": target_date,
            "start_time": start_time,
            "end_time": end_time,
            "title": title,
        }
    )
    task_queue.append(
        {
            "id": "calendar-write-1",
            "tool": "calendar_mock",
            "action": "write_event",
            "date": target_date,
            "start_time": start_time,
            "end_time": end_time,
            "title": title,
            "description": user_text[:240],
            "origin": "local start",
            "destination": title,
            "color": "#5B7FBF",
        }
    )

    return {
        "status": "ready",
        "assistantMessage": f"Drafted a plan for {title.lower()} on {target_date}.",
        "task_queue": task_queue,
        "schedule_draft": {
            "status": "ready",
            "assistantMessage": f"Drafted a plan for {title.lower()} on {target_date}.",
            "tasks": task_queue,
            "events": [],
            "calendar_checks": [],
            "weather": [],
            "notes": ["fallback plan"],
        },
    }


def planner_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    messages = state.get("messages", [])
    run_id = _clean_text(state.get("run_id"), 128)
    system_prompt = _build_system_prompt(context)
    response_text = client.chat(
        [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
    )
    sanitized_output = _strip_think_blocks(response_text)
    next_state: AgentState = deepcopy(state)
    next_state["raw_model_output"] = sanitized_output
    _log_event(
        context,
        "model_output_summary",
        {
            "chars": len(response_text),
            "sanitized_chars": len(sanitized_output),
            "has_think_block": "<|think|>" in response_text.lower(),
            "has_json_object": "{" in sanitized_output and "}" in sanitized_output,
        },
        run_id=run_id,
    )
    return next_state


def parse_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    run_id = _clean_text(state.get("run_id"), 128)
    raw_output = state.get("raw_model_output", "")
    parsed = _extract_json_object(raw_output)
    parsed_from_fallback = False
    if not parsed:
        parsed = _build_fallback_plan(state, context)
        parsed_from_fallback = True

    task_queue = parsed.get("task_queue")
    if not isinstance(task_queue, list):
        task_queue = []

    schedule_draft = parsed.get("schedule_draft")
    if not isinstance(schedule_draft, dict):
        schedule_draft = {}

    status = parsed.get("status", "needs_clarification")
    assistant_message = _clean_text(parsed.get("assistantMessage"), 400) or "Need a little more detail."

    next_state: AgentState = deepcopy(state)
    next_state["planner_payload"] = parsed
    next_state["task_queue"] = [task for task in task_queue if isinstance(task, dict)]
    next_state["decomposition"] = parsed.get("decomposition") if isinstance(parsed.get("decomposition"), dict) else {}
    next_state["schedule_draft"] = {
        "status": status,
        "assistantMessage": assistant_message,
        "tasks": next_state["task_queue"],
        "events": list(schedule_draft.get("events", []) if isinstance(schedule_draft.get("events"), list) else []),
        "calendar_checks": list(schedule_draft.get("calendar_checks", []) if isinstance(schedule_draft.get("calendar_checks"), list) else []),
        "weather": list(schedule_draft.get("weather", []) if isinstance(schedule_draft.get("weather"), list) else []),
        "notes": list(schedule_draft.get("notes", []) if isinstance(schedule_draft.get("notes"), list) else []),
        "blocks": list(schedule_draft.get("blocks", []) if isinstance(schedule_draft.get("blocks"), list) else []),
        "validation": schedule_draft.get("validation") if isinstance(schedule_draft.get("validation"), dict) else {},
        "clarification_options": list(
            schedule_draft.get("clarification_options", [])
            if isinstance(schedule_draft.get("clarification_options"), list)
            else []
        ),
        "resolution_path": list(schedule_draft.get("resolution_path", []) if isinstance(schedule_draft.get("resolution_path"), list) else []),
    }
    next_state["conflict_history"] = list(state.get("conflict_history") or [])
    next_state["replan_attempts"] = int(state.get("replan_attempts") or 0)
    next_state["latest_resolution_path"] = list(state.get("latest_resolution_path") or ["planner", "parse"])
    next_state["clarification_options"] = list(state.get("clarification_options") or [])

    _log_event(
        context,
        "parsed_planner_payload_summary",
        {
            "status": status,
            "task_count": len(next_state["task_queue"]),
            "event_count": len(next_state["schedule_draft"].get("events", [])),
            "note_count": len(next_state["schedule_draft"].get("notes", [])),
            "assistant_message": assistant_message,
            "used_fallback": parsed_from_fallback,
        },
        run_id=run_id,
    )

    transition = "planned" if status == "ready" else "clarification"
    _log_event(
        context,
        "state_transition",
        {
            "state": transition,
            "status": status,
            "task_count": len(next_state["task_queue"]),
        },
        run_id=run_id,
    )
    return next_state


def decompose_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    run_id = _clean_text(state.get("run_id"), 128)
    blocks = _build_decomposition_blocks(state, context)
    task_queue = _build_task_queue_from_blocks(blocks)
    draft = deepcopy(state.get("schedule_draft") or {})
    draft.update(
        {
            "status": draft.get("status", "ready"),
            "tasks": task_queue,
            "blocks": blocks,
            "calendar_checks": [],
            "weather": [],
            "events": list(draft.get("events") or []),
            "notes": list(draft.get("notes") or []),
            "validation": {},
            "clarification_options": list(draft.get("clarification_options") or []),
            "resolution_path": list(draft.get("resolution_path") or []) + ["decompose"],
        }
    )
    next_state: AgentState = deepcopy(state)
    next_state["task_queue"] = task_queue
    next_state["schedule_draft"] = draft
    next_state["decomposition"] = {
        "status": "ready",
        "blocks": blocks,
        "task_count": len(task_queue),
        "segment_count": len(blocks),
    }
    next_state["latest_resolution_path"] = list(next_state.get("latest_resolution_path") or []) + ["decompose"]

    _log_event(
        context,
        "decomposition_summary",
        {
            "block_count": len(blocks),
            "task_count": len(task_queue),
            "titles": [block.get("title", "") for block in blocks],
        },
        run_id=run_id,
    )
    return next_state


def validate_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    run_id = _clean_text(state.get("run_id"), 128)
    draft = deepcopy(state.get("schedule_draft") or {})
    blocks = list(draft.get("blocks") or [])
    memory_preferences = _extract_memory_preferences(context.get("retrievedMemory") or {})
    calendar_checks: list[dict[str, Any]] = []
    weather_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for block in blocks:
        weather_issues, weather_report = _weather_conflicts_for_block(block, context)
        if weather_report:
            weather_rows.append(weather_report)
        if weather_issues:
            issues.extend(
                {
                    **issue,
                    "task_id": f"calendar-write-{block.get('id')}",
                    "block_id": block.get("id"),
                    "priority": block.get("priority", 0),
                }
                for issue in weather_issues
            )
        check_result = MockCalendarTool.check_availability(
            date=_clean_text(block.get("date"), 20),
            start_time=_clean_text(block.get("start_time"), 5),
            end_time=_clean_text(block.get("end_time"), 5),
            title=_clean_text(block.get("title"), 120),
        )
        calendar_checks.append(
            {
                "task_id": f"calendar-check-{block.get('id')}",
                "status": check_result.status,
                "reason": check_result.reason,
                "conflicting_events": check_result.conflicting_events,
                "block_id": block.get("id"),
            }
        )
        if check_result.status != "available":
            issues.append(
                {
                    "kind": "temporal",
                    "code": "calendar_conflict",
                    "message": check_result.reason,
                    "task_id": f"calendar-check-{block.get('id')}",
                    "block_id": block.get("id"),
                    "priority": block.get("priority", 0),
                    "conflicts": check_result.conflicting_events,
                }
            )

        context_conflicts = _find_conflicting_events(
            date=_clean_text(block.get("date"), 20),
            start_time=_clean_text(block.get("start_time"), 5),
            end_time=_clean_text(block.get("end_time"), 5),
            title=_clean_text(block.get("title"), 120),
            context=context,
        )
        if context_conflicts:
            issues.append(
                {
                    "kind": "temporal",
                    "code": "existing_event_conflict",
                    "message": "requested time overlaps an existing calendar event",
                    "task_id": f"calendar-check-{block.get('id')}",
                    "block_id": block.get("id"),
                    "priority": block.get("priority", 0),
                    "conflicts": context_conflicts,
                }
            )

        memory_issues = _memory_conflicts_for_block(block, memory_preferences)
        for issue in memory_issues:
            issues.append(
                {
                    **issue,
                    "task_id": f"calendar-write-{block.get('id')}",
                    "block_id": block.get("id"),
                    "priority": block.get("priority", 0),
                }
            )

        if block.get("depends_on"):
            previous_id = block["depends_on"][0]
            prev = next((item for item in blocks if item.get("id") == previous_id), None)
            if isinstance(prev, dict):
                prev_end = _time_to_minutes(str(prev.get("end_time") or "")) or 0
                block_start = _time_to_minutes(str(block.get("start_time") or "")) or 0
                if block_start < prev_end + DEFAULT_BUFFER_MINUTES:
                    issues.append(
                        {
                            "kind": "buffer",
                            "code": "missing_buffer",
                            "message": f"{block.get('title', 'Next item')} needs {DEFAULT_BUFFER_MINUTES} minutes of buffer after {prev.get('title', 'the prior item')}.",
                            "task_id": f"calendar-write-{block.get('id')}",
                            "block_id": block.get("id"),
                            "priority": block.get("priority", 0),
                        }
                    )

        _log_event(
            context,
            "tool_result_summary",
            {
                "tool": "calendar_mock",
                "action": "check_availability",
                "date": _clean_text(block.get("date"), 20),
                "status": check_result.status,
                "reason": check_result.reason,
                "conflict_count": len(check_result.conflicting_events),
            },
            run_id=run_id,
        )
        if weather_report:
            _log_event(
                context,
                "tool_result_summary",
                {
                    "tool": "weather_mock",
                    "action": "forecast",
                    "date": weather_report.get("date"),
                    "summary": weather_report.get("summary"),
                },
                run_id=run_id,
            )

    conflict_detected = bool(issues)
    needs_replan = conflict_detected and any(
        issue.get("kind") in {"temporal", "buffer", "weather", "memory"} and next(
            (block for block in blocks if block.get("id") == issue.get("block_id")),
            {},
        ).get("flexible", False)
        for issue in issues
    )
    has_unavoidable_conflict = conflict_detected and not needs_replan

    validation_report = {
        "status": "conflict" if conflict_detected else "ready",
        "issues": issues,
        "needs_replan": needs_replan,
        "unavoidable": has_unavoidable_conflict,
        "memory_preferences": memory_preferences,
        "calendar_checks": calendar_checks,
        "weather": weather_rows,
    }

    draft["calendar_checks"] = calendar_checks
    draft["weather"] = weather_rows
    draft["validation"] = validation_report
    draft["notes"] = list(draft.get("notes") or [])
    draft["status"] = "needs_clarification" if has_unavoidable_conflict else "ready"
    if has_unavoidable_conflict:
        draft["clarification_options"] = _clarification_options_from_blocks(blocks, context)
        draft["assistantMessage"] = _compose_clarification_message(blocks, draft["clarification_options"])
    elif issues:
        draft["assistantMessage"] = "I found a conflict that can likely be rebalanced."
    else:
        draft["assistantMessage"] = draft.get("assistantMessage") or "Drafted a plan."

    next_state: AgentState = deepcopy(state)
    next_state["schedule_draft"] = draft
    next_state["validation_report"] = validation_report
    next_state["conflict_history"] = list(state.get("conflict_history") or []) + [validation_report]
    next_state["latest_resolution_path"] = list(next_state.get("latest_resolution_path") or []) + ["validate"]
    next_state["clarification_options"] = list(draft.get("clarification_options") or [])

    _log_event(
        context,
        "validation_summary",
        {
            "status": validation_report["status"],
            "issue_count": len(issues),
            "needs_replan": needs_replan,
            "unavoidable": has_unavoidable_conflict,
        },
        run_id=run_id,
    )
    return next_state


def replan_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    run_id = _clean_text(state.get("run_id"), 128)
    attempts = int(state.get("replan_attempts") or 0)
    limit = int(context.get("replanAttemptLimit") or DEFAULT_REPLAN_LIMIT)
    draft = deepcopy(state.get("schedule_draft") or {})
    blocks = list(draft.get("blocks") or [])
    validation_report = state.get("validation_report") or {}
    issues = list(validation_report.get("issues") or [])
    memory_preferences = _extract_memory_preferences(context.get("retrievedMemory") or {})
    next_state: AgentState = deepcopy(state)

    if attempts >= limit or not issues:
        clarification_options = _clarification_options_from_blocks(blocks, context)
        draft["status"] = "needs_clarification"
        draft["clarification_options"] = clarification_options
        draft["assistantMessage"] = _compose_clarification_message(blocks, clarification_options)
        draft["validation"] = {
            **(draft.get("validation") if isinstance(draft.get("validation"), dict) else {}),
            "replan_attempts": attempts,
            "replan_limit": limit,
        }
        next_state["clarification_options"] = clarification_options
        next_state["schedule_draft"] = draft
        next_state["replan_attempts"] = attempts
        next_state["latest_resolution_path"] = list(next_state.get("latest_resolution_path") or []) + ["clarify"]
        _log_event(
            context,
            "replan_summary",
            {
                "attempts": attempts,
                "limit": limit,
                "status": "clarify",
                "issue_count": len(issues),
            },
            run_id=run_id,
        )
        return next_state

    changed = False
    sorted_blocks = sorted(blocks, key=lambda block: int(block.get("priority") or 0))
    issue_map = {str(issue.get("block_id")): issue for issue in issues if issue.get("block_id")}
    for block in sorted_blocks:
        issue = issue_map.get(str(block.get("id")))
        if not issue and not block.get("flexible"):
            continue
        allow_date_shift = bool(issue and issue.get("kind") == "weather")
        candidate = _find_best_slot(
            block,
            context=context,
            memory_preferences=memory_preferences,
            allow_date_shift=allow_date_shift or bool(block.get("flexible")),
        )
        if candidate:
            if (
                candidate.get("date") != block.get("date")
                or candidate.get("start_time") != block.get("start_time")
                or candidate.get("end_time") != block.get("end_time")
            ):
                block.update(
                    {
                        "date": candidate["date"],
                        "start_time": candidate["start_time"],
                        "end_time": candidate["end_time"],
                    }
                )
                changed = True

    if blocks:
        for index, block in enumerate(blocks):
            if index == 0:
                continue
            prev = blocks[index - 1]
            if block.get("date") == prev.get("date"):
                prev_end = _time_to_minutes(str(prev.get("end_time") or "")) or 0
                desired_start = prev_end + DEFAULT_BUFFER_MINUTES
                current_start = _time_to_minutes(str(block.get("start_time") or "")) or 0
                if current_start < desired_start:
                    duration = max(30, int(block.get("duration_minutes") or 60))
                    block["start_time"] = _minutes_to_time(desired_start)
                    block["end_time"] = _minutes_to_time(desired_start + duration)
                    changed = True

    if not changed:
        clarification_options = _clarification_options_from_blocks(blocks, context)
        draft["status"] = "needs_clarification"
        draft["clarification_options"] = clarification_options
        draft["assistantMessage"] = _compose_clarification_message(blocks, clarification_options)
        next_state["clarification_options"] = clarification_options
        next_state["schedule_draft"] = draft
        next_state["replan_attempts"] = attempts + 1
        next_state["latest_resolution_path"] = list(next_state.get("latest_resolution_path") or []) + ["clarify"]
        _log_event(
            context,
            "replan_summary",
            {
                "attempts": attempts + 1,
                "limit": limit,
                "status": "no-change",
                "issue_count": len(issues),
            },
            run_id=run_id,
        )
        return next_state

    draft["blocks"] = blocks
    draft["tasks"] = _build_task_queue_from_blocks(blocks)
    draft["calendar_checks"] = []
    draft["weather"] = []
    draft["validation"] = {
        "status": "replanned",
        "issues": issues,
        "needs_replan": True,
        "replan_attempts": attempts + 1,
        "replan_limit": limit,
    }
    draft["notes"] = list(draft.get("notes") or []) + ["rebalanced"]
    draft["status"] = "ready"
    draft["assistantMessage"] = "I rebalanced the plan to avoid the conflicts."

    next_state["task_queue"] = draft["tasks"]
    next_state["schedule_draft"] = draft
    next_state["replan_attempts"] = attempts + 1
    next_state["validation_report"] = {
        "status": "replanned",
        "issues": issues,
        "needs_replan": True,
        "unavoidable": False,
        "memory_preferences": memory_preferences,
        "calendar_checks": [],
        "weather": [],
    }
    next_state["latest_resolution_path"] = list(next_state.get("latest_resolution_path") or []) + ["replan"]

    _log_event(
        context,
        "replan_summary",
        {
            "attempts": attempts + 1,
            "limit": limit,
            "status": "rebalanced",
            "issue_count": len(issues),
            "block_titles": [block.get("title", "") for block in blocks],
        },
        run_id=run_id,
    )
    return next_state


def execute_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    run_id = _clean_text(state.get("run_id"), 128)
    draft = deepcopy(state.get("schedule_draft") or {})
    task_queue = list(state.get("task_queue") or [])
    blocks = list(draft.get("blocks") or [])
    events = list(draft.get("events") or [])
    notes = list(draft.get("notes") or [])
    validation = draft.get("validation") if isinstance(draft.get("validation"), dict) else {}
    if draft.get("status") == "needs_clarification":
        next_state: AgentState = deepcopy(state)
        next_state["schedule_draft"] = draft
        return next_state

    for block in blocks:
        check_status = "available"
        if isinstance(validation, dict):
            for check in validation.get("calendar_checks") or []:
                if isinstance(check, dict) and check.get("block_id") == block.get("id"):
                    check_status = str(check.get("status") or "available")
                    break

        if check_status != "available":
            notes.append(f"event-skipped:{block.get('id')}")
            continue

        event = {
            "title": _clean_text(block.get("title"), 120) or "Untitled",
            "description": _clean_text(block.get("segment"), 320),
            "date": _clean_text(block.get("date"), 20),
            "start_time": _clean_text(block.get("start_time"), 5),
            "end_time": _clean_text(block.get("end_time"), 5),
            "origin": _clean_text(block.get("origin"), 120),
            "destination": _clean_text(block.get("destination"), 120),
            "color": "#5B7FBF",
        }
        created = MockCalendarTool.write_event(event)
        events.append(created)
        notes.append(f"event:{created['id']}")
        _log_event(
            context,
            "tool_result_summary",
            {
                "tool": "calendar_mock",
                "action": "write_event",
                "status": "written",
                "event_id": created["id"],
                "title": created["title"],
            },
            run_id=run_id,
        )

    draft["events"] = events
    draft["notes"] = notes
    draft["tasks"] = task_queue
    draft["assistantMessage"] = draft.get("assistantMessage") or "Drafted a plan."
    next_state = deepcopy(state)
    next_state["schedule_draft"] = draft
    next_state["latest_resolution_path"] = list(next_state.get("latest_resolution_path") or []) + ["execute"]
    return next_state


def finalize_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    run_id = _clean_text(state.get("run_id"), 128)
    draft = deepcopy(state.get("schedule_draft") or {})
    response = {
        "status": draft.get("status", "needs_clarification"),
        "assistantMessage": draft.get("assistantMessage", "Need a little more detail."),
        "events": draft.get("events", []),
        "task_queue": state.get("task_queue", []),
        "schedule_draft": draft,
    }
    _log_event(
        context,
        "state_transition",
        {
            "state": "completed",
            "status": response["status"],
            "event_count": len(response["events"]),
        },
        run_id=run_id,
    )
    _log_event(
        context,
        "final_response",
        {
            "status": response["status"],
            "assistant_message": _clean_text(response["assistantMessage"], 400),
            "event_count": len(response["events"]),
            "task_count": len(response["task_queue"]),
        },
        run_id=run_id,
    )
    next_state: AgentState = deepcopy(state)
    next_state["response"] = response
    next_state["latest_resolution_path"] = list(next_state.get("latest_resolution_path") or []) + ["finalize"]
    return next_state


class OllamaClient:
    def __init__(self, *, base_url: str, model: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def chat(self, messages: list[dict[str, str]]) -> str:
        import json
        from urllib import request as urlrequest
        from urllib.error import HTTPError, URLError

        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "messages": messages,
                "options": {"temperature": 0.1},
            }
        ).encode("utf-8")
        req = urlrequest.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
            data = json.loads(body)
            content = data.get("message", {}).get("content", "")
            return str(content)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
            return ""


class LangGraphRuntime:
    def __init__(self, graph: Any, *, thread_id: str) -> None:
        self.graph = graph
        self.thread_id = thread_id

    def invoke(self, state: AgentState) -> AgentState:
        return self.graph.invoke(state, config={"configurable": {"thread_id": self.thread_id}})


class GraphRunner:
    def __init__(
        self,
        nodes: list[tuple[str, Callable[[AgentState], AgentState]]],
        *,
        checkpoint_store: SQLitePlannerCheckpointStore | None,
        calendar_id: str,
        run_id: str,
        replan_limit: int,
    ) -> None:
        self.nodes = nodes
        self.checkpoint_store = checkpoint_store
        self.calendar_id = calendar_id
        self.run_id = run_id
        self.replan_limit = replan_limit

    def _save_checkpoint(self, step_name: str, state: AgentState) -> None:
        if not self.checkpoint_store:
            return
        snapshot = deepcopy(state)
        snapshot["raw_model_output"] = _strip_think_blocks(snapshot.get("raw_model_output", ""))
        self.checkpoint_store.save(
            calendar_id=self.calendar_id,
            run_id=self.run_id,
            step_name=step_name,
            state=snapshot,
            created_at=utc_now_iso(),
        )

    def invoke(self, state: AgentState) -> AgentState:
        current: AgentState = deepcopy(state)
        self._save_checkpoint("received", current)
        step_map = {name: node for name, node in self.nodes}
        order = [name for name, _ in self.nodes]
        step_index = 0
        while step_index < len(order):
            step_name = order[step_index]
            node = step_map[step_name]
            current = node(current)
            self._save_checkpoint(step_name, current)

            if step_name == "validate":
                draft = current.get("schedule_draft") or {}
                report = current.get("validation_report") or {}
                if draft.get("status") == "needs_clarification" or report.get("unavoidable"):
                    step_index = order.index("finalize")
                    continue
                if report.get("needs_replan"):
                    step_index = order.index("replan")
                    continue
                step_index = order.index("execute")
                continue

            if step_name == "replan":
                draft = current.get("schedule_draft") or {}
                if draft.get("status") == "needs_clarification":
                    step_index = order.index("finalize")
                else:
                    step_index = order.index("validate")
                continue

            if step_name == "execute":
                step_index = order.index("finalize")
                continue

            if step_name == "finalize":
                break

            step_index += 1
        return current


def _build_langgraph_checkpointer(checkpoint_db_path: str) -> Any | None:
    if SqliteSaver is None:
        return None
    try:
        conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:
        try:
            return SqliteSaver.from_conn_string(checkpoint_db_path)
        except Exception:
            return None


def build_graph_runner(
    context: dict[str, Any],
    client: Any,
    *,
    checkpoint_store: SQLitePlannerCheckpointStore | None = None,
    thread_id: str | None = None,
) -> Any:
    def planner_wrapper(inner_state: AgentState) -> AgentState:
        return planner_node(inner_state, context, client)

    def parse_wrapper(inner_state: AgentState) -> AgentState:
        return parse_node(inner_state, context, client)

    def decompose_wrapper(inner_state: AgentState) -> AgentState:
        return decompose_node(inner_state, context, client)

    def validate_wrapper(inner_state: AgentState) -> AgentState:
        return validate_node(inner_state, context, client)

    def replan_wrapper(inner_state: AgentState) -> AgentState:
        return replan_node(inner_state, context, client)

    def execute_wrapper(inner_state: AgentState) -> AgentState:
        return execute_node(inner_state, context, client)

    def finalize_wrapper(inner_state: AgentState) -> AgentState:
        return finalize_node(inner_state, context, client)

    if StateGraph is not None:
        graph = StateGraph(AgentState)
        graph.add_node("planner", planner_wrapper)
        graph.add_node("parse", parse_wrapper)
        graph.add_node("decompose", decompose_wrapper)
        graph.add_node("validate", validate_wrapper)
        graph.add_node("replan", replan_wrapper)
        graph.add_node("execute", execute_wrapper)
        graph.add_node("finalize", finalize_wrapper)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "parse")
        graph.add_edge("parse", "decompose")
        graph.add_edge("decompose", "validate")
        graph.add_conditional_edges(
            "validate",
            lambda inner_state: (
                "finalize"
                if (inner_state.get("schedule_draft") or {}).get("status") == "needs_clarification"
                or (inner_state.get("validation_report") or {}).get("unavoidable")
                else ("replan" if (inner_state.get("validation_report") or {}).get("needs_replan") else "execute")
            ),
            {
                "replan": "replan",
                "execute": "execute",
                "finalize": "finalize",
            },
        )
        graph.add_conditional_edges(
            "replan",
            lambda inner_state: "finalize"
            if (inner_state.get("schedule_draft") or {}).get("status") == "needs_clarification"
            else "validate",
            {
                "validate": "validate",
                "finalize": "finalize",
            },
        )
        graph.add_edge("execute", "finalize")
        graph.add_edge("finalize", END)
        checkpointer = _build_langgraph_checkpointer(context["checkpoint_db_path"])
        if checkpointer is not None and thread_id:
            return LangGraphRuntime(graph.compile(checkpointer=checkpointer), thread_id=thread_id)
        return LangGraphRuntime(graph.compile(), thread_id=thread_id or context["calendar_id"])

    return GraphRunner(
        [
            ("planner", planner_wrapper),
            ("parse", parse_wrapper),
            ("decompose", decompose_wrapper),
            ("validate", validate_wrapper),
            ("replan", replan_wrapper),
            ("execute", execute_wrapper),
            ("finalize", finalize_wrapper),
        ],
        checkpoint_store=checkpoint_store,
        calendar_id=context["calendar_id"],
        run_id=context["run_id"],
        replan_limit=int(context.get("replanAttemptLimit") or DEFAULT_REPLAN_LIMIT),
    )


def _normalize_calendar_id(payload: dict[str, Any]) -> str:
    calendar_id = _clean_text(payload.get("calendarId"), 128)
    if calendar_id:
        return calendar_id
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    calendar_id = _clean_text(context.get("calendarId"), 128)
    return calendar_id or "default-calendar"


def run_planner_graph(
    payload: dict[str, Any],
    *,
    ollama_url: str,
    model: str,
    timeout_seconds: float,
    data_root: str = "./data",
) -> AgentState:
    calendar_id = _normalize_calendar_id(payload)
    paths = build_planner_paths(data_root, calendar_id)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.sessions_dir.mkdir(parents=True, exist_ok=True)
    paths.exports_dir.mkdir(parents=True, exist_ok=True)

    normalized_messages = _normalize_messages(payload.get("messages"))
    session_id = calendar_id
    event_logger = SessionEventLogger(paths.session_log_path, calendar_id, session_id)
    run_index = event_logger.count_events("planner_request_received") + 1
    run_id = f"{calendar_id}-run-{run_index:04d}"
    prior_events = read_recent_session_events(paths.session_log_path, limit=8)
    recent_context = tail_context_from_events(prior_events, limit=8)

    context_payload = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    calendar_window = {
        "start": context_payload.get("calendarStartDate"),
        "end": context_payload.get("calendarEndDate"),
        "visible_days": context_payload.get("visibleDays"),
        "day_start_hour": context_payload.get("dayStartHour"),
        "day_end_hour": context_payload.get("dayEndHour"),
        "timezone": context_payload.get("timezone"),
    }

    event_logger.append(
        "planner_request_received",
        {
            "run_index": run_index,
            "run_id": run_id,
            "calendar_window": calendar_window,
            "message_count": len(normalized_messages),
            "latest_user_message": _clean_text(_latest_user_message(normalized_messages), 240),
            "messages": _compact_message_batch(normalized_messages),
        },
        run_id=run_id,
    )
    event_logger.append(
        "normalized_user_message_batch",
        {
            "run_index": run_index,
            "messages": _compact_message_batch(normalized_messages),
        },
        run_id=run_id,
    )

    latest_user_message = sanitize_query_text(_latest_user_message(normalized_messages))

    memory_store = LongTermMemoryStore(paths.data_root / "chromadb", ollama_url=ollama_url)
    retrieved_memory = memory_store.query(
        calendar_id=calendar_id,
        query_text=latest_user_message,
        recent_context=recent_context,
    )
    event_logger.append(
        "retrieved_memory_summary",
        {
            "run_index": run_index,
            "retrieved_memory": retrieved_memory,
            "summary_text": retrieved_memory.get("summary_text", ""),
        },
        run_id=run_id,
    )

    client = OllamaClient(base_url=ollama_url, model=model, timeout_seconds=timeout_seconds)
    checkpoint_store = SQLitePlannerCheckpointStore(paths.checkpoint_db_path)
    context = {
        "calendar_id": calendar_id,
        "session_id": session_id,
        "run_id": run_id,
        "event_logger": event_logger,
        "checkpoint_db_path": str(paths.checkpoint_db_path),
        "event_log_path": str(paths.session_log_path),
        "retrievedMemory": retrieved_memory,
        "calendarStartDate": context_payload.get("calendarStartDate"),
        "calendarEndDate": context_payload.get("calendarEndDate"),
        "visibleDays": context_payload.get("visibleDays"),
        "dayStartHour": context_payload.get("dayStartHour"),
        "dayEndHour": context_payload.get("dayEndHour"),
        "timezone": context_payload.get("timezone"),
        "existingEvents": context_payload.get("existingEvents") or [],
        "relativeDayHints": context_payload.get("relativeDayHints") or [],
        "replanAttemptLimit": DEFAULT_REPLAN_LIMIT,
    }

    state: AgentState = {
        "calendar_id": calendar_id,
        "session_id": session_id,
        "run_id": run_id,
        "event_log_path": str(paths.session_log_path),
        "retrieved_memory": retrieved_memory,
        "distilled_facts": [],
        "messages": normalized_messages,
        "task_queue": [],
        "decomposition": {},
        "validation_report": {},
        "conflict_history": [],
        "replan_attempts": 0,
        "latest_resolution_path": [],
        "clarification_options": [],
        "schedule_draft": {
            "status": "needs_clarification",
            "assistantMessage": "Need a little more detail.",
            "tasks": [],
            "events": [],
            "calendar_checks": [],
            "weather": [],
            "notes": [],
            "blocks": [],
            "validation": {},
            "clarification_options": [],
            "resolution_path": [],
        },
    }

    runner = build_graph_runner(
        context,
        client,
        checkpoint_store=checkpoint_store,
        thread_id=calendar_id,
    )
    final_state = runner.invoke(state)

    response = final_state.get("response") or {
        "status": "needs_clarification",
        "assistantMessage": "Need a little more detail.",
        "events": [],
        "task_queue": [],
        "schedule_draft": final_state.get("schedule_draft", {}),
    }
    final_state["response"] = response

    checkpoint_store.save(
        calendar_id=calendar_id,
        run_id=run_id,
        step_name="final",
        state=deepcopy(final_state),
        created_at=utc_now_iso(),
    )

    if response.get("status") == "ready":
        distillation = distill_session_log(
            session_log_path=str(paths.session_log_path),
            calendar_id=calendar_id,
            session_id=session_id,
        )
        if distillation.facts:
            memory_store.upsert_facts(distillation.facts)
        final_state["distilled_facts"] = [
            {
                "category": fact.category,
                "normalized_value": fact.normalized_value,
                "confidence": fact.confidence,
                "document_id": fact.document_id,
            }
            for fact in distillation.facts
        ]
        event_logger.append(
            "distillation_results",
            {
                "run_index": run_index,
                "summary": distillation.summary,
                "upserted_facts": final_state["distilled_facts"],
                "skipped": distillation.skipped,
            },
            run_id=run_id,
        )

    return final_state
