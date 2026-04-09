from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any


def _to_minutes(value: str) -> int | None:
    try:
        hour, minute = value.split(":", 1)
        hour_int = int(hour)
        minute_int = int(minute)
    except Exception:
        return None

    if hour_int < 0 or hour_int > 23 or minute_int < 0 or minute_int > 59:
        return None
    return hour_int * 60 + minute_int


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


def _stable_hash(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


@dataclass(frozen=True)
class CalendarCheckResult:
    status: str
    reason: str
    conflicting_events: list[dict[str, Any]]


class MockCalendarTool:
    """Deterministic local calendar stub."""

    seed_events = [
        {"date": "2026-04-09", "start_time": "09:00", "end_time": "10:00", "title": "Morning hold"},
        {"date": "2026-04-09", "start_time": "13:00", "end_time": "14:00", "title": "Lunch hold"},
        {"date": "2026-04-10", "start_time": "15:00", "end_time": "16:00", "title": "Team sync"},
    ]

    @classmethod
    def check_availability(
        cls,
        *,
        date: str,
        start_time: str,
        end_time: str,
        title: str = "",
    ) -> CalendarCheckResult:
        start_minutes = _to_minutes(start_time)
        end_minutes = _to_minutes(end_time)
        if start_minutes is None or end_minutes is None or end_minutes <= start_minutes:
            return CalendarCheckResult(
                status="invalid",
                reason="invalid time window",
                conflicting_events=[],
            )

        matches: list[dict[str, Any]] = []
        for event in cls.seed_events:
            if event["date"] != date:
                continue
            event_start = _to_minutes(event["start_time"])
            event_end = _to_minutes(event["end_time"])
            if event_start is None or event_end is None:
                continue
            if _overlaps(start_minutes, end_minutes, event_start, event_end):
                matches.append(event)

        if matches:
            return CalendarCheckResult(
                status="conflict",
                reason="requested time overlaps a seeded calendar hold",
                conflicting_events=matches,
            )

        if "busy" in title.lower():
            return CalendarCheckResult(
                status="conflict",
                reason="title flagged as busy by deterministic rule",
                conflicting_events=[],
            )

        return CalendarCheckResult(
            status="available",
            reason="no conflict detected",
            conflicting_events=[],
        )

    @classmethod
    def write_event(cls, event: dict[str, Any]) -> dict[str, Any]:
        title = str(event.get("title", "Untitled")).strip() or "Untitled"
        date = str(event.get("date", "")).strip()
        start_time = str(event.get("start_time", "")).strip()
        end_time = str(event.get("end_time", "")).strip()
        slug = title.lower().replace(" ", "-")[:24] or "event"
        event_id = f"mock-{date}-{start_time.replace(':', '')}-{slug}"
        return {
            "id": event_id,
            "title": title,
            "date": date,
            "startTime": start_time,
            "endTime": end_time,
            "origin": event.get("origin") or "",
            "destination": event.get("destination") or "",
            "color": event.get("color") or "#5B7FBF",
            "written": True,
        }


class MockWeatherTool:
    """Deterministic local weather stub."""

    conditions = ["sunny", "partly cloudy", "cloudy", "windy", "light rain"]

    @classmethod
    def forecast(cls, *, date: str, location: str = "local area") -> dict[str, Any]:
        seed = _stable_hash(date, location or "local area")
        condition = cls.conditions[seed % len(cls.conditions)]
        high = 64 + (seed % 18)
        low = high - (7 + seed % 6)
        humidity = 35 + (seed % 40)
        return {
            "date": date,
            "location": location or "local area",
            "condition": condition,
            "high_f": high,
            "low_f": low,
            "humidity_pct": humidity,
            "summary": f"{condition} with a high of {high}F and a low of {low}F.",
        }
