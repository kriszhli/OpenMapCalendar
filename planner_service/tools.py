from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
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
    """Deterministic local calendar helper."""

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
    """Deterministic local weather helper."""

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


def _stable_coordinate(text: str, *, axis: str) -> float:
    digest = hashlib.sha256(f"{axis}::{text}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:4], "big")
    if axis == "lat":
        return (raw / 2**32) * 140.0 - 70.0
    return (raw / 2**32) * 360.0 - 180.0


def _normalize_place_name(value: str) -> str:
    text = " ".join(part for part in value.replace("_", " ").split() if part)
    return text.strip().title() or "Local Area"


def _haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    radius_km = 6371.0
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    d_lat = math.radians(b_lat - a_lat)
    d_lng = math.radians(b_lng - a_lng)
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    return 2 * radius_km * math.atan2(math.sqrt(h), math.sqrt(1 - h))


class MockLocationTool:
    """Deterministic local place normalizer."""

    @classmethod
    def normalize_place(
        cls,
        query: str,
        *,
        focus: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        name = _normalize_place_name(query or "Local Area")
        lat = _stable_coordinate(name, axis="lat")
        lng = _stable_coordinate(name, axis="lng")
        if focus and isinstance(focus.get("lat"), (int, float)) and isinstance(focus.get("lng"), (int, float)):
            lat = (lat + float(focus["lat"])) / 2.0
            lng = (lng + float(focus["lng"])) / 2.0
        return {
            "query": query,
            "name": name,
            "display_name": name,
            "lat": lat,
            "lng": lng,
            "source": "mock",
        }


class MockRouteTool:
    """Deterministic local route estimator."""

    @classmethod
    def estimate_route(
        cls,
        *,
        origin: str,
        destination: str,
        mode: str = "driving",
        focus: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        origin_place = MockLocationTool.normalize_place(origin or "Origin", focus=focus)
        destination_place = MockLocationTool.normalize_place(destination or "Destination", focus=focus)
        distance_km = _haversine_km(
            float(origin_place["lat"]),
            float(origin_place["lng"]),
            float(destination_place["lat"]),
            float(destination_place["lng"]),
        )
        if distance_km < 0.5:
            distance_km = 1.5
        speed_kmh = 32.0 if mode == "driving" else 20.0 if mode == "transit" else 5.0
        duration_minutes = max(5, int(round((distance_km / speed_kmh) * 60.0)))
        return {
            "origin": origin_place,
            "destination": destination_place,
            "mode": mode,
            "distance_km": round(distance_km, 2),
            "duration_minutes": duration_minutes,
            "summary": (
                f"Estimated {duration_minutes} minute {mode} trip from "
                f"{origin_place['name']} to {destination_place['name']}."
            ),
            "source": "mock",
        }
