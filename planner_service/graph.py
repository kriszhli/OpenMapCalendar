from __future__ import annotations

from copy import deepcopy
from datetime import date as date_cls
from typing import Any, Literal, NotRequired, TypedDict

from .tools import MockCalendarTool, MockWeatherTool

try:
    from langgraph.graph import END, StateGraph  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    END = "__end__"
    StateGraph = None


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


class AgentState(TypedDict, total=False):
    messages: list[ChatMessage]
    task_queue: list[TaskItem]
    schedule_draft: ScheduleDraft
    raw_model_output: str
    planner_payload: dict[str, Any]
    response: dict[str, Any]


class GraphRunner:
    def __init__(self, nodes: list[Any]):
        self.nodes = nodes

    def invoke(self, state: AgentState) -> AgentState:
        current: AgentState = deepcopy(state)
        for node in self.nodes:
            current = node(current)
        return current


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


def _build_system_prompt(context: dict[str, Any]) -> str:
    existing_events = context.get("existingEvents") or []
    relative_day_hints = context.get("relativeDayHints") or []

    return "\n".join(
        [
            "<|think|>",
            "You are the local planning core for a calendar assistant.",
            "Think privately, then emit one valid JSON object only.",
            "Do not reveal your reasoning outside the <|think|> block.",
            "The JSON object must have these keys:",
            '{"status":"ready|needs_clarification","assistantMessage":"string","task_queue":[...],"schedule_draft":{...}}',
            "Use task_queue entries with tool names calendar_mock or weather_mock.",
            "When the user request is incomplete, set status to needs_clarification and keep task_queue empty.",
            f"Calendar window: {context.get('calendarStartDate')} to {context.get('calendarEndDate')}",
            f"Visible days: {context.get('visibleDays')}",
            f"Hours: {context.get('dayStartHour')}:00-{context.get('dayEndHour')}:00",
            f"Timezone: {context.get('timezone')}",
            f"Existing events snapshot: {existing_events}",
            f"Relative day hints: {relative_day_hints}",
            "Prefer this task order when ready: calendar availability check, weather check, calendar write.",
            "Use exact ISO dates and 24-hour times.",
        ]
    )


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
        from datetime import timedelta

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
    system_prompt = _build_system_prompt(context)
    response_text = client.chat(
        [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
    )
    next_state: AgentState = deepcopy(state)
    next_state["raw_model_output"] = response_text
    return next_state


def parse_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    raw_output = state.get("raw_model_output", "")
    parsed = _extract_json_object(_strip_think_blocks(raw_output))
    if not parsed:
        parsed = _build_fallback_plan(state, context)

    task_queue = parsed.get("task_queue")
    if not isinstance(task_queue, list):
        task_queue = []

    schedule_draft = parsed.get("schedule_draft")
    if not isinstance(schedule_draft, dict):
        schedule_draft = {}

    next_state: AgentState = deepcopy(state)
    next_state["planner_payload"] = parsed
    next_state["task_queue"] = [task for task in task_queue if isinstance(task, dict)]
    next_state["schedule_draft"] = {
        "status": parsed.get("status", "needs_clarification"),
        "assistantMessage": _clean_text(parsed.get("assistantMessage"), 400) or "Need a little more detail.",
        "tasks": next_state["task_queue"],
        "events": list(schedule_draft.get("events", []) if isinstance(schedule_draft.get("events"), list) else []),
        "calendar_checks": list(schedule_draft.get("calendar_checks", []) if isinstance(schedule_draft.get("calendar_checks"), list) else []),
        "weather": list(schedule_draft.get("weather", []) if isinstance(schedule_draft.get("weather"), list) else []),
        "notes": list(schedule_draft.get("notes", []) if isinstance(schedule_draft.get("notes"), list) else []),
    }
    return next_state


def executor_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    draft = deepcopy(state.get("schedule_draft") or {})
    task_queue = list(state.get("task_queue") or [])
    calendar_checks = list(draft.get("calendar_checks") or [])
    weather_rows = list(draft.get("weather") or [])
    events = list(draft.get("events") or [])
    notes = list(draft.get("notes") or [])

    for task in task_queue:
        tool = task.get("tool")
        action = task.get("action")
        if tool == "weather_mock" and action == "forecast":
            report = MockWeatherTool.forecast(
                date=_clean_text(task.get("date"), 20),
                location=_clean_text(task.get("location"), 120) or "local area",
            )
            weather_rows.append(report)
            notes.append(f"weather:{report['condition']}")
            continue

        if tool == "calendar_mock" and action == "check_availability":
            result = MockCalendarTool.check_availability(
                date=_clean_text(task.get("date"), 20),
                start_time=_clean_text(task.get("start_time"), 5),
                end_time=_clean_text(task.get("end_time"), 5),
                title=_clean_text(task.get("title"), 120),
            )
            calendar_checks.append(
                {
                    "task_id": task.get("id"),
                    "status": result.status,
                    "reason": result.reason,
                    "conflicting_events": result.conflicting_events,
                }
            )
            if result.status != "available":
                notes.append(f"calendar:{result.status}")
            continue

        if tool == "calendar_mock" and action == "write_event":
            check_status = "available"
            if calendar_checks:
                check_status = calendar_checks[-1].get("status", "available")

            event = {
                "title": _clean_text(task.get("title"), 120) or "Untitled",
                "description": _clean_text(task.get("description"), 320),
                "date": _clean_text(task.get("date"), 20),
                "start_time": _clean_text(task.get("start_time"), 5),
                "end_time": _clean_text(task.get("end_time"), 5),
                "origin": _clean_text(task.get("origin"), 120),
                "destination": _clean_text(task.get("destination"), 120),
                "color": _clean_text(task.get("color"), 20) or "#5B7FBF",
            }
            if check_status == "available":
                created = MockCalendarTool.write_event(event)
                events.append(created)
                notes.append(f"event:{created['id']}")
            else:
                notes.append("event-skipped:conflict")
            continue

    draft["calendar_checks"] = calendar_checks
    draft["weather"] = weather_rows
    draft["events"] = events
    draft["notes"] = notes
    if draft.get("status") != "needs_clarification" and not events and calendar_checks:
        if any(check.get("status") != "available" for check in calendar_checks):
            draft["assistantMessage"] = "I found a calendar conflict, so I left the event unscheduled."

    next_state: AgentState = deepcopy(state)
    next_state["schedule_draft"] = draft
    return next_state


def finalize_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    draft = state.get("schedule_draft") or {}
    response = {
        "status": draft.get("status", "needs_clarification"),
        "assistantMessage": draft.get("assistantMessage", "Need a little more detail."),
        "events": draft.get("events", []),
        "task_queue": state.get("task_queue", []),
        "schedule_draft": draft,
    }
    next_state: AgentState = deepcopy(state)
    next_state["response"] = response
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


def build_graph_runner(context: dict[str, Any], client: Any) -> Any:
    def planner_wrapper(inner_state: AgentState) -> AgentState:
        return planner_node(inner_state, context, client)

    def parse_wrapper(inner_state: AgentState) -> AgentState:
        return parse_node(inner_state, context, client)

    def executor_wrapper(inner_state: AgentState) -> AgentState:
        return executor_node(inner_state, context, client)

    def finalize_wrapper(inner_state: AgentState) -> AgentState:
        return finalize_node(inner_state, context, client)

    if StateGraph is not None:
        graph = StateGraph(AgentState)
        graph.add_node("planner", planner_wrapper)
        graph.add_node("parse", parse_wrapper)
        graph.add_node("executor", executor_wrapper)
        graph.add_node("finalize", finalize_wrapper)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "parse")
        graph.add_edge("parse", "executor")
        graph.add_edge("executor", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    return GraphRunner([planner_wrapper, parse_wrapper, executor_wrapper, finalize_wrapper])


def run_planner_graph(
    payload: dict[str, Any],
    *,
    ollama_url: str,
    model: str,
    timeout_seconds: float,
) -> AgentState:
    client = OllamaClient(base_url=ollama_url, model=model, timeout_seconds=timeout_seconds)
    state: AgentState = {
        "messages": _normalize_messages(payload.get("messages")),
        "task_queue": [],
        "schedule_draft": {
            "status": "needs_clarification",
            "assistantMessage": "Need a little more detail.",
            "tasks": [],
            "events": [],
            "calendar_checks": [],
            "weather": [],
            "notes": [],
        },
    }

    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    runner = build_graph_runner(context, client)
    return runner.invoke(state)
