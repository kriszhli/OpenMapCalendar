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
    raw_model_output: str
    planner_payload: dict[str, Any]
    response: dict[str, Any]


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
            '{"status":"ready|needs_clarification","assistantMessage":"string","task_queue":[...],"schedule_draft":{...}}',
            "Use task_queue entries with tool names calendar_mock or weather_mock.",
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
    next_state["schedule_draft"] = {
        "status": status,
        "assistantMessage": assistant_message,
        "tasks": next_state["task_queue"],
        "events": list(schedule_draft.get("events", []) if isinstance(schedule_draft.get("events"), list) else []),
        "calendar_checks": list(schedule_draft.get("calendar_checks", []) if isinstance(schedule_draft.get("calendar_checks"), list) else []),
        "weather": list(schedule_draft.get("weather", []) if isinstance(schedule_draft.get("weather"), list) else []),
        "notes": list(schedule_draft.get("notes", []) if isinstance(schedule_draft.get("notes"), list) else []),
    }

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


def executor_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    run_id = _clean_text(state.get("run_id"), 128)
    draft = deepcopy(state.get("schedule_draft") or {})
    task_queue = list(state.get("task_queue") or [])
    calendar_checks = list(draft.get("calendar_checks") or [])
    weather_rows = list(draft.get("weather") or [])
    events = list(draft.get("events") or [])
    notes = list(draft.get("notes") or [])
    conflict_detected = False

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
            _log_event(
                context,
                "tool_result_summary",
                {
                    "tool": "weather_mock",
                    "action": "forecast",
                    "date": report["date"],
                    "summary": report["summary"],
                },
                run_id=run_id,
            )
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
                conflict_detected = True
                notes.append(f"calendar:{result.status}")
                _log_event(
                    context,
                    "state_transition",
                    {
                        "state": "conflict",
                        "tool": "calendar_mock",
                        "task_id": task.get("id"),
                        "status": result.status,
                    },
                    run_id=run_id,
                )
            _log_event(
                context,
                "tool_result_summary",
                {
                    "tool": "calendar_mock",
                    "action": "check_availability",
                    "date": _clean_text(task.get("date"), 20),
                    "status": result.status,
                    "reason": result.reason,
                    "conflict_count": len(result.conflicting_events),
                },
                run_id=run_id,
            )
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
            else:
                conflict_detected = True
                notes.append("event-skipped:conflict")
                _log_event(
                    context,
                    "tool_result_summary",
                    {
                        "tool": "calendar_mock",
                        "action": "write_event",
                        "status": "skipped",
                        "reason": "calendar conflict",
                        "title": event["title"],
                    },
                    run_id=run_id,
                )
            continue

    draft["calendar_checks"] = calendar_checks
    draft["weather"] = weather_rows
    draft["events"] = events
    draft["notes"] = notes
    if draft.get("status") != "needs_clarification" and not events and calendar_checks:
        if any(check.get("status") != "available" for check in calendar_checks):
            draft["assistantMessage"] = "I found a calendar conflict, so I left the event unscheduled."
            conflict_detected = True

    if conflict_detected:
        _log_event(
            context,
            "state_transition",
            {
                "state": "conflict",
                "status": "conflict-detected",
                "task_count": len(task_queue),
            },
            run_id=run_id,
        )

    next_state: AgentState = deepcopy(state)
    next_state["schedule_draft"] = draft
    return next_state


def finalize_node(state: AgentState, context: dict[str, Any], client: Any) -> AgentState:
    run_id = _clean_text(state.get("run_id"), 128)
    draft = state.get("schedule_draft") or {}
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
    ) -> None:
        self.nodes = nodes
        self.checkpoint_store = checkpoint_store
        self.calendar_id = calendar_id
        self.run_id = run_id

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
        for step_name, node in self.nodes:
            current = node(current)
            self._save_checkpoint(step_name, current)
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
        checkpointer = _build_langgraph_checkpointer(context["checkpoint_db_path"])
        if checkpointer is not None and thread_id:
            return LangGraphRuntime(graph.compile(checkpointer=checkpointer), thread_id=thread_id)
        return LangGraphRuntime(graph.compile(), thread_id=thread_id or context["calendar_id"])

    return GraphRunner(
        [
            ("planner", planner_wrapper),
            ("parse", parse_wrapper),
            ("executor", executor_wrapper),
            ("finalize", finalize_wrapper),
        ],
        checkpoint_store=checkpoint_store,
        calendar_id=context["calendar_id"],
        run_id=context["run_id"],
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
