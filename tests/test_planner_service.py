from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

from planner_service.distillation import distill_session_log
from planner_service.export import export_training_rows
from planner_service.graph import run_planner_graph
from planner_service.memory import LongTermMemoryStore, MemoryFact
from planner_service.persistence import SessionEventLogger, load_jsonl
from planner_service.service import _append_confirmation_record, run_health_check
from planner_service.tools import MockWeatherTool


def _base_context() -> dict[str, object]:
    return {
        "calendarStartDate": "2026-04-08",
        "calendarEndDate": "2026-04-12",
        "visibleDays": 5,
        "dayStartHour": 7,
        "dayEndHour": 22,
        "timezone": "America/Chicago",
        "existingEvents": [],
        "relativeDayHints": [],
    }


def _context_with_dates(
    start_date: str,
    end_date: str | None = None,
    *,
    visible_days: int = 5,
    existing_events: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    context = _base_context()
    context["calendarStartDate"] = start_date
    context["calendarEndDate"] = end_date or start_date
    context["visibleDays"] = visible_days
    context["existingEvents"] = existing_events or []
    return context


class PlannerServiceTests(unittest.TestCase):
    def test_smoke_creates_checkpoint_and_session_log(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-test-") as tmp:
            payload = {
                "calendarId": "cal-smoke",
                "messages": [{"role": "user", "text": "Plan a museum visit before lunch."}],
                "context": _base_context(),
            }
            state = run_planner_graph(
                payload,
                ollama_url="http://127.0.0.1:11434",
                model="gemma4:e2b",
                timeout_seconds=1.0,
                data_root=tmp,
            )

            response = state.get("response") or {}
            self.assertIn(response.get("status"), {"ready", "needs_clarification"})
            self.assertIn("assistantMessage", response)
            self.assertIn("events", response)
            self.assertIn("task_queue", response)
            self.assertIn("schedule_draft", response)
            self.assertIn("run_id", response)
            if response.get("status") == "ready" and response.get("events"):
                self.assertTrue(response["events"][0].get("pending"))

            checkpoint_path = Path(tmp) / "checkpoints" / "langgraph.sqlite"
            session_log_path = Path(tmp) / "sessions" / "cal-smoke.jsonl"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(session_log_path.exists())
            self.assertGreater(len(load_jsonl(session_log_path)), 0)

    def test_multi_intent_decomposition_creates_atomic_blocks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-decompose-") as tmp:
            payload = {
                "calendarId": "cal-decompose",
                "messages": [
                    {"role": "user", "text": "On the second day, schedule a museum visit and dinner."}
                ],
                "context": _base_context(),
            }
            state = run_planner_graph(
                payload,
                ollama_url="http://127.0.0.1:11434",
                model="gemma4:e2b",
                timeout_seconds=1.0,
                data_root=tmp,
            )

            decomposition = state.get("decomposition") or {}
            blocks = decomposition.get("blocks") if isinstance(decomposition, dict) else []
            self.assertIsInstance(blocks, list)
            self.assertGreaterEqual(len(blocks), 2)
            titles = " ".join(str(block.get("title", "")) for block in blocks if isinstance(block, dict))
            self.assertIn("Museum", titles)
            self.assertIn("Dinner", titles)
            if len(blocks) > 1 and isinstance(blocks[0], dict) and isinstance(blocks[1], dict):
                self.assertEqual(blocks[1].get("depends_on"), [blocks[0].get("id")])
            self.assertGreaterEqual(len(state.get("task_queue") or []), 4)

    def test_weather_conflict_triggers_reranking(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-weather-") as tmp:
            base = date(2026, 4, 8)
            bad_start = None
            for offset in range(0, 10):
                candidate = (base + timedelta(days=offset)).isoformat()
                forecast = MockWeatherTool.forecast(date=candidate, location="local area")
                if forecast.get("condition") not in {"sunny", "partly cloudy"}:
                    later_good = any(
                        MockWeatherTool.forecast(
                            date=(base + timedelta(days=offset + step)).isoformat(),
                            location="local area",
                        ).get("condition") in {"sunny", "partly cloudy"}
                        for step in range(1, 5)
                    )
                    if later_good:
                        bad_start = candidate
                        break

            self.assertIsNotNone(bad_start)
            assert bad_start is not None
            payload = {
                "calendarId": "cal-weather",
                "messages": [{"role": "user", "text": "Plan an outdoor walk."}],
                "context": _context_with_dates(
                    bad_start,
                    (date.fromisoformat(bad_start) + timedelta(days=4)).isoformat(),
                    visible_days=5,
                ),
            }
            state = run_planner_graph(
                payload,
                ollama_url="http://127.0.0.1:11434",
                model="gemma4:e2b",
                timeout_seconds=1.0,
                data_root=tmp,
            )

            response = state.get("response") or {}
            self.assertEqual(response.get("status"), "ready")
            self.assertGreaterEqual(int(state.get("replan_attempts") or 0), 1)
            events = response.get("events") or []
            self.assertTrue(events)
            self.assertNotEqual(events[0].get("date"), bad_start)

    def test_memory_preference_conflict_is_rebalanced(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-memory-replan-") as tmp:
            store = LongTermMemoryStore(Path(tmp) / "chromadb")
            store.upsert_facts(
                [
                    MemoryFact(
                        calendar_id="cal-memory-replan",
                        category="lunch_avoidance",
                        normalized_value="avoid lunch window",
                        confidence=0.94,
                        summary="User avoids lunch meetings.",
                        source_session_ids=["seed-session"],
                        source_run_ids=["seed-run"],
                        created_at="2026-04-08T00:00:00Z",
                        updated_at="2026-04-08T00:00:00Z",
                    )
                ]
            )

            payload = {
                "calendarId": "cal-memory-replan",
                "messages": [{"role": "user", "text": "Schedule a lunch check-in sometime today."}],
                "context": _base_context(),
            }
            state = run_planner_graph(
                payload,
                ollama_url="http://127.0.0.1:11434",
                model="gemma4:e2b",
                timeout_seconds=1.0,
                data_root=tmp,
            )

            response = state.get("response") or {}
            self.assertEqual(response.get("status"), "ready")
            self.assertGreaterEqual(int(state.get("replan_attempts") or 0), 1)
            blocks = (state.get("schedule_draft") or {}).get("blocks") or []
            self.assertTrue(blocks)
            first_block = blocks[0]
            self.assertNotEqual(first_block.get("start_time"), "12:00")
            self.assertNotEqual(first_block.get("end_time"), "13:00")

    def test_replan_loop_limit_falls_back_to_clarification(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-loop-") as tmp:
            blocked_events: list[dict[str, object]] = []
            for hour in range(7, 22):
                blocked_events.append(
                    {
                        "date": "2026-04-09",
                        "startTime": f"{hour:02d}:00",
                        "endTime": f"{hour:02d}:30",
                        "title": f"Blocked {hour:02d}:00",
                    }
                )
                blocked_events.append(
                    {
                        "date": "2026-04-09",
                        "startTime": f"{hour:02d}:30",
                        "endTime": f"{hour + 1:02d}:00",
                        "title": f"Blocked {hour:02d}:30",
                    }
                )

            payload = {
                "calendarId": "cal-loop-limit",
                "messages": [{"role": "user", "text": "Schedule a lunch check-in."}],
                "context": _context_with_dates(
                    "2026-04-09",
                    "2026-04-09",
                    visible_days=1,
                    existing_events=blocked_events,
                ),
            }
            state = run_planner_graph(
                payload,
                ollama_url="http://127.0.0.1:11434",
                model="gemma4:e2b",
                timeout_seconds=1.0,
                data_root=tmp,
            )

            response = state.get("response") or {}
            draft = response.get("schedule_draft") or {}
            self.assertEqual(response.get("status"), "needs_clarification")
            self.assertLessEqual(int(state.get("replan_attempts") or 0), 2)
            options = draft.get("clarification_options") or []
            self.assertEqual(len(options), 2)
            self.assertIn("Choose one", str(response.get("assistantMessage") or ""))

    def test_memory_retrieval_is_injected_before_planning(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-memory-") as tmp:
            store = LongTermMemoryStore(Path(tmp) / "chromadb")
            store.upsert_facts(
                [
                    MemoryFact(
                        calendar_id="cal-memory",
                        category="lunch_avoidance",
                        normalized_value="prefer before lunch",
                        confidence=0.91,
                        summary="User prefers meetings before lunch.",
                        source_session_ids=["seed-session"],
                        source_run_ids=["seed-run"],
                        created_at="2026-04-08T00:00:00Z",
                        updated_at="2026-04-08T00:00:00Z",
                    )
                ]
            )

            payload = {
                "calendarId": "cal-memory",
                "messages": [{"role": "user", "text": "Schedule this before lunch."}],
                "context": _base_context(),
            }
            state = run_planner_graph(
                payload,
                ollama_url="http://127.0.0.1:11434",
                model="gemma4:e2b",
                timeout_seconds=1.0,
                data_root=tmp,
            )

            retrieved = state.get("retrieved_memory") or {}
            facts = retrieved.get("facts") if isinstance(retrieved, dict) else None
            self.assertTrue(facts)
            summary_text = retrieved.get("summary_text", "")
            self.assertIn("lunch_avoidance", summary_text)

    def test_distillation_emits_only_stable_facts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-distill-") as tmp:
            session_path = Path(tmp) / "sessions" / "cal-distill.jsonl"
            logger = SessionEventLogger(session_path, "cal-distill", "cal-distill")

            run_id = "cal-distill-run-0001"
            common_window = {
                "start": "2026-04-08",
                "end": "2026-04-12",
                "visible_days": 5,
                "day_start_hour": 7,
                "day_end_hour": 22,
                "timezone": "America/Chicago",
            }
            logger.append(
                "planner_request_received",
                {
                    "run_index": 1,
                    "run_id": run_id,
                    "calendar_window": common_window,
                    "message_count": 5,
                    "latest_user_message": "I work from 9am to 5pm and need a 2 hour focus block.",
                    "messages": [
                        {"role": "user", "content": "I work from 9am to 5pm."},
                        {"role": "user", "content": "I work from 9am to 5pm and need a 2 hour focus block."},
                        {"role": "user", "content": "I avoid lunch meetings."},
                        {"role": "user", "content": "I need a 30 minute commute buffer."},
                        {"role": "user", "content": "I am traveling to Denver next Tuesday."},
                    ],
                },
                run_id=run_id,
            )
            logger.append(
                "normalized_user_message_batch",
                {
                    "run_index": 1,
                    "messages": [
                        {"role": "user", "content": "I work from 9am to 5pm."},
                        {"role": "user", "content": "I work from 9am to 5pm and need a 2 hour focus block."},
                        {"role": "user", "content": "I avoid lunch meetings."},
                        {"role": "user", "content": "I need a 30 minute commute buffer."},
                        {"role": "user", "content": "I am traveling to Denver next Tuesday."},
                    ],
                },
                run_id=run_id,
            )
            logger.append(
                "final_response",
                {
                    "status": "ready",
                    "assistant_message": "Drafted a plan.",
                    "event_count": 1,
                    "task_count": 2,
                },
                run_id=run_id,
            )

            result = distill_session_log(
                session_log_path=str(session_path),
                calendar_id="cal-distill",
                session_id="cal-distill",
            )
            categories = sorted(fact.category for fact in result.facts)
            self.assertEqual(categories, ["commute_sensitivity", "focus_block", "lunch_avoidance", "work_hours"])
            self.assertTrue(all(fact.document_id for fact in result.facts))
            doc_ids = [fact.document_id for fact in result.facts]
            self.assertEqual(doc_ids, [fact.document_id for fact in result.facts])

    def test_export_writes_confirmed_outcomes_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-export-") as tmp:
            session_path = Path(tmp) / "sessions" / "cal-export.jsonl"
            logger = SessionEventLogger(session_path, "cal-export", "cal-export")
            run_id = "cal-export-run-0001"

            logger.append(
                "planner_request_received",
                {
                    "run_index": 1,
                    "run_id": run_id,
                    "calendar_window": {
                        "start": "2026-04-08",
                        "end": "2026-04-12",
                        "visible_days": 5,
                        "day_start_hour": 7,
                        "day_end_hour": 22,
                        "timezone": "America/Chicago",
                    },
                    "messages": [{"role": "user", "content": "Plan a museum visit before lunch."}],
                },
                run_id=run_id,
            )
            logger.append(
                "model_output_summary",
                {"chars": 128, "has_json_object": True, "sanitized_chars": 96, "has_think_block": False},
                run_id=run_id,
            )
            logger.append(
                "parsed_planner_payload_summary",
                {"status": "ready", "task_count": 2, "event_count": 0, "note_count": 1, "assistant_message": "Drafted."},
                run_id=run_id,
            )
            logger.append(
                "tool_result_summary",
                {"tool": "calendar_mock", "action": "check_availability", "status": "available"},
                run_id=run_id,
            )
            logger.append(
                "final_response",
                {
                    "status": "ready",
                    "assistant_message": "Drafted a plan.",
                    "event_count": 1,
                    "task_count": 2,
                },
                run_id=run_id,
            )
            logger.append(
                "plan_confirmation_received",
                {
                    "user_confirmed": True,
                    "confirmed_schedule": {
                        "status": "ready",
                        "assistantMessage": "Drafted a plan.",
                        "events": [{"date": "2026-04-08", "startTime": "10:00", "endTime": "11:00"}],
                    },
                    "schedule_draft": {"status": "ready"},
                    "correction_diff": {
                        "summary": "1 edited event",
                        "event_diffs": [],
                        "preference_updates": [],
                    },
                    "preference_updates": [],
                },
                run_id=run_id,
            )
            logger.append(
                "distillation_results",
                {"summary": {"fact_count": 1}, "upserted_facts": [{"category": "lunch_avoidance"}], "skipped": []},
                run_id=run_id,
            )

            batch = export_training_rows(data_root=tmp, calendar_id="cal-export")
            exported_text = batch.path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(batch.row_count, 1)
            self.assertEqual(len(exported_text), 1)

            row = json.loads(exported_text[0])
            self.assertEqual(row["calendar_id"], "cal-export")
            self.assertEqual(row["approval"]["user_confirmed"], True)
            self.assertEqual(row["confirmed_schedule"]["status"], "ready")
            self.assertEqual(row["schedule_draft"]["status"], "ready")
            self.assertEqual(row["user_input"]["messages"][0]["content"], "Plan a museum visit before lunch.")
            self.assertEqual(len(row["output"]["tool_actions"]), 1)

            batch2 = export_training_rows(data_root=tmp, calendar_id="cal-export")
            self.assertEqual(batch.path.read_text(encoding="utf-8"), batch2.path.read_text(encoding="utf-8"))

    def test_export_skips_unconfirmed_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-export-unconfirmed-") as tmp:
            session_path = Path(tmp) / "sessions" / "cal-export-unconfirmed.jsonl"
            logger = SessionEventLogger(session_path, "cal-export-unconfirmed", "cal-export-unconfirmed")
            run_id = "cal-export-unconfirmed-run-0001"

            logger.append(
                "planner_request_received",
                {
                    "run_index": 1,
                    "run_id": run_id,
                    "calendar_window": {
                        "start": "2026-04-08",
                        "end": "2026-04-12",
                        "visible_days": 5,
                        "day_start_hour": 7,
                        "day_end_hour": 22,
                        "timezone": "America/Chicago",
                    },
                    "messages": [{"role": "user", "content": "Plan a museum visit before lunch."}],
                },
                run_id=run_id,
            )
            logger.append(
                "final_response",
                {
                    "status": "ready",
                    "assistant_message": "Drafted a plan.",
                    "event_count": 1,
                    "task_count": 2,
                },
                run_id=run_id,
            )
            logger.append(
                "plan_confirmation_received",
                {
                    "user_confirmed": False,
                    "confirmed_schedule": {
                        "status": "ready",
                        "assistantMessage": "Drafted a plan.",
                        "events": [{"date": "2026-04-08", "startTime": "10:00", "endTime": "11:00"}],
                    },
                    "schedule_draft": {"status": "ready"},
                    "correction_diff": {
                        "summary": "1 edited event",
                        "event_diffs": [],
                        "preference_updates": [],
                    },
                    "preference_updates": [],
                },
                run_id=run_id,
            )

            batch = export_training_rows(data_root=tmp, calendar_id="cal-export-unconfirmed")
            self.assertEqual(batch.row_count, 0)
            self.assertEqual(batch.path.read_text(encoding="utf-8"), "")

    def test_confirmation_requires_user_confirmation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-confirmation-") as tmp:
            payload = {
                "calendar_id": "cal-confirmation",
                "run_id": "run-confirmation-0001",
                "user_confirmed": False,
                "confirmed_schedule": {"status": "ready", "events": []},
                "schedule_draft": {"status": "ready"},
                "correction_diff": {},
                "preference_updates": [],
            }

            with self.assertRaises(ValueError):
                _append_confirmation_record(data_root=tmp, payload=payload)

    def test_manual_correction_distills_forced_memory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-correction-") as tmp:
            session_path = Path(tmp) / "sessions" / "cal-correction.jsonl"
            logger = SessionEventLogger(session_path, "cal-correction", "cal-correction")
            run_id = "cal-correction-run-0001"

            logger.append(
                "plan_confirmation_received",
                {
                    "user_confirmed": True,
                    "confirmed_schedule": {
                        "status": "ready",
                        "assistantMessage": "Committed.",
                        "events": [{"date": "2026-04-08", "startTime": "09:00", "endTime": "10:00"}],
                    },
                    "schedule_draft": {"status": "ready"},
                    "correction_diff": {
                        "summary": "User moved the plan away from lunch.",
                        "event_diffs": [
                            {
                                "index": 0,
                                "changedFields": ["startTime", "endTime"],
                                "before": {"title": "Lunch check-in", "date": "2026-04-08", "startTime": "12:00", "endTime": "13:00"},
                                "after": {"title": "Lunch check-in", "date": "2026-04-08", "startTime": "09:00", "endTime": "10:00"},
                            }
                        ],
                        "preference_updates": [
                            {
                                "category": "lunch_avoidance",
                                "normalized_value": "avoid lunch window",
                                "summary": "User moved the plan away from lunch.",
                                "confidence": 1.0,
                                "forced": True,
                                "priority": 100,
                            }
                        ],
                    },
                    "preference_updates": [
                        {
                            "category": "lunch_avoidance",
                            "normalized_value": "avoid lunch window",
                            "summary": "User moved the plan away from lunch.",
                            "confidence": 1.0,
                            "forced": True,
                            "priority": 100,
                        }
                    ],
                },
                run_id=run_id,
            )

            result = distill_session_log(
                session_log_path=str(session_path),
                calendar_id="cal-correction",
                session_id="cal-correction",
            )
            self.assertTrue(result.facts)
            facts_by_category = {fact.category: fact for fact in result.facts}
            self.assertIn("lunch_avoidance", facts_by_category)
            self.assertTrue(facts_by_category["lunch_avoidance"].forced)
            self.assertGreaterEqual(facts_by_category["lunch_avoidance"].priority, 100)

    def test_health_check_runs_locally(self) -> None:
        with tempfile.TemporaryDirectory(prefix="planner-health-") as tmp:
            exit_code = run_health_check(
                data_root=tmp,
                ollama_url="http://127.0.0.1:11434",
                model="gemma4:e2b",
                timeout_seconds=1.0,
                calendar_id="cal-health",
            )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
