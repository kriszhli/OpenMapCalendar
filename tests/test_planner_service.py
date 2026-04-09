from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from planner_service.distillation import distill_session_log
from planner_service.export import export_training_rows
from planner_service.graph import run_planner_graph
from planner_service.memory import LongTermMemoryStore, MemoryFact
from planner_service.persistence import SessionEventLogger, load_jsonl


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

            checkpoint_path = Path(tmp) / "checkpoints" / "langgraph.sqlite"
            session_log_path = Path(tmp) / "sessions" / "cal-smoke.jsonl"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(session_log_path.exists())
            self.assertGreater(len(load_jsonl(session_log_path)), 0)

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

    def test_export_writes_terminal_outcomes_only(self) -> None:
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
            self.assertEqual(row["output"]["status"], "ready")
            self.assertEqual(row["output"]["assistant_message"], "Drafted a plan.")
            self.assertEqual(len(row["output"]["tool_actions"]), 1)
            self.assertEqual(row["input"]["messages"][0]["content"], "Plan a museum visit before lunch.")

            batch2 = export_training_rows(data_root=tmp, calendar_id="cal-export")
            self.assertEqual(batch.path.read_text(encoding="utf-8"), batch2.path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
