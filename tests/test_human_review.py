import base64
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_ark.ark_eval.human_review import (  # noqa: E402
    REVIEW_NOTES_FILENAME,
    REVIEW_NOTES_SCHEMA,
    _read_review_notes,
    _extract_authored_task_prompt,
    _extract_task_prompt,
    _write_review_notes,
    build_replay_summary,
    discover_tasks,
    select_reward_extremes,
    select_result_entries,
    select_trajectory_entries,
)


def _record(seed, score, *, frame_count=1):
    encoded = {
        "__agentark_type__": "agentark.pil_image_png_base64.v1",
        "mime_type": "image/png",
        "size": [2, 2],
        "data": base64.b64encode(f"png-{seed}".encode()).decode(),
    }
    frames = [encoded for _ in range(frame_count)]
    return {
        "source": {"case_id": f"seed-{seed}", "model_name": "test-model"},
        "task": {"task_name": "TaskA", "group_seed": seed},
        "rollout": {
            "score_reward": score,
            "last_attempt_reward": score,
            "best_attempt_reward": score,
            "rollout_success": score > 0,
            "rollout_truncated": score <= 0,
            "max_attempts": 1,
            "max_steps_per_attempt": 1,
        },
        "history_snapshot": {
            "0": [[{
                "obs": {"reset_context": "start", "vis": [frames]},
                "next_obs": {"step_msg": "changed", "vis": [frames]},
                "action": '<tool_call>{"name":"Act","arguments":{"value":1}}</tool_call>',
                "reward": score,
                "done": True,
            }]]
        },
    }


class HumanReviewTest(unittest.TestCase):
    def test_extract_task_prompt_accepts_records_without_section_marker(self):
        record = _record(1, 1.0)
        record["history_snapshot"]["0"][0][0]["obs"]["task_prompt"] = "Plain task prompt"

        self.assertEqual(_extract_task_prompt(record), "Plain task prompt")

    def test_extract_authored_prompt_from_prefab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefab = Path(tmpdir) / "Task.prefab"
            prefab.write_text(
                "  taskDescription: '[task prompt]\n\n    Do the visible thing.'\n"
                "  taskCodeWrapper: \n",
                encoding="utf-8",
            )

            self.assertEqual(
                _extract_authored_task_prompt(Path(tmpdir)),
                "[task prompt]\n\nDo the visible thing.",
            )

    def test_empty_prefab_prompt_falls_back_to_csharp_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "Task.prefab").write_text(
                "  taskDescription: \n  taskCodeWrapper: \n",
                encoding="utf-8",
            )
            (root / "Task.cs").write_text(
                'private const string DefaultDescription =\n'
                '    "First sentence; still the prompt. " +\n'
                '    "Second sentence.\\n\\nFinal paragraph.";\n',
                encoding="utf-8",
            )

            self.assertEqual(
                _extract_authored_task_prompt(root),
                "First sentence; still the prompt. Second sentence.\n\nFinal paragraph.",
            )

    def test_csharp_verbatim_default_prompt_is_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "Task.cs").write_text(
                'private const string DefaultDescription = @"First line; yes.\n\n'
                'Second ""quoted"" line.";\n',
                encoding="utf-8",
            )

            self.assertEqual(
                _extract_authored_task_prompt(root),
                'First line; yes.\n\nSecond "quoted" line.',
            )

    def test_observation_only_capture_is_disclosed_and_rendered(self):
        encoded = {
            "__agentark_type__": "agentark.pil_image_png_base64.v1",
            "mime_type": "image/png",
            "size": [2, 2],
            "data": base64.b64encode(b"captured-png").decode(),
        }
        record = _record(3, 1.0)
        record["history_snapshot"] = {}
        record["runtime_request_capture"] = {
            "turns": [
                {"agents": {"0": {"images": [encoded]}}},
                {"agents": {"0": {"images": [encoded]}}},
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = build_replay_summary(
                record,
                kind="high",
                task_dir=Path(tmpdir),
                task_relative_dir="tasks/task_003",
                semantic_images=True,
            )

        self.assertEqual(summary["trajectory_detail"], "observation_only")
        self.assertTrue(summary["record_limitation_zh"])
        self.assertEqual(len(summary["steps"]), 2)
        self.assertEqual(summary["frame_count"], 1)

    def test_results_restore_actions_when_history_snapshot_is_absent(self):
        encoded = {
            "__agentark_type__": "agentark.pil_image_png_base64.v1",
            "mime_type": "image/png",
            "size": [2, 2],
            "data": base64.b64encode(b"captured-png").decode(),
        }
        trajectory = _record(5, -1.0)
        trajectory["history_snapshot"] = {}
        trajectory["runtime_request_capture"] = {
            "turns": [{"agents": {"0": {"images": [encoded]}}}]
        }
        result = {
            "steps": [
                {
                    "action_preview": (
                        '<tool_call>{"name":"Submit","arguments":{"answer":2}}</tool_call>'
                    ),
                    "reward_total": -1.0,
                    "done": {"__all__": True},
                    "info": {"attempt": {"index": 1}},
                    "step_messages": [
                        {"step_msg": "['<step_context>\\nWRONG\\n</step_context>']"}
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = build_replay_summary(
                trajectory,
                kind="low",
                task_dir=Path(tmpdir),
                task_relative_dir="tasks/task_005",
                semantic_images=True,
                result_record=result,
            )

        self.assertEqual(summary["trajectory_detail"], "result_supplemented")
        self.assertEqual(summary["steps"][0]["tool"]["name"], "Submit")
        self.assertEqual(summary["steps"][0]["reward"], -1.0)
        self.assertIn("WRONG", summary["steps"][0]["after_text"])
        self.assertEqual(summary["frame_count"], 1)

    def test_select_trajectory_entries_supports_legacy_registry_paths(self) -> None:
        registry = [
            {
                "kind": "record",
                "path": "artifacts/Task117_seeds1_10_results.jsonl",
                "records": 10,
                "task_id": 117,
            },
            {
                "kind": "record",
                "path": "artifacts/Task117_seeds1_10_trajectories.jsonl",
                "records": 10,
                "task_id": 117,
            },
        ]

        selected = select_trajectory_entries(registry, [117])

        self.assertEqual(
            selected[117]["path"],
            "artifacts/Task117_seeds1_10_trajectories.jsonl",
        )

    def test_select_result_entries_uses_exact_trajectory_sibling(self) -> None:
        registry = [
            {
                "path": "artifacts/run-a/Task153_seeds1_10_trajectories.jsonl",
                "record_kind": "trajectories",
                "task_id": 153,
            },
            {
                "path": "artifacts/run-a/Task153_seeds1_10_results.jsonl",
                "record_kind": "results",
                "task_id": 153,
            },
            {
                "path": "artifacts/run-b/Task153_seeds1_10_results.jsonl",
                "record_kind": "results",
                "task_id": 153,
            },
        ]
        trajectories = select_trajectory_entries(registry, [153])

        selected = select_result_entries(registry, trajectories)

        self.assertEqual(
            selected[153]["path"],
            "artifacts/run-a/Task153_seeds1_10_results.jsonl",
        )

    def test_select_reward_extremes_uses_different_seeds_for_ties(self):
        records = [_record(1, 1.0), _record(2, 1.0), _record(3, 1.0)]
        high, low = select_reward_extremes(records)
        self.assertEqual(high["task"]["group_seed"], 1)
        self.assertEqual(low["task"]["group_seed"], 3)

    def test_select_reward_extremes_orders_by_score(self):
        high, low = select_reward_extremes([_record(4, -1.0), _record(5, 0.5)])
        self.assertEqual(high["rollout"]["score_reward"], 0.5)
        self.assertEqual(low["rollout"]["score_reward"], -1.0)

    def test_build_replay_summary_extracts_steps_and_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = build_replay_summary(
                _record(7, 1.0, frame_count=2),
                kind="high",
                task_dir=Path(tmpdir),
                task_relative_dir="tasks/task_007",
                semantic_images=True,
            )
            self.assertEqual(summary["seed"], 7)
            self.assertEqual(len(summary["steps"]), 1)
            self.assertEqual(summary["steps"][0]["tool"]["name"], "Act")
            self.assertTrue(summary["has_agent_transition_video"])
            self.assertEqual(summary["frame_count"], 1)  # identical consecutive frames are deduplicated
            self.assertTrue(list(Path(tmpdir).glob("high/*.png")))

    def test_build_replay_summary_recovers_omitted_terminal_result_step(self):
        trajectory = _record(3, -1.0)
        history_step = trajectory["history_snapshot"]["0"][0][0]
        history_step["reward"] = 0.0
        history_step["done"] = False
        trajectory["rollout"]["attempt_rewards"] = [
            {"index": 1, "reward_total": -1.0, "turns": 2, "done": True}
        ]
        terminal_action = '<tool_call>{"name":"ExecuteBypass","arguments":{}}</tool_call>'
        result = {
            "start_attempt_index": 1,
            "steps": [
                {
                    "action_preview": history_step["action"],
                    "reward_total": 0.0,
                    "done": {"0": False, "__all__": False},
                    "info": {"attempt": {"index": 1, "auto_reset": False}},
                    "step_messages": [
                        {"step_msg": "['<step_context>\\nchanged\\n</step_context>']"}
                    ],
                },
                {
                    "action_preview": terminal_action,
                    "reward_total": -1.0,
                    "done": {"0": True, "__all__": True},
                    "info": {"attempt": {"index": 1, "auto_reset": False}},
                    "step_messages": [
                        {
                            "step_msg": (
                                "['<step_context>\\nUNSAFE | BYPASS FAILED AT 0>2"
                                "\\n</step_context>']"
                            )
                        }
                    ],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = build_replay_summary(
                trajectory,
                kind="low",
                task_dir=Path(tmpdir),
                task_relative_dir="tasks/task_153",
                semantic_images=True,
                result_record=result,
            )

        self.assertEqual(summary["trajectory_detail"], "result_supplemented")
        self.assertEqual(len(summary["steps"]), 2)
        self.assertEqual(summary["steps"][-1]["tool"]["name"], "ExecuteBypass")
        self.assertEqual(summary["steps"][-1]["reward"], -1.0)
        self.assertTrue(summary["steps"][-1]["done"])
        self.assertTrue(summary["steps"][-1]["record_result_only"])
        self.assertIn("UNSAFE", summary["steps"][-1]["after_text"])
        self.assertIn("不会伪造", summary["record_limitation_zh"])

    def test_build_replay_summary_does_not_duplicate_complete_history(self):
        trajectory = _record(4, 1.0)
        history_step = trajectory["history_snapshot"]["0"][0][0]
        result = {
            "steps": [
                {
                    "action_preview": history_step["action"],
                    "reward_total": 1.0,
                    "done": {"__all__": True},
                    "info": {"attempt": {"index": 1}},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = build_replay_summary(
                trajectory,
                kind="high",
                task_dir=Path(tmpdir),
                task_relative_dir="tasks/task_004",
                semantic_images=True,
                result_record=result,
            )

        self.assertEqual(summary["trajectory_detail"], "full")
        self.assertEqual(len(summary["steps"]), 1)
        self.assertNotIn("record_result_only", summary["steps"][0])

    def test_text_modality_does_not_emit_compatibility_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = build_replay_summary(
                _record(9, 0.0),
                kind="low",
                task_dir=Path(tmpdir),
                task_relative_dir="tasks/task_009",
                semantic_images=False,
            )
            self.assertEqual(summary["frame_count"], 0)
            self.assertFalse(list(Path(tmpdir).glob("low/*.png")))

    def test_review_notes_are_atomically_persisted_to_the_workbench(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            document = _write_review_notes(root, {"notes": {"41": "纵向后坐力偏小"}})
            self.assertEqual(document["schema"], REVIEW_NOTES_SCHEMA)
            self.assertEqual(_read_review_notes(root), document)
            self.assertTrue((root / REVIEW_NOTES_FILENAME).is_file())
            self.assertFalse((root / f"{REVIEW_NOTES_FILENAME}.tmp").exists())

    def test_review_notes_reject_non_task_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                _write_review_notes(Path(tmpdir), {"notes": {"task-41": "invalid"}})

    def test_discover_tasks_can_include_gui_and_packaged_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            packaged = root / "packaged"
            gui = source / "Task187_GuiEvidence"
            external = packaged / "ExternalTask"
            gui.mkdir(parents=True)
            (external / "cfg").mkdir(parents=True)
            (gui / "task_config.yaml").write_text(
                "task_name: GuiEvidence\n"
                "task_info:\n"
                "  name: GuiEvidence\n"
                "  tags: [gui]\n",
                encoding="utf-8",
            )
            (external / "cfg" / "task_config.yaml").write_text(
                "task_name: ExternalTask\n"
                "task_info:\n"
                "  name: ExternalTask\n"
                "  tags: [3d]\n"
                "  legacy_names: [Task142_ExternalTask]\n",
                encoding="utf-8",
            )

            default = discover_tasks([source, packaged], [142, 187])
            self.assertEqual(set(default), {142})
            included = discover_tasks([source, packaged], [142, 187], include_gui=True)
            self.assertEqual(set(included), {142, 187})
            self.assertEqual(included[142]["config_path"].name, "task_config.yaml")


if __name__ == "__main__":
    unittest.main()
