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
    _write_review_notes,
    build_replay_summary,
    select_reward_extremes,
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


if __name__ == "__main__":
    unittest.main()
