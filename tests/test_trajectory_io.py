import tempfile
import unittest
import asyncio
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_eval.run_api_agent import (  # noqa: E402
    apply_trajectory_load_to_cases,
    maybe_save_eval_trajectory,
)
from agent_ark.ark_eval.trajectory_io import (  # noqa: E402
    TrajectoryJsonlWriter,
    count_history_prefix_attempts,
    decode_history_snapshot,
    encode_history_snapshot,
    history_snapshot_from_record,
    load_trajectory_records,
)
from agent_ark.ark_env.direct_env import EnvWrapper  # noqa: E402
from agent_ark.ark_env.serving.env_client import EnvHttpClient  # noqa: E402

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


class FakeTrajectoryEnv:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def export_finalized_attempts(self, *, prefix_attempts=None):
        if prefix_attempts is None:
            return self.snapshot
        return {unity_id: attempts[:prefix_attempts] for unity_id, attempts in self.snapshot.items()}


class TrajectoryIoTest(unittest.TestCase):
    def test_history_snapshot_round_trip_with_images(self):
        image = Image.new('RGB', (2, 2), color=(255, 0, 0)) if Image is not None else 'image-fallback'
        snapshot = {
            1: [
                [
                    {
                        'obs': {'step_msg': 'start', 'vis': [[image]]},
                        'action': 'go',
                        'next_obs': {'step_msg': 'end', 'vis': [[image]]},
                        'reward': 1.0,
                        'done': True,
                    }
                ]
            ]
        }

        encoded = encode_history_snapshot(snapshot, include_images=True)
        decoded = decode_history_snapshot(encoded)

        self.assertEqual(sorted(decoded.keys()), [1])
        self.assertEqual(decoded[1][0][0]['action'], 'go')
        self.assertEqual(decoded[1][0][0]['reward'], 1.0)
        if Image is not None:
            self.assertTrue(hasattr(decoded[1][0][0]['obs']['vis'][0][0], 'save'))

    def test_maybe_save_eval_trajectory_and_load_prefix(self):
        snapshot = {
            1: [
                [{'obs': {'step_msg': 'attempt1'}, 'action': 'a1', 'next_obs': {}, 'reward': 0.0, 'done': True}],
                [{'obs': {'step_msg': 'attempt2'}, 'action': 'a2', 'next_obs': {}, 'reward': 1.0, 'done': True}],
            ]
        }
        result = {
            'status': 'ok',
            'case_id': 'case-a',
            'model_name': 'model-a',
            'model': 'm',
            'provider': 'fake',
            'requested_task_name': 'task-a',
            'requested_group_seed': 7,
            'requested_env_id': 0,
            'actual_task_name': 'task-a',
            'actual_rollout_group_seed': 7,
            'actual_env_id': 0,
            'max_attempts': 3,
            'max_steps_per_attempt': 1,
            'rollout_step_budget': 3,
            'attempt_group_seed_history': [7, 7],
            'score_reward': 1.0,
            'last_attempt_reward': 1.0,
            'best_attempt_reward': 1.0,
            'rollout_success': True,
            'ever_attempt_success': True,
            'rollout_truncated': False,
            'attempt_rewards': [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'trajectories.jsonl'
            writer = TrajectoryJsonlWriter(str(path))
            record = maybe_save_eval_trajectory(
                env=FakeTrajectoryEnv(snapshot),
                result=result,
                case={'case_id': 'case-a', 'task_name': 'task-a', 'group_seed': 7, 'env_id': 0},
                model_runtime={'name': 'model-a', 'model': 'm', 'provider': 'fake'},
                eval_cfg={'trajectory_save': {'enabled': True, 'output_path': str(path), 'prefix_attempts': 1}},
                writer=writer,
            )

            self.assertIsNotNone(record)
            self.assertEqual(result['trajectory']['prefix_attempt_count'], 1)
            loaded = load_trajectory_records(str(path))[0]
            prefix = history_snapshot_from_record(loaded)
            self.assertEqual(count_history_prefix_attempts(prefix), 1)
            self.assertEqual(prefix[1][0][0]['action'], 'a1')

    def test_apply_trajectory_load_to_cases_sets_history_and_start_attempt(self):
        snapshot = {
            1: [
                [{'obs': {'step_msg': 'attempt1'}, 'action': 'a1', 'next_obs': {}, 'reward': 0.0, 'done': True}],
                [{'obs': {'step_msg': 'attempt2'}, 'action': 'a2', 'next_obs': {}, 'reward': 0.0, 'done': True}],
            ]
        }
        result = {
            'status': 'ok',
            'case_id': 'case-a',
            'model_name': 'model-a',
            'model': 'm',
            'provider': 'fake',
            'requested_task_name': 'task-a',
            'requested_group_seed': 7,
            'requested_env_id': 0,
            'actual_task_name': 'task-a',
            'actual_rollout_group_seed': 7,
            'actual_env_id': 0,
            'max_attempts': 3,
            'max_steps_per_attempt': 1,
            'rollout_step_budget': 3,
            'attempt_group_seed_history': [7, 7],
            'score_reward': 0.0,
            'last_attempt_reward': 0.0,
            'best_attempt_reward': 0.0,
            'rollout_success': False,
            'ever_attempt_success': False,
            'rollout_truncated': True,
            'attempt_rewards': [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'trajectories.jsonl'
            writer = TrajectoryJsonlWriter(str(path))
            maybe_save_eval_trajectory(
                env=FakeTrajectoryEnv(snapshot),
                result=result,
                case={'case_id': 'case-a', 'task_name': 'task-a', 'group_seed': 7, 'env_id': 0},
                model_runtime={'name': 'model-a', 'model': 'm', 'provider': 'fake'},
                eval_cfg={'trajectory_save': {'enabled': True, 'output_path': str(path), 'prefix_attempts': 2}},
                writer=writer,
            )

            cases = apply_trajectory_load_to_cases(
                [{'case_id': 'case-a', 'task_name': 'task-a', 'group_seed': 7, 'env_id': 0}],
                {'trajectory_load': {'enabled': True, 'path': str(path), 'prefix_attempts': 1}},
            )

            self.assertEqual(len(cases), 1)
            self.assertEqual(count_history_prefix_attempts(cases[0]['history_snapshot']), 1)
            self.assertEqual(cases[0]['start_attempt_index'], 2)
            self.assertEqual(cases[0]['trajectory_ref_resolved']['prefix_attempt_count'], 1)

    def test_envwrapper_resolves_attempt_index_from_loaded_prefix(self):
        snapshot = {1: [[{'action': 'a1'}], [{'action': 'a2'}]]}

        self.assertEqual(EnvWrapper._resolve_start_attempt_index(None, snapshot), 3)
        self.assertEqual(EnvWrapper._resolve_start_attempt_index(5, snapshot), 5)

    def test_verl_client_sends_start_attempt_index(self):
        client = EnvHttpClient('http://example.test')
        captured = []

        def fake_post(path, payload):
            captured.append((path, payload))
            return {'env_id': 'env-a'}

        client._post = fake_post

        asyncio.run(client.astart_env(
            'env-a',
            task_name='task-a',
            group_seed=7,
            history_snapshot={1: [[{'action': 'a1'}]]},
            start_attempt_index=2,
        ))
        asyncio.run(client.aacquire_start_env(
            {},
            task_name='task-a',
            group_seed=7,
            history_snapshot={1: [[{'action': 'a1'}]]},
            start_attempt_index=2,
        ))

        self.assertEqual(captured[0][0], '/v1/envs/env-a/start')
        self.assertEqual(captured[0][1]['start_attempt_index'], 2)
        self.assertEqual(captured[1][0], '/v1/envs/acquire_start')
        self.assertEqual(captured[1][1]['start_attempt_index'], 2)

    def test_env_client_sends_uid_for_server_managed_tasks(self):
        client = EnvHttpClient('http://example.test')
        captured = []

        def fake_post(path, payload):
            captured.append((path, payload))
            return {'env_id': 'env-a'}

        client._post = fake_post

        asyncio.run(client.astart_env('env-a', uid='group-a'))
        asyncio.run(client.aacquire_start_env({}, uid='group-b'))

        self.assertEqual(captured[0][0], '/v1/envs/env-a/start')
        self.assertEqual(captured[0][1]['uid'], 'group-a')
        self.assertEqual(captured[1][0], '/v1/envs/acquire_start')
        self.assertEqual(captured[1][1]['uid'], 'group-b')


if __name__ == '__main__':
    unittest.main()
