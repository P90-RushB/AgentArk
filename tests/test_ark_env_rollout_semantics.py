import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.ark_env import ArkEnv  # noqa: E402
from agent_ark.ark_env.ark_sub_env import ArkSubEnv  # noqa: E402
from agent_ark.ark_eval.run_api_agent import summarize_attempt_rewards  # noqa: E402


class FakeSubEnv:
    def __init__(self, step_results, *, max_attempts=2, max_steps_per_attempt=1):
        self.ml_unity_id_map = {1: 1}
        self.step_results = list(step_results)
        self.step_calls = 0
        self.reset_calls = []
        self.env_info_mgr = SimpleNamespace(
            env_config={
                'task_name': 'task-a',
                'group_seed': 123,
                'env_id': 0,
                'max_attempts': max_attempts,
                'max_steps_per_attempt': max_steps_per_attempt,
            }
        )

    def step(self, actions, info=None):
        del actions, info
        result = self.step_results[self.step_calls]
        self.step_calls += 1
        return deepcopy(result)

    def reset(self, task_name=None, group_seed=None, env_id=None, coordination=None):
        del coordination
        self.reset_calls.append({'task_name': task_name, 'group_seed': group_seed, 'env_id': env_id})
        self.env_info_mgr.env_config.update({
            'task_name': task_name,
            'group_seed': group_seed,
            'env_id': env_id,
        })
        return {1: {'step_msg': f'reset-{len(self.reset_calls)}'}}, {'reset': True}


class FakeRolloutContext:
    def __init__(self):
        self.finalized_attempts = []
        self.reset_history_snapshot = None

    def record_transition(self, **kwargs):
        del kwargs

    def finalize_attempt(self, unity_ids):
        self.finalized_attempts.append(list(unity_ids))

    def take_finalized_attempts(self):
        return {}

    def on_reset(self, **kwargs):
        self.reset_history_snapshot = kwargs.get('history_snapshot')
        return kwargs['obs']

    def on_attempt_reset(self, **kwargs):
        return kwargs['obs']

    def finalize_obs(self, obs, rollout_done=None):
        del rollout_done
        return obs


def make_env(step_results, *, max_attempts=2, current_attempt_index=1):
    env = object.__new__(ArkEnv)
    env.cfg = {}
    env.sub_env = FakeSubEnv(step_results, max_attempts=max_attempts)
    env.rollout_ctx = FakeRolloutContext()
    env.hooks = SimpleNamespace(enabled=False)
    env._hook_text_max_chars = 6000
    env._hook_max_images_per_observation = 4
    env.shared_coordinator = None
    env.local_history_store = SimpleNamespace(sample_snapshot=lambda *args, **kwargs: {})
    env.rollout_started = True
    env.current_attempt_index = current_attempt_index
    env.max_attempts = max_attempts
    env.max_steps_per_attempt = 1
    env._selected_task_name = 'task-a'
    env._selected_rollout_group_seed = 123
    env._current_attempt_group_seed = 123
    env._attempt_group_seed_history = [123 for _ in range(current_attempt_index)]
    env._reroll_group_seed_on_same_task = False
    env._selected_env_id = 0
    env._selected_env_cfg = dict(env.sub_env.env_info_mgr.env_config)
    env._last_reset_plan = {}
    env._active_history_bucket_key = None
    env._rollout_finalized_attempts = {}
    env._publish_finalized_attempts = lambda finalized_attempts=None: None
    return env


class ArkEnvRolloutSemanticsTest(unittest.TestCase):
    def test_intermediate_success_auto_resets_instead_of_terminating_rollout(self):
        env = make_env([
            ({1: {'step_msg': 'attempt-1-done'}}, {1: 1.0}, {1: True, '__all__': True}, {}),
        ], max_attempts=2)

        obs, reward, done, info = env.step({1: '<tool_call>{}</tool_call>'})

        self.assertEqual(obs[1]['step_msg'], 'reset-1')
        self.assertEqual(reward, {1: 1.0})
        self.assertEqual(done, {1: False, '__all__': False})
        self.assertTrue(info['attempt']['done'])
        self.assertTrue(info['attempt']['success'])
        self.assertTrue(info['attempt']['auto_reset'])
        self.assertFalse(info['rollout']['success'])
        self.assertFalse(info['rollout']['truncated'])
        self.assertEqual(info['rollout']['current_attempt_index'], 2)
        self.assertEqual(env.current_attempt_index, 2)
        self.assertTrue(env.rollout_started)
        self.assertEqual(len(env.sub_env.reset_calls), 1)

    def test_final_success_terminates_rollout(self):
        env = make_env([
            ({1: {'step_msg': 'attempt-2-done'}}, {1: 1.0}, {1: True, '__all__': True}, {}),
        ], max_attempts=2, current_attempt_index=2)

        _, _, done, info = env.step({1: '<tool_call>{}</tool_call>'})

        self.assertEqual(done, {1: True, '__all__': True})
        self.assertTrue(info['attempt']['success'])
        self.assertTrue(info['rollout']['success'])
        self.assertFalse(info['rollout']['truncated'])
        self.assertFalse(env.rollout_started)
        self.assertEqual(len(env.sub_env.reset_calls), 0)

    def test_final_failure_truncates_rollout(self):
        env = make_env([
            ({1: {'step_msg': 'attempt-2-done'}}, {1: 0.25}, {1: True, '__all__': True}, {}),
        ], max_attempts=2, current_attempt_index=2)

        _, _, done, info = env.step({1: '<tool_call>{}</tool_call>'})
        self.assertEqual(done, {1: True, '__all__': True})
        self.assertFalse(info['attempt']['success'])
        self.assertFalse(info['rollout']['success'])
        self.assertTrue(info['rollout']['truncated'])
        self.assertFalse(env.rollout_started)
        self.assertEqual(len(env.sub_env.reset_calls), 0)

    def test_max_attempts_one_keeps_success_terminal(self):
        env = make_env([
            ({1: {'step_msg': 'attempt-1-done'}}, {1: 1.0}, {1: True, '__all__': True}, {}),
        ], max_attempts=1)

        _, _, done, info = env.step({1: '<tool_call>{}</tool_call>'})

        self.assertEqual(done, {1: True, '__all__': True})
        self.assertTrue(info['rollout']['success'])
        self.assertFalse(info['rollout']['truncated'])
        self.assertEqual(len(env.sub_env.reset_calls), 0)

    def test_reset_can_start_from_loaded_attempt_prefix(self):
        env = make_env([], max_attempts=5)

        obs, info = env.reset(
            task_name='task-a',
            group_seed=123,
            env_id=0,
            history_snapshot={1: [[{'action': 'a1'}], [{'action': 'a2'}]]},
            max_attempts=5,
            start_attempt_index=3,
        )

        self.assertEqual(env.current_attempt_index, 3)
        self.assertEqual(info['attempt']['index'], 3)
        self.assertTrue(info['rollout']['history_snapshot_seeded'])
        self.assertEqual(info['rollout']['current_attempt_index'], 3)
        self.assertEqual(obs[1]['step_msg'], 'reset-2')
        self.assertEqual(len(env.sub_env.reset_calls), 2)
        self.assertEqual(env.rollout_ctx.reset_history_snapshot[1][1][0]['action'], 'a2')


class EvalAttemptSummaryTest(unittest.TestCase):
    def test_summary_scores_final_attempt_and_keeps_ever_success_diagnostics(self):
        summary = summarize_attempt_rewards([
            {
                'reward_total': 1.0,
                'info': {'attempt': {'index': 1, 'done': True, 'success': True, 'auto_reset': True}},
            },
            {
                'reward_total': 0.25,
                'info': {'attempt': {'index': 2, 'done': True, 'success': False, 'auto_reset': False}},
            },
        ])

        self.assertEqual(summary['last_attempt_index'], 2)
        self.assertEqual(summary['last_attempt_reward'], 0.25)
        self.assertEqual(summary['best_attempt_reward'], 1.0)
        self.assertFalse(summary['final_attempt_success'])
        self.assertTrue(summary['ever_attempt_success'])
        self.assertEqual(summary['first_success_attempt_index'], 1)
        self.assertEqual(summary['success_attempt_count'], 1)


class ArkSubEnvReuseTest(unittest.TestCase):
    def _make_sub_env(self, *, recreate=None, width=128, height=128, last_resolution=(64, 64)):
        sub_env = object.__new__(ArkSubEnv)
        sub_env.env = object()
        sub_env.cfg = {}
        sub_env._last_env_resolution = last_resolution
        wrapper_cfg = {}
        if recreate is not None:
            wrapper_cfg['recreate_unity_on_resolution_change'] = recreate
        sub_env.env_info_mgr = SimpleNamespace(
            env_config={
                'width': width,
                'height': height,
                'env_wrapper_cfg': wrapper_cfg,
            }
        )
        return sub_env

    def test_reuses_live_unity_env_when_resolution_is_stable(self):
        sub_env = self._make_sub_env(width=384, height=384, last_resolution=(384, 384))

        self.assertFalse(sub_env._should_recreate_env())

    def test_recreates_on_resolution_change_by_default(self):
        sub_env = self._make_sub_env()

        self.assertTrue(sub_env._should_recreate_env())

    def test_can_opt_into_resolution_change_recreate_for_compatibility(self):
        sub_env = self._make_sub_env(recreate=True)

        self.assertTrue(sub_env._should_recreate_env())

    def test_can_opt_out_of_resolution_change_recreate(self):
        sub_env = self._make_sub_env(recreate=False)

        self.assertFalse(sub_env._should_recreate_env())


if __name__ == '__main__':
    unittest.main()
