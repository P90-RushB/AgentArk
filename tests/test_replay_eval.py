import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_eval.run_replay import (  # noqa: E402
    ReplayAgent,
    extract_replay_action,
    replay_case_from_record,
    run_replay,
    select_replay_records,
)


def _sample_record():
    return {
        'status': 'ok',
        'case_id': 'task-a-seed-0007',
        'model_name': 'model-a',
        'model': 'fake-model',
        'requested_task_name': 'TaskA',
        'requested_group_seed': 7,
        'requested_env_id': 0,
        'actual_task_name': 'TaskA',
        'actual_rollout_group_seed': 7,
        'actual_env_id': 0,
        'turns': 2,
        'max_attempts': 1,
        'max_steps_per_attempt': 2,
        'score_reward': 3.0,
        'total_reward': 3.0,
        'rollout_success': True,
        'rollout_truncated': False,
        'steps': [
            {
                'turn_index': 0,
                'action_preview': '<tool_call>{"name":"A","arguments":{}}</tool_call>',
                'raw_request_by_agent': {
                    '0': {'request_messages': [{'role': 'user', 'content': 'live request 1'}]},
                },
                'raw_response_by_agent': {
                    '0': {
                        'assistant_raw': '<think>one</think><tool_call>{"name":"A","arguments":{}}</tool_call>',
                        'action_extracted': '<tool_call>{"name":"A","arguments":{}}</tool_call>',
                    },
                },
            },
            {
                'turn_index': 1,
                'action_preview': '<tool_call>{"name":"B","arguments":{}}</tool_call>',
                'raw_request_by_agent': {
                    '0': {'request_messages': [{'role': 'user', 'content': 'live request 2'}]},
                },
                'raw_response_by_agent': {
                    '0': {
                        'assistant_raw': '<think>two</think><tool_call>{"name":"B","arguments":{}}</tool_call>',
                        'action_extracted': '<tool_call>{"name":"B","arguments":{}}</tool_call>',
                    },
                },
            },
        ],
    }


class FakeReplayEnv:
    def __init__(self, cfg):
        self.cfg = dict(cfg or {})
        self.max_attempts = 1
        self.max_steps_per_attempt = 2
        self.current_attempt_index = 1
        self.turns = 0
        self.actions = []
        self._selected_env_cfg = {}

    def reset(self, *, task_name, group_seed, env_id=None, history_snapshot=None, max_attempts=None, start_attempt_index=None):
        self.max_attempts = int(max_attempts or 1)
        self.turns = 0
        self.actions = []
        self._selected_env_cfg = {
            'task_name': task_name,
            'rollout_group_seed': int(group_seed),
            'attempt_group_seed': int(group_seed),
            'attempt_group_seed_history': [int(group_seed)],
            'env_id': int(env_id or 0),
        }
        return {0: {'messages': [{'role': 'user', 'content': 'start'}], 'step_msg': 'start'}}, {
            'attempt': {'index': 1},
        }

    def step(self, code_act, info=None):
        payload = code_act[0]
        self.actions.append(payload['action'] if isinstance(payload, dict) else payload)
        self.turns += 1
        success = self.turns >= 2
        reward = 1.0 if self.turns == 1 else 2.0
        step_info = {
            'attempt': {
                'index': 1,
                'done': success,
                'success': success,
                'auto_reset': False,
            },
            'rollout': {
                'success': success,
                'truncated': False,
                'task_name': self._selected_env_cfg['task_name'],
                'group_seed': self._selected_env_cfg['rollout_group_seed'],
                'rollout_group_seed': self._selected_env_cfg['rollout_group_seed'],
                'current_attempt_group_seed': self._selected_env_cfg['attempt_group_seed'],
                'attempt_group_seed_history': self._selected_env_cfg['attempt_group_seed_history'],
                'env_id': self._selected_env_cfg['env_id'],
                'current_attempt_index': 1,
                'max_attempts': self.max_attempts,
                'max_steps_per_attempt': self.max_steps_per_attempt,
                'max_rollout_steps': self.max_attempts * self.max_steps_per_attempt,
            },
        }
        return (
            {0: {'messages': [{'role': 'user', 'content': f'next {self.turns}'}], 'step_msg': f'next {self.turns}'}},
            {0: reward},
            {0: success, '__all__': success},
            step_info,
        )

    def close(self):
        return


class ReplayEvalTest(unittest.TestCase):
    def test_extract_replay_action_prefers_saved_action(self):
        record = _sample_record()
        action = extract_replay_action(record['steps'][0], 0)
        self.assertEqual(action, '<tool_call>{"name":"A","arguments":{}}</tool_call>')

    def test_replay_agent_returns_saved_actions_in_order(self):
        agent = ReplayAgent(_sample_record())

        requests = agent.build_request_messages({0: {'messages': []}})
        self.assertEqual(requests[0][0]['content'], 'live request 1')

        first, first_trace = agent.forward_with_trace({0: {'messages': []}})
        second, second_trace = agent.forward_with_trace({0: {'messages': []}})

        self.assertIn('"A"', first[0]['action'])
        self.assertIn('"B"', second[0]['action'])
        self.assertIn('<think>one</think>', first_trace[0]['assistant_raw'])
        self.assertIn('<think>two</think>', second_trace[0]['assistant_raw'])

    def test_select_replay_records_filters_by_model_and_seed(self):
        records = [
            dict(_sample_record(), _replay_record_index=0),
            dict(_sample_record(), _replay_record_index=1, model_name='model-b', requested_group_seed=8),
        ]

        selected = select_replay_records(records, {'model_name': 'model-b', 'group_seed': 8})

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['model_name'], 'model-b')

    def test_replay_case_uses_record_identity_and_budget(self):
        case = replay_case_from_record(_sample_record(), {})

        self.assertEqual(case['task_name'], 'TaskA')
        self.assertEqual(case['group_seed'], 7)
        self.assertEqual(case['env_id'], 0)
        self.assertEqual(case['max_attempts'], 1)

    def test_run_replay_with_fake_env_matches_source_result(self):
        with TemporaryDirectory() as tmpdir:
            records_path = Path(tmpdir) / 'records.jsonl'
            records_path.write_text(json.dumps(_sample_record()) + '\n', encoding='utf-8')

            results = run_replay(
                {
                    'env_cfg': {},
                    'replay': {
                        'records_path': str(records_path),
                        'record_index': 0,
                        'step_delay_s': 0,
                    },
                    'hooks': {},
                },
                env_factory=FakeReplayEnv,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['status'], 'ok')
        self.assertTrue(results[0]['replay']['match'])
        self.assertEqual(results[0]['score_reward'], 3.0)


if __name__ == '__main__':
    unittest.main()
