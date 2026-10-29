"""Interrupted evaluation evidence is additive and replayed as a prefix only."""
import base64
import io
import json
from copy import deepcopy

from PIL import Image
import pytest

from agent_ark.ark_eval.run_api_agent import (
    build_error_result, evaluate_case, apply_trajectory_load_to_cases, maybe_save_eval_trajectory,
)
from agent_ark.ark_eval.run_replay import (
    extract_replay_action, replay_record, select_replay_records,
)
from agent_ark.ark_eval.trajectory_io import decode_trajectory_value


CASE = {'case_id': 'partial', 'task_name': 'VisibleTask', 'group_seed': 7, 'env_id': 0}


def observation(index):
    image = Image.new('RGB', (2, 2), (index * 25, 10, 30))
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')
    return {0: {'image': image, 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': f'observation {index}'},
        {'type': 'image_url', 'image_url': {'url': url}},
    ]}]}}


class Env:
    max_attempts = 1
    max_steps_per_attempt = 5
    current_attempt_index = 1

    def __init__(self, fail_step=None):
        self.fail_step = fail_step
        self.actions = []
        self._selected_env_cfg = {}

    def reset(self, **kwargs):
        self.actions = []
        self._selected_env_cfg = dict(task_name=kwargs['task_name'], rollout_group_seed=7,
                                      attempt_group_seed=7, attempt_group_seed_history=[7], env_id=0)
        return observation(0), {'attempt': {'index': 1}}

    def step(self, actions, info=None):
        self.actions.append(actions[0]['action'])
        if self.fail_step == len(self.actions):
            raise TimeoutError('environment response lost after action was sent')
        terminal = len(self.actions) >= 5
        return observation(len(self.actions)), {0: 1.0}, {0: terminal, '__all__': terminal}, {
            'attempt': {'index': 1, 'group_seed': 7, 'done': terminal},
            'rollout': {'success': terminal, 'truncated': False},
        }


class Agent:
    def __init__(self, fail_turn=2, trace=True, stale=False, action_length=40):
        self.fail_turn, self.trace, self.stale = fail_turn, trace, stale
        self.action = 'action-' + 'x' * action_length
        self.last_trace_by_agent = {}
        if not trace:
            self.forward_with_trace = None

    def reset(self):
        self.turn = 0
        self.last_trace_by_agent = {}

    def forward_with_details(self, obs):
        result, _ = self._forward(obs)
        return result

    def forward_with_trace(self, obs):
        return self._forward(obs)

    def _forward(self, obs):
        if self.turn == self.fail_turn and self.stale:
            raise TimeoutError('preparation failed before a new trace')
        failed = self.turn == self.fail_turn
        trace = {0: {'assistant_raw': 'brief explanation', 'action_extracted': self.action,
                     'model_requests': [{'purpose': 'summary' if failed else 'action',
                                         'status': 'failed' if failed else 'completed',
                                         'request_messages': obs[0]['messages']}],
                     'agent_events': [{'kind': 'tool_exchange', 'name': 'write_note', 'result': 'saved'}]}}
        self.last_trace_by_agent = trace
        if failed:
            raise TimeoutError('model response lost')
        self.turn += 1
        return {0: {'action': self.action, 'assistant': 'brief explanation'}}, trace


def runtime(agent):
    return dict(name='model', model='fake', provider='fake', agent_type='simple',
                base_url=None, api_key_env=None, agent=agent)


def failed_record(env=None, agent=None):
    env, model = env or Env(), runtime(agent or Agent())
    with pytest.raises(TimeoutError) as caught:
        evaluate_case(env, model, CASE)
    record = build_error_result(model, CASE, caught.value)
    # This is the serialized artifact boundary, not a private object reference.
    return env, json.loads(json.dumps(record)), caught.value


def test_forward_failure_preserves_completed_prefix_images_and_failed_tools():
    env, record, original = failed_record()
    assert type(original) is TimeoutError
    assert record['status'] == 'error' and record['rollout_complete'] is False
    assert record['turns'] == 2 and len(env.actions) == 2
    assert record['total_reward'] == 2
    assert [s['actions']['0']['action'] for s in record['steps']] == env.actions
    assert record['partial_rollout']['failed_turn']['env_action_status'] == 'not_submitted'
    assert record['partial_rollout']['failed_turn']['phase'] == 'agent_forward'
    assert len(record['model_trace_images']) == 3
    assert record['agent_failure_trace']['0']['model_requests'][0]['purpose'] == 'summary'
    assert record['steps'][0]['raw_response_by_agent']['0']['agent_events'][0]['name'] == 'write_note'
    assert record['agent_failure_trace']['0']['agent_events'][0]['name'] == 'write_note'
    current = decode_trajectory_value(record['partial_rollout']['current_observation'])
    assert current['0']['image'].getpixel((0, 0)) == (50, 10, 30)
    assert record['partial_rollout']['observation_coverage']['completed_model_requests'] is True
    assert record['partial_rollout']['observation_coverage']['all_intermediate_raw_observations'] is False


def test_env_step_failure_saves_proposal_as_unknown_not_completed_or_stale_failure_trace():
    env, record, _ = failed_record(Env(fail_step=2), Agent(fail_turn=99))
    assert len(env.actions) == 2
    assert record['turns'] == len(record['steps']) == 1
    assert record['partial_rollout']['failed_turn']['phase'] == 'env_step'
    assert record['partial_rollout']['failed_turn']['env_action_status'] == 'unknown'
    assert record['partial_rollout']['failed_turn']['actions']['0']['action'] == env.actions[-1]
    assert 'agent_failure_trace' not in record


def test_failure_before_new_trace_does_not_claim_previous_decision():
    _, record, _ = failed_record(agent=Agent(stale=True))
    assert record['turns'] == 2
    assert 'agent_failure_trace' not in record
    assert record['partial_rollout']['failed_turn']['raw_response_by_agent'] == {}


@pytest.mark.parametrize('trace', [True, False])
def test_exact_long_actions_survive_partial_record_and_replay(trace):
    env, record, _ = failed_record(agent=Agent(trace=trace, action_length=16000))
    assert record['steps'][0]['actions']['0']['action'] == env.actions[0]
    assert extract_replay_action(record['steps'][0]) == env.actions[0]
    replay_env = Env()
    result = replay_record(env=replay_env, record=record,
                           replay_cfg={'allow_partial': True, 'require_match': True})
    assert replay_env.actions == env.actions
    assert result['status'] == 'error' and result['score_reward'] is None
    assert result['rollout_complete'] is False
    assert result['replay']['partial'] is True and result['replay']['match'] is True
    assert result['rollout_truncated'] is False and result['rollout_budget_exhausted'] is False
    assert result['steps'][0]['partial_step_version'] == 1
    assert record['partial_rollout']['observation_coverage']['completed_model_requests'] is trace


def test_partial_replay_requires_explicit_opt_in_and_default_filter_stays_ok():
    _, record, _ = failed_record()
    assert select_replay_records([record], {}) == []
    with pytest.raises(ValueError, match='allow_partial'):
        replay_record(env=Env(), record=record, replay_cfg={})
    with pytest.raises(ValueError, match='no supported partial'):
        replay_record(env=Env(), record={**record, 'partial_rollout': None, 'steps': []},
                      replay_cfg={'allow_partial': True})


def test_zero_step_partial_replay_resets_without_executing_failed_action():
    env, record, _ = failed_record(agent=Agent(fail_turn=0))
    replay_env = Env()
    result = replay_record(env=replay_env, record=record,
                           replay_cfg={'allow_partial': True, 'require_match': True})
    assert env.actions == replay_env.actions == []
    assert result['turns'] == 0 and result['replay']['match']


def test_legacy_full_error_record_remains_replayable_and_old_action_priority_unchanged():
    model = runtime(Agent(fail_turn=99))
    record = evaluate_case(Env(), model, CASE)
    assert 'partial_rollout' not in record and 'rollout_complete' not in record
    record['status'] = 'error'
    record['error_type'] = 'PlayerFeedbackGateError'
    result = replay_record(env=Env(), record=record, replay_cfg={'require_match': True})
    assert result['status'] == 'ok' and result['replay']['match']
    assert 'partial' not in result['replay']
    assert extract_replay_action({'actions': {'0': {'action': 'new'}}, 'action_preview': 'legacy'}) == 'legacy'
    assert extract_replay_action({'actions': {'0': {'action': 'new'}},
                                  'raw_response_by_agent': {'0': {'action_extracted': 'legacy'}}}) == 'legacy'


def test_partial_replay_detects_stepwise_reward_difference_even_when_total_matches():
    _, record, _ = failed_record()
    tampered = deepcopy(record)
    tampered['steps'][0]['reward_total'] = 2
    tampered['steps'][1]['reward_total'] = 0
    with pytest.raises(RuntimeError, match='mismatch'):
        replay_record(env=Env(), record=tampered,
                      replay_cfg={'allow_partial': True, 'require_match': True})


def test_partial_evidence_is_not_a_formal_trajectory_prefix(tmp_path):
    _, record, _ = failed_record()
    source = tmp_path / 'error.jsonl'
    source.write_text(json.dumps(record) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='finalized trajectory prefix'):
        apply_trajectory_load_to_cases([CASE], {'trajectory_load': {'enabled': True, 'path': str(source), 'index': 0}})
    assert maybe_save_eval_trajectory(env=None, result=record, case=CASE, model_runtime={},
                                      eval_cfg={}, writer=object()) is None


@pytest.mark.parametrize('action', [None, '', 'actual environment action'])
def test_partial_exact_submitted_action_overrides_assistant_and_conflicting_trace(action):
    agent = Agent()
    agent.action = action
    env, record, _ = failed_record(agent=agent)
    for step in record['steps']:
        step['raw_response_by_agent']['0']['action_extracted'] = 'incorrect trace action'
    replay_env = Env()
    result = replay_record(env=replay_env, record=record,
                           replay_cfg={'allow_partial': True, 'require_match': True})
    assert replay_env.actions == env.actions == [action, action]
    assert result['replay']['match']
    assert [step['actions']['0']['action'] for step in result['steps']] == replay_env.actions


def test_partial_comparison_uses_actual_submitted_actions_not_copied_source(monkeypatch):
    from agent_ark.ark_eval import run_replay as module
    _, record, _ = failed_record()
    real_forward = module.ReplayAgent.forward_with_trace

    def tampered_forward(self, obs):
        actions, trace = real_forward(self, obs)
        actions[0]['action'] = 'wrong action actually sent'
        return actions, trace

    monkeypatch.setattr(module.ReplayAgent, 'forward_with_trace', tampered_forward)
    env = Env()
    result = replay_record(env=env, record=record, replay_cfg={'allow_partial': True})
    assert env.actions == ['wrong action actually sent'] * 2
    assert result['replay']['checks']['completed_steps_match'] is False
    assert result['replay']['match'] is False


def test_partial_multiple_agent_actions_are_rejected_before_any_execution():
    _, record, _ = failed_record()
    record['steps'][0]['actions']['1'] = {'action': 'unexpected second agent'}
    env = Env()
    with pytest.raises(ValueError, match='exactly one submitted agent'):
        replay_record(env=env, record=record, replay_cfg={'allow_partial': True})
    assert env.actions == []
