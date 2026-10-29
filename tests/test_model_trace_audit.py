"""Audit storage must improve visibility without changing replay or history."""
import base64
from copy import deepcopy
from hashlib import sha256

from agent_ark.agent.simple_agent.encoding import encode_png
from agent_ark.ark_eval.model_trace import capture_image_reference
from agent_ark.ark_eval.run_api_agent import _run_case_rollout, normalize_response_trace
from agent_ark.ark_eval.run_replay import ReplayAgent
from agent_ark.ark_eval.trajectory_io import build_eval_trajectory_record, history_snapshot_from_record
from agent_ark.interaction.hooks import HookManager


def _image():
    raw = encode_png(bytes([255, 0, 0]), 1, 1)
    return sha256(raw).hexdigest(), 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')


def test_model_trace_images_are_deduplicated_and_originals_unchanged():
    digest, url = _image()
    message = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}]}
    trace = {0: {'model_requests': [
        {'request_messages': [message], 'request_transport': {'inputs': [{'type': 'image', 'url': url}]}},
        {'request_messages': [message]},
    ], 'assistant_raw': 'unchanged', 'action_extracted': 'unchanged'}}
    original = deepcopy(trace)
    assets = {}
    saved = normalize_response_trace(trace, image_assets=assets)
    assert trace == original
    assert list(assets) == [digest]
    assert assets[digest]['data_url'] == url
    expected = 'agentark-trace-image:' + digest
    assert saved['0']['model_requests'][0]['request_messages'][0]['content'][0]['image_url']['url'] == expected
    assert saved['0']['model_requests'][0]['request_transport']['inputs'][0]['url'] == expected
    assert 'data:image' not in str(saved)
    # Existing standalone trace consumers retain their image-omission behavior.
    assert 'data:image' not in str(normalize_response_trace(trace))


def test_invalid_inline_images_are_not_claimed_as_available():
    assert capture_image_reference('data:image/png;base64,broken?', {}) is None
    assert capture_image_reference('https://example.com/image.png', {}) is None


def test_corrupt_saved_image_assets_are_unavailable_without_changing_actions():
    _, url = _image()
    agent = ReplayAgent({'steps': [{'raw_response_by_agent': {'0': {'action_extracted': 'action'}}}],
                         'model_trace_images': {'0' * 64: {'data_url': url}}})
    assert agent.model_trace_images == {}
    assert agent.forward({0: {}}) == {0: 'action'}


class _Env:
    max_attempts = 1
    max_steps_per_attempt = 2
    current_attempt_index = 1
    _selected_env_cfg = {'task_name': 'Task', 'group_seed': 7}

    def reset(self, **kwargs):
        self.actions = []
        return {0: {'messages': [{'role': 'user', 'content': 'visible'}]}}, {}

    def step(self, actions, info=None):
        self.actions.append(actions[0]['action'])
        last = len(self.actions) == 2
        return {0: {'messages': [{'role': 'user', 'content': 'feedback'}]}}, {0: 0}, {0: last}, {
            'attempt': {'index': 1}, 'rollout': {'success': last, 'truncated': False}}


class _Hook:
    def __init__(self):
        self.events = []

    def handle_event(self, event):
        self.events.append(event)


def test_failed_calls_are_visible_without_execution_or_provider_exception_text():
    import pytest

    class FailedAgent:
        last_trace_by_agent = {}

        def reset(self):
            pass

        def forward_with_trace(self, obs):
            self.last_trace_by_agent = {0: {'model_requests': [{'purpose': 'action', 'status': 'failed',
                'request_messages': [{'role': 'user', 'content': 'visible'}], 'error_type': 'TimeoutError'}]}}
            raise TimeoutError('provider-secret-error')

    env, collector = _Env(), _Hook()
    with pytest.raises(TimeoutError):
        _run_case_rollout(env, {'name': 'failed', 'agent': FailedAgent()},
                          {'case_id': 'case', 'task_name': 'Task', 'group_seed': 7},
                          hook_manager=HookManager([collector], strict=True))
    event = [item['payload'] for item in collector.events if item['event'] == 'agent_response'][0]
    assert event['status'] == 'error'
    assert event['raw_trace_by_agent']['0']['model_requests'][0]['status'] == 'failed'
    assert 'provider-secret-error' not in str(event)
    assert env.actions == []


def test_replay_preserves_actual_calls_and_emits_exact_original_images_once():
    digest, url = _image()
    ref = 'agentark-trace-image:' + digest
    steps = [{'turn_index': index, 'raw_response_by_agent': {'0': {
        'action_extracted': f'action-{index}', 'assistant_raw': f'original-{index}',
        'model_requests': [{'purpose': 'action', 'completion': '', 'tool_calls': [{'name': 'submit_action'}],
                            'request_messages': [{'role': 'user', 'content': [
                                {'type': 'image_url', 'image_url': {'url': ref}}]}]}],
        'agent_events': [{'kind': 'action_translation'}],
    }}} for index in range(2)]
    record = {'steps': steps, 'model_trace_images': {digest: {'mime_type': 'image/png', 'data_url': url}}}
    before = deepcopy(record)
    agent = ReplayAgent(record)
    env, collector = _Env(), _Hook()
    rollout = _run_case_rollout(env, {'name': 'replay', 'agent': agent},
                                {'case_id': 'case', 'task_name': 'Task', 'group_seed': 7},
                                hook_manager=HookManager([collector], strict=True))
    assert env.actions == ['action-0', 'action-1']
    assert record == before
    assert rollout['model_trace_images'] == record['model_trace_images']
    response_events = [event['payload'] for event in collector.events if event['event'] == 'agent_response']
    assert response_events[0]['model_trace_images'] == record['model_trace_images']
    assert 'model_trace_images' not in response_events[1]
    assert response_events[0]['raw_trace_by_agent']['0']['replay'] is True
    assert rollout['step_records'][1]['raw_response_by_agent']['0']['model_requests'] == steps[1]['raw_response_by_agent']['0']['model_requests']


def test_optional_trajectory_audit_respects_prefix_and_never_enters_resume_history():
    digest, url = _image()
    ref = 'agentark-trace-image:' + digest
    snapshot = {0: [[{'action': 'old-a'}], [{'action': 'old-b'}]]}
    result = {'steps': [
        {'info': {'attempt': {'index': index}}, 'raw_response_by_agent': {'0': {
            'action_extracted': f'action-{index}', 'model_requests': [{'request_messages': [
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': ref}}]}]}]}}}
        for index in (1, 2)
    ], 'model_trace_images': {digest: {'mime_type': 'image/png', 'data_url': url}}}
    kwargs = {'result': result, 'case': {'case_id': 'c', 'task_name': 'Task', 'group_seed': 7},
              'model_runtime': {'name': 'm'}, 'history_snapshot': snapshot, 'prefix_attempts': 1}
    saved = build_eval_trajectory_record(**kwargs)
    assert saved['version'] == 1
    assert len(saved['agent_trace']['steps']) == 1
    assert saved['agent_trace']['model_trace_images'] == result['model_trace_images']
    assert history_snapshot_from_record(saved) == {0: [snapshot[0][0]]}
    no_images = build_eval_trajectory_record(**kwargs, include_images=False)
    assert no_images['agent_trace']['model_trace_images'] == {}
    assert 'data:image' not in str(no_images)
    legacy = build_eval_trajectory_record(**kwargs, include_agent_trace=False)
    assert 'agent_trace' not in legacy
    assert legacy['history_snapshot'] == saved['history_snapshot']
