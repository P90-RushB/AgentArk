"""Compatibility contracts for additive agent selection and message ownership."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_ark.agent.agent_registry import (
    AgentType, agent_constructor_options, configure_agent_environment,
    resolve_agent_type,
)
from agent_ark.ark_eval.run_api_agent import build_model_runtimes, normalize_response_trace


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.thinking_type = None
        self.reasoning_effort = kwargs.get('reasoning_effort')
        self.max_completion_tokens = kwargs.get('max_completion_tokens')


def test_old_api_and_codex_configs_keep_selection_and_constructor_defaults():
    with patch('agent_ark.ark_eval.run_api_agent.APIAgent', wraps=None) as api:
        api._normalize_max_completion_tokens.return_value = 30000
        api.side_effect = FakeAgent
        runtime = build_model_runtimes([{'model': 'old-http', 'api_key': 'fake'}])[0]
    assert 'agent_type' not in runtime
    assert runtime['agent'].kwargs['max_retries'] == 2
    assert 'simple_agent_config' not in runtime['agent'].kwargs
    with patch('agent_ark.ark_eval.run_api_agent.CodexAgent', side_effect=FakeAgent):
        runtime = build_model_runtimes([{'provider': 'codex', 'model': 'old-codex'}])[0]
    assert 'agent_type' not in runtime
    assert runtime['agent'].kwargs['thread_mode'] == 'per_agent'
    assert runtime['agent'].kwargs['context_mode'] == 'lean'


def test_simple_selection_is_separate_from_provider():
    cfg = {'agent_type': 'simple', 'provider': 'openrouter', 'model': 'qwen/test',
           'api_key': 'fake', 'simple_agent': {'context': {'recent_turns': 3}}}
    with patch.object(AgentType, 'load_class', return_value=FakeAgent):
        runtime = build_model_runtimes([cfg])[0]
    assert runtime['agent_type'] == 'simple'
    assert runtime['provider'] == 'openrouter'
    assert runtime['agent'].kwargs['max_retries'] == 0
    assert runtime['agent'].kwargs['simple_agent_config'] == cfg['simple_agent']
    runtime['agent'].kwargs['simple_agent_config']['context']['recent_turns'] = 9
    assert cfg['simple_agent']['context']['recent_turns'] == 3


@pytest.mark.parametrize('cfg', [
    {'agent_type': 'missing'},
    {'agent_type': 'simple', 'provider': 'codex'},
    {'agent_type': 'api', 'provider': 'codex'},
    {'agent_type': 'codex', 'provider': 'openrouter'},
    {'simple_agent': {}},
])
def test_bad_agent_combinations_fail_before_any_client_or_environment(cfg):
    with pytest.raises(ValueError):
        resolve_agent_type(cfg)


def test_simple_incremental_contract_is_scoped_and_legacy_settings_unchanged():
    cfg = {'env_config_overrides': {'env_wrapper_cfg': {'context_manager': {
        'messages': {'return_mode': 'full', 'max_images_per_section': 2},
        'history': {'sample_size': 3},
    }}}}
    original = deepcopy(cfg)
    assert configure_agent_environment(cfg, {'provider': 'openrouter'}) is cfg
    assert configure_agent_environment(cfg, {'provider': 'codex'}) is cfg
    delta = configure_agent_environment(cfg, {'agent_type': 'simple', 'provider': 'openrouter'})
    assert cfg == original
    msg = delta['env_config_overrides']['env_wrapper_cfg']['context_manager']['messages']
    assert msg == {'return_mode': 'delta', 'max_images_per_section': 2,
                   'enabled': True, 'append_only': True, 'only_return_messages': True}
    assert delta['env_config_overrides']['env_wrapper_cfg']['context_manager']['history'] == {'sample_size': 3}


def test_simple_options_are_validated_and_trace_keeps_inner_model_calls():
    with pytest.raises(ValueError, match='object'):
        agent_constructor_options(resolve_agent_type({'agent_type': 'simple'}), {'simple_agent': True})
    trace = {'0': {'assistant_raw': 'final', 'action_extracted': 'action',
                   'agent_events': [{'kind': 'summary', 'text': 'memory'}],
                   'model_requests': [{'messages': [{'role': 'user', 'content': [
                       {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abcd'}}
                   ]}]}], 'agent_usage': {'summary_calls': 1, 'action_calls': 2}}}
    saved = normalize_response_trace(trace)['0']
    assert saved['agent_events'] == trace['0']['agent_events']
    assert saved['agent_usage']['summary_calls'] == 1
    assert saved['action_extracted'] == 'action'
    assert 'data:image' not in str(saved['model_requests'])
    assert 'data:image' in str(trace['0']['model_requests'])


def test_simple_tool_rounds_and_compaction_submit_one_environment_step_per_decision(tmp_path):
    import base64
    import json

    from agent_ark.agent.simple_agent import SimpleAgent, ModelCompletion, ModelUsage, ToolCall
    from agent_ark.agent.simple_agent.encoding import encode_png
    from agent_ark.ark_eval.run_api_agent import _run_case_rollout

    class Model:
        def __init__(self):
            self.requests = []
            self.prepared = False
            self.decisions = 0

        async def complete(self, request):
            self.requests.append(request)
            if request.purpose == 'summary':
                return ModelCompletion('Earlier visible step and selected tap retained.', ModelUsage(5, 3, 8))
            if not self.prepared:
                self.prepared = True
                return ModelCompletion('', ModelUsage(5, 3, 8), (
                    ToolCall(f'files-{self.decisions}', 'list_files', '{"path":"records"}'),))
            self.prepared = False
            self.decisions += 1
            content = json.dumps({'name': 'tap', 'arguments': {'step': self.decisions}})
            return ModelCompletion('', ModelUsage(5, 3, 8), (
                ToolCall(f'submit-{self.decisions}', 'submit_action',
                         json.dumps({'format': 'tool_call', 'content': content})),))

    class Env:
        max_attempts = 1
        max_steps_per_attempt = 3
        _selected_env_cfg = {'task_name': 'visible-tap', 'env_id': 0}
        sub_env = SimpleNamespace(env_info_mgr=SimpleNamespace(env_config=_selected_env_cfg))

        def __init__(self):
            self.steps = []

        def observation(self, step, assistant=None):
            image = encode_png(bytes([step, 0, 0]), 1, 1)
            messages = []
            if step == 0:
                messages.append({'role': 'system', 'content': 'Submit a tap action in the task format.'})
            if assistant is not None:
                messages.append({'role': 'assistant', 'content': assistant})
            messages.append({'role': 'user', 'content': [
                {'type': 'text', 'text': f'Current observation (step_msg): visible step {step}'},
                {'type': 'image_url', 'image_url': {
                    'url': 'data:image/png;base64,' + base64.b64encode(image).decode('ascii')}},
            ]})
            return {0: {'messages': messages}}

        def reset(self, **kwargs):
            return self.observation(0), {}

        def step(self, actions, info=None):
            self.steps.append((actions, info))
            selected = actions[0]
            assert selected['action'].startswith('<tool_call>')
            assert json.loads(selected['action'][11:-12])['arguments']['step'] == len(self.steps)
            last = len(self.steps) == 3
            return (self.observation(len(self.steps), selected['assistant']), {0: 0.0},
                    {0: last}, {'rollout': {'success': last, 'truncated': False}})

    backend = Model()
    agent = SimpleAgent(model_backend=backend, simple_agent_config={
        'context': {'max_images': 2, 'recent_turns': 1},
        'tools': {'enabled': True, 'workspace_parent': str(tmp_path)},
    })
    env = Env()
    try:
        result = _run_case_rollout(env, {'name': 'simple-test', 'agent': agent},
                                  {'case_id': 'test', 'task_name': 'visible-tap', 'group_seed': 1})
        assert result['turns'] == len(env.steps) == backend.decisions == 3
        assert [info for _, info in env.steps] == [{'turn_index': i} for i in range(3)]
        assert sum(request.purpose == 'action' for request in backend.requests) == 6
        assert sum(request.purpose == 'summary' for request in backend.requests) == 1
        assert sum(entry.kind == 'summary_completion'
                   for entry in agent._states[0].runtime.history_snapshot) == 1
        assert all('<tool_call>' not in part.text
                   for request in backend.requests if request.purpose == 'action'
                   for message in request.messages if message.role == 'assistant'
                   for part in message.parts if hasattr(part, 'text'))
    finally:
        agent.close()
