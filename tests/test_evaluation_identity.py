"""Harness migrations must not reuse or delete another evaluation's results."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from agent_ark.ark_eval.evaluation_identity import build_evaluation_identity
from agent_ark.ark_eval.run_api_agent import (
    JsonlWriter, build_error_result, build_model_runtimes, compact_eval_results_file,
    load_existing_eval_results, write_eval_result,
)
from agent_ark.ark_eval.run_parallel_api_eval import (
    ThreadSafeJsonlWriter, _model_stub, build_parallel_eval_jobs,
    compact_parallel_results_file, filter_existing_parallel_jobs,
)


CASE = {'case_id': 'one', 'task_name': 'TaskA', 'group_seed': 1, 'env_id': 0}
API = {'name': 'same-model', 'model': 'test-model', 'provider': 'openai'}
SIMPLE = {**API, 'agent_type': 'simple', 'simple_agent': {
    'context': {'recent_turns': 3},
    'tools': {'enabled': True, 'workspace_parent': 'tmp/notes'},
}}


def runtime(config):
    return _model_stub(config, 0)


def row(config=None, *, status='ok'):
    value = {
        'status': status, 'requested_task_name': CASE['task_name'],
        'requested_group_seed': 1, 'requested_env_id': 0,
        'model_name': API['name'], 'score_reward': 1,
    }
    if config is not None:
        stub = runtime(config)
        value.update({key: stub[key] for key in ('model', 'provider', 'agent_type', 'evaluation_identity')})
    return value


def save(path, rows):
    path.write_text(''.join(json.dumps(value) + '\n' for value in rows), encoding='utf-8')


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


@pytest.mark.parametrize('parallel', [False, True])
def test_new_simple_does_not_skip_old_api_or_unknown_simple_config(tmp_path, parallel):
    path = tmp_path / 'results.jsonl'
    old_api = row()
    old_simple = {**row(), 'agent_type': 'simple', 'provider': 'openai'}
    save(path, [old_api, old_simple])
    if parallel:
        pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [SIMPLE]), str(path))
        assert len(pending) == 1
    else:
        existing = load_existing_eval_results(str(path), [CASE], [runtime(SIMPLE)])
    assert not existing


@pytest.mark.parametrize('parallel', [False, True])
def test_same_simple_identity_resumes_but_different_budget_does_not(tmp_path, parallel):
    path = tmp_path / 'results.jsonl'
    save(path, [row(SIMPLE)])
    changed = deepcopy(SIMPLE)
    changed['simple_agent']['context']['recent_turns'] += 1
    for cfg, expected in [(SIMPLE, 1), (changed, 0)]:
        if parallel:
            pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [cfg]), str(path))
            assert len(pending) == 1 - expected
        else:
            existing = load_existing_eval_results(str(path), [CASE], [runtime(cfg)])
        assert len(existing) == expected


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('provider', ['openai', 'codex'])
def test_legacy_harness_can_resume_own_historical_rows(tmp_path, parallel, provider):
    path = tmp_path / 'results.jsonl'
    cfg = {**API, 'provider': provider}
    legacy = {**row(), 'provider': provider, 'model': cfg['model']}
    save(path, [legacy])
    if parallel:
        pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [cfg]), str(path))
        assert not pending
    else:
        existing = load_existing_eval_results(str(path), [CASE], [runtime(cfg)])
    assert len(existing) == 1


@pytest.mark.parametrize('parallel', [False, True])
def test_compaction_and_retry_preserve_other_harness_and_config(tmp_path, parallel):
    path = tmp_path / 'results.jsonl'
    changed = deepcopy(SIMPLE)
    changed['simple_agent']['context']['recent_turns'] = 5
    keep = [row(API), row(changed), row()]
    save(path, keep + [row(SIMPLE, status='error'), row(SIMPLE)])
    if parallel:
        removed = compact_parallel_results_file(str(path), build_parallel_eval_jobs([CASE], [SIMPLE]))
        writer = ThreadSafeJsonlWriter(str(path), append=True, replace_existing_by_resume_key=True)
        writer.write({**row(SIMPLE), 'score_reward': 2})
    else:
        removed = compact_eval_results_file(str(path), [CASE], [runtime(SIMPLE)])
        writer = JsonlWriter(str(path), append=True)
        write_eval_result(writer, {**row(SIMPLE), 'score_reward': 2}, replace_existing_by_resume_key=True)
    assert removed == 1
    remaining = read(path)
    assert remaining[:3] == keep
    assert len(remaining) == 4
    assert remaining[-1]['score_reward'] == 2


def test_identity_excludes_credentials_local_paths_and_mapping_order():
    original = deepcopy(SIMPLE)
    original.update(api_key='first-secret', api_key_env='FIRST_KEY', codex_bin='C:/first/codex', cwd='C:/first')
    changed = deepcopy(original)
    changed.update(api_key='second-secret', api_key_env='SECOND_KEY', codex_bin='/second/codex', cwd='/second')
    changed['simple_agent']['tools']['workspace_parent'] = '/second/workspaces'
    assert build_evaluation_identity(original) == build_evaluation_identity(changed)
    assert build_evaluation_identity(dict(reversed(list(original.items())))) == build_evaluation_identity(original)
    assert set(build_evaluation_identity(original)) == {'version', 'agent_type', 'config_sha256'}


def test_omitted_simple_settings_equal_explicit_effective_defaults():
    from dataclasses import asdict
    from agent_ark.agent.simple_agent.runtime import SimpleRuntimeConfig
    from agent_ark.agent.simple_agent.tools import ToolLimits

    minimal = {**API, 'agent_type': 'simple'}
    explicit = {**minimal, 'simple_agent': {
        'context': asdict(SimpleRuntimeConfig()), 'tool_choice': 'required',
        'output_token_parameter': 'max_completion_tokens',
        'tools': {'enabled': False, 'limits': asdict(ToolLimits(max_records=512))},
    }}
    assert build_evaluation_identity(minimal) == build_evaluation_identity(explicit)


@pytest.mark.parametrize('config,old_hash', [
    ({'agent_type': 'simple', 'provider': 'openai', 'model': 'test-model'},
     'e57a2f3f5b16b81648ce586f423e2356f61f6ae3c798826094dfa06284ac9bc8'),
    (SIMPLE, '2504b7efa393b6c898a9b6d3d4934c992c0e51455e84e87500e158d721e5ef5e'),
])
def test_explicit_whole_turn_policy_preserves_pre_policy_fingerprint(config, old_hash):
    legacy = deepcopy(config)
    legacy.setdefault('simple_agent', {}).setdefault('context', {})['compaction_policy'] = 'whole_turn_v1'
    legacy['simple_agent']['context']['output_budget_policy'] = 'legacy_v1'
    legacy['simple_agent']['context']['round_budget_policy'] = 'legacy_v1'
    identity = build_evaluation_identity(legacy)
    assert identity == {'version': 1, 'agent_type': 'simple', 'config_sha256': old_hash}
    assert build_evaluation_identity(config)['config_sha256'] != old_hash


@pytest.mark.parametrize('parallel', [False, True])
def test_old_simple_fingerprint_resumes_only_with_legacy_policy_and_survives_new_results(tmp_path, parallel):
    old = {**row(), 'model': 'test-model', 'provider': 'openai', 'agent_type': 'simple',
           'evaluation_identity': {'version': 1, 'agent_type': 'simple',
                                  'config_sha256': '2504b7efa393b6c898a9b6d3d4934c992c0e51455e84e87500e158d721e5ef5e'}}
    path = tmp_path / 'results.jsonl'
    save(path, [old])
    legacy = deepcopy(SIMPLE)
    legacy['simple_agent']['context']['compaction_policy'] = 'whole_turn_v1'
    legacy['simple_agent']['context']['output_budget_policy'] = 'legacy_v1'
    legacy['simple_agent']['context']['round_budget_policy'] = 'legacy_v1'
    for config, expected in [(legacy, 1), (SIMPLE, 0)]:
        if parallel:
            pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [config]), str(path))
            assert len(pending) == 1 - expected
        else:
            existing = load_existing_eval_results(str(path), [CASE], [runtime(config)])
        assert len(existing) == expected
    if parallel:
        removed = compact_parallel_results_file(str(path), build_parallel_eval_jobs([CASE], [SIMPLE]))
        writer = ThreadSafeJsonlWriter(str(path), append=True, replace_existing_by_resume_key=True)
        writer.write(row(SIMPLE))
    else:
        removed = compact_eval_results_file(str(path), [CASE], [runtime(SIMPLE)])
        writer = JsonlWriter(str(path), append=True)
        write_eval_result(writer, row(SIMPLE), replace_existing_by_resume_key=True)
    assert removed == 0
    assert read(path) == [old, row(SIMPLE)]


def test_sequential_and_parallel_creation_record_same_identity():
    class FakeAgent:
        def __init__(self, **kwargs):
            self.thinking_type = None
            self.reasoning_effort = kwargs.get('reasoning_effort')
            self.max_completion_tokens = kwargs.get('max_completion_tokens')

    with patch('agent_ark.agent.agent_registry.AgentType.load_class', return_value=FakeAgent):
        actual = build_model_runtimes([{**SIMPLE, 'api_key': 'fake'}])[0]
    assert actual['evaluation_identity'] == runtime(SIMPLE)['evaluation_identity']
    error = build_error_result(actual, CASE, RuntimeError('failed'))
    assert error['agent_type'] == 'simple'
    assert error['evaluation_identity'] == actual['evaluation_identity']


@pytest.mark.parametrize('config,expected', [
    ({'agent_type': 'simple', 'provider': 'openai', 'model': 'test-model'},
     'a39bc9d739968a2564b029e6a3f9bba984b066e5c0a47cf616309e82306a8fb2'),
    ({'provider': 'openai', 'model': 'test-model'},
     '0b8894134cb4283560e4351ec01f6d99dde337f442701cf3d6c999e418a01bcf'),
    ({'agent_type': 'simple', 'provider': 'codex', 'model': 'gpt-6.1-sol'},
     '8d953341101e328c32882dd86ff92dd17d9ff6fdd6080f6833f2c16d4d50e69e'),
])
def test_optional_litellm_does_not_change_existing_identity(config, expected):
    config = deepcopy(config)
    if config.get('agent_type') == 'simple':
        config['simple_agent'] = {'context': {'output_budget_policy': 'legacy_v1',
                                            'round_budget_policy': 'legacy_v1'}}
    assert build_evaluation_identity(config)['config_sha256'] == expected
    if config.get('agent_type') == 'simple':
        explicit = deepcopy(config)
        explicit.setdefault('simple_agent', {})['backend'] = 'native'
        assert build_evaluation_identity(explicit) == build_evaluation_identity(config)


@pytest.mark.parametrize('config,old_hash', [
    ({'agent_type': 'simple', 'provider': 'openai', 'model': 'test-model'},
     '047216dbe08c318aec600d7b3903e8f3d22bb7d5bc41f8007199527ffa4a6694'),
    ({'agent_type': 'simple', 'provider': 'codex', 'model': 'gpt-6.1-sol'},
     'd4f812358261971c9a2379b713cce25854bb1ea95a86e83570b1a9e42255316b'),
])
def test_explicit_message_chunks_retains_its_identity_after_default_migration(config, old_hash):
    historical = deepcopy(config)
    historical['simple_agent'] = {'context': {'compaction_policy': 'message_chunks_v1',
                                            'output_budget_policy': 'legacy_v1',
                                            'round_budget_policy': 'legacy_v1'}}
    assert build_evaluation_identity(historical)['config_sha256'] == old_hash
    assert build_evaluation_identity(config)['config_sha256'] != old_hash


@pytest.mark.parametrize('compaction,old_hash', [
    ('whole_turn_v1', 'e57a2f3f5b16b81648ce586f423e2356f61f6ae3c798826094dfa06284ac9bc8'),
    ('message_chunks_v1', '047216dbe08c318aec600d7b3903e8f3d22bb7d5bc41f8007199527ffa4a6694'),
    ('tool_rounds_v1', 'a39bc9d739968a2564b029e6a3f9bba984b066e5c0a47cf616309e82306a8fb2'),
])
def test_legacy_output_budget_preserves_each_previous_compaction_identity(compaction, old_hash):
    config = {'agent_type': 'simple', 'provider': 'openai', 'model': 'test-model',
              'simple_agent': {'context': {'compaction_policy': compaction,
                                           'output_budget_policy': 'legacy_v1',
                                           'round_budget_policy': 'legacy_v1'}}}
    assert build_evaluation_identity(config)['config_sha256'] == old_hash
    # This limit is inactive under legacy accounting and must not fork its identity.
    config['simple_agent']['context']['max_protocol_output_bytes'] = 17
    assert build_evaluation_identity(config)['config_sha256'] == old_hash
    config['simple_agent']['context']['output_budget_policy'] = 'separate_native_v1'
    assert build_evaluation_identity(config)['config_sha256'] != old_hash


def test_separate_native_protocol_budget_is_an_evaluation_condition():
    config = {'agent_type': 'simple', 'provider': 'openai', 'model': 'test-model'}
    changed = {**config, 'simple_agent': {'context': {'max_protocol_output_bytes': 500_000}}}
    assert build_evaluation_identity(config) != build_evaluation_identity(changed)


@pytest.mark.parametrize('compaction,output_policy,old_hash', [
    ('whole_turn_v1', 'legacy_v1',
     'e57a2f3f5b16b81648ce586f423e2356f61f6ae3c798826094dfa06284ac9bc8'),
    ('message_chunks_v1', 'legacy_v1',
     '047216dbe08c318aec600d7b3903e8f3d22bb7d5bc41f8007199527ffa4a6694'),
    ('tool_rounds_v1', 'legacy_v1',
     'a39bc9d739968a2564b029e6a3f9bba984b066e5c0a47cf616309e82306a8fb2'),
    ('whole_turn_v1', 'separate_native_v1',
     '4af2af1e523251928428bbc1fa2c78ac3572041418cba75ed706a34e132e0c84'),
    ('message_chunks_v1', 'separate_native_v1',
     '28ffc9c9e453281b3a7a91ea7badff9d2244a2cfd300314509b5238f7bdc4d7a'),
    ('tool_rounds_v1', 'separate_native_v1',
     '59acda90bfa91630c219713297ffd69c613fcbea0dede7e4cb05300b40140040'),
])
def test_legacy_round_notices_preserve_all_previous_policy_fingerprints(compaction, output_policy, old_hash):
    config = {'agent_type': 'simple', 'provider': 'openai', 'model': 'test-model',
              'simple_agent': {'context': {'compaction_policy': compaction,
                                           'output_budget_policy': output_policy,
                                           'round_budget_policy': 'legacy_v1'}}}
    assert build_evaluation_identity(config)['config_sha256'] == old_hash
    config['simple_agent']['context']['round_budget_policy'] = 'visible_v1'
    assert build_evaluation_identity(config)['config_sha256'] != old_hash


@pytest.mark.parametrize('enabled', [False, True])
def test_round_notice_policy_is_an_explicit_condition_even_without_builtin_tools(enabled):
    config = {'agent_type': 'simple', 'provider': 'openai', 'model': 'test-model',
              'simple_agent': {'tools': {'enabled': enabled}}}
    visible = deepcopy(config)
    visible['simple_agent']['context'] = {'round_budget_policy': 'visible_v1'}
    legacy = deepcopy(config)
    legacy['simple_agent']['context'] = {'round_budget_policy': 'legacy_v1'}
    assert build_evaluation_identity(config) == build_evaluation_identity(visible)
    assert build_evaluation_identity(config) != build_evaluation_identity(legacy)


@pytest.mark.parametrize('agent_type,provider', [('api', 'openai'), ('codex', 'codex')])
def test_simple_round_notice_settings_do_not_change_other_harness_identity(agent_type, provider):
    config = {**API, 'agent_type': agent_type, 'provider': provider}
    original = build_evaluation_identity(config)
    for policy in ('visible_v1', 'legacy_v1'):
        unrelated = {**config, 'simple_agent': {'context': {'round_budget_policy': policy}}}
        assert build_evaluation_identity(unrelated) == original


@pytest.mark.parametrize('parallel', [False, True])
def test_pre_notice_separate_budget_record_resumes_only_explicit_legacy_round_policy(tmp_path, parallel):
    config = {**API, 'agent_type': 'simple'}
    legacy = {**config, 'simple_agent': {'context': {'round_budget_policy': 'legacy_v1'}}}
    old = {**row(), 'model': 'test-model', 'provider': 'openai', 'agent_type': 'simple',
           'evaluation_identity': {'version': 1, 'agent_type': 'simple',
                                  'config_sha256': '59acda90bfa91630c219713297ffd69c613fcbea0dede7e4cb05300b40140040'}}
    path = tmp_path / 'results.jsonl'
    save(path, [old])
    for selected, expected in [(legacy, 1), (config, 0)]:
        if parallel:
            pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [selected]), str(path))
            assert len(pending) == 1 - expected
        else:
            existing = load_existing_eval_results(str(path), [CASE], [runtime(selected)])
        assert len(existing) == expected
    assert read(path) == [old]


def test_litellm_identity_distinguishes_protocol_model_and_provider_default_url():
    config = deepcopy(SIMPLE)
    config['simple_agent'].update(backend='litellm', litellm={
        'model': 'openai/gpt-6-astra', 'api': 'responses'})
    original = build_evaluation_identity(config)
    assert original != build_evaluation_identity(SIMPLE)
    for change in (
        {'model': 'openai/gpt-6.1-sol', 'api': 'responses'},
        {'model': 'openai/gpt-6-astra', 'api': 'chat_completions'},
    ):
        changed = deepcopy(config)
        changed['simple_agent']['litellm'] = change
        assert build_evaluation_identity(changed) != original
    changed = deepcopy(config)
    changed['base_url'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert build_evaluation_identity(changed) != original
    alias = deepcopy(config)
    alias['simple_agent']['litellm']['model'] = 'openai/responses/gpt-6-astra'
    assert build_evaluation_identity(alias) == original


@pytest.mark.parametrize('parallel', [False, True])
def test_litellm_results_cannot_resume_native_chat_cases(tmp_path, parallel):
    config = deepcopy(SIMPLE)
    config['simple_agent'].update(backend='litellm', litellm={
        'model': 'openai/gpt-6-astra', 'api': 'responses'})
    path = tmp_path / 'results.jsonl'
    save(path, [row(config)])
    if parallel:
        pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [SIMPLE]), str(path))
        assert len(pending) == 1
    else:
        existing = load_existing_eval_results(str(path), [CASE], [runtime(SIMPLE)])
    assert not existing


def test_unknown_or_invalid_record_identity_does_not_crash_or_resume(tmp_path):
    path = tmp_path / 'results.jsonl'
    save(path, [
        {**row(), 'agent_type': 'retired-plugin'},
        {**row(), 'evaluation_identity': {'version': 2}},
    ])
    assert not load_existing_eval_results(str(path), [CASE], [runtime(API)])


def test_ambiguous_legacy_row_cannot_choose_between_new_configurations(tmp_path):
    path = tmp_path / 'results.jsonl'
    save(path, [row()])
    changed = {**API, 'temperature': 0.7}
    assert not load_existing_eval_results(str(path), [CASE], [runtime(API), runtime(changed)])


def test_legacy_row_with_known_different_output_budget_does_not_resume(tmp_path):
    path = tmp_path / 'results.jsonl'
    save(path, [{**row(), 'max_completion_tokens': 4096}])
    assert not load_existing_eval_results(str(path), [CASE], [runtime(API)])


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('same_endpoint', [False, True])
def test_legacy_endpoint_is_compared_without_credentials_or_query(tmp_path, parallel, same_endpoint):
    path = tmp_path / 'results.jsonl'
    save(path, [{**row(), 'base_url': 'https://old-user:old-secret@proxy.example/api/v1/?key=old'}])
    endpoint = ('https://new-user:new-secret@proxy.example/api/v1?key=new'
                if same_endpoint else 'https://another.example/api/v1')
    cfg = {**API, 'base_url': endpoint}
    if parallel:
        pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [cfg]), str(path))
        assert bool(pending) is not same_endpoint
    else:
        existing = load_existing_eval_results(str(path), [CASE], [runtime(cfg)])
    assert bool(existing) is same_endpoint


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('field,old,new', [
    ('thinking', {'type': 'enabled'}, {'type': 'disabled'}),
    ('reasoning_effort', 'low', 'high'),
    ('max_completion_tokens', 4096, 8192),
])
def test_known_legacy_behavior_mismatch_never_skips(tmp_path, parallel, field, old, new):
    path = tmp_path / 'results.jsonl'
    save(path, [{**row(), field: old}])
    cfg = {**API, field: new}
    if parallel:
        pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [cfg]), str(path))
        assert len(pending) == 1
    else:
        existing = load_existing_eval_results(str(path), [CASE], [runtime(cfg)])
    assert not existing


@pytest.mark.parametrize('parallel', [False, True])
def test_missing_legacy_behavior_remains_unknown_and_thinking_aliases_match(tmp_path, parallel):
    path = tmp_path / 'results.jsonl'
    cfg = {**API, 'thinking': True, 'reasoning_effort': 'high'}
    for legacy in [row(), {**row(), 'thinking': {'type': 'enabled'}}]:
        save(path, [legacy])
        if parallel:
            pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [cfg]), str(path))
            assert not pending
        else:
            existing = load_existing_eval_results(str(path), [CASE], [runtime(cfg)])
        assert len(existing) == 1


def normalized_serial_runtime(config):
    """Exercise production serial normalization without constructing clients."""
    from agent_ark.agent.api_agent import APIAgent as OriginalAPIAgent

    class FakeAgent:
        _normalize_max_completion_tokens = staticmethod(OriginalAPIAgent._normalize_max_completion_tokens)

        def __init__(self, **kwargs):
            self.thinking_type = OriginalAPIAgent._normalize_thinking_type(kwargs.get('thinking'))
            self.reasoning_effort = OriginalAPIAgent._normalize_reasoning_effort(kwargs.get('reasoning_effort'))
            self.max_completion_tokens = kwargs.get('max_completion_tokens')

    with patch('agent_ark.ark_eval.run_api_agent.APIAgent', FakeAgent), \
            patch('agent_ark.ark_eval.run_api_agent.CodexAgent', FakeAgent), \
            patch('agent_ark.agent.agent_registry.AgentType.load_class', return_value=FakeAgent):
        return build_model_runtimes([{**config, 'api_key': 'fake'}])[0]


@pytest.mark.parametrize('config', [
    {'name': API['name'], 'provider': 'codex'},
    {**API, 'provider': 'codex', 'codex_context_mode': 'default'},
    {**API, 'provider': 'codex', 'agent_type': 'simple'},
    {**API, 'agent_type': 'codex', 'provider': 'auto', 'effort': 'HIGH'},
    {**API, 'provider': 'codex', 'codex_context_mode': 'LEAN',
     'codex_disabled_mcp_servers': [' node_repl ', 'other', 'other']},
    {**API, 'temperature': '0.7', 'thinking': True,
     'reasoning_effort': 'HIGH', 'max_completion_tokens': '8192'},
])
def test_parallel_metadata_matches_production_serial_effective_values(config):
    serial, parallel = normalized_serial_runtime(config), runtime(config)
    for key in ('model', 'provider', 'base_url', 'api_key_env', 'temperature', 'thinking',
                'reasoning_effort', 'max_completion_tokens', 'thread_mode',
                'codex_context_mode', 'codex_disabled_mcp_servers'):
        assert serial.get(key) == parallel.get(key), key


@pytest.mark.parametrize('old,new,expected', [
    ({'provider': 'codex'}, {'provider': 'codex'}, True),
    ({'provider': 'codex', 'codex_context_mode': 'default'},
     {'provider': 'codex', 'codex_context_mode': 'default'}, True),
    ({'provider': 'codex', 'codex_context_mode': 'default'},
     {'provider': 'codex', 'codex_context_mode': 'lean', 'max_completion_tokens': None}, False),
    ({'provider': 'codex', 'codex_context_mode': 'lean', 'codex_disabled_mcp_servers': ['one']},
     {'provider': 'codex', 'codex_context_mode': 'lean', 'codex_disabled_mcp_servers': ['two']}, False),
    ({'provider': 'codex', 'codex_context_mode': 'default', 'codex_disabled_mcp_servers': ['one']},
     {'provider': 'codex', 'codex_context_mode': 'default', 'codex_disabled_mcp_servers': ['two']}, True),
    ({'provider': 'codex', 'reasoning_effort': 'low'},
     {'provider': 'codex', 'reasoning_effort': 'high'}, False),
    ({'provider': 'codex', 'thread_mode': 'per_turn'},
     {'provider': 'codex', 'thread_mode': 'per_agent'}, False),
    ({'provider': 'codex', 'temperature': 0.2},
     {'provider': 'codex', 'temperature': 0.8}, True),
    ({'temperature': 0.2}, {'temperature': 0.8}, False),
])
def test_real_legacy_rows_have_serial_parallel_resume_parity(tmp_path, old, new, expected):
    path = tmp_path / 'results.jsonl'
    old_cfg = {'name': API['name'], **old}
    new_cfg = {'name': API['name'], **new}
    if old_cfg.get('provider') != 'codex':
        old_cfg['model'] = new_cfg['model'] = 'test-model'
    old_runtime = normalized_serial_runtime(old_cfg)
    # These are precisely the known fields written by historical result rows,
    # including null values for unsupported legacy Codex output/sampling caps.
    legacy = {**row(), **{key: old_runtime.get(key) for key in (
        'model', 'provider', 'base_url', 'temperature', 'thinking', 'reasoning_effort',
        'max_completion_tokens', 'codex_context_mode', 'codex_disabled_mcp_servers',
    )}}
    if 'thread_mode' in old_runtime:
        legacy['thread_mode'] = old_runtime['thread_mode']
    save(path, [legacy])
    serial_existing = load_existing_eval_results(str(path), [CASE], [normalized_serial_runtime(new_cfg)])
    pending, parallel_existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [new_cfg]), str(path))
    assert bool(serial_existing) is expected
    assert bool(parallel_existing) is expected
    assert bool(pending) is not expected


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('old_url,new_url,expected', [
    ('https://different-proxy.example/v1', None, False),
    ('https://different-proxy.example/v1', '', False),
    ('https://different-proxy.example/v1', '  ', False),
    (None, 'https://different-proxy.example/v1', False),
    ('', 'https://different-proxy.example/v1', False),
    ('  ', 'https://different-proxy.example/v1', False),
    (None, None, True),
    ('', 'https://dashscope.aliyuncs.com/compatible-mode/v1/', True),
    ('https://dashscope.aliyuncs.com/compatible-mode/v1', None, True),
])
def test_known_legacy_default_http_endpoint_is_not_unknown(tmp_path, parallel, old_url, new_url, expected):
    path = tmp_path / 'results.jsonl'
    save(path, [{**row(), 'base_url': old_url}])
    cfg = dict(API)
    if new_url is not None:
        cfg['base_url'] = new_url
    if parallel:
        pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [cfg]), str(path))
        assert bool(pending) is not expected
    else:
        existing = load_existing_eval_results(str(path), [CASE], [normalized_serial_runtime(cfg)])
    assert bool(existing) is expected


@pytest.mark.parametrize('parallel', [False, True])
def test_missing_historical_endpoint_remains_unknown_and_codex_ignores_url(tmp_path, parallel):
    path = tmp_path / 'results.jsonl'
    for cfg, legacy in [
        ({**API, 'base_url': 'https://different-proxy.example/v1'}, row()),
        ({'name': API['name'], 'provider': 'codex'},
         {**row(), 'provider': 'codex', 'base_url': 'https://ignored.example/v1'}),
    ]:
        save(path, [legacy])
        if parallel:
            pending, existing = filter_existing_parallel_jobs(build_parallel_eval_jobs([CASE], [cfg]), str(path))
            assert not pending
        else:
            existing = load_existing_eval_results(str(path), [CASE], [runtime(cfg)])
        assert len(existing) == 1


@pytest.mark.parametrize('agent_type', ['api', 'simple'])
def test_default_endpoint_has_same_fingerprint_as_explicit_default(agent_type):
    minimal = {**API, 'agent_type': agent_type}
    expected = build_evaluation_identity(minimal)
    for endpoint in (None, '', 'https://dashscope.aliyuncs.com/compatible-mode/v1/'):
        assert build_evaluation_identity({**minimal, 'base_url': endpoint}) == expected
    if agent_type == 'api':
        assert build_evaluation_identity({**minimal, 'base_url': '  '}) == expected
