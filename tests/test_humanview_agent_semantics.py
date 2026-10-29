"""HumanView distinguishes public replies, environment actions and effective instructions."""
from copy import deepcopy
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from agent_ark.interaction.local_viewer import _VIEWER_HTML


NODE = shutil.which('node')
HARNESS = Path(__file__).with_name('viewer_dom_harness.js')
pytestmark = pytest.mark.skipif(not NODE, reason='Node.js is required for production viewer DOM checks')
ACTION = '<tool_call>{"name":"MoveRight","arguments":{"steps":3}}</tool_call>'
RATIONALE = 'Move right three cells toward the food.'


def action_call(*, instruction='Actual native harness policy', rationale=RATIONALE):
    return {'purpose': 'action', 'status': 'completed',
            'request_messages': [{'role': 'system', 'content': instruction},
                                 {'role': 'developer', 'content': 'Adapted task instructions: use submit_action.'},
                                 {'role': 'user', 'content': 'CURRENT_OBSERVATION_IS_NOT_A_PIN'}],
            'completion': rationale, 'tool_calls': [{'name': 'submit_action', 'arguments_json': '{}'}]}


def response(seq=1, *, action=ACTION, assistant=None, rationale=RATIONALE,
             case='seed-1', model='simple-model', agent='0', calls=None, native=True):
    trace = {'assistant_raw': rationale, 'action_extracted': action,
             'model_requests': deepcopy(calls if calls is not None else [action_call(rationale=rationale)])}
    if native:
        trace['simple_agent'] = {'native_action_submission': True}
    return {'seq': seq, 'event': 'agent_response', 'payload': {
        'case_id': case, 'model_name': model, 'turn_index': seq-1,
        'actions': {agent: {'action': action, 'assistant': action if assistant is None else assistant}},
        'raw_trace_by_agent': {agent: trace},
    }}


def render(events=(), *, state_events=(), clear_at=(), expand=True):
    completed = subprocess.run([NODE, str(HARNESS)],
        input=json.dumps({'html': _VIEWER_HTML, 'events': events, 'state_events': state_events,
                          'clear_at': clear_at, 'expand_model_calls': expand}),
        text=True, encoding='utf-8', capture_output=True, check=True, timeout=20)
    return json.loads(completed.stdout)


def rows(result, role):
    return [row['html'] for row in result['rows'] if row['className'] == 'bubble-row ' + role]


def instructions(result):
    return '\n'.join(card['html'] for card in result['model_instructions'])


def test_native_public_content_and_exact_action_share_one_assistant_card():
    # Real SimpleAgent payload.assistant echoes the translated action, while
    # the last action completion records the public explanation.
    result = render([response()])
    cards = rows(result, 'assistant')
    assert len(cards) == 1
    assert 'Assistant content' in cards[0] and 'Environment action' in cards[0]
    assert RATIONALE in cards[0]
    assert html.escape(ACTION, quote=False) in cards[0]
    assert cards[0].count(html.escape(ACTION, quote=False)) == 1


def test_response_with_separate_public_payload_content_still_shows_its_action():
    result = render([response(assistant=RATIONALE)])
    assert RATIONALE in rows(result, 'assistant')[0]
    assert html.escape(ACTION, quote=False) in rows(result, 'assistant')[0]


@pytest.mark.parametrize(('action', 'rationale'), [
    ('1', 'I will choose option 1.'),
    ('0', 'I will choose option 0.'),
    ('left', 'I will move left.'),
    ('left, 0.25', 'I will use left, 0.25 to move cautiously.'),
    ('1', '1'),
    (ACTION, ACTION),
])
def test_native_short_or_identical_content_never_merges_with_exact_environment_action(action, rationale):
    card = rows(render([response(action=action, rationale=rationale)]), 'assistant')[0]
    assert 'Assistant reply / Environment action (combined)' not in card
    content = re.search(r'Assistant content</div>\s*<pre>([\s\S]*?)</pre>', card)
    environment_action = re.search(r'Environment action</div>\s*<pre>([\s\S]*?)</pre>', card)
    assert content is not None and content.group(1) == html.escape(rationale, quote=False)
    assert environment_action is not None and environment_action.group(1) == html.escape(action, quote=False)


def test_native_action_echo_is_suppressed_using_the_original_payload_assistant():
    echo = {'seq': 2, 'event': 'env_step', 'payload': {'message_mode': 'delta',
        'observation_id': 'after-one', 'next_obs': {'0': {'messages': [
            {'role': 'assistant', 'content': ACTION}, {'role': 'user', 'content': 'Step reward=1'}]}}}}
    result = render([response(), echo])
    assert len(rows(result, 'assistant')) == 1
    assert 'Step reward=1' in '\n'.join(rows(result, 'user'))


def test_empty_public_content_does_not_hide_a_selected_environment_action():
    result = render([response(rationale='')])
    cards = rows(result, 'assistant')
    assert len(cards) == 1 and 'Environment action' in cards[0]
    assert cards[0].count(html.escape(ACTION, quote=False)) == 1


@pytest.mark.parametrize('action', [None, ''])
def test_known_none_and_empty_actions_are_not_guessed_from_assistant_text(action):
    result = render([response(action=action, assistant=RATIONALE)])
    card = rows(result, 'assistant')[0]
    assert card.count(RATIONALE) == 1
    assert 'Environment action' in card
    assert re.search('none|no (?:environment )?action|not selected|empty', card, re.I)


def test_legacy_combined_reply_keeps_original_reasoning_without_repeating_action():
    combined = '<think>Legacy public explanation.</think>' + ACTION
    result = render([response(native=False, assistant=combined, rationale=combined, calls=[])])
    card = rows(result, 'assistant')[0]
    assert html.escape(combined, quote=False) in card
    assert card.count(html.escape(ACTION, quote=False)) == 1


def test_matching_truncated_action_preview_can_use_the_saved_complete_translation():
    full = 'CODE_START\n' + 'code_line();\n' * 1800 + 'CODE_END'
    event = response(action=full[:12000]+'\n...[truncated]', assistant=RATIONALE)
    event['payload']['raw_trace_by_agent']['0']['action_extracted'] = full
    card = rows(render([event]), 'assistant')[0]
    assert 'CODE_START' in card and 'CODE_END' in card
    assert html.escape(full, quote=False) in card


def test_conflicting_trace_cannot_silently_replace_a_truncated_payload_action():
    event = response(action='THE_SUBMITTED_PREVIEW\n...[truncated]', assistant=RATIONALE)
    event['payload']['raw_trace_by_agent']['0']['action_extracted'] = 'DIFFERENT_ACTION_NEVER_SUBMITTED'
    card = rows(render([event]), 'assistant')[0]
    assert 'THE_SUBMITTED_PREVIEW' in card and 'truncated' in card
    assert 'DIFFERENT_ACTION_NEVER_SUBMITTED' not in card


def test_error_without_selected_actions_does_not_invent_execution_from_failure_trace():
    event = response(action='NEVER_EXECUTED')
    event['payload'].update(status='error', actions={})
    result = render([event])
    assert not rows(result, 'assistant')
    assert 'NEVER_EXECUTED' not in '\n'.join(row['html'] for row in result['rows'])


def test_model_instruction_cards_use_action_calls_not_summary_or_request_preview():
    summary = {'purpose': 'summary', 'status': 'completed', 'completion': 'Useful notes',
               'request_messages': [{'role': 'system', 'content': 'SUMMARY_SYSTEM_NOT_THE_ACTION_POLICY'}]}
    call = action_call()
    call['request_messages'].append({'role': 'system', 'content': 'LATE_SYSTEM_NOT_A_LEADING_PIN'})
    request = {'seq': 1, 'event': 'agent_request', 'payload': {'raw_trace_by_agent': {'0': {
        'request_messages': [{'role': 'system', 'content': 'PRECOMPACTION_PREVIEW_NOT_ACTUAL'}]}}}}
    result = render([request, response(2, calls=[summary, call])])
    card = instructions(result)
    assert 'Actual native harness policy' in card
    assert 'Adapted task instructions: use submit_action.' in card
    assert 'SUMMARY_SYSTEM_NOT_THE_ACTION_POLICY' not in card
    assert 'PRECOMPACTION_PREVIEW_NOT_ACTUAL' not in card
    assert 'CURRENT_OBSERVATION_IS_NOT_A_PIN' not in card
    assert 'LATE_SYSTEM_NOT_A_LEADING_PIN' not in card


def test_original_environment_prompt_is_kept_separate_and_collapsible():
    raw = '<system_prompt>Return a &lt;think&gt; block and a &lt;tool_call&gt; envelope.</system_prompt>\nTask objective.'
    reset = {'seq': 1, 'event': 'env_reset', 'payload': {'message_mode': 'delta', 'obs': {
        '0': {'step_msg': raw, 'images': []}}}}
    result = render([reset, response(2)])
    environment = '\n'.join(rows(result, 'user'))
    assert 'Environment original system prompt' in environment
    assert '<details' in environment and 'Task objective.' in environment
    assert html.escape(raw.split('\n')[0], quote=False) in environment
    assert 'Actual native harness policy' in instructions(result)
    assert html.escape(raw.split('\n')[0], quote=False) not in instructions(result)


def test_codex_instruction_card_includes_outer_protocol_without_repeating_observations():
    call = action_call()
    call['request_transport'] = {
        'transport': 'codex_sdk_structured_json', 'boundary': 'sdk_turn_input',
        'inputs': [{'type': 'text', 'text': 'OUTER_JSON_WRAPPER: return text and tool_calls.'},
                   {'type': 'text', 'text': 'BEGIN SUPPLIED MESSAGE {"role":"system"}'},
                   {'type': 'text', 'text': 'DO_NOT_DUPLICATE_ALL_SDK_MESSAGE_INPUTS'}],
        'output_schema': {'type': 'object', 'properties': {'text': {'type': 'string'}, 'tool_calls': {'type': 'array'}}},
        'thread_base_instructions': 'SAVED_THREAD_BASE_INSTRUCTION',
        'thread_developer_instructions': 'SAVED_THREAD_DEVELOPER_INSTRUCTION',
        'sdk_added_context': 'not_exposed_by_sdk',
    }
    card = instructions(render([response(calls=[call])]))
    for value in ('OUTER_JSON_WRAPPER', 'text', 'tool_calls', 'SAVED_THREAD_BASE_INSTRUCTION',
                  'SAVED_THREAD_DEVELOPER_INSTRUCTION', 'output_schema', 'not_exposed_by_sdk'):
        assert value in card
    assert 'DO_NOT_DUPLICATE_ALL_SDK_MESSAGE_INPUTS' not in card


def test_simple_agent_tools_disabled_still_shows_recorded_additional_instructions():
    event = response(calls=[action_call(instruction='Environment base policy')])
    event['payload']['raw_trace_by_agent']['0']['simple_agent'] = {'native_action_submission': False, 'tools_enabled': False}
    event['payload']['raw_trace_by_agent']['0']['model_requests'][0]['request_messages'][1]['content'] = 'Additional SimpleAgent memory instructions'
    assert 'Additional SimpleAgent memory instructions' in instructions(render([event]))


def test_instruction_cards_are_lazy_deduplicated_and_isolated_by_scope():
    events = [response(1), response(2), response(3, case='seed-2'),
              response(4, case='seed-2', model='other-model'),
              response(5, case='seed-2', model='other-model', agent='8')]
    result = render(events, expand=False)
    assert len(result['model_instructions']) == 4
    assert 'Actual native harness policy' not in instructions(result)
    expanded = render(events)
    assert len(expanded['model_instructions']) == 4
    assert 'Actual native harness policy' in instructions(expanded)
    assert 'seed-1' in expanded['model_instructions'][0]['html']
    assert 'seed-2' not in expanded['model_instructions'][0]['html']
    assert 'other-model' in expanded['model_instructions'][-1]['html']
    assert '8' in expanded['model_instructions'][-1]['html']


def test_instruction_changes_get_an_additional_card_and_clear_resets_deduplication():
    changed = response(3, calls=[action_call(instruction='UPDATED_HARNESS_POLICY')])
    result = render([response(1), response(2), changed])
    assert len(result['model_instructions']) == 2
    assert 'UPDATED_HARNESS_POLICY' in result['model_instructions'][1]['html']
    cleared = render([response(1), response(2)], clear_at=[1])
    assert len(cleared['model_instructions']) == 1


def test_bootstrap_and_sse_replay_do_not_repeat_instruction_or_action_cards():
    events = [{'seq': 1, 'event': 'run_start', 'payload': {}}, response(2)]
    result = render(events=events, state_events=events)
    assert len(rows(result, 'assistant')) == 1
    assert len(result['model_instructions']) == 1


def test_missing_actual_requests_never_invent_harness_instructions():
    event = response(calls=[])
    result = render([event])
    assert not result['model_instructions']
    assert re.search('not recorded|not saved|unavailable', result['model_instruction_help'], re.I)


def test_untrusted_explanation_action_and_harness_prompts_are_escaped():
    attack = '<script>untrusted()</script><img src=x onerror="bad()">'
    event = response(action=attack, rationale=attack, calls=[action_call(instruction=attack, rationale=attack)])
    result = render([event])
    body = '\n'.join(rows(result, 'assistant')) + instructions(result)
    assert '<script>' not in body and '<img src=x' not in body
    assert '&lt;script&gt;' in body


def test_context_notes_shown_in_chat_are_the_notes_actually_used_by_the_action():
    summary = {'purpose': 'summary', 'status': 'completed', 'completion': 'Fresh summary facts'}
    call = action_call()
    call['request_messages'].insert(2, {'role': 'user', 'content':
        'Model notes; not environment facts. Source archive entries 1,2:\nFresh summary facts'})
    result = render([response(calls=[summary, call])])
    context = '\n'.join(row for row in rows(result, 'system') if 'harness-context' in row)
    assert 'Fresh summary facts' in context
    assert 'Model notes; not environment facts.' in context


def test_failed_summary_attempt_with_old_notes_does_not_claim_new_compaction():
    summary = {'purpose': 'summary', 'status': 'failed', 'error_type': 'SummaryResponseError'}
    call = action_call()
    call['request_messages'].insert(2, {'role': 'user', 'content':
        'Model notes; not environment facts. Source archive entries 1,2:\nPreviously retained notes'})
    result = render([response(calls=[summary, call])])
    context = '\n'.join(row for row in rows(result, 'system') if 'harness-context' in row)
    assert 'Context compressed' not in context


def test_summary_output_not_committed_to_action_context_is_not_shown_as_memory():
    summary = {'purpose': 'summary', 'status': 'completed', 'completion': 'UNCOMMITTED_NOTES'}
    result = render([response(calls=[summary, action_call()])])
    assert not any('harness-context' in row for row in rows(result, 'system'))
