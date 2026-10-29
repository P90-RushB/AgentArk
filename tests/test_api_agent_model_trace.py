"""Actual HTTP-boundary trace contracts without sending model requests."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent_ark.agent.api_agent import APIAgent, LLMClient


def agent_with_responses(responses):
    pending = list(responses)
    wire = []

    def create(**kwargs):
        wire.append(deepcopy(kwargs))
        value = pending.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    client = object.__new__(LLMClient)
    client.model = 'trace-test-model'
    client.provider = 'openai'
    client.timeout_s = None
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                                     api_key='client-secret-must-not-be-logged')
    agent = object.__new__(APIAgent)
    agent.client = client
    agent.temperature = None
    agent.thinking_type = None
    agent.reasoning_effort = 'low'
    agent.max_completion_tokens = 1234
    return agent, wire


def completion(content='  Public explanation.\n<tool_call>{"name":"Tap","arguments":{}}</tool_call>  ', **fields):
    return SimpleNamespace(
        id='response-1', model='trace-test-model',
        choices=[SimpleNamespace(index=0, finish_reason='stop', message=SimpleNamespace(
            role='assistant', content=content, model_extra=fields))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19),
        headers={'authorization': 'response-header-must-not-be-logged'},
    )


def test_api_trace_matches_actual_request_and_keeps_original_response_fields():
    response = completion(reasoning_content='  native summary  ', encrypted_content='opaque\nbytes',
                          reasoning_details=[{'type': 'reasoning.encrypted', 'data': 'ciphertext'}],
                          arbitrary_credential='unknown-extra-must-not-be-logged')
    agent, wire = agent_with_responses([response])
    messages = [{'role': 'system', 'content': 'Original wrapper prompt <think> unchanged'},
                {'role': 'user', 'content': [{'type': 'text', 'text': 'State'},
                                           {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]}]
    original = deepcopy(messages)
    actions, traces = agent.forward_with_trace({7: {'messages': messages}})

    call = traces[7]['model_requests'][0]
    assert wire == [{'model': 'trace-test-model', 'messages': original, 'extra_body': {},
                     'reasoning_effort': 'low', 'max_completion_tokens': 1234}]
    assert call['request_messages'] == wire[0]['messages']
    assert call['parameters'] == {key: value for key, value in wire[0].items() if key != 'messages'}
    assert call['completion'] == response.choices[0].message.content.strip()
    assert call['response_metadata']['message'] == {
        'role': 'assistant', 'content': response.choices[0].message.content,
        'reasoning_content': '  native summary  ', 'encrypted_content': 'opaque\nbytes',
        'reasoning_details': [{'type': 'reasoning.encrypted', 'data': 'ciphertext'}],
    }
    assert call['response_metadata']['finish_reason'] == 'stop'
    assert call['usage'] == {'prompt_tokens': 12, 'completion_tokens': 7, 'total_tokens': 19}
    assert call['status'] == 'completed'
    assert actions[7]['action'] == '<tool_call>{"name":"Tap","arguments":{}}</tool_call>'
    assert actions[7]['assistant'].startswith('<think>\nnative summary\n</think>')
    assert 'must-not-be-logged' not in str(traces)
    assert messages == original
    call['request_messages'][0]['content'] = 'changed audit copy'
    assert wire[0]['messages'] == original


def test_api_trace_keeps_null_content_and_native_tool_call_without_new_execution():
    response = completion(None)
    response.choices[0].finish_reason = 'tool_calls'
    response.choices[0].message.tool_calls = [SimpleNamespace(
        id='call-1', type='function', function=SimpleNamespace(name='unhandled', arguments='{"x":1}'))]
    agent, _ = agent_with_responses([response])
    actions, traces = agent.forward_with_trace({0: {'messages': [{'role': 'user', 'content': 'state'}]}})
    call = traces[0]['model_requests'][0]
    assert call['response_metadata']['message']['content'] is None
    assert call['tool_calls'] == [{'id': 'call-1', 'type': 'function',
                                  'function': {'name': 'unhandled', 'arguments': '{"x":1}'}}]
    assert call['completion'] == ''
    # Legacy APIAgent still only extracts from returned text. Auditing does not
    # introduce native tool execution or convert a missing answer to an action.
    assert actions[0] == {'action': None, 'assistant': ''}


def test_api_trace_retains_every_explicit_empty_choice_retry(monkeypatch):
    monkeypatch.setenv('AGENTARK_EMPTY_CHOICES_RETRIES', '1')
    monkeypatch.setenv('AGENTARK_EMPTY_CHOICES_RETRY_DELAY_S', '0')
    agent, wire = agent_with_responses([
        SimpleNamespace(choices=[], usage={'prompt_tokens': 9, 'completion_tokens': 2}), completion(),
    ])
    _, traces = agent.forward_with_trace({0: {'messages': [{'role': 'user', 'content': 'state'}]}})
    calls = traces[0]['model_requests']
    assert [call['status'] for call in calls] == ['empty_choices', 'completed']
    assert [call['attempt'] for call in calls] == [1, 2]
    assert calls[0]['usage'] == {'prompt_tokens': 9, 'completion_tokens': 2}
    assert calls[0]['response_metadata']['choices'] == []
    assert [call['request_messages'] for call in calls] == [item['messages'] for item in wire]


def test_api_trace_exposes_failure_type_without_provider_exception_contents():
    agent, wire = agent_with_responses([TimeoutError('authorization=must-not-be-logged')])
    with pytest.raises(TimeoutError, match='authorization'):
        agent.forward_with_trace({0: {'messages': [{'role': 'user', 'content': 'state'}]}})
    call = agent.last_trace_by_agent[0]['model_requests'][0]
    assert call['status'] == 'failed'
    assert call['error_type'] == 'TimeoutError'
    assert call['request_messages'] == wire[0]['messages']
    assert 'must-not-be-logged' not in str(agent.last_trace_by_agent)
    agent.reset()
    assert agent.last_trace_by_agent == {}
