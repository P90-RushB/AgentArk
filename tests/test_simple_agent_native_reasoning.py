"""Offline regressions for opaque API reasoning replay; no HTTP, SDK or Unity."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent import AgentError, CallableModel, SimpleRuntime, SimpleRuntimeConfig
from agent_ark.agent.simple_agent.agent import SimpleAgent
from agent_ark.agent.simple_agent.codex_model import CodexModel
from agent_ark.agent.simple_agent.encoding import api_chat_messages, chat_messages, encoded_messages, image_from_pixels
from agent_ark.agent.simple_agent.models import ModelResponseError
from agent_ark.agent.simple_agent.runtime import _metrics
from agent_ark.agent.simple_agent.types import (
    AgentInputIdentity, AgentLimitError, AgentRequest, ArchiveEntry, Message,
    ModelCompletion, ModelRequest, NativeAssistantState, TextPart, ToolCall, ToolSpec,
)


ACTION = '<tool_call>{"name":"tap","arguments":{"x":3}}</tool_call>'
SCOPE = "openrouter:test-model"
PRIVATE = "OPAQUE_REASONING_MUST_NOT_BECOME_CONTENT"
DETAILS = [
    {"type": "reasoning.encrypted", "data": "opaque+/=\\n密文", "id": "enc-2", "format": "vendor-v2", "index": 8},
    {"type": "reasoning.text", "text": PRIVATE, "signature": "signed-A+/=", "index": 2},
]


def state(content="answer", *, scope=SCOPE, calls=(), **reasoning):
    payload = {"role": "assistant", "content": content, **(reasoning or {"reasoning_details": deepcopy(DETAILS)})}
    if calls:
        payload["tool_calls"] = [wire_call(call) for call in calls]
    return NativeAssistantState(scope, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def wire_call(call):
    return {"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": call.arguments_json}}


def response(content, *, calls=(), finish_reason=None, **fields):
    return {"choices": [{"finish_reason": finish_reason or ("tool_calls" if calls else "stop"), "message": {
        "content": content, **({"tool_calls": [wire_call(call) for call in calls]} if calls else {}), **fields}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}


class FakeChatClient:
    """The production API adapter calls this transport, without creating HTTP clients."""

    def __init__(self, respond):
        self.respond = respond
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return self.respond(kwargs)

    async def close(self):
        pass


def fake_agent(respond, **kwargs):
    agent = SimpleAgent(api_key="unused-fake-key", provider="openrouter",
                        base_url="https://openrouter.ai/api/v1", model="test/model", **kwargs)
    client = FakeChatClient(respond)
    agent._backend.client = client
    return agent, client


def request(number=1, text="visible observation"):
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "agent", 1, str(number)), (
        Message.text("system", "Use the visible observation."), Message.text("user", text)))


def texts(messages):
    return "\n".join(part.text for message in messages for part in message.parts if isinstance(part, TextPart))


def test_native_state_is_frozen_detached_and_constructor_defaults_remain_compatible():
    native = state()
    first = native.message
    first["reasoning_details"][0]["data"] = "mutated"
    first["content"] = "mutated"
    assert native.message["content"] == "answer"
    assert native.message["reasoning_details"] == DETAILS
    with pytest.raises(FrozenInstanceError):
        native.scope = "different"
    assert Message.text("assistant", "answer").native_assistant is None
    assert ModelCompletion("answer").native_assistant is None
    message = Message("assistant", (TextPart("answer"),), (), None, native)
    completion = ModelCompletion("answer", native_assistant=native)
    assert message.native_assistant is completion.native_assistant is native


@pytest.mark.parametrize("scope", ["", None, 1, True, "x" * 257])
def test_native_state_rejects_invalid_scope(scope):
    with pytest.raises((ValueError, TypeError)):
        state(scope=scope)


@pytest.mark.parametrize("payload", [
    [], None, {}, {"role": "user", "content": "x", "reasoning": "private"},
    {"role": "assistant", "reasoning": "private"},
    {"role": "assistant", "content": "x"},
    {"role": "assistant", "content": "x", "reasoning": ""},
    {"role": "assistant", "content": "x", "reasoning": []},
    {"role": "assistant", "content": "x", "reasoning_details": []},
    {"role": "assistant", "content": "x", "reasoning_details": ["private"]},
    {"role": "assistant", "content": "x", "reasoning_details": {}},
    {"role": "assistant", "content": "x", "reasoning": "private", "reasoning_details": DETAILS},
    {"role": "assistant", "content": "x", "reasoning_content": "private"},
    {"role": "assistant", "content": "x", "reasoning": "private", "unexpected": "injection"},
    {"role": "assistant", "content": 1, "reasoning": "private"},
    {"role": "assistant", "content": {}, "reasoning": "private"},
])
def test_native_state_rejects_non_assistant_or_non_reasoning_payloads(payload):
    with pytest.raises((ValueError, TypeError)):
        NativeAssistantState(SCOPE, json.dumps(payload))


@pytest.mark.parametrize("value", [None, {}, b"{}", "", "not JSON"])
def test_native_state_requires_json_string(value):
    with pytest.raises((ValueError, TypeError)):
        NativeAssistantState(SCOPE, value)


def test_native_state_scope_and_utf8_json_hard_bound_are_inclusive():
    limit = 1024 * 1024
    payload = {"role": "assistant", "content": None, "reasoning": ""}
    overhead = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    payload["reasoning"] = "密" + "x" * (limit - overhead - len("密".encode("utf-8")))
    exact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert len(exact.encode("utf-8")) == limit
    assert NativeAssistantState("s" * 256, exact).message == payload
    payload["reasoning"] += "x"
    with pytest.raises((ValueError, TypeError)):
        NativeAssistantState(SCOPE, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


@pytest.mark.parametrize("role", ["system", "developer", "user", "tool"])
def test_native_sidecar_is_assistant_only_and_environment_cannot_supply_it(role):
    native = state()
    with pytest.raises((ValueError, TypeError)):
        Message(role, (TextPart("answer"),), tool_call_id="call" if role == "tool" else None,
                native_assistant=native)
    with pytest.raises((ValueError, TypeError)):
        AgentRequest(request().identity, (Message("assistant", (TextPart("answer"),), native_assistant=native),))


@pytest.mark.parametrize("native", [{}, "json", 1])
def test_message_and_completion_reject_wrong_native_state_type(native):
    with pytest.raises((ValueError, TypeError)):
        Message("assistant", (TextPart("answer"),), native_assistant=native)
    with pytest.raises((ValueError, TypeError)):
        ModelCompletion("answer", native_assistant=native)


@pytest.mark.parametrize("content,wrong_text", [
    ("answer", "different answer"),
    (" answer ", " answer "),
    (None, "invented answer"),
    (" \n\t ", "invented answer"),
    ([{"text": "first"}, {"type": "text", "text": "second"}], "firstsecond"),
])
def test_native_content_projection_must_match_message_and_completion_body(content, wrong_text):
    native = state(content)
    with pytest.raises((ValueError, TypeError)):
        Message("assistant", (TextPart(wrong_text),), native_assistant=native)
    with pytest.raises((ValueError, TypeError)):
        ModelCompletion(wrong_text, native_assistant=native)


def test_native_message_cannot_hide_an_image_in_portable_body():
    native = state("answer")
    image = image_from_pixels(b"\x01\x02\x03", 1, 1)
    with pytest.raises((ValueError, TypeError)):
        Message("assistant", (TextPart("answer"), image), native_assistant=native)


def test_native_message_accepts_multiple_text_parts_with_the_same_concatenated_projection():
    native = state([{"text": " first "}, {"type": "text", "text": " second "}])
    message = Message("assistant", (TextPart("first\n"), TextPart("second")), native_assistant=native)
    assert "".join(part.text for part in message.parts) == native.content_text == "first\nsecond"
    assert api_chat_messages((message,), native_scope=SCOPE)[0] == native.message


@pytest.mark.parametrize("content,expected", [
    (None, ""),
    ("", ""),
    (" \n\t ", ""),
    (" \nanswer \t", "answer"),
    ([], ""),
    ([{"text": ""}, {"type": "text", "text": " \n "}], ""),
    ([{"text": " first "}, {"type": "text", "text": " \n "}, {"type": "text", "text": " second\t"}], "first\nsecond"),
    ([{"type": "text", "text": " answer "}], "answer"),
])
def test_native_content_projection_accepts_null_whitespace_and_text_lists_without_changing_raw_content(content, expected):
    native = state(content)
    assert native.content_text == expected
    message = Message("assistant", (TextPart(expected),), native_assistant=native)
    completion = ModelCompletion(expected, native_assistant=native)
    assert completion.text == expected
    assert native.message["content"] == content
    assert api_chat_messages((message,), native_scope=SCOPE)[0]["content"] == content
    assert chat_messages((message,))[0]["content"] == [{"type": "text", "text": expected}]
    sdk = SimpleNamespace(TextInput=lambda **kw: {"type": "text", **kw},
                          ImageInput=lambda **kw: {"type": "image", **kw})
    inputs = CodexModel._inputs(ModelRequest((message,)), sdk)
    assert any(item.get("text") == expected for item in inputs)


@pytest.mark.parametrize("content", [
    [7], [None], ["answer"], [{}], [{"text": 7}], [{"type": "text"}],
    [{"type": None, "text": "answer"}],
    [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
    [{"type": "image_url", "text": "answer"}],
    [{"type": "text", "text": "answer"}, {"unrelated": "not text"}],
])
def test_native_content_rejects_non_text_lists_in_state_and_actual_api_completion(content):
    with pytest.raises((ValueError, TypeError)):
        state(content)

    async def run():
        agent, _ = fake_agent(lambda _: response(content, reasoning="private native reasoning"))
        try:
            with pytest.raises(ModelResponseError):
                await agent._backend.complete(ModelRequest((Message.text("user", "visible"),)))
        finally:
            await agent._backend.close()
            agent.close()
    asyncio.run(run())


@pytest.mark.parametrize("field", ["call_id", "name", "arguments_json", "missing"])
def test_sidecar_tool_calls_must_match_trusted_call_fields(field):
    call = ToolCall("original-id", "echo", ' {"text":"exact spacing"} ')
    native = state(None, calls=(call,))
    changed = () if field == "missing" else (replace(call, **{field: "different"}),)
    with pytest.raises((ValueError, TypeError)):
        Message("assistant", (TextPart(""),), changed, native_assistant=native)
    with pytest.raises((ValueError, TypeError)):
        ModelCompletion("", tool_calls=changed, native_assistant=native)


def test_native_encoding_replays_only_matching_scope_and_is_detached_from_state():
    call = ToolCall("exact-id", "echo", ' { "text" : "原始\\ntext" } ')
    native = state(" \n\t ", calls=(call,))
    message = Message("assistant", (), (call,), native_assistant=native)
    messages = (Message.text("user", "visible"), message,
                Message("tool", (TextPart("result"),), tool_call_id=call.call_id))
    encoded = api_chat_messages(messages, native_scope=SCOPE)
    assert encoded[1] == native.message
    assert encoded[2]["tool_call_id"] == call.call_id
    encoded[1]["reasoning_details"][0]["data"] = "mutated"
    assert api_chat_messages(messages, native_scope=SCOPE)[1] == native.message
    wire = chat_messages(messages)
    assert all("reasoning" not in item and "reasoning_details" not in item for item in wire)
    assert PRIVATE not in json.dumps(wire)
    assert wire[1]["tool_calls"] == [wire_call(call)]
    with pytest.raises(ModelResponseError, match="(?i)scope|endpoint|model"):
        api_chat_messages(messages, native_scope="other-model")


@pytest.mark.parametrize("raw_content", [None, "", " \n\t "])
def test_fake_api_host_tool_and_submission_replay_exact_details_across_env_steps(raw_content):
    host = ToolCall("host-opaque-id", "echo", ' { "text" : "visible result" } ')
    submit = ToolCall("submit-opaque-id", "submit_action", json.dumps({
        "format": "tool_call", "content": ACTION[11:-12]}, separators=(",", ":")))
    next_submit = replace(submit, call_id="next-submit")
    originals = [response(raw_content, calls=(host,), reasoning_details=deepcopy(DETAILS), reasoning="do not duplicate"),
                 response(raw_content, calls=(submit,), reasoning_details=list(reversed(deepcopy(DETAILS))), reasoning_content="do not duplicate"),
                 response(None, calls=(next_submit,), reasoning="next native reasoning")]
    pristine = deepcopy(originals)
    executions = []

    def setup(registry):
        registry.register(ToolSpec("echo", "Return own visible text.",
            '{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}'),
            lambda args, session: (executions.append(args["text"]) or TextPart(args["text"]),))

    agent, client = fake_agent(lambda _: originals.pop(0), tool_setup=setup)
    try:
        result, traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "visible step 1"}]}})
        assert result[0]["action"] == ACTION
        assert executions == ["visible result"]
        assert len(client.requests) == 2
        expected_host = {"role": "assistant", **pristine[0]["choices"][0]["message"]}
        expected_host.pop("reasoning")
        assert next(m for m in client.requests[1]["messages"] if m.get("tool_calls")) == expected_host
        assert traces[0]["model_requests"][1]["request_messages"] == client.requests[1]["messages"]
        assert PRIVATE not in result[0]["assistant"]
        agent.forward({0: {"messages": [{"role": "user", "content": "actual environment feedback: tapped"}]}})
        final_messages = client.requests[2]["messages"]
        assistant_calls = [m for m in final_messages if m.get("tool_calls")]
        expected_submission = {"role": "assistant", **pristine[1]["choices"][0]["message"]}
        expected_submission.pop("reasoning_content")
        assert assistant_calls == [expected_host, expected_submission]
        assert all("reasoning" not in m and "reasoning_content" not in m for m in assistant_calls)
        assert final_messages[-1]["role"] == "user"
        assert "actual environment feedback" in str(final_messages[-1]["content"])
        assert any(m["role"] == "tool" and m["tool_call_id"] == submit.call_id
                   and "not executed yet" in m["content"] for m in final_messages)
        archive = agent._states[0].runtime.history_snapshot
        assert next(e for e in archive if e.kind == "tool_exchange").messages[0].native_assistant.message == expected_host
        assert next(e for e in archive if e.kind == "completion").messages[0].native_assistant.message == expected_submission
    finally:
        agent.close()


@pytest.mark.parametrize("fields", [
    {"reasoning": "  original native reasoning\n"},
    {"reasoning_content": "  original native reasoning\n"},
    {"model_extra": {"reasoning": "  original native reasoning\n"}},
    {"model_extra": {"reasoning_content": "  original native reasoning\n"}},
])
def test_reasoning_aliases_replay_under_reasoning_without_content_think_wrappers(fields):
    agent, client = fake_agent(lambda _: response(" \n" + ACTION + " \t", **fields))
    try:
        first, _ = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "visible"}]}})
        assert first[0] == {"action": ACTION, "assistant": ACTION}
        agent.forward({0: {"messages": [{"role": "user", "content": "next feedback"}]}})
        assistant = next(m for m in client.requests[-1]["messages"] if m["role"] == "assistant")
        assert assistant == {"role": "assistant", "content": " \n" + ACTION + " \t",
                             "reasoning": "  original native reasoning\n"}
        assert "<think>" not in json.dumps(client.requests[-1]["messages"])
    finally:
        agent.close()


def test_structured_response_model_extra_details_remain_opaque_and_take_precedence():
    async def run():
        message = SimpleNamespace(content=ACTION, tool_calls=None, refusal=None,
            reasoning="must not duplicate", model_extra={"reasoning_details": deepcopy(DETAILS)})
        agent, _ = fake_agent(lambda _: SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)], usage=None))
        try:
            completion = await agent._backend.complete(ModelRequest((Message.text("user", "visible"),)))
            assert completion.text == ACTION
            assert completion.native_assistant.message == {"role": "assistant", "content": ACTION,
                                                           "reasoning_details": DETAILS}
            message.model_extra["reasoning_details"][0]["data"] = "mutated source"
            assert completion.native_assistant.message["reasoning_details"] == DETAILS
        finally:
            await agent._backend.close()
            agent.close()
    asyncio.run(run())


@pytest.mark.parametrize("details", [None, []])
@pytest.mark.parametrize("in_extra", [False, True])
@pytest.mark.parametrize("reasoning", [None, "original native reasoning"])
def test_absent_or_empty_details_fall_back_to_reasoning_or_create_no_sidecar(details, in_extra, reasoning):
    async def run():
        fields = {"reasoning_details": details}
        if reasoning is not None:
            fields["reasoning_content"] = reasoning
        if in_extra:
            fields = {"model_extra": fields}
        agent, _ = fake_agent(lambda _: response(ACTION, **fields))
        try:
            completion = await agent._backend.complete(ModelRequest((Message.text("user", "visible"),)))
            assert completion.text == ACTION
            if reasoning is None:
                assert completion.native_assistant is None
            else:
                assert completion.native_assistant.message == {"role": "assistant", "content": ACTION,
                                                               "reasoning": reasoning}
        finally:
            await agent._backend.close()
            agent.close()
    asyncio.run(run())


@pytest.mark.parametrize("content", [None, "", " \n\t ", []])
def test_reasoning_only_completion_without_tools_cannot_supply_action(content):
    agent, client = fake_agent(lambda _: response(content, reasoning=ACTION))
    try:
        with pytest.raises(ModelResponseError):
            agent.forward({0: {"messages": [{"role": "user", "content": "visible"}]}})
        assert len(client.requests) == 1
        assert all(e.kind != "completion" for e in agent._states[0].runtime.history_snapshot)
    finally:
        agent.close()


def test_no_reasoning_completion_retains_ordinary_behavior_without_sidecar():
    async def run():
        agent, _ = fake_agent(lambda _: response(" \n" + ACTION + " "))
        try:
            completion = await agent._backend.complete(ModelRequest((Message.text("user", "visible"),)))
            assert completion.text == ACTION and completion.native_assistant is None
        finally:
            await agent._backend.close()
            agent.close()
    asyncio.run(run())


def record_agent(directory, *, limits=None):
    host = ToolCall("read-own-records", "list_files", '{"path":"records","offset":0,"limit":10}')
    submit = ToolCall("choose-action", "submit_action", json.dumps({
        "format": "tool_call", "content": ACTION[11:-12]}, separators=(",", ":")))
    replies = [response(None, calls=(host,), reasoning_details=deepcopy(DETAILS)),
               response("", calls=(submit,), reasoning_content="native final selection")]
    return fake_agent(lambda _: replies.pop(0), simple_agent_config={"tools": {
        "enabled": True, "workspace_parent": str(directory), "limits": limits or {}}})


def test_final_submission_rationale_stays_in_assistant_content_records_and_next_step_history(tmp_path):
    rationale = "Tap the visible point at x=3 to select the highlighted option."
    legacy_unity_instructions = "Unity task: tap the highlighted option. Reply with <think>reasoning</think> and the action."
    action_arguments = {"format": "tool_call", "content": ACTION[11:-12]}
    submit = ToolCall("step-one-selection", "submit_action", json.dumps(action_arguments, separators=(",", ":")))
    next_submit = replace(submit, call_id="step-two-selection")
    replies = [response(rationale, calls=(submit,), reasoning_details=deepcopy(DETAILS)),
               response("Tap the visible point again to confirm the selection.", calls=(next_submit,), reasoning="private follow-up")]
    agent, client = fake_agent(lambda _: replies.pop(0), simple_agent_config={"tools": {
        "enabled": True, "workspace_parent": str(tmp_path / "workspace")}})

    def body(message):
        content = message["content"]
        return content if isinstance(content, str) else "\n".join(part["text"] for part in content or [] if part["type"] == "text")

    try:
        result, traces = agent.forward_with_trace({0: {"messages": [
            {"role": "system", "content": legacy_unity_instructions},
            {"role": "user", "content": "The option at x=3 is highlighted."}]}})
        first_messages = client.requests[0]["messages"]
        assert first_messages[0]["role"] == "system"
        protocol = body(first_messages[0])
        assert "For every environment step" in protocol
        assert "action rationale" in protocol and "one or two sentences" in protocol
        assert "without private reasoning or explanation tags" in protocol
        for tag in ("<think>", "</think>", "<rationale>", "</rationale>"):
            assert tag not in protocol
        assert "only the task action payload, never the rationale" in protocol
        assert legacy_unity_instructions not in protocol
        assert any(message["role"] == "developer" and legacy_unity_instructions in body(message)
                   for message in first_messages)
        assert result[0]["action"] == ACTION
        trace_call = traces[0]["model_requests"][0]
        assert trace_call["completion"] == rationale
        assert trace_call["native_assistant"]["message"]["content"] == rationale
        assert trace_call["native_assistant"]["message"]["reasoning_details"] == DETAILS
        assert PRIVATE not in trace_call["completion"] and "<think>" not in trace_call["completion"]
        assert json.loads(trace_call["tool_calls"][0]["arguments_json"]) == action_arguments
        assert rationale not in trace_call["tool_calls"][0]["arguments_json"]
        first_snapshot = agent._states[0].workspace._records["turn-000001"][0]
        files = dict(first_snapshot.files)
        assert files["assistant.txt"].decode("utf-8") == rationale
        assert files["action.txt"].decode("utf-8") == ACTION

        agent.forward({0: {"messages": [{"role": "user", "content": "Environment feedback: the option is selected."}]}})
        previous_assistant = next(message for message in client.requests[1]["messages"]
                                  if message.get("tool_calls", [{}])[0].get("id") == submit.call_id)
        assert previous_assistant == {"role": "assistant", "content": rationale,
                                      "tool_calls": [wire_call(submit)], "reasoning_details": DETAILS}
        assert json.loads(previous_assistant["tool_calls"][0]["function"]["arguments"]) == action_arguments
        assert any(message["role"] == "tool" and message["tool_call_id"] == submit.call_id
                   for message in client.requests[1]["messages"])
    finally:
        agent.close()


@pytest.fixture
def published_native_record(tmp_path):
    agent, client = record_agent(tmp_path / "workspace")
    try:
        result, trace = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "visible record input"}]}})
        runtime_state = agent._states[0]
        snapshot = runtime_state.workspace._records["turn-000001"][0]
        yield agent, client, result, trace[0], dict(snapshot.files)
    finally:
        agent.close()


def test_published_native_records_and_trace_preserve_both_host_and_submission(published_native_record):
    agent, client, result, trace, files = published_native_record
    assert result[0]["action"] == ACTION
    records = json.loads(files["native_assistants.json"])
    assert records == {"format": "native-assistants-v1", "calls": [
        {"call_index": index, "purpose": "action", **call["native_assistant"]}
        for index, call in enumerate(trace["model_requests"], start=1)]}
    assert records["calls"][0]["message"]["reasoning_details"] == DETAILS
    assert records["calls"][1]["message"]["reasoning"] == "native final selection"
    assert all(call["scope"] == agent._backend.native_scope for call in records["calls"])
    assert PRIVATE.encode("utf-8") not in files["assistant.txt"]
    assert files["action.txt"].decode("utf-8") == ACTION
    assert PRIVATE.encode("utf-8") not in files["messages.json"]
    assert [call["request_messages"] for call in trace["model_requests"]] == [
        call["messages"] for call in client.requests]
    assert trace["model_requests"][1]["request_messages"][-2] == records["calls"][0]["message"]
    snapshot_event = next(event for event in trace["agent_events"] if event["kind"] == "snapshot")
    assert snapshot_event["bytes"] == agent._states[0].published_bytes == sum(map(len, files.values()))
    assert snapshot_event["file_count"] == len(files) == 4


@pytest.mark.parametrize("bound", ["max_record_files", "max_record_file_bytes", "max_record_bytes", "max_published_bytes"])
def test_published_native_record_obeys_existing_workspace_quotas(bound, published_native_record, tmp_path):
    _, _, _, _, files = published_native_record
    native_bytes = len(files["native_assistants.json"])
    if bound == "max_record_file_bytes":
        # The native sidecar is the only file over this threshold; omitting it
        # from quota accounting would make this publication incorrectly pass.
        assert native_bytes > max(len(data) for name, data in files.items() if name != "native_assistants.json")
        value = native_bytes - 1
    elif bound == "max_record_files":
        value = len(files) - 1
    else:
        value = sum(map(len, files.values())) - 1
    agent, _ = record_agent(tmp_path / "limited-workspace", limits={bound: value})
    try:
        with pytest.raises(AgentLimitError, match="snapshot capacity"):
            agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "visible record input"}]}})
        assert agent._states[0].published_records == agent._states[0].published_bytes == 0
        assert not agent._states[0].workspace._records
        assert agent._states[0].runtime.history_snapshot[-1].kind == "completion"
    finally:
        agent.close()


def test_summary_and_codex_inputs_exclude_native_state_even_with_matching_api_scope():
    async def run():
        agent, client = fake_agent(lambda _: response("concise summary", reasoning_details=deepcopy(DETAILS)))
        try:
            native = state(ACTION, scope=agent._backend.native_scope)
            assistant = Message("assistant", (TextPart(ACTION),), native_assistant=native)
            req = ModelRequest((Message.text("user", "visible"), assistant), purpose="summary")
            completion = await agent._backend.complete(req)
            assert completion.text == "concise summary" and completion.native_assistant is None
            assert PRIVATE not in json.dumps(client.requests[0]["messages"])
            assert all("reasoning_details" not in m for m in client.requests[0]["messages"])
            sdk = SimpleNamespace(TextInput=lambda **kw: {"type": "text", **kw},
                                  ImageInput=lambda **kw: {"type": "image", **kw})
            inputs = CodexModel._inputs(replace(req, purpose="action"), sdk)
            assert PRIVATE not in json.dumps(inputs)
            assert any(item.get("text") == ACTION for item in inputs)
        finally:
            await agent._backend.close()
            agent.close()
    asyncio.run(run())


def test_native_state_counts_toward_context_tokens_bytes_characters_and_archive():
    native = state("answer", reasoning=PRIVATE + "密" * 500)
    plain = (Message.text("assistant", "answer"),)
    messages = (replace(plain[0], native_assistant=native),)
    runtime = SimpleRuntime(CallableModel(lambda _: asyncio.sleep(0)))
    native_metrics, plain_metrics = _metrics(messages), _metrics(plain)
    assert native_metrics[0] >= plain_metrics[0] + len(native.message_json)
    assert native_metrics[1] >= plain_metrics[1] + len(native.message_json.encode("utf-8"))
    assert runtime._tokens(messages, native_metrics[1]) > runtime._tokens(plain, plain_metrics[1])
    native_entry = ArchiveEntry(1, "completion", request().identity, messages, text="answer")
    plain_entry = replace(native_entry, messages=plain)
    assert runtime._archive_cost(native_entry)[0] > runtime._archive_cost(plain_entry)[0]
    assert runtime._archive_cost(native_entry)[1] > runtime._archive_cost(plain_entry)[1]
    assert PRIVATE not in encoded_messages(messages).decode("utf-8")
    counted = SimpleRuntime(CallableModel(lambda _: asyncio.sleep(0)), token_counter=lambda _: 7)
    assert counted._tokens(messages, native_metrics[1]) > counted._tokens(plain, plain_metrics[1]) == 7


@pytest.mark.parametrize("bound", ["max_context_characters", "max_context_bytes", "max_context_tokens"])
def test_native_state_changes_context_fit_for_every_budget(bound):
    messages = (Message("assistant", (TextPart("answer"),), native_assistant=state("answer", reasoning="密" * 500)),)
    plain = (Message.text("assistant", "answer"),)
    runtime = SimpleRuntime(CallableModel(lambda _: asyncio.sleep(0)))
    native_characters, native_bytes, _, _ = _metrics(messages)
    boundary = {"max_context_characters": native_characters,
                "max_context_bytes": native_bytes,
                "max_context_tokens": runtime._tokens(messages, native_bytes) + runtime.config.output_reserve_tokens}[bound]
    exact = SimpleRuntime(CallableModel(lambda _: asyncio.sleep(0)), replace(runtime.config, **{bound: boundary}))
    below = SimpleRuntime(CallableModel(lambda _: asyncio.sleep(0)), replace(runtime.config, **{bound: boundary - 1}))
    assert exact._fits(messages)
    assert below._fits(plain)
    assert not below._fits(messages)


@pytest.mark.parametrize("bound", ["max_archive_characters", "max_archive_bytes"])
def test_native_state_enforces_archive_hard_bound(bound):
    async def run():
        async def model(req):
            return ModelCompletion("answer", native_assistant=state("answer", reasoning=PRIVATE * 20))
        probe = SimpleRuntime(CallableModel(model))
        await probe.complete(request())
        offset = 0 if bound == "max_archive_characters" else 1
        total = sum(probe._archive_cost(entry)[offset] for entry in probe.history_snapshot)
        exact = SimpleRuntime(CallableModel(model), replace(probe.config, **{bound: total}))
        below = SimpleRuntime(CallableModel(model), replace(probe.config, **{bound: total - 1}))
        assert (await exact.complete(request())).text == "answer"
        with pytest.raises(AgentLimitError, match="archive"):
            await below.complete(request())
        assert [e.kind for e in below.history_snapshot] == ["input"]
    asyncio.run(run())


@pytest.mark.parametrize("bound", ["max_action_output_characters", "max_action_output_bytes"])
def test_action_output_limit_counts_native_state_and_preserves_original_archive(bound):
    async def run():
        native = state("answer", reasoning=PRIVATE * 20)
        async def model(req):
            return ModelCompletion("answer", native_assistant=native)
        amount = len(native.message_json) if bound.endswith("characters") else len(native.message_json.encode("utf-8"))
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(**{bound: len("answer") + amount - 1}))
        with pytest.raises(AgentLimitError, match="action output"):
            await runtime.complete(request())
        completion = runtime.history_snapshot[-1]
        assert completion.kind == "completion" and completion.text == "answer"
        assert completion.messages[0].native_assistant == native
    asyncio.run(run())


def test_compaction_strips_retained_sidecars_but_preserves_immutable_archive_and_summary_boundary():
    async def run():
        seen = []
        async def model(req):
            seen.append(req)
            if req.purpose == "summary":
                assert all(message.native_assistant is None for message in req.messages)
                assert PRIVATE not in texts(req.messages)
                return ModelCompletion("Earlier complete turns retained.")
            return ModelCompletion("answer", native_assistant=state("answer"))

        cfg = SimpleRuntimeConfig(max_context_characters=1800, max_context_tokens=20000,
            output_reserve_tokens=64, recent_turns=1, summary_output_tokens=128)
        runtime = SimpleRuntime(CallableModel(model), cfg)
        await runtime.complete(request(1, "old " + "x" * 500))
        await runtime.complete(request(2, "recent " + "x" * 500))
        saved = next(e for e in runtime.history_snapshot if e.kind == "completion")
        await runtime.complete(request(3, "current " + "x" * 500))
        assert any(req.purpose == "summary" for req in seen)
        after_compaction = seen[-1]
        assert after_compaction.purpose == "action"
        assert all(message.native_assistant is None for message in after_compaction.messages)
        assert "Model notes; not environment facts" in texts(after_compaction.messages)
        assert saved.messages[0].native_assistant is not None
        assert saved in runtime.history_snapshot
        assert all(not e.messages for e in runtime.history_snapshot if e.kind == "summary_completion")
    asyncio.run(run())


def test_runtime_rejects_summary_completion_with_native_state_and_preserves_working_transaction():
    async def run():
        async def model(req):
            if req.purpose == "summary":
                return ModelCompletion("Earlier complete turns retained.", native_assistant=state("Earlier complete turns retained."))
            return ModelCompletion("answer", native_assistant=state("answer"))
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            max_context_characters=1800, max_context_tokens=20000,
            output_reserve_tokens=64, recent_turns=1, summary_output_tokens=128))
        await runtime.complete(request(1, "old " + "x" * 500))
        await runtime.complete(request(2, "recent " + "x" * 500))
        third = request(3, "current " + "x" * 500)
        runtime.accept_input(third)
        before, archive = runtime.working_snapshot, runtime.history_snapshot
        with pytest.raises(AgentError, match="summary model returned an invalid completion"):
            await runtime.complete(third)
        assert runtime.working_snapshot == before
        assert runtime.history_snapshot == archive
        assert runtime.usage.summary_calls == 1
    asyncio.run(run())


def test_current_tool_continuation_cannot_compact_away_native_prefix_to_fit():
    async def run():
        from agent_ark.agent.simple_agent import ToolRegistry
        tool = ToolCall("host-current", "echo", '{"text":"x"}')
        registry = ToolRegistry()
        registry.register(ToolSpec("echo", "Return visible own text.", '{"type":"object"}'),
                          lambda args, session: (TextPart("large result " + "x" * 1500),))
        calls = []
        async def model(req):
            calls.append(req)
            assert req.purpose == "action"
            return ModelCompletion("", tool_calls=(tool,), native_assistant=state(None, calls=(tool,)))
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            max_context_characters=1800, max_context_tokens=20000,
            output_reserve_tokens=64, recent_turns=1), tools=registry)
        with pytest.raises(AgentLimitError, match="(?i)native.*(continuation|prefix)|(?i:continuation).*native"):
            await runtime.complete(request())
        assert len(calls) == 1 and runtime.usage.summary_calls == 0
        exchange = next(e for e in runtime.history_snapshot if e.kind == "tool_exchange")
        assert exchange.messages[0].native_assistant is not None
        assert exchange.messages[0] in runtime.working_snapshot
    asyncio.run(run())
