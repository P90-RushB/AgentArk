"""Offline contracts for Volcengine's opaque assistant continuation fields."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent import CallableModel, SimpleRuntime, SimpleRuntimeConfig, ToolRegistry
from agent_ark.agent.simple_agent.agent import SimpleAgent
from agent_ark.agent.simple_agent.codex_model import CodexModel
from agent_ark.agent.simple_agent.encoding import api_chat_messages, chat_messages
from agent_ark.agent.simple_agent.models import ModelResponseError
from agent_ark.agent.simple_agent.types import (
    AgentInputIdentity, AgentLimitError, AgentRequest, Message, ModelCompletion,
    ModelRequest, NativeAssistantState, TextPart, ToolCall, ToolSpec,
)


ACTION = '<tool_call>{"name":"tap","arguments":{"x":3}}</tool_call>'
SCOPE = "volcengine:offline-model"
SUMMARY = "  PRIVATE_NATIVE_SUMMARY\n原文 must never become ordinary content\t"
ENCRYPTED = " opaque+/=\\n密文\n\t{not an action} "
DETAILS = [{"type": "reasoning.encrypted", "data": "openrouter opaque", "index": 4}]


def wire_call(call):
    return {"id": call.call_id, "type": "function", "function": {
        "name": call.name, "arguments": call.arguments_json}}


def response(content, *, calls=(), **fields):
    return {"choices": [{"finish_reason": "tool_calls" if calls else "stop", "message": {
        "content": content, **({"tool_calls": [wire_call(call) for call in calls]} if calls else {}),
        **fields}}]}


class FakeChatClient:
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
    agent = SimpleAgent(api_key="unused-fake-key", provider="auto",
                        base_url="https://ark.cn-beijing.volces.com/api/v3", model="offline-model", **kwargs)
    client = FakeChatClient(respond)
    agent._backend.client = client
    return agent, client


def native_state(content="answer", *, calls=(), scope=SCOPE, **fields):
    payload = {"role": "assistant", "content": content,
               **({"tool_calls": [wire_call(call) for call in calls]} if calls else {}), **fields}
    return NativeAssistantState(scope, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def request(number=1, text="visible observation"):
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "agent", 1, str(number)), (
        Message.text("system", "Use the visible observation."), Message.text("user", text)))


def body_text(messages):
    return "\n".join(part.text for message in messages for part in message.parts if isinstance(part, TextPart))


@pytest.mark.parametrize("fields", [
    {"reasoning_content": SUMMARY, "encrypted_content": ENCRYPTED},
    {"encrypted_content": ENCRYPTED},
    {"reasoning_content": None, "encrypted_content": ENCRYPTED},
    {"reasoning_content": "", "encrypted_content": ENCRYPTED},
    {"reasoning_content": SUMMARY, "encrypted_content": None},
    {"reasoning_content": None, "encrypted_content": None},
    {"encrypted_content": None},
    {"reasoning_content": "", "encrypted_content": ""},
    {"encrypted_content": ""},
])
def test_native_pair_preserves_field_presence_and_values_in_api_replay_only(fields):
    async def run():
        agent, client = fake_agent(lambda _: response(" \n" + ACTION + " \t", **fields))
        try:
            completion = await agent._backend.complete(ModelRequest((Message.text("user", "visible"),)))
            native = completion.native_assistant
            expected = {"role": "assistant", "content": " \n" + ACTION + " \t", **fields}
            assert completion.text == ACTION
            assert native.message == expected
            assert agent._backend.last_response_metadata["native_assistant"]["message"] == expected
            assert agent._backend.last_response_metadata["assistant_raw"] == ACTION
            assistant = Message("assistant", (TextPart(ACTION),), native_assistant=native)
            replay = api_chat_messages((assistant,), native_scope=agent._backend.native_scope)
            assert replay == [expected]
            replay[0]["encrypted_content"] = "mutated detached copy"
            assert native.message == expected
            portable = chat_messages((assistant,))
            assert portable == [{"role": "assistant", "content": [{"type": "text", "text": ACTION}]}]
            sdk = SimpleNamespace(TextInput=lambda **kw: {"type": "text", **kw},
                                  ImageInput=lambda **kw: {"type": "image", **kw})
            inputs = CodexModel._inputs(ModelRequest((assistant,)), sdk)
            assert "PRIVATE_NATIVE_SUMMARY" not in json.dumps(inputs) and "opaque+/=" not in json.dumps(inputs)
            with pytest.raises(ModelResponseError, match="(?i)scope|endpoint|model"):
                api_chat_messages((assistant,), native_scope="different-endpoint-or-model")
            assert len(client.requests) == 1
        finally:
            await agent._backend.close()
            agent.close()
    asyncio.run(run())


@pytest.mark.parametrize("explicit,extra,expected_fields", [
    ({"content", "reasoning_content"}, {}, {"reasoning": SUMMARY}),
    ({"content", "reasoning_content", "encrypted_content"}, {},
     {"reasoning_content": SUMMARY, "encrypted_content": None}),
    ({"content", "reasoning_content"}, {"encrypted_content": ""},
     {"reasoning_content": SUMMARY, "encrypted_content": ""}),
    ({"content"}, {"reasoning_content": None, "encrypted_content": ENCRYPTED},
     {"reasoning_content": None, "encrypted_content": ENCRYPTED}),
])
def test_sdk_declared_defaults_do_not_turn_missing_fields_into_explicit_nulls(explicit, extra, expected_fields):
    async def run():
        # SDK objects expose default attributes even when their JSON key was absent.
        message = SimpleNamespace(content=ACTION, tool_calls=None, refusal=None,
            reasoning_content=SUMMARY if "reasoning_content" in explicit else None,
            encrypted_content=None, model_fields_set=explicit, model_extra=deepcopy(extra))
        agent, _ = fake_agent(lambda _: SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=message)], usage=None))
        try:
            completion = await agent._backend.complete(ModelRequest((Message.text("user", "visible"),)))
            assert completion.native_assistant.message == {
                "role": "assistant", "content": ACTION, **expected_fields}
        finally:
            await agent._backend.close()
            agent.close()
    asyncio.run(run())


@pytest.mark.parametrize("fields,expected_fields", [
    ({"reasoning_content": SUMMARY}, {"reasoning": SUMMARY}),
    ({"reasoning_content": SUMMARY, "encrypted_content": None},
     {"reasoning_content": SUMMARY, "encrypted_content": None}),
    ({"encrypted_content": ENCRYPTED}, {"encrypted_content": ENCRYPTED}),
])
def test_real_openai_sdk_message_extra_fields_preserve_original_dialect(fields, expected_fields):
    from openai.types.chat import ChatCompletionMessage

    async def run():
        message = ChatCompletionMessage(role="assistant", content=ACTION, **fields)
        agent, _ = fake_agent(lambda _: SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=message)], usage=None))
        try:
            completion = await agent._backend.complete(ModelRequest((Message.text("user", "visible"),)))
            assert completion.native_assistant.message == {
                "role": "assistant", "content": ACTION, **expected_fields}
        finally:
            await agent._backend.close()
            agent.close()
    asyncio.run(run())


@pytest.mark.parametrize("fields,expected", [
    ({"reasoning_content": SUMMARY}, {"reasoning": SUMMARY}),
    ({"model_extra": {"reasoning_content": SUMMARY}}, {"reasoning": SUMMARY}),
    ({"reasoning": "existing reasoning", "reasoning_content": SUMMARY}, {"reasoning": "existing reasoning"}),
    ({"reasoning_details": deepcopy(DETAILS), "reasoning_content": SUMMARY, "encrypted_content": ENCRYPTED},
     {"reasoning_details": DETAILS}),
    ({"reasoning_details": deepcopy(DETAILS), "reasoning_content": SUMMARY, "encrypted_content": []},
     {"reasoning_details": DETAILS}),
    ({"reasoning_details": [], "reasoning_content": SUMMARY, "encrypted_content": ENCRYPTED},
     {"reasoning_content": SUMMARY, "encrypted_content": ENCRYPTED}),
    ({"reasoning": "existing reasoning", "encrypted_content": None}, {"reasoning": "existing reasoning"}),
    ({"reasoning": "existing reasoning", "encrypted_content": ""}, {"reasoning": "existing reasoning"}),
    ({"model_extra": {"reasoning": "existing reasoning", "encrypted_content": ""}},
     {"reasoning": "existing reasoning"}),
    ({"reasoning": "existing reasoning", "reasoning_content": None, "encrypted_content": ""},
     {"reasoning_content": None, "encrypted_content": ""}),
])
def test_legacy_alias_and_details_precedence_remain_compatible(fields, expected):
    async def run():
        agent, _ = fake_agent(lambda _: response(ACTION, **fields))
        try:
            completion = await agent._backend.complete(ModelRequest((Message.text("user", "visible"),)))
            assert completion.text == ACTION
            assert completion.native_assistant.message == {"role": "assistant", "content": ACTION, **expected}
        finally:
            await agent._backend.close()
            agent.close()
    asyncio.run(run())


@pytest.mark.parametrize("raw_content", [None, "", " \n\t "])
def test_host_continuations_submission_step_history_and_records_replay_native_fields(tmp_path, raw_content):
    host = ToolCall("host-one", "echo", ' { "text" : "first visible result" } ')
    again = ToolCall("host-two", "echo", '{"text":"second visible result"}')
    submit = ToolCall("submit-one", "submit_action", json.dumps({
        "format": "tool_call", "content": ACTION[11:-12]}, separators=(",", ":")))
    next_submit = replace(submit, call_id="submit-two")
    originals = [response(raw_content, calls=(host,), encrypted_content=ENCRYPTED),
                 response(raw_content, calls=(again,), reasoning_content=None, encrypted_content=""),
                 response("Visible action rationale.", calls=(submit,), reasoning_content=SUMMARY,
                          encrypted_content=ENCRYPTED),
                 response(None, calls=(next_submit,), encrypted_content=None)]
    pristine = deepcopy(originals)
    executions = []

    def setup(registry):
        registry.register(ToolSpec("echo", "Return own visible text.",
            '{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}'),
            lambda args, session: (executions.append(args["text"]) or TextPart(args["text"]),))

    agent, client = fake_agent(lambda _: originals.pop(0), tool_setup=setup,
        simple_agent_config={"tools": {"enabled": True, "workspace_parent": str(tmp_path / "workspace")}})
    try:
        result, traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "visible step one"}]}})
        assert result[0] == {"action": ACTION, "assistant": ACTION}
        assert executions == ["first visible result", "second visible result"]
        expected = [{"role": "assistant", **item["choices"][0]["message"]} for item in pristine]
        assert [m for m in client.requests[1]["messages"] if m.get("tool_calls")] == expected[:1]
        assert [m for m in client.requests[2]["messages"] if m.get("tool_calls")] == expected[:2]
        assert all("reasoning" not in m and "reasoning_details" not in m for m in expected)
        assert [call["request_messages"] for call in traces[0]["model_requests"]] == [
            call["messages"] for call in client.requests]
        current = agent._states[0]
        files = dict(current.workspace._records["turn-000001"][0].files)
        records = json.loads(files["native_assistants.json"])
        assert records == {"format": "native-assistants-v1", "calls": [
            {"call_index": index, "purpose": "action", "scope": agent._backend.native_scope, "message": message}
            for index, message in enumerate(expected[:3], start=1)]}
        assert files["assistant.txt"].decode("utf-8") == "Visible action rationale."
        assert files["action.txt"].decode("utf-8") == ACTION
        assert b"PRIVATE_NATIVE_SUMMARY" not in files["messages.json"]
        assert b"opaque+/=" not in files["messages.json"]
        archived = [entry.messages[0].native_assistant.message for entry in current.runtime.history_snapshot
                    if entry.kind in ("tool_exchange", "completion") and entry.messages]
        assert archived == expected[:3]
        agent.forward({0: {"messages": [{"role": "user", "content": "actual environment feedback: tapped"}]}})
        following = client.requests[3]["messages"]
        assert [m for m in following if m.get("tool_calls")] == expected[:3]
        assert any(m["role"] == "tool" and m["tool_call_id"] == submit.call_id
                   and "not executed yet" in m["content"] for m in following)
        assert following[-1]["role"] == "user" and "actual environment feedback" in str(following[-1]["content"])
        assert records["calls"][0]["message"] == expected[0]
    finally:
        agent.close()


@pytest.mark.parametrize("fields", [
    {"encrypted_content": []}, {"encrypted_content": {}}, {"encrypted_content": True},
    {"encrypted_content": 7}, {"encrypted_content": ENCRYPTED, "reasoning_content": []},
    {"encrypted_content": ENCRYPTED, "reasoning_content": {}},
    {"encrypted_content": ENCRYPTED, "reasoning_content": False},
    {"model_extra": {"encrypted_content": ENCRYPTED, "reasoning_content": {}}},
])
def test_malformed_native_fields_are_rejected_before_host_tool_side_effects(fields):
    executions = []
    host = ToolCall("must-not-execute", "echo", '{}')

    def setup(registry):
        registry.register(ToolSpec("echo", "Return visible text.", '{"type":"object"}'),
                          lambda args, session: (executions.append("executed") or TextPart("result"),))

    agent, client = fake_agent(lambda _: response(None, calls=(host,), **fields), tool_setup=setup)
    try:
        with pytest.raises(ModelResponseError, match="(?i)native|reasoning"):
            agent.forward({0: {"messages": [{"role": "user", "content": "visible"}]}})
        assert not executions and len(client.requests) == 1
        assert all(entry.kind not in ("tool_exchange", "completion")
                   for entry in agent._states[0].runtime.history_snapshot)
    finally:
        agent.close()


@pytest.mark.parametrize("fields", [
    {"encrypted_content": [], "reasoning_content": SUMMARY},
    {"encrypted_content": ENCRYPTED, "reasoning_content": 1},
    {"encrypted_content": ENCRYPTED, "reasoning": "mixed family"},
    {"encrypted_content": ENCRYPTED, "reasoning_details": DETAILS},
    {"reasoning_content": SUMMARY},
    {"encrypted_content": ENCRYPTED, "unexpected": "arbitrary provider kwargs"},
])
def test_state_contract_rejects_invalid_shapes_and_mixed_reasoning_families(fields):
    with pytest.raises((ValueError, TypeError)):
        native_state(**fields)


def test_native_pair_obeys_inclusive_utf8_json_byte_limit_and_duplicate_key_rejection():
    limit = 1024 * 1024
    payload = {"role": "assistant", "content": None, "reasoning_content": None, "encrypted_content": ""}
    overhead = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    payload["encrypted_content"] = "密" + "x" * (limit - overhead - len("密".encode("utf-8")))
    exact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert len(exact.encode("utf-8")) == limit
    assert NativeAssistantState("s" * 256, exact).message == payload
    with pytest.raises((ValueError, TypeError)):
        NativeAssistantState(SCOPE, exact[:-2] + "x\"}")
    duplicate = '{"role":"assistant","content":null,"encrypted_content":"one","encrypted_content":"two"}'
    with pytest.raises((ValueError, TypeError)):
        NativeAssistantState(SCOPE, duplicate)


@pytest.mark.parametrize("content", [None, "", " \n\t "])
def test_native_summary_and_encrypted_text_never_supply_the_environment_action(content):
    agent, client = fake_agent(lambda _: response(content, reasoning_content=ACTION, encrypted_content=ACTION))
    try:
        with pytest.raises(ModelResponseError):
            agent.forward({0: {"messages": [{"role": "user", "content": "visible"}]}})
        assert len(client.requests) == 1
        assert all(entry.kind != "completion" for entry in agent._states[0].runtime.history_snapshot)
    finally:
        agent.close()


def test_summary_model_inputs_and_output_exclude_native_pair_even_for_matching_scope():
    async def run():
        agent, client = fake_agent(lambda _: response("concise visible summary", reasoning_content=SUMMARY,
                                                     encrypted_content=ENCRYPTED))
        try:
            native = native_state(ACTION, scope=agent._backend.native_scope,
                                  reasoning_content=SUMMARY, encrypted_content=ENCRYPTED)
            assistant = Message("assistant", (TextPart(ACTION),), native_assistant=native)
            completion = await agent._backend.complete(ModelRequest((Message.text("user", "visible"), assistant),
                                                                   purpose="summary"))
            assert completion.text == "concise visible summary" and completion.native_assistant is None
            assert all("reasoning_content" not in message and "encrypted_content" not in message
                       for message in client.requests[0]["messages"])
            assert "PRIVATE_NATIVE_SUMMARY" not in json.dumps(client.requests[0]["messages"])
            assert "opaque+/=" not in json.dumps(client.requests[0]["messages"])
        finally:
            await agent._backend.close()
            agent.close()
    asyncio.run(run())


def test_successful_compaction_clears_working_native_pair_and_keeps_original_archive():
    async def run():
        seen = []

        async def model(req):
            seen.append(req)
            if req.purpose == "summary":
                assert all(message.native_assistant is None for message in req.messages)
                assert SUMMARY not in body_text(req.messages) and ENCRYPTED not in body_text(req.messages)
                return ModelCompletion("Earlier complete visible turns retained.")
            return ModelCompletion("answer", native_assistant=native_state(
                reasoning_content=SUMMARY, encrypted_content=ENCRYPTED))

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            max_context_characters=1800, max_context_tokens=20000, output_reserve_tokens=64,
            recent_turns=1, summary_output_tokens=128))
        await runtime.complete(request(1, "old " + "x" * 500))
        await runtime.complete(request(2, "recent " + "x" * 500))
        originals = [entry for entry in runtime.history_snapshot if entry.kind == "completion"]
        await runtime.complete(request(3, "current " + "x" * 500))
        assert any(req.purpose == "summary" for req in seen)
        assert seen[-1].purpose == "action"
        assert all(message.native_assistant is None for message in seen[-1].messages)
        assert "Model notes; not environment facts" in body_text(seen[-1].messages)
        assert all(entry in runtime.history_snapshot for entry in originals)
        assert all(entry.messages[0].native_assistant.message["encrypted_content"] == ENCRYPTED
                   for entry in originals)
    asyncio.run(run())


@pytest.mark.parametrize("encrypted", [ENCRYPTED, "", None])
def test_only_encrypted_native_tool_turn_cannot_compact_its_prefix_mid_continuation(encrypted):
    async def run():
        host = ToolCall("host-current", "echo", '{}')
        registry = ToolRegistry()
        registry.register(ToolSpec("echo", "Return own visible text.", '{"type":"object"}'),
                          lambda args, session: (TextPart("large result " + "x" * 1500),))
        calls = []

        async def model(req):
            calls.append(req)
            assert req.purpose == "action"
            return ModelCompletion("", tool_calls=(host,), native_assistant=native_state(
                None, calls=(host,), encrypted_content=encrypted))

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            max_context_characters=1800, max_context_tokens=20000, output_reserve_tokens=64,
            recent_turns=1), tools=registry)
        with pytest.raises(AgentLimitError, match="(?i)native.*(continuation|prefix)"):
            await runtime.complete(request())
        assert len(calls) == 1 and runtime.usage.summary_calls == 0
        exchange = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert exchange.messages[0].native_assistant.message["encrypted_content"] == encrypted
        assert exchange.messages[0] in runtime.working_snapshot
    asyncio.run(run())
