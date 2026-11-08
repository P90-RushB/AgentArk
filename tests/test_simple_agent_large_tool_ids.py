"""Opaque IDs above 64 KiB retain protocol, context and archive accounting."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent import CallableModel, SimpleAgent, SimpleRuntime, SimpleRuntimeConfig, ToolRegistry
from agent_ark.agent.simple_agent.action_submission import SUBMISSION_SPEC, translate_submission
from agent_ark.agent.simple_agent.encoding import encode_png
from agent_ark.agent.simple_agent.litellm_model import LiteLLMModel
from agent_ark.agent.simple_agent.runtime import _metrics
from agent_ark.agent.simple_agent.types import (
    AgentInputIdentity, AgentLimitError, AgentRequest, ImagePart, LiteLLMAssistantState, Message, ModelCompletion,
    ModelRequest, NativeAssistantState, TextPart, ToolCall, ToolSpec,
)


HOST = ToolSpec("host_operation", "Return visible evidence.", '{"type":"object"}')
ARGUMENTS = '{"text":"visible argument"}'


def opaque_id(size):
    prefix = "call_synthetic__mpthought__"
    return (prefix + "Synthetic+/=._-" * size)[:size]


def adapter():
    return LiteLLMModel(SimpleNamespace(litellm_config={"model": "openai/synthetic-model", "api": "responses"},
                                      base_url="https://proxy.example/openapi"))


def response(call, text="Brief reason."):
    return {"status": "completed", "output": [
        {"type": "message", "id": f"msg_synthetic_{len(call.call_id)}", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": text}]},
        {"type": "function_call", "id": f"fc_synthetic_{len(call.call_id)}", "status": "completed", "call_id": call.call_id,
         "name": call.name, "arguments": call.arguments_json}],
        "usage": {"input_tokens": 7, "output_tokens": 11, "total_tokens": 18}}


def parsed(call):
    model = adapter()
    return model._responses_completion(response(call), ModelRequest((Message.text("user", "Observe."),),
        tools=(HOST, SUBMISSION_SPEC)))


def request():
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "agent", 0, "1"),
                        (Message.text("user", "Preserve visible evidence and finish."),))


def config(**updates):
    return SimpleRuntimeConfig(**dict(max_context_tokens=1_000_000, max_context_characters=2_000_000,
                                     output_reserve_tokens=128, **updates))


def registry(executed, result=None):
    tools = ToolRegistry()
    tools.register(HOST, lambda arguments, session: executed.append(arguments) or
                   (result if result is not None else (TextPart("visible result"),)))
    return tools


@pytest.mark.parametrize("size", [71838, 95221])
def test_production_responses_parser_accepts_observed_lengths_without_rewriting(size):
    call = ToolCall(opaque_id(size), HOST.name, ARGUMENTS)
    model = adapter()
    source = response(call)
    completion = model._responses_completion(source, ModelRequest((Message.text("user", "Observe."),), tools=(HOST,)))
    assert completion.tool_calls == (call,)
    assert completion.native_assistant.message["responses_output"] == source["output"]
    assistant = Message("assistant", (TextPart(completion.text),), (call,), native_assistant=completion.native_assistant)
    result = Message("tool", (TextPart("visible result"),), tool_call_id=call.call_id)
    wire = model.request_messages(ModelRequest((assistant, result), tools=(HOST,)))
    assert wire[:-1] == source["output"]
    assert wire[-1] == {"type": "function_call_output", "call_id": call.call_id, "output": "visible result"}


def test_production_agent_host_and_terminal_ack_keep_large_ids_in_following_wire(monkeypatch, tmp_path):
    ids = (opaque_id(95221), opaque_id(71838))
    action = {"name": "ApplyImpulse", "arguments": {"forceScale": 1.5}}
    calls = [ToolCall(ids[0], "list_files", '{"path":"records","offset":0,"limit":1}'),
             ToolCall(ids[1], "submit_action", json.dumps({"format": "tool_call", "content": json.dumps(action)})),
             ToolCall("terminal-short", "submit_action", '{"format":"raw","content":"next action"}')]
    replies = iter(response(call) for call in calls)
    seen = []
    async def aresponses(**kwargs):
        seen.append(deepcopy(kwargs))
        return next(replies)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(aresponses=aresponses))
    agent = SimpleAgent(api_key="synthetic-key", temperature=None, max_retries=0,
        simple_agent_config={"backend": "litellm", "litellm": {"model": "openai/synthetic-model", "api": "responses"},
            "context": {"max_context_characters": 2_000_000, "max_context_tokens": 1_000_000},
            "tools": {"enabled": True, "workspace_parent": str(tmp_path)}})
    try:
        result, _ = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "Visible state."}]}})
        assert json.loads(result[0]["action"].removeprefix("<tool_call>").removesuffix("</tool_call>")) == action
        second, _ = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "Next observation."}]}})
        assert second[0]["action"] == "next action"
        assert len(seen) == 3
        host = next(item for item in seen[1]["input"] if item.get("type") == "function_call")
        host_result = next(item for item in seen[1]["input"] if item.get("type") == "function_call_output")
        assert host["call_id"].encode() == host_result["call_id"].encode() == ids[0].encode()
        terminal = next(item for item in seen[2]["input"] if item.get("type") == "function_call" and item["name"] == "submit_action")
        ack = next(item for item in seen[2]["input"] if item.get("type") == "function_call_output" and item["call_id"] == ids[1])
        assert terminal["call_id"].encode() == ack["call_id"].encode() == ids[1].encode()
        assert seen[2]["input"][:len(seen[1]["input"])] == seen[1]["input"]
        archive = agent._states[0].runtime.history_snapshot
        assert any(message.native_assistant and message.native_assistant.tool_calls == (calls[0],)
                   for entry in archive for message in entry.messages)
        assert any(message.tool_call_id == ids[1] for entry in archive for message in entry.messages)
    finally:
        agent.close()


@pytest.mark.parametrize("terminal", [False, True])
def test_large_ids_share_the_existing_exact_aggregate_protocol_limit(terminal):
    async def run():
        call = ToolCall(opaque_id(95221), "submit_action" if terminal else HOST.name,
                        '{"format":"raw","content":"selected action"}' if terminal else ARGUMENTS)
        source = parsed(call)
        completion, _ = translate_submission(source, max_bytes=1000, max_characters=1000) if terminal else (source, None)
        amount = len(call.call_id.encode()) * (2 if terminal else 1)
        amount += len(source.native_assistant.message_json.encode()) + len(source.native_assistant.scope.encode())
        for limit, accepted in ((amount, True), (amount - 1, False)):
            executed, seen = [], []
            async def model(req):
                seen.append(req)
                return completion if len(seen) == 1 else ModelCompletion("done")
            runtime = SimpleRuntime(CallableModel(model), config(max_protocol_output_bytes=limit), tools=registry(executed))
            if accepted:
                assert (await runtime.complete(request())).text == ("selected action" if terminal else "done")
                assert len(executed) == (0 if terminal else 1)
            else:
                with pytest.raises(AgentLimitError, match="protocol output"):
                    await runtime.complete(request())
                assert len(seen) == 1 and executed == []
    asyncio.run(run())


def test_multiple_large_portable_ids_share_one_default_protocol_allowance():
    async def run():
        calls = tuple(ToolCall(str(index) + opaque_id(95221), HOST.name, ARGUMENTS) for index in range(12))
        assert sum(len(call.call_id.encode()) for call in calls) > 1024 * 1024
        executed = []
        async def model(req):
            return ModelCompletion("", tool_calls=calls)
        runtime = SimpleRuntime(CallableModel(model), config(), tools=registry(executed))
        with pytest.raises(AgentLimitError, match="protocol output"):
            await runtime.complete(request())
        assert executed == []
    asyncio.run(run())


@pytest.mark.parametrize("state_type,reasoning", [
    (NativeAssistantState, {"reasoning": "synthetic"}),
    (LiteLLMAssistantState, {"reasoning_content": "synthetic"}),
])
def test_maximum_structural_id_does_not_enlarge_native_json_envelopes(state_type, reasoning):
    call = ToolCall("x" * (1024 * 1024), HOST.name, ARGUMENTS)
    packet = {"role": "assistant", "content": None, "tool_calls": [{"id": call.call_id,
        "type": "function", "function": {"name": call.name, "arguments": call.arguments_json}}], **reasoning}
    with pytest.raises(ValueError, match="state exceeds"):
        state_type("synthetic:scope", json.dumps(packet))


@pytest.mark.parametrize("dimension,limit", [("characters", 8000), ("bytes", 8000), ("tokens", 2000)])
def test_large_id_counts_in_context_and_restarts_without_losing_latest_result(dimension, limit):
    async def run():
        image = ImagePart(encode_png(bytes((0, 0, 255)) * 4, 2, 2), width=2, height=2)
        result = (TextPart("Exact latest result."), image)
        call = ToolCall(opaque_id(95221), HOST.name, ARGUMENTS)
        completion = parsed(call)
        seen, executed = [], []
        async def model(req):
            assert req.purpose == "action", "Dropping only opaque replay should suffice without a summary"
            seen.append(req)
            return completion if len(seen) == 1 else ModelCompletion("done")
        cfg = replace(config(), **{"max_context_" + dimension: limit})
        runtime = SimpleRuntime(CallableModel(model), cfg, tools=registry(executed, result))
        assert (await runtime.complete(request())).text == "done"
        assert len(executed) == 1 and len(seen) == 2
        following = seen[1].messages
        assert all(not message.tool_calls and message.native_assistant is None and message.role != "tool" for message in following)
        assert any(isinstance(part, TextPart) and part.text == ARGUMENTS for message in following for part in message.parts)
        assert any(isinstance(part, TextPart) and part.text == result[0].text for message in following for part in message.parts)
        assert any(part is image for message in following for part in message.parts)
        assert any(entry.compaction_json and json.loads(entry.compaction_json).get("stage") == "restart"
                   for entry in runtime.history_snapshot)
        original = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert original.messages[0].native_assistant == completion.native_assistant
        assert original.messages[0].tool_calls == (call,) and original.messages[1].tool_call_id == call.call_id
        assert _metrics(original.messages)[0] > len(call.call_id)
    asyncio.run(run())


@pytest.mark.parametrize("dimension", ["characters", "bytes"])
def test_large_id_remains_bounded_by_immutable_archive_limits_before_host_effects(dimension):
    async def run():
        completion = parsed(ToolCall(opaque_id(95221), HOST.name, ARGUMENTS))
        executed, seen = [], []
        async def model(req):
            seen.append(req)
            return completion
        cfg = replace(config(), **{"max_archive_" + dimension: 8000})
        runtime = SimpleRuntime(CallableModel(model), cfg, tools=registry(executed))
        with pytest.raises(AgentLimitError, match="archive"):
            await runtime.complete(request())
        assert len(seen) == 1 and executed == []
    asyncio.run(run())


def test_large_id_native_packet_still_obeys_legacy_combined_action_budget():
    async def run():
        completion = parsed(ToolCall(opaque_id(95221), HOST.name, ARGUMENTS))
        assert len(completion.native_assistant.message_json) > 120_000
        executed = []
        async def model(req):
            return completion
        runtime = SimpleRuntime(CallableModel(model), config(output_budget_policy="legacy_v1"), tools=registry(executed))
        with pytest.raises(AgentLimitError, match="action output"):
            await runtime.complete(request())
        assert executed == []
    asyncio.run(run())
