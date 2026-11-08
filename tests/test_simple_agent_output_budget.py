"""Separate semantic output admission from opaque replay; no HTTP or Unity."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json

import pytest

from agent_ark.agent.simple_agent import CallableModel, SimpleRuntime, SimpleRuntimeConfig, ToolRegistry
from agent_ark.agent.simple_agent.action_submission import translate_submission
from agent_ark.agent.simple_agent.models import SubmissionResponseError
from agent_ark.agent.simple_agent.runtime import _metrics
from agent_ark.agent.simple_agent.types import (
    AgentInputIdentity, AgentLimitError, AgentRequest, LiteLLMAssistantState,
    Message, ModelCompletion, NativeAssistantState, TextPart, ToolCall, ToolSpec,
)


SCOPE = "offline:output-budget"
ARGUMENTS = '{"text":"visible argument"}'


def request(number=1):
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "agent", 0, str(number)), (
        Message.text("user", "Complete the visible operation."),))


def native_completion(call, text=""):
    wire = {"id": call.call_id, "type": "function", "function": {
        "name": call.name, "arguments": call.arguments_json}}
    payload = {"role": "assistant", "content": text or None,
               "tool_calls": [wire], "responses_output": [{"type": "function_call",
                   "id": "fc_" + call.call_id, "call_id": call.call_id,
                   "name": call.name, "arguments": call.arguments_json}]}
    if text:
        payload["responses_output"].insert(0, {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": text}]})
    state = LiteLLMAssistantState(SCOPE, json.dumps(payload, separators=(",", ":")))
    return ModelCompletion(text, tool_calls=(call,), native_assistant=state)


def registry(executed):
    tools = ToolRegistry()
    tools.register(ToolSpec("host_operation", "Perform one visible operation.",
        '{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}'),
        lambda arguments, session: executed.append(arguments["text"]) or (TextPart("visible result"),))
    return tools


def test_new_default_and_explicit_legacy_output_budget_policy():
    assert SimpleRuntimeConfig().output_budget_policy == "separate_native_v1"
    assert SimpleRuntimeConfig(output_budget_policy="legacy_v1").output_budget_policy == "legacy_v1"
    with pytest.raises(ValueError, match="output budget policy"):
        SimpleRuntimeConfig(output_budget_policy="unknown")
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_protocol_output_bytes"):
            SimpleRuntimeConfig(max_protocol_output_bytes=value)


def test_long_opaque_call_is_admitted_and_original_packet_reaches_next_request():
    async def run():
        call = ToolCall("call_" + "opaque" * 6972, "host_operation", ARGUMENTS)
        first = native_completion(call, "Brief reason.")
        assert len(first.native_assistant.message_json) > 120_000
        seen, executed = [], []
        async def model(req):
            seen.append(req)
            return first if len(seen) == 1 else ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), tools=registry(executed))
        assert (await runtime.complete(request())).text == "done"
        assert executed == ["visible argument"]
        assistant = next(message for message in seen[1].messages if message.tool_calls)
        assert assistant.tool_calls == (call,)
        assert assistant.native_assistant.message_json == first.native_assistant.message_json
        archived = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert archived.messages[0].native_assistant.message_json == first.native_assistant.message_json
        assert not any(entry.compaction_json for entry in runtime.history_snapshot)
    asyncio.run(run())


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("dimension", ["characters", "bytes"])
def test_full_tool_arguments_are_bounded_before_host_side_effect(native, dimension):
    async def run():
        arguments = '{"text":"' + "密" * 30 + '"}'
        call = ToolCall("short", "host_operation", arguments)
        first = native_completion(call, "Reason.") if native else ModelCompletion("Reason.", tool_calls=(call,))
        values = ("Reason.", call.name, arguments)
        amount = sum(len(value) if dimension == "characters" else len(value.encode("utf-8")) for value in values)
        cfg = SimpleRuntimeConfig(**{"max_action_output_" + dimension: amount - 1})
        executed = []
        async def model(req):
            return first
        runtime = SimpleRuntime(CallableModel(model), cfg, tools=registry(executed))
        with pytest.raises(AgentLimitError, match="action output"):
            await runtime.complete(request())
        assert executed == []
        assert runtime.history_snapshot[-1].messages[0].tool_calls == (call,)
        assert runtime.history_snapshot[-1].messages[0].native_assistant == first.native_assistant
    asyncio.run(run())


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("dimension", ["characters", "bytes"])
def test_visible_final_text_retains_its_existing_action_limit(native, dimension):
    async def run():
        text = "可见回复"
        state = NativeAssistantState(SCOPE, json.dumps({"role": "assistant", "content": text,
            "reasoning": "opaque" * 1000}, ensure_ascii=False)) if native else None
        async def model(req):
            return ModelCompletion(text, native_assistant=state)
        amount = len(text) if dimension == "characters" else len(text.encode("utf-8"))
        cfg = SimpleRuntimeConfig(**{"max_action_output_" + dimension: amount})
        exact = SimpleRuntime(CallableModel(model), cfg)
        assert (await exact.complete(request())).text == text
        below = SimpleRuntime(CallableModel(model), replace(cfg, **{"max_action_output_" + dimension: amount - 1}))
        with pytest.raises(AgentLimitError, match="action output"):
            await below.complete(request())
    asyncio.run(run())


@pytest.mark.parametrize("native", [False, True])
def test_protocol_budget_bounds_native_packets_and_portable_ids_before_execution(native):
    async def run():
        call = ToolCall("opaque-" + "x" * 300, "host_operation", ARGUMENTS)
        first = native_completion(call) if native else ModelCompletion("", tool_calls=(call,))
        amount = len(call.call_id.encode())
        if native:
            amount += len(first.native_assistant.message_json.encode()) + len(SCOPE.encode())
        turns = 0
        async def model(req):
            nonlocal turns
            turns += 1
            return first if turns == 1 else ModelCompletion("done")
        executed = []
        exact = SimpleRuntime(CallableModel(model),
            SimpleRuntimeConfig(max_protocol_output_bytes=amount), tools=registry(executed))
        assert (await exact.complete(request())).text == "done"
        assert executed == ["visible argument"]
        executed.clear()
        turns = 0
        runtime = SimpleRuntime(CallableModel(model),
            SimpleRuntimeConfig(max_protocol_output_bytes=amount - 1), tools=registry(executed))
        with pytest.raises(AgentLimitError, match="protocol output"):
            await runtime.complete(request())
        assert executed == []
        assert runtime.history_snapshot[-1].messages[0].tool_calls == (call,)
    asyncio.run(run())


def test_submission_pair_protocol_budget_includes_call_and_result_ids():
    async def run():
        call = ToolCall("state-" + "x" * 1000, "submit_action", '{"format":"raw","content":"selected"}')
        source = ModelCompletion("Reason.", tool_calls=(call,))
        translated, _ = translate_submission(source, max_bytes=1000, max_characters=1000)
        async def model(req):
            return translated
        cfg = SimpleRuntimeConfig(max_protocol_output_bytes=2 * len(call.call_id.encode()))
        exact = SimpleRuntime(CallableModel(model), cfg)
        assert (await exact.complete(request())).text == "selected"
        below = SimpleRuntime(CallableModel(model), replace(cfg,
            max_protocol_output_bytes=cfg.max_protocol_output_bytes - 1))
        with pytest.raises(AgentLimitError, match="protocol output"):
            await below.complete(request())
        assert below.history_snapshot[-1].messages == translated.context_messages
    asyncio.run(run())


def test_portable_batch_ids_share_one_protocol_budget():
    async def run():
        calls = tuple(ToolCall(str(i) + "x" * 100, "host_operation", ARGUMENTS) for i in range(3))
        async def model(req):
            return ModelCompletion("", tool_calls=calls)
        executed = []
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_protocol_output_bytes=302),
                                tools=registry(executed))
        with pytest.raises(AgentLimitError, match="protocol output"):
            await runtime.complete(request())
        assert executed == []
    asyncio.run(run())


@pytest.mark.parametrize("native", [False, True])
def test_translated_submission_counts_original_reason_and_arguments_once(native):
    async def run():
        arguments = '{"format":"raw","content":"selected action"}'
        call = ToolCall("signature-" + "x" * 1000, "submit_action", arguments)
        source = native_completion(call, "Original visible reason.") if native else ModelCompletion(
            "Original visible reason.", tool_calls=(call,))
        translated, event = translate_submission(source, max_bytes=1000, max_characters=1000)
        assert event is not None and translated.text == "selected action"
        exact_amount = len(source.text) + len(call.name) + len(arguments)
        async def model(req):
            return translated
        cfg = SimpleRuntimeConfig(max_action_output_characters=exact_amount)
        exact = SimpleRuntime(CallableModel(model), cfg)
        assert (await exact.complete(request())).text == "selected action"
        assert exact.history_snapshot[-1].messages == translated.context_messages
        below = SimpleRuntime(CallableModel(model), replace(cfg, max_action_output_characters=exact_amount - 1))
        with pytest.raises(AgentLimitError, match="action output"):
            await below.complete(request())
        # Original rationale must count, even when the selected action itself fits.
        only_arguments = SimpleRuntime(CallableModel(model), replace(cfg,
            max_action_output_characters=len(call.name) + len(arguments)))
        with pytest.raises(AgentLimitError, match="action output"):
            await only_arguments.complete(request())
    asyncio.run(run())


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reason", ["r" * 50, "密" * 170])
@pytest.mark.parametrize("dimension", ["characters", "bytes"])
def test_expanding_submission_bounds_each_complete_representation_once(native, reason, dimension):
    async def run():
        content = json.dumps({"name": "x", "arguments": {"s": "<" * 100}}, separators=(",", ":"))
        arguments = json.dumps({"format": "tool_call", "content": content},
                               ensure_ascii=False, separators=(",", ":"))
        call = ToolCall("expanding-submission", "submit_action", arguments)
        source = native_completion(call, reason) if native else ModelCompletion(reason, tool_calls=(call,))
        translated, event = translate_submission(source, max_bytes=10_000, max_characters=10_000)
        assert event is not None and "\\u003c" in translated.text
        original_characters = len(reason) + len(call.name) + len(arguments)
        original_bytes = sum(len(value.encode("utf-8")) for value in (reason, call.name, arguments))
        translated_characters = len(translated.text)
        translated_bytes = len(translated.text.encode("utf-8"))
        assert translated_characters > original_characters
        if not reason.isascii():
            # UTF-8 admission must independently protect the original model
            # output even when character admission is dominated by expansion.
            assert original_bytes > translated_bytes
        amount = max(original_characters, translated_characters) if dimension == "characters" else max(
            original_bytes, translated_bytes)
        async def model(req):
            return translated
        cfg = SimpleRuntimeConfig(**{"max_action_output_" + dimension: amount})
        exact = SimpleRuntime(CallableModel(model), cfg)
        assert (await exact.complete(request())).text == translated.text
        assert exact.history_snapshot[-1].messages == translated.context_messages
        below = SimpleRuntime(CallableModel(model), replace(cfg,
            **{"max_action_output_" + dimension: amount - 1}))
        with pytest.raises(AgentLimitError, match="action output"):
            await below.complete(request())
        assert below.history_snapshot[-1].messages == translated.context_messages
    asyncio.run(run())


def test_existing_type_bounds_reject_oversized_ids_and_native_packets():
    with pytest.raises(ValueError, match="tool call exceeds"):
        ToolCall("x" * (64 * 1024 + 1), "host_operation", ARGUMENTS)
    with pytest.raises(ValueError, match="state exceeds"):
        NativeAssistantState(SCOPE, json.dumps({"role": "assistant", "content": "done",
                                               "reasoning": "x" * 1024 * 1024}))


@pytest.mark.parametrize("native", [False, True])
def test_oversized_rejected_submission_arguments_stop_before_feedback(native):
    async def run():
        call = ToolCall("bad-selection", "submit_action", json.dumps({"format": "code", "content": "x" * 100}))
        source = native_completion(call, "Reason.") if native else ModelCompletion("Reason.", tool_calls=(call,))
        requests, executed = [], []
        async def model(req):
            requests.append(req)
            raise SubmissionResponseError(source, "invalid submit_action: decoded action exceeds its bound")
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_action_output_bytes=64),
                                tools=registry(executed))
        with pytest.raises(AgentLimitError, match="action output"):
            await runtime.complete(request())
        assert len(requests) == 1 and executed == []
        assert not any(entry.kind == "tool_exchange" for entry in runtime.history_snapshot)
        archived = runtime.history_snapshot[-1]
        assert archived.kind == "completion" and archived.text == source.text
        assert archived.messages[0].tool_calls == (call,)
        assert archived.messages[0].native_assistant == source.native_assistant
    asyncio.run(run())


def test_long_signature_still_counts_toward_context_tokens_and_triggers_fresh_context():
    async def run():
        first = native_completion(ToolCall("call_" + "opaque" * 6972, "host_operation", ARGUMENTS))
        seen, executed = [], []
        async def model(req):
            seen.append(req)
            if req.purpose == "summary":
                raise AssertionError("Opaque-state retirement alone should fit")
            return first if len(seen) == 1 else ModelCompletion("done")
        cfg = SimpleRuntimeConfig(max_context_characters=20_000,
            max_context_tokens=8000, output_reserve_tokens=128)
        runtime = SimpleRuntime(CallableModel(model), cfg, tools=registry(executed))
        original = Message("assistant", (), first.tool_calls, native_assistant=first.native_assistant)
        plain = Message.text("assistant", "")
        assert _metrics((original,))[0] > cfg.max_context_characters
        assert runtime._tokens((original,), 0) > cfg.max_context_tokens
        assert runtime._tokens((plain,), 0) < cfg.max_context_tokens
        assert not runtime._fits((original,))
        assert (await runtime.complete(request())).text == "done"
        assert executed == ["visible argument"]
        following = seen[1].messages
        assert all(message.native_assistant is None and not message.tool_calls and message.role != "tool"
                   for message in following)
        visible = "\n".join(part.text for message in following for part in message.parts if isinstance(part, TextPart))
        assert ARGUMENTS in visible and "visible result" in visible
        assert "call_opaque" not in visible
        restart = next(entry for entry in runtime.history_snapshot if entry.compaction_json)
        assert json.loads(restart.compaction_json)["stage"] == "restart"
        archived = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert archived.messages[0].native_assistant.message_json == first.native_assistant.message_json
    asyncio.run(run())


def test_explicit_legacy_policy_reproduces_native_action_guard_and_unchanged_archive():
    async def run():
        first = native_completion(ToolCall("call_" + "opaque" * 6972, "host_operation", ARGUMENTS))
        async def model(req):
            return first
        executed = []
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(output_budget_policy="legacy_v1"),
                                tools=registry(executed))
        with pytest.raises(AgentLimitError, match="action output"):
            await runtime.complete(request())
        assert executed == []
        assert runtime.history_snapshot[-1].messages[0].native_assistant == first.native_assistant
    asyncio.run(run())
