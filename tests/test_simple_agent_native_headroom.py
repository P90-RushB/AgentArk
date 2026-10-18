"""Offline fresh-decision headroom; no HTTP, Player, SDK or model access."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from unittest.mock import patch

import pytest

from agent_ark.agent.simple_agent import CallableModel, SimpleRuntime, SimpleRuntimeConfig, ToolRegistry
from agent_ark.agent.simple_agent.encoding import api_chat_messages
from agent_ark.agent.simple_agent.runtime import _metrics
from agent_ark.agent.simple_agent.types import (
    AgentEvent, AgentInputIdentity, AgentLimitError, AgentRequest, ImagePart, Message,
    ModelCompletion, ModelUsage, NativeAssistantState, TextPart, ToolCall, ToolSpec,
)


SCOPE = "offline:headroom"
PRIVATE = "opaque-private-state"
USAGE = ModelUsage(3, 5, 8)


def request(number, text="observation", image=None):
    parts = (TextPart(text),) + ((image,) if image is not None else ())
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "agent", 0, str(number)),
        (Message.text("system", "Use the visible observation."), Message("user", parts)))


def native(text="a", *, reason=PRIVATE, calls=()):
    payload = {"role": "assistant", "content": text, "reasoning": reason}
    if calls:
        payload["tool_calls"] = [{"id": call.call_id, "type": "function", "function": {
            "name": call.name, "arguments": call.arguments_json}} for call in calls]
    return NativeAssistantState(SCOPE, json.dumps(payload, separators=(",", ":")))


def registry(results=None):
    tools = ToolRegistry()
    tools.register(ToolSpec("echo", "Return visible text.", '{"type":"object"}'),
        lambda args, session: (TextPart((results or {}).get(args.get("index"), "visible reply")),))
    return tools


def metrics(runtime, messages=None):
    return _metrics((runtime.working_snapshot if messages is None else messages)
                    + (runtime._tool_budget_message(),))


def exchange_size(round_number, target, metric):
    """A valid native host exchange with an exact additive measured cost."""
    call = ToolCall(f"host-{round_number}", "echo", json.dumps({"index": round_number}))
    sidecar = native(None, calls=(call,))
    assistant = Message("assistant", (), (call,), native_assistant=sidecar)
    empty = Message("tool", (TextPart(""),), tool_call_id=call.call_id)
    prefix = (Message.text("user", "prefix"),)
    overhead = _metrics(prefix + (assistant, empty))[metric] - _metrics(prefix)[metric]
    assert target > overhead
    return ModelCompletion("", USAGE, (call,), native_assistant=sidecar), "r" * (target - overhead)


@pytest.mark.parametrize("history_count,target,metric,increments,image_bytes,reason_size", [
    (4, 489_182, 0, (63_660,), 0, 80_000),
    (40, 15_943_648, 1, (14_635, 14_636, 14_636, 14_636), 298_000, 100),
])
@pytest.mark.parametrize("soft_policy", [False, True])
def test_recorded_pressure_values_compact_before_valid_native_host_rounds(
        history_count, target, metric, increments, image_bytes, reason_size, soft_policy):
    """Synthetic clean prefixes match the measured Opus/DeepSeek pressures.

    These are size regressions, not copies of source audit records. The old
    policy reproduces 552842 characters / 16002191 bytes exactly. The new
    policy summarizes before opening that native turn, then keeps its prefix.
    """
    async def run():
        host_pairs = [exchange_size(i, size, metric) for i, size in enumerate(increments)]
        results = {i: text for i, (_, text) in enumerate(host_pairs)}
        history_reply = ModelCompletion("a" * (10_000 if not image_bytes else 1), USAGE,
            native_assistant=native("a" * (10_000 if not image_bytes else 1), reason="o" * reason_size))
        replies = iter([history_reply] * history_count + [reply for reply, _ in host_pairs]
                       + [ModelCompletion("finished", USAGE, native_assistant=native("finished"))])
        calls = []

        async def model(value):
            calls.append(value)
            if value.purpose == "summary":
                assert all(message.native_assistant is None for message in value.messages)
                assert PRIVATE not in repr(value.messages)
                return ModelCompletion("Earlier complete observations and actions retained.", USAGE)
            return next(replies)

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(), tools=registry(results))
        image = ImagePart(b"i" * image_bytes) if image_bytes else None
        # Rehydrate only complete old decisions under their original admission
        # policy. No failed partial tool prefix enters the new decision.
        with patch.object(runtime, "_fresh_decision_fits", runtime._fits):
            for number in range(history_count):
                await runtime.complete(request(number, image=image))
        saved_history = runtime.history_snapshot
        blank = request(history_count, "")
        base = metrics(runtime, runtime.working_snapshot + blank.messages[1:])[metric]
        current = request(history_count, "x" * (target - base))
        runtime.accept_input(current)
        assert metrics(runtime)[metric] == target
        assert runtime._fits(runtime.working_snapshot)
        assert not runtime._fresh_decision_fits(runtime.working_snapshot)
        before_calls = len(calls)
        before_usage = runtime.usage
        policy = patch.object(runtime, "_fresh_decision_fits", runtime._fits) if not soft_policy else patch.object(
            runtime, "_fresh_decision_fits", wraps=runtime._fresh_decision_fits)
        with policy, patch.object(runtime, "_compact", wraps=runtime._compact) as compact:
            if soft_policy:
                assert (await runtime.complete(current)).text == "finished"
            else:
                with pytest.raises(AgentLimitError, match="native tool continuation cannot fit"):
                    await runtime.complete(current)
                assert metrics(runtime)[metric] == target + sum(increments)
            assert compact.call_count == 1  # Never compact inside the native turn.
        fresh_calls = calls[before_calls:]
        if soft_policy:
            assert fresh_calls[0].purpose == "summary"
            actions = [value for value in fresh_calls if value.purpose == "action"]
            assert len(actions) == len(increments) + 1
            assert runtime._fresh_decision_fits(actions[0].messages)
            assert all(entry in runtime.history_snapshot for entry in saved_history)
            assert any(message.native_assistant is not None for entry in saved_history for message in entry.messages)
            for previous, following in zip(actions, actions[1:]):
                assert following.messages[:len(previous.messages)] == previous.messages
                assert following.messages[-2].native_assistant is not None
            assert runtime.usage.summary_calls > before_usage.summary_calls
        else:
            assert len(fresh_calls) == len(increments)
            assert runtime.usage.summary_calls == before_usage.summary_calls == 0
        assert all(value.max_output_tokens == (runtime.config.summary_output_tokens if value.purpose == "summary"
                   else runtime.config.output_reserve_tokens) for value in calls)
    asyncio.run(run())


@pytest.mark.parametrize("metric", [0, 1])
def test_native_watermark_is_inclusive_and_keeps_hard_caps(metric):
    runtime = SimpleRuntime(CallableModel(lambda _: asyncio.sleep(0)), tools=registry())
    messages = request(1, "text").messages
    cost = metrics(runtime, messages)[metric]
    # Choose a hard cap whose integer 75% watermark is exactly this cost.
    limit = (cost * 4 + 2) // 3
    name = "max_context_characters" if metric == 0 else "max_context_bytes"
    runtime.config = replace(runtime.config, **{name: limit})
    assert limit * 3 // 4 == cost
    assert runtime._fresh_decision_fits(messages)
    larger = (*messages[:-1], Message.text("user", "textx"))
    assert not runtime._fresh_decision_fits(larger)
    assert runtime._fits(larger)
    runtime.config = replace(runtime.config, **{name: cost})
    assert runtime._fits(messages)
    assert not runtime._fits(larger)


@pytest.mark.parametrize("summary_calls", [0, 1])
def test_protected_first_input_above_watermark_falls_back_without_summary(summary_calls):
    async def run():
        seen = []
        async def model(value):
            seen.append(value)
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_summary_calls=summary_calls), tools=registry())
        current = request(1, "p" * 800)
        exact = metrics(runtime, current.messages)[0]
        runtime.config = replace(runtime.config, max_context_characters=exact)
        assert not runtime._fresh_decision_fits(current.messages)
        assert (await runtime.complete(current)).text == "done"
        assert [value.purpose for value in seen] == ["action"]
        assert runtime.usage.summary_calls == 0
    asyncio.run(run())


@pytest.mark.parametrize("summary_calls", [0, 1])
def test_soft_summary_allowance_is_best_effort_but_hard_overflow_still_fails(summary_calls):
    async def run():
        seen = []
        async def model(value):
            seen.append(value.purpose)
            return ModelCompletion("" if value.purpose == "summary" else "done")
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            max_summary_calls=summary_calls, recent_turns=1), tools=registry())
        with patch.object(runtime, "_fresh_decision_fits", runtime._fits):
            await runtime.complete(request(1, "old " + "o" * 1_000))
        current = request(2, "current " + "c" * 1_000)
        runtime.accept_input(current)
        hard = metrics(runtime)[0]
        runtime.config = replace(runtime.config, max_context_characters=hard)
        assert not runtime._fresh_decision_fits(runtime.working_snapshot)
        assert (await runtime.complete(current)).text == "done"
        assert runtime.usage.summary_calls == summary_calls
        assert runtime._fits(runtime.working_snapshot[:-1])
        # Summary retries do not enlarge or reset any existing allowance.
        assert runtime._summary_attempts.get(current.identity, 0) == summary_calls
        oversized = request(3, "z" * (hard + 1))
        with pytest.raises(AgentLimitError):
            await runtime.complete(oversized)
    asyncio.run(run())


@pytest.mark.parametrize("tools_enabled", [False, True])
def test_below_pressure_wire_prefix_and_tools_disabled_policy_are_unchanged(tools_enabled):
    async def run(policy):
        seen = []
        async def model(value):
            seen.append(value)
            return ModelCompletion("answer", native_assistant=native("answer"))
        runtime = SimpleRuntime(CallableModel(model), tools=registry() if tools_enabled else None)
        with patch.object(runtime, "_fresh_decision_fits", policy(runtime)):
            await runtime.complete(request(1))
            await runtime.complete(request(2))
        return [api_chat_messages(value.messages, native_scope=SCOPE) for value in seen], runtime
    old, _ = asyncio.run(run(lambda runtime: runtime._fits))
    new, runtime = asyncio.run(run(lambda runtime: runtime._fresh_decision_fits))
    assert new == old
    assert runtime.usage.summary_calls == 0
    if not tools_enabled:
        large = request(3, "x" * 400_000).messages
        assert runtime._fresh_decision_fits(large) == runtime._fits(large) is True



def test_native_continuation_crosses_soft_watermark_without_compaction_or_prefix_change():
    async def run():
        reply, result = exchange_size(0, 1_000, 0)
        replies = iter((reply, ModelCompletion("finished", USAGE)))
        calls = []
        async def model(value):
            calls.append(value)
            assert value.purpose == "action"
            return next(replies)
        runtime = SimpleRuntime(CallableModel(model), tools=registry({0: result}))
        current = request(1)
        initial_cost = metrics(runtime, current.messages)[0]
        runtime.config = replace(runtime.config, max_context_characters=initial_cost + 1_010)
        assert runtime._fresh_decision_fits(current.messages)
        with patch.object(runtime, "_compact", wraps=runtime._compact) as compact:
            assert (await runtime.complete(current)).text == "finished"
            assert compact.call_count == 1
        first, continuation = calls
        assert metrics(runtime, continuation.messages)[0] == initial_cost + 1_000
        assert runtime._fits(continuation.messages)
        assert not runtime._fresh_decision_fits(continuation.messages)
        assert continuation.messages[:len(first.messages)] == first.messages
        assert api_chat_messages(continuation.messages[:len(first.messages)], native_scope=SCOPE) == (
            api_chat_messages(first.messages, native_scope=SCOPE))
        assert continuation.messages[-2].native_assistant is reply.native_assistant
        assert continuation.messages[-1].tool_call_id == reply.tool_calls[0].call_id
        assert runtime.usage.action_calls == 2 and runtime.usage.summary_calls == 0
    asyncio.run(run())


def test_successful_summary_above_soft_watermark_commits_after_summary_allowance_exhaustion():
    async def run():
        calls = []
        async def model(value):
            calls.append(value)
            if value.purpose == "summary":
                assert all(message.native_assistant is None for message in value.messages)
                return ModelCompletion("Keep the useful earlier observation.", USAGE)
            if len([item for item in calls if item.purpose == "action"]) <= 2:
                return ModelCompletion("a", USAGE, native_assistant=native("a"))
            return ModelCompletion("finished", USAGE)
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            max_summary_calls=1, recent_turns=2), tools=registry())
        with patch.object(runtime, "_fresh_decision_fits", runtime._fits):
            await runtime.complete(request(1, "old " + "o" * 800))
            await runtime.complete(request(2, "retained " + "r" * 1_000))
        runtime.observe_result(AgentEvent(request(2).identity, "required-feedback", "action.result",
                                         json.dumps({"feedback": "f" * 300})))
        current = request(3, "current " + "c" * 4_000)
        runtime.accept_input(current)
        before = runtime.working_snapshot
        archive = runtime.history_snapshot
        hard = metrics(runtime)[0]
        runtime.config = replace(runtime.config, max_context_characters=hard)
        assert runtime._fits(before) and not runtime._fresh_decision_fits(before)
        assert (await runtime.complete(current)).text == "finished"
        action = calls[-1]
        assert action.purpose == "action"
        assert metrics(runtime, action.messages)[0] < hard
        assert runtime._fits(action.messages) and not runtime._fresh_decision_fits(action.messages)
        assert runtime.usage.summary_calls == runtime._summary_attempts[current.identity] == 1
        assert [item.purpose for item in calls] == ["action", "action", "summary", "action"]
        assert action.messages == runtime.working_snapshot[:-1]
        assert all(message.native_assistant is None for message in action.messages)
        assert any("Model notes; not environment facts" in part.text
                   for message in action.messages for part in message.parts if isinstance(part, TextPart))
        assert any("required-feedback" == entry.event_id for entry in archive)
        assert any("f" * 300 in part.text for message in action.messages for part in message.parts
                   if isinstance(part, TextPart))
        assert runtime.history_snapshot[:len(archive)] == archive
        assert sum(message.native_assistant is not None for entry in archive for message in entry.messages) == 2
    asyncio.run(run())
