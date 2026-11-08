"""Append-only main-response budget snapshots; no SDK, HTTP or Unity."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json

import pytest

from agent_ark.agent.simple_agent import CallableModel, SimpleRuntime, SimpleRuntimeConfig, ToolLimits, ToolRegistry
from agent_ark.agent.simple_agent.action_submission import SUBMISSION_SPEC, translate_submission
from agent_ark.agent.simple_agent.encoding import api_chat_messages, encode_png
from agent_ark.agent.simple_agent.runtime import _metrics
from agent_ark.agent.simple_agent.types import (
    AgentInputIdentity, AgentLimitError, AgentRequest, ImagePart, Message, ModelCompletion,
    NativeAssistantState, TextPart, ToolCall, ToolSpec,
)


LABEL = "Runtime decision-budget snapshot for the upcoming main-model response; earlier snapshots are historical.\n"
SCOPE = "offline:round-budget"


def request(number=1, text="Observe and finish the visible task."):
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "agent", 0, str(number)), (
        Message.text("user", text),))


def snapshots(messages):
    return [json.loads(part.text.splitlines()[1]) for message in messages for part in message.parts
            if isinstance(part, TextPart) and part.text.startswith(LABEL)]


def budget_events(runtime):
    result = []
    for entry in runtime.history_snapshot:
        if entry.kind != "event":
            continue
        try:
            value = json.loads(entry.text)
        except ValueError:
            continue
        if value.get("kind") == "round_budget":
            result.append((entry, value))
    return result


def registry(executed, *, rounds=3, calls=8, submit=False, handler=None):
    tools = ToolRegistry(limits=ToolLimits(max_model_rounds=rounds, max_calls=calls))
    def host(arguments, session):
        executed.append(arguments["number"])
        return handler(arguments, session) if handler is not None else (TextPart(f"Result {arguments['number']}"),)
    tools.register(ToolSpec("host_operation", "One visible operation.",
        '{"type":"object","properties":{"number":{"type":"integer"}},"required":["number"]}'), host)
    if submit:
        def forbidden(arguments, session):
            raise AssertionError("A translated final submission must not execute as a host tool")
        tools.register(SUBMISSION_SPEC, forbidden)
    return tools


def host_call(index, *, identifier=None):
    return ToolCall(identifier or f"host-{index}", "host_operation", json.dumps({"number": index}))


def native(calls):
    value = {"role": "assistant", "content": None, "reasoning": "opaque original state",
        "tool_calls": [{"id": call.call_id, "type": "function", "function": {
            "name": call.name, "arguments": call.arguments_json}} for call in calls]}
    return NativeAssistantState(SCOPE, json.dumps(value, separators=(",", ":")))


def cfg(**updates):
    return SimpleRuntimeConfig(**dict(max_context_tokens=100_000, output_reserve_tokens=128, **updates))


def test_budget_policy_default_and_explicit_legacy():
    assert SimpleRuntimeConfig().round_budget_policy == "visible_v1"
    assert SimpleRuntimeConfig(round_budget_policy="legacy_v1").round_budget_policy == "legacy_v1"
    with pytest.raises(ValueError, match="round budget policy"):
        SimpleRuntimeConfig(round_budget_policy="unknown")


def test_every_main_request_sees_counts_and_final_response_hint_without_inventing_a_tool():
    async def run():
        seen, executed = [], []
        async def model(req):
            seen.append(req)
            if len(seen) < 3:
                return ModelCompletion("", tool_calls=(host_call(len(seen)),))
            return ModelCompletion("final answer")
        runtime = SimpleRuntime(CallableModel(model), tools=registry(executed))
        assert (await runtime.complete(request())).text == "final answer"
        assert executed == [1, 2]
        for index, req in enumerate(seen, 1):
            values = snapshots(req.messages)
            assert len(values) == index
            assert [value["current_round"] for value in values] == list(range(1, index + 1))
            current = values[-1]
            assert current == {"kind": "round_budget", "policy": "visible_v1", "current_round": index,
                "total_rounds": 3, "remaining_rounds_including_this_response": 4 - index,
                "host_calls_used": index - 1, "host_calls_limit": 8, "host_calls_remaining": 9 - index,
                "final_response_required": index == 3}
            assert req.messages[-1].role == "user"
            assert req.tools == seen[0].tools
        last = seen[-1].messages[-1].parts[0].text
        assert "final available main-model response" in last and "submit_action" not in last
        assert "the task's required response format" in last
        assert len(budget_events(runtime)) == 3
        assert not runtime._required_events
        assert all(len(entry.messages) == 2 and entry.messages[0].role == "assistant"
                   and entry.messages[1].role == "tool" for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
    asyncio.run(run())


def test_parallel_host_calls_consume_one_main_round_and_separate_host_slots():
    async def run():
        seen, executed = [], []
        async def model(req):
            seen.append(req)
            if len(seen) == 1:
                return ModelCompletion("", tool_calls=(host_call(1), host_call(2)))
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), tools=registry(executed, rounds=2))
        await runtime.complete(request())
        assert executed == [1, 2]
        assert snapshots(seen[1].messages)[-1]["current_round"] == 2
        assert snapshots(seen[1].messages)[-1]["host_calls_used"] == 2
        assert snapshots(seen[1].messages)[-1]["remaining_rounds_including_this_response"] == 1
    asyncio.run(run())


def test_final_submit_uses_the_last_slot_without_being_executed_as_a_host_call():
    async def run():
        seen, executed = [], []
        async def model(req):
            seen.append(req)
            if len(seen) < 3:
                return ModelCompletion("", tool_calls=(host_call(len(seen)),))
            assert "exposed submit_action tool and its current schema" in req.messages[-1].parts[0].text
            selection = ModelCompletion("Ready.", tool_calls=(ToolCall("selected", "submit_action",
                '{"format":"raw","content":"selected action"}'),))
            translated, event = translate_submission(selection, max_bytes=1000, max_characters=1000)
            assert event is not None
            return translated
        runtime = SimpleRuntime(CallableModel(model), tools=registry(executed, submit=True))
        assert (await runtime.complete(request())).text == "selected action"
        assert len(seen) == 3 and executed == [1, 2]
        assert runtime.usage.action_calls == 3
        assert snapshots(seen[-1].messages)[-1]["host_calls_used"] == 2
    asyncio.run(run())


def test_ignoring_final_hint_retains_original_hard_round_exhaustion():
    async def run():
        seen, executed = [], []
        async def model(req):
            seen.append(req)
            return ModelCompletion("", tool_calls=(host_call(len(seen)),))
        runtime = SimpleRuntime(CallableModel(model), tools=registry(executed, rounds=2, submit=True))
        with pytest.raises(AgentLimitError, match="^model tool round budget exhausted$"):
            await runtime.complete(request())
        assert len(seen) == 2 and executed == [1, 2]
        assert snapshots(seen[-1].messages)[-1]["final_response_required"] is True
        assert seen[0].tools == seen[1].tools
        assert runtime.history_snapshot[-1].kind == "tool_exchange"
    asyncio.run(run())


def test_round_and_host_counts_reset_for_each_decision_without_clearing_history():
    async def run():
        seen, executed = [], []
        async def model(req):
            seen.append(req)
            current = snapshots(req.messages)[-1]
            if current["current_round"] == 1:
                return ModelCompletion("", tool_calls=(host_call(len(seen)),))
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), tools=registry(executed, rounds=2))
        await runtime.complete(request(1))
        previous = tuple(budget_events(runtime))
        await runtime.complete(request(2))
        current = snapshots(seen[2].messages)[-1]
        assert current["current_round"] == 1 and current["host_calls_used"] == 0
        assert current["remaining_rounds_including_this_response"] == 2
        assert tuple(budget_events(runtime))[:2] == previous
        assert runtime.usage.action_calls == 4
    asyncio.run(run())


def test_native_continuation_prefix_and_all_prior_snapshots_stay_byte_exact():
    async def run():
        seen, executed = [], []
        call = host_call(1)
        original = native((call,))
        async def model(req):
            seen.append(req)
            return ModelCompletion("", tool_calls=(call,), native_assistant=original) if len(seen) == 1 else ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), tools=registry(executed, rounds=2))
        await runtime.complete(request())
        prefix = api_chat_messages(seen[0].messages, SCOPE)
        following = api_chat_messages(seen[1].messages, SCOPE)
        assert following[:len(prefix)] == prefix
        assert following[len(prefix)] == original.message
        assert following[len(prefix) + 1]["role"] == "tool"
        assert following[-1]["role"] == "user"
        assert snapshots(seen[1].messages)[0] == snapshots(seen[0].messages)[0]
        assert next(message.native_assistant for message in seen[1].messages if message.native_assistant).message_json == original.message_json
    asyncio.run(run())


@pytest.mark.parametrize("policy", ["visible_v1", "legacy_v1"])
def test_without_tools_both_policies_preserve_the_same_wire_and_archive(policy):
    async def run():
        seen = []
        async def model(req):
            seen.append(req)
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), cfg(round_budget_policy=policy))
        await runtime.complete(request())
        assert seen[0].messages == request().messages
        assert not snapshots(seen[0].messages) and not budget_events(runtime)
        assert [entry.kind for entry in runtime.history_snapshot] == ["input", "completion"]
    asyncio.run(run())


def test_explicit_legacy_preserves_original_tool_wire_metrics_and_archive_structure():
    async def run():
        seen, executed = [], []
        call = host_call(1)
        async def model(req):
            seen.append(req)
            return ModelCompletion("", tool_calls=(call,)) if len(seen) == 1 else ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), cfg(round_budget_policy="legacy_v1"), tools=registry(executed))
        await runtime.complete(request())
        assert seen[0].messages == request().messages
        assert seen[1].messages == (*request().messages,
            Message("assistant", (), (call,)), Message("tool", (TextPart("Result 1"),), tool_call_id=call.call_id))
        assert not budget_events(runtime)
        assert [entry.kind for entry in runtime.history_snapshot] == ["input", "event", "tool_exchange", "completion"]
    asyncio.run(run())


@pytest.mark.parametrize("dimension", ["characters", "bytes", "tokens"])
def test_current_snapshot_counts_toward_every_context_admission_limit(dimension):
    async def run():
        seen = []
        async def model(req):
            seen.append(req)
            return ModelCompletion("done")
        probe = SimpleRuntime(CallableModel(model), tools=registry([]))
        await probe.complete(request())
        messages = seen[0].messages
        counted = (*messages, probe._tool_budget_message())
        characters, byte_count, _, _ = _metrics(counted)
        amount = {"characters": characters, "bytes": byte_count,
                  "tokens": probe._tokens(counted, byte_count) + probe.config.output_reserve_tokens}[dimension]
        assert _metrics(counted)[0] > _metrics((*request().messages, probe._tool_budget_message()))[0]
        cfg_below = replace(probe.config, **{"max_context_" + dimension: amount - 1})
        seen.clear()
        bounded = SimpleRuntime(CallableModel(model), cfg_below, tools=registry([]))
        with pytest.raises(AgentLimitError, match="context bounds"):
            await bounded.complete(request())
        assert seen == [] and len(budget_events(bounded)) == 1
    asyncio.run(run())


@pytest.mark.parametrize("dimension", ["entries", "characters", "bytes"])
def test_snapshot_is_an_immutable_archive_entry_under_existing_hard_limits(dimension):
    async def run():
        seen = []
        async def model(req):
            seen.append(req)
            return ModelCompletion("done")
        probe = SimpleRuntime(CallableModel(model), tools=registry([]))
        await probe.complete(request())
        budget = budget_events(probe)[0][0]
        if dimension == "entries":
            amount = budget.sequence
        else:
            index = 0 if dimension == "characters" else 1
            amount = sum(probe._archive_cost(entry)[index] for entry in probe.history_snapshot if entry.sequence <= budget.sequence)
        seen.clear()
        bounded = SimpleRuntime(CallableModel(model), replace(probe.config,
            **{"max_archive_" + dimension: amount - 1}), tools=registry([]))
        with pytest.raises(AgentLimitError, match="immutable archive"):
            await bounded.complete(request())
        assert seen == [] and [entry.kind for entry in bounded.history_snapshot] == ["input"]
    asyncio.run(run())


def test_summary_helpers_skip_internal_snapshots_but_keep_identical_environment_text():
    runtime = SimpleRuntime(CallableModel(lambda req: None), tools=registry([]))
    initial = request()
    runtime.accept_input(initial)
    session = runtime.tools.begin(initial)
    try:
        budget = runtime._append_round_budget(initial.identity, session, 1)
        same_text = budget.messages[0].parts[0].text
        runtime.accept_input(request(2, same_text))
        actual_input = runtime._working[-1]
        span = (budget, actual_input)
        summary = runtime._summary_messages(span)
        assert sum(part.text == same_text for message in summary for part in message.parts if isinstance(part, TextPart)) == 1
        for atoms in (runtime._chunk_atoms(span), runtime._tool_round_atoms(span)):
            assert sum(part.text == same_text for atom in atoms for message in atom.messages
                       for part in message.parts if isinstance(part, TextPart)) == 1
            assert all(reference.archive_sequence != budget.sequence for atom in atoms for reference in atom.references)
    finally:
        session.close()


def test_fresh_context_retires_old_snapshot_preserves_current_counter_and_unread_result():
    async def run():
        image = ImagePart(encode_png(bytes((0, 0, 255)) * 4, 2, 2), width=2, height=2)
        result = (TextPart("Exact latest visible tool result."), image)
        seen, executed = [], []
        call = host_call(1, identifier="opaque-" + "x" * 6000)
        first = ModelCompletion("", tool_calls=(call,), native_assistant=native((call,)))
        async def model(req):
            seen.append(req)
            if req.purpose == "summary":
                raise AssertionError("Retiring opaque state and old snapshots should suffice")
            return first if len(seen) == 1 else ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), cfg(max_context_characters=5000),
            tools=registry(executed, rounds=2, handler=lambda args, session: result))
        assert (await runtime.complete(request())).text == "done"
        assert executed == [1]
        following = seen[1].messages
        assert [value["current_round"] for value in snapshots(following)] == [2]
        current = snapshots(following)[0]
        assert current["host_calls_used"] == 1 and current["remaining_rounds_including_this_response"] == 1
        assert all(not message.tool_calls and message.native_assistant is None and message.role != "tool" for message in following)
        assert any(part is image for message in following for part in message.parts)
        assert any(isinstance(part, TextPart) and part.text == result[0].text for message in following for part in message.parts)
        assert any(isinstance(part, TextPart) and part.text == call.arguments_json for message in following for part in message.parts)
        assert len(budget_events(runtime)) == 2
        assert not runtime._required_events
        assert any(entry.compaction_json and json.loads(entry.compaction_json).get("stage") == "restart"
                   for entry in runtime.history_snapshot)
    asyncio.run(run())


def test_actual_summary_never_receives_snapshots_or_consumes_current_decision_rounds():
    async def run():
        seen, executed, main_calls = [], [], 0
        async def model(req):
            nonlocal main_calls
            seen.append(req)
            if req.purpose == "summary":
                assert not snapshots(req.messages)
                return ModelCompletion("Earlier visible facts retained.")
            main_calls += 1
            current = snapshots(req.messages)[-1]
            if main_calls == 2:
                assert current["current_round"] == 1
                return ModelCompletion("", tool_calls=(host_call(1),))
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), cfg(max_context_characters=4500, recent_turns=1),
                                tools=registry(executed, rounds=2))
        await runtime.complete(request(1, "Old visible facts: " + "x" * 2100))
        await runtime.complete(request(2, "Current visible observation: " + "y" * 2100))
        assert any(req.purpose == "summary" for req in seen)
        current = [value for _, value in budget_events(runtime) if value["current_round"] == 2]
        assert len(current) == 1 and current[0]["remaining_rounds_including_this_response"] == 1
        assert runtime.usage.action_calls == 3 and runtime.usage.summary_calls >= 1
        assert executed == [1]
    asyncio.run(run())
