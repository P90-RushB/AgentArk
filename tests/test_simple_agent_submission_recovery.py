"""Bounded native action-format recovery; no model, SDK, or environment I/O."""
import asyncio
import json
from unittest.mock import patch

import pytest

from agent_ark.agent.simple_agent.agent import SimpleAgent, _TracedModel
from agent_ark.agent.simple_agent.models import ModelResponseError
from agent_ark.agent.simple_agent.runtime import _metrics
from agent_ark.agent.simple_agent.types import (
    AgentInputIdentity, AgentLimitError, AgentRequest, AgentReuseError, Message,
    ModelCompletion, ModelRequest, ModelUsage, NativeAssistantState, ToolCall,
)


USAGE = ModelUsage(5, 7, 12)


def invalid(number=0, **kwargs):
    return ModelCompletion("", USAGE, (ToolCall(f"bad-{number}", "submit_action",
        '{"format":"raw","content":"a","extra":1}'),), **kwargs)


def valid():
    return ModelCompletion("", USAGE, (ToolCall("good", "submit_action",
        '{"format":"raw","content":"a"}'),))


class SequenceModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        result = next(self.replies)
        if isinstance(result, BaseException):
            raise result
        return result


def make_agent(replies, *, context=None, limits=None):
    backend = SequenceModel(replies)
    agent = SimpleAgent(provider="openai", model_backend=backend, tool_setup=lambda registry: None,
        simple_agent_config={"context": context or {}, "tools": {"limits": limits or {}}})
    return agent, backend, agent._state(0)


def request():
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "0", 0, "input"),
                        (Message.text("user", "visible observation"),))


def test_repeated_rejections_exhaust_existing_rounds_and_cannot_restart_same_input():
    async def run():
        agent, backend, state = make_agent([invalid(i) for i in range(4)], limits={"max_model_rounds": 3})
        current = request()
        try:
            with pytest.raises(AgentLimitError, match="model tool round budget exhausted"):
                await state.runtime.complete(current)
            with pytest.raises(AgentReuseError):
                await state.runtime.complete(current)
            assert len(backend.requests) == state.runtime.usage.action_calls == 3
            assert state.runtime.usage.action_usage == (USAGE,) * 3
            assert [record["status"] for record in state.model.calls] == ["rejected"] * 3
            assert state.model.usages == [("action", USAGE)] * 3
            exchanges = [entry for entry in state.runtime.history_snapshot if entry.kind == "tool_exchange"]
            assert len(exchanges) == 3 and all(entry.usage == USAGE for entry in exchanges)
            assert not state.runtime._completed
            for exchange in exchanges:
                assistant, feedback = exchange.messages
                assert feedback.tool_call_id == assistant.tool_calls[0].call_id
                assert json.loads(feedback.parts[0].text)["batch_rejected"] is True
        finally:
            agent.close()
    asyncio.run(run())


@pytest.mark.parametrize("context,limits,expected,calls", [
    ({"max_model_calls": 2}, {}, "lifetime model call budget", 2),
    ({}, {"max_calls": 1}, "tool call/time budget", 2),
    ({}, {"max_text_bytes": 1}, "tool result budget", 1),
    ({}, {"max_total_text_bytes": 1}, "tool result budget", 1),
])
def test_recovery_charges_model_call_and_feedback_budgets(context, limits, expected, calls):
    async def run():
        agent, backend, state = make_agent([invalid(i) for i in range(4)], context=context, limits=limits)
        try:
            with pytest.raises(AgentLimitError, match=expected):
                await state.runtime.complete(request())
            assert len(backend.requests) == state.runtime.usage.action_calls == calls
            assert state.runtime.usage.action_usage == (USAGE,) * calls
            assert state.model.usages == [("action", USAGE)] * calls
        finally:
            agent.close()
    asyncio.run(run())


def test_entire_mixed_batch_is_charged_before_any_feedback_or_host_dispatch():
    async def run():
        bad = ModelCompletion("", USAGE, (ToolCall("host-first", "write_note", "{}"), *invalid().tool_calls))
        agent, backend, state = make_agent([bad, valid()], limits={"max_calls": 1})
        try:
            with patch.object(state.runtime.tools, "execute") as execute:
                with pytest.raises(AgentLimitError, match="tool call/time budget"):
                    await state.runtime.complete(request())
                execute.assert_not_called()
            assert len(backend.requests) == 1
            assert not any(entry.kind == "tool_exchange" for entry in state.runtime.history_snapshot)
        finally:
            agent.close()
    asyncio.run(run())


def test_recovery_context_limit_does_not_compact_or_issue_another_request():
    async def run():
        agent, backend, state = make_agent([invalid(), valid()])
        runtime = state.runtime
        first_cost = _metrics(request().messages + (runtime._tool_budget_message(),))[0]
        # Preserve all other bounds while fitting exactly the initial request.
        from dataclasses import replace
        runtime.config = replace(runtime.config, max_context_characters=first_cost)
        try:
            with patch.object(runtime, "_compact", wraps=runtime._compact) as compact:
                with pytest.raises(AgentLimitError, match="native tool continuation cannot fit"):
                    await runtime.complete(request())
                assert compact.call_count == 1
            assert len(backend.requests) == 1 and runtime.usage.summary_calls == 0
            assert len(runtime.history_snapshot[-1].messages) == 2
        finally:
            agent.close()
    asyncio.run(run())


def test_rejected_native_output_limit_stops_before_feedback_and_keeps_original_audit():
    async def run():
        calls = invalid().tool_calls
        original = {"role": "assistant", "content": None, "reasoning": "opaque" * 100,
            "tool_calls": [{"id": calls[0].call_id, "type": "function", "function": {
                "name": calls[0].name, "arguments": calls[0].arguments_json}}]}
        native = NativeAssistantState("test", json.dumps(original))
        bad = invalid(native_assistant=native)
        agent, backend, state = make_agent([bad, valid()], context={"max_action_output_characters": 200})
        try:
            with pytest.raises(AgentLimitError, match="action output exceeds its hard bounds"):
                await state.runtime.complete(request())
            assert len(backend.requests) == 1
            assert state.runtime.history_snapshot[-1].messages[0].native_assistant is native
            assert not any(entry.kind == "tool_exchange" for entry in state.runtime.history_snapshot)
            assert state.runtime.usage.action_usage == (USAGE,)
        finally:
            agent.close()
    asyncio.run(run())


def test_archive_limit_stops_recovery_without_issuing_another_request():
    async def run():
        agent, backend, state = make_agent([invalid(), valid()], context={"max_archive_entries": 1})
        try:
            with pytest.raises(AgentLimitError, match="immutable archive bound exhausted"):
                await state.runtime.complete(request())
            assert len(backend.requests) == 1
            assert [entry.kind for entry in state.runtime.history_snapshot] == ["input"]
        finally:
            agent.close()
    asyncio.run(run())


@pytest.mark.parametrize("error", [ModelResponseError("provider refused"), RuntimeError("network failed"),
                                   ValueError("programming error"), asyncio.CancelledError()])
def test_provider_and_program_errors_do_not_enter_submission_recovery(error):
    async def run():
        agent, backend, state = make_agent([error, valid()])
        try:
            with pytest.raises(type(error)) as caught:
                await state.runtime.complete(request())
            assert caught.value is error
            assert len(backend.requests) == 1
            assert not any(entry.kind == "tool_exchange" for entry in state.runtime.history_snapshot)
        finally:
            agent.close()
    asyncio.run(run())


def test_original_deadline_prevents_recovery_after_provider_swallows_cancellation():
    async def run():
        agent, backend, state = make_agent([], context={"timeout_seconds": 0.01})
        async def slow(request):
            backend.requests.append(request)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                return invalid()
        backend.complete = slow
        try:
            with pytest.raises(AgentLimitError, match="decision time budget exhausted"):
                await state.runtime.complete(request())
            assert len(backend.requests) == 1
            assert not any(entry.kind == "tool_exchange" for entry in state.runtime.history_snapshot)
        finally:
            agent.close()
    asyncio.run(run())


def test_legacy_transport_does_not_interpret_or_recover_native_submission():
    async def run():
        bad = invalid()
        backend = SequenceModel([bad])
        traced = _TracedModel(backend, native_submission=False)
        assert await traced.complete(ModelRequest((Message.text("user", "observation"),))) is bad
        assert len(backend.requests) == 1 and traced.calls[0]["status"] == "completed"
    asyncio.run(run())
