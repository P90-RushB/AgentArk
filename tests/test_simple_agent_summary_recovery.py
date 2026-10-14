"""Bounded summary recovery, using injected clients and in-memory models only."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent.agent import SimpleAgent
from agent_ark.agent.simple_agent.encoding import chat_messages, encoded_messages
from agent_ark.agent.simple_agent.models import CallableModel, ChatCompletionsModel, ModelResponseError, SummaryResponseError
from agent_ark.agent.simple_agent.runtime import SimpleRuntime, SimpleRuntimeConfig
from agent_ark.agent.simple_agent.types import (
    AgentInputIdentity, AgentLimitError, AgentRequest, Message, ModelCompletion, ModelRequest, ModelUsage, TextPart,
)


ACTION = '<tool_call>{"name":"tap","arguments":{"x":3}}</tool_call>'
SUMMARY = "Useful facts retained."
THOUGHT = "SUMMARY_THOUGHT_MUST_NOT_BE_REPLAYED"
BAD_BODY = "FAILED_SUMMARY_BODY_MUST_NOT_BE_REPLAYED"
USAGE = {"prompt_tokens": 21, "completion_tokens": 39, "total_tokens": 60,
         "completion_tokens_details": {"reasoning_tokens": 31}}
CONTEXT = {"max_context_characters": 1000, "max_context_tokens": 10000,
           "output_reserve_tokens": 64, "recent_turns": 1, "summary_output_tokens": 256}


def reply(content=SUMMARY, *, finish="stop", **fields):
    return {"choices": [{"finish_reason": finish,
                         "message": {"content": content, "reasoning": THOUGHT, **fields}}], "usage": USAGE}


def wire_text(messages):
    return "\n".join(
        content if isinstance(content := message.get("content"), str)
        else "\n".join(part["text"] for part in content or [] if part.get("type") == "text")
        for message in messages)


class FakeClient:
    def __init__(self, summaries):
        self.summaries = iter(summaries)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if wire_text(kwargs["messages"]).startswith("Summarize this agent's earlier complete conversation turns."):
            result = next(self.summaries)
            if isinstance(result, BaseException):
                raise result
            return result
        return reply(ACTION, reasoning="Use the visible tap.")

    async def close(self):
        pass


def make_agent(summaries, **context):
    agent = SimpleAgent(api_key="unused-fake-key", provider="openai", simple_agent_config={
        "context": {**CONTEXT, **context}})
    client = FakeClient(summaries)
    agent._backend.client = client
    return agent, client


def request(number, text=None):
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "0", 0, str(number)),
                        (Message.text("user", text if text is not None else str(number) + "x" * 399),))


def accepted_three(runtime):
    for number in range(1, 4):
        runtime.accept_input(request(number))
    return request(3)


@pytest.mark.parametrize("first,reason", [(reply(None), "no summary content"),
                                         (reply(BAD_BODY, finish="length"), "output token limit")])
def test_provider_summary_failure_recovers_in_harness_and_retains_all_billed_usage(first, reason):
    agent, client = make_agent([first, reply()], max_context_characters=2100)
    try:
        trace = None
        for turn in range(7):
            result, traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content":
                "Current observation (step_msg):\nobservation " + str(turn) + " " + "x" * 480}]}})
            assert result[0]["action"] == ACTION
            if traces[0]["agent_usage"]["summary_calls"]:
                trace = traces[0]
                break
        assert trace is not None
        records = trace["model_requests"]
        assert [record["purpose"] for record in records] == ["summary", "summary", "action"]
        assert [record["status"] for record in records] == ["failed", "completed", "completed"]
        assert records[0]["error_type"] == "SummaryResponseError"
        retry = wire_text(records[1]["request_messages"])
        assert "Summary retry instructions:" in retry and reason in retry
        assert THOUGHT not in retry and BAD_BODY not in retry
        action_context = wire_text(client.requests[-1]["messages"])
        assert SUMMARY in action_context
        assert "Summary retry instructions:" not in action_context
        assert THOUGHT not in action_context and BAD_BODY not in action_context
        assert [event["text"] for event in trace["agent_events"] if event["kind"] == "summary_completion"] == [SUMMARY]
        assert all(record["response_metadata"]["usage"] == USAGE for record in records)
        assert all(THOUGHT in record["response_metadata"]["assistant_raw"] for record in records[:2])
        assert trace["agent_usage"]["summary"] == {"input_tokens": 42, "output_tokens": 78, "total_tokens": 120}
        assert trace["usage"] == {"input_tokens": 63, "output_tokens": 117, "total_tokens": 180}
        assert agent._states[0].runtime.usage.summary_usage == (ModelUsage(21, 39, 60),) * 2
        assert records[0]["max_output_tokens"] == records[1]["max_output_tokens"] == 256
    finally:
        agent.close()


def test_four_failed_attempts_exhaust_existing_budget_without_action_or_working_changes():
    async def run():
        agent, client = make_agent([reply(None)] * 4)
        state = agent._state(0)
        current = accepted_three(state.runtime)
        before = state.runtime.working_snapshot
        archive = state.runtime.history_snapshot
        try:
            for _ in range(2):
                with pytest.raises(AgentLimitError, match="summary call bound exhausted"):
                    await state.runtime.complete(current)
            assert len(client.requests) == state.runtime.usage.summary_calls == 4
            assert state.runtime.usage.action_calls == 0
            assert state.runtime.working_snapshot == before
            assert state.runtime.history_snapshot == archive
            assert state.runtime.usage.summary_usage == (ModelUsage(21, 39, 60),) * 4
            assert len(state.model.usages) == 4
            assert sum(usage.total_tokens for _, usage in state.model.usages) == 240
            assert all(record["status"] == "failed" and record["response_metadata"]["usage"] == USAGE
                       for record in state.model.calls)
            assert all(record["response_metadata"]["assistant_raw"] == THOUGHT for record in state.model.calls)
            assert all(THOUGHT not in wire_text(record["request_messages"]) for record in state.model.calls)
        finally:
            await agent._backend.close()
            agent.close()

    asyncio.run(run())


@pytest.mark.parametrize("bad,limits,reason", [
    (" \n\t ", {}, "no summary content"),
    (BAD_BODY * 5, {"summary_output_characters": 32}, "exceeded its output bounds"),
    ("界" * 20, {"summary_output_bytes": 32}, "exceeded its output bounds"),
    (BAD_BODY * 4, {"summary_output_tokens": 16}, "exceeded its output bounds"),
    (BAD_BODY * 30, {"summary_output_tokens": 1024}, "did not reduce the source records"),
])
def test_runtime_invalid_summary_text_is_audited_and_retried_without_replaying_it(bad, limits, reason):
    async def run():
        seen = []
        replies = iter((bad, "Facts."))

        async def model(value):
            seen.append(value)
            return ModelCompletion(next(replies) if value.purpose == "summary" else ACTION, ModelUsage(2, 3, 5))

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(**{**CONTEXT, **limits}))
        current = accepted_three(runtime)
        await runtime.complete(current)
        assert [value.purpose for value in seen] == ["summary", "summary", "action"]
        retry = wire_text(chat_messages(seen[1].messages))
        assert reason in retry
        if bad.strip():
            assert bad.strip() not in retry
        assert [entry.text for entry in runtime.history_snapshot if entry.kind == "summary_completion"] == [bad, "Facts."]
        working = "\n".join(part.text for message in runtime.working_snapshot for part in message.parts if isinstance(part, TextPart))
        assert "Facts." in working and "Summary retry instructions:" not in working
        assert BAD_BODY not in working
        assert runtime.usage.summary_calls == 2 and runtime.usage.action_calls == 1
        assert runtime.usage.summary_usage == (ModelUsage(2, 3, 5),) * 2

    asyncio.run(run())


@pytest.mark.parametrize("bound", ["characters", "bytes", "tokens"])
def test_retry_feedback_is_budgeted_and_shrinks_only_complete_source_prefixes(bound):
    async def run():
        source = [request(1, "FIRST_OLD_RECORD " * 40), request(2, "SECOND_OLD_RECORD " * 40),
                  request(3, "CURRENT_VISIBLE_RECORD " * 10)]
        probe = SimpleRuntime(CallableModel(lambda _: None), SimpleRuntimeConfig(**CONTEXT))
        for value in source:
            probe.accept_input(value)
        initial = probe._summary_messages(tuple(probe._working[:2]))
        budget = {"characters": sum(len(part.text) for message in initial for part in message.parts),
                  "bytes": len(encoded_messages(initial)), "tokens": probe._tokens(initial, len(encoded_messages(initial)))}[bound]
        agent, client = make_agent([reply(None), reply()], max_context_characters=1200,
                                   **{"summary_input_" + bound: budget})
        runtime = agent._state(0).runtime
        for value in source:
            runtime.accept_input(value)
        try:
            await runtime.complete(source[-1])
            summaries = agent._states[0].model.calls[:2]
            assert "FIRST_OLD_RECORD" in wire_text(summaries[0]["request_messages"])
            assert "SECOND_OLD_RECORD" in wire_text(summaries[0]["request_messages"])
            assert "FIRST_OLD_RECORD" in wire_text(summaries[1]["request_messages"])
            assert "SECOND_OLD_RECORD" not in wire_text(summaries[1]["request_messages"])
            for record in summaries:
                messages = record["request_messages"]
                measured = {"characters": sum(len(part["text"]) for message in messages for part in message["content"] or []),
                            "bytes": len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()),
                            "tokens": record["input_tokens"]}[bound]
                assert measured <= budget
            action = wire_text(client.requests[-1]["messages"])
            assert SUMMARY in action and "SECOND_OLD_RECORD" in action and "CURRENT_VISIBLE_RECORD" in action
            assert "Summary retry instructions:" not in action
            assert runtime.usage.summary_calls == 2
        finally:
            await agent._backend.close()
            agent.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", [
    PermissionError("invalid API credentials"), RuntimeError("provider HTTP failure"),
    ValueError("programming defect"), asyncio.CancelledError(), AgentLimitError("decision time budget exhausted"),
    reply(refusal="Refused"), reply(finish="length", refusal="Refused"),
    reply(finish="length", tool_calls=[{"id": "call", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]),
    {**reply(finish="length"), "usage": {"completion_tokens": -1}},
])
def test_unrecoverable_provider_or_program_errors_are_never_retried(failure):
    async def run():
        agent, client = make_agent([failure, reply()])
        runtime = agent._state(0).runtime
        current = accepted_three(runtime)
        before = runtime.working_snapshot
        expected = type(failure) if isinstance(failure, BaseException) else ModelResponseError
        try:
            with pytest.raises(expected) as caught:
                await runtime.complete(current)
            if isinstance(failure, BaseException):
                assert caught.value is failure
            assert len(client.requests) == runtime.usage.summary_calls == 1
            assert runtime.usage.action_calls == 0 and runtime.working_snapshot == before
        finally:
            await agent._backend.close()
            agent.close()

    asyncio.run(run())


def test_archive_capacity_failure_stops_without_retrying_completed_summary():
    async def run():
        agent, client = make_agent([reply(BAD_BODY * 20), reply()], max_archive_entries=3)
        runtime = agent._state(0).runtime
        current = accepted_three(runtime)
        before = runtime.working_snapshot
        try:
            with pytest.raises(AgentLimitError, match="immutable archive bound exhausted"):
                await runtime.complete(current)
            assert len(client.requests) == runtime.usage.summary_calls == 1
            assert runtime.working_snapshot == before and runtime.usage.action_calls == 0
        finally:
            await agent._backend.close()
            agent.close()

    asyncio.run(run())


def test_lifetime_model_budget_stops_recovery_before_an_extra_provider_call():
    async def run():
        agent, client = make_agent([reply(None)] * 4, max_model_calls=2)
        runtime = agent._state(0).runtime
        current = accepted_three(runtime)
        before = runtime.working_snapshot
        try:
            with pytest.raises(AgentLimitError, match="lifetime model call budget exhausted"):
                await runtime.complete(current)
            assert len(client.requests) == runtime.usage.summary_calls == 2
            assert runtime.usage.action_calls == 0 and runtime.working_snapshot == before
        finally:
            await agent._backend.close()
            agent.close()

    asyncio.run(run())


def test_elapsed_deadline_cannot_be_swallowed_by_a_recoverable_provider_response():
    async def run():
        agent, client = make_agent([], timeout_seconds=0.01)

        async def slow_create(**kwargs):
            client.requests.append(kwargs)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                return reply(None)

        client.chat.completions.create = slow_create
        runtime = agent._state(0).runtime
        current = accepted_three(runtime)
        before = runtime.working_snapshot
        try:
            with pytest.raises(AgentLimitError, match="decision time budget exhausted"):
                await runtime.complete(current)
            assert len(client.requests) == runtime.usage.summary_calls == 1
            assert runtime.usage.action_calls == 0 and runtime.working_snapshot == before
        finally:
            await agent._backend.close()
            agent.close()

    asyncio.run(run())


@pytest.mark.parametrize("first", [reply(None), reply(BAD_BODY, finish="length")])
def test_generic_chat_adapter_exposes_only_typed_summary_generation_failures(first):
    async def run():
        client = FakeClient([first])
        adapter = ChatCompletionsModel(client, "fake-model")
        with pytest.raises(SummaryResponseError) as caught:
            await adapter.complete(ModelRequest((Message.text("system", "Summarize this agent's earlier complete conversation turns."),), purpose="summary"))
        assert caught.value.usage == ModelUsage(21, 39, 60)

    asyncio.run(run())


@pytest.mark.parametrize("first", [reply(None), reply(BAD_BODY, finish="length")])
def test_generic_backend_summary_recovery_trace_keeps_usage_from_failed_typed_response(first):
    client = FakeClient([first, reply()])
    agent = SimpleAgent(provider="openai", model_backend=ChatCompletionsModel(client, "fake-model"),
                        simple_agent_config={"context": {**CONTEXT, "max_context_characters": 2100}})
    try:
        trace = None
        for turn in range(7):
            result, traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content":
                "Current observation (step_msg):\nobservation " + str(turn) + " " + "x" * 480}]}})
            assert result[0]["action"] == ACTION
            if traces[0]["agent_usage"]["summary_calls"]:
                trace = traces[0]
                break
        assert trace is not None
        assert [record["status"] for record in trace["model_requests"]] == ["failed", "completed", "completed"]
        assert all(record["usage"] == {"input_tokens": 21, "output_tokens": 39, "total_tokens": 60}
                   for record in trace["model_requests"])
        assert trace["agent_usage"]["summary"] == {"input_tokens": 42, "output_tokens": 78, "total_tokens": 120}
        assert trace["usage"] == {"input_tokens": 63, "output_tokens": 117, "total_tokens": 180}
        action_context = wire_text(client.requests[-1]["messages"])
        assert SUMMARY in action_context and THOUGHT not in action_context
        assert BAD_BODY not in action_context and "Summary retry instructions:" not in action_context
    finally:
        agent.close()
