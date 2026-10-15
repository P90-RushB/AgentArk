"""Offline provider-response regressions for SimpleAgent's summary memory."""
import asyncio
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent.agent import SimpleAgent, _TracedModel
from agent_ark.agent.simple_agent.models import ModelResponseError
from agent_ark.agent.simple_agent.types import (
    AgentInputIdentity, AgentLimitError, AgentRequest, Message, ModelRequest, ModelUsage, TextPart,
)


ACTION = '<tool_call>{"name":"tap","arguments":{"x":3}}</tool_call>'
SUMMARY = "Earlier trials selected the left path."
SUMMARY_REASONING = ("SUMMARY_REASONING_MUST_NOT_BECOME_MEMORY " + "analysis " * 200).rstrip()
ACTION_USAGE = {
    "prompt_tokens": 11, "completion_tokens": 19, "total_tokens": 30,
    "completion_tokens_details": {"reasoning_tokens": 15},
}
SUMMARY_USAGE = {
    "prompt_tokens": 21, "completion_tokens": 39, "total_tokens": 60,
    "completion_tokens_details": {"reasoning_tokens": 31},
}


class FakeChatClient:
    """Injected before any request, so no SDK HTTP client can be constructed."""

    def __init__(self, respond):
        self.respond = respond
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.respond(kwargs)

    async def close(self):
        pass


def response(message, *, finish_reason="stop", usage=SUMMARY_USAGE):
    return {"choices": [{"finish_reason": finish_reason, "message": message}], "usage": usage}


def fake_agent(respond, **kwargs):
    agent = SimpleAgent(api_key="unused-fake-key", provider="openai", **kwargs)
    client = FakeChatClient(respond)
    agent._backend.client = client
    return agent, client


def wire_text(messages):
    return "\n".join(
        content if isinstance(content := message.get("content"), str)
        else "\n".join(part["text"] for part in content or [] if part.get("type") == "text")
        for message in messages
    )


def is_summary(kwargs):
    return wire_text(kwargs["messages"]).startswith("Summarize this agent's earlier complete conversation turns.")


@pytest.mark.parametrize("reasoning_fields", [
    {"reasoning": SUMMARY_REASONING},
    {"reasoning_content": SUMMARY_REASONING},
    {"model_extra": {"reasoning": SUMMARY_REASONING}},
    {"model_extra": {"reasoning_content": SUMMARY_REASONING}},
])
def test_summary_completion_uses_only_coerced_content_but_trace_keeps_reasoning(reasoning_fields):
    async def run():
        agent, client = fake_agent(lambda _: response({
            "content": [{"type": "text", "text": "  Earlier trials "}, {"text": "selected the left path.  "}],
            **reasoning_fields,
        }))
        traced = _TracedModel(agent._backend)
        try:
            completion = await traced.complete(ModelRequest((Message.text("user", "old records"),), purpose="summary"))
            assert completion.text == "Earlier trials\nselected the left path."
            assert completion.native_assistant is None
            assert completion.usage == ModelUsage(21, 39, 60)
            record = traced.calls[0]
            assert record["completion"] == completion.text
            assert record["response_metadata"]["assistant_raw"] == (
                "<think>\n" + SUMMARY_REASONING + "\n</think>\n" + completion.text)
            assert record["response_metadata"]["usage"] == SUMMARY_USAGE
            assert record["usage"] == {"input_tokens": 21, "output_tokens": 39, "total_tokens": 60}
            assert len(client.requests) == 1
        finally:
            await agent._backend.close()
            agent.close()

    asyncio.run(run())


def test_summary_preserves_literal_think_tags_in_content():
    async def run():
        body = "The observed label is <think>literal text</think>; preserve it."
        agent, _ = fake_agent(lambda _: response({"content": body}))
        try:
            completion = await agent._backend.complete(ModelRequest((Message.text("user", "records"),), purpose="summary"))
            assert completion.text == body
        finally:
            await agent._backend.close()
            agent.close()

    asyncio.run(run())


@pytest.mark.parametrize("content,reasoning_fields", [
    (None, {"reasoning": SUMMARY_REASONING}),
    ("", {"model_extra": {"reasoning_content": SUMMARY_REASONING}}),
    (" \n\t ", {"reasoning_content": SUMMARY_REASONING}),
    ([], {"model_extra": {"reasoning": SUMMARY_REASONING}}),
])
def test_reasoning_only_summary_rejects_without_changing_working_context_and_keeps_audit(content, reasoning_fields):
    async def run():
        agent, client = fake_agent(lambda kwargs: (
            response({"content": content, **reasoning_fields}) if is_summary(kwargs)
            else response({"content": ACTION}, usage=ACTION_USAGE)
        ), simple_agent_config={"context": {
            "max_context_characters": 1200, "max_context_tokens": 10000,
            "output_reserve_tokens": 64, "recent_turns": 1, "summary_output_tokens": 256,
            "max_summary_calls": 1,
        }})
        state = agent._state(0)
        runtime = state.runtime

        def request(turn):
            identity = AgentInputIdentity("run", "session", "episode", "0", 0, "turn-" + str(turn))
            return AgentRequest(identity, (Message.text("user", str(turn) + "x" * 399),))

        try:
            await runtime.complete(request(1))
            await runtime.complete(request(2))
            third = request(3)
            runtime.accept_input(third)
            before_context, before_archive = runtime.working_snapshot, runtime.history_snapshot
            with pytest.raises(AgentLimitError, match="summary call bound exhausted: chat client returned empty summary content"):
                await runtime.complete(third)
            assert runtime.working_snapshot == before_context
            assert runtime.history_snapshot == before_archive
            assert runtime.usage.action_calls == 2
            assert runtime.usage.summary_calls == 1
            assert len(client.requests) == 3
            record = state.model.calls[-1]
            assert record["purpose"] == "summary" and record["status"] == "failed"
            assert record["error_type"] == "SummaryResponseError"
            assert record["response_metadata"]["assistant_raw"] == SUMMARY_REASONING
            assert record["response_metadata"]["usage"] == SUMMARY_USAGE
            assert record["usage"] == {"input_tokens": 21, "output_tokens": 39, "total_tokens": 60}
            assert state.model.usages[-1] == ("summary", ModelUsage(21, 39, 60))
            assert sum(usage.total_tokens for _, usage in state.model.usages) == 120
        finally:
            await agent._backend.close()
            agent.close()

    asyncio.run(run())


def test_compacted_body_reaches_actual_later_action_requests_without_summary_reasoning():
    agent, client = fake_agent(lambda kwargs: (
        response({"content": SUMMARY, "model_extra": {"reasoning_content": SUMMARY_REASONING}})
        if is_summary(kwargs) else response({"content": ACTION, "reasoning": "Choose the visible tap."}, usage=ACTION_USAGE)
    ), simple_agent_config={"context": {
        "max_context_characters": 2100, "max_context_tokens": 10000,
        "output_reserve_tokens": 64, "recent_turns": 1,
        "summary_output_tokens": 256, "summary_output_characters": 512,
    }})
    try:
        found = None
        for turn in range(7):
            value = {0: {"messages": [{"role": "user", "content":
                "Current observation (step_msg):\nobservation " + str(turn) + " " + "x" * 480}]}}
            result, traces = agent.forward_with_trace(value)
            assert result[0]["action"] == ACTION
            if traces[0]["agent_usage"]["summary_calls"]:
                found = traces[0]
                break
        assert found is not None
        assert [record["purpose"] for record in found["model_requests"]] == ["summary", "action"]
        assert found["model_requests"][0]["completion"] == SUMMARY
        assert SUMMARY_REASONING in found["model_requests"][0]["response_metadata"]["assistant_raw"]
        assert found["model_requests"][0]["response_metadata"]["usage"] == SUMMARY_USAGE
        assert found["agent_usage"]["summary"] == {"input_tokens": 21, "output_tokens": 39, "total_tokens": 60}
        assert found["usage"] == {"input_tokens": 32, "output_tokens": 58, "total_tokens": 90}
        assert [event["text"] for event in found["agent_events"] if event["kind"] == "summary_completion"] == [SUMMARY]
        assert SUMMARY in wire_text(client.requests[-1]["messages"])
        assert "SUMMARY_REASONING_MUST_NOT_BECOME_MEMORY" not in wire_text(client.requests[-1]["messages"])
        agent.forward({0: {"messages": [{"role": "user", "content": "A later visible result."}]}})
        assert SUMMARY in wire_text(client.requests[-1]["messages"])
        assert "SUMMARY_REASONING_MUST_NOT_BECOME_MEMORY" not in wire_text(client.requests[-1]["messages"])
        assert all(entry.text == SUMMARY for entry in agent._states[0].runtime.history_snapshot
                   if entry.kind == "summary_completion")
        assert "SUMMARY_REASONING_MUST_NOT_BECOME_MEMORY" not in "\n".join(
            part.text for message in agent._states[0].runtime.working_snapshot
            for part in message.parts if isinstance(part, TextPart))
    finally:
        agent.close()


@pytest.mark.parametrize("message,finish_reason,error", [
    ({"content": SUMMARY}, "length", "not complete"),
    ({"content": SUMMARY, "refusal": "refused"}, "stop", "refused completion"),
    ({"content": SUMMARY, "tool_calls": None}, "tool_calls", "no tool calls"),
    ({"content": SUMMARY, "tool_calls": [{"id": "call", "type": "function",
      "function": {"name": "unavailable", "arguments": "{}"}}]}, "tool_calls", "unavailable tools"),
    ({"content": SUMMARY, "tool_calls": "invalid"}, "stop", "invalid tool calls"),
])
def test_summary_still_rejects_incomplete_refused_or_tool_responses(message, finish_reason, error):
    async def run():
        agent, _ = fake_agent(lambda _: response({"reasoning": SUMMARY_REASONING, **message}, finish_reason=finish_reason))
        traced = _TracedModel(agent._backend)
        try:
            with pytest.raises(ModelResponseError, match=error):
                await traced.complete(ModelRequest((Message.text("user", "old records"),), purpose="summary"))
            record = traced.calls[0]
            assert record["status"] == "failed"
            assert record["response_metadata"]["usage"] == SUMMARY_USAGE
            assert SUMMARY_REASONING in record["response_metadata"]["assistant_raw"]
        finally:
            await agent._backend.close()
            agent.close()

    asyncio.run(run())


@pytest.mark.parametrize("content,reasoning,expected_action", [
    (ACTION, "Choose the visible tap.", ACTION),
    ("<code>public sealed class AgentAction : CodeAction {}</code>", "Use the task API.",
     "public sealed class AgentAction : CodeAction {}"),
])
def test_action_responses_extract_only_content_and_keep_full_usage(content, reasoning, expected_action):
    agent, client = fake_agent(lambda _: response({"content": content, "reasoning": reasoning}, usage=ACTION_USAGE))
    try:
        result, traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "visible state"}]}})
        raw = "<think>\n" + reasoning + "\n</think>\n" + content if content else reasoning
        assert result[0] == {"action": expected_action, "assistant": content}
        record = traces[0]["model_requests"][0]
        assert record["completion"] == content
        assert record["response_metadata"]["assistant_raw"] == raw
        assert record["response_metadata"]["usage"] == ACTION_USAGE
        assert traces[0]["usage"] == {"input_tokens": 11, "output_tokens": 19, "total_tokens": 30}
        assert len(client.requests) == 1
    finally:
        agent.close()
