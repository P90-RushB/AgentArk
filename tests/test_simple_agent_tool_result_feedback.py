"""Offline result-budget recovery; no HTTP, Player, SDK, or environment I/O."""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_ark.agent.simple_agent import (
    AgentInputIdentity, AgentLimitError, AgentRequest, AgentReuseError,
    CallableModel, ImagePart, Message, ModelCompletion, NativeAssistantState,
    RecordSnapshot, SimpleRuntime, TextPart, ToolCall, ToolLimits, ToolRegistry,
    ToolSpec, encode_png,
)
from agent_ark.agent.simple_agent.agent import SimpleAgent
from agent_ark.agent.simple_agent.runtime import _metrics


ERROR = "Tool result omitted: result budget exhausted. Operation may already have run; do not retry blindly."
MINIMUM_FEEDBACK = json.dumps({"error": ERROR}, separators=(",", ":"))
MINIMUM_BYTES = len(MINIMUM_FEEDBACK.encode("utf-8"))
SECRET = "PRIVATE_RESULT_CONTENT"


def request():
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "agent", 0, "input"),
                        (Message.text("user", "Visible synthetic observation."),))


def call(number=1):
    return ToolCall(str(number), "inspect", "{}")


def registry(handler, limits=None, *, fresh=False):
    tools = ToolRegistry(limits=limits)
    tools.register(ToolSpec("inspect", "Read synthetic visible information.", '{"type":"object"}'), handler)
    if fresh:
        return tools, None
    tools.accept_input(request())
    return tools, tools.begin(request())


def text_size(parts):
    return sum(len(part.text.encode("utf-8")) for part in parts if isinstance(part, TextPart))


def assert_omitted(result):
    assert not result.success
    assert len(result.parts) == 1 and isinstance(result.parts[0], TextPart)
    feedback = json.loads(result.parts[0].text)
    assert feedback["error"] == ERROR
    assert SECRET not in result.parts[0].text
    assert "private/path" not in result.parts[0].text
    assert "executed" not in feedback
    return feedback


@pytest.mark.parametrize("limits,initial,oversized,following", [
    (ToolLimits(max_text_bytes=256), (),
     (TextPart(SECRET * 30), ImagePart(b"image")), (TextPart("ok"),)),
    # Character count fits; its UTF-8 byte count exceeds the result limit.
    (ToolLimits(max_text_bytes=256), (), (TextPart("\u4e09" * 86),), (TextPart("ok"),)),
    (ToolLimits(max_text_bytes=512, max_total_text_bytes=530), (TextPart("x" * 300),),
     (TextPart(SECRET * 15), ImagePart(b"image")), (TextPart("ok"),)),
    (ToolLimits(max_image_bytes=3), (),
     (TextPart(SECRET), ImagePart(b"large")), (ImagePart(b"ok"),)),
    (ToolLimits(max_total_image_bytes=3), (ImagePart(b"12"),),
     (TextPart(SECRET), ImagePart(b"34")), (ImagePart(b"5"),)),
    (ToolLimits(max_images=1), (ImagePart(b"first"),),
     (TextPart(SECRET + " private/path"), ImagePart(b"second")), (TextPart("ok"),)),
], ids=["text", "utf8-text", "total-text", "image-bytes", "total-image-bytes", "image-count"])
def test_oversized_result_is_wholly_omitted_and_only_feedback_is_charged(limits, initial, oversized, following):
    outputs = iter(([initial] if initial else []) + [oversized, following])
    tools, session = registry(lambda *_: next(outputs), limits)
    if initial:
        assert tools.execute(call(), session).parts == initial
    before = session.text_bytes, session.images, session.image_bytes, session.calls
    denied = tools.execute(call(session.calls + 1), session)
    feedback = assert_omitted(denied)
    assert denied.call_id == str(before[3] + 1)
    assert session.text_bytes == before[0] + text_size(denied.parts)
    assert (session.images, session.image_bytes, session.calls) == (before[1], before[2], before[3] + 1)
    assert text_size(denied.parts) <= limits.max_text_bytes
    assert session.text_bytes <= limits.max_total_text_bytes
    if "remaining_images" in feedback:
        assert feedback["remaining_calls"] == limits.max_calls - session.calls
        assert feedback["remaining_images"] == limits.max_images - before[1]
    if "remaining_image_bytes" in feedback:
        assert feedback["remaining_image_bytes"] == limits.max_total_image_bytes - before[2]
        assert feedback["max_image_bytes"] == limits.max_image_bytes
    delivered = tools.execute(call(session.calls + 1), session)
    assert delivered.success and delivered.parts == following
    assert session.text_bytes == before[0] + text_size(denied.parts) + text_size(following)
    assert session.images == before[1] + sum(isinstance(part, ImagePart) for part in following)
    assert session.image_bytes == before[2] + sum(len(part.data) for part in following if isinstance(part, ImagePart))


@pytest.mark.parametrize("field", ["max_text_bytes", "max_total_text_bytes"])
def test_minimum_feedback_fits_exactly_without_optional_limit_fields(field):
    limits = ToolLimits(max_image_bytes=1, **{field: MINIMUM_BYTES})
    tools, session = registry(lambda *_: (ImagePart(b"too large"),), limits)
    result = tools.execute(call(), session)
    assert_omitted(result)
    assert result.parts == (TextPart(MINIMUM_FEEDBACK),)
    assert (session.calls, session.text_bytes, session.images, session.image_bytes) == (1, MINIMUM_BYTES, 0, 0)


def test_call_and_image_headroom_is_retained_when_optional_byte_details_do_not_fit():
    expected = json.dumps({"error": ERROR, "remaining_calls": 31, "remaining_images": 4}, separators=(",", ":"))
    tools, session = registry(lambda *_: (ImagePart(b"too large"),),
                              ToolLimits(max_image_bytes=1, max_text_bytes=len(expected.encode("utf-8"))))
    result = tools.execute(call(), session)
    assert_omitted(result)
    assert result.parts == (TextPart(expected),)
    assert session.text_bytes == len(expected.encode("utf-8"))


@pytest.mark.parametrize("field", ["max_text_bytes", "max_total_text_bytes"])
def test_insufficient_room_for_minimum_feedback_remains_a_hard_limit(field):
    invoked = []
    tools, session = registry(lambda *_: invoked.append(True) or (ImagePart(b"too large"),),
                              ToolLimits(max_image_bytes=1, **{field: MINIMUM_BYTES - 1}))
    with pytest.raises(AgentLimitError, match="tool result budget exhausted"):
        tools.execute(call(), session)
    assert invoked == [True]
    assert (session.calls, session.text_bytes, session.images, session.image_bytes) == (1, 0, 0, 0)


def test_total_feedback_room_uses_bytes_already_delivered_in_this_session():
    outputs = iter(((TextPart("abc"),), (ImagePart(b"too large"),)))
    tools, session = registry(lambda *_: next(outputs),
                              ToolLimits(max_image_bytes=1, max_total_text_bytes=MINIMUM_BYTES + 2))
    assert tools.execute(call(), session).success
    with pytest.raises(AgentLimitError, match="tool result budget exhausted"):
        tools.execute(call(2), session)
    assert (session.calls, session.text_bytes, session.images, session.image_bytes) == (2, 3, 0, 0)


def test_successful_exact_bound_results_preserve_parts_and_accounting():
    parts = (TextPart("\u4e09x"), ImagePart(b"12"), ImagePart(b"34"))
    limits = ToolLimits(max_text_bytes=4, max_total_text_bytes=4, max_images=2,
                        max_image_bytes=2, max_total_image_bytes=4)
    tools, session = registry(lambda *_: parts, limits)
    result = tools.execute(call(), session)
    assert result.success and result.parts is parts
    assert (session.calls, session.text_bytes, session.images, session.image_bytes) == (1, 4, 2, 4)


def test_handler_side_effect_runs_once_and_feedback_never_claims_it_was_rolled_back(tmp_path):
    invocations = []
    target = tmp_path / "effect.txt"

    def operation(*_):
        invocations.append(True)
        target.write_text("committed", encoding="utf-8")
        return (TextPart(SECRET + " private/path " + "x" * 300), ImagePart(b"PRIVATE_IMAGE_PAYLOAD"))

    tools, session = registry(operation, ToolLimits(max_text_bytes=256))
    result = tools.execute(call(), session)
    assert_omitted(result)
    assert str(target) not in result.parts[0].text
    assert "PRIVATE_IMAGE_PAYLOAD" not in result.parts[0].text
    assert target.read_text(encoding="utf-8") == "committed"
    with pytest.raises(AgentReuseError, match="already executed"):
        tools.execute(call(), session)
    assert invocations == [True] and session.calls == 1


def test_result_feedback_does_not_extend_call_or_time_limits(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("agent_ark.agent.simple_agent.tools.time.monotonic", lambda: now[0])
    invoked = []

    def operation(*_):
        invoked.append(True)
        return (ImagePart(b"oversized"),)

    tools, session = registry(operation, ToolLimits(max_calls=2, max_image_bytes=1, timeout_seconds=1))
    assert_omitted(tools.execute(call(), session))
    assert_omitted(tools.execute(call(2), session))
    with pytest.raises(AgentLimitError, match="tool call/time budget exhausted"):
        tools.execute(call(3), session)
    assert invoked == [True, True]

    tools, session = registry(operation, ToolLimits(max_image_bytes=1, timeout_seconds=1))
    now[0] += 2
    with pytest.raises(AgentLimitError, match="tool call/time budget exhausted"):
        tools.execute(call(), session)
    assert session.calls == 0 and invoked == [True, True]


def test_deadline_after_oversized_operation_still_stops_without_retry(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("agent_ark.agent.simple_agent.tools.time.monotonic", lambda: now[0])
    invoked = []

    def operation(*_):
        invoked.append(True)
        now[0] += 2
        return (ImagePart(b"oversized"),)

    tools, session = registry(operation, ToolLimits(max_image_bytes=1, timeout_seconds=1))
    with pytest.raises(AgentLimitError, match="tool time budget exhausted after operation"):
        tools.execute(call(), session)
    with pytest.raises(AgentLimitError, match="tool call/time budget exhausted"):
        tools.execute(call(2), session)
    assert invoked == [True] and session.calls == 1
    assert session.images == session.image_bytes == 0


@pytest.mark.parametrize("bound", ["rounds", "context"])
def test_feedback_does_not_extend_model_rounds_or_native_context_hard_limits(bound):
    async def run():
        requests = []

        async def model(value):
            requests.append(value)
            host_call = call(len(requests))
            native = NativeAssistantState("offline", json.dumps({
                "role": "assistant", "content": None, "reasoning": "opaque",
                "tool_calls": [wire_call(host_call)]}))
            return ModelCompletion("", tool_calls=(host_call,), native_assistant=native)

        tools, _ = registry(lambda *_: (ImagePart(b"oversized"),),
                            ToolLimits(max_image_bytes=1, max_model_rounds=2), fresh=True)
        runtime = SimpleRuntime(CallableModel(model), tools=tools)
        if bound == "context":
            initial = _metrics(request().messages + (runtime._tool_budget_message(),))[0]
            runtime.config = replace(runtime.config, max_context_characters=initial)
        with patch.object(runtime, "_compact", wraps=runtime._compact) as compact:
            with pytest.raises(AgentLimitError, match=("model tool round budget exhausted" if bound == "rounds"
                                                      else "native tool continuation cannot fit")):
                await runtime.complete(request())
            assert compact.call_count == 1
        assert len(requests) == (2 if bound == "rounds" else 1)
        assert runtime.usage.summary_calls == 0
    asyncio.run(run())


def wire_call(value):
    return {"id": value.call_id, "type": "function", "function": {
        "name": value.name, "arguments": value.arguments_json}}


def test_default_four_image_batch_recovers_to_submission_with_exact_openrouter_prefix(tmp_path):
    images = tuple(encode_png(bytes([number, 30, 200]) * 4, 2, 2) for number in range(9))
    requests = []
    replies = []

    async def create(**kwargs):
        requests.append(deepcopy(kwargs))
        number = len(requests)
        if number < 3:
            indexes = range(3) if number == 1 else range(3, 9)
            calls = [ToolCall(f"image-{index}", "view_image", json.dumps({
                "path": f"records/own/images/{index}.png"})) for index in indexes]
            if number == 1:
                calls += [ToolCall(f"read-{index}", "read_file", json.dumps({
                    "path": "records/own/info.txt", "start_line": 1, "max_lines": 1})) for index in range(19)]
        else:
            assert number == 3
            calls = [ToolCall("finish", "submit_action", '{"format":"raw","content":"selected-action"}')]
        message = {"role": "assistant", "content": None,
                   "tool_calls": [wire_call(value) for value in calls],
                   "reasoning_details": [{"type": "reasoning.encrypted", "id": f"opaque-{number}",
                                          "data": "signed+/=\n", "index": 0, "format": "vendor-v2"}]}
        replies.append(deepcopy(message))
        return {"choices": [{"finish_reason": "tool_calls", "message": message}]}

    agent = SimpleAgent(api_key="offline-test-key", provider="openrouter",
                        base_url="https://openrouter.ai/api/v1", model="offline/test-model",
                        simple_agent_config={"tools": {"enabled": True, "workspace_parent": str(tmp_path)}})
    async def close():
        pass

    agent._backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close)
    observed = []
    sessions = []
    obs = {"messages": [{"role": "user", "content": "Visible synthetic observation."}]}
    try:
        state, current, *_ = agent._prepare(0, obs)
        files = tuple((f"images/{index}.png", image) for index, image in enumerate(images)) + (("info.txt", b"visible\n"),)
        state.workspace.publish(RecordSnapshot("own", files), current.identity)
        execute = state.workspace.execute

        def capture(value, session):
            result = execute(value, session)
            observed.append(result)
            sessions.append(session)
            return result

        with patch.object(state.workspace, "execute", side_effect=capture):
            result, _ = agent.forward_with_trace({0: obs})
        assert result[0]["action"] == "selected-action"
        assert len(requests) == 3 and state.runtime.usage.action_calls == 3
        assert state.runtime.usage.summary_calls == 0
        assert len(observed) == 28 and all(item.success for item in observed[:23])
        assert len(observed[23:]) == 5
        for item in observed[23:]:
            assert assert_omitted(item)["remaining_images"] == 0
        assert json.loads(observed[-1].parts[0].text)["remaining_calls"] == 4
        assert len({id(session) for session in sessions}) == 1
        session = sessions[0]
        assert session.closed and session.calls == 28
        assert state.workspace.limits.max_calls == 32 and state.workspace.limits.max_images == 4
        assert session.images == 4 and session.image_bytes == sum(map(len, images[:4]))
        assert session.text_bytes == sum(text_size(item.parts) for item in observed)
        for earlier, later, reply in zip(requests, requests[1:], replies):
            prefix = earlier["messages"]
            assert later["messages"][:len(prefix)] == prefix
            assert later["messages"][len(prefix)] == reply
            assert {key: value for key, value in earlier.items() if key != "messages"} == {
                key: value for key, value in later.items() if key != "messages"}
        delivered = [base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                     for message in requests[-1]["messages"] if isinstance(message["content"], list)
                     for part in message["content"] if part["type"] == "image_url"]
        assert delivered == list(images[:4])
        feedback = [message for message in requests[-1]["messages"] if message["role"] == "tool"]
        assert [message["tool_call_id"] for message in feedback] == (
            [f"image-{index}" for index in range(3)] + [f"read-{index}" for index in range(19)]
            + [f"image-{index}" for index in range(3, 9)])
        assert all(json.loads(message["content"])["error"] == ERROR for message in feedback[23:])
    finally:
        agent.close()
