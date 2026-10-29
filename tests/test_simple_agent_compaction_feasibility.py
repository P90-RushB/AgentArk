"""Compaction infeasibility proofs; no SDK, HTTP, Player or task assets."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent_ark.agent.simple_agent import (
    AgentEvent, AgentInputIdentity, AgentLimitError, AgentRequest, CallableModel, ImagePart,
    Message, ModelCompletion, SimpleRuntime, SimpleRuntimeConfig, TextPart,
)
from agent_ark.agent.simple_agent.runtime import _WorkingEntry
from agent_ark.agent.simple_agent.types import ArchiveEntry, NativeAssistantState


def request(number, images=0, text="observation"):
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "agent", 0, str(number)),
                        (Message.text("system", "Keep the task instructions."),
                         Message("user", (TextPart(text),) + (ImagePart(b"small-image"),) * images)))


def text(messages):
    return "\n".join(part.text for message in messages for part in message.parts if isinstance(part, TextPart))


def test_default_video_turn_fails_without_any_useless_summary_and_keeps_archive_and_working_state():
    async def run():
        calls = []

        async def model(value):
            calls.append(value)
            return ModelCompletion("Earlier facts." if value.purpose == "summary" else "action")

        runtime = SimpleRuntime(CallableModel(model))
        for number, images in enumerate((1, 83, 81)):
            await runtime.complete(request(number, images))
        current = request(3, 77)
        runtime.accept_input(current)
        before = runtime.working_snapshot
        archive = runtime.history_snapshot
        for _ in range(2):
            with pytest.raises(AgentLimitError, match=r"complete earlier turn exceeds summary input bounds \(summary_input_tokens\).*max_context_tokens"):
                await runtime.complete(current)
        assert [call.purpose for call in calls] == ["action"] * 3
        assert runtime.usage.action_calls == 3 and runtime.usage.summary_calls == 0
        assert runtime.working_snapshot == before and runtime.history_snapshot == archive
        assert len([item for item in archive if item.kind == "input"]) == 4
        assert "small-image" not in str(runtime._compaction_obstruction(runtime._working, runtime._pinned, frozenset()))

    asyncio.run(run())


@pytest.mark.parametrize("bound", ["characters", "bytes", "images", "image_bytes"])
def test_infeasibility_identifies_static_bounds_without_sending_a_summary(bound):
    async def run():
        seen = []

        async def model(value):
            seen.append(value)
            return ModelCompletion("action")

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(recent_turns=1))
        await runtime.complete(request(1, images=3, text="old-observation " * 100))
        runtime.accept_input(request(2, images=3, text="current-observation " * 100))
        cfg = {"summary_input_characters": 100} if bound == "characters" else {"summary_input_bytes": 100} if bound == "bytes" else {"summary_input_tokens": 1}
        cfg.update({"max_context_characters": 100} if bound == "characters" else
                   {"max_context_bytes": 100} if bound == "bytes" else
                   {"max_images": 2} if bound == "images" else {"max_image_bytes": 10})
        runtime.config = replace(runtime.config, **cfg)
        with pytest.raises(AgentLimitError, match="complete earlier turn exceeds summary input bounds") as failure:
            await runtime.complete(request(2, images=3, text="current-observation " * 100))
        assert ("max_" + ("context_" if bound in ("characters", "bytes") else "") + bound) in str(failure.value)
        assert len(str(failure.value)) < 512
        assert [call.purpose for call in seen] == ["action"]

    asyncio.run(run())


def test_larger_explicit_summary_budget_keeps_the_successful_video_path():
    async def run():
        seen = []

        async def model(value):
            seen.append(value)
            return ModelCompletion("Earlier facts." if value.purpose == "summary" else "action")

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(summary_input_tokens=196608, recent_turns=1))
        for number, images in enumerate((1, 83, 81, 77)):
            await runtime.complete(request(number, images))
        assert [call.purpose for call in seen] == ["action", "action", "action", "summary", "action"]
        assert len([part for message in seen[-1].messages for part in message.parts if isinstance(part, ImagePart)]) == 77
        assert "Earlier facts." in text(seen[-1].messages)
        assert len([item for item in runtime.history_snapshot if item.kind == "input"]) == 4

    asyncio.run(run())


def test_summary_only_can_shrink_large_notes_then_include_the_next_complete_turn():
    async def run():
        calls = []

        async def model(value):
            calls.append(value)
            return ModelCompletion("Short memory." if value.purpose == "summary" else "action")

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            recent_turns=1, max_context_characters=1000, summary_input_characters=1800))
        current = request(3, text="current " * 60)
        runtime.accept_input(current)
        old = request(2, text="old " * 125)
        note = Message.text("user", "Model notes; not environment facts. " + "n" * 900)
        runtime._working = [
            _WorkingEntry(1, "summary", old.identity, (note,), (1, 2)),
            _WorkingEntry(3, "input", old.identity, old.messages[1:], (3,)),
            _WorkingEntry(4, "completion", old.identity, (Message.text("assistant", "old action"),), (4,)),
            _WorkingEntry(5, "input", current.identity, current.messages[1:], (5,)),
        ]
        assert runtime._summary_span(runtime._working, frozenset())[0].kind == "summary"
        assert len(runtime._summary_span(runtime._working, frozenset())) == 1
        assert (await runtime.complete(current)).text == "action"
        assert [value.purpose for value in calls] == ["summary", "summary", "action"]
        assert calls[0].purpose == "summary" and "n" * 900 in text(calls[0].messages)
        assert "old " * 125 in text(calls[1].messages)
        assert calls[-1].purpose == "action" and "Short memory." in text(calls[-1].messages)

    asyncio.run(run())


def test_notes_only_context_can_still_be_summarized():
    async def run():
        calls = []

        async def model(value):
            calls.append(value)
            return ModelCompletion("Short facts." if value.purpose == "summary" else "action")

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(recent_turns=1, max_context_characters=500))
        current = request(2)
        runtime.accept_input(current)
        runtime._working.insert(0, _WorkingEntry(0, "summary", request(1).identity,
                                                (Message.text("user", "earlier notes " * 100),), (0,)))
        await runtime.complete(current)
        assert [call.purpose for call in calls] == ["summary", "action"]

    asyncio.run(run())


def test_removable_native_replay_is_not_treated_as_an_uncompressible_suffix():
    runtime = SimpleRuntime(CallableModel(lambda _: None), SimpleRuntimeConfig(
        recent_turns=1, max_context_bytes=5000, summary_input_tokens=1000))
    first, current = request(1, images=2), request(2)
    runtime.accept_input(first)
    runtime.accept_input(current)
    native = NativeAssistantState("test", '{"role":"assistant","content":"a","reasoning":"' + "n" * 10000 + '"}')
    runtime._working[-1] = replace(runtime._working[-1], messages=(
        current.messages[-1], Message("assistant", (TextPart("a"),), native_assistant=native)))
    assert not runtime._fits(runtime.working_snapshot)
    assert runtime._compaction_obstruction(runtime._working, runtime._pinned, frozenset()) is None


def test_arbitrary_custom_token_counter_is_not_used_as_a_monotone_impossibility_proof():
    runtime = SimpleRuntime(CallableModel(lambda _: None), SimpleRuntimeConfig(), token_counter=lambda _: 999999)
    for number in range(4):
        runtime.accept_input(request(number, images=(1, 83, 81, 77)[number]))
    assert runtime._compaction_obstruction(runtime._working, runtime._pinned, frozenset()) is None


def test_soft_watermark_does_not_become_an_early_hard_failure():
    runtime = SimpleRuntime(CallableModel(lambda _: None), SimpleRuntimeConfig(
        recent_turns=1, max_context_characters=3000, summary_input_tokens=1000))
    runtime.accept_input(request(1, images=2))
    runtime.accept_input(request(2, text="current " * 300))
    assert runtime._fits(runtime.working_snapshot)
    assert not runtime._fits(runtime.working_snapshot, native_headroom=True)
    assert runtime._compaction_obstruction(runtime._working, runtime._pinned, frozenset()) is None


def test_consumed_event_after_a_completion_can_stay_outside_its_summary_span():
    async def run():
        calls = []

        def source(number, characters):
            return AgentRequest(AgentInputIdentity("r", "s", "e", "a", 0, str(number)),
                                (Message.text("system", "Keep"), Message.text("user", "x" * characters)))

        async def model(value):
            calls.append(value)
            return ModelCompletion("Short facts." if value.purpose == "summary" else "A" * 350)

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            recent_turns=1, max_context_characters=1600, summary_input_characters=1300))
        await runtime.complete(source(1, 300))
        runtime.observe_result(AgentEvent(source(1, 300).identity, "ev1", "action.result", "E" * 200))
        await runtime.complete(source(2, 250))
        assert not runtime._required_events
        original_event = next(item for item in runtime.history_snapshot if item.kind == "event")
        await runtime.complete(source(3, 300))
        assert [value.purpose for value in calls] == ["action", "action", "summary", "action"]
        assert "E" * 200 not in text(calls[2].messages)
        assert "E" * 200 in text(calls[3].messages)
        assert original_event in runtime.history_snapshot
        assert original_event.messages[0] in runtime.working_snapshot

    asyncio.run(run())


def test_coalesced_input_can_close_at_an_earlier_inputs_late_completion():
    async def run():
        calls = []

        async def model(value):
            calls.append(value)
            return ModelCompletion("Short facts." if value.purpose == "summary" else "action")

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            recent_turns=1, max_context_characters=1600, summary_input_characters=1300))
        first = request(1, text="x" * 10)
        runtime.accept_input(first)
        runtime.accept_input(request(2, text="x" * 300))
        # A previously running action may finish after a newer observation.
        message = Message.text("assistant", "A" * 350)
        runtime._append_archive(ArchiveEntry(3, "completion", first.identity, (message,)))
        runtime._working.append(_WorkingEntry(3, "completion", first.identity, (message,), (3,)))
        runtime.observe_result(AgentEvent(first.identity, "ev1", "action.result", "E" * 200))
        runtime._required_events.clear()  # The next completed decision consumed it.
        current = request(3, text="x" * 800)
        await runtime.complete(current)
        assert [value.purpose for value in calls] == ["summary", "action"]
        assert "E" * 200 not in text(calls[0].messages)
        assert "E" * 200 in text(calls[1].messages)

    asyncio.run(run())
