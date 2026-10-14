# SPDX-License-Identifier: Apache-2.0
# Adapted with permission from the ArkArena stdlib core (ark_arena/harness).
from __future__ import annotations

import asyncio
import random
import struct
import zlib
from dataclasses import FrozenInstanceError, replace

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_ark.agent.simple_agent import (
    CallableModel, AgentEvent, AgentInputIdentity, AgentLimitError,
    AgentRequest, AgentReuseError, ImagePart, Message, ModelCompletion,
    ModelUsage, SimpleRuntime, SimpleRuntimeConfig, TextPart, chat_messages,
    encode_png, image_from_pixels,
)
from agent_ark.agent.simple_agent.encoding import encoded_messages


def identity(number=1, *, episode="episode-1", incarnation=1, agent="agent-1", run="run-1", session="session-1"):
    return AgentInputIdentity(run, session, episode, agent, incarnation, str(number))


def request(number=1, text="current observation", **scope):
    return AgentRequest(identity(number, **scope), (
        Message.text("system", "Keep instructions"),
        Message.text("developer", '{"action_schema":"complete code or JSON"}'),
        Message.text("user", text),
    ))


def texts(messages):
    return "\n".join(part.text for message in messages for part in message.parts if isinstance(part, TextPart))


def config(**changes):
    return SimpleRuntimeConfig(**{
        "max_context_characters": 1_300, "max_context_bytes": 20_000,
        "max_context_tokens": 20_000, "output_reserve_tokens": 128,
        "recent_turns": 1, "summary_output_tokens": 512,
        **changes,
    })


def test_frozen_messages_and_real_png_origin():
    pixels = bytes([255, 0, 0, 0, 0, 255])  # Bottom row red, top row blue.
    png = encode_png(pixels, 1, 2, origin="bottom_left")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    position = 8
    compressed = b""
    while position < len(png):
        size = struct.unpack(">I", png[position:position + 4])[0]
        kind = png[position + 4:position + 8]
        if kind == b"IDAT":
            compressed += png[position + 8:position + 8 + size]
        position += size + 12
    assert zlib.decompress(compressed) == bytes([0, 0, 0, 255, 0, 255, 0, 0])
    image = image_from_pixels(pixels, 1, 2, origin="bottom_left")
    assert image.origin == "top_left" and image.data == png
    with pytest.raises(FrozenInstanceError):
        image.data = b"changed"
    with pytest.raises(ValueError, match="immutable"):
        ImagePart(bytearray(png))
    with pytest.raises(ValueError, match="tuple"):
        Message("user", [TextPart("mutable")])
    with pytest.raises(ValueError, match="top-left"):
        chat_messages((Message("user", (ImagePart(png, origin="bottom_left"),)),))


def test_current_input_images_code_and_instructions_are_complete_and_archived():
    async def run():
        calls = []
        code = 'public class Policy { string value = "a\\b\\c"; }\n' * 500
        image = image_from_pixels(bytes([1, 2, 3]), 1, 1)

        async def model(value):
            calls.append(value)
            return ModelCompletion(code, ModelUsage(12, 20, 32))

        runtime = SimpleRuntime(CallableModel(model))
        source = replace(request(), messages=(*request().messages, Message("user", (image,))))
        response = await runtime.complete(source)
        assert response.text == code
        assert calls[0].messages == source.messages
        assert calls[0].token_count_is_estimate is True
        assert runtime.history_snapshot[0].messages == source.messages
        assert runtime.history_snapshot[1].text == code
        assert runtime.usage.action_calls == 1 and runtime.usage.summary_calls == 0
        assert runtime.usage.action_usage == (ModelUsage(12, 20, 32),)
        runtime.accept_input(source)
        assert len(runtime.history_snapshot) == 2
        with pytest.raises(AgentReuseError, match="completed"):
            await runtime.complete(source)
        with pytest.raises(AgentReuseError, match="conflicting"):
            runtime.accept_input(replace(source, messages=request(text="changed").messages))

    asyncio.run(run())


def test_large_camera_uses_separate_image_estimate_and_byte_bounds():
    async def run():
        # Incompressible normal camera pixels exceed the old 512k JSON bound.
        pixels = random.Random(17).randbytes(1024 * 768 * 3)
        image = image_from_pixels(pixels, 1024, 768)
        source = replace(request(), messages=(*request().messages, Message("user", (image,))))
        calls = []

        async def model(value):
            calls.append(value)
            return ModelCompletion("{}")

        runtime = SimpleRuntime(CallableModel(model))
        await runtime.complete(source)
        assert len(encoded_messages(source.messages)) > 512_000
        assert calls[0].input_tokens < 2_000
        assert calls[0].messages[-1].parts[0].data == image.data
        limited = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_image_bytes=10))
        with pytest.raises(AgentLimitError):
            await limited.complete(source)
        assert limited.history_snapshot[0].messages[-1].parts[0].data == image.data
        assert len(calls) == 1  # No image was silently dropped to allow a call.

    asyncio.run(run())


def test_compaction_retains_early_facts_and_originals_and_separates_usage():
    async def run():
        calls = []

        async def model(value):
            calls.append(value)
            if value.purpose == "summary":
                assert "early fact: copper key" in texts(value.messages)
                return ModelCompletion("Remember early fact: copper key.", ModelUsage(20, 8, 28))
            return ModelCompletion("action " + "A" * 280, ModelUsage(30, 90, 120))

        runtime = SimpleRuntime(CallableModel(model), config())
        for number in range(1, 5):
            await runtime.complete(request(number, "early fact: copper key " + "x" * 380 if number == 1 else "x" * 400))
        assert any(call.purpose == "summary" for call in calls)
        assert "copper key" in texts(calls[-1].messages)
        assert "Model notes; not environment facts" in texts(calls[-1].messages)
        assert sum(message.role == "system" for message in calls[-1].messages) == 1
        assert runtime.history_snapshot[0].messages[-1].parts[0].text.startswith("early fact:")
        assert len([entry for entry in runtime.history_snapshot if entry.kind == "input"]) == 4
        assert runtime.usage.action_calls == 4
        assert runtime.usage.summary_calls >= 1
        assert len(runtime.usage.action_usage) == 4
        assert all(usage.output_tokens == 8 for usage in runtime.usage.summary_usage)

    asyncio.run(run())


def test_results_appended_during_summary_are_not_lost():
    async def run():
        summarizing = asyncio.Event()
        release = asyncio.Event()
        action_calls = []

        async def model(value):
            if value.purpose == "summary":
                summarizing.set()
                await release.wait()
                return ModelCompletion("Earlier facts retained.")
            action_calls.append(value)
            return ModelCompletion("a" * 280)

        runtime = SimpleRuntime(CallableModel(model), config())
        await runtime.complete(request(1, "x" * 400))
        await runtime.complete(request(2, "x" * 400))
        pending = asyncio.create_task(runtime.complete(request(3, "latest input " + "x" * 350)))
        await asyncio.wait_for(summarizing.wait(), 1)
        event = AgentEvent(identity(2), "late-result", "action.result", '{"accepted":true}')
        runtime.observe_result(event)
        release.set()
        await pending
        assert "late-result" == next(entry.event_id for entry in runtime.history_snapshot if entry.kind == "event")
        assert '"accepted":true' in texts(runtime.working_snapshot)
        assert '"accepted":true' not in texts(action_calls[-1].messages)  # A was frozen before this result arrived.
        assert "latest input" in texts(action_calls[-1].messages)
        assert runtime.usage.action_calls == 3
        await runtime.complete(request(4, "next input"))
        assert '"accepted":true' in texts(action_calls[-1].messages)
        assert runtime.usage.action_calls == 4

    asyncio.run(run())


def test_default_image_limit_compacts_only_after_128_images():
    async def run():
        calls = []
        image = image_from_pixels(bytes([1, 2, 3]), 1, 1)

        async def model(value):
            calls.append(value)
            return ModelCompletion("Earlier camera facts retained." if value.purpose == "summary" else "{}")

        # Isolate the image-count policy from the independent token limit.
        runtime = SimpleRuntime(CallableModel(model), token_counter=lambda messages: 1)
        for number in range(1, 129):
            source = replace(request(number), messages=(*request(number).messages, Message("user", (image,))))
            await runtime.complete(source)
        assert runtime.usage.summary_calls == 0
        assert sum(isinstance(part, ImagePart) for message in calls[-1].messages for part in message.parts) == 128

        source = replace(request(129), messages=(*request(129).messages, Message("user", (image,))))
        await runtime.complete(source)
        assert runtime.usage.summary_calls == 1
        assert sum(isinstance(part, ImagePart) for message in calls[-1].messages for part in message.parts) == 2
        assert sum(isinstance(part, ImagePart) for entry in runtime.history_snapshot
                   for message in entry.messages for part in message.parts) == 129

    asyncio.run(run())


def test_compaction_with_large_prior_image_keeps_original_and_sends_real_image_to_summary():
    async def run():
        image = image_from_pixels(random.Random(19).randbytes(1024 * 768 * 3), 1024, 768)
        seen = []

        async def model(value):
            seen.append(value)
            return ModelCompletion("Prior camera facts retained." if value.purpose == "summary" else "{}")

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_images=2, recent_turns=1))
        for number in range(1, 4):
            source = replace(request(number), messages=(*request(number).messages, Message("user", (image,))))
            await runtime.complete(source)
        summaries = [value for value in seen if value.purpose == "summary"]
        assert len(summaries) == 1
        assert len(encoded_messages(summaries[0].messages)) > 512_000
        assert any(part == image for message in summaries[0].messages for part in message.parts)
        assert sum(part == image for entry in runtime.history_snapshot for message in entry.messages for part in message.parts) == 3
        assert "Model notes" in texts(seen[-1].messages)

    asyncio.run(run())


def test_summary_await_new_input_does_not_rebind_original_model_request():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        actions = []

        async def model(value):
            if value.purpose == "summary":
                entered.set()
                await release.wait()
                return ModelCompletion("Earlier facts retained.")
            actions.append(value)
            return ModelCompletion("a" * 280)

        runtime = SimpleRuntime(CallableModel(model), config())
        await runtime.complete(request(1, "x" * 400))
        await runtime.complete(request(2, "x" * 400))
        pending = asyncio.create_task(runtime.complete(request(3, "A observation " + "x" * 350)))
        await asyncio.wait_for(entered.wait(), 1)
        replacement = request(4, "B observation")
        runtime.accept_input(replacement)
        runtime.observe_result(AgentEvent(identity(4), "B-feedback", "action.result", "B own feedback"))
        release.set()
        await pending
        assert "A observation" in texts(actions[-1].messages)
        assert "B observation" not in texts(actions[-1].messages)
        assert "B own feedback" not in texts(actions[-1].messages)
        assert "B observation" in texts(runtime.working_snapshot)
        await runtime.complete(replacement)
        assert "B observation" in texts(actions[-1].messages)
        assert "B own feedback" in texts(actions[-1].messages)
        assert next(entry for entry in runtime.history_snapshot if entry.kind == "completion" and entry.identity.input_id == "3").identity == identity(3)

    asyncio.run(run())


def test_summary_transaction_rolls_back_after_second_summary_failure():
    async def run():
        summaries = 0

        async def model(value):
            nonlocal summaries
            assert value.purpose == "summary"
            summaries += 1
            if summaries == 2:
                raise RuntimeError("second summary failed")
            return ModelCompletion("Facts retained.")

        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=700, summary_input_characters=600))
        for number in range(1, 5):
            runtime.accept_input(request(number, "x" * 300))
        before = runtime.working_snapshot
        with pytest.raises(RuntimeError, match="second summary"):
            await runtime.complete(request(4, "x" * 300))
        assert runtime.working_snapshot == before
        assert runtime.usage.action_calls == 0 and runtime.usage.summary_calls == 2
        assert runtime.history_snapshot[-1].kind == "summary_completion"

    asyncio.run(run())


def test_summary_input_budget_never_splits_a_turn_before_its_completion():
    async def run():
        calls = []
        runtime = None

        async def model(value):
            calls.append(value)
            assert value.purpose == "action"
            if len(calls) == 1:
                runtime.observe_result(AgentEvent(identity(1), "mid-turn", "action.result", "e" * 140))
            return ModelCompletion("a" * 300)

        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=1_200, summary_input_characters=600))
        await runtime.complete(request(1, "x" * 140))
        await runtime.complete(request(2, "x" * 140))
        source = request(3, "x" * 500)
        runtime.accept_input(source)
        before = runtime.working_snapshot
        with pytest.raises(AgentLimitError):
            await runtime.complete(source)
        assert runtime.working_snapshot == before
        assert runtime.usage.summary_calls == 0
        assert runtime.history_snapshot[1].kind == "event"
        assert runtime.history_snapshot[2].kind == "completion"

    asyncio.run(run())


def test_interleaved_capture_and_late_completion_cannot_be_partially_summarized():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def model(value):
            calls.append(value)
            if len(calls) == 1:
                entered.set()
                await release.wait()
            return ModelCompletion("a" * 300 if value.purpose == "action" else "Facts retained.")

        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=650))
        pending = asyncio.create_task(runtime.complete(request(1, "A" * 120)))
        await asyncio.wait_for(entered.wait(), 1)
        replacement = request(2, "B" * 300)
        runtime.accept_input(replacement)
        release.set()
        await pending
        before = runtime.working_snapshot
        with pytest.raises(AgentLimitError):
            await runtime.complete(replacement)
        assert runtime.working_snapshot == before and len(calls) == 1
        # A later observation closes the coalesced B turn, allowing A and its
        # interleaved completion to be summarized together.
        await runtime.complete(request(3, "C" * 200))
        assert runtime.usage.summary_calls == 1
        assert "A" * 120 in texts(calls[1].messages)
        assert "a" * 300 in texts(calls[1].messages)

    asyncio.run(run())


def test_action_cancellation_cannot_repeat_the_same_model_call():
    async def run():
        entered = asyncio.Event()

        async def model(value):
            entered.set()
            await asyncio.Event().wait()

        runtime = SimpleRuntime(CallableModel(model))
        pending = asyncio.create_task(runtime.complete(request()))
        await asyncio.wait_for(entered.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        with pytest.raises(AgentReuseError, match="completed"):
            await runtime.complete(request())
        assert runtime.usage.action_calls == 1
        runtime.close()
        runtime.reset()
        runtime.validate_fresh()

    asyncio.run(run())


def test_summary_output_bound_archives_raw_text_and_preserves_original_context():
    async def run():
        async def model(value):
            return ModelCompletion("too long summary text" if value.purpose == "summary" else "a" * 280)

        runtime = SimpleRuntime(CallableModel(model), config(summary_output_characters=5))
        await runtime.complete(request(1, "x" * 400))
        await runtime.complete(request(2, "x" * 400))
        source = request(3, "x" * 400)
        runtime.accept_input(source)
        before = runtime.working_snapshot
        with pytest.raises(AgentLimitError, match="summary output"):
            await runtime.complete(source)
        assert runtime.working_snapshot == before
        assert runtime.history_snapshot[-1].kind == "summary_completion"
        assert runtime.history_snapshot[-1].text == "too long summary text"
        assert runtime.usage.action_calls == 2

    asyncio.run(run())


def test_summary_call_limit_is_not_reset_by_retrying_same_input():
    async def run():
        async def model(value):
            if value.purpose == "summary":
                raise RuntimeError("failed summary")
            return ModelCompletion("a" * 280)

        runtime = SimpleRuntime(CallableModel(model), config(max_summary_calls=1))
        await runtime.complete(request(1, "x" * 400))
        await runtime.complete(request(2, "x" * 400))
        source = request(3, "x" * 400)
        with pytest.raises(RuntimeError, match="failed summary"):
            await runtime.complete(source)
        with pytest.raises(AgentLimitError, match="summary call bound"):
            await runtime.complete(source)
        assert runtime.usage.summary_calls == 1 and runtime.usage.action_calls == 2

    asyncio.run(run())


def test_caller_token_counter_is_labelled_and_independent_hard_bounds_remain():
    async def run():
        calls = []

        async def model(value):
            calls.append(value)
            return ModelCompletion("{}")

        runtime = SimpleRuntime(CallableModel(model), token_counter=lambda messages: 17)
        runtime.validate_fresh()
        await runtime.complete(request())
        assert calls[0].input_tokens == 17 and calls[0].token_count_is_estimate is False
        with pytest.raises(AgentReuseError, match="fresh"):
            runtime.validate_fresh()
        too_small = SimpleRuntime(CallableModel(model), config(max_context_bytes=1), token_counter=lambda messages: 0)
        with pytest.raises(AgentLimitError):
            await too_small.complete(request())
        assert len(calls) == 1

    asyncio.run(run())


def test_first_input_does_not_revive_prior_incarnation_feedback():
    async def run():
        async def model(value):
            assert "prior incarnation" not in texts(value.messages)
            assert "global early feedback" in texts(value.messages)
            return ModelCompletion("{}")

        runtime = SimpleRuntime(CallableModel(model))
        runtime.observe_result(AgentEvent(identity("old", incarnation=1), "old-event", "action.result", "prior incarnation"))
        runtime.observe_result(AgentEvent(identity("global", incarnation=0), "global-event", "reward.event", "global early feedback"))
        await runtime.complete(request(incarnation=2))
        assert runtime.history_snapshot[0].identity.incarnation == 1
        assert "prior incarnation" not in texts(runtime.working_snapshot)

    asyncio.run(run())


@pytest.mark.parametrize("new_episode", [False, True])
def test_scope_change_during_multiple_summaries_keeps_original_required_feedback_protected(new_episode):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        summaries = []

        async def model(value):
            assert value.purpose == "summary"  # The original input cannot fit; no action call is permitted.
            summaries.append(value)
            if len(summaries) == 1:
                entered.set()
                await release.wait()
                return ModelCompletion("Earlier observation facts. " * 4)
            return ModelCompletion("K")

        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=1_000, summary_input_characters=2_000))
        runtime.accept_input(request(1, "old observation " + "x" * 280))
        runtime.accept_input(request(2, "old observation " + "y" * 280))
        event = AgentEvent(identity(2), "must-retain", "action.result", "mandatory original feedback " + "f" * 780)
        runtime.observe_result(event)
        original = request(3, "A current observation " + "a" * 580)
        runtime.accept_input(original)
        pending = asyncio.create_task(runtime.complete(original))
        await asyncio.wait_for(entered.wait(), 1)
        replacement = request(4, "B current observation", episode="episode-2" if new_episode else "episode-1", incarnation=2)
        runtime.accept_input(replacement)
        replacement_context = runtime.working_snapshot
        assert "mandatory original feedback" not in texts(replacement_context)
        release.set()
        with pytest.raises(AgentLimitError):
            await pending
        assert len(summaries) >= 2
        assert all("mandatory original feedback" not in texts(value.messages) for value in summaries)
        assert runtime.working_snapshot == replacement_context
        assert "B current observation" in texts(runtime.working_snapshot)
        assert runtime.usage.action_calls == 0
        assert next(entry for entry in runtime.history_snapshot if entry.event_id == "must-retain").identity == identity(2)

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_summary_failure_or_cancellation_preserves_working_transaction(cancel):
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def model(value):
            if value.purpose == "summary":
                entered.set()
                await release.wait()
                raise RuntimeError("summary unavailable")
            return ModelCompletion("a" * 280)

        runtime = SimpleRuntime(CallableModel(model), config())
        await runtime.complete(request(1, "x" * 400))
        await runtime.complete(request(2, "x" * 400))
        source = request(3, "x" * 400)
        runtime.accept_input(source)
        before = runtime.working_snapshot
        pending = asyncio.create_task(runtime.complete(source))
        await asyncio.wait_for(entered.wait(), 1)
        event = AgentEvent(identity(2), "observed-while-awaiting", "action.result", "kept")
        runtime.observe_result(event)
        if cancel:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            release.set()
            with pytest.raises(RuntimeError, match="unavailable"):
                await pending
        assert runtime.working_snapshot[:-1] == before
        assert "kept" in texts(runtime.working_snapshot[-1:])
        assert runtime.usage.action_calls == 2 and runtime.usage.summary_calls == 1
        assert len([entry for entry in runtime.history_snapshot if entry.kind == "completion"]) == 2

    asyncio.run(run())


def test_coalesced_inputs_can_compact_without_fake_action_calls():
    async def run():
        purposes = []

        async def model(value):
            purposes.append(value.purpose)
            return ModelCompletion("Older observations retained." if value.purpose == "summary" else "{}")

        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=750))
        for number in range(1, 4):
            runtime.accept_input(request(number, "coalesced observation " + "x" * 300))
        await runtime.complete(request(4, "last observation " + "x" * 300))
        assert purposes == ["summary", "action"]
        assert runtime.usage.action_calls == 1
        assert len([entry for entry in runtime.history_snapshot if entry.kind == "input"]) == 4

    asyncio.run(run())


def test_scope_boundaries_raw_late_results_and_explicit_reset():
    async def run():
        async def model(value):
            return ModelCompletion("{}")

        runtime = SimpleRuntime(CallableModel(model))
        initial = AgentEvent(identity("initial"), "initial-result", "episode.opened", "initial feedback")
        runtime.observe_result(initial)
        await runtime.complete(request(1, "old episode fact"))
        assert "initial feedback" in texts(runtime.working_snapshot)
        await runtime.complete(replace(request(2, "new episode", episode="episode-2", incarnation=2), history_policy="carry"))
        assert "old episode fact" in texts(runtime.working_snapshot)
        await runtime.complete(request(3, "rejoined", episode="episode-2", incarnation=3))
        assert "old episode fact" not in texts(runtime.working_snapshot)
        old_event = AgentEvent(identity(1), "final-old-result", "action.result", "late original result")
        runtime.observe_result(old_event)
        assert runtime.history_snapshot[-1].identity == identity(1)
        assert "late original result" not in texts(runtime.working_snapshot)
        await runtime.complete(request(4, "third episode", episode="episode-3"))
        assert "rejoined" not in texts(runtime.working_snapshot)
        for changed in (request(run="another"), request(session="another"), request(agent="another")):
            with pytest.raises(AgentReuseError, match="bound"):
                runtime.accept_input(changed)
        with pytest.raises(AgentReuseError, match="closed"):
            runtime.reset()
        runtime.close()
        runtime.reset()
        assert runtime.history_snapshot == () and runtime.usage.action_calls == 0
        await runtime.complete(request(run="new-run", session="new-session"))
        assert len(runtime.history_snapshot) == 2

    asyncio.run(run())


def test_live_instruction_changes_fail_before_archiving_and_events_deduplicate():
    async def run():
        runtime = SimpleRuntime(CallableModel(lambda value: asyncio.sleep(0, result=ModelCompletion("{}"))))
        await runtime.complete(request())
        changed = replace(request(2), messages=(Message.text("system", "new rules"), *request(2).messages[1:]))
        with pytest.raises(AgentReuseError, match="instructions"):
            runtime.accept_input(changed)
        assert len(runtime.history_snapshot) == 2
        event = AgentEvent(identity(), "same-event", "action.result", "complete feedback")
        runtime.observe_result(event)
        runtime.observe_result(event)
        assert len(runtime.history_snapshot) == 3
        with pytest.raises(AgentReuseError, match="conflicting"):
            runtime.observe_result(replace(event, payload_json="different"))

    asyncio.run(run())


def test_complete_code_budget_accounts_json_escaping_without_truncation():
    async def run():
        called = False

        async def model(value):
            nonlocal called
            called = True
            return ModelCompletion("{}")

        code = 'var s = "\\\\\\\\";\n' * 200
        source = request(text=code)
        assert len(encoded_messages(source.messages)) > len(code.encode("utf-8"))
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=20_000, max_context_bytes=len(code.encode("utf-8"))))
        with pytest.raises(AgentLimitError):
            await runtime.complete(source)
        assert not called
        assert runtime.history_snapshot[0].messages[-1].parts == (TextPart(code),)

    asyncio.run(run())


def test_archive_exhaustion_fails_explicitly_with_raw_completion_and_input_preserved():
    async def run():
        runtime = SimpleRuntime(CallableModel(lambda value: asyncio.sleep(0, result=ModelCompletion("bad JSON raw"))), SimpleRuntimeConfig(max_archive_entries=2))
        response = await runtime.complete(request())
        assert response.text == "bad JSON raw"  # Parsing is exclusively the bridge's job.
        with pytest.raises(AgentLimitError, match="archive"):
            runtime.accept_input(request(2))
        assert len(runtime.history_snapshot) == 2
        assert runtime.history_snapshot[1].text == "bad JSON raw"

    asyncio.run(run())


def test_two_concurrent_agent_agents_never_share_memory():
    async def run():
        seen = []

        async def model(value):
            seen.append(texts(value.messages))
            await asyncio.sleep(0)
            return ModelCompletion("{}")

        shared_model = CallableModel(model)
        left, right = SimpleRuntime(shared_model), SimpleRuntime(shared_model)
        await asyncio.gather(left.complete(request(text="left private", agent="left")), right.complete(request(text="right private", agent="right")))
        assert all(not ("left private" in text and "right private" in text) for text in seen)
        assert "right private" not in texts(left.working_snapshot)
        assert "left private" not in texts(right.working_snapshot)

    asyncio.run(run())



import asyncio
import base64
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent import (
    CallableModel, ChatCompletionsModel, Message, ModelCompletion,
    ModelRequest, ModelResponseError, ModelUsage, TextPart, image_from_pixels,
)


def response(**message):
    return {"choices": [{"finish_reason": "stop", "message": {"content": "full code\n", **message}}], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}


def client(return_value, calls):
    async def create(**kwargs):
        calls.append(kwargs)
        return return_value
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_chat_client_multimodal_serialization_and_caller_ownership():
    async def run():
        calls = []
        image = image_from_pixels(bytes([255, 0, 0, 128]), 1, 1, channels=4)
        messages = (Message("user", (TextPart('quote " and code\\\n'), image)),)
        compatible = client(response(), calls)
        model = ChatCompletionsModel(compatible, "caller-model")
        completion = await model.complete(ModelRequest(messages, max_output_tokens=77))
        assert completion == ModelCompletion("full code\n", ModelUsage(12, 3, 15))
        assert calls[0]["model"] == "caller-model"
        assert calls[0]["max_completion_tokens"] == 77
        assert "max_tokens" not in calls[0]
        content = calls[0]["messages"][0]["content"]
        assert content[0]["text"] == 'quote " and code\\\n'
        assert base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]) == image.data
        calls[0]["messages"][0]["content"][0]["text"] = "mutated caller copy"
        assert messages[0].parts[0].text == 'quote " and code\\\n'
        legacy = ChatCompletionsModel(compatible, "compatible-model", output_token_parameter="max_tokens")
        await legacy.complete(ModelRequest(messages, max_output_tokens=33))
        assert calls[1]["max_tokens"] == 33
        assert not hasattr(model, "close")  # The adapter never owns/enters/exits this client.

    asyncio.run(run())


@pytest.mark.parametrize("bad", [
    {}, {"choices": []}, {"choices": [{}, {}]},
    {"choices": [{"finish_reason": "length", "message": {"content": "cut code"}}]},
    {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]},
    response(content=None), response(content=[{"type": "text", "text": "unsupported"}]),
    response(tool_calls=[{"id": "tool"}]), response(refusal="no"),
    {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": True}},
])
def test_chat_client_response_failures_never_retry(bad):
    async def run():
        calls = []
        model = ChatCompletionsModel(client(bad, calls), "model")
        with pytest.raises(ModelResponseError):
            await model.complete(ModelRequest((Message.text("user", "current"),)))
        assert len(calls) == 1

    asyncio.run(run())


def test_chat_object_response_and_no_usage_are_supported():
    async def run():
        value = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="{}"))])
        model = ChatCompletionsModel(client(value, []), "model")
        assert await model.complete(ModelRequest((Message.text("user", "current"),))) == ModelCompletion("{}")

    asyncio.run(run())


def test_callable_requires_async_immutable_completion():
    async def run():
        source = ModelRequest((Message.text("user", "current"),))
        valid = CallableModel(lambda value: asyncio.sleep(0, result=ModelCompletion("{}")))
        assert await valid.complete(source) == ModelCompletion("{}")
        for function in (lambda value: ModelCompletion("{}"), lambda value: asyncio.sleep(0, result="wrong")):
            with pytest.raises(ModelResponseError):
                await CallableModel(function).complete(source)

    asyncio.run(run())


def test_sync_chat_client_rejected_without_second_call():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response()

    async def run():
        compatible = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with pytest.raises(ModelResponseError, match="asynchronous"):
            await ChatCompletionsModel(compatible, "model").complete(ModelRequest((Message.text("user", "current"),)))

    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("suppress", [False, True])
def test_timeout_is_bounded_even_when_provider_suppresses_cancellation(suppress):
    async def run():
        calls = 0
        async def model(value):
            nonlocal calls
            calls += 1
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if suppress:
                    return ModelCompletion("must never become an action")
                raise
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(timeout_seconds=0.01))
        with pytest.raises(AgentLimitError, match="time budget"):
            await runtime.complete(request())
        assert calls == 1 and runtime.usage.action_calls == 1
        assert not any(entry.kind == "completion" for entry in runtime.history_snapshot)
        with pytest.raises(AgentReuseError):
            await runtime.complete(request())
        runtime.close()
    asyncio.run(run())


def test_external_cancellation_is_not_reported_as_timeout():
    async def run():
        started = asyncio.Event()
        async def model(value):
            started.set()
            await asyncio.Event().wait()
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(timeout_seconds=1))
        work = asyncio.create_task(runtime.complete(request()))
        await started.wait()
        work.cancel("caller cancelled")
        with pytest.raises(asyncio.CancelledError) as stopped:
            await work
        # Python 3.10 may discard the cancel message when awaiting a Task.
        assert isinstance(stopped.value, asyncio.CancelledError)
        assert not any(entry.kind == "completion" for entry in runtime.history_snapshot)
        runtime.close()
    asyncio.run(run())


def test_synchronous_provider_overrun_is_checked_before_accepting_output():
    import time
    async def run():
        async def model(value):
            time.sleep(0.02)
            return ModelCompletion("late output")
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(timeout_seconds=0.005))
        with pytest.raises(AgentLimitError, match="time budget"):
            await runtime.complete(request())
        assert not any(entry.kind == "completion" for entry in runtime.history_snapshot)
    asyncio.run(run())


def test_model_call_and_action_output_limits_fail_without_partial_return():
    async def run():
        async def model(value):
            return ModelCompletion("完整代码" * 10)
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_action_output_bytes=20))
        with pytest.raises(AgentLimitError, match="action output"):
            await runtime.complete(request())
        assert runtime.history_snapshot[-1].text == "完整代码" * 10
        assert "完整代码" not in texts(runtime.working_snapshot)
        bounded = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_model_calls=1))
        assert (await bounded.complete(request())).text == "完整代码" * 10
        with pytest.raises(AgentLimitError, match="model call budget"):
            await bounded.complete(request(2))
        assert bounded.usage.action_calls == 1
    asyncio.run(run())


def test_recent_complete_code_and_required_feedback_are_never_cut_into_summary():
    async def run():
        seen = []
        code = "public sealed class AgentAction : CodeAction { /* full */ }\n" * 5
        async def model(value):
            seen.append(value)
            return ModelCompletion("Earlier complete turns." if value.purpose == "summary" else code)
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=1300, recent_turns=2))
        await runtime.complete(request(1, "old fact " + "x" * 260))
        await runtime.complete(request(2, "retained second observation " + "y" * 260))
        runtime.observe_result(AgentEvent(identity(2), "required", "action.result", "KEEP REQUIRED FEEDBACK"))
        await runtime.complete(request(3, "current third observation " + "z" * 260))
        summary = next(value for value in seen if value.purpose == "summary")
        assert "retained second observation" not in texts(summary.messages)
        assert "KEEP REQUIRED FEEDBACK" not in texts(summary.messages)
        action = seen[-1]
        assert any(message.parts == (TextPart(code),) for message in action.messages)
        assert "KEEP REQUIRED FEEDBACK" in texts(action.messages)
        assert action.messages[:2] == request().messages[:2]
    asyncio.run(run())


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "120"])
def test_timeout_configuration_rejects_invalid_bounds(value):
    with pytest.raises(ValueError):
        SimpleRuntimeConfig(timeout_seconds=value)


def test_rejected_archive_input_does_not_supersede_notes_scope(tmp_path):
    from agent_ark.agent.simple_agent import FileWorkspace, ToolCall
    async def run():
        workspace = FileWorkspace(tmp_path)
        runtime = SimpleRuntime(CallableModel(lambda r: asyncio.sleep(0, result=ModelCompletion("ok"))),
                                SimpleRuntimeConfig(max_archive_entries=2), tools=workspace)
        await runtime.complete(request())
        original_generation = workspace._generation
        with pytest.raises(AgentLimitError, match="archive"):
            runtime.accept_input(request(2, episode="other", incarnation=2))
        assert workspace._generation == original_generation
        session = workspace.begin(request())
        assert workspace.execute(ToolCall("note", "write_note", '{"path":"notes/own.md","content":"preserved"}'), session).success
    asyncio.run(run())


def test_core_import_does_not_load_arkarena_or_provider_sdks():
    import os
    import subprocess
    code = """
import sys
class BlockProviders:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'ark_arena', 'openai', 'openai_codex'}:
            raise RuntimeError('provider dependency prohibited')
sys.meta_path.insert(0, BlockProviders())
from agent_ark.agent.simple_agent import SimpleRuntime, CallableModel, FileWorkspace
assert not any(name.split('.')[0] in {'ark_arena', 'openai', 'openai_codex'} for name in sys.modules)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("failure", ["calls", "result", "time"])
def test_completed_tool_result_is_archived_before_later_batch_budget_failure(tmp_path, failure):
    import hashlib
    import json
    from agent_ark.agent.simple_agent import FileWorkspace, ToolCall, ToolLimits, ToolSpec
    async def run():
        limits = ToolLimits(max_calls=1) if failure == "calls" else (
            ToolLimits(max_total_text_bytes=200) if failure == "result" else ToolLimits())
        workspace = FileWorkspace(tmp_path, limits=limits)
        if failure == "time":
            def expire(args, session):
                session.deadline = 0
                return (TextPart("finished after the local deadline"),)
            workspace.register(ToolSpec("expire", "Expire this synthetic test session.",
                                        '{"type":"object","additionalProperties":false}'), expire)
        write = ToolCall("write", "write_note", '{"path":"notes/own.md","content":"retained"}')
        second = ToolCall("next", "expire", "{}") if failure == "time" else ToolCall(
            "next", "list_files", '{"path":"notes","offset":0,"limit":20}')
        rounds = 0
        def assert_complete_pairs(messages):
            pending = set()
            for message in messages:
                if message.tool_calls:
                    assert not pending
                    pending.update(call.call_id for call in message.tool_calls)
                elif message.role == "tool":
                    assert message.tool_call_id in pending
                    pending.remove(message.tool_call_id)
                else:
                    assert not pending
            assert not pending
        async def model(value):
            nonlocal rounds
            rounds += 1
            assert_complete_pairs(value.messages)
            if rounds == 1:
                return ModelCompletion("", tool_calls=(write, second))
            if rounds == 2:
                # Failed batch audit entries never create a dangling partial
                # tool chain in the subsequent decision's model context.
                assert not any(message.tool_calls or message.role == "tool" for message in value.messages)
                if failure == "result":
                    return ModelCompletion("", tool_calls=(ToolCall("recover", "list_files",
                        '{"path":"notes","offset":0,"limit":20}'),))
                return ModelCompletion("", tool_calls=(ToolCall("recover", "read_file",
                    '{"path":"notes/own.md","start_line":1,"max_lines":10}'),))
            assert ("notes/own.md" if failure == "result" else "retained") in texts(value.messages)
            return ModelCompletion("recovered")
        runtime = SimpleRuntime(CallableModel(model), tools=workspace)
        with pytest.raises(AgentLimitError):
            await runtime.complete(request())
        audits = [entry for entry in runtime.history_snapshot if entry.kind == "event"]
        assert [entry.sequence for entry in audits] == [2, 3]
        assert [entry.event_id for entry in audits] == ["tool-result-2", "tool-result-3"]
        assert json.loads(audits[0].text) == {"kind": "tool_result", "call_id": "write",
                                             "name": "write_note", "success": True, "result_returned": True}
        exact_result = json.loads(audits[0].messages[1].parts[0].text)
        assert exact_result == {"path": "notes/own.md", "bytes": 8,
                               "sha256": hashlib.sha256(b"retained").hexdigest(), "model_memory": True}
        assert audits[0].messages[0].tool_calls == (write,)
        assert audits[0].messages[1].tool_call_id == "write"
        assert json.loads(audits[1].text)["error_type"] == "AgentLimitError"
        assert json.loads(audits[1].text)["result_returned"] is False
        assert not any(entry.kind == "tool_exchange" for entry in runtime.history_snapshot)
        assert_complete_pairs(runtime.working_snapshot)
        assert not any(message.tool_calls or message.role == "tool" for message in runtime.working_snapshot)
        assert next(workspace._root.rglob("own.md")).read_bytes() == b"retained"
        before = runtime.history_snapshot
        assert (await runtime.complete(request(2, "recover this visible note"))).text == "recovered"
        assert runtime.history_snapshot[:len(before)] == before
        assert_complete_pairs(runtime.working_snapshot)
    asyncio.run(run())


def test_archive_capacity_is_reserved_before_note_side_effect(tmp_path):
    from agent_ark.agent.simple_agent import FileWorkspace, ToolCall
    async def run():
        workspace = FileWorkspace(tmp_path)
        async def model(value):
            return ModelCompletion("", tool_calls=(ToolCall("write", "write_note",
                                    '{"path":"notes/own.md","content":"must not be written"}'),))
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_archive_entries=1), tools=workspace)
        with pytest.raises(AgentLimitError, match="before operation"):
            await runtime.complete(request())
        assert workspace._root is None
        assert len(runtime.history_snapshot) == 1
        assert not any(message.tool_calls for message in runtime.working_snapshot)
    asyncio.run(run())


def test_tool_result_survives_full_batch_archive_exhaustion(tmp_path):
    from agent_ark.agent.simple_agent import FileWorkspace, ToolCall
    async def run():
        workspace = FileWorkspace(tmp_path)
        async def model(value):
            return ModelCompletion("", tool_calls=(ToolCall("write", "write_note",
                                    '{"path":"notes/own.md","content":"retained"}'),))
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_archive_entries=2), tools=workspace)
        with pytest.raises(AgentLimitError, match="archive"):
            await runtime.complete(request())
        assert runtime.history_snapshot[-1].event_id == "tool-result-2"
        assert "retained" not in texts(runtime.working_snapshot)
        assert not any(message.tool_calls or message.role == "tool" for message in runtime.working_snapshot)
        assert next(workspace._root.rglob("own.md")).read_bytes() == b"retained"
    asyncio.run(run())


def native_submission_context(code, *, call_id="submit-1"):
    import json
    from agent_ark.agent.simple_agent import ToolCall
    call = ToolCall(call_id, "submit_action", json.dumps({"code": code}, ensure_ascii=False))
    return (Message("assistant", (), (call,)), Message("tool", (TextPart(
        "Adapter selected action; not executed yet; actual feedback follows"),), tool_call_id=call_id))


def test_optional_native_context_preserves_original_completion_constructor():
    from agent_ark.agent.simple_agent import ToolCall
    call = ToolCall("own", "read_file", "{}")
    usage = ModelUsage(1, 2, 3)
    ordinary = ModelCompletion("unchanged", usage, (call,))
    assert ordinary.text == "unchanged" and ordinary.usage == usage
    assert ordinary.tool_calls == (call,) and ordinary.context_messages == ()
    assert ModelCompletion("plain").context_messages == ()
    context = native_submission_context("complete source")
    derived = ModelCompletion("translated source", usage, (), context)
    assert derived.context_messages == context and derived.tool_calls == ()
    with pytest.raises(FrozenInstanceError):
        derived.context_messages = ()


@pytest.mark.parametrize("malformed", ["list", "short", "long", "system", "developer", "user",
                                          "no_call", "many_calls", "wrong_result", "wrong_id", "coexist"])
def test_native_completion_context_rejects_unclosed_or_untrusted_roles(malformed):
    from agent_ark.agent.simple_agent import ToolCall
    context = native_submission_context("source")
    assistant, result = context
    calls = ()
    if malformed == "list":
        context = list(context)
    elif malformed == "short":
        context = (assistant,)
    elif malformed == "long":
        context = (*context, Message.text("user", "extra observation"))
    elif malformed in ("system", "developer", "user"):
        context = (Message.text(malformed, "instruction injection"), result)
    elif malformed == "no_call":
        context = (Message.text("assistant", "no native call"), result)
    elif malformed == "many_calls":
        context = (Message("assistant", (), (*assistant.tool_calls, ToolCall("extra", "submit_action", "{}"))), result)
    elif malformed == "wrong_result":
        context = (assistant, Message.text("assistant", "ack"))
    elif malformed == "wrong_id":
        context = (assistant, Message("tool", (TextPart("ack"),), tool_call_id="other"))
    elif malformed == "coexist":
        calls = (ToolCall("other", "read_file", "{}"),)
    with pytest.raises(ValueError, match="completion context"):
        ModelCompletion("source", tool_calls=calls, context_messages=context)


def test_native_submission_context_is_retained_without_dispatch_or_task_xml_echo():
    async def run():
        calls = []
        task_action = '<tool_call>{"tool":"task.move","args":{"x":1}}</tool_call>'
        context = native_submission_context(task_action)
        async def model(value):
            calls.append(value)
            if len(calls) == 1:
                return ModelCompletion(task_action, ModelUsage(10, 4, 14), (), context)
            assert value.messages[-3:-1] == context
            assert all(message != Message.text("assistant", task_action) for message in value.messages)
            assert value.messages[-1] == request(2, "actual environment feedback").messages[-1]
            wire = chat_messages(value.messages)
            assert wire[-3]["tool_calls"][0]["function"]["name"] == "submit_action"
            assert wire[-2]["role"] == "tool" and wire[-2]["tool_call_id"] == "submit-1"
            return ModelCompletion("next final action")
        # There is no ToolRegistry: the recorded native call has no authority
        # to execute a host tool or ask for an additional model round.
        runtime = SimpleRuntime(CallableModel(model))
        assert (await runtime.complete(request())).text == task_action
        completion = runtime.history_snapshot[-1]
        assert completion.kind == "completion" and completion.text == task_action
        assert completion.messages == context
        assert runtime.working_snapshot[-2:] == context
        assert (await runtime.complete(request(2, "actual environment feedback"))).text == "next final action"
        assert runtime.usage.action_calls == 2
        assert runtime.history_snapshot[1] == completion
    asyncio.run(run())


def test_native_context_preserves_decoded_full_csharp_and_accounts_both_archive_payloads():
    import json
    async def run():
        code = 'public sealed class AgentAction : CodeAction { string value = "完整\\\\code"; }\n' * 350
        context = native_submission_context(code)
        async def model(value):
            return ModelCompletion(code, context_messages=context)
        runtime = SimpleRuntime(CallableModel(model))
        assert (await runtime.complete(request())).text == code
        original = runtime.history_snapshot[-1]
        assert original.text == code and original.messages == context
        decoded = json.loads(chat_messages(context)[0]["tool_calls"][0]["function"]["arguments"])
        assert decoded["code"] == code
        assert runtime.working_snapshot[-2:] == context
        characters, byte_count, image_bytes = runtime._archive_cost(original)
        assert characters >= len(code) + len(context[0].tool_calls[0].arguments_json)
        assert byte_count >= len(code.encode("utf-8")) + len(encoded_messages(context))
        assert image_bytes == 0
        raw_only_character_budget = len(code) + sum(len(p.text) for m in request().messages for p in m.parts) + 100
        char_limited = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_archive_characters=raw_only_character_budget))
        with pytest.raises(AgentLimitError, match="archive"):
            await char_limited.complete(request())
        raw_only_byte_budget = len(code.encode("utf-8")) + len(encoded_messages(request().messages)) + 1500
        byte_limited = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_archive_bytes=raw_only_byte_budget))
        with pytest.raises(AgentLimitError, match="archive"):
            await byte_limited.complete(request())
        assert len(char_limited.history_snapshot) == len(byte_limited.history_snapshot) == 1
    asyncio.run(run())


def test_summary_rejects_native_context_and_keeps_original_working_transaction():
    from agent_ark.agent.simple_agent import AgentError
    async def run():
        calls = []
        context = native_submission_context("A" * 280)
        async def model(value):
            calls.append(value)
            if value.purpose == "summary":
                assert not value.tools
                return ModelCompletion("must not become a summary", context_messages=context)
            return ModelCompletion("A" * 280, context_messages=context)
        runtime = SimpleRuntime(CallableModel(model), config())
        await runtime.complete(request(1, "x" * 400))
        await runtime.complete(request(2, "x" * 400))
        source = request(3, "x" * 400)
        runtime.accept_input(source)
        before = runtime.working_snapshot
        with pytest.raises(AgentError, match="summary model"):
            await runtime.complete(source)
        assert runtime.working_snapshot == before
        assert runtime.usage.action_calls == 2 and runtime.usage.summary_calls == 1
        assert all(entry.kind != "summary_completion" for entry in runtime.history_snapshot)
        assert all(message in runtime.working_snapshot for message in context)
    asyncio.run(run())


def test_native_submission_pairs_are_plain_records_in_summary_model_requests():
    async def run():
        summaries = []
        rounds = 0
        async def model(value):
            nonlocal rounds
            pending = set()
            for message in value.messages:
                if message.tool_calls:
                    assert not pending
                    pending.update(call.call_id for call in message.tool_calls)
                elif message.role == "tool":
                    assert message.tool_call_id in pending
                    pending.remove(message.tool_call_id)
                else:
                    assert not pending
            assert not pending
            if value.purpose == "summary":
                assert value.tools == ()
                assert not any(message.tool_calls or message.role == "tool" for message in value.messages)
                summaries.append(value)
                return ModelCompletion("Earlier native submissions and actual feedback retained.")
            rounds += 1
            code = "action " + "A" * 280
            return ModelCompletion(code, context_messages=native_submission_context(code, call_id=str(rounds)))
        runtime = SimpleRuntime(CallableModel(model), config())
        for number in range(1, 5):
            await runtime.complete(request(number, "observation " + "x" * 360))
        assert summaries and "submit_action" in texts(summaries[0].messages)
        assert "Exact original arguments JSON follows:" in texts(summaries[0].messages)
        assert "Model notes; not environment facts" in texts(runtime.working_snapshot)
        originals = [entry for entry in runtime.history_snapshot if entry.kind == "completion"]
        assert len(originals) == 4 and all(len(entry.messages) == 2 for entry in originals)
        assert all(entry.text == "action " + "A" * 280 for entry in originals)
    asyncio.run(run())


def test_summary_transcript_preserves_exact_calls_reply_parts_images_and_source_order():
    import json
    from agent_ark.agent.simple_agent import ToolCall
    from agent_ark.agent.simple_agent.runtime import _WorkingEntry
    first_image = image_from_pixels(bytes([255, 0, 0]), 1, 1)
    second_image = image_from_pixels(bytes([0, 0, 255]), 1, 1)
    first = ToolCall('call-"one', "look", '{ "tag" : "original\\nspacing" }')
    second = ToolCall("call-two", "read_file", '{"path":"notes/own.md"}')
    assistant = Message("assistant", (TextPart("original assistant commentary"), first_image), (first, second))
    reply = Message("tool", (TextPart('  EXACT RESULT "quoted" \\n\n'), second_image,
                              TextPart("original reply tail")), tool_call_id=first.call_id)
    next_reply = Message("tool", (TextPart("second exact reply"),), tool_call_id=second.call_id)
    native = native_submission_context("complete C# source", call_id="submit-native")
    observation = Message.text("user", "original observation")
    feedback = Message.text("user", "actual environment feedback after selection")
    entries = (
        _WorkingEntry(1, "input", identity(), (observation,), (1,)),
        _WorkingEntry(2, "tool_exchange", identity(), (assistant, reply, next_reply), (2,)),
        _WorkingEntry(3, "completion", identity(), native, (3,)),
        _WorkingEntry(4, "event", identity(), (feedback,), (4,)),
    )
    original = tuple(entry.messages for entry in entries)
    summary = SimpleRuntime._summary_messages(entries)
    assert not any(message.tool_calls or message.role == "tool" for message in summary)
    assert summary[1] is observation and summary[-1] is feedback
    assert summary[2].role == summary[3].role == summary[4].role == "user"
    assert summary[2].parts[1:3] == assistant.parts
    assert summary[3].parts[1:] == reply.parts
    assert summary[4].parts[1:] == next_reply.parts
    assert TextPart(first.arguments_json) in summary[2].parts
    assert TextPart(second.arguments_json) in summary[2].parts
    metadata = [part.text.split("\n", 1)[0].removeprefix("Recorded call metadata: ")
                for part in summary[2].parts if isinstance(part, TextPart) and part.text.startswith("Recorded call metadata: ")]
    assert [json.loads(value) for value in metadata] == [
        {"call_id": first.call_id, "name": first.name}, {"call_id": second.call_id, "name": second.name}]
    assert "Source archive entries: [2]" in summary[2].parts[0].text
    assert "Source archive entries: [3]" in summary[5].parts[0].text
    images = [part for message in summary for part in message.parts if isinstance(part, ImagePart)]
    assert len(images) == 2 and images[0] is first_image and images[1] is second_image
    assert tuple(entry.messages for entry in entries) == original
    wire = chat_messages(summary)
    assert all(message["role"] != "tool" and "tool_calls" not in message for message in wire)


def test_actual_summary_model_receives_plain_tool_records_and_original_image_payload(tmp_path):
    from agent_ark.agent.simple_agent import ToolCall, ToolRegistry, ToolSpec
    async def run():
        image = image_from_pixels(bytes([255, 125, 0]), 1, 1)
        raw_reply = '  {"observed":"orange","literal":"line\\nquote\\\""}\nEXACT RESULT TAIL'
        arguments = ' {"tag" : "orange\\nframe"} '
        tools = ToolRegistry()
        tools.register(ToolSpec("look", "Read this synthetic own image.",
                                '{"type":"object","properties":{"tag":{"type":"string"}},"required":["tag"],"additionalProperties":false}'),
                       lambda args, session: (TextPart(raw_reply), image))
        rounds = 0
        summaries = []
        code = "public sealed class AgentAction : CodeAction { /* 完整 */ }" * 4
        async def model(value):
            nonlocal rounds
            if value.purpose == "summary":
                summaries.append(value)
                assert value.tools == ()
                assert not any(message.tool_calls or message.role == "tool" for message in value.messages)
                assert any(part == TextPart(arguments) for message in value.messages for part in message.parts)
                assert any(part == TextPart(raw_reply) for message in value.messages for part in message.parts)
                assert any(part is image for message in value.messages for part in message.parts)
                assert "submit_action" in texts(value.messages)
                assert "not executed yet" in texts(value.messages)
                return ModelCompletion("Earlier observed orange frames and selected source were recorded.")
            rounds += 1
            if rounds % 2:
                return ModelCompletion("", tool_calls=(ToolCall(str(rounds), "look", arguments),))
            return ModelCompletion(code, context_messages=native_submission_context(code, call_id=str(rounds)))
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=10_000, max_images=2), tools=tools)
        for number in range(1, 5):
            assert (await runtime.complete(request(number, "visible observation " + "x" * 100))).text == code
        assert summaries
        exchanges = [entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange"]
        completions = [entry for entry in runtime.history_snapshot if entry.kind == "completion"]
        assert len(exchanges) == len(completions) == 4
        assert all(entry.messages[0].tool_calls[0].arguments_json == arguments for entry in exchanges)
        assert all(entry.messages[1].parts == (TextPart(raw_reply), image) for entry in exchanges)
        assert all(entry.text == code and entry.messages[0].tool_calls for entry in completions)
    asyncio.run(run())


@pytest.mark.parametrize("bound", ["characters", "bytes", "tokens"])
def test_transformed_summary_budget_keeps_complete_native_turn_and_rolls_back(bound):
    async def run():
        from agent_ark.agent.simple_agent.runtime import _metrics
        image = image_from_pixels(bytes([2, 4, 6]), 1, 1)
        native = native_submission_context("A" * 180)
        context = (native[0], Message("tool", (TextPart("exact original acknowledgement"), image,
                                              TextPart("exact trailing reply")), tool_call_id="submit-1"))
        requests = []
        async def model(value):
            requests.append(value)
            assert value.purpose == "action"
            return ModelCompletion("A" * 180, context_messages=context)
        first = request(1, "x" * 260)
        probe = SimpleRuntime(CallableModel(model), config(max_context_characters=700))
        await probe.complete(first)
        transformed = SimpleRuntime._summary_messages(tuple(probe._working))
        raw = (transformed[0],) + tuple(message for entry in probe._working for message in entry.messages)
        transformed_characters, transformed_bytes, _, _ = _metrics(transformed)
        raw_characters, raw_bytes, _, _ = _metrics(raw)
        assert raw_characters < transformed_characters - 1
        assert raw_bytes < transformed_bytes - 1
        updates = {"summary_input_" + bound: (transformed_characters if bound == "characters" else transformed_bytes) - 1}
        counter = (lambda messages: len(encoded_messages(messages))) if bound == "tokens" else None
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=700, **updates), token_counter=counter)
        await runtime.complete(first)
        next_input = request(2, "y" * 260)
        runtime.accept_input(next_input)
        before = runtime.working_snapshot
        originals = runtime.history_snapshot
        with pytest.raises(AgentLimitError, match="cannot fit"):
            await runtime.complete(next_input)
        assert runtime.working_snapshot == before
        assert runtime.history_snapshot == originals
        assert runtime.usage.action_calls == 1 and runtime.usage.summary_calls == 0
        assert runtime.history_snapshot[1].messages == context
        assert runtime.working_snapshot[-3:-1] == context
        assert not any(value.purpose == "summary" for value in requests)
    asyncio.run(run())
