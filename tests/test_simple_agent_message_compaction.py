"""Message-chunk compaction regressions using immutable in-memory inputs only."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json

import pytest

from agent_ark.agent.simple_agent import (
    AgentError, AgentEvent, AgentInputIdentity, AgentLimitError, AgentRequest,
    CallableModel, ImagePart, Message, ModelCompletion, ModelUsage,
    SimpleRuntime, SimpleRuntimeConfig, TextPart, ToolCall, ToolRegistry, ToolSpec,
)
from agent_ark.agent.simple_agent.encoding import encoded_messages
from agent_ark.agent.simple_agent.runtime import _WorkingEntry
from agent_ark.agent.simple_agent.types import ArchiveEntry, NativeAssistantState


USAGE = ModelUsage(3, 5, 8)


def request(number, *, text="observation", images=(), parts=None):
    if isinstance(images, int):
        images = tuple(ImagePart(f"image-{number}-{index:03}".encode()) for index in range(images))
    return AgentRequest(
        AgentInputIdentity("run", "session", "episode", "agent", 0, str(number)),
        (Message.text("system", "PINNED_INSTRUCTIONS_MUST_REMAIN"),
         Message("user", parts if parts is not None else (TextPart(text), *images))),
    )


def texts(messages):
    return "\n".join(part.text for message in messages for part in message.parts if isinstance(part, TextPart))


def image_parts(messages):
    return [part for message in messages for part in message.parts if isinstance(part, ImagePart)]


def metadata(value):
    assert value.compaction_json is not None
    result = json.loads(value.compaction_json)
    assert result["version"] == 1
    assert result["policy"] == "message_chunks_v1"
    assert len(value.compaction_json.encode("utf-8")) <= 64 * 1024
    return result


def chunks(calls, stage="leaf"):
    return [call for call in calls if call.purpose == "summary" and metadata(call)["stage"] == stage]


def assert_summary_bounds(runtime, calls):
    for call in calls:
        if call.purpose == "summary":
            assert runtime._fits(call.messages, summary=True)
            assert call.input_tokens <= runtime.config.summary_input_tokens
            assert call.max_output_tokens == runtime.config.summary_output_tokens
            assert not call.tools
            assert all(message.native_assistant is None for message in call.messages)


def recorder(calls, *, summary="Concise earlier facts.", action="action"):
    async def model(value):
        calls.append(value)
        return ModelCompletion(summary if value.purpose == "summary" else action, USAGE)
    return CallableModel(model)


def image_pressure(model, **config):
    """80 old images need two default-sized leaves and a merge."""
    runtime = SimpleRuntime(model, SimpleRuntimeConfig(recent_turns=1, max_images=100, **config))
    old = request(1, images=80)
    current = request(2, images=30)
    runtime.accept_input(old)
    runtime.accept_input(current)
    return runtime, old, current


def native(text="answer"):
    return NativeAssistantState("offline:message-chunks", json.dumps({
        "role": "assistant", "content": text, "reasoning": "PRIVATE_REPLAY_MUST_STAY_ARCHIVED",
    }))


def test_default_video_frames_now_fit_without_changing_summary_budgets_or_original_images():
    async def run():
        calls = []
        runtime = SimpleRuntime(recorder(calls))
        cfg = runtime.config
        assert cfg.compaction_policy == "message_chunks_v1"
        assert (cfg.summary_input_tokens, cfg.max_summary_calls) == (48_000, 4)
        originals = []
        for number, count in enumerate((1, 83, 81, 77)):
            value = request(number, images=count)
            originals.append(value)
            assert (await runtime.complete(value)).text == "action"
        assert runtime.config == cfg
        assert runtime.usage.action_calls == 4
        assert 3 <= runtime.usage.summary_calls <= 4
        leaves, merges = chunks(calls), chunks(calls, "merge")
        assert 2 <= len(leaves) <= 3 and len(merges) == 1
        source_calls = [call for call in calls if call.purpose == "summary" and metadata(call)["stage"] != "merge"]
        summarized = image_parts(tuple(message for call in source_calls for message in call.messages))
        assert summarized == [part for value in originals[:2] for part in image_parts(value.messages)]
        assert all(actual is original for actual, original in zip(summarized,
            [part for value in originals[:2] for part in image_parts(value.messages)]))
        assert image_parts(calls[-1].messages) == image_parts(originals[2].messages) + image_parts(originals[3].messages)
        assert [entry.messages for entry in runtime.history_snapshot if entry.kind == "input"] == [value.messages for value in originals]
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_eighty_images_use_ordered_whole_images_and_one_shared_leaf_merge_plan():
    async def run():
        calls = []
        runtime, old, current = image_pressure(recorder(calls))
        original_archive = runtime.history_snapshot
        assert (await runtime.complete(current)).text == "action"
        leaves, merges = chunks(calls), chunks(calls, "merge")
        assert len(leaves) in (2, 3) and len(merges) == 1
        all_meta = [metadata(call) for call in (*leaves, *merges)]
        assert len({value["plan_id"] for value in all_meta}) == 1
        assert [metadata(call)["leaf_index"] for call in leaves] == list(range(len(leaves)))
        assert all(metadata(call)["leaf_count"] == len(leaves) for call in leaves)
        observed = [part for call in leaves for part in image_parts(call.messages)]
        assert observed == image_parts(old.messages)
        assert all(a is b for a, b in zip(observed, image_parts(old.messages)))
        source_images = [fragment for call in leaves for fragment in metadata(call)["fragments"]
                         if fragment["kind"] == "image"]
        assert len(source_images) == 80
        for index, fragment in enumerate(source_images):
            assert fragment["archive_sequence"] == 1 and fragment["message_index"] == 1
            assert fragment["part_index"] == index + 1 and fragment["image_whole"] is True
            assert fragment["sha256"] == hashlib.sha256(observed[index].data).hexdigest()
        assert runtime.history_snapshot[:len(original_archive)] == original_archive
        audit = [entry for entry in runtime.history_snapshot if entry.kind == "summary_completion"]
        assert [metadata(entry) for entry in audit] == [metadata(call) for call in calls if call.purpose == "summary"]
        assert all(not entry.messages for entry in audit)
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_multibyte_single_text_part_has_exact_unicode_range_coverage_and_hashes():
    async def run():
        calls = []
        source = "汉🙂e\u0301" * 700
        old, current = request(1, text=source), request(2, text="CURRENT_VISIBLE")
        runtime = SimpleRuntime(recorder(calls), SimpleRuntimeConfig(
            recent_turns=1, max_context_characters=1000,
            summary_input_characters=2300, summary_input_bytes=7000,
            summary_output_tokens=256))
        runtime.accept_input(old)
        runtime.accept_input(current)
        original = runtime.history_snapshot
        await runtime.complete(current)
        leaves = chunks(calls)
        assert 2 <= len(leaves) <= 3
        fragments = [(call, fragment) for call in leaves for fragment in metadata(call)["fragments"]
                     if fragment["archive_sequence"] == 1 and fragment["kind"] == "text"]
        cursor = 0
        for call, fragment in fragments:
            assert fragment["message_index"] == 1 and fragment["part_index"] == 0
            assert fragment["text_start"] == cursor
            end = fragment["text_end"]
            assert cursor < end <= len(source)
            piece = source[cursor:end]
            assert fragment["sha256"] == hashlib.sha256(piece.encode("utf-8")).hexdigest()
            assert any(isinstance(part, TextPart) and part.text == piece for message in call.messages for part in message.parts)
            cursor = end
        assert cursor == len(source)
        assert runtime.history_snapshot[:len(original)] == original
        assert runtime.history_snapshot[0].messages == old.messages
        assert source not in texts(calls[-1].messages)
        assert "CURRENT_VISIBLE" in texts(calls[-1].messages)
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_interleaved_text_and_images_keep_original_part_order_and_immutable_bytes():
    async def run():
        calls = []
        pictures = (ImagePart(b"FIRST-WHOLE-IMAGE"), ImagePart(b"SECOND-WHOLE-IMAGE"))
        parts = (TextPart("before " + "汉" * 1300), pictures[0], TextPart("between " + "🙂" * 1300),
                 pictures[1], TextPart("after " + "z" * 900))
        old, current = request(1, parts=parts), request(2, text="current")
        runtime = SimpleRuntime(recorder(calls), SimpleRuntimeConfig(recent_turns=1,
            max_context_characters=1000, summary_input_characters=2600,
            summary_input_bytes=9000, summary_output_tokens=256))
        runtime.accept_input(old)
        runtime.accept_input(current)
        await runtime.complete(current)
        sequence = []
        ranges = {index: [] for index, part in enumerate(parts) if isinstance(part, TextPart)}
        for call in chunks(calls):
            expected_payload = []
            for fragment in metadata(call)["fragments"]:
                if fragment["archive_sequence"] != 1:
                    continue
                index = fragment["part_index"]
                assert fragment["message_index"] == 1
                sequence.append(index)
                original = parts[index]
                if isinstance(original, ImagePart):
                    assert fragment["kind"] == "image" and fragment["image_whole"] is True
                    expected_payload.append(original)
                    assert fragment["sha256"] == hashlib.sha256(original.data).hexdigest()
                else:
                    start, end = fragment["text_start"], fragment["text_end"]
                    ranges[index].append((start, end))
                    expected_payload.append(TextPart(original.text[start:end]))
            # Source pieces must remain a subsequence of the summary's labelled parts.
            actual = iter(part for message in call.messages for part in message.parts)
            for expected in expected_payload:
                match = next((part for part in actual if part == expected), None)
                assert match is not None
                if isinstance(expected, ImagePart):
                    assert match is expected
        assert sequence == sorted(sequence)
        assert [part for call in chunks(calls) for part in image_parts(call.messages)] == list(pictures)
        for index, spans in ranges.items():
            assert spans[0][0] == 0 and spans[-1][1] == len(parts[index].text)
            assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
        assert runtime.history_snapshot[0].messages[1].parts is parts
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_each_video_leaf_keeps_the_frame_caption_with_its_associated_image():
    async def run():
        calls = []
        pictures = tuple(ImagePart(f"caption-frame-{index}".encode()) for index in range(5))
        captions = tuple(f"Frame {index} @ time {index * 0.25:.2f}s: object position {index}." for index in range(5))
        parts = tuple(part for caption, picture in zip(captions, pictures) for part in (TextPart(caption), picture))
        old = request(1, parts=parts)
        runtime = SimpleRuntime(recorder(calls), SimpleRuntimeConfig(recent_turns=1,
            max_images=5, summary_input_tokens=2500, summary_output_tokens=256))
        runtime.accept_input(old)
        # A sixth current image creates pressure without lowering the action
        # token budget or the five-frame historical sequence's image allowance.
        current = request(2, text="CURRENT_FRAME_KEEP", images=1)
        runtime.accept_input(current)
        await runtime.complete(current)
        leaves = chunks(calls)
        assert len(leaves) == 3 and len(chunks(calls, "merge")) == 1
        seen = []
        for call in leaves:
            body = texts(call.messages)
            for image in image_parts(call.messages):
                index = next(index for index, original in enumerate(pictures) if image is original)
                # Provenance IDs alone do not carry a frame's temporal meaning.
                # Every leaf must expose its actual caption or repeated context.
                assert captions[index] in body
                seen.append(index)
        assert seen == list(range(5))
        assert "CURRENT_FRAME_KEEP" in texts(calls[-1].messages)
        assert image_parts(calls[-1].messages) == image_parts(current.messages)
        assert runtime.history_snapshot[0].messages == old.messages
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_small_tail_in_a_second_text_part_does_not_block_an_overall_reducing_plan():
    async def run():
        calls = []
        old = request(1, parts=(TextPart("x" * 1200), TextPart("tail")))
        current = request(2, text="current")
        runtime = SimpleRuntime(recorder(calls, summary="Concise useful facts."), SimpleRuntimeConfig(
            recent_turns=1, summary_input_characters=1501, max_context_characters=600,
            summary_output_tokens=256))
        runtime.accept_input(old)
        runtime.accept_input(current)
        original = runtime.history_snapshot
        assert (await runtime.complete(current)).text == "action"
        assert runtime.usage.summary_calls <= 4
        fragments = [fragment for call in chunks(calls) for fragment in metadata(call)["fragments"]
                     if fragment["archive_sequence"] == 1]
        for part_index, source in enumerate(("x" * 1200, "tail")):
            spans = [(value["text_start"], value["text_end"]) for value in fragments
                     if value["part_index"] == part_index]
            assert spans and spans[0][0] == 0 and spans[-1][1] == len(source)
            assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
        assert "Concise useful facts." in texts(calls[-1].messages)
        assert runtime.history_snapshot[:len(original)] == original
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_many_small_turns_select_bounded_provenance_prefixes_and_keep_every_source():
    async def run():
        calls = []
        runtime = SimpleRuntime(recorder(calls), SimpleRuntimeConfig(recent_turns=1,
            max_context_characters=1000, summary_output_tokens=256))
        inputs = [request(number, text=f"TURN_{number:04}") for number in range(600)]
        for value in inputs:
            runtime.accept_input(value)
        original = runtime.history_snapshot
        assert (await runtime.complete(inputs[-1])).text == "action"
        summaries = [call for call in calls if call.purpose == "summary"]
        assert 1 <= len(summaries) <= 4
        summarized_sequences = {fragment["archive_sequence"] for call in summaries
                                for fragment in metadata(call)["fragments"]}
        retained_parts = {part.text for message in calls[-1].messages for part in message.parts if isinstance(part, TextPart)}
        for entry, value in zip(original, inputs):
            assert entry.sequence in summarized_sequences or value.messages[-1].parts[0].text in retained_parts
        assert 600 not in summarized_sequences
        assert "TURN_0599" in texts(calls[-1].messages)
        assert runtime.history_snapshot[:len(original)] == original
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_merge_reduction_uses_original_prefix_with_many_short_turns_and_large_video_turn():
    async def run():
        calls = []
        runtime = SimpleRuntime(recorder(calls, summary="Concise useful facts."),
                                SimpleRuntimeConfig(recent_turns=1, max_images=100))
        for number in range(100):
            runtime.accept_input(request(number, text=f"short old {number}"))
        old_video = request(100, images=80)
        current = request(101, images=30)
        runtime.accept_input(old_video)
        runtime.accept_input(current)
        archive, before = runtime.history_snapshot, runtime.working_snapshot
        assert (await runtime.complete(current)).text == "action"
        summaries = [call for call in calls if call.purpose == "summary"]
        assert [metadata(call)["stage"] for call in summaries] == ["leaf", "leaf", "merge"]
        assert runtime.usage.summary_calls == 3 and runtime.usage.action_calls == 1
        assert image_parts(calls[-1].messages) == image_parts(current.messages)
        assert all(a is b for a, b in zip(image_parts(calls[-1].messages), image_parts(current.messages)))
        assert runtime.history_snapshot[:len(archive)] == archive
        assert len(runtime.working_snapshot) < len(before)
        assert len(encoded_messages(calls[-1].messages)) < len(encoded_messages(before))
        assert "Concise useful facts." in texts(calls[-1].messages)
        leaf_sources = {fragment["archive_sequence"] for call in summaries[:2]
                        for fragment in metadata(call)["fragments"]}
        assert leaf_sources == set(range(1, 102))
        audit = [entry for entry in runtime.history_snapshot if entry.kind == "summary_completion"]
        assert [metadata(entry) for entry in audit] == [metadata(call) for call in summaries]
        assert {fragment["archive_sequence"] for fragment in metadata(summaries[-1])["fragments"]} == {
            entry.sequence for entry in audit[:2]}
        assert runtime._fits(calls[-1].messages)
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_one_indivisible_image_over_summary_byte_limit_fails_before_any_model_call():
    async def run():
        calls = []
        runtime = SimpleRuntime(recorder(calls), SimpleRuntimeConfig(recent_turns=1,
            max_images=1, summary_input_bytes=1500))
        runtime.accept_input(request(1, images=(ImagePart(b"x" * 3000),)))
        current = request(2, images=1)
        runtime.accept_input(current)
        before, archive = runtime.working_snapshot, runtime.history_snapshot
        with pytest.raises(AgentLimitError, match="(?i)(indivisible|image|fragment).*(summary|bound|fit)|summary.*(image|fragment)"):
            await runtime.complete(current)
        assert not calls and runtime.usage.summary_calls == runtime.usage.action_calls == 0
        assert runtime.working_snapshot == before and runtime.history_snapshot == archive
    asyncio.run(run())


def test_custom_nonmonotonic_counter_cannot_make_one_image_a_false_impossibility_proof():
    async def run():
        calls = []

        def counter(messages):
            count = len(image_parts(messages))
            # Rejecting a one-image candidate does not prove that every larger
            # source chunk fails; this is intentionally not a monotone counter.
            return 999_999 if count == 1 else count * 1024

        runtime = SimpleRuntime(recorder(calls), SimpleRuntimeConfig(recent_turns=1, max_images=100),
                                token_counter=counter)
        old, current = request(1, images=80), request(2, images=30)
        runtime.accept_input(old)
        assert (await runtime.complete(current)).text == "action"
        assert runtime.usage.summary_calls == 3
        assert [part for call in chunks(calls) for part in image_parts(call.messages)] == image_parts(old.messages)
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


async def completed_tool_batch(calls, *, input_characters=20, result_characters=700):
    registry = ToolRegistry()
    registry.register(ToolSpec("echo", "Return only visible own text.", '{"type":"object"}'),
        lambda args, session: (TextPart(f"RESULT_{args['index']}_" + "r" * result_characters),))
    batch = tuple(ToolCall(f"opaque id {index}/+", "echo", json.dumps({"index": index})) for index in range(3))
    replies = iter((ModelCompletion("BATCH_ASSISTANT", USAGE, batch), ModelCompletion("old action", USAGE)))

    async def model(value):
        calls.append(value)
        if value.purpose == "summary":
            return ModelCompletion("Closed batch facts retained.", USAGE)
        return next(replies, ModelCompletion("new action", USAGE))

    runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
        recent_turns=1, max_context_characters=100_000, output_reserve_tokens=64), tools=registry)
    await runtime.complete(request(1, text="O" * input_characters))
    return runtime, batch


def test_oversized_closed_tool_batch_is_indivisible_and_rejected_without_summary_calls():
    async def run():
        calls = []
        runtime, _ = await completed_tool_batch(calls)
        current = request(2, text="current")
        runtime.accept_input(current)
        runtime.config = replace(runtime.config, max_context_characters=1000,
                                 summary_input_characters=2100, summary_output_tokens=256)
        before, archive = runtime.working_snapshot, runtime.history_snapshot
        with pytest.raises(AgentLimitError, match="(?i)(tool|batch|group).*(summary|bound|fit)|summary.*(tool|batch|group)"):
            await runtime.complete(current)
        assert all(call.purpose == "action" for call in calls)
        assert runtime.usage.summary_calls == 0
        assert runtime.working_snapshot == before and runtime.history_snapshot == archive
    asyncio.run(run())


def test_closed_multi_result_batch_never_sends_orphan_calls_or_replies_to_a_leaf():
    async def run():
        calls = []
        runtime, batch = await completed_tool_batch(calls, input_characters=4000, result_characters=250)
        original = runtime.history_snapshot
        current = request(2, text="current")
        runtime.accept_input(current)
        runtime.config = replace(runtime.config, max_context_characters=1000,
                                 summary_input_characters=3000, summary_output_tokens=256)
        assert (await runtime.complete(current)).text == "new action"
        leaves = chunks(calls)
        group_leaves = [call for call in leaves if "BATCH_ASSISTANT" in texts(call.messages)]
        assert len(group_leaves) == 1
        closed = texts(group_leaves[0].messages)
        for index, call in enumerate(batch):
            assert call.call_id in closed and call.arguments_json in closed
            assert f"RESULT_{index}_" + "r" * 250 in closed
        assert all(not message.tool_calls and message.role != "tool" for call in leaves for message in call.messages)
        assert all(not any(f"RESULT_{index}_" in texts(call.messages) for index in range(3))
                   for call in leaves if call is not group_leaves[0])
        assert runtime.history_snapshot[:len(original)] == original
        archived_group = next(entry for entry in original if entry.kind == "tool_exchange")
        assert archived_group.messages[0].tool_calls == batch
        assert [message.tool_call_id for message in archived_group.messages[1:]] == [call.call_id for call in batch]
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


@pytest.mark.parametrize("failure_stage", [2, 3])
@pytest.mark.parametrize("failure", ["provider", "cancel", "invalid"])
def test_later_leaf_or_merge_failure_rolls_back_every_staged_change_but_keeps_audit(failure_stage, failure):
    async def run():
        calls = []
        seen_summary = 0

        async def model(value):
            nonlocal seen_summary
            calls.append(value)
            assert value.purpose == "summary"
            seen_summary += 1
            if seen_summary == failure_stage:
                if failure == "provider":
                    raise RuntimeError("offline provider failure")
                if failure == "cancel":
                    raise asyncio.CancelledError()
                return ModelCompletion("invalid", USAGE, native_assistant=native("invalid"))
            return ModelCompletion("Earlier facts retained.", USAGE)

        runtime, _, current = image_pressure(CallableModel(model))
        before, archive = runtime.working_snapshot, runtime.history_snapshot
        expected = RuntimeError if failure == "provider" else asyncio.CancelledError if failure == "cancel" else AgentError
        with pytest.raises(expected):
            await runtime.complete(current)
        assert runtime.working_snapshot == before
        assert runtime.history_snapshot[:len(archive)] == archive
        assert runtime.usage.summary_calls == failure_stage and runtime.usage.action_calls == 0
        success = [entry for entry in runtime.history_snapshot if entry.kind == "summary_completion"]
        returned = failure_stage if failure == "invalid" else failure_stage - 1
        assert len(success) == returned
        assert runtime.usage.summary_usage == (USAGE,) * returned
        assert [metadata(entry)["stage"] for entry in success[:failure_stage - 1]] == ["leaf"] * (failure_stage - 1)
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_repeating_the_same_input_after_partial_failure_cannot_reset_four_call_allowance():
    async def run():
        calls = []

        async def model(value):
            calls.append(value)
            assert value.purpose == "summary"
            if len(calls) == 2:
                raise RuntimeError("second leaf failed")
            return ModelCompletion("Earlier facts retained.", USAGE)

        runtime, _, current = image_pressure(CallableModel(model))
        before = runtime.working_snapshot
        with pytest.raises(RuntimeError, match="second leaf"):
            await runtime.complete(current)
        with pytest.raises(AgentLimitError, match="summary.*(bound|budget|allowance)"):
            await runtime.complete(current)
        # A new complete plan may be rejected before billing if it needs more
        # than the allowance remaining from this identity's earlier failure.
        assert 2 <= len(calls) == runtime.usage.summary_calls <= 4
        assert runtime.working_snapshot == before and runtime.usage.action_calls == 0
        billed = len(calls)
        with pytest.raises(AgentLimitError):
            await runtime.complete(current)
        assert len(calls) == billed
        assert len([entry for entry in runtime.history_snapshot if entry.kind == "summary_completion"]) == billed - 1
    asyncio.run(run())


def test_recoverable_second_leaf_response_and_merge_share_existing_four_calls():
    async def run():
        calls = []

        async def model(value):
            calls.append(value)
            if value.purpose == "action":
                return ModelCompletion("action", USAGE)
            return ModelCompletion("" if len(calls) == 2 else "Earlier facts retained.", USAGE)

        runtime, _, current = image_pressure(CallableModel(model))
        assert (await runtime.complete(current)).text == "action"
        assert runtime.usage.summary_calls == 4 and runtime.usage.action_calls == 1
        assert [metadata(call)["stage"] for call in calls if call.purpose == "summary"] == ["leaf", "leaf", "leaf", "merge"]
        assert "Summary retry instructions:" in texts(calls[2].messages)
        assert "Summary retry instructions:" not in texts(calls[-1].messages)
        assert runtime.usage.summary_usage == (USAGE,) * 4
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_one_deadline_covers_all_leaves_and_merge_including_provider_swallowing_cancellation():
    async def run():
        calls = []

        async def model(value):
            calls.append(value)
            assert value.purpose == "summary"
            try:
                # Each individual response fits the deadline; their sum does not.
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                return ModelCompletion("Late response cannot commit.", USAGE)
            return ModelCompletion("Earlier facts retained.", USAGE)

        runtime, _, current = image_pressure(CallableModel(model), timeout_seconds=0.16)
        before, archive = runtime.working_snapshot, runtime.history_snapshot
        with pytest.raises(AgentLimitError, match="decision time budget exhausted"):
            await runtime.complete(current)
        assert len(calls) == runtime.usage.summary_calls == 2
        assert runtime.usage.action_calls == 0 and runtime.working_snapshot == before
        assert runtime.history_snapshot[:len(archive)] == archive
        assert len([entry for entry in runtime.history_snapshot if entry.kind == "summary_completion"]) == 1
    asyncio.run(run())


def test_successful_leaf_responses_do_not_commit_when_final_notes_cannot_fit_action_context():
    async def run():
        calls = []
        runtime, _, current = image_pressure(recorder(calls), max_context_characters=80)
        before, archive = runtime.working_snapshot, runtime.history_snapshot
        with pytest.raises(AgentLimitError):
            await runtime.complete(current)
        assert runtime.working_snapshot == before
        assert runtime.history_snapshot[:len(archive)] == archive
        assert not any(call.purpose == "action" for call in calls)
        assert runtime.usage.summary_calls <= 4
        assert runtime.usage.action_calls == 0
        assert all(entry.kind == "summary_completion" for entry in runtime.history_snapshot[len(archive):])
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_pins_current_recent_and_required_feedback_survive_compaction_and_native_state_stays_archived():
    async def run():
        calls = []

        async def model(value):
            calls.append(value)
            if value.purpose == "summary":
                return ModelCompletion("Earlier useful facts.", USAGE)
            return ModelCompletion("answer", USAGE, native_assistant=native())

        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_context_characters=50_000,
            recent_turns=2, summary_input_characters=3000, summary_output_tokens=256))
        old, recent = request(1, text="OLD_SOURCE_" + "o" * 4000), request(2, text="RECENT_KEEP_" + "r" * 100)
        await runtime.complete(old)
        await runtime.complete(recent)
        runtime.observe_result(AgentEvent(old.identity, "required", "action.result", "REQUIRED_FEEDBACK_KEEP"))
        current = request(3, text="CURRENT_KEEP_" + "c" * 100)
        runtime.accept_input(current)
        archive = runtime.history_snapshot
        runtime.config = replace(runtime.config, max_context_characters=1500)
        await runtime.complete(current)
        leaves = chunks(calls)
        assert leaves
        for call in leaves:
            body = texts(call.messages)
            assert "PINNED_INSTRUCTIONS_MUST_REMAIN" not in body
            assert "RECENT_KEEP_" not in body and "CURRENT_KEEP_" not in body
            assert "REQUIRED_FEEDBACK_KEEP" not in body
            assert "PRIVATE_REPLAY_MUST_STAY_ARCHIVED" not in body
        final = calls[-1]
        assert final.purpose == "action"
        for marker in ("PINNED_INSTRUCTIONS_MUST_REMAIN", "RECENT_KEEP_", "CURRENT_KEEP_", "REQUIRED_FEEDBACK_KEEP"):
            assert marker in texts(final.messages)
        assert all(message.native_assistant is None for message in final.messages)
        assert runtime.history_snapshot[:len(archive)] == archive
        assert sum(message.native_assistant is not None for entry in archive for message in entry.messages) == 2
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())


def test_late_completion_after_protected_interleaved_input_does_not_allow_half_turn_summary():
    async def run():
        calls = []
        runtime = SimpleRuntime(recorder(calls), SimpleRuntimeConfig(recent_turns=2,
            max_context_characters=1000, summary_input_characters=2500, summary_output_tokens=256))
        first, second = request("A", text="A_" + "a" * 4000), request("B", text="B_PROTECTED")
        runtime.accept_input(first)
        runtime.accept_input(second)
        late = Message.text("assistant", "LATE_A_COMPLETION_" + "l" * 300)
        runtime._append_archive(ArchiveEntry(3, "completion", first.identity, (late,), text=late.parts[0].text))
        runtime._working.append(_WorkingEntry(3, "completion", first.identity, (late,), (3,)))
        current = request("C", text="CURRENT_PROTECTED")
        runtime.accept_input(current)
        before, archive = runtime.working_snapshot, runtime.history_snapshot
        with pytest.raises(AgentLimitError):
            await runtime.complete(current)
        assert runtime.working_snapshot == before and runtime.history_snapshot == archive
        assert not calls
    asyncio.run(run())


def test_eligible_interleaved_inputs_and_late_completion_are_summarized_as_one_closed_prefix():
    async def run():
        calls = []
        runtime = SimpleRuntime(recorder(calls), SimpleRuntimeConfig(recent_turns=1,
            max_context_characters=1000, summary_input_characters=3500, summary_output_tokens=256))
        first, second = request("A", text="A_" + "a" * 4000), request("B", text="B_" + "b" * 500)
        runtime.accept_input(first)
        runtime.accept_input(second)
        late = Message.text("assistant", "LATE_A_COMPLETION_" + "l" * 300)
        runtime._append_archive(ArchiveEntry(3, "completion", first.identity, (late,), text=late.parts[0].text))
        runtime._working.append(_WorkingEntry(3, "completion", first.identity, (late,), (3,)))
        current = request("C", text="CURRENT_PROTECTED")
        runtime.accept_input(current)
        archive = runtime.history_snapshot
        assert (await runtime.complete(current)).text == "action"
        sources = [fragment["archive_sequence"] for call in chunks(calls) for fragment in metadata(call)["fragments"]]
        assert sources == sorted(sources) and set(sources) == {1, 2, 3}
        assert "LATE_A_COMPLETION_" not in texts(calls[-1].messages)
        assert "A_" + "a" * 4000 not in texts(calls[-1].messages)
        assert "B_" + "b" * 500 not in texts(calls[-1].messages)
        assert "CURRENT_PROTECTED" in texts(calls[-1].messages)
        assert runtime.history_snapshot[:len(archive)] == archive
        assert_summary_bounds(runtime, calls)
    asyncio.run(run())
