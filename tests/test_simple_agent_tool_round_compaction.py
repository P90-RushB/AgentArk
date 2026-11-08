"""Offline generic host-chain compaction; no SDK, HTTP, files, or Unity."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json

import pytest

from agent_ark.agent.simple_agent import CallableModel, SimpleRuntime, SimpleRuntimeConfig, ToolLimits, ToolRegistry
from agent_ark.agent.simple_agent.encoding import api_chat_messages, encode_png
from agent_ark.agent.simple_agent.models import SubmissionResponseError
from agent_ark.agent.simple_agent.runtime import _metrics
from agent_ark.agent.simple_agent.types import (
    AgentEvent, AgentInputIdentity, AgentLimitError, AgentRequest, AgentReuseError,
    ImagePart, LiteLLMAssistantState, Message, ModelCompletion, ModelUsage, NativeAssistantState,
    TextPart, ToolCall, ToolSpec,
)


PRIVATE = "opaque-private-provider-state"
USAGE = ModelUsage(3, 5, 8)
SCOPE = "offline:tool-rounds"


def request(number=1):
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "agent", 0, str(number)), (
        Message.text("system", "Use only visible observations and completed host results."),
        Message.text("user", f"Current observation {number}: preserve this instruction exactly.")), "carry")


def config(**updates):
    values = dict(max_context_characters=4000,
        max_context_tokens=100_000, output_reserve_tokens=128, recent_turns=2,
        summary_output_tokens=256)
    values.update(updates)
    return SimpleRuntimeConfig(**values)


def wire_call(call):
    return {"id": call.call_id, "type": "function", "function": {
        "name": call.name, "arguments": call.arguments_json}}


def state(calls=(), *, kind="native", private=PRIVATE):
    payload = {"role": "assistant", "content": None}
    if calls:
        payload["tool_calls"] = [wire_call(call) for call in calls]
    if kind == "native":
        payload["reasoning"] = private
    elif kind == "encrypted":
        payload.update(reasoning_content=None, encrypted_content=private)
    elif kind == "litellm_chat":
        payload["provider_specific_fields"] = {"opaque_state": private}
    elif kind == "responses":
        payload["responses_output"] = [
            {"type": "reasoning", "id": "reason-" + calls[0].call_id,
             "summary": [], "encrypted_content": private},
            *({"type": "function_call", "call_id": call.call_id,
               "name": call.name, "arguments": call.arguments_json} for call in calls)]
    else:
        return None
    cls = LiteLLMAssistantState if kind.startswith("litellm") or kind == "responses" else NativeAssistantState
    return cls(SCOPE, json.dumps(payload, separators=(",", ":")))


def registry(handler, *, limits=None):
    tools = ToolRegistry(limits=limits)
    tools.register(ToolSpec("host_operation", "Perform one visible bounded operation.",
        '{"type":"object","properties":{"number":{"type":"integer"}},"required":["number"]}'), handler)
    return tools


def text(messages):
    return "\n".join(part.text for message in messages for part in message.parts if isinstance(part, TextPart))


def restarts(runtime):
    return [entry for entry in runtime.history_snapshot if entry.compaction_json is not None
            and json.loads(entry.compaction_json).get("stage") == "restart"]


def assert_fresh(messages):
    assert not any(message.native_assistant is not None or message.tool_calls or message.role == "tool"
                   for message in messages)


def assert_original_provenance(runtime, summaries):
    """Resolve hashes/ranges against saved originals, never working wrappers."""
    for request in summaries:
        for fragment in json.loads(request.compaction_json)["fragments"]:
            entry = runtime.history_snapshot[fragment["archive_sequence"] - 1]
            kind = fragment["kind"]
            if kind in ("leaf_summary", "model_notes", "completion_text"):
                source = entry.text
            elif kind in ("tool_call", "tool_arguments"):
                call = entry.messages[fragment["message_index"]].tool_calls[fragment["call_index"]]
                source = (call.arguments_json if kind == "tool_arguments" else json.dumps(
                    {"call_id": call.call_id, "name": call.name, "arguments_json": call.arguments_json},
                    ensure_ascii=False, separators=(",", ":")))
            else:
                part = entry.messages[fragment["message_index"]].parts[fragment["part_index"]]
                source = part.data if isinstance(part, ImagePart) else part.text
            if "text_start" in fragment:
                source = source[fragment["text_start"]:fragment["text_end"]]
            payload = source if isinstance(source, bytes) else source.encode("utf-8")
            assert hashlib.sha256(payload).hexdigest() == fragment["sha256"]


def test_generic_compaction_is_default_and_legacy_policies_remain_selectable():
    assert SimpleRuntimeConfig().compaction_policy == "tool_rounds_v1"
    assert SimpleRuntimeConfig(compaction_policy="message_chunks_v1").compaction_policy == "message_chunks_v1"
    assert SimpleRuntimeConfig(compaction_policy="whole_turn_v1").compaction_policy == "whole_turn_v1"


def test_default_latest_image_batch_overflow_fails_without_summarizing_unseen_images():
    async def run():
        images = tuple(ImagePart(encode_png(bytes(color) * 4, 2, 2), width=2, height=2)
                       for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255)))
        call = ToolCall("image-batch", "host_operation", '{"number":1}')
        executed, summaries = [], []
        action_round = 0
        def handler(args, session):
            executed.append(args["number"])
            return tuple(part for index, image in enumerate(images, 1)
                         for part in (TextPart(f"Visible image {index}."), image))
        async def model(req):
            nonlocal action_round
            if req.purpose == "summary":
                summaries.append(req)
                pytest.fail("An unseen latest image batch cannot be summarized to bypass its hard image bound.")
            action_round += 1
            if action_round == 1:
                return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,)))
            pytest.fail("The overflowing newest result must not produce another action request.")
        runtime = SimpleRuntime(CallableModel(model), config(max_images=1), tools=registry(handler))
        with pytest.raises(AgentLimitError):
            await runtime.complete(request())
        assert executed == [1] and action_round == 1
        assert summaries == [] and runtime.usage.summary_calls == 0
        group = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert tuple(part for part in group.messages[1].parts if isinstance(part, ImagePart)) == images
        assert not restarts(runtime)
    asyncio.run(run())


def test_previously_observed_images_are_preserved_through_bounded_leaves_and_merge():
    async def run():
        images = tuple(ImagePart(encode_png(bytes(color) * 4, 2, 2), width=2, height=2)
                       for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255)))
        call = ToolCall("older-image-batch", "host_operation", '{"number":1}')
        actions, summaries, seen_images = [], [], []
        def handler(args, session):
            return tuple(part for index, image in enumerate(images, 1)
                         for part in (TextPart(f"Visible image {index}."), image))
        async def model(req):
            if req.purpose == "summary":
                summaries.append(req)
                assert runtime._fits(req.messages, summary=True)
                assert_fresh(req.messages)
                actual = [part for message in req.messages for part in message.parts if isinstance(part, ImagePart)]
                if actual:
                    assert len(actual) == 1
                    number = images.index(actual[0]) + 1
                    assert f"Visible image {number}." in text(req.messages)
                    seen_images.extend(actual)
                    return ModelCompletion(f"Visual fact from image {number} retained.")
                assert json.loads(req.compaction_json)["stage"] == "merge"
                assert all(f"Visual fact from image {number} retained." in text(req.messages) for number in (1, 2, 3))
                return ModelCompletion("Visual facts from all three older images retained.")
            actions.append(req)
            if len(actions) == 1:
                return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,)))
            if len(actions) == 2:
                actual = tuple(part for message in req.messages for part in message.parts if isinstance(part, ImagePart))
                assert actual == images  # The main model observed the entire result before it became old.
                return ModelCompletion("First decision completed.")
            assert_fresh(req.messages)
            assert not any(isinstance(part, ImagePart) for message in req.messages for part in message.parts)
            assert "Visual facts from all three older images retained." in text(req.messages)
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), config(max_images=3), tools=registry(handler))
        assert (await runtime.complete(request(1))).text == "First decision completed."
        runtime.config = replace(runtime.config, max_images=1)
        assert (await runtime.complete(request(2))).text == "done"
        assert seen_images == list(images) and len(summaries) == 4
        assert len(restarts(runtime)) == 1
    asyncio.run(run())


def test_tools_free_default_compacts_old_visual_input_without_changing_current_images():
    async def run():
        images = tuple(ImagePart(encode_png(bytes((index, 255 - index, index // 2)) * 4, 2, 2), width=2, height=2)
                       for index in range(110))
        older_base, current_base = request(1), request(2)
        older = replace(older_base, messages=(older_base.messages[0],
            Message("user", (TextPart("Earlier ordered visual sequence."), *images[:80]))))
        current = replace(current_base, messages=(current_base.messages[0],
            Message("user", (TextPart("Current ordered visual sequence; preserve every image."), *images[80:]))))
        summaries, seen_old_images, actions = [], [], []
        async def model(req):
            assert not req.tools
            if req.purpose == "summary":
                summaries.append(req)
                assert runtime._fits(req.messages, summary=True)
                actual = [part for message in req.messages for part in message.parts if isinstance(part, ImagePart)]
                seen_old_images.extend(actual)
                if actual:
                    return ModelCompletion("Earlier visual evidence retained in chronological order.")
                assert json.loads(req.compaction_json)["stage"] == "merge"
                return ModelCompletion("All earlier visual evidence retained in chronological order.")
            actions.append(req)
            if len(actions) == 2:
                assert runtime._fits(req.messages)
                assert current.messages[1] in req.messages
                actual = tuple(part for message in req.messages for part in message.parts if isinstance(part, ImagePart))
                assert actual == images[80:]
                assert "All earlier visual evidence retained in chronological order." in text(req.messages)
            return ModelCompletion("answer")
        runtime = SimpleRuntime(CallableModel(model), config(max_images=100, recent_turns=1))
        assert (await runtime.complete(older)).text == "answer"
        original_input = runtime.history_snapshot[0]
        assert (await runtime.complete(current)).text == "answer"
        assert len(actions) == 2 and len(summaries) == 3
        assert seen_old_images == list(images[:80])
        assert sum(json.loads(req.compaction_json)["stage"] == "merge" for req in summaries) == 1
        assert original_input in runtime.history_snapshot and original_input.messages == older.messages
        assert len(restarts(runtime)) == 1
        assert_original_provenance(runtime, summaries)
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["native", "encrypted", "litellm_chat", "responses", "plain"])
def test_arbitrary_host_chain_compacts_more_than_once_without_repeating_side_effects(kind):
    async def run():
        executed, sessions, requests, summaries = [], [], [], []
        action_round = 0
        just_summarized = False
        def handler(args, session):
            executed.append(args["number"])
            sessions.append(session)
            return (TextPart(f"Completed operation {args['number']}. " + "r" * 1600),)
        tools = registry(handler)
        async def model(req):
            nonlocal action_round, just_summarized
            requests.append(req)
            if req.purpose == "summary":
                summaries.append(req)
                just_summarized = True
                assert PRIVATE not in text(req.messages)
                assert_fresh(req.messages)
                return ModelCompletion("Earlier host operations completed successfully; preserve their results.", USAGE)
            action_round += 1
            assert runtime._fits(req.messages)
            assert request().messages[1] in req.messages
            if just_summarized:
                assert_fresh(req.messages)
                just_summarized = False
            if action_round <= 6:
                call = ToolCall(f"operation-{action_round}", "host_operation", json.dumps({"number": action_round}))
                return ModelCompletion("", USAGE, (call,), native_assistant=state((call,), kind=kind))
            return ModelCompletion("final action", USAGE)
        runtime = SimpleRuntime(CallableModel(model), config(max_summary_calls=8), tools=tools)
        assert (await runtime.complete(request())).text == "final action"
        assert executed == list(range(1, 7)) and all(session is sessions[0] for session in sessions)
        assert sessions[0].calls == 6
        assert len(summaries) >= 2 and len(restarts(runtime)) >= 2
        assert runtime.usage.action_calls == 7 and runtime.usage.summary_calls == len(summaries)
        assert runtime.usage.action_usage == (USAGE,) * 7
        groups = [entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange"]
        assert len(groups) == 6
        for number, group in enumerate(groups, 1):
            assistant, result = group.messages
            assert assistant.tool_calls[0].call_id == result.tool_call_id == f"operation-{number}"
            assert (assistant.native_assistant is not None) == (kind != "plain")
        assert all(json.loads(entry.compaction_json)["version"] == 2 for entry in restarts(runtime))
        assert_original_provenance(runtime, summaries)
    asyncio.run(run())


def test_long_opaque_ids_are_projected_to_aliases_without_summarizing_newest_result():
    async def run():
        opaque_id = "call__mpthought__" + "q" * 41_839
        call = ToolCall(opaque_id, "host_operation", '{"number":1}')
        original = state((call,))
        action_calls, summaries, executed = [], [], []
        def handler(args, session):
            executed.append(args["number"])
            return (TextPart("Visible operation completed."),)
        async def model(req):
            if req.purpose == "summary":
                summaries.append(req)
                assert opaque_id not in text(req.messages)
                assert "record-" in text(req.messages)
                assert runtime._fits(req.messages, summary=True)
                return ModelCompletion("The visible operation already completed successfully.")
            action_calls.append(req)
            if len(action_calls) == 1:
                return ModelCompletion("", tool_calls=(call,), native_assistant=original)
            assert_fresh(req.messages)
            assert "Visible operation completed." in text(req.messages)
            assert call.arguments_json in text(req.messages)
            assert opaque_id not in text(req.messages)
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), config(
            max_context_characters=50_000, summary_input_characters=4000), tools=registry(handler))
        assert (await runtime.complete(request())).text == "done"
        assert summaries == [] and executed == [1] and len(restarts(runtime)) == 1
        group = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert group.messages[0].native_assistant is original
        assert group.messages[0].tool_calls[0].call_id == group.messages[1].tool_call_id == opaque_id
        assert opaque_id in json.dumps(api_chat_messages(group.messages, SCOPE))
    asyncio.run(run())


def test_latest_tool_image_and_caption_reach_main_model_verbatim_after_restart():
    async def run():
        image = ImagePart(encode_png(bytes((250, 40, 90)) * 4, 2, 2), width=2, height=2)
        caption = TextPart("Newest unseen image: inspect its red marker before choosing an action.")
        opaque_id = "opaque__mpthought__" + "q" * 41_839
        call = ToolCall(opaque_id, "host_operation", '{"number":1}')
        actions, dispatched = [], []
        def handler(args, session):
            dispatched.append(args["number"])
            return (caption, image)
        async def model(req):
            assert req.purpose == "action", "A newly returned image must not be summarized before the main model sees it."
            actions.append(req)
            if len(actions) == 1:
                return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,)))
            assert_fresh(req.messages)
            assert any(caption in message.parts and image in message.parts for message in req.messages)
            parts = next(message.parts for message in req.messages if image in message.parts)
            assert parts[parts.index(image) - 1] is caption
            assert call.arguments_json in text(req.messages) and opaque_id not in text(req.messages)
            return ModelCompletion("Image inspected; final action.")
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=50_000,
            max_images=1), tools=registry(handler))
        assert (await runtime.complete(request())).text == "Image inspected; final action."
        assert dispatched == [1] and len(actions) == 2 and runtime.usage.summary_calls == 0
        assert len(restarts(runtime)) == 1
        group = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert group.messages[1].parts == (caption, image)
    asyncio.run(run())


def test_latest_protected_result_overflow_fails_without_summarizing_unseen_content():
    async def run():
        result = TextPart("Unseen required result " + "r" * 3000)
        call = ToolCall("latest-overflow", "host_operation", '{"number":1}')
        requests, dispatched = [], []
        def handler(args, session):
            dispatched.append(args["number"])
            return (result,)
        async def model(req):
            requests.append(req)
            assert req.purpose == "action"
            return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,)))
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=2600), tools=registry(handler))
        with pytest.raises(AgentLimitError):
            await runtime.complete(request())
        assert dispatched == [1] and len(requests) == 1 and runtime.usage.summary_calls == 0
        assert not restarts(runtime)
        assert any(result in message.parts for message in runtime.working_snapshot)
        assert any(message.native_assistant is not None for message in runtime.working_snapshot)
    asyncio.run(run())


def test_large_closed_result_is_chunked_with_complete_ordered_source_coverage():
    async def run():
        call = ToolCall("large-result", "host_operation", json.dumps({"number": 1,
            "large_argument": "BEGIN_ARGUMENT " + "a" * 4000 + " END_ARGUMENT"}))
        result_text = json.dumps({"status": "failed", "executed": False,
            "detail": "BEGIN_VISIBLE_RESULT " + "r" * 6000 + " END_VISIBLE_RESULT"}, separators=(",", ":"))
        summaries, actions = [], []
        newest_result = "NEWEST_VISIBLE_RESULT " + "n" * 10_000
        def handler(args, session):
            return (TextPart(result_text if args["number"] == 1 else newest_result),)
        async def model(req):
            if req.purpose == "summary":
                summaries.append(req)
                assert runtime._fits(req.messages, summary=True)
                assert_fresh(req.messages)
                return ModelCompletion("The earlier host result reported failure without execution.")
            actions.append(req)
            if len(actions) == 1:
                return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,)))
            if len(actions) == 2:
                assert result_text in text(req.messages)
                next_call = ToolCall("new-result", "host_operation", '{"number":2}')
                return ModelCompletion("", tool_calls=(next_call,), native_assistant=state((next_call,)))
            assert_fresh(req.messages)
            assert newest_result in text(req.messages)
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=20000,
            summary_input_characters=6500), tools=registry(handler))
        assert (await runtime.complete(request())).text == "done"
        audits = [json.loads(req.compaction_json) for req in summaries]
        leaves = [audit for audit in audits if audit["stage"] == "leaf"]
        assert len(leaves) > 1 and any(audit["stage"] == "merge" for audit in audits)
        for kind, source in (("tool_arguments", call.arguments_json), ("tool_result_text", result_text)):
            ranges = [(fragment["text_start"], fragment["text_end"]) for leaf in leaves
                      for fragment in leaf["fragments"] if fragment["kind"] == kind]
            assert ranges[0][0] == 0 and ranges[-1][1] == len(source)
            assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        summary_text = "\n".join(text(req.messages) for req in summaries)
        assert '"status":"failed","executed":false' in summary_text
        group = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert group.messages[0].tool_calls == (call,) and group.messages[1].parts[0].text == result_text
        assert_original_provenance(runtime, summaries)
    asyncio.run(run())


def test_native_only_tiny_previous_answer_is_retained_losslessly_without_summary():
    async def run():
        original = NativeAssistantState(SCOPE, json.dumps({"role": "assistant", "content": "x",
            "reasoning": PRIVATE * 180}, separators=(",", ":")))
        actions, summaries = [], []
        async def model(req):
            if req.purpose == "summary":
                summaries.append(req)
                assert PRIVATE not in text(req.messages)
                return ModelCompletion("The earlier answer was x.")
            actions.append(req)
            if len(actions) == 1:
                return ModelCompletion("x", native_assistant=original)
            assert_fresh(req.messages)
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=10000))
        assert (await runtime.complete(request(1))).text == "x"
        saved = runtime.history_snapshot[-1]
        runtime.config = replace(runtime.config, max_context_characters=3000)
        assert (await runtime.complete(request(2))).text == "done"
        assert summaries == [] and saved in runtime.history_snapshot
        assert "x" in text(actions[-1].messages)
        assert saved.messages[0].native_assistant is original
        assert len(restarts(runtime)) == 1
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["empty", "oversized"])
@pytest.mark.parametrize("pressure", ["soft", "hard"])
def test_fresh_soft_watermark_summary_failure_falls_back_only_when_original_hard_fits(failure, pressure):
    async def run():
        original = NativeAssistantState(SCOPE, json.dumps({"role": "assistant", "content": "x",
            "reasoning": PRIVATE}, separators=(",", ":")))
        actions, summaries = [], []
        def handler(args, session):
            pytest.fail("No host operation was requested.")
        async def model(req):
            if req.purpose == "summary":
                summaries.append(req)
                return ModelCompletion("" if failure == "empty" else "n" * 200, USAGE)
            actions.append(req)
            if len(actions) == 1:
                return ModelCompletion("x", USAGE, native_assistant=original)
            assert req.messages[:-1] == before
            assert req.messages[-1].role == "user" and "Runtime decision-budget snapshot" in text(req.messages[-1:])
            assert any(message.native_assistant is original for message in req.messages)
            return ModelCompletion("done", USAGE)
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=10000, recent_turns=1,
            max_summary_calls=2, summary_output_characters=100), tools=registry(handler))
        earlier = replace(request(1), messages=(request(1).messages[0],
            Message.text("user", "Earlier visible facts " + "v" * 5000)))
        assert (await runtime.complete(earlier)).text == "x"
        current = request(2)
        runtime.accept_input(current)
        before = runtime.working_snapshot
        characters = _metrics(before + (runtime._tool_budget_message(),))[0]
        portable = tuple(replace(message, native_assistant=None) for message in before)
        portable_characters = _metrics(portable + (runtime._tool_budget_message(),))[0]
        cap = characters * 10 // 9 if pressure == "soft" else portable_characters - 1
        runtime.config = replace(runtime.config, max_context_characters=cap)
        assert not runtime._fresh_decision_fits(before)
        assert runtime._fits(before) == (pressure == "soft")
        if pressure == "soft":
            assert (await runtime.complete(current)).text == "done"
            assert runtime.working_snapshot[:len(before)] == before
        else:
            with pytest.raises(AgentLimitError, match="summary call bound"):
                await runtime.complete(current)
            assert runtime.working_snapshot[:-1] == before
            assert "Runtime decision-budget snapshot" in text(runtime.working_snapshot[-1:])
        assert len(summaries) == runtime.usage.summary_calls == 2
        assert runtime.usage.summary_usage == (USAGE, USAGE)
        assert len(actions) == (2 if pressure == "soft" else 1)
        assert not restarts(runtime)
    asyncio.run(run())


def test_below_pressure_keeps_native_continuation_prefix_and_ids_exact():
    async def run():
        call = ToolCall("opaque__mpthought__" + "s" * 1000, "host_operation", '{"number":1}')
        native = state((call,))
        requests = []
        def handler(args, session):
            return (TextPart("Visible successful result."),)
        async def model(req):
            requests.append(req)
            assert req.purpose == "action"
            if len(requests) == 1:
                return ModelCompletion("", tool_calls=(call,), native_assistant=native)
            assert req.messages[:len(requests[0].messages)] == requests[0].messages
            assert req.messages[-3].native_assistant is native
            assert req.messages[-2].tool_call_id == call.call_id
            assert "Runtime decision-budget snapshot" in text(req.messages[-1:])
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=10000), tools=registry(handler))
        assert (await runtime.complete(request())).text == "done"
        assert len(requests) == 2 and runtime.usage.summary_calls == 0 and not restarts(runtime)
    asyncio.run(run())


def test_multicall_batch_closes_before_restart_and_keeps_one_session():
    async def run():
        executed, sessions = [], []
        calls = tuple(ToolCall(f"batch-{number}", "host_operation", json.dumps({"number": number}))
                      for number in range(1, 4))
        action_round = 0
        def handler(args, session):
            executed.append(args["number"])
            sessions.append(session)
            return (TextPart("Completed. " + "r" * (900 if args["number"] < 4 else 4000)),)
        async def model(req):
            nonlocal action_round
            if req.purpose == "summary":
                assert executed == [1, 2, 3, 4]
                assert sum("tool_arguments" in part.text for message in req.messages for part in message.parts) >= 3
                return ModelCompletion("Operations one, two, and three already completed.")
            action_round += 1
            if action_round == 1:
                return ModelCompletion("", tool_calls=calls, native_assistant=state(calls))
            if action_round == 2:
                assert any(message.tool_calls == calls for message in req.messages)
                call = ToolCall("after-restart", "host_operation", '{"number":4}')
                return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,)))
            assert_fresh(req.messages)
            assert "Completed. " + "r" * 4000 in text(req.messages)
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=7000), tools=registry(handler))
        assert (await runtime.complete(request())).text == "done"
        assert executed == [1, 2, 3, 4] and all(session is sessions[0] for session in sessions)
        assert sessions[0].calls == 4
        first = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert len(first.messages) == 4 and len(first.messages[0].tool_calls) == 3
    asyncio.run(run())


def test_restart_does_not_reset_executed_call_ids():
    async def run():
        executed = []
        call = ToolCall("same-id", "host_operation", '{"number":1}')
        def handler(args, session):
            executed.append(args["number"])
            return (TextPart("r" * 1600),)
        async def model(req):
            if req.purpose == "summary":
                return ModelCompletion("Operation one already executed.")
            return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,), private=PRIVATE * 80))
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=3500), tools=registry(handler))
        with pytest.raises(AgentReuseError, match="already executed"):
            await runtime.complete(request())
        assert executed == [1] and len(restarts(runtime)) == 1
        with pytest.raises(AgentReuseError):
            await runtime.complete(request())
    asyncio.run(run())


@pytest.mark.parametrize("bound", ["rounds", "calls", "lifetime"])
def test_compaction_does_not_reset_model_or_tool_budgets(bound):
    async def run():
        executed, actions = [], []
        def handler(args, session):
            executed.append(args["number"])
            return (TextPart("r" * 1600),)
        async def model(req):
            if req.purpose == "summary":
                return ModelCompletion("The first operation already completed.")
            actions.append(req)
            number = len(actions)
            call = ToolCall(f"quota-{number}", "host_operation", json.dumps({"number": number}))
            return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,)))
        limits = ToolLimits(max_model_rounds=3) if bound == "rounds" else ToolLimits(max_calls=2) if bound == "calls" else None
        updates = {"max_model_calls": 3} if bound == "lifetime" else {}
        # Include visible snapshots while retaining pressure after two host
        # operations, before the third main response; quota assertions stay exact.
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=4000, **updates),
                                tools=registry(handler, limits=limits))
        expected = {"rounds": "model tool round budget", "calls": "tool call/time budget",
                    "lifetime": "lifetime model call budget"}[bound]
        with pytest.raises(AgentLimitError, match=expected):
            await runtime.complete(request())
        assert len(restarts(runtime)) == 1 and runtime.usage.summary_calls == 1
        assert executed == ([1, 2, 3] if bound == "rounds" else [1, 2])
        assert len(actions) == (2 if bound == "lifetime" else 3)
    asyncio.run(run())


def test_summary_allowance_is_cumulative_across_restarts_within_one_decision():
    async def run():
        actions, executed = [], []
        def handler(args, session):
            executed.append(args["number"])
            return (TextPart("r" * 1600),)
        async def model(req):
            if req.purpose == "summary":
                return ModelCompletion("One earlier operation is recorded.")
            actions.append(req)
            call = ToolCall(f"operation-{len(actions)}", "host_operation", json.dumps({"number": len(actions)}))
            return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,)))
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=4000,
            max_summary_calls=1), tools=registry(handler))
        with pytest.raises(AgentLimitError, match="summary call bound"):
            await runtime.complete(request())
        assert executed == [1, 2, 3] and len(actions) == 3
        assert runtime.usage.summary_calls == 1 and len(restarts(runtime)) == 1
        assert any(message.native_assistant is not None for message in runtime.working_snapshot)
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["empty", "oversized", "cancelled", "restart_archive"])
def test_failed_restart_leaves_original_working_batch_and_side_effects_intact(failure):
    async def run():
        executed, before_summary = [], []
        action_round = 0
        def handler(args, session):
            executed.append(args["number"])
            return (TextPart("r" * 1600),)
        async def model(req):
            nonlocal action_round
            if req.purpose == "summary":
                before_summary.append(runtime.working_snapshot)
                if failure == "cancelled":
                    raise asyncio.CancelledError()
                if failure == "restart_archive":
                    # One result can be archived; the restart receipt cannot.
                    runtime.config = replace(runtime.config, max_archive_entries=len(runtime.history_snapshot) + 1)
                    return ModelCompletion("The operation already completed.", USAGE)
                return ModelCompletion("" if failure == "empty" else "x" * 200, USAGE)
            action_round += 1
            call = ToolCall(f"once-{action_round}", "host_operation", json.dumps({"number": action_round}))
            return ModelCompletion("", USAGE, (call,), native_assistant=state((call,)))
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=4000,
            max_summary_calls=2, summary_output_characters=100), tools=registry(handler))
        error = asyncio.CancelledError if failure == "cancelled" else AgentLimitError
        with pytest.raises(error):
            await runtime.complete(request())
        assert before_summary and runtime.working_snapshot == before_summary[0]
        assert executed == [1, 2] and not restarts(runtime)
        assert runtime.usage.action_calls == 2
        assert any(message.native_assistant is not None for message in runtime.working_snapshot)
        with pytest.raises(AgentReuseError):
            await runtime.complete(request())
    asyncio.run(run())


def test_decision_deadline_covers_summary_and_keeps_original_working_batch():
    async def run():
        before_summary, executed = [], []
        action_round = 0
        def handler(args, session):
            executed.append(args["number"])
            return (TextPart("r" * 1600),)
        async def model(req):
            nonlocal action_round
            if req.purpose == "summary":
                before_summary.append(runtime.working_snapshot)
                await asyncio.Event().wait()
            action_round += 1
            call = ToolCall(f"deadline-{action_round}", "host_operation", json.dumps({"number": action_round}))
            return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,)))
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=4000,
            timeout_seconds=0.02), tools=registry(handler))
        with pytest.raises(AgentLimitError, match="decision time budget"):
            await runtime.complete(request())
        assert before_summary and runtime.working_snapshot == before_summary[0]
        assert executed == [1, 2] and not restarts(runtime)
    asyncio.run(run())


def test_partial_multicall_failure_is_not_compacted_or_reexecuted():
    async def run():
        executed = []
        calls = tuple(ToolCall(str(number), "host_operation", json.dumps({"number": number})) for number in (1, 2))
        def handler(args, session):
            executed.append(args["number"])
            return (TextPart("r" * 3000),)
        async def model(req):
            assert req.purpose == "action"
            return ModelCompletion("", tool_calls=calls, native_assistant=state(calls))
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=2600),
            tools=registry(handler, limits=ToolLimits(max_calls=1)))
        with pytest.raises(AgentLimitError, match="tool call/time budget"):
            await runtime.complete(request())
        assert executed == [1] and runtime.usage.summary_calls == 0
        assert not any(entry.kind == "tool_exchange" for entry in runtime.history_snapshot)
        with pytest.raises(AgentReuseError):
            await runtime.complete(request())
    asyncio.run(run())


def test_rejected_submission_feedback_is_a_generic_closed_batch_for_compaction():
    async def run():
        dispatched, actions, summaries = [], [], []
        calls = (ToolCall("registered-host", "host_operation", '{"number":1}'),
                 ToolCall("rejected-batch", "arbitrary_rejected_tool", "{}"))
        rejected = ModelCompletion("", USAGE, calls, native_assistant=state(calls, private=PRIVATE * 80))
        def handler(args, session):
            dispatched.append(args)
            return (TextPart("unexpected"),)
        async def model(req):
            if req.purpose == "summary":
                summaries.append(req)
                assert "invalid_action_submission" in text(req.messages)
                assert "rejected; executed=false" in text(req.messages)
                assert "operation already completed" not in text(req.messages)
                assert PRIVATE not in text(req.messages)
                return ModelCompletion("The rejected batch executed nothing; correct the request.")
            actions.append(req)
            if len(actions) == 1:
                raise SubmissionResponseError(rejected, "synthetic invalid action selection")
            assert_fresh(req.messages)
            assert "invalid_action_submission" in text(req.messages)
            assert "rejected; executed=false" in text(req.messages)
            assert "operation already completed" not in text(req.messages)
            assert PRIVATE not in text(req.messages)
            return ModelCompletion("recovered action", USAGE)
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=3500), tools=registry(handler))
        assert (await runtime.complete(request())).text == "recovered action"
        assert dispatched == [] and summaries == [] and len(restarts(runtime)) == 1
        group = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert group.messages[0].native_assistant is rejected.native_assistant
        assert len(group.messages) == 3
        assert all(json.loads(message.parts[0].text)["executed"] is False for message in group.messages[1:])
    asyncio.run(run())


def test_current_observation_and_required_feedback_keep_their_chronological_positions():
    async def run():
        current, earlier = request(2), request(1)
        actions = 0
        def handler(args, session):
            return (TextPart("r" * 1600),)
        async def model(req):
            nonlocal actions
            if req.purpose == "summary":
                return ModelCompletion("Earlier completed operation retained.")
            actions += 1
            if actions == 1:
                call = ToolCall("operation", "host_operation", '{"number":1}')
                return ModelCompletion("", tool_calls=(call,), native_assistant=state((call,), private=PRIVATE * 80))
            assert_fresh(req.messages)
            ordered = [message for message in req.messages if message in (earlier.messages[1], feedback, current.messages[1])]
            assert ordered == [earlier.messages[1], feedback, current.messages[1]]
            return ModelCompletion("done")
        runtime = SimpleRuntime(CallableModel(model), config(max_context_characters=4000), tools=registry(handler))
        runtime.accept_input(earlier)
        runtime.observe_result(AgentEvent(earlier.identity, "required", "observation", '{"visible":"feedback"}'))
        feedback = runtime.working_snapshot[-1]
        assert (await runtime.complete(current)).text == "done"
        assert feedback in runtime.working_snapshot
    asyncio.run(run())
