# SPDX-License-Identifier: Apache-2.0
# Adapted from ArkArena workspace-tool tests, licensed under Apache-2.0.
"""Synthetic own records; no game, network, Codex SDK or hidden task source."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent import (
    CallableModel, ChatCompletionsModel, FileWorkspace, AgentInputIdentity,
    AgentLimitError, AgentRequest, AgentReuseError, ImagePart, Message,
    ModelCompletion, ModelUsage, RecordSnapshot, SimpleRuntime, SimpleRuntimeConfig,
    TextPart, ToolCall, ToolError, ToolLimits, ToolRegistry, ToolSpec, encode_png,
)
from agent_ark.agent.simple_agent.encoding import chat_messages


def request(number=1, episode="e1", incarnation=1, history="carry", agent="alpha", session="session"):
    return AgentRequest(AgentInputIdentity("run", session, episode, agent, incarnation, str(number)), (
        Message.text("system", "Read permitted synthetic records and return only final Action JSON."),
        Message.text("user", "Available: records/match01/. Review selected records and write own notes.")), history)


def call(name, args, number=1):
    return ToolCall(str(number), name, json.dumps(args))


def workspace(tmp_path, limits=None):
    tools = FileWorkspace(tmp_path, limits=limits)
    png = encode_png(bytes([250, 10, 5]) * (16 * 16), 16, 16)
    tools.publish(RecordSnapshot("match01", (("index.json", b'{"events":"events.jsonl","frame":"frame.png"}'),
                                             ("events.jsonl", b'{"event":"synthetic","marker":"OWN42"}\n'),
                                             ("frame.png", png))), request().identity)
    return tools, png


def read(path, start=1, lines=100):
    return {"path": path, "start_line": start, "max_lines": lines}


def test_api_tool_loop_reads_image_writes_reads_note_and_returns_final_action(tmp_path):
    async def run():
        tools, png = workspace(tmp_path)
        requests = []
        async def create(**kwargs):
            requests.append(kwargs)
            messages = kwargs["messages"]
            round = len(requests)
            results = [m for m in messages if m["role"] == "tool"]
            if round == 1:
                name, args = "list_files", {"path": "records", "offset": 0, "limit": 20}
            elif round == 2:
                paths = [row["path"] for row in json.loads(results[-1]["content"])["files"]]
                name, args = "read_file", read(next(p for p in paths if p.endswith("index.json")))
            elif round == 3:
                index = json.loads(json.loads(results[-1]["content"])["text"])
                name, args = "read_file", read("records/match01/" + index["events"])
            elif round == 4:
                assert "OWN42" in results[-1]["content"]
                name, args = "view_image", {"path": "records/match01/frame.png"}
            elif round == 5:
                image = next(p for m in messages if m["role"] == "user" for p in m["content"] if p["type"] == "image_url")
                assert base64.b64decode(image["image_url"]["url"].split(",", 1)[1]) == png
                name, args = "write_note", {"path": "notes/review.md", "content": "Synthetic OWN42; red frame."}
            elif round == 6:
                name, args = "read_file", read("notes/review.md")
            else:
                assert "Synthetic OWN42; red frame." in results[-1]["content"]
                assert len(results) == 6
                return {"choices": [{"finish_reason": "stop", "message": {"content": '{"payload":"public class Strategy {}"}'}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
            return {"choices": [{"finish_reason": "tool_calls", "message": {"content": None,
                       "tool_calls": [{"id": str(round), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        harness = SimpleRuntime(ChatCompletionsModel(client, "api-model"), tools=tools)
        result = await harness.complete(request())
        assert json.loads(result.text)["payload"] == "public class Strategy {}"
        assert result.usage == ModelUsage(70, 14, 84)
        assert harness.usage.action_calls == 7 and harness.usage.summary_calls == 0
        assert len([entry for entry in harness.history_snapshot if entry.kind == "tool_exchange"]) == 6
        assert all(len(r["tools"]) == 4 for r in requests)
        assert requests[-1]["messages"].count(requests[0]["messages"][0]) == 1
    asyncio.run(run())


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "C:/secret", "notes/../secret", "notes/a:b", "notes/NUL", "notes/a\\b", "notes/dir/file", "notes/file."])
def test_unsafe_paths_are_recoverable_bounded_errors_without_machine_paths(tmp_path, path):
    tools, _ = workspace(tmp_path)
    tools.accept_input(request())
    session = tools.begin(request())
    result = tools.execute(call("write_note", {"path": path, "content": "bad"}), session)
    assert result.success is False
    assert str(tmp_path) not in result.parts[0].text
    assert not list(tmp_path.rglob("*.md"))
    assert tools.execute(call("write_note", {"path": "notes/ok.md", "content": "ok"}, 2), session).success


def test_record_snapshot_identity_immutable_and_publish_does_not_change_active_reads(tmp_path):
    root = tmp_path / "records"
    root.mkdir()
    (root / "log.txt").write_bytes(b"first\nsecond\n")
    first = RecordSnapshot.from_directory("fixed", root)
    tools = FileWorkspace(tmp_path)
    tools.publish(first, request().identity)
    tools.accept_input(request())
    active = tools.begin(request())
    (root / "log.txt").write_text("replacement", encoding="utf-8")
    result = tools.execute(call("read_file", read("records/fixed/log.txt", 2, 1)), active)
    assert json.loads(result.parts[0].text)["text"] == "second\n"
    with pytest.raises(AgentReuseError):
        tools.publish(RecordSnapshot.from_directory("fixed", root), request().identity)
    tools.publish(RecordSnapshot("later", (("text", b"new"),)), request().identity)
    result = tools.execute(call("read_file", read("records/later/text"), 2), active)
    assert not result.success


def test_notes_carry_omit_churn_and_quota_across_namespaces(tmp_path):
    tools, _ = workspace(tmp_path, ToolLimits(max_workspace_bytes=10))
    source = request()
    tools.accept_input(source)
    old = tools.begin(source)
    tools.execute(call("write_note", {"path": "notes/n.md", "content": "123456"}), old)
    carry = request(2, "e2", 2)
    tools.accept_input(carry)
    with pytest.raises(AgentReuseError):
        tools.execute(call("read_file", read("notes/n.md"), 2), old)
    fresh = tools.begin(carry)
    assert tools.execute(call("read_file", read("notes/n.md")), fresh).success
    assert "match01" in fresh.records
    omit = request(3, "e3", 1, "omit")
    tools.accept_input(omit)
    empty = tools.begin(omit)
    assert not empty.records
    assert not tools.execute(call("read_file", read("notes/n.md")), empty).success
    assert not tools.execute(call("write_note", {"path": "notes/n.md", "content": "12345"}, 2), empty).success
    assert tools.execute(call("write_note", {"path": "notes/n.md", "content": "12"}, 3), empty).success
    churn = request(4, "e3", 2)
    tools.accept_input(churn)
    assert not tools.execute(call("read_file", read("notes/n.md")), tools.begin(churn)).success
    with pytest.raises(AgentReuseError):
        tools.accept_input(request(session="different"))
    with pytest.raises(AgentReuseError):
        tools.publish(RecordSnapshot("peer", (("x", b"hidden"),)), request(agent="beta").identity)


def test_duplicate_calls_do_not_repeat_write_and_limits_are_finite(tmp_path):
    tools, _ = workspace(tmp_path, ToolLimits(max_calls=2))
    tools.accept_input(request())
    session = tools.begin(request())
    write = call("write_note", {"path": "notes/n.md", "content": "original"})
    tools.execute(write, session)
    with pytest.raises(AgentReuseError):
        tools.execute(write, session)
    result = tools.execute(call("read_file", read("notes/n.md"), 2), session)
    assert "original" in result.parts[0].text
    with pytest.raises(AgentLimitError):
        tools.execute(call("list_files", {"path": ".", "offset": 0, "limit": 5}, 3), session)


def test_snapshot_rejects_symlinks(tmp_path):
    directory = tmp_path / "record"
    directory.mkdir()
    source = tmp_path / "outside"
    source.write_bytes(b"secret")
    link = directory / "linked"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation requires OS privilege")
    with pytest.raises(ToolError):
        RecordSnapshot.from_directory("r", directory)


def test_snapshot_rejects_hardlinks_and_size_limits(tmp_path):
    directory = tmp_path / "record"
    directory.mkdir()
    source = tmp_path / "outside"
    source.write_bytes(b"secret")
    link = directory / "linked"
    import os
    os.link(source, link)
    with pytest.raises(ToolError):
        RecordSnapshot.from_directory("r", directory)
    link.unlink()
    link.write_bytes(b"x" * 20)
    with pytest.raises(ToolError):
        RecordSnapshot.from_directory("r", directory, limits=ToolLimits(max_record_file_bytes=10))


def test_record_listing_can_paginate_beyond_4096(tmp_path):
    tools = FileWorkspace(tmp_path)
    files = tuple((f"{n:04}.txt", b"x") for n in range(1000))
    for n in range(5):
        tools.publish(RecordSnapshot("r" + str(n), files), request().identity)
    tools.accept_input(request())
    session = tools.begin(request())
    result = tools.execute(call("list_files", {"path": "records", "offset": 4100, "limit": 100}), session)
    rows = json.loads(result.parts[0].text)
    assert result.success and len(rows["files"]) == 100 and rows["next_offset"] == 4200


def test_new_input_during_api_work_never_enters_old_tool_context(tmp_path):
    async def run():
        tools, _ = workspace(tmp_path)
        started, release = asyncio.Event(), asyncio.Event()
        requests = []
        async def model(req):
            requests.append(req)
            if len(requests) == 1:
                started.set()
                await release.wait()
                return ModelCompletion("", tool_calls=(call("read_file", read("records/match01/events.jsonl")),))
            assert not any("NEW OBSERVATION" in p.text for m in req.messages for p in m.parts if isinstance(p, TextPart))
            return ModelCompletion('{"payload":"original"}')
        harness = SimpleRuntime(CallableModel(model), tools=tools)
        work = asyncio.create_task(harness.complete(request()))
        await started.wait()
        newer = replace(request(2), messages=(request().messages[0], Message.text("user", "NEW OBSERVATION")))
        harness.accept_input(newer)
        release.set()
        assert (await work).text == '{"payload":"original"}'
        assert any("NEW OBSERVATION" in p.text for m in harness.working_snapshot for p in m.parts if isinstance(p, TextPart))
    asyncio.run(run())


def test_context_images_batch_order_and_unknown_usage(tmp_path):
    async def run():
        tools, _ = workspace(tmp_path)
        calls = 0
        async def model(req):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelCompletion("", ModelUsage(1, 1, 2), (
                    call("view_image", {"path": "records/match01/frame.png"}),
                    call("read_file", read("records/match01/events.jsonl"), 2)))
            wire = chat_messages(req.messages)
            assert [m["role"] for m in wire[-3:]] == ["tool", "tool", "user"]
            return ModelCompletion('{"payload":"ok"}')
        result = await SimpleRuntime(CallableModel(model), tools=tools).complete(request())
        assert result.usage == ModelUsage()
    asyncio.run(run())


def test_no_codex_dependency_in_fresh_python_process():
    code = """
import sys
class BlockDependencies:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'ark_arena', 'openai_codex', 'openai', 'numpy', 'PIL', 'yaml'}:
            raise RuntimeError('External dependency import prohibited')
sys.meta_path.insert(0, BlockDependencies())
from agent_ark.agent.simple_agent import SimpleRuntime, ChatCompletionsModel, FileWorkspace, ToolCall
assert not any(name.startswith('openai_codex') for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_custom_tools_schema_argument_errors_and_owner_isolation(tmp_path):
    tools = ToolRegistry()
    tools.register(ToolSpec("echo", "Return bounded own text.", '{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}'),
                   lambda args, session: (TextPart(args["text"]),))
    tools.accept_input(request())
    session = tools.begin(request())
    assert not tools.execute(ToolCall("1", "echo", '{"text":"a","text":"b"}'), session).success
    assert not tools.execute(call("echo", {"text": 42}, 2), session).success
    assert tools.execute(call("echo", {"text": "ok"}, 3), session).parts == (TextPart("ok"),)
    tools.claim(object())
    with pytest.raises(AgentReuseError):
        SimpleRuntime(CallableModel(lambda r: asyncio.sleep(0)), tools=tools)


def test_complete_tool_groups_survive_compaction_without_orphan_calls():
    async def run():
        registry = ToolRegistry()
        registry.register(ToolSpec("echo", "Return own text.", '{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}'),
                          lambda args, _: (TextPart(args["text"]),))
        action_round = 0
        summaries = []
        def assert_pairs(messages):
            pending = set()
            for message in messages:
                if message.tool_calls:
                    assert not pending
                    pending.update(c.call_id for c in message.tool_calls)
                elif message.role == "tool":
                    assert message.tool_call_id in pending
                    pending.remove(message.tool_call_id)
                else:
                    assert not pending
            assert not pending
        async def model(req):
            nonlocal action_round
            assert_pairs(req.messages)
            if req.purpose == "summary":
                assert not req.tools
                summaries.append(req.messages)
                return ModelCompletion("Earlier synthetic tool outputs were read and retained as model notes.")
            action_round += 1
            if action_round % 2:
                return ModelCompletion("", tool_calls=(call("echo", {"text": "x" * 400}, action_round),))
            return ModelCompletion('{"payload":"ok"}')
        config = SimpleRuntimeConfig(max_context_characters=2200, max_context_tokens=2000,
                                     output_reserve_tokens=32, recent_turns=0, summary_output_tokens=128)
        harness = SimpleRuntime(CallableModel(model), config, tools=registry)
        for n in range(1, 6):
            assert (await harness.complete(request(n))).text == '{"payload":"ok"}'
        assert summaries
        assert_pairs(harness.working_snapshot)
        assert len([entry for entry in harness.history_snapshot if entry.kind == "tool_exchange"]) == 5
    asyncio.run(run())


@pytest.mark.parametrize("path", [
    "../secret", "/absolute", "C:/secret", "C:\\secret", "log.txt:secret", "a//b",
    "a/./b", "a/../b", "a\\b", "NUL", "CON.txt", "COM1.log", "trailing.",
    "trailing ", "control\x00", "a" * 129,
])
def test_snapshot_rejects_unsafe_logical_paths(path):
    with pytest.raises(ToolError):
        RecordSnapshot("own", ((path, b"data"),))


@pytest.mark.parametrize("files", [
    [("x", b"data")],
    (["x", b"data"],),
    (("x", bytearray(b"mutable")),),
    (("x", b"first"), ("X", b"second")),
    (("x",),),
])
def test_snapshot_requires_immutable_unique_entries(files):
    with pytest.raises(ValueError):
        RecordSnapshot("own", files)


@pytest.mark.parametrize("record_id", ["", "../outside", "with/slash", "id:stream", "a" * 65])
def test_snapshot_record_id_is_a_bounded_logical_identifier(record_id):
    with pytest.raises(ValueError):
        RecordSnapshot(record_id, (("x", b"data"),))


def test_safe_tools_declare_only_workspace_capabilities(tmp_path):
    tools = FileWorkspace(tmp_path)
    assert {spec.name for spec in tools.specs} == {"list_files", "read_file", "view_image", "write_note"}
    assert all(spec.schema["additionalProperties"] is False for spec in tools.specs)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field,value", [
    ("max_calls", 0), ("max_record_bytes", -1), ("max_note_files", True),
    ("max_published_bytes", 1.5), ("timeout_seconds", 0),
    ("timeout_seconds", float("inf")), ("timeout_seconds", float("nan")),
    ("timeout_seconds", 3601), ("timeout_seconds", True),
])
def test_tool_limits_reject_nonfinite_or_nonpositive_values(field, value):
    with pytest.raises(ValueError):
        ToolLimits(**{field: value})


def test_registry_owner_and_session_cannot_cross_registry_or_closed_scope():
    tools = ToolRegistry()
    other = ToolRegistry()
    source = request()
    for registry in (tools, other):
        registry.accept_input(source)
    session = tools.begin(source)
    with pytest.raises(AgentReuseError):
        other.execute(call("unknown", {}), session)
    session.close()
    with pytest.raises(AgentReuseError):
        tools.execute(call("unknown", {}), session)
    owner = object()
    tools.claim(owner)
    tools.claim(owner)
    with pytest.raises(AgentReuseError):
        tools.claim(object())
    with pytest.raises(AgentReuseError):
        tools.validate_fresh()
    with pytest.raises(AgentReuseError):
        tools.begin(request(2))


@pytest.mark.parametrize("identity", [
    AgentInputIdentity("other-run", "session", "e1", "alpha", 1, "2"),
    AgentInputIdentity("run", "other-session", "e1", "alpha", 1, "2"),
    AgentInputIdentity("run", "session", "e1", "other-agent", 1, "2"),
])
def test_published_records_bind_input_and_publication_to_owner(tmp_path, identity):
    tools, _ = workspace(tmp_path)
    foreign = AgentRequest(identity, request().messages)
    with pytest.raises(AgentReuseError):
        tools.accept_input(foreign)
    with pytest.raises(AgentReuseError):
        tools.publish(RecordSnapshot("foreign", (("x", b"secret"),)), identity)


def test_scope_count_is_bounded_even_when_history_is_omitted():
    tools = ToolRegistry(limits=ToolLimits(max_scope_entries=2))
    tools.accept_input(request())
    tools.accept_input(request(2, "e2", history="omit"))
    with pytest.raises(AgentLimitError):
        tools.accept_input(request(3, "e3", history="omit"))
    assert tools.begin(request(2, "e2", history="omit"))


def test_record_publication_limits_are_cumulative_and_idempotent(tmp_path):
    limits = ToolLimits(max_records=2, max_published_bytes=5, max_record_bytes=4,
                        max_record_file_bytes=3, max_record_files=2)
    tools = FileWorkspace(tmp_path, limits=limits)
    own = request().identity
    first = RecordSnapshot("first", (("x", b"123"),))
    tools.publish(first, own)
    tools.publish(first, own)
    with pytest.raises(AgentLimitError):
        tools.publish(RecordSnapshot("second", (("x", b"123"),)), own)
    tools.publish(RecordSnapshot("second", (("x", b"12"),)), own)
    with pytest.raises(AgentLimitError):
        tools.publish(RecordSnapshot("third", (("x", b"1"),)), own)
    with pytest.raises(AgentReuseError):
        tools.publish(first, request(2, "other").identity)


@pytest.mark.parametrize("files", [
    (("x", b"1234"),),
    (("x", b"123"), ("y", b"12")),
    (("x", b"1"), ("y", b"1"), ("z", b"1")),
])
def test_record_per_file_total_and_count_bounds(tmp_path, files):
    tools = FileWorkspace(tmp_path, limits=ToolLimits(max_record_file_bytes=3,
                          max_record_bytes=4, max_record_files=2))
    with pytest.raises(AgentLimitError):
        tools.publish(RecordSnapshot("own", files), request().identity)


def test_record_directory_total_and_entry_count_bounds(tmp_path):
    source = tmp_path / "record"
    source.mkdir()
    (source / "a").write_bytes(b"123")
    (source / "b").write_bytes(b"456")
    with pytest.raises(ToolError, match="capacity"):
        RecordSnapshot.from_directory("own", source, limits=ToolLimits(max_record_bytes=5))
    (source / "a").unlink()
    (source / "b").unlink()
    for name in ("a", "b", "c"):
        (source / name).mkdir()
    with pytest.raises(ToolError, match="entry limit"):
        RecordSnapshot.from_directory("own", source, limits=ToolLimits(max_record_files=1))


def test_note_utf8_byte_count_replacement_and_file_quota(tmp_path):
    tools = FileWorkspace(tmp_path, limits=ToolLimits(max_note_bytes=4,
                          max_workspace_bytes=5, max_note_files=1))
    source = request()
    tools.accept_input(source)
    session = tools.begin(source)
    assert not tools.execute(call("write_note", {"path": "notes/n.md", "content": "三三"}), session).success
    assert tools.execute(call("write_note", {"path": "notes/n.md", "content": "1234"}, 2), session).success
    assert tools.execute(call("write_note", {"path": "notes/n.md", "content": "12"}, 3), session).success
    assert not tools.execute(call("write_note", {"path": "notes/other.md", "content": "1"}, 4), session).success
    assert json.loads(tools.execute(call("read_file", read("notes/n.md"), 5), session).parts[0].text)["text"] == "12"
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) == 2


def test_unowned_or_modified_notes_are_not_read_or_overwritten(tmp_path):
    tools = FileWorkspace(tmp_path)
    source = request()
    tools.accept_input(source)
    session = tools.begin(source)
    tools.execute(call("write_note", {"path": "notes/n.md", "content": "own"}), session)
    owned = next(tmp_path.rglob("n.md"))
    owned.write_bytes(b"outside change")
    assert not tools.execute(call("read_file", read("notes/n.md"), 2), session).success
    assert not tools.execute(call("write_note", {"path": "notes/n.md", "content": "replace"}, 3), session).success
    unknown = owned.parent / "unknown.md"
    unknown.write_bytes(b"foreign")
    assert not tools.execute(call("read_file", read("notes/unknown.md"), 4), session).success
    assert not tools.execute(call("write_note", {"path": "notes/unknown.md", "content": "replace"}, 5), session).success
    assert unknown.read_bytes() == b"foreign"
    owned.unlink()
    assert not tools.execute(call("write_note", {"path": "notes/n.md", "content": "replace"}, 6), session).success


def test_hardlinked_notes_cannot_expose_or_modify_external_files(tmp_path):
    tools = FileWorkspace(tmp_path)
    source = request()
    tools.accept_input(source)
    session = tools.begin(source)
    tools.execute(call("write_note", {"path": "notes/n.md", "content": "own"}), session)
    target = next(tmp_path.rglob("n.md"))
    target.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    os.link(outside, target)
    for number, name, args in (
        (2, "read_file", read("notes/n.md")),
        (3, "write_note", {"path": "notes/n.md", "content": "replacement"}),
    ):
        assert not tools.execute(call(name, args, number), session).success
    assert outside.read_bytes() == b"secret"


def test_symlink_workspace_root_is_rejected_before_creating_notes(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation requires OS privilege")
    tools = FileWorkspace(link)
    tools.accept_input(request())
    result = tools.execute(call("write_note", {"path": "notes/n.md", "content": "bad"}), tools.begin(request()))
    assert not result.success
    assert not list(actual.iterdir())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction fixture")
def test_windows_junction_root_cannot_publish_or_write_external_content(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "private.txt").write_bytes(b"secret")
    junction = tmp_path / "junction"
    completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(actual)],
                               capture_output=True, text=True)
    if completed.returncode:
        pytest.skip("junction creation unavailable on this filesystem")
    with pytest.raises(ToolError):
        RecordSnapshot.from_directory("own", junction)
    tools = FileWorkspace(junction)
    tools.accept_input(request())
    result = tools.execute(call("write_note", {"path": "notes/n.md", "content": "bad"}), tools.begin(request()))
    assert not result.success
    assert [path.name for path in actual.iterdir()] == ["private.txt"]


def test_tool_errors_are_bounded_and_never_disclose_os_exception(tmp_path):
    tools = FileWorkspace(tmp_path / "missing")
    tools.accept_input(request())
    session = tools.begin(request())
    result = tools.execute(call("write_note", {"path": "notes/n.md", "content": "note"}), session)
    assert not result.success
    assert str(tmp_path) not in result.parts[0].text
    assert "FileNotFoundError" not in result.parts[0].text
    unknown = tools.execute(call("submit_action", {}, 2), session)
    assert not unknown.success and "Unknown tool" in unknown.parts[0].text


@pytest.mark.parametrize("arguments", [
    '{"text":"a","text":"b"}', '["not object"]', '{"text":NaN}',
    '{"text":42}', '{"text":"ok","extra":"bad"}', '{"wrong":"missing"}',
])
def test_custom_tool_invalid_arguments_never_reach_callback(arguments):
    invoked = []
    tools = ToolRegistry()
    tools.register(ToolSpec("echo", "Return own text.", '{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}'),
                   lambda args, _: invoked.append(args) or (TextPart(args["text"]),))
    tools.accept_input(request())
    result = tools.execute(ToolCall("1", "echo", arguments), tools.begin(request()))
    assert not result.success and not invoked


def test_unsupported_schema_registration_fails_closed():
    tools = ToolRegistry()
    with pytest.raises(ValueError, match="unsupported"):
        tools.register(ToolSpec("unsafe", "Invalid schema.", '{"type":"string","pattern":".*"}'), lambda *_: (TextPart("x"),))
    assert not tools.specs


def test_text_and_image_result_budgets_are_finite():
    tools = ToolRegistry(limits=ToolLimits(max_text_bytes=5, max_total_text_bytes=5,
                          max_image_bytes=3, max_total_image_bytes=3, max_images=1))
    tools.register(ToolSpec("text", "Own text.", '{"type":"object"}'), lambda *_: (TextPart("123"),))
    tools.register(ToolSpec("image", "Own image.", '{"type":"object"}'), lambda *_: (ImagePart(b"123"),))
    tools.register(ToolSpec("large", "Own text.", '{"type":"object"}'), lambda *_: (TextPart("123456"),))
    tools.accept_input(request())
    session = tools.begin(request())
    assert tools.execute(call("text", {}), session).success
    with pytest.raises(AgentLimitError, match="result budget"):
        tools.execute(call("text", {}, 2), session)
    assert tools.execute(call("image", {}, 3), session).success
    with pytest.raises(AgentLimitError, match="result budget"):
        tools.execute(call("image", {}, 4), session)
    with pytest.raises(AgentLimitError, match="result budget"):
        tools.execute(call("large", {}, 5), session)


def test_timeout_before_and_after_operation_does_not_retry(monkeypatch):
    from agent_ark.agent.simple_agent import tools as module
    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    invocations = []
    tools = ToolRegistry(limits=ToolLimits(timeout_seconds=1))
    def operation(*_):
        invocations.append(True)
        now[0] += 2
        return (TextPart("own result"),)
    tools.register(ToolSpec("slow", "Bounded operation.", '{"type":"object"}'), operation)
    tools.accept_input(request())
    session = tools.begin(request())
    with pytest.raises(AgentLimitError, match="after operation"):
        tools.execute(call("slow", {}), session)
    with pytest.raises(AgentLimitError, match="call/time budget"):
        tools.execute(call("slow", {}, 2), session)
    assert len(invocations) == 1
