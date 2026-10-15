"""Executable workspace guidance, with synthetic observations and no providers."""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import re

import pytest

from agent_ark.agent.simple_agent import (
    AgentInputIdentity, AgentRequest, CallableModel, FileWorkspace, ImagePart,
    Message, ModelCompletion, RecordSnapshot, SimpleRuntime, TextPart, ToolCall,
    ToolRegistry, ToolSpec, encode_png,
)
from agent_ark.agent.simple_agent.encoding import tool_declarations


FILE_TOOLS = {"list_files", "read_file", "view_image", "write_note"}


def examples(specs):
    """Exercise the exact JSON examples delivered to the model, not copies."""
    result = {}
    for spec in specs:
        if spec.name in FILE_TOOLS:
            match = re.search(r"\b" + spec.name + r"\((\{.*?\})\)", spec.description)
            assert match, spec.name
            result[spec.name] = json.loads(match.group(1))
    return result


def request(number=1):
    return AgentRequest(AgentInputIdentity("run", "case", "attempt", "agent", 0, str(number)), (
        Message.text("system", "Synthetic task instructions."),
        Message.text("user", "Synthetic current observation.")), "carry")


def call(name, arguments, number):
    return ToolCall(str(number), name, json.dumps(arguments))


def test_workspace_guidance_is_in_real_tool_schemas_and_native_declarations(tmp_path):
    specs = FileWorkspace(tmp_path).specs
    assert {spec.name for spec in specs} == FILE_TOOLS
    documented = examples(specs)
    declarations = tool_declarations(specs)
    for spec, declaration in zip(specs, declarations):
        schema = json.loads(spec.parameters_json)
        assert schema == spec.schema == declaration["function"]["parameters"]
        assert declaration["function"]["description"] == spec.description
        assert set(documented[spec.name]) == set(schema["required"])
        assert schema["additionalProperties"] is False
        assert all(field.get("description") for field in schema["properties"].values())
    guidance = " ".join(spec.description for spec in specs)
    assert "virtual workspace" in guidance and "records/<record_id>/" in guidance
    assert "notes/<name>" in guidance and "next_offset" in guidance
    assert "Compaction" in guidance and "published originals" in guidance
    assert "current decision is published only after completion" in guidance
    assert "SimpleAgent.reset() or SimpleAgent.close() clears" in guidance
    assert "not environment facts" in guidance


def test_guidance_examples_reopen_actual_published_image_after_real_compaction(tmp_path):
    from agent_ark.agent.simple_agent.agent import SimpleAgent

    original = encode_png(bytes([240, 25, 40]) * (8 * 8), 8, 8)
    newer = encode_png(bytes([20, 60, 230]) * (8 * 8), 8, 8)
    reference_label = "Original reference.\n" + "Synthetic reference detail. " * 80
    stages = []
    summaries = []
    documented = {}
    discovered = {}

    def images(value):
        return [part.data for message in value.messages for part in message.parts if isinstance(part, ImagePart)]

    def last_result(value):
        return json.loads(next(message for message in reversed(value.messages) if message.role == "tool").parts[0].text)

    def submit(stage):
        return ModelCompletion("Selected synthetic action.", tool_calls=(
            call("submit_action", {"format": "raw", "content": "synthetic-action-" + str(stage)}, stage),))

    async def model(value):
        if value.purpose == "summary":
            assert not value.tools
            summaries.append(value)
            return ModelCompletion("Earlier synthetic observations and decisions were recorded.")
        stage = len(stages)
        stages.append(value)
        if stage == 0:
            documented.update(examples(value.tools))
            assert images(value) == [original]
            return submit(stage)
        if stage == 1:
            assert summaries and original not in images(value)
            assert images(value) == [newer]
            return submit(stage)
        if stage == 2:
            assert original not in images(value)
            return ModelCompletion("", tool_calls=(call("list_files", documented["list_files"], stage),))
        if stage == 3:
            listing = last_result(value)
            assert listing["next_offset"] is None
            rows = listing["files"]
            assert all(row["read_only"] for row in rows)
            assert {row["record_id"] for row in rows} == {"turn-000001", "turn-000002"}
            paths = {row["path"] for row in rows}
            # IDs and paths are confirmed by list_files before using the example.
            assert documented["read_file"]["path"] in paths
            discovered["paths"] = paths
            return ModelCompletion("", tool_calls=(call("read_file", documented["read_file"], stage),))
        if stage == 4:
            record = json.loads(last_result(value)["text"])
            assert record["format"] == "visible-messages-v1"
            chunks = next(part["text_chunks"] for message in record["messages"]
                          for part in message["content"] if "text_chunks" in part)
            assert "".join(chunks) == "Current observation (step_msg): " + reference_label
            image = next(part for message in record["messages"] for part in message["content"] if part["type"] == "image")
            root = documented["read_file"]["path"].rsplit("/", 1)[0]
            path = root + "/" + image["path"]
            assert path in discovered["paths"]
            assert documented["view_image"] == {"path": path}
            return ModelCompletion("", tool_calls=(call("view_image", documented["view_image"], stage),))
        if stage == 5:
            # This is the original PNG returned as model-visible vision content,
            # although its observation was removed from the working request.
            assert images(value) == [original]
            returned = next(message for message in reversed(value.messages) if message.role == "tool")
            assert returned.parts[1] == ImagePart(original)
            assert json.loads(returned.parts[0].text)["path"] == documented["view_image"]["path"]
            return ModelCompletion("", tool_calls=(call("write_note", documented["write_note"], stage),))
        if stage == 6:
            note = last_result(value)
            assert note["model_memory"] is True
            assert note["path"] == documented["write_note"]["path"]
            return ModelCompletion("", tool_calls=(call("read_file", {
                "path": note["path"], "start_line": 1, "max_lines": 100}, stage),))
        assert stage == 7
        assert last_result(value)["text"] == documented["write_note"]["content"]
        return submit(stage)

    agent = SimpleAgent(model_backend=CallableModel(model), simple_agent_config={
        "context": {"max_images": 1, "recent_turns": 1},
        "tools": {"enabled": True, "workspace_parent": str(tmp_path)}})

    def observation(label, png=None):
        content = [{"type": "text", "text": "Current observation (step_msg): " + label}]
        if png is not None:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}})
        return {0: {"messages": [{"role": "user", "content": content}], "message_mode": "delta"}}

    try:
        agent.forward_with_trace(observation(reference_label, original))
        first = agent._states[0].workspace._records["turn-000001"][0]
        assert isinstance(first, RecordSnapshot)
        assert set(dict(first.files)) == {"messages.json", "assistant.txt", "action.txt", "images/0000.png"}
        assert dict(first.files)["images/0000.png"] == original
        assert dict(first.files)["assistant.txt"] == b"Selected synthetic action."
        assert dict(first.files)["action.txt"] == b"synthetic-action-0"
        agent.forward_with_trace(observation("Newer frame.", newer))
        result, trace = agent.forward_with_trace(observation("Recheck the original reference."))
        assert result[0]["action"] == "synthetic-action-7"
        assert len(stages) == 8 and len(summaries) >= 2
        state = agent._states[0]
        assert state.workspace._records["turn-000001"][0] is first
        assert any(part.data == original for entry in state.runtime.history_snapshot
                   for message in entry.messages for part in message.parts if isinstance(part, ImagePart))
        assert trace[0]["agent_usage"]["action_calls"] == 6
        # Notes and published snapshots belong only to this case's workspace.
        directory = state.directory.name
        agent.reset()
        assert not agent._states and not Path(directory).exists()
    finally:
        agent.close()


def test_listing_prefix_pagination_and_writable_namespace_match_guidance(tmp_path):
    workspace = FileWorkspace(tmp_path)
    text = b'{\n  "visible": true\n}\n'
    snapshot = RecordSnapshot("listed", (("messages.json", text), ("action.txt", b"action")))
    workspace.publish(snapshot, request().identity)
    workspace.accept_input(request())
    session = workspace.begin(request())
    assert workspace.execute(call("write_note", {"path": "notes/plan.md", "content": "model interpretation"}, 1), session).success
    first = json.loads(workspace.execute(call("list_files", {"path": "records/listed", "offset": 0, "limit": 1}, 2), session).parts[0].text)
    second = json.loads(workspace.execute(call("list_files", {"path": "records/listed", "offset": first["next_offset"], "limit": 1}, 3), session).parts[0].text)
    assert first["next_offset"] == 1 and second["next_offset"] is None
    assert [first["files"][0]["path"], second["files"][0]["path"]] == ["records/listed/action.txt", "records/listed/messages.json"]
    notes = json.loads(workspace.execute(call("list_files", {"path": "notes", "offset": 0, "limit": 20}, 4), session).parts[0].text)
    assert notes["files"][0]["path"] == "notes/plan.md" and notes["files"][0]["read_only"] is False
    for number, path in enumerate(("records/listed/action.txt", "notes/nested/plan.md", "/outside.txt"), 5):
        assert not workspace.execute(call("write_note", {"path": path, "content": "invalid"}, number), session).success
    page = json.loads(workspace.execute(call("read_file", {
        "path": "records/listed/messages.json", "start_line": 1, "max_lines": 1}, 8), session).parts[0].text)
    rest = json.loads(workspace.execute(call("read_file", {
        "path": page["path"], "start_line": page["next_line"], "max_lines": 100}, 9), session).parts[0].text)
    assert page["next_line"] == 2 and rest["next_line"] is None
    assert (page["text"] + rest["text"]).encode("utf-8") == text
    assert page["digest"] == rest["digest"] == snapshot.digest


@pytest.mark.parametrize("custom_tool", [False, True])
def test_workspace_guidance_is_absent_without_filesystem_tools(custom_tool):
    async def run():
        registry = None
        if custom_tool:
            registry = ToolRegistry()
            registry.register(ToolSpec("echo", "Return the supplied text.",
                '{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}'),
                lambda args, _: (TextPart(args["text"]),))
        seen = []

        async def model(value):
            seen.append(value)
            return ModelCompletion("done")

        runtime = SimpleRuntime(CallableModel(model), tools=registry)
        assert (await runtime.complete(request())).text == "done"
        assert seen[0].messages == request().messages
        assert {spec.name for spec in seen[0].tools} == ({"echo"} if custom_tool else set())
        assert not FILE_TOOLS.intersection(spec.name for spec in seen[0].tools)
        assert "records/" not in json.dumps(tool_declarations(seen[0].tools))

    asyncio.run(run())
