"""Synthetic opaque tool IDs survive parsing and tool-result continuation."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent import SimpleAgent
from agent_ark.agent.simple_agent.agent import _OpenAIModel
from agent_ark.agent.simple_agent.encoding import chat_messages
from agent_ark.agent.simple_agent.models import ChatCompletionsModel, ModelResponseError
from agent_ark.agent.simple_agent.types import (
    Message, ModelRequest, ModelUsage, NativeAssistantState, TextPart, ToolCall, ToolSpec,
)


ID_BYTE_LIMIT = 64 * 1024
TOOLS = (ToolSpec("list_files", "List available own records.", '{"type":"object"}'),)


def opaque_id(size):
    # Deliberately synthetic; no live provider signature belongs in test fixtures.
    prefix = "call_synthetic__mpthought__"
    return (prefix + "Synthetic+/=._-" * size)[:size]


def wire_call(call_id, *, name="list_files", arguments='{"path":"records"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def response(calls, **reasoning):
    return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "Inspect own records.", "tool_calls": calls, **reasoning}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return deepcopy(self.reply)


def model_for(adapter, reply):
    client = FakeClient(reply)
    if adapter == "generic":
        return ChatCompletionsModel(client, "synthetic-model"), client
    owner = SimpleNamespace(
        base_url="https://proxy.example/openapi", model="synthetic-model", extra_body={},
        output_token_parameter="max_completion_tokens", max_completion_tokens=None,
        temperature=None, _request_reasoning_effort=None, timeout_s=None, tool_choice="required",
    )
    model = _OpenAIModel(owner)
    model.client = client
    return model, client


@pytest.mark.parametrize("adapter", ["simple", "generic"])
@pytest.mark.parametrize("call_id", ["call-old-1", opaque_id(364), opaque_id(4096)],
                         ids=["legacy-short", "opaque-364-bytes", "opaque-4096-bytes"])
def test_provider_ids_parse_and_encode_without_changes(adapter, call_id):
    call = wire_call(call_id)
    model, _ = model_for(adapter, response([call]))
    completion = asyncio.run(model.complete(ModelRequest((Message.text("user", "Visible state"),), tools=TOOLS)))
    assert completion.tool_calls == (ToolCall(call_id, "list_files", call["function"]["arguments"]),)
    assert completion.usage == ModelUsage(2, 3, 5)
    assistant = Message("assistant", (TextPart(completion.text),), completion.tool_calls)
    result = Message("tool", (TextPart('{"files":[]}'),), tool_call_id=call_id)
    encoded = chat_messages((assistant, result))
    assert encoded[0]["tool_calls"] == [call]
    assert encoded[1]["tool_call_id"].encode("utf-8") == call_id.encode("utf-8")


@pytest.mark.parametrize("call_id", ["", None, 7, b"call-bytes", "x" * (ID_BYTE_LIMIT + 1)],
                         ids=["empty", "null", "integer", "bytes", "oversize"])
@pytest.mark.parametrize("adapter", ["simple", "generic"])
def test_invalid_ids_are_still_rejected_by_both_adapters(adapter, call_id):
    model, _ = model_for(adapter, response([wire_call(call_id)]))
    with pytest.raises(ModelResponseError):
        asyncio.run(model.complete(ModelRequest((Message.text("user", "Visible state"),), tools=TOOLS)))


@pytest.mark.parametrize("adapter", ["simple", "generic"])
def test_duplicate_opaque_ids_are_still_rejected(adapter):
    call = wire_call(opaque_id(364))
    model, _ = model_for(adapter, response([call, deepcopy(call)]))
    with pytest.raises(ModelResponseError):
        asyncio.run(model.complete(ModelRequest((Message.text("user", "Visible state"),), tools=TOOLS)))


@pytest.mark.parametrize("unit", ["x", "密", "😀"])
def test_tool_id_limit_counts_utf8_bytes_and_preserves_exact_boundary(unit):
    width = len(unit.encode("utf-8"))
    exact = unit * (ID_BYTE_LIMIT // width) + "x" * (ID_BYTE_LIMIT % width)
    assert len(exact.encode("utf-8")) == ID_BYTE_LIMIT
    assert ToolCall(exact, "list_files", "{}").call_id == exact
    with pytest.raises(ValueError, match="bound"):
        ToolCall(exact + "x", "list_files", "{}")


def test_other_tool_call_bounds_are_unchanged():
    ToolCall(opaque_id(364), "n" * 64, "x" * 131072)
    with pytest.raises(ValueError, match="bound"):
        ToolCall("call-old", "n" * 65, "{}")
    with pytest.raises(ValueError, match="bound"):
        ToolCall("call-old", "list_files", "x" * 131073)


@pytest.mark.parametrize("call_id", [opaque_id(364), "密" * (ID_BYTE_LIMIT // 3) + "x"],
                         ids=["opaque-364-bytes", "utf8-exact-boundary"])
def test_native_reasoning_reuses_opaque_id_validation_and_exact_replay(call_id):
    call = wire_call(call_id)
    reasoning = {"reasoning_details": [{"type": "reasoning.encrypted", "data": "SYNTHETIC-ONLY+/="}]}
    reply = response([call], **reasoning)
    model, _ = model_for("simple", reply)
    completion = asyncio.run(model.complete(ModelRequest((Message.text("user", "Visible state"),), tools=TOOLS)))
    native = completion.native_assistant
    assert native is not None
    assert native.tool_calls[0].call_id == call_id
    assistant = Message("assistant", (TextPart(completion.text),), completion.tool_calls, native_assistant=native)
    result = Message("tool", (TextPart('{"files":[]}'),), tool_call_id=call_id)
    replay = model.request_messages(ModelRequest((assistant, result), tools=TOOLS))
    assert replay[0] == reply["choices"][0]["message"]
    assert replay[1]["tool_call_id"].encode("utf-8") == call_id.encode("utf-8")

    original = native.message
    for calls in ([call, deepcopy(call)], [wire_call("")], [wire_call(None)],
                  [wire_call("x" * (ID_BYTE_LIMIT + 1))]):
        with pytest.raises(ValueError):
            NativeAssistantState(native.scope, json.dumps({**original, "tool_calls": calls}))


@pytest.mark.parametrize("native_reasoning", [False, True])
def test_real_sdk_host_tool_continuation_keeps_opaque_ids(tmp_path, native_reasoning):
    """Exercise the actual SDK wire path through list_files and submit_action."""
    received = []
    ids = [opaque_id(364), opaque_id(4096)]
    reasoning = {"reasoning_details": [{"type": "reasoning.encrypted", "data": "SYNTHETIC-ONLY+/="}]} if native_reasoning else {}
    action = {"name": "ExecutePlan", "arguments": {"plan": "U1,L9"}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            index = len(received) - 1
            name = "list_files" if index == 0 else "submit_action"
            arguments = {"path": "records", "offset": 0, "limit": 10} if index == 0 else {
                "format": "tool_call", "content": json.dumps(action)}
            payload = response([wire_call(ids[index], name=name, arguments=json.dumps(arguments))], **reasoning)
            payload.update({"id": "synthetic-response", "object": "chat.completion", "created": 0, "model": "synthetic-model"})
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent = None
    try:
        agent = SimpleAgent(api_key="synthetic-key", provider="openai", model="synthetic-model",
                            base_url=f"http://127.0.0.1:{server.server_port}/openapi", max_retries=0,
                            simple_agent_config={"tools": {"enabled": True, "workspace_parent": str(tmp_path)}})
        results, traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "Visible state"}]}})
        assert json.loads(results[0]["action"].removeprefix("<tool_call>").removesuffix("</tool_call>")) == action
        assert len(received) == 2
        continuation = received[1]["messages"]
        assistant = next(message for message in continuation if message.get("tool_calls"))
        result = next(message for message in continuation if message["role"] == "tool")
        assert assistant["tool_calls"][0]["id"].encode("utf-8") == ids[0].encode("utf-8")
        assert result["tool_call_id"].encode("utf-8") == ids[0].encode("utf-8")
        assert ("reasoning_details" in assistant) == native_reasoning
        if native_reasoning:
            assert assistant["reasoning_details"] == reasoning["reasoning_details"]
        exchanges = [entry for entry in agent._states[0].runtime.history_snapshot if entry.kind == "tool_exchange"]
        assert exchanges[0].messages[0].tool_calls[0].call_id == ids[0]
        assert exchanges[0].messages[1].tool_call_id == ids[0]
        assert traces[0]["simple_agent"]["native_action_submission"] is True
    finally:
        if agent is not None:
            agent.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
