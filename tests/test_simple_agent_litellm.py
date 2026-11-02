"""Offline contracts for the opt-in LiteLLM backend; no provider or Unity calls."""
from __future__ import annotations

import asyncio
import base64
import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from hashlib import sha256
import sys
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent import CallableModel, SimpleRuntime, SimpleRuntimeConfig, ToolRegistry
from agent_ark.agent.simple_agent.agent import SimpleAgent
from agent_ark.agent.simple_agent.encoding import encoded_messages, image_from_pixels
from agent_ark.agent.simple_agent.models import ModelResponseError, SummaryResponseError
from agent_ark.agent.simple_agent.runtime import _metrics
from agent_ark.agent.simple_agent.types import (
    AgentInputIdentity, AgentLimitError, AgentRequest, ArchiveEntry, LiteLLMAssistantState,
    Message, ModelCompletion, ModelRequest, ModelUsage, NativeAssistantState, TextPart, ToolCall, ToolSpec,
)


ACTION = '<tool_call>{"name":"tap","arguments":{"x":3}}</tool_call>'
PRIVATE = "SYNTHETIC_OPAQUE_CONTINUATION_ONLY"
REASONING_ITEMS = [
    {"type": "reasoning", "id": "rs-second", "encrypted_content": PRIVATE + "+/=\\n密文", "summary": [], "phase": "analysis"},
    {"type": "reasoning", "id": "rs-first", "summary": [{"type": "summary_text", "text": "Synthetic summary."}], "phase": "commentary"},
]
THINKING_BLOCKS = [
    {"type": "thinking", "thinking": PRIVATE, "signature": "synthetic-signed+/="},
    {"type": "redacted_thinking", "data": "synthetic-redacted+/="},
]
USAGE = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12,
         "completion_tokens_details": {"reasoning_tokens": 4}}


def wire_call(call, **extra):
    return {"id": call.call_id, "type": "function", "function": {
        "name": call.name, "arguments": call.arguments_json}, **extra}


def reply(content=ACTION, *, calls=(), finish=None, **fields):
    return {"choices": [{"finish_reason": finish or ("tool_calls" if calls else "stop"), "message": {
        "role": "assistant", "content": content,
        **({"tool_calls": list(calls)} if calls else {}), **fields}}], "usage": deepcopy(USAGE)}


def responses_reply(content="Visible rationale.", *, call=None, status="completed", refusal=False):
    output = [deepcopy(REASONING_ITEMS[0]),
              {"type": "message", "id": "msg-synthetic", "role": "assistant", "status": "completed", "phase": "commentary",
               "content": [{"type": "refusal", "refusal": "Refused"}] if refusal else
                          [{"type": "output_text", "text": content, "annotations": []}]},
              deepcopy(REASONING_ITEMS[1])]
    if call is not None:
        output.append({"type": "function_call", "id": "fc-" + call.call_id, "call_id": call.call_id,
                       "name": call.name, "arguments": call.arguments_json, "status": "completed"})
    return {"id": "resp-synthetic", "object": "response", "model": "test-model", "status": status,
            "output": output, "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
            **({"incomplete_details": {"reason": "max_output_tokens"}} if status == "incomplete" else {})}


class FakeLiteLLM:
    """The public async SDK entry point, with fully detached request captures."""
    def __init__(self, replies):
        self.replies = replies if callable(replies) else iter(replies)
        self.requests = []

    async def acompletion(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        value = self.replies(kwargs) if callable(self.replies) else next(self.replies)
        if isinstance(value, BaseException):
            raise value
        return deepcopy(value)

    aresponses = acompletion


def make_agent(monkeypatch, replies, *, config=None, **kwargs):
    sdk = FakeLiteLLM(replies)
    monkeypatch.setitem(sys.modules, "litellm", sdk)
    agent = SimpleAgent(**{"api_key": "synthetic-explicit-key", "temperature": None, **kwargs},
                        simple_agent_config={"backend": "litellm", "litellm": {"model": "openai/test-model"},
                                             **(config or {})})
    return agent, sdk


def obs(text="Visible state", index=0):
    return {index: {"messages": [{"role": "user", "content": text}]}}


def request(number=1, text="Visible state"):
    return AgentRequest(AgentInputIdentity("run", "session", "episode", "agent", 1, str(number)),
                        (Message.text("user", text),))


def state(content="answer", *, calls=(), scope="litellm:synthetic", **fields):
    payload = {"role": "assistant", "content": content,
               **({"tool_calls": [wire_call(call) for call in calls]} if calls else {}),
               **(fields or {"reasoning_items": deepcopy(REASONING_ITEMS)})}
    return LiteLLMAssistantState(scope, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def text_of(messages):
    return "\n".join(part.text for message in messages for part in message.parts if isinstance(part, TextPart))


def wire_text(messages):
    return "\n".join(content if isinstance(content := message.get("content"), str)
                     else "\n".join(part.get("text", "") for part in content or []) for message in messages)


def test_native_default_runs_without_importing_optional_litellm(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "litellm" or name.startswith("litellm."):
            raise AssertionError("native backend imported its optional alternative")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    agent = SimpleAgent(api_key="synthetic-native-key", provider="openai")
    async def create(**kwargs):
        return reply()
    async def close():
        pass
    agent._backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close)
    try:
        assert agent.forward(obs())[0] == ACTION
        assert "litellm" not in agent.configuration_summary()
    finally:
        agent.close()


def test_missing_optional_dependency_fails_clearly_only_when_selected(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)
    agent = None
    try:
        with pytest.raises((ImportError, RuntimeError), match="(?i)litellm"):
            agent = SimpleAgent(simple_agent_config={"backend": "litellm", "litellm": {"model": "openai/test-model"}})
            agent.forward(obs())
    finally:
        if agent is not None:
            agent.close()


@pytest.mark.parametrize("config", [
    {"backend": "other"},
    {"backend": "litellm"},
    {"backend": "litellm", "litellm": {"model": "unqualified"}},
    {"backend": "litellm", "litellm": {"model": "openai/test", "api": "automatic"}},
    {"backend": "litellm", "litellm": {"model": "openai/responses/test", "api": "chat_completions"}},
    {"backend": "litellm", "litellm": {"model": "anthropic/test", "api": "responses"}},
    {"backend": "litellm", "litellm": {"model": "openai/test", "api_key": "hidden-in-config"}},
    {"backend": "litellm", "litellm": {"model": "openai/test", "drop_params": True}},
    {"backend": "litellm", "litellm": {"model": "openai/test", "fallbacks": ["openai/other"]}},
])
def test_backend_config_is_explicit_and_does_not_accept_arbitrary_sdk_kwargs(config):
    with pytest.raises((ValueError, TypeError)):
        SimpleAgent(simple_agent_config=config)


def test_litellm_rejects_legacy_thinking_parameter():
    with pytest.raises(ValueError, match="(?i)thinking"):
        SimpleAgent(thinking={"type": "enabled"}, simple_agent_config={
            "backend": "litellm", "litellm": {"model": "openai/test-model"}})


@pytest.mark.parametrize("api,model,expected", [
    ("chat_completions", "anthropic/test", "anthropic/test"),
])
def test_sdk_routing_output_budget_tools_and_image_use_existing_runtime(monkeypatch, api, model, expected):
    agent, sdk = make_agent(monkeypatch, [reply()], base_url="https://synthetic.invalid/v1", reasoning_effort="high",
                            max_completion_tokens=31, timeout_s=25, max_retries=2,
                            config={"litellm": {"model": model, "api": api}})
    image = image_from_pixels(b"\x01\x02\x03", 1, 1)
    tool = ToolSpec("echo", "Return visible own text.", '{"type":"object"}')
    try:
        completion = asyncio.run(agent._backend.complete(ModelRequest((Message("user", (TextPart("Visible state"), image)),),
                                                                      tools=(tool,), max_output_tokens=99)))
        kwargs = sdk.requests[0]
        assert kwargs["model"] == expected
        assert kwargs["api_base"] == "https://synthetic.invalid/v1"
        assert kwargs["api_key"] == "synthetic-explicit-key"
        assert kwargs["max_completion_tokens"] == 31
        assert kwargs["reasoning_effort"] == "high" and kwargs["timeout"] == 25
        assert kwargs["num_retries"] is None and kwargs["max_retries"] == 2
        assert kwargs["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert kwargs["tools"][0]["function"]["name"] == "echo"
        assert completion.usage == ModelUsage(5, 7, 12)
    finally:
        agent.close()


def test_litellm_leaves_provider_environment_auth_resolution_to_sdk(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-other-provider-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "synthetic-legacy-key")
    agent, sdk = make_agent(monkeypatch, [reply()], api_key=None)
    try:
        agent.forward(obs())
        assert "api_key" not in sdk.requests[0] and "api_base" not in sdk.requests[0]
        assert "synthetic-other-provider-key" not in json.dumps(agent.configuration_summary())
        assert "synthetic-legacy-key" not in json.dumps(agent.configuration_summary())
    finally:
        agent.close()


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_audit_redacts_api_key_and_url_credentials_while_sdk_receives_original_connection(monkeypatch, api):
    endpoint = "https://synthetic-user:synthetic-password@synthetic.invalid/v1?api_key=synthetic-query-secret#synthetic-fragment"
    agent, sdk = make_agent(monkeypatch, [responses_reply(ACTION) if api == "responses" else reply()],
                            base_url=endpoint, config={"litellm": {"model": "openai/test-model", "api": api}})
    try:
        result, traces = agent.forward_with_trace(obs())
        assert result[0]["action"] == ACTION
        assert sdk.requests[0]["api_base"] == endpoint
        assert sdk.requests[0]["api_key"] == "synthetic-explicit-key"
        call = traces[0]["model_requests"][0]
        assert "api_base" not in call["parameters"] and "api_key" not in call["parameters"]
        assert call["request_transport"]["endpoint"] == "https://synthetic.invalid/v1"
        assert call["request_transport"]["kind"] == "logical_sdk_request"
        audit = json.dumps({"trace": traces, "configuration": agent.configuration_summary()})
        assert all(secret not in audit for secret in (
            "synthetic-explicit-key", "synthetic-user", "synthetic-password", "synthetic-query-secret", "synthetic-fragment"))
    finally:
        agent.close()


def test_responses_output_budget_below_sdk_minimum_fails_before_provider_call(monkeypatch):
    agent, sdk = make_agent(monkeypatch, [responses_reply(ACTION)], config={
        "litellm": {"model": "openai/test-model", "api": "responses"}})
    try:
        with pytest.raises(ValueError, match="(?i)(16|output.*budget)"):
            asyncio.run(agent._backend.complete(ModelRequest((Message.text("user", "Visible"),), max_output_tokens=15)))
        assert not sdk.requests
    finally:
        agent.close()


@pytest.mark.parametrize("name,value", [("drop_params", True), ("modify_params", True),
                                        ("model_fallbacks", ["openai/other"]), ("num_retries", 1)])
def test_global_sdk_fallbacks_and_parameter_rewrites_fail_closed_without_mutation(monkeypatch, name, value):
    agent, sdk = make_agent(monkeypatch, [reply()])
    setattr(sdk, name, deepcopy(value))
    try:
        with pytest.raises(ValueError, match="(?i)(global|fallback|modif|retr)"):
            agent.forward(obs())
        assert not sdk.requests and getattr(sdk, name) == value
    finally:
        agent.close()


@pytest.mark.parametrize("model,expected", [("openai/openai/gpt6", "openai/openai/gpt6"),
                                            ("openai/responses/test", "openai/test")])
def test_native_responses_api_replays_original_output_items_in_order_and_uses_call_id(monkeypatch, tmp_path, model, expected):
    host = ToolCall("host-opaque+/=" + "signed_" * 20, "list_files", '{"path":"records","offset":0,"limit":10}')
    submit = ToolCall("selection-one", "submit_action", json.dumps({"format": "tool_call", "content": ACTION[11:-12]}))
    first = responses_reply(call=host)
    second = responses_reply("Tap the visible point.", call=submit)
    third = responses_reply("Tap again.", call=replace(submit, call_id="selection-two"))
    agent, sdk = make_agent(monkeypatch, [first, second, third], config={
        "litellm": {"model": model, "api": "responses"},
        "tools": {"enabled": True, "workspace_parent": str(tmp_path)}})
    try:
        result, traces = agent.forward_with_trace(obs())
        assert result[0]["action"] == ACTION
        initial, continued = sdk.requests
        assert initial["model"] == expected and "messages" not in initial
        assert initial["store"] is False
        assert initial["include"] == ["reasoning.encrypted_content"]
        assert all(tool["type"] == "function" and tool["strict"] is False for tool in initial["tools"])
        offset = len(initial["input"])
        assert continued["input"][:offset] == initial["input"]
        assert continued["input"][offset:offset + len(first["output"])] == first["output"]
        result_item = continued["input"][offset + len(first["output"])]
        assert result_item["type"] == "function_call_output" and result_item["call_id"] == host.call_id
        assert "tool_call_id" not in result_item
        native = traces[0]["model_requests"][0]["native_assistant"]["message"]
        assert native["responses_output"] == first["output"]
        agent.forward(obs("Actual environment feedback."))
        next_items = sdk.requests[-1]["input"]
        first_start = next(index for index, item in enumerate(next_items) if item.get("id") == first["output"][0]["id"])
        assert next_items[first_start:first_start + len(first["output"])] == first["output"]
        assert PRIVATE not in result[0]["assistant"]
    finally:
        agent.close()


@pytest.mark.parametrize("status,refusal", [("incomplete", False), ("failed", False), ("completed", True)])
def test_responses_incomplete_failed_and_refused_actions_are_billed_but_rejected(monkeypatch, status, refusal):
    agent, _ = make_agent(monkeypatch, [responses_reply(status=status, refusal=refusal)],
                          config={"litellm": {"model": "openai/test-model", "api": "responses"}})
    try:
        with pytest.raises(ModelResponseError):
            agent.forward(obs())
        assert agent.last_trace_by_agent[0]["usage"]["total_tokens"] == 12
        assert all(entry.kind != "completion" for entry in agent._states[0].runtime.history_snapshot)
    finally:
        agent.close()


@pytest.mark.parametrize("bad_id", [None, "", "duplicate"])
def test_responses_tool_calls_require_valid_unique_call_ids(monkeypatch, bad_id):
    value = responses_reply(call=ToolCall("call-trusted", "echo", "{}"))
    if bad_id == "duplicate":
        value["output"].append(deepcopy(value["output"][-1]))
    else:
        value["output"][-1]["call_id"] = bad_id
    agent, _ = make_agent(monkeypatch, [value], config={"litellm": {"model": "openai/test", "api": "responses"}})
    try:
        with pytest.raises(ModelResponseError):
            asyncio.run(agent._backend.complete(ModelRequest((Message.text("user", "Visible"),),
                tools=(ToolSpec("echo", "Return visible text.", '{"type":"object"}'),))))
    finally:
        agent.close()


@pytest.mark.parametrize("status", [None, "completed", "in_progress", "incomplete"])
def test_responses_nullable_function_status_preserves_completed_response_contract(monkeypatch, status):
    call = ToolCall("call-nullable-status", "echo", "{}")
    value = responses_reply(call=call)
    value["output"][-1]["status"] = status
    agent, _ = make_agent(monkeypatch, [value], config={
        "litellm": {"model": "openai/test-model", "api": "responses"}})
    request = ModelRequest((Message.text("user", "Visible"),),
                           tools=(ToolSpec("echo", "Return visible text.", '{"type":"object"}'),))
    try:
        if status in ("in_progress", "incomplete"):
            with pytest.raises(ModelResponseError):
                asyncio.run(agent._backend.complete(request))
        else:
            completion = asyncio.run(agent._backend.complete(request))
            assert completion.tool_calls == (call,)
            assert completion.native_assistant.message["responses_output"] == value["output"]
            assistant = Message("assistant", (TextPart(completion.text),), completion.tool_calls,
                                native_assistant=completion.native_assistant)
            continued = agent._backend.request_messages(ModelRequest(request.messages + (assistant,), tools=request.tools))
            assert continued[-len(value["output"]):] == value["output"]
    finally:
        agent.close()


@pytest.mark.parametrize("message_status", ["completed", "incomplete"])
def test_responses_output_token_limit_is_a_recoverable_summary_failure_with_usage(monkeypatch, message_status):
    failed = responses_reply("FAILED_SUMMARY", status="incomplete")
    failed["output"][1]["status"] = message_status
    agent, sdk = make_agent(monkeypatch, [failed, responses_reply("Earlier useful facts.")], config={
        "litellm": {"model": "openai/test-model", "api": "responses"}})
    try:
        value = ModelRequest((Message.text("user", "Summarize earlier visible history."),), purpose="summary", max_output_tokens=256)
        with pytest.raises(SummaryResponseError) as error:
            asyncio.run(agent._backend.complete(value))
        assert error.value.usage == ModelUsage(5, 7, 12)
        completion = asyncio.run(agent._backend.complete(value))
        assert completion.text == "Earlier useful facts." and completion.native_assistant is None
        assert len(sdk.requests) == 2 and all(not call.get("tools") for call in sdk.requests)
    finally:
        agent.close()


def test_responses_observation_image_is_deduplicated_in_saved_audit_and_preserved_by_replay(monkeypatch, tmp_path):
    from agent_ark.ark_eval.run_api_agent import forward_agent_with_trace
    from agent_ark.ark_eval.run_replay import ReplayAgent

    host = ToolCall("host-image-record", "list_files", '{"path":"records","offset":0,"limit":10}')
    submit = ToolCall("submit-image-action", "submit_action", json.dumps({"format": "tool_call", "content": ACTION[11:-12]}))
    agent, sdk = make_agent(monkeypatch, [responses_reply(call=host), responses_reply(call=submit)], config={
        "litellm": {"model": "openai/test-model", "api": "responses"},
        "tools": {"enabled": True, "workspace_parent": str(tmp_path)}})
    image = image_from_pixels(b"\x01\x02\x03", 1, 1)
    url = "data:image/png;base64," + base64.b64encode(image.data).decode("ascii")
    digest = sha256(image.data).hexdigest()
    observation = {0: {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Visible image state"}, {"type": "image_url", "image_url": {"url": url}}]}]}}
    assets = {}
    try:
        actions, saved = forward_agent_with_trace(agent, observation, image_assets=assets)
        assert actions[0]["action"] == ACTION
        assert list(assets) == [digest] and assets[digest]["data_url"] == url
        assert "data:image" not in json.dumps(saved)
        expected_reference = "agentark-trace-image:" + digest
        for call in saved["0"]["model_requests"]:
            parts = [part for item in call["request_messages"] for part in item.get("content", [])
                     if isinstance(part, dict) and part.get("type") == "input_image"]
            assert len(parts) == 1 and parts[0]["image_url"] == expected_reference
        assert url in json.dumps(sdk.requests)
        record = {"steps": [{"turn_index": 0, "raw_response_by_agent": saved}], "model_trace_images": assets}
        original = deepcopy(record)
        replay = ReplayAgent(record)
        replay_actions, replay_trace = replay.forward_with_trace({0: {}})
        assert replay_actions[0]["action"] == ACTION
        assert replay_trace[0]["model_requests"] == saved["0"]["model_requests"]
        assert replay.model_trace_images == assets and record == original
        assert len(sdk.requests) == 2
    finally:
        agent.close()


@pytest.mark.parametrize("fields", [{"reasoning_items": REASONING_ITEMS}, {"thinking_blocks": THINKING_BLOCKS}])
def test_native_host_tools_and_submission_replay_all_provider_state_into_next_environment_step(monkeypatch, tmp_path, fields):
    opaque_id = "synthetic-call__mpthought__" + "Signed+/=" * 70
    host = ToolCall(opaque_id, "list_files", '{"path":"records","offset":0,"limit":10}')
    submit = ToolCall("selection-one", "submit_action", json.dumps({"format": "tool_call", "content": ACTION[11:-12]}))
    call_extras = {"provider_specific_fields": {"thought_signature": "synthetic-tool-signature+/="}}
    first = reply(None, calls=(wire_call(host, **call_extras),), provider_specific_fields={"phase": "analysis"}, **fields)
    second = reply("Tap the visible point.", calls=(wire_call(submit, **call_extras),), **fields)
    third = reply("Tap again.", calls=(wire_call(replace(submit, call_id="selection-two")),), **fields)
    agent, sdk = make_agent(monkeypatch, [first, second, third], config={
        "tools": {"enabled": True, "workspace_parent": str(tmp_path)}})
    try:
        result, traces = agent.forward_with_trace(obs())
        assert result[0]["action"] == ACTION
        assert PRIVATE not in result[0]["assistant"]
        assert sdk.requests[1]["messages"][:len(sdk.requests[0]["messages"])] == sdk.requests[0]["messages"]
        assert sdk.requests[1]["messages"][-2] == first["choices"][0]["message"]
        assert sdk.requests[1]["messages"][-1]["tool_call_id"] == opaque_id
        assert all(tool["function"]["name"] for tool in sdk.requests[0]["tools"])
        assert sdk.requests[0]["tool_choice"] == "required"
        assert traces[0]["usage"] == {"input_tokens": 10, "output_tokens": 14, "total_tokens": 24}
        assert traces[0]["model_requests"][0]["native_assistant"]["message"] == first["choices"][0]["message"]
        assert "synthetic-explicit-key" not in json.dumps(traces)
        assert "synthetic-explicit-key" not in json.dumps(agent.configuration_summary())
        agent.forward(obs("Actual environment feedback: selected."))
        replay = [message for message in sdk.requests[2]["messages"] if message.get("tool_calls")]
        assert replay == [first["choices"][0]["message"], second["choices"][0]["message"]]
        assert any(message.get("tool_call_id") == submit.call_id for message in sdk.requests[2]["messages"])
        files = dict(agent._states[0].workspace._records["turn-000001"][0].files)
        assert PRIVATE.encode() not in files["assistant.txt"]
        assert PRIVATE.encode() not in files["messages.json"]
        assert json.loads(files["native_assistants.json"])["calls"][0]["message"] == first["choices"][0]["message"]
    finally:
        agent.close()


def test_invalid_submission_preserves_entire_batch_and_recovers_without_host_side_effects(monkeypatch, tmp_path):
    host = ToolCall("must-not-execute", "write_note", '{"path":"notes/forbidden.txt","content":"SIDE EFFECT"}')
    bad = ToolCall("bad-selection", "submit_action", '{"format":"raw","content":"a"}')
    good = ToolCall("good-selection", "submit_action", '{"format":"raw","content":"a"}')
    original = reply("Visible rationale.", calls=(wire_call(host), wire_call(bad)), reasoning_items=REASONING_ITEMS)
    agent, sdk = make_agent(monkeypatch, [original, reply("Corrected.", calls=(wire_call(good),))], config={
        "tools": {"enabled": True, "workspace_parent": str(tmp_path)}})
    try:
        result, traces = agent.forward_with_trace(obs())
        assert result[0]["action"] == "a"
        continuation = sdk.requests[1]["messages"][len(sdk.requests[0]["messages"]):]
        assert continuation[0] == original["choices"][0]["message"]
        assert [message["tool_call_id"] for message in continuation[1:]] == [host.call_id, bad.call_id]
        assert all(json.loads(message["content"])["executed"] is False for message in continuation[1:])
        assert [record["status"] for record in traces[0]["model_requests"]] == ["rejected", "completed"]
        assert traces[0]["usage"]["total_tokens"] == 24
        assert not list(tmp_path.rglob("forbidden.txt"))
    finally:
        agent.close()


@pytest.mark.parametrize("mutate", [
    lambda calls: [{**calls[0], "id": ""}],
    lambda calls: [{**calls[0], "id": None}],
    lambda calls: [calls[0], deepcopy(calls[0])],
    lambda calls: [{**calls[0], "type": "custom"}],
])
def test_opaque_replay_does_not_relax_tool_id_or_function_validation(monkeypatch, mutate):
    call = ToolCall("synthetic-opaque+/=", "echo", "{}")
    agent, _ = make_agent(monkeypatch, [reply(None, calls=mutate([wire_call(call)]), thinking_blocks=THINKING_BLOCKS)])
    try:
        with pytest.raises(ModelResponseError):
            asyncio.run(agent._backend.complete(ModelRequest((Message.text("user", "Visible"),),
                tools=(ToolSpec("echo", "Return visible text.", '{"type":"object"}'),))))
    finally:
        agent.close()


@pytest.mark.parametrize("fields,finish", [({"refusal": "Refused"}, "stop"), ({}, "length"), ({}, "content_filter")])
def test_rejected_or_incomplete_actions_keep_billed_usage_but_never_become_actions(monkeypatch, fields, finish):
    agent, _ = make_agent(monkeypatch, [reply(finish=finish, **fields)])
    try:
        with pytest.raises(ModelResponseError):
            agent.forward(obs())
        trace = agent.last_trace_by_agent[0]
        assert trace["model_requests"][0]["status"] == "failed"
        assert trace["usage"]["total_tokens"] == 12
        assert all(entry.kind != "completion" for entry in agent._states[0].runtime.history_snapshot)
    finally:
        agent.close()


def test_scope_mismatch_fails_closed_and_summaries_exclude_all_opaque_state(monkeypatch):
    agent, sdk = make_agent(monkeypatch, [reply("Earlier facts.", reasoning_items=REASONING_ITEMS)])
    native = state(ACTION, scope="another-endpoint-and-model")
    assistant = Message("assistant", (TextPart(ACTION),), native_assistant=native)
    try:
        with pytest.raises(ModelResponseError, match="(?i)(scope|endpoint|model)"):
            agent._backend.request_messages(ModelRequest((assistant,)))
        summary = asyncio.run(agent._backend.complete(ModelRequest((assistant,), purpose="summary")))
        assert summary.text == "Earlier facts." and summary.native_assistant is None
        assert PRIVATE not in json.dumps(sdk.requests[0]["messages"])
        assert all("reasoning_items" not in message for message in sdk.requests[0]["messages"])
        assert not sdk.requests[0].get("tools")
    finally:
        agent.close()


def test_summary_failure_retries_with_usage_and_without_failed_or_opaque_content(monkeypatch):
    seen_summaries = []
    def respond(kwargs):
        if wire_text(kwargs["messages"]).startswith("Summarize this agent's earlier complete conversation turns."):
            seen_summaries.append(kwargs)
            return reply("FAILED_SUMMARY" if len(seen_summaries) == 1 else "Earlier useful facts.",
                         finish="length" if len(seen_summaries) == 1 else "stop", thinking_blocks=THINKING_BLOCKS)
        return reply(reasoning_items=REASONING_ITEMS)
    agent, _ = make_agent(monkeypatch, respond, config={"context": {
        "max_context_characters": 2400, "max_context_tokens": 10000, "output_reserve_tokens": 64,
        "recent_turns": 1, "summary_output_tokens": 256}})
    try:
        trace = None
        for turn in range(8):
            _, traces = agent.forward_with_trace(obs("Current observation (step_msg):\n" + str(turn) + "x" * 480))
            if traces[0]["agent_usage"]["summary_calls"]:
                trace = traces[0]
                break
        assert trace is not None and len(seen_summaries) >= 2
        assert "Summary retry instructions:" in wire_text(seen_summaries[1]["messages"])
        assert "FAILED_SUMMARY" not in wire_text(seen_summaries[1]["messages"])
        assert PRIVATE not in json.dumps(seen_summaries)
        assert [record["status"] for record in trace["model_requests"][:2]] == ["failed", "completed"]
        assert trace["model_requests"][0]["error_type"] == "SummaryResponseError"
        assert trace["agent_usage"]["summary"]["total_tokens"] >= 24
    finally:
        agent.close()


def test_agents_and_reset_do_not_share_opaque_conversation_or_provider_cache(monkeypatch):
    counter = iter(range(10))
    def respond(kwargs):
        return reply(reasoning_items=[{"type": "reasoning", "id": "rs-" + str(next(counter)),
                                       "encrypted_content": PRIVATE, "summary": []}])
    agent, sdk = make_agent(monkeypatch, respond)
    try:
        agent.forward({**obs("Agent zero visible state", 0), **obs("Agent one visible state", 1)})
        assert "Agent zero" not in json.dumps(sdk.requests[1]["messages"])
        assert PRIVATE not in json.dumps(sdk.requests[1]["messages"])
        agent.forward(obs("Agent zero feedback", 0))
        assert "rs-0" in json.dumps(sdk.requests[2]["messages"])
        assert "rs-1" not in json.dumps(sdk.requests[2]["messages"])
        agent.close()
        agent.reset()
        agent.forward(obs("Fresh case", 0))
        assert PRIVATE not in json.dumps(sdk.requests[3]["messages"])
        assert "Agent zero" not in json.dumps(sdk.requests[3]["messages"])
    finally:
        agent.close()


def test_litellm_state_is_frozen_detached_bounded_and_keeps_native_contract_separate():
    native = state()
    changed = native.message
    changed["reasoning_items"].reverse()
    assert native.message["reasoning_items"] == REASONING_ITEMS
    with pytest.raises(FrozenInstanceError):
        native.scope = "changed"
    with pytest.raises(ValueError):
        NativeAssistantState(native.scope, native.message_json)
    with pytest.raises((ValueError, TypeError)):
        state(reasoning_items=[{"encrypted_content": "x" * (1024 * 1024)}])
    with pytest.raises(ValueError):
        ModelCompletion("wrong text", native_assistant=native)
    with pytest.raises(ValueError):
        AgentRequest(request().identity, (Message("assistant", (TextPart("answer"),), native_assistant=native),))


def test_litellm_state_counts_toward_context_archive_tokens_and_output_limits():
    async def run():
        native = state(reasoning_items=[{"type": "reasoning", "encrypted_content": PRIVATE + "密" * 400}])
        messages = (Message("assistant", (TextPart("answer"),), native_assistant=native),)
        plain = (Message.text("assistant", "answer"),)
        runtime = SimpleRuntime(CallableModel(lambda _: None))
        assert _metrics(messages)[0] >= _metrics(plain)[0] + len(native.message_json)
        assert _metrics(messages)[1] >= _metrics(plain)[1] + len(native.message_json.encode())
        assert runtime._tokens(messages, _metrics(messages)[1]) > runtime._tokens(plain, _metrics(plain)[1])
        archived = ArchiveEntry(1, "completion", request().identity, messages, text="answer")
        assert runtime._archive_cost(archived)[1] > runtime._archive_cost(replace(archived, messages=plain))[1]
        assert PRIVATE not in encoded_messages(messages).decode()
        async def model(req):
            return ModelCompletion("answer", native_assistant=native)
        bounded = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(max_action_output_bytes=100))
        with pytest.raises(AgentLimitError, match="action output"):
            await bounded.complete(request())
        assert bounded.history_snapshot[-1].messages[0].native_assistant == native
    asyncio.run(run())


def test_compaction_strips_litellm_working_state_preserves_archive_and_summary_boundary():
    async def run():
        seen = []
        async def model(req):
            seen.append(req)
            if req.purpose == "summary":
                assert all(message.native_assistant is None for message in req.messages)
                assert PRIVATE not in text_of(req.messages)
                return ModelCompletion("Earlier complete turns retained.")
            return ModelCompletion("answer", native_assistant=state("answer"))
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            max_context_characters=1800, max_context_tokens=20000, output_reserve_tokens=64,
            recent_turns=1, summary_output_tokens=128))
        for number in range(1, 4):
            await runtime.complete(request(number, str(number) + "x" * 500))
        assert any(req.purpose == "summary" for req in seen)
        assert all(message.native_assistant is None for message in seen[-1].messages)
        assert any(entry.kind == "completion" and entry.messages[0].native_assistant is not None
                   for entry in runtime.history_snapshot)
        assert all(not entry.messages for entry in runtime.history_snapshot if entry.kind == "summary_completion")
    asyncio.run(run())


def test_current_litellm_tool_continuation_cannot_strip_opaque_prefix_to_fit():
    async def run():
        call = ToolCall("current-host", "echo", "{}")
        tools = ToolRegistry()
        tools.register(ToolSpec("echo", "Return visible own text.", '{"type":"object"}'),
                       lambda args, session: (TextPart("large result " + "x" * 1800),))
        seen = []
        async def model(req):
            seen.append(req)
            return ModelCompletion("", tool_calls=(call,), native_assistant=state(None, calls=(call,)))
        runtime = SimpleRuntime(CallableModel(model), SimpleRuntimeConfig(
            max_context_characters=1800, max_context_tokens=20000, output_reserve_tokens=64, recent_turns=1), tools=tools)
        with pytest.raises(AgentLimitError, match="(?i)(native.*continuation|continuation.*native|native.*prefix)"):
            await runtime.complete(request())
        assert len(seen) == 1 and runtime.usage.summary_calls == 0
        exchange = next(entry for entry in runtime.history_snapshot if entry.kind == "tool_exchange")
        assert exchange.messages[0].native_assistant is not None
        assert exchange.messages[0] in runtime.working_snapshot
    asyncio.run(run())
