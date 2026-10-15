"""Offline Codex transport contracts; these tests never launch a real CLI."""
import asyncio
import json
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_ark.agent.agent_registry import AgentType, configure_agent_environment, resolve_agent_type
from agent_ark.agent.simple_agent.agent import SimpleAgent
from agent_ark.agent.simple_agent.codex_model import CodexModel
from agent_ark.agent.simple_agent.encoding import encode_png
from agent_ark.agent.simple_agent.models import ModelResponseError, SummaryResponseError
from agent_ark.agent.simple_agent.types import ImagePart, Message, ModelRequest, TextPart, ToolSpec
from agent_ark.ark_eval.run_api_agent import build_error_result, build_model_runtimes, normalize_response_trace


def event(method, payload):
    return SimpleNamespace(method=method, payload=payload)


def envelope(text="", calls=()):
    return json.dumps({"text": text, "tool_calls": list(calls)}, ensure_ascii=False)


def call(name, args, id="call_1"):
    return {"id": id, "name": name, "arguments_json": json.dumps(args)}


def successful(raw, *, reasoning=None, usage=True):
    items = []
    if reasoning:
        items.append(event("item/completed", {"item": {
            "type": "reasoning", "id": "reason", "content": [reasoning], "summary": []}}))
    items += [event("item/completed", {"item": {
        "type": "agentMessage", "id": "answer", "phase": "final_answer", "text": raw}})]
    if usage:
        items.append(event("thread/tokenUsage/updated", {"tokenUsage": {
            "last": {"inputTokens": 7, "outputTokens": 4, "totalTokens": 11,
                     "cachedInputTokens": 3, "reasoningOutputTokens": 2},
            "total": {"inputTokens": 17, "outputTokens": 14, "totalTokens": 31,
                      "cachedInputTokens": 8, "reasoningOutputTokens": 9},
            "modelContextWindow": 200000}}))
    items.append(event("turn/completed", {"turn": {"id": "turn", "status": "completed"}}))
    return items


def sdk_error(*, retry, code="responseStreamDisconnected", message="Connection failed", details=None, status=502):
    info = {code: {"httpStatusCode": status}} if code.startswith(("http", "response")) else code
    return event("error", {"error": {"message": message, "codexErrorInfo": info,
                                    "additionalDetails": details}, "willRetry": retry})


@pytest.fixture
def fake_sdk(tmp_path, monkeypatch):
    auth_home = tmp_path / "source-home"
    auth_home.mkdir()
    (auth_home / "auth.json").write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"fake": "test"}}))
    (auth_home / "config.toml").write_text("do_not_inherit = true")
    (auth_home / "AGENTS.md").write_text("DO NOT INHERIT USER INSTRUCTIONS")
    monkeypatch.setenv("CODEX_HOME", str(auth_home))
    state = SimpleNamespace(scripts=[], instances=[], starts=[], requests=[], interrupts=[], closes=[],
                            entered=asyncio.Event(), auth_inputs=[], refresh_tokens=None,
                            skill_names=[], disabled_skills=set(), disables_work=True)

    class Turn:
        id = "turn"

        def __init__(self, script):
            self.script = script
            self.stopped = asyncio.Event()

        async def interrupt(self):
            state.interrupts.append(self)
            self.stopped.set()

        async def stream(self):
            state.entered.set()
            if self.script == "block":
                await self.stopped.wait()
                yield event("turn/completed", {"turn": {"id": "turn", "status": "interrupted"}})
                return
            for item in self.script:
                if isinstance(item, BaseException):
                    raise item
                if item == "block":
                    await self.stopped.wait()
                    yield event("turn/completed", {"turn": {"id": "turn", "status": "interrupted"}})
                    return
                yield item

    class Thread:
        def __init__(self, index):
            self.id = f"thread_{index}"

        async def turn(self, inputs, **kwargs):
            state.requests.append((inputs, kwargs))
            return Turn(state.scripts.pop(0))

    class Codex:
        def __init__(self, *, config):
            self.config = config
            self._client = SimpleNamespace(request=self.request, start=self.start)
            state.instances.append(self)

        async def start(self):
            pass

        async def __aenter__(self):
            await self._client.start()
            return self

        async def request(self, method, params, **kwargs):
            if method == "skills/config/write":
                if state.disables_work:
                    state.disabled_skills.add(params["path"])
                return {"effectiveEnabled": not state.disables_work}
            assert method == "skills/list"
            paths = [str(Path(self.config.env["CODEX_HOME"]) / "skills" / name / "SKILL.md")
                     for name in state.skill_names]
            return {"data": [{"cwd": self.config.cwd, "errors": [], "skills": [
                {"path": path, "enabled": path not in state.disabled_skills} for path in paths]}]}

        async def thread_start(self, **kwargs):
            assert list(Path(kwargs["cwd"]).iterdir()) == []
            home = Path(self.config.env["CODEX_HOME"])
            assert {path.name for path in home.iterdir()} == {"auth.json"}
            state.auth_inputs.append(json.loads((home / "auth.json").read_text()))
            assert self.config.env["OPENAI_API_KEY"] == ""
            state.starts.append(kwargs)
            return Thread(len(state.instances))

        async def close(self):
            if state.refresh_tokens is not None:
                (Path(self.config.env["CODEX_HOME"]) / "auth.json").write_text(
                    json.dumps({"auth_mode": "chatgpt", "tokens": state.refresh_tokens}))
            state.closes.append(self)

    sdk = SimpleNamespace(AsyncCodex=Codex, CodexConfig=lambda **kwargs: SimpleNamespace(**kwargs),
                          TextInput=lambda **kwargs: {"type": "text", **kwargs},
                          ImageInput=lambda **kwargs: {"type": "image", **kwargs},
                          ApprovalMode=SimpleNamespace(deny_all="deny_all"), Sandbox=SimpleNamespace(read_only="read_only"))
    monkeypatch.setitem(sys.modules, "openai_codex", sdk)
    monkeypatch.setitem(sys.modules, "openai_codex.generated.v2_all", SimpleNamespace(
        SkillsConfigWriteResponse=object, SkillsListResponse=object))
    state.auth_home = auth_home
    return state


def request(*, tools=(), purpose="action", image=False):
    parts = [TextPart("Visible task observation")]
    if image:
        parts.append(ImagePart(encode_png(b"\x01\x02\x03", 1, 1), width=1, height=1))
    return ModelRequest((Message("user", tuple(parts)),), tools=tools, purpose=purpose, max_output_tokens=128)


def test_default_http_backend_does_not_require_codex_sdk(monkeypatch):
    from agent_ark.agent.simple_agent.agent import _OpenAIModel

    monkeypatch.setitem(sys.modules, "openai_codex", None)
    agent = SimpleAgent()
    try:
        assert agent.provider == "auto"
        assert agent.model == "qwen3-vl-plus"
        assert isinstance(agent._backend, _OpenAIModel)
    finally:
        agent.close()


def test_codex_selection_is_lazy_and_has_no_http_or_key_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai_codex", None)
    monkeypatch.setenv("AGENTARK_CODEX_BIN", "fake-codex-executable")
    spec = resolve_agent_type({"agent_type": "simple", "provider": "codex"})
    assert spec.name == "simple"
    runtime = build_model_runtimes([{"agent_type": "simple", "provider": "codex",
                                     "api_key_env": "UNSET_TEST_KEY"}])[0]
    agent = runtime["agent"]
    try:
        assert isinstance(agent._backend, CodexModel)
        assert agent.codex_bin == "fake-codex-executable"
        assert runtime["model"] == "gpt-6.1-sol"
        assert runtime["reasoning_effort"] == "medium"
        assert runtime["max_output_tokens_supported"] is False
        cfg = configure_agent_environment({}, runtime)
        assert cfg["env_config_overrides"]["env_wrapper_cfg"]["context_manager"]["messages"]["return_mode"] == "delta"
    finally:
        agent.close()


def test_each_request_gets_new_process_home_cwd_thread_and_complete_usage(fake_sdk):
    fake_sdk.scripts = [successful(envelope("first")), successful(envelope("second"), usage=False)]
    backend = CodexModel(codex_bin="selected-bin")

    async def run():
        first = await backend.complete(request(image=True))
        metadata = dict(backend.last_response_metadata)
        second = await backend.complete(request())
        return first, metadata, second

    first, metadata, second = asyncio.run(run())
    assert (first.usage.input_tokens, first.usage.output_tokens, first.usage.total_tokens) == (17, 14, 31)
    assert second.usage.total_tokens is None
    assert metadata["sdk_usage"]["total"]["cachedInputTokens"] == 8
    assert metadata["sdk_usage"]["total"]["reasoningOutputTokens"] == 9
    assert metadata["max_output_tokens"]["supported"] is False
    assert metadata["builtin_tool_execution_observed"] is False
    assert len(fake_sdk.starts) == len(fake_sdk.requests) == len(fake_sdk.instances) == len(fake_sdk.closes) == 2
    assert metadata["sdk_thread_id"] != backend.last_response_metadata["sdk_thread_id"]
    assert len({entry["cwd"] for entry in fake_sdk.starts}) == 2
    for instance, start in zip(fake_sdk.instances, fake_sdk.starts):
        assert instance.config.codex_bin == "selected-bin"
        assert not Path(instance.config.cwd).exists()
        assert not Path(instance.config.env["CODEX_HOME"]).exists()
        assert start["base_instructions"] == start["developer_instructions"] == ""
        assert start["ephemeral"] is True
        assert start["approval_mode"] == "deny_all"
        assert start["config"]["features"]["shell_tool"] is False
        assert start["config"]["mcp_servers"] == {}
        assert start["config"]["project_doc_max_bytes"] == 0
    assert any(item["type"] == "image" for item in fake_sdk.requests[0][0])
    assert "max_output_tokens" not in fake_sdk.requests[0][1]
    assert (fake_sdk.auth_home / "auth.json").exists()


def test_summary_uses_final_text_only_and_reasoning_stays_in_audit(fake_sdk):
    fake_sdk.scripts = [successful(envelope("concise memory"), reasoning="audit-only-reasoning")]
    backend = CodexModel()
    result = asyncio.run(backend.complete(request(purpose="summary")))
    assert result.text == "concise memory"
    assert "audit-only-reasoning" not in result.text
    assert "audit-only-reasoning" in str(backend.last_response_metadata["sdk_items"])


def test_sdk_retry_notification_preserves_same_turn_and_single_action(fake_sdk, tmp_path):
    action = {"name": "tap", "arguments": {"x": 3}}
    fake_sdk.scripts = [[sdk_error(retry=True)] + successful(envelope(calls=[
        call("submit_action", {"format": "tool_call", "content": json.dumps(action)})]))]
    agent = SimpleAgent(provider="codex", model="gpt-6.1-sol", reasoning_effort="medium",
        simple_agent_config={"tools": {"enabled": True, "workspace_parent": str(tmp_path / "workspace")}})
    try:
        result, trace = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "Visible task"}]}})
        assert json.loads(result[0]["action"][11:-12]) == action
        assert len(trace[0]["model_requests"]) == 1
        metadata = trace[0]["model_requests"][0]["response_metadata"]
        assert metadata["sdk_error_events"][0]["will_retry"] is True
        assert metadata["sdk_error_events"][0]["http_status"] == 502
        assert metadata["status"] == "completed"
        assert metadata["usage"]["total_tokens"] == 31
        assert metadata["sdk_turn_starts"] == 1
        assert len(fake_sdk.starts) == len(fake_sdk.requests) == len(fake_sdk.closes) == 1
        assert fake_sdk.interrupts == []
    finally:
        agent.close()


def test_terminal_error_waits_for_usage_and_preserves_safe_terminal_detail(fake_sdk):
    notification = sdk_error(retry=False, code="httpConnectionFailed", status=400,
                             message="Request rejected", details="Too many images: maximum image limit exceeded")
    terminal_error = {"message": "Invalid image input", "codexErrorInfo": "badRequest",
                      "additionalDetails": "Image limit exceeded"}
    usage = successful(envelope())[1]
    fake_sdk.scripts = [[notification, usage,
                        event("turn/completed", {"turn": {"status": "failed", "error": terminal_error}})]]
    backend = CodexModel()
    with pytest.raises(ModelResponseError, match="image_input.*badRequest"):
        asyncio.run(backend.complete(request(image=True)))
    metadata = backend.last_response_metadata
    assert metadata["sdk_error_events"][0]["will_retry"] is False
    assert metadata["sdk_error_events"][0]["http_status"] == 400
    assert metadata["sdk_terminal_error"]["category"] == "image_input"
    assert metadata["finish_reason"] == "failed"
    assert metadata["usage"]["total_tokens"] == 31
    assert len(fake_sdk.requests) == len(fake_sdk.closes) == 1


@pytest.mark.parametrize(("code", "status", "message", "category"), [
    ("contextWindowExceeded", None, "", "context_limit"),
    ("usageLimitExceeded", None, "", "quota"),
    ("httpConnectionFailed", 429, "", "quota"),
    ("unauthorized", None, "", "authentication"),
    ("httpConnectionFailed", 401, "", "authentication"),
    ("serverOverloaded", None, "", "server"),
    ("responseStreamDisconnected", 200, "", "network"),
    ("badRequest", None, "Too many images in the request", "image_input"),
    ("badRequest", None, "Input too long for model context window", "context_limit"),
    ("badRequest", None, "Request rejected", "request"),
    ("unrecognized-secret-code", None, "unrecognized provider error", "unknown"),
])
def test_error_categories_are_typed_or_fixed_signals_and_never_reissue(fake_sdk, code, status, message, category):
    failure = sdk_error(retry=False, code=code, status=status, message=message)
    fake_sdk.scripts = [[failure, event("turn/completed", {"turn": {"status": "failed"}})]]
    backend = CodexModel()
    with pytest.raises(ModelResponseError, match=category):
        asyncio.run(backend.complete(request()))
    assert backend.last_response_metadata["sdk_error_events"][0]["category"] == category
    assert backend.last_response_metadata["usage"] == {}
    assert backend.last_response_metadata["sdk_usage"] is None
    assert len(fake_sdk.requests) == len(fake_sdk.closes) == 1


def test_error_audit_omits_all_provider_freeform_auth_images_and_urls(fake_sdk):
    secret = 'Bearer private-auth-token data:image/png;base64,c2VjcmV0 https://example.invalid/?token=private'
    failure = sdk_error(retry=False, code="httpConnectionFailed", status=401,
                        message=secret, details=secret)
    fake_sdk.scripts = [[failure, event("turn/completed", {"turn": {"status": "failed"}})]]
    backend = CodexModel()
    with pytest.raises(ModelResponseError) as error:
        asyncio.run(backend.complete(request()))
    saved = json.dumps(backend.last_response_metadata) + str(error.value)
    assert not any(value in saved for value in ("private-auth-token", "data:image", "c2VjcmV0", "example.invalid"))
    details = backend.last_response_metadata["sdk_error_events"][0]
    assert details["message_characters"] == details["additional_details_characters"] == len(secret)
    assert details["freeform_details_omitted"] is True


@pytest.mark.parametrize(("stage", "error_class"), [
    ("startup", OSError), ("rpc", ImportError), ("stream", RuntimeError), ("close", RuntimeError),
])
def test_sdk_exceptions_are_sanitized_in_error_result_and_traceback(fake_sdk, monkeypatch, stage, error_class):
    secret = "network error Bearer FAKE_SECRET data:image/png;base64,FAKE_IMAGE https://example.test/?token=FAKE"
    failure = error_class(secret)
    fake_sdk.scripts = [successful(envelope("unused"))]
    sdk = sys.modules["openai_codex"]

    async def fail(*args, **kwargs):
        raise failure

    if stage == "startup":
        monkeypatch.setattr(sdk.AsyncCodex, "start", fail)
    elif stage == "rpc":
        monkeypatch.setattr(sdk.AsyncCodex, "request", fail)
    elif stage == "stream":
        # Usage can arrive before a transport exception; keep it auditable.
        fake_sdk.scripts = [[*successful(envelope("unused"))[:-1], failure]]
    else:
        original_close = sdk.AsyncCodex.close

        async def fail_after_close(self):
            await original_close(self)
            raise failure

        monkeypatch.setattr(sdk.AsyncCodex, "close", fail_after_close)

    backend = CodexModel()
    try:
        asyncio.run(backend.complete(request()))
    except ModelResponseError as exc:
        agent = SimpleNamespace(last_trace_by_agent={0: {"model_requests": [
            {"response_metadata": backend.last_response_metadata}]}})
        report = build_error_result(
            {"name": "offline", "model": "fake", "provider": "codex", "agent_type": "simple",
             "base_url": None, "api_key_env": None, "agent": agent},
            {"case_id": "offline", "task_name": "visible", "group_seed": 1}, exc)
        assert exc.__suppress_context__ is True
    else:
        pytest.fail("SDK exceptions must fail the request")

    saved = json.dumps(report)
    assert all(value not in saved for value in ("FAKE_SECRET", "data:image", "FAKE_IMAGE", "example.test"))
    assert report["error_type"] == "ModelResponseError"
    assert "ModelResponseError" in report["traceback"]
    detail = backend.last_response_metadata["sdk_exception"]
    assert detail["category"] == "network"
    assert detail["message_characters"] == len(secret)
    assert detail["freeform_details_omitted"] is True
    assert backend.last_response_metadata["status"] == "sdk_error"
    if stage in ("stream", "close"):
        assert backend.last_response_metadata["usage"]["total_tokens"] == 31
    assert len(fake_sdk.instances) == len(fake_sdk.closes) == 1
    assert not backend._active
    assert not Path(fake_sdk.instances[0].config.cwd).exists()


def test_missing_optional_sdk_preserves_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai_codex", None)
    with pytest.raises(ImportError, match="install the project's codex extra") as error:
        asyncio.run(CodexModel().complete(request()))
    assert error.value.__suppress_context__ is True


@pytest.mark.parametrize("retry", [None, "true", 1, {}])
def test_malformed_retry_flag_fails_closed_without_request_retry(fake_sdk, retry):
    fake_sdk.scripts = [[sdk_error(retry=retry)]]
    backend = CodexModel()
    with pytest.raises(ModelResponseError, match="invalid error retry flag"):
        asyncio.run(backend.complete(request()))
    assert backend.last_response_metadata["sdk_error_events"][0]["will_retry"] is None
    assert len(fake_sdk.requests) == len(fake_sdk.interrupts) == len(fake_sdk.closes) == 1


def test_repeated_sdk_retry_errors_are_bounded_without_request_reissue(fake_sdk):
    fake_sdk.scripts = [[sdk_error(retry=True)] * 20]
    backend = CodexModel(timeout_s=None)
    with pytest.raises(ModelResponseError, match="bounded same-turn error"):
        asyncio.run(backend.complete(request()))
    assert len(backend.last_response_metadata["sdk_error_events"]) == backend.request_parameters(request())["sdk_error_event_limit"]
    assert backend.last_response_metadata["sdk_error_event_limit_exceeded"] is True
    assert len(fake_sdk.requests) == len(fake_sdk.interrupts) == len(fake_sdk.closes) == 1


def test_retry_notification_followed_by_end_of_stream_keeps_diagnostic(fake_sdk):
    fake_sdk.scripts = [[sdk_error(retry=True, code="responseStreamDisconnected", status=200)]]
    backend = CodexModel()
    with pytest.raises(ModelResponseError, match="network.*stream ended before turn completion"):
        asyncio.run(backend.complete(request()))
    assert backend.last_response_metadata["status"] == "sdk_error"
    assert len(fake_sdk.requests) == len(fake_sdk.closes) == 1


@pytest.mark.parametrize("retry", [True, False])
def test_sdk_error_wait_does_not_extend_the_original_request_deadline(fake_sdk, retry):
    fake_sdk.scripts = [[sdk_error(retry=retry), "block"]]
    backend = CodexModel(timeout_s=1.0)
    with pytest.raises(TimeoutError):
        asyncio.run(backend.complete(request()))
    assert backend.last_response_metadata["status"] == "timeout"
    assert backend.last_response_metadata["sdk_error_events"][0]["will_retry"] is retry
    assert len(fake_sdk.requests) == len(fake_sdk.interrupts) == len(fake_sdk.closes) == 1


def test_failed_turn_without_error_notification_preserves_typed_error(fake_sdk):
    fake_sdk.scripts = [[event("turn/completed", {"turn": {"status": "failed", "error": {
        "message": "Context limit", "codexErrorInfo": "contextWindowExceeded"}}})]]
    backend = CodexModel()
    with pytest.raises(ModelResponseError, match="context_limit"):
        asyncio.run(backend.complete(request()))
    assert backend.last_response_metadata["sdk_error_events"] == []
    assert backend.last_response_metadata["sdk_terminal_error"]["code"] == "contextWindowExceeded"


def test_dynamic_skill_preflight_disables_and_verifies_catalog_before_model_turn(fake_sdk):
    fake_sdk.skill_names = ["system-skill-alpha", "future-new-system-skill"]
    fake_sdk.scripts = [successful(envelope("answer"))]
    backend = CodexModel()
    asyncio.run(backend.complete(request()))
    assert backend.last_response_metadata["skills_disabled_count"] == 2
    assert backend.last_response_metadata["skills_verified_disabled"] is True
    assert "future-new-system-skill" not in str(backend.last_response_metadata)
    assert len(fake_sdk.disabled_skills) == 2
    assert all(entry["enabled"] is False for entry in fake_sdk.starts[0]["config"]["skills"]["config"])
    env = fake_sdk.instances[0].config.env
    assert env["HOME"] == env["USERPROFILE"] == env["CODEX_HOME"]
    assert (fake_sdk.auth_home / "config.toml").read_text() == "do_not_inherit = true"


def test_ineffective_skill_disable_fails_before_any_model_turn(fake_sdk):
    fake_sdk.skill_names = ["cannot-disable"]
    fake_sdk.disables_work = False
    with pytest.raises(ModelResponseError, match="disable"):
        asyncio.run(CodexModel().complete(request()))
    assert fake_sdk.requests == []
    assert len(fake_sdk.closes) == 1


def test_missing_skill_rpc_capability_fails_before_starting_transport(fake_sdk, monkeypatch):
    monkeypatch.setattr(sys.modules["openai_codex"].AsyncCodex, "request", None)
    with pytest.raises(ModelResponseError, match="skill RPCs"):
        asyncio.run(CodexModel().complete(request()))
    assert fake_sdk.requests == []
    assert len(fake_sdk.closes) == 1


def test_refresh_survives_only_in_backend_memory_and_source_auth_stays_unchanged(fake_sdk):
    original = (fake_sdk.auth_home / "auth.json").read_bytes()
    fake_sdk.scripts = [successful(envelope("first")), successful(envelope("second"))]
    fake_sdk.refresh_tokens = {"fake": "rotated-test-token"}
    backend = CodexModel()

    async def run():
        await backend.complete(request())
        await backend.complete(request())
        assert fake_sdk.auth_inputs[0]["tokens"] == {"fake": "test"}
        assert fake_sdk.auth_inputs[1]["tokens"] == fake_sdk.refresh_tokens
        assert "rotated-test-token" not in str(backend.last_response_metadata)
        await backend.close()
        assert backend._auth_bytes is None

    asyncio.run(run())
    assert (fake_sdk.auth_home / "auth.json").read_bytes() == original


@pytest.mark.parametrize("raw", [
    '{"text":"a","text":"b","tool_calls":[]}',
    envelope(calls=[call("unknown", {})]),
    envelope(calls=[call("read_file", {}, id="bad id")]),
    envelope(calls=[call("read_file", {}, id="same"), call("read_file", {}, id="same")]),
    envelope(calls=[{"id": "a", "name": "read_file", "arguments_json": '{"a":1,"a":2}'}]),
    envelope(calls=[{"id": "a", "name": "read_file", "arguments_json": '{"a":NaN}'}]),
    envelope(calls=[{"id": "a", "name": "read_file", "arguments_json": '[]'}]),
    '```json\n{"text":"hello","tool_calls":[]}\n```',
])
def test_invalid_structured_transport_fails_closed_without_retry(fake_sdk, raw):
    fake_sdk.scripts = [successful(raw)]
    tools = (ToolSpec("read_file", "Read a visible file", '{"type":"object"}'),)
    with pytest.raises(ModelResponseError):
        asyncio.run(CodexModel().complete(request(tools=tools)))
    assert len(fake_sdk.requests) == len(fake_sdk.instances) == len(fake_sdk.closes) == 1


@pytest.mark.parametrize("kind", ["commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall",
                                  "webSearch", "imageView", "collabAgentToolCall", "contextCompaction", "hookPrompt"])
def test_unexpected_builtin_or_compaction_interrupts_and_closes(fake_sdk, kind):
    fake_sdk.scripts = [[event("item/started", {"item": {"id": "unexpected", "type": kind}})]]
    backend = CodexModel()
    with pytest.raises(ModelResponseError, match="undeclared"):
        asyncio.run(backend.complete(request()))
    assert backend.last_response_metadata["unexpected_sdk_item_type"] == kind
    assert len(fake_sdk.interrupts) == len(fake_sdk.closes) == 1


def test_timeout_interrupts_and_terminates_request_process(fake_sdk):
    fake_sdk.scripts = ["block"]
    # Cover the live-turn timeout. Startup/RPC deadlines have separate tests;
    # leave enough time for Windows temporary-file/SDK preflight scheduling.
    backend = CodexModel(timeout_s=1.0)
    with pytest.raises(TimeoutError):
        asyncio.run(backend.complete(request()))
    assert backend.last_response_metadata["status"] == "timeout"
    assert fake_sdk.entered.is_set()
    assert len(fake_sdk.interrupts) == len(fake_sdk.closes) == 1
    assert not Path(fake_sdk.instances[0].config.cwd).exists()


def test_external_cancellation_interrupts_and_terminates_request_process(fake_sdk):
    fake_sdk.scripts = ["block"]
    backend = CodexModel(timeout_s=None)

    async def run():
        pending = asyncio.create_task(backend.complete(request()))
        await fake_sdk.entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(run())
    assert backend.last_response_metadata["status"] == "cancelled"
    assert len(fake_sdk.interrupts) == len(fake_sdk.closes) == 1


def test_empty_summary_is_bounded_runtime_retry_candidate(fake_sdk):
    fake_sdk.scripts = [successful(envelope(""), reasoning="not summary content")]
    with pytest.raises(SummaryResponseError) as error:
        asyncio.run(CodexModel().complete(request(purpose="summary")))
    assert error.value.reason == "empty_content"
    assert error.value.usage.total_tokens == 31


def test_runtime_executes_host_tools_then_submit_action_without_codex_memory(fake_sdk, tmp_path):
    action = {"name": "tap", "arguments": {"x": 3}}
    rationale = "Tap the visible button to advance the task."
    fake_sdk.scripts = [
        successful(envelope(calls=[call("list_files", {"path": "records"})])),
        successful(envelope(rationale, calls=[call("submit_action", {"format": "tool_call", "content": json.dumps(action)})])),
    ]
    agent = SimpleAgent(provider="codex", model="gpt-6.1-sol", reasoning_effort="medium",
        simple_agent_config={"tools": {"enabled": True, "workspace_parent": str(tmp_path / "workspace")}})
    try:
        legacy_format = "First output <think>OLD_FORMAT</think>, then <tool_call> JSON </tool_call>."
        result, trace = agent.forward_with_trace({0: {"messages": [
            {"role": "system", "content": legacy_format}, {"role": "user", "content": "Visible task"}]}})
        assert json.loads(result[0]["action"][11:-12]) == action
        assert len(trace[0]["model_requests"]) == 2
        assert len(fake_sdk.instances) == 2
        ordered_input = str(fake_sdk.requests[0][0])
        assert "<think>...</think> reply are superseded" in ordered_input
        assert "native model/backend reasoning when available" in ordered_input
        assert "hidden or absent" in ordered_input
        assert "exactly one standalone native submit_action" in ordered_input
        assert "user-facing action rationale in the assistant message's content" in ordered_input
        assert "belongs in text, which represents assistant-message content" in ordered_input
        assert "Keep it out of the submit_action content argument" in ordered_input
        assert legacy_format in ordered_input
        assert '"tool_call_id": "call_1"' in str(fake_sdk.requests[1][0])
        assert all(item["response_metadata"]["sdk_native_function_tools"] is False
                   for item in trace[0]["model_requests"])
        assert trace[0]["assistant_raw"] == rationale
        submitted = agent._states[0].runtime.history_snapshot[-1].messages[0]
        assert submitted.parts == (TextPart(rationale),)
        arguments = json.loads(submitted.tool_calls[0].arguments_json)
        assert json.loads(arguments["content"]) == action
        record = dict(agent._states[0].workspace._records["turn-000001"][0].files)
        assert record["assistant.txt"] == rationale.encode("utf-8")
        assert json.loads(record["action.txt"].decode("utf-8")[11:-12]) == action
        old_backend = agent._backend
        agent.close()
        agent.reset()
        assert isinstance(agent._backend, CodexModel) and agent._backend is not old_backend
    finally:
        agent.close()


def test_isolation_never_falls_back_to_user_home_without_file_credentials(fake_sdk):
    (fake_sdk.auth_home / "auth.json").unlink()
    with pytest.raises(ModelResponseError, match="auth.json"):
        asyncio.run(CodexModel().complete(request()))
    assert fake_sdk.instances == []


def test_saved_sdk_item_audit_omits_image_urls_without_mutating_raw_items():
    trace = {0: {"model_requests": [{"response_metadata": {"sdk_items": [
        {"type": "userMessage", "content": [{"type": "image", "url": "data:image/png;base64,abcd"}]}
    ]}}]}}
    saved = normalize_response_trace(trace)
    assert "data:image" not in str(saved)
    assert "data:image/png;base64,abcd" in str(trace)


@pytest.fixture
def threaded_sdk(fake_sdk, monkeypatch):
    """Model the SDK's uncancellable executor queue.get and delayed Popen."""
    state = SimpleNamespace(queue=queue.Queue(), registered=False, started=threading.Event(),
        worker_done=threading.Event(), release_start=threading.Event(), start_entered=threading.Event(),
        delay_start=False, spawned=False, process_closed=False, interrupt_completes=True,
        close_saw_registered=False, turn_starts=0, closes=0, original_start_calls=0,
        blocked_stage=None, rpc_queue=queue.Queue(), rpc_entered=threading.Event(),
        rpc_done=threading.Event(), rpc_registered=False, leading_events=())

    async def maybe_block(stage):
        if state.blocked_stage != stage:
            return

        def worker():
            state.rpc_entered.set()
            try:
                return state.rpc_queue.get()
            finally:
                state.rpc_done.set()

        state.rpc_registered = True
        try:
            value = await asyncio.to_thread(worker)
            if isinstance(value, BaseException):
                raise value
        finally:
            state.rpc_registered = False

    class Turn:
        id = "thread-backed-turn"

        async def interrupt(self):
            if state.interrupt_completes and state.registered:
                state.queue.put(event("turn/completed", {"turn": {"id": self.id, "status": "interrupted"}}))

        def _get(self):
            state.started.set()
            try:
                return state.queue.get()
            finally:
                state.worker_done.set()

        async def stream(self):
            state.registered = True
            try:
                for notification in state.leading_events:
                    yield notification
                value = await asyncio.to_thread(self._get)
                if isinstance(value, BaseException):
                    raise value
                yield value
            finally:
                state.registered = False

    class Thread:
        id = "thread-backed-thread"

        async def turn(self, inputs, **kwargs):
            state.turn_starts += 1
            await maybe_block("turn_start")
            return Turn()

    class Codex:
        def __init__(self, *, config):
            self.config = config
            self._client = SimpleNamespace(request=self.request, start=self.start)
            state.config = config
            state.codex = self

        def _spawn(self):
            state.original_start_calls += 1
            state.start_entered.set()
            if state.delay_start:
                state.release_start.wait()
            state.spawned = True

        async def start(self):
            await asyncio.to_thread(self._spawn)

        async def __aenter__(self):
            await self._client.start()
            await maybe_block("initialize")
            return self

        async def request(self, method, params, **kwargs):
            assert method == "skills/list"
            await maybe_block("skills")
            return {"data": [{"cwd": self.config.cwd, "errors": [], "skills": []}]}

        async def thread_start(self, **kwargs):
            await maybe_block("thread_start")
            return Thread()

        async def close(self):
            state.closes += 1
            if state.spawned:
                state.process_closed = True
            if state.registered:
                state.close_saw_registered = True
                state.queue.put(RuntimeError("fake transport closed"))
            if state.rpc_registered:
                state.rpc_queue.put(RuntimeError("fake RPC transport closed"))

    original = sys.modules["openai_codex"]
    monkeypatch.setitem(sys.modules, "openai_codex", SimpleNamespace(
        **{**vars(original), "AsyncCodex": Codex}))
    return state


@pytest.mark.parametrize("reason", ["timeout", "cancel"])
@pytest.mark.parametrize("interrupt_completes", [True, False])
def test_thread_backed_stream_is_woken_before_unregistration(threaded_sdk, reason, interrupt_completes):
    state = threaded_sdk
    state.interrupt_completes = interrupt_completes
    backend = CodexModel(timeout_s=1.0 if reason == "timeout" else None)

    async def run():
        pending = asyncio.create_task(backend.complete(request()))
        try:
            assert await asyncio.to_thread(state.started.wait, 1.0)
            if reason == "cancel":
                pending.cancel()
            done, _ = await asyncio.wait({pending}, timeout=6.0)
            assert done, "request cleanup must not strand the queue worker"
            with pytest.raises(TimeoutError if reason == "timeout" else asyncio.CancelledError):
                await pending
            assert state.worker_done.is_set()
            assert not state.registered
            assert state.process_closed
            if not interrupt_completes:
                assert state.close_saw_registered
            assert not Path(state.config.cwd).exists()
        finally:
            # Release the original queue even if an assertion catches a
            # regression, so asyncio.run's executor shutdown cannot hang pytest.
            state.queue.put(RuntimeError("test cleanup"))
            state.release_start.set()
            await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["timeout", "cancel"])
@pytest.mark.parametrize("stage", ["stream", "skills", "turn_start"])
def test_cleanup_error_preserves_failure_and_still_settles_workers(threaded_sdk, monkeypatch, reason, stage):
    state = threaded_sdk
    state.interrupt_completes = False
    state.blocked_stage = None if stage == "stream" else stage
    usage_event = next(item for item in successful(envelope()) if item.method == "thread/tokenUsage/updated")
    state.leading_events = (usage_event,)
    sdk = sys.modules["openai_codex"]
    original_close = sdk.AsyncCodex.close
    secret = "Bearer FAKE_CLEANUP_SECRET data:image/png;base64,FAKE_IMAGE https://example.test/?token=FAKE"

    async def fail_after_close(self):
        await original_close(self)
        raise RuntimeError(secret)

    monkeypatch.setattr(sdk.AsyncCodex, "close", fail_after_close)
    backend = CodexModel(timeout_s=1.0 if reason == "timeout" else None)

    async def run():
        pending = asyncio.create_task(backend.complete(request()))
        try:
            entered = state.started if stage == "stream" else state.rpc_entered
            assert await asyncio.to_thread(entered.wait, 1.0)
            if reason == "cancel":
                pending.cancel()
            done, _ = await asyncio.wait({pending}, timeout=6.0)
            assert done, "cleanup errors must not bypass worker settlement"
            try:
                await pending
            except (TimeoutError, asyncio.CancelledError) as exc:
                assert isinstance(exc, TimeoutError if reason == "timeout" else asyncio.CancelledError)
                report = build_error_result(
                    {"name": "offline", "model": "fake", "provider": "codex", "base_url": None, "api_key_env": None},
                    {"case_id": "offline", "task_name": "visible", "group_seed": 1}, exc)
            else:
                pytest.fail("original request failure must be preserved")
            saved = json.dumps({"report": report, "metadata": backend.last_response_metadata})
            assert all(value not in saved for value in ("FAKE_CLEANUP_SECRET", "data:image", "FAKE_IMAGE", "example.test"))
            assert backend.last_response_metadata["status"] == ("timeout" if reason == "timeout" else "cancelled")
            assert backend.last_response_metadata["sdk_cleanup_error"]["freeform_details_omitted"] is True
            assert state.process_closed and state.closes == 1
            if stage == "stream":
                assert state.worker_done.is_set() and not state.registered
                assert state.close_saw_registered
                assert backend.last_response_metadata["usage"]["total_tokens"] == 31
            else:
                assert state.rpc_done.is_set() and not state.rpc_registered
            assert not Path(state.config.cwd).exists()
            assert not backend._active
        finally:
            state.release_start.set()
            state.rpc_queue.put(RuntimeError("test RPC cleanup"))
            state.queue.put(RuntimeError("test stream cleanup"))
            await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(run())


def test_cleanup_error_preserves_summary_retry_reason_and_usage(fake_sdk, monkeypatch):
    fake_sdk.scripts = [successful(envelope(""))]
    sdk = sys.modules["openai_codex"]
    original_close = sdk.AsyncCodex.close
    secret = "Bearer FAKE_CLEANUP_SECRET data:image/png;base64,FAKE_IMAGE https://example.test/?token=FAKE"

    async def fail_after_close(self):
        await original_close(self)
        raise RuntimeError(secret)

    monkeypatch.setattr(sdk.AsyncCodex, "close", fail_after_close)
    backend = CodexModel()
    try:
        asyncio.run(backend.complete(request(purpose="summary")))
    except SummaryResponseError as exc:
        assert exc.reason == "empty_content"
        assert exc.usage.total_tokens == 31
        report = build_error_result(
            {"name": "offline", "model": "fake", "provider": "codex", "base_url": None, "api_key_env": None},
            {"case_id": "offline", "task_name": "visible", "group_seed": 1}, exc)
    else:
        pytest.fail("summary retry candidate must survive cleanup errors")
    saved = json.dumps({"report": report, "metadata": backend.last_response_metadata})
    assert all(value not in saved for value in ("FAKE_CLEANUP_SECRET", "data:image", "FAKE_IMAGE", "example.test"))
    assert backend.last_response_metadata["usage"]["total_tokens"] == 31
    assert backend.last_response_metadata["sdk_cleanup_error"]["freeform_details_omitted"] is True
    assert len(fake_sdk.closes) == 1
    assert not backend._active
    assert not Path(fake_sdk.instances[0].config.cwd).exists()


@pytest.mark.parametrize("reason", ["timeout", "cancel"])
def test_delayed_startup_is_settled_then_closed_before_temp_cleanup(threaded_sdk, reason):
    state = threaded_sdk
    state.delay_start = True
    backend = CodexModel(timeout_s=0.01 if reason == "timeout" else None)

    async def run():
        pending = asyncio.create_task(backend.complete(request()))
        try:
            assert await asyncio.to_thread(state.start_entered.wait, 1.0)
            if reason == "cancel":
                pending.cancel()
            await asyncio.sleep(0.04)
            assert not pending.done(), "cleanup must wait for the outstanding Popen worker"
            state.release_start.set()
            done, _ = await asyncio.wait({pending}, timeout=2.0)
            assert done
            with pytest.raises(TimeoutError if reason == "timeout" else asyncio.CancelledError):
                await pending
            assert state.spawned and state.process_closed
            assert state.turn_starts == 0
            assert state.closes == 1
            await state.codex._client.start()
            assert state.original_start_calls == 1
            assert not Path(state.config.cwd).exists()
        finally:
            state.release_start.set()
            state.queue.put(RuntimeError("test cleanup"))
            await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["timeout", "cancel"])
@pytest.mark.parametrize("stage", ["initialize", "skills", "thread_start", "turn_start"])
def test_blocked_sdk_rpc_is_closed_then_settled_without_hanging(threaded_sdk, reason, stage):
    state = threaded_sdk
    state.blocked_stage = stage
    backend = CodexModel(timeout_s=1.0 if reason == "timeout" else None)

    async def run():
        pending = asyncio.create_task(backend.complete(request()))
        try:
            assert await asyncio.to_thread(state.rpc_entered.wait, 1.0)
            if reason == "cancel":
                pending.cancel()
            done, _ = await asyncio.wait({pending}, timeout=2.0)
            assert done, "RPC worker must be woken by transport closure before settlement"
            with pytest.raises(TimeoutError if reason == "timeout" else asyncio.CancelledError):
                await pending
            assert state.rpc_done.is_set() and not state.rpc_registered
            assert state.process_closed
            await state.codex._client.start()
            assert state.original_start_calls == 1
        finally:
            state.release_start.set()
            state.rpc_queue.put(RuntimeError("test RPC cleanup"))
            state.queue.put(RuntimeError("test stream cleanup"))
            await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(run())
