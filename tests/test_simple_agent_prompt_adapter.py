"""Offline native prompt adaptation through real wrapper/message parsing."""
from __future__ import annotations

from copy import deepcopy
from importlib import resources
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image
import pytest

from agent_ark.agent.api_agent import APIAgent
from agent_ark.agent.simple_agent.agent import SimpleAgent
from agent_ark.agent.simple_agent.encoding import chat_messages
from agent_ark.agent.simple_agent.types import ImagePart, Message, ModelCompletion, TextPart, ToolCall
from agent_ark.ark_env.context_manager import MessageContext
from agent_ark.ark_env.direct_env import EnvInfoManager, EnvWrapper


PROJECT = Path(__file__).resolve().parents[1]
ACTION_BODY = '{"name":"Paint","arguments":{"label":"门","count":2}}'
ACTION = "<tool_call>" + ACTION_BODY + "</tool_call>"
SOURCE = "public sealed class AgentAction : CodeAction { public override void Start() { Context.CompleteStep(); } }"
TASK = ('TASK_RULE: Paint only the highlighted cell. Keep literal <think>label</think>, '
        '<code>label</code>, and <tool_call>label</tool_call> in the picture.\n'
        '<tool_docs>Only Paint(string label, int count) is exposed. Never inspect hidden cells.</tool_docs>\n'
        'TASK_CODE_CAPABILITY: use only Context.Call("paint", CodeValue.String("门")).')
CASES = [(mode, explanation) for mode in ("func", "code", None)
         for explanation in ("legacy_think", "brief_rationale")]
RESOURCES = [f"native_system_{mode}_{explanation}.txt" for mode in ("func", "code", "legacy")
             for explanation in ("legacy_think", "brief_rationale")] + ["native_system_func_zh.txt", "native_system_prompt.txt"]


def native_helper():
    from agent_ark.agent.simple_agent.prompt_adapter import adapt_native_environment_pin
    return adapt_native_environment_pin


def real_messages(mode="func", explanation="legacy_think", *, chinese=False):
    manager = object.__new__(EnvInfoManager)
    manager.env_config = {"action_mode": mode, "max_attempts": 2, "max_steps_per_attempt": 8,
                          "prompt": {"explanation_mode": explanation}}
    manager.system_prompt = manager._get_system_prompt(action_mode=mode)
    if chinese:
        manager.system_prompt = resources.files("agent_ark").joinpath(
            "ark_env", "info", "system_prompt_func_zh.txt").read_text(encoding="utf-8")
    wrapper = object.__new__(EnvWrapper)
    wrapper.env_info_mgr = manager
    payload = wrapper._build_reset_obs_payload(
        "<task_prompt>" + TASK + "</task_prompt><reset_context>The highlighted cell is red.</reset_context>")
    payload["vis"] = [[Image.new("RGB", (3, 2), "red")]]
    context = MessageContext({"enabled": True, "max_images_per_section": 1,
                              "text_max_chars": 100_000, "merge_system_task_prompt": False})
    return context.build_chat_messages(payload, [], [])


def submission(kind="tool_call", content=ACTION_BODY, call_id="submit", text="Paint the visible highlighted cell."):
    return ModelCompletion(text, tool_calls=(ToolCall(call_id, "submit_action", json.dumps({
        "format": kind, "content": content}, ensure_ascii=False)),))


class FakeModel:
    def __init__(self, replies=None):
        self.requests = []
        self.replies = list(replies or [submission()])

    async def complete(self, request):
        self.requests.append(request)
        return self.replies.pop(0)


def text_of(messages):
    return "\n".join(part.text for message in messages for part in message.parts if isinstance(part, TextPart))


def record_text(message):
    return "".join(part.get("text", "".join(part.get("text_chunks", [])))
                   for part in message["content"] if part["type"] == "text")


@pytest.mark.parametrize("mode,explanation", CASES)
def test_real_wrapper_templates_adapt_in_preview_forward_and_model_readable_records(tmp_path, mode, explanation):
    messages = real_messages(mode, explanation)
    original = deepcopy(messages)
    original_system = messages[0]["content"]
    assert messages[0]["role"] == "system" and not original_system.startswith("<system_prompt>")
    backend = FakeModel()
    agent = SimpleAgent(model_backend=backend, simple_agent_config={"tools": {
        "enabled": True, "workspace_parent": str(tmp_path)}})
    try:
        preview = agent.build_request_messages({0: {"messages": messages}})
        assert not agent._states
        result, trace = agent.forward_with_trace({0: {"messages": messages}})
        actual = backend.requests[0].messages
        assert preview[0] == chat_messages(actual) == trace[0]["model_requests"][0]["request_messages"]
        assert result[0]["action"] == ACTION
        assert messages == original
        assert agent._states[0].env_pins == (Message.text("system", original_system),)
        replacement = native_helper()(Message.text("system", original_system))
        assert replacement.parts[0].text != original_system
        assert replacement.parts[0].text in text_of(actual)
        policy = text_of(actual[:2])
        assert "submit_action" in policy
        for tag in ("<think>", "</think>", "<tool_call>", "</tool_call>", "<code>", "</code>", "<params>"):
            assert tag not in policy
        assert original_system not in text_of(actual)
        assert TASK in text_of(actual)
        original_parts = agent._messages({"messages": original})
        assert [part.data for message in actual for part in message.parts if isinstance(part, ImagePart)] == [
            part.data for message in original_parts for part in message.parts if isinstance(part, ImagePart)]
        files = dict(agent._states[0].workspace._records["turn-000001"][0].files)
        record = json.loads(files["messages.json"])
        assert record_text(record["messages"][0]) == replacement.parts[0].text
        assert record_text(record["messages"][1]) == record_text({"content": original[1]["content"]})
        assert original_system not in "\n".join(record_text(m) for m in record["messages"])
        assert TASK in "\n".join(record_text(m) for m in record["messages"])
        stored_image = next(part for m in record["messages"] for part in m["content"] if part["type"] == "image")
        assert files[stored_image["path"]] == next(part.data for m in original_parts for part in m.parts if isinstance(part, ImagePart))
    finally:
        agent.close()


def test_chinese_func_template_is_recognized_after_real_message_context_parsing():
    source = real_messages(chinese=True)[0]["content"]
    original = Message.text("system", source)
    adapted = native_helper()(original)
    assert adapted != original and "submit_action" in adapted.parts[0].text
    assert "<think>" not in adapted.parts[0].text and "<tool_call>" not in adapted.parts[0].text


@pytest.mark.parametrize("mode,explanation", CASES)
def test_replacement_retains_payload_and_runtime_constraints(mode, explanation):
    source = real_messages(mode, explanation)[0]["content"]
    replacement = native_helper()(Message.text("system", source)).parts[0].text
    if mode == "code":
        for contract in ("public sealed class AgentAction : CodeAction", "Start", "Tick", "FixedTick", "Stop",
                         "CodeValue", "Context.CompleteStep()", "Context.StopSelf()", "checkpoint"):
            assert contract in replacement
        for restricted in ("MonoBehaviour", "Unity", "scene", "file", "network", "thread", "reflection",
                           "delegate", "async", "exception", "undeclared"):
            assert restricted.casefold() in replacement.casefold()
    elif mode is None:
        for contract in ("MonoBehaviour", "Start", "Update", "FixedUpdate", "UnityEditor", "localPosition",
                         "ActRouter", "router.Call", "router.Get", "router.Set"):
            assert contract in replacement
        assert "子节点" in replacement or "subtree" in replacement.casefold()
        assert "反射" in replacement or "reflection" in replacement.casefold()
    else:
        for contract in ("TaskApi", "JSON", "true", "false"):
            assert contract in replacement
        assert "tool documentation" in replacement or "工具文档" in replacement


@pytest.mark.parametrize("role", ["system", "developer"])
def test_matching_normalizes_crlf_but_preserves_source_pins(role):
    source = real_messages()[0]["content"]
    source_crlf = source.replace("\n", "\r\n")
    assert native_helper()(Message.text(role, source)).parts == native_helper()(Message.text(role, source_crlf)).parts
    assert native_helper()(Message.text(role, source_crlf)).role == role


@pytest.mark.parametrize("variant", ["custom", "appendix", "prefix", "trailing_newline", "leading_newline",
                                     "wrapped", "two_text_parts", "image", "user"])
def test_custom_near_matches_appendices_and_mixed_parts_are_not_rewritten(variant):
    source = real_messages()[0]["content"]
    if variant == "custom":
        message = Message.text("system", source.replace("THINK_CONTENT", "CUSTOM_PRIVATE_FORMAT"))
    elif variant == "appendix":
        message = Message.text("system", source + "\nTASK_APPENDIX: Keep this exact task policy.")
    elif variant == "prefix":
        message = Message.text("developer", "CUSTOM_PREFIX\n" + source)
    elif variant == "trailing_newline":
        message = Message.text("system", source + "\n")
    elif variant == "leading_newline":
        message = Message.text("system", "\n" + source)
    elif variant == "wrapped":
        message = Message.text("system", "<system_prompt>\n" + source + "\n</system_prompt>")
    elif variant == "two_text_parts":
        message = Message("system", (TextPart(source[:20]), TextPart(source[20:])))
    elif variant == "image":
        message = Message("system", (TextPart(source), ImagePart(b"image-content")))
    else:
        message = Message.text("user", source)
    assert native_helper()(message) is message


def test_unknown_pin_and_literal_task_text_remain_in_native_request_and_records(tmp_path):
    source = real_messages()[0]["content"] + "\nCUSTOM_SYSTEM_APPENDIX"
    messages = [{"role": "system", "content": source}, {"role": "developer", "content": "CUSTOM_TASK_RULE"},
                {"role": "user", "content": TASK}]
    backend = FakeModel()
    agent = SimpleAgent(model_backend=backend, simple_agent_config={"tools": {
        "enabled": True, "workspace_parent": str(tmp_path)}})
    try:
        agent.forward({0: {"messages": messages}})
        assert source in text_of(backend.requests[0].messages)
        assert TASK in text_of(backend.requests[0].messages)
        files = dict(agent._states[0].workspace._records["turn-000001"][0].files)
        record = json.loads(files["messages.json"])
        assert [record_text(message) for message in record["messages"]] == [source, "CUSTOM_TASK_RULE", TASK]
    finally:
        agent.close()


def test_original_environment_pin_change_is_rejected_before_another_model_call(tmp_path):
    initial = real_messages("func", "legacy_think")
    updated = real_messages("func", "brief_rationale")
    backend = FakeModel([submission(call_id="first"), submission(call_id="second")])
    agent = SimpleAgent(model_backend=backend, simple_agent_config={"tools": {
        "enabled": True, "workspace_parent": str(tmp_path)}})
    try:
        agent.forward({0: {"messages": initial}})
        with pytest.raises(ValueError, match="environment instructions changed"):
            agent.build_request_messages({0: {"messages": updated}})
        with pytest.raises(ValueError, match="environment instructions changed"):
            agent.forward({0: {"messages": updated}})
        assert len(backend.requests) == 1
        assert agent._states[0].env_pins[0].parts[0].text == initial[0]["content"]
    finally:
        agent.close()


@pytest.mark.parametrize("mode,explanation", CASES)
def test_tools_disabled_path_keeps_original_wrapper_prompt(mode, explanation):
    messages = real_messages(mode, explanation)
    original = deepcopy(messages)
    backend = FakeModel([ModelCompletion("<think>Legacy thought.</think>\n" + ACTION)])
    agent = SimpleAgent(model_backend=backend)
    try:
        preview = agent.build_request_messages({0: {"messages": messages}})
        result = agent.forward({0: {"messages": messages}})
        assert result[0] == ACTION
        assert backend.requests[0].messages[0] == Message.text("system", messages[0]["content"])
        assert preview[0] == chat_messages(backend.requests[0].messages)
        assert not backend.requests[0].tools and "submit_action" not in text_of(backend.requests[0].messages)
        assert messages == original
    finally:
        agent.close()


def test_multistep_native_submission_payloads_and_closed_history_survive_prompt_adaptation(tmp_path):
    payloads = [("tool_call", ACTION_BODY, ACTION), ("params", "L2,U3", "<params>L2,U3</params>"),
                ("code", SOURCE, SOURCE), ("raw", "SELECT LEFT", "SELECT LEFT")]
    backend = FakeModel([submission(kind, body, call_id=f"submit-{index}", text=f"Visible rationale {index}.")
                         for index, (kind, body, _) in enumerate(payloads, 1)])
    agent = SimpleAgent(model_backend=backend, simple_agent_config={"tools": {
        "enabled": True, "workspace_parent": str(tmp_path)}})
    try:
        for index, (_, _, expected) in enumerate(payloads, 1):
            messages = real_messages() if index == 1 else [{"role": "user", "content": f"Execution feedback {index - 1}."}]
            result = agent.forward({0: {"messages": messages}})
            assert result[0] == expected
            req = backend.requests[index - 1]
            assert "submit_action" in text_of(req.messages[:2])
            previous_calls = [message for message in req.messages if message.tool_calls]
            assert [message.tool_calls[0].call_id for message in previous_calls] == [f"submit-{i}" for i in range(1, index)]
            for message in previous_calls:
                call = message.tool_calls[0]
                assert any(reply.role == "tool" and reply.tool_call_id == call.call_id
                           and "not executed yet" in text_of((reply,)) for reply in req.messages)
    finally:
        agent.close()


def test_apiagent_default_legacy_prompt_and_response_are_unchanged():
    messages = real_messages()
    original = deepcopy(messages)
    api = object.__new__(APIAgent)
    seen = []
    reply = "<think>Use the legacy response recipe.</think>\n" + ACTION
    api._call_api_with_usage = lambda request: (seen.append(deepcopy(request)) or reply, {"prompt_tokens": 2})
    assert api.build_request_messages({0: {"messages": messages}})[0] == original
    result, trace = api.forward_with_trace({0: {"messages": messages}})
    assert seen == [original]
    assert result[0] == {"action": ACTION, "assistant": reply}
    assert trace[0]["assistant_raw"] == reply
    assert messages == original
    assert "<think>" in seen[0][0]["content"]


@pytest.mark.parametrize("name", RESOURCES)
def test_every_native_resource_example_is_valid_for_the_real_submission_parser(name):
    from agent_ark.agent.simple_agent.action_submission import translate_submission
    text = resources.files("agent_ark.agent.simple_agent").joinpath("info", name).read_text(encoding="utf-8")
    examples = [json.loads(line) for line in text.splitlines() if line.startswith('{"format"')]
    assert examples, "Every replacement carries a usable native submit_action example"
    for index, arguments in enumerate(examples):
        assert set(arguments) == {"format", "content"}
        completion = submission(arguments["format"], arguments["content"], call_id=f"example-{index}")
        translated, event = translate_submission(completion, max_bytes=120_000, max_characters=120_000)
        assert translated.text and not translated.tool_calls and event["source"] == "simple_agent_adapter"
        assert translated.context_messages[0].tool_calls[0].name == "submit_action"


def test_native_prompt_resources_are_available_in_an_installed_wheel_away_from_checkout(tmp_path):
    project = tmp_path / "wheel source 中文"
    project.mkdir()
    shutil.copytree(PROJECT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(PROJECT / name, project / name)
    built = subprocess.run([sys.executable, "-B", "-m", "pip", "wheel", "--no-index", "--no-deps",
                            "--no-build-isolation", "--wheel-dir", str(tmp_path / "dist"), str(project)],
                           cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next((tmp_path / "dist").glob("*.whl"))
    installed = subprocess.run([sys.executable, "-B", "-m", "pip", "install", "--no-index", "--no-deps",
                                "--target", str(tmp_path / "installed package 中文"), str(wheel)],
                               cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert installed.returncode == 0, installed.stdout + installed.stderr
    check = """
import importlib.resources as r,json,pathlib,sys
sys.path.insert(0,sys.argv[1])
from agent_ark.agent.simple_agent.prompt_adapter import adapt_native_environment_pin
from agent_ark.agent.simple_agent.types import Message
from agent_ark.ark_env.direct_env import EnvInfoManager
root=r.files('agent_ark.agent.simple_agent').joinpath('info')
names=json.loads(sys.argv[2])
for name in names:
 text=root.joinpath(name).read_text(encoding='utf-8')
 assert text and 'submit_action' in text,name
manager=object.__new__(EnvInfoManager);manager.env_config={}
from agent_ark.utils.parse_utils import parse_system_task_prompt
source=parse_system_task_prompt(manager._get_system_prompt(action_mode='func'))[0]
adapted=adapt_native_environment_pin(Message.text('system',source))
assert adapted.parts[0].text!=source and 'submit_action' in adapted.parts[0].text
print('installed wheel native prompt resources passed')
"""
    checked = subprocess.run([sys.executable, "-I", "-B", "-c", check, str(tmp_path / "installed package 中文"), json.dumps(RESOURCES)],
                             cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "installed wheel native prompt resources passed" in checked.stdout
