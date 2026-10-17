"""Offline summary generation budgets; reasoning parameters stay inherited."""
import asyncio
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent.agent import SimpleAgent
from agent_ark.agent.simple_agent.models import CallableModel
from agent_ark.agent.simple_agent.runtime import SimpleRuntime, SimpleRuntimeConfig
from agent_ark.agent.simple_agent.types import Message, ModelRequest


@pytest.mark.parametrize("provider,host,effort,thinking,extra,wire_effort", [
    ("openai", "api.openai.com", "max", None, {}, "max"),
    ("generic", "example.invalid", "max", None, {}, "max"),
    ("none", "example.invalid", "high", None, {}, "high"),
    ("auto", "api.openai.com", "max", None, {}, "max"),
    ("openrouter", "openrouter.ai", "max", True, {"reasoning": {"enabled": True, "effort": "max"}}, None),
    ("auto", "openrouter.ai", "max", None, {"reasoning": {"enabled": True, "effort": "max"}}, None),
    ("openrouter", "openrouter.ai", None, None, {"reasoning": {"enabled": True}}, None),
    ("openrouter", "openrouter.ai", None, False, {"reasoning": {"enabled": False}}, None),
    ("volcengine", "example.invalid", "high", True, {"thinking": {"type": "enabled"}}, "high"),
    ("volcengine", "example.invalid", None, False, {"thinking": {"type": "disabled"}}, None),
    ("dashscope", "dashscope.aliyuncs.com", None, None,
     {"enable_thinking": True, "thinking_budget": 81920}, None),
    ("qwen", "dashscope.aliyuncs.com", None, None,
     {"enable_thinking": True, "thinking_budget": 81920}, None),
    ("auto", "dashscope.aliyuncs.com", None, None,
     {"enable_thinking": True, "thinking_budget": 81920}, None),
    ("dashscope", "dashscope.aliyuncs.com", None, False, {"enable_thinking": False}, None),
    ("openai", "api.openai.com", "none", None, {}, "none"),
    ("openai", "api.openai.com", None, None, {}, None),
])
@pytest.mark.parametrize("token_parameter", ["max_completion_tokens", "max_tokens"])
def test_actual_http_wire_separates_summary_budget_and_preserves_action_reasoning(
        provider, host, effort, thinking, extra, wire_effort, token_parameter):
    agent = SimpleAgent(provider=provider, base_url="https://" + host + "/v1", reasoning_effort=effort,
        thinking=thinking, max_completion_tokens=20480,
        simple_agent_config={"output_token_parameter": token_parameter})
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Complete notes."}}]}

    agent._backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    messages = (Message.text("user", "visible"),)

    async def run():
        await agent._backend.complete(ModelRequest(messages, purpose="action", max_output_tokens=24000))
        await agent._backend.complete(ModelRequest(messages, purpose="summary",
                                                  max_output_tokens=agent.context_config.summary_output_tokens))

    try:
        asyncio.run(run())
        action, summary = calls
        assert action[token_parameter] == 20480
        assert summary[token_parameter] == 32768
        assert action["extra_body"] == summary["extra_body"] == extra
        assert action.get("reasoning_effort") == summary.get("reasoning_effort") == wire_effort
        assert ("reasoning_effort" in action) is (wire_effort is not None)
        assert ("reasoning_effort" in summary) is (wire_effort is not None)
        assert agent.configuration_summary()["context"]["summary_output_tokens"] == 32768
    finally:
        agent.close()


def test_custom_summary_budget_is_independent_of_smaller_action_cap():
    agent = SimpleAgent(provider="openai", reasoning_effort="max", max_completion_tokens=64,
        simple_agent_config={"context": {"summary_output_tokens": 4096}})
    try:
        messages = (Message.text("user", "visible"),)
        runtime = agent._state(0).runtime
        action = agent._backend.request_parameters(runtime._model_request(messages, "action"))
        summary = agent._backend.request_parameters(runtime._model_request(messages, "summary"))
        assert action["max_completion_tokens"] == 64
        assert summary["max_completion_tokens"] == 4096
        assert action["reasoning_effort"] == summary["reasoning_effort"] == "max"
    finally:
        agent.close()


def test_first_summary_prompt_has_actual_custom_budgets_and_includes_reasoning():
    runtime = SimpleRuntime(CallableModel(lambda _: None), SimpleRuntimeConfig(
        summary_output_tokens=3456, summary_output_characters=7890, summary_output_bytes=12345))
    prompt = runtime._summary_messages(())[0].parts[0].text
    assert "3456 tokens, including any reasoning tokens" in prompt
    assert "7890 characters" in prompt and "12345 UTF-8 bytes" in prompt
    assert "only concise plain text model notes" in prompt
    assert "without private reasoning" in prompt
    assert "Summary retry instructions:" not in prompt
