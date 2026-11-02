"""Optional SDK selection, evaluation records and resume through public runners.

Only the SDK and Unity environment are fakes: config loading, agent registry,
runtime construction, rollout, JSONL writing and resume filtering are real.
"""
from __future__ import annotations

from copy import deepcopy
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from agent_ark.ark_eval import run_api_agent, run_parallel_api_eval


ACTION = '<tool_call>{"name":"tap","arguments":{"x":3}}</tool_call>'


@pytest.mark.parametrize("agent_type,provider", [("api", "openai"), ("codex", "codex"), ("simple", "codex")])
def test_invalid_litellm_agent_combination_is_rejected_before_parallel_resume(tmp_path, agent_type, provider):
    """An existing success cannot hide a new invalid backend selection."""
    case = {"case_id": "existing-case", "task_name": "SyntheticVisibleTarget", "group_seed": 1, "env_id": 0}
    model = {"name": "same-visible-model", "model": "display-model", "agent_type": agent_type,
             "provider": provider, "simple_agent": {"backend": "litellm", "litellm": {
                 "model": "openai/synthetic-model", "api": "responses"}}}
    output = tmp_path / "existing-results.jsonl"
    output.write_text(json.dumps({"status": "ok", "requested_task_name": case["task_name"],
                                  "requested_group_seed": 1, "requested_env_id": 0,
                                  "model_name": model["name"], "model": model["model"],
                                  "provider": provider, "agent_type": agent_type}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(litellm|simple_agent)"):
        jobs = run_parallel_api_eval.build_parallel_eval_jobs([case], [model])
        run_parallel_api_eval.filter_existing_parallel_jobs(jobs, str(output))


class FakeLiteLLM:
    def __init__(self):
        self.requests = []
        self._lock = threading.Lock()

    async def acompletion(self, **kwargs):
        with self._lock:
            self.requests.append(("chat_completions", deepcopy(kwargs)))
        return {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": ACTION}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}

    async def aresponses(self, **kwargs):
        with self._lock:
            self.requests.append(("responses", deepcopy(kwargs)))
        return {"id": "synthetic-response", "status": "completed", "output": [
            {"type": "message", "id": "synthetic-message", "role": "assistant", "status": "completed",
             "phase": "final_answer", "content": [{"type": "output_text", "text": ACTION, "annotations": []}]}],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
@pytest.mark.parametrize("first_api", ["chat_completions", "responses"])
def test_litellm_public_evaluation_entry_records_and_resumes_own_protocol(tmp_path, monkeypatch, parallel, first_api):
    sdk = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", sdk)
    environments = []
    environment_lock = threading.Lock()

    class FakeEnv:
        max_attempts = 1
        max_steps_per_attempt = 1
        current_attempt_index = 1

        def __init__(self, config):
            messages = config["env_config_overrides"]["env_wrapper_cfg"]["context_manager"]["messages"]
            assert messages == {"enabled": True, "only_return_messages": True,
                                "append_only": True, "return_mode": "delta"}
            self.config = config
            self.actions = []
            self.closed = False
            self._selected_env_cfg = {}
            with environment_lock:
                environments.append(self)

        def reset(self, *, task_name, group_seed, env_id, **kwargs):
            self._selected_env_cfg = {"task_name": task_name, "group_seed": group_seed, "env_id": env_id}
            return {0: {"messages": [{"role": "system", "content": "Return the visible tap action."},
                                     {"role": "user", "content": "Visible target x=3."}],
                        "step_msg": "Visible target x=3.", "message_mode": "delta"}}, {"attempt": {"index": 1}}

        def step(self, action, *, info):
            # Assert the actual environment payload, not a preview or the SDK's
            # full response. The evaluator preserves the action extraction ABI.
            assert action == {0: {"action": ACTION, "assistant": ACTION}}
            self.actions.append(deepcopy(action))
            return ({0: {"messages": [{"role": "user", "content": "Target selected."}],
                         "step_msg": "Target selected."}},
                    {0: 1.0}, {0: True},
                    {"rollout": {"success": True, "truncated": False},
                     "attempt": {"index": 1, "done": True, "success": True}})

        def close(self):
            self.closed = True

    # The public runners still build the registered SimpleAgent and invoke the
    # real evaluator; only machine/runtime setup is replaced alongside FakeEnv.
    for runner in (run_api_agent, run_parallel_api_eval):
        monkeypatch.setattr(runner, "ArkEnv", FakeEnv)
        monkeypatch.setattr(runner, "ensure_runtime_pool_range", lambda *args, **kwargs: None)
        monkeypatch.setattr(runner, "resolve_runtime_sandbox_cfg", lambda value: value)
    monkeypatch.setattr(run_api_agent, "_eval_env_uses_task_store", lambda _: False)

    output = tmp_path / "results.jsonl"
    config_path = tmp_path / "evaluation.json"
    # Two cases cause the actual parallel executor to construct separate agents
    # on worker threads. The serial entry follows the same two one-step cases.
    cases = [{"case_id": "seed-" + str(seed), "task_name": "SyntheticVisibleTarget", "group_seed": seed, "env_id": 0}
             for seed in (1, 2)]
    model = {"name": "same-visible-model", "model": "display-model", "provider": "openai",
             "agent_type": "simple", "api_key": "synthetic-explicit-key", "temperature": None,
             "max_retries": 0, "simple_agent": {"backend": "litellm", "litellm": {
                 "model": "openai/synthetic-model", "api": first_api}}}
    config = {"env_cfg": {"mod_path": str(tmp_path / "unused-fake-mods")}, "models": [model], "eval": {
        "cases": cases, "output_path": str(output), "skip_existing_results": True,
        "max_parallel_envs": 2, "allow_shared_runtime_parallel": True, "stop_on_error": True}}
    runner = run_parallel_api_eval if parallel else run_api_agent

    def run_current_config():
        config_path.write_text(json.dumps(config), encoding="utf-8")
        runner.main(SimpleNamespace(config=str(config_path)))
        return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    first = run_current_config()
    assert len(first) == 2
    assert len(sdk.requests) == 2
    assert {api for api, _ in sdk.requests} == {first_api}
    assert all(request["model"] == "openai/synthetic-model" and request["api_key"] == "synthetic-explicit-key"
               for _, request in sdk.requests)
    assert len(environments) == 2
    for row in first:
        assert row["status"] == "ok"
        assert row["agent_type"] == "simple" and row["model"] == "display-model"
        assert row["rollout_success"] is True and row["score_reward"] == 1.0 and row["turns"] == 1
        assert row["evaluation_identity"]["agent_type"] == "simple"
        trace = row["steps"][0]["raw_response_by_agent"]["0"]
        assert trace["simple_agent"]["backend"] == "litellm"
        assert trace["simple_agent"]["litellm"]["api"] == first_api
        assert trace["model_requests"][0]["request_transport"]["sdk_method"] == (
            "aresponses" if first_api == "responses" else "acompletion")
        if parallel:
            assert row["parallel_eval"] is True and row["parallel_slot_index"] in (0, 1)
    if parallel:
        assert {row["parallel_slot_index"] for row in first} == {0, 1}
    assert all(env.closed and len(env.actions) == 1 for env in environments)

    # Same LiteLLM configuration must resume its saved successful cases without
    # another model request or environment; the JSONL is left byte-for-byte intact.
    saved_bytes = output.read_bytes()
    assert run_current_config() == first
    assert output.read_bytes() == saved_bytes
    assert len(sdk.requests) == 2 and len(environments) == 2

    # Reusing name/model/seed while changing the selected API must run both cases
    # and retain the previous protocol's records. This checks the observable
    # resume contract, without reproducing the identity hashing implementation.
    second_api = "responses" if first_api == "chat_completions" else "chat_completions"
    model["simple_agent"]["litellm"]["api"] = second_api
    combined = run_current_config()
    assert len(combined) == 4 and combined[:2] == first
    assert [api for api, _ in sdk.requests].count(second_api) == 2
    assert len(environments) == 4 and all(env.closed and len(env.actions) == 1 for env in environments)
    assert {row["evaluation_identity"]["config_sha256"] for row in combined[2:]}.isdisjoint(
        {row["evaluation_identity"]["config_sha256"] for row in first})
    assert all(row["status"] == "ok" for row in combined)
