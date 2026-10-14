"""Case-level failure handling stays separate from bounded summary recovery."""
import json
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent.types import AgentLimitError
from agent_ark.ark_eval import run_api_agent


@pytest.mark.parametrize("stop_on_error", [False, True])
def test_exhausted_summary_is_recorded_and_batch_continuation_respects_policy(tmp_path, monkeypatch, stop_on_error):
    output = tmp_path / "results.jsonl"
    cases = [{"case_id": name, "task_name": "visible-task", "group_seed": 1}
             for name in ("summary-failed", "next-case")]
    agent = SimpleNamespace(last_trace_by_agent={0: {
        "model_requests": [{"purpose": "summary", "status": "failed", "usage": {
            "input_tokens": 10, "output_tokens": 20, "total_tokens": 30}}],
        "agent_usage": {"summary_calls": 4},
    }})
    closed_agents = []
    agent.close = lambda: closed_agents.append(True)
    runtime = {"name": "simple-test", "model": "fake", "provider": "openai",
               "agent_type": "simple", "base_url": None, "api_key_env": None, "agent": agent}
    hooks = SimpleNamespace(enabled=False, start=lambda: None,
                            emit=lambda *args, **kwargs: None, close=lambda: None)
    environments = []

    class FakeEnv:
        def __init__(self, config):
            self.closed = False
            environments.append(self)

        def close(self):
            self.closed = True

    visited = []

    def evaluate_case(**kwargs):
        case = kwargs["case"]
        visited.append(case["case_id"])
        if case["case_id"] == "summary-failed":
            raise AgentLimitError("summary call bound exhausted")
        return {"status": "ok", "case_id": case["case_id"], "model_name": runtime["name"],
                "turns": 1, "rollout_success": True, "rollout_truncated": False, "total_reward": 1.0}

    monkeypatch.setattr(run_api_agent, "load_eval_config", lambda _: {
        "env_cfg": {}, "models": [], "eval": {"output_path": str(output), "stop_on_error": stop_on_error}})
    monkeypatch.setattr(run_api_agent, "ensure_runtime_pool_range", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_api_agent, "resolve_runtime_sandbox_cfg", lambda value: value)
    monkeypatch.setattr(run_api_agent, "build_eval_hook_manager", lambda _: (hooks, None))
    monkeypatch.setattr(run_api_agent, "build_model_runtimes", lambda *args, **kwargs: [runtime])
    monkeypatch.setattr(run_api_agent, "build_eval_cases", lambda *args: cases)
    monkeypatch.setattr(run_api_agent, "ArkEnv", FakeEnv)
    monkeypatch.setattr(run_api_agent, "evaluate_case", evaluate_case)
    monkeypatch.setattr(run_api_agent, "maybe_save_eval_trajectory", lambda **kwargs: None)

    if stop_on_error:
        with pytest.raises(AgentLimitError, match="summary call bound exhausted"):
            run_api_agent.main(SimpleNamespace(config="unused.yaml"))
    else:
        run_api_agent.main(SimpleNamespace(config="unused.yaml"))

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[0]["status"] == "error"
    assert records[0]["error_type"] == "AgentLimitError"
    assert records[0]["agent_failure_trace"]["0"]["model_requests"][0]["purpose"] == "summary"
    assert records[0]["agent_failure_trace"]["0"]["agent_usage"]["summary_calls"] == 4
    assert visited == (["summary-failed"] if stop_on_error else ["summary-failed", "next-case"])
    assert len(records) == (1 if stop_on_error else 2)
    if not stop_on_error:
        assert records[1]["status"] == "ok"
    assert all(env.closed for env in environments)
    assert len(closed_agents) == len(visited)
