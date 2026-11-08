"""Policy changes must not resume or reinterpret another policy's records."""
from copy import deepcopy
import json
import sys

import pytest

from agent_ark.agent.simple_agent.runtime import SimpleRuntime
from agent_ark.ark_eval.evaluation_identity import build_evaluation_identity
from agent_ark.ark_eval.run_api_agent import load_existing_eval_results
from agent_ark.ark_eval.run_parallel_api_eval import (
    _model_stub, build_parallel_eval_jobs, filter_existing_parallel_jobs,
)
from agent_ark.ark_eval.run_replay import ReplayAgent


CASE = {"case_id": "one", "task_name": "TaskA", "group_seed": 1, "env_id": 0}
BASE = {"name": "same-model", "agent_type": "simple", "provider": "openai", "model": "test-model"}


def configured(policy):
    config = deepcopy(BASE)
    if policy is not None:
        config["simple_agent"] = {"context": {"compaction_policy": policy}}
    return config


def completed_row(config):
    stub = _model_stub(config, 0)
    return {
        "status": "ok", "requested_task_name": CASE["task_name"],
        "requested_group_seed": 1, "requested_env_id": 0,
        "model_name": BASE["name"], "score_reward": 1,
        **{key: stub[key] for key in ("model", "provider", "agent_type", "evaluation_identity")},
    }


@pytest.mark.parametrize("old_policy", ["whole_turn_v1", "message_chunks_v1"])
@pytest.mark.parametrize("parallel", [False, True])
def test_completed_results_resume_only_the_selected_policy(tmp_path, old_policy, parallel):
    old, new = configured(old_policy), configured("tool_rounds_v1")
    assert build_evaluation_identity(old) != build_evaluation_identity(new)
    path = tmp_path / "results.jsonl"
    for saved, selected, expected in ((old, new, 0), (new, old, 0), (new, new, 1)):
        original = json.dumps(completed_row(saved)) + "\n"
        path.write_text(original, encoding="utf-8")
        if parallel:
            pending, existing = filter_existing_parallel_jobs(
                build_parallel_eval_jobs([CASE], [selected]), str(path))
            assert len(pending) == 1 - expected
        else:
            existing = load_existing_eval_results(str(path), [CASE], [_model_stub(selected, 0)])
        assert len(existing) == expected
        assert path.read_text(encoding="utf-8") == original


def test_omitted_policy_identifies_the_new_tool_round_default():
    from agent_ark.agent.simple_agent.runtime import SimpleRuntimeConfig

    assert SimpleRuntimeConfig().compaction_policy == "tool_rounds_v1"
    assert build_evaluation_identity(configured(None)) == build_evaluation_identity(configured("tool_rounds_v1"))


@pytest.mark.parametrize("policy", [None, "whole_turn_v1", "message_chunks_v1", "tool_rounds_v1"])
def test_saved_action_replay_preserves_audit_without_running_compaction(monkeypatch, policy):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Saved-action replay invoked model compaction")

    monkeypatch.setattr(SimpleRuntime, "complete", forbidden)
    action = '<tool_call>{"name":"ApplyImpulse","arguments":{"forceScale":1.8}}</tool_call>'
    payload = {
        "assistant_raw": "A short saved rationale.", "action_extracted": action,
        "model_requests": [{"purpose": "summary", "status": "completed",
            "compaction": {"version": 1, "policy": policy or "historical", "stage": "leaf"},
            "completion": "Saved memory; preserve the executed tool result."}],
        "agent_events": [{"kind": "compaction", "policy": policy,
                          "source_sequences": [1, 2, 3]}],
        "simple_agent": {"context": {} if policy is None else {"compaction_policy": policy}},
    }
    record = {"steps": [{"turn_index": 0, "raw_response_by_agent": {"0": payload}}]}
    original = deepcopy(record)
    replay = ReplayAgent(record)
    responses, traces = replay.forward_with_trace({0: {"messages": []}})
    assert responses[0] == {"action": action, "assistant": payload["assistant_raw"]}
    for key in ("model_requests", "agent_events", "simple_agent"):
        assert traces[0][key] == payload[key]
    assert record == original
    traces[0]["model_requests"][0]["completion"] = "Mutation in a viewer must not rewrite the record."
    assert record == original


def test_litellm_responses_compacts_inside_one_environment_decision(monkeypatch, tmp_path):
    from agent_ark.agent.simple_agent.agent import SimpleAgent
    from agent_ark.agent.simple_agent.types import TextPart, ToolSpec

    executed, requests = [], []
    action_round = 0
    summary_rounds = 0

    class SDK:
        async def aresponses(self, **parameters):
            nonlocal action_round, summary_rounds
            requests.append(deepcopy(parameters))
            source = parameters["input"]
            if not parameters.get("tools"):
                summary_rounds += 1
                output = [{"type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text":
                        f"{len(executed)} host operations completed. Their results are retained; do not repeat them."}]}]
            else:
                action_round += 1
                if action_round == 2:
                    assert any(item.get("type") == "function_call_output" for item in source)
                if action_round == 3:
                    # A fresh provider conversation, with the completed chain
                    # represented by notes rather than altered signed calls.
                    assert not any(item.get("type") in ("function_call", "function_call_output", "reasoning")
                                   for item in source)
                    assert "Model notes; not environment facts" in json.dumps(source)
                    assert "Completed visible operation result. " + "r" * 7000 in json.dumps(source)
                if action_round <= 2:
                    name = "host_operation"
                    arguments = {"number": action_round}
                else:
                    name = "submit_action"
                    arguments = {"format": "tool_call", "content": json.dumps({
                        "name": "ApplyImpulse", "arguments": {"forceScale": 1.8}})}
                output = [{"type": "reasoning", "id": f"reason-{action_round}", "summary": [],
                           "encrypted_content": "synthetic-opaque-state"},
                          {"type": "function_call", "id": f"fc-{action_round}",
                           "call_id": f"call-{action_round}", "name": name,
                           "arguments": json.dumps(arguments), "status": "completed"}]
            return {"status": "completed", "output": output,
                    "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}}

    monkeypatch.setitem(sys.modules, "litellm", SDK())

    def setup(registry):
        def operation(arguments, session):
            executed.append(arguments["number"])
            return (TextPart("Completed visible operation result. " + "r" * 7000),)

        registry.register(ToolSpec("host_operation", "Perform a visible operation.", json.dumps({
            "type": "object", "properties": {"number": {"type": "integer"}},
            "required": ["number"]})), operation)

    agent = SimpleAgent(api_key="synthetic-only", temperature=None, tool_setup=setup,
        simple_agent_config={"backend": "litellm",
            "litellm": {"model": "openai/synthetic-model", "api": "responses"},
            # One fresh result plus instructions and declared tools fits;
            # two accumulated results require older-history compaction.
            "context": {"max_context_characters": 18000,
                        "max_context_tokens": 100000, "output_reserve_tokens": 128},
            "tools": {"enabled": True, "workspace_parent": str(tmp_path)}})
    try:
        responses, traces = agent.forward_with_trace({0: {"messages": [
            {"role": "user", "content": "Choose the impulse after completing two host operations."}]}})
        expected = '<tool_call>{"name":"ApplyImpulse","arguments":{"forceScale":1.8}}</tool_call>'
        assert responses[0]["action"] == expected
        assert executed == [1, 2]
        assert action_round == 3 and summary_rounds == 1
        assert len(requests) == 4
        trace = traces[0]
        assert [call["purpose"] for call in trace["model_requests"]] == [
            "action", "action", "summary", "action"]
        assert len([event for event in trace["agent_events"] if event["kind"] == "action_translation"]) == 1
        record = {"steps": [{"turn_index": 0, "raw_response_by_agent": {"0": trace}}]}
        replayed, replay_trace = ReplayAgent(record).forward_with_trace({0: {"messages": []}})
        assert replayed[0]["action"] == expected
        assert replay_trace[0]["model_requests"] == trace["model_requests"]
        assert executed == [1, 2] and len(requests) == 4
    finally:
        agent.close()
