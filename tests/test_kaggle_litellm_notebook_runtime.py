"""The formal LiteLLM notebook imports and runs offline against local fakes."""
from __future__ import annotations

import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_ark.ark_eval.evaluation_identity import build_evaluation_identity
from agent_ark.ark_eval.run_api_agent import build_model_runtimes
from test_kaggle_notebook_runtime import load_notebook, prepare_full_run


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "benchmarks/kaggle"
NOTEBOOK = KAGGLE / "agentark_litellm_task.ipynb"


def generator():
    spec = importlib.util.spec_from_file_location("litellm_kaggle_generator", KAGGLE / "generate_agentark_kaggle_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def notebook(tmp_path, monkeypatch):
    return load_notebook(NOTEBOOK, tmp_path, monkeypatch,
                         expected_commit=generator().DEFAULT_LITELLM_WRAPPER_COMMIT)


def test_formal_litellm_notebook_import_keeps_generic_defaults_and_distinct_artifacts(notebook):
    assert notebook["DEFAULT_MODEL_BACKEND"] == "litellm"
    assert notebook["DEFAULT_TOOL_CHOICE"] is None
    assert notebook["DEFAULT_TASK_NAME"] == "MarbleStop"
    assert notebook["DEFAULT_SEED_START"] == 1 and notebook["DEFAULT_SEED_END"] == 10
    assert notebook["DEFAULT_TASK_SCORE_SCALE_MIN"] == 0 and notebook["DEFAULT_TASK_SCORE_SCALE_MAX"] == 1
    assert notebook["DEFAULT_BENCHMARK_TASK_NAME"].endswith("-litellm")
    assert "_litellm_" in notebook["RESULT_STEM"]
    summary = json.loads(notebook["EXPORTED_SUMMARY_PATH"].read_text())
    assert summary["evaluation_status"] == "creation_smoke" and summary["full_eval"] is False


def test_formal_litellm_notebook_failure_exports_diagnostics_without_a_score(notebook):
    rows = [{"requested_group_seed": seed, "status": "error", "error_type": "BadRequestError",
             "error": "provider rejected a declared parameter", "score_reward": 0.0, "turns": 0}
            for seed in [1, 2]]
    prepare_full_run(notebook, rows)
    with pytest.raises(RuntimeError, match="no leaderboard score"):
        notebook["agentark_task_seeds_1_10"](None)
    summary = json.loads(notebook["EXPORTED_SUMMARY_PATH"].read_text())
    assert summary["evaluation_status"] == "failed" and summary["error_count"] == 2
    assert [json.loads(line) for line in notebook["EXPORTED_RESULT_PATH"].read_text().splitlines()] == rows


def test_formal_litellm_notebook_returns_normalized_mean_score(notebook):
    rows = [{"requested_group_seed": seed, "status": "ok", "score_reward": score, "turns": 6}
            for seed, score in [(1, -1.0), (2, 1.0)]]
    notebook["DEFAULT_TASK_SCORE_SCALE_MIN"] = -1.0
    notebook["DEFAULT_TASK_SCORE_SCALE_MAX"] = 1.0
    prepare_full_run(notebook, rows)
    assert notebook["agentark_task_seeds_1_10"](None) == 0.5
    summary = json.loads(notebook["EXPORTED_SUMMARY_PATH"].read_text())
    assert summary["evaluation_status"] == "complete" and summary["error_count"] == 0
    assert summary["mean_score_reward"] == 0.0 and summary["normalized_mean_score_reward"] == 0.5


@pytest.mark.parametrize("wire_model,api,tool_choice", [
    ("openai/gpt-6-astra", "responses", "required"),
    ("google/gemini-2.5-pro", "chat_completions", "required"),
    ("anthropic/claude-opus-5-5@default", "chat_completions", "auto"),
])
def test_formal_notebook_recipe_uses_real_sdk_local_http_and_native_submission(notebook, monkeypatch, wire_model, api, tool_choice):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    pytest.importorskip("litellm")
    received = []
    action = {"name": "Wait", "arguments": {"seconds": 1}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            first = len(received) == 1
            name = "list_files" if first else "submit_action"
            arguments = {"path": "records", "offset": 0, "limit": 10} if first else {
                "format": "tool_call", "content": json.dumps(action)}
            if api == "responses":
                payload = {"id": "synthetic-response", "object": "response", "created_at": 0, "model": wire_model,
                    "status": "completed", "output": [
                        {"type": "message", "id": "synthetic-message", "role": "assistant", "status": "completed", "phase": "commentary",
                         "content": [{"type": "output_text", "text": "Inspect visible state." if first else "Wait one second.", "annotations": []}]},
                        {"type": "function_call", "id": "fc-" + str(len(received)), "call_id": "call-" + str(len(received)),
                         "name": name, "arguments": json.dumps(arguments), "status": "completed"}],
                    "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}}
            else:
                payload = {"id": "synthetic-chat", "object": "chat.completion", "created": 0, "model": wire_model,
                    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant",
                        "content": "Inspect visible state." if first else "Wait one second.",
                        "tool_calls": [{"id": "call-" + str(len(received)), "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}
            encoded = json.dumps(payload).encode()
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
        notebook["_write_eval_config"](task_name="DoorSpin3Turns", seed_start=1, seed_end=2, env_id=0,
            model=wire_model, base_url=f"http://127.0.0.1:{server.server_port}/models/openapi/v1",
            max_completion_tokens=30000)
        recipe = yaml.safe_load(notebook["CONFIG_PATH"].read_text())["models"][0]
        monkeypatch.setenv("MODEL_PROXY_API_KEY", "synthetic-local-key")
        recipe["timeout_s"] = 20
        runtime = build_model_runtimes([recipe])[0]
        agent = runtime["agent"]
        assert agent.tool_choice == recipe["simple_agent"]["tool_choice"] == tool_choice
        assert runtime["evaluation_identity"] == build_evaluation_identity(recipe)
        other_choice = dict(recipe, simple_agent=dict(recipe["simple_agent"], tool_choice="required" if tool_choice == "auto" else "auto"))
        assert runtime["evaluation_identity"] != build_evaluation_identity(other_choice)
        actions, traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "Visible door state."}]}})
        assert json.loads(actions[0]["action"][11:-12]) == action
        expected_path = "/models/openapi/responses" if api == "responses" else "/models/openapi/v1/chat/completions"
        assert len(received) == 2 and all(path == expected_path and body["model"] == wire_model for path, body in received)
        assert all(body["tool_choice"] == tool_choice for path, body in received)
        assert traces[0]["usage"]["total_tokens"] == 24
        if api == "responses":
            assert all(body["store"] is False and "temperature" not in body for path, body in received)
            assert all(tool["strict"] is False for tool in received[0][1]["tools"])
            assert any(item.get("type") == "function_call_output" and item["call_id"] == "call-1"
                       for item in received[1][1]["input"])
        else:
            assert any(item.get("role") == "tool" and item["tool_call_id"] == "call-1"
                       for item in received[1][1]["messages"])
    finally:
        if agent is not None:
            agent.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("backend,wire_model,api_override,expected_api,temperature,tool_choice", [
    ("native", "google/gemini-2.5-pro", None, None, 1.0, "required"),
    ("native", "anthropic/claude-opus-5-5@default", None, None, 1.0, "required"),
    ("litellm", "openai/gpt-6-astra", None, "responses", None, "required"),
    ("litellm", "gpt-6-astra", "chat_completions", "chat_completions", None, "required"),
    ("litellm", "google/gemini-2.5-pro", None, "chat_completions", 1.0, "required"),
    ("litellm", "selected-model", "responses", "responses", 1.0, "required"),
    ("litellm", "anthropic/claude-opus-5-5@default", None, "chat_completions", 1.0, "auto"),
    ("litellm", "claude-opus-5-5-default", None, "chat_completions", 1.0, "auto"),
])
def test_backend_selection_preserves_wire_model_and_authors_explicit_sdk_route(
        notebook, monkeypatch, backend, wire_model, api_override, expected_api, temperature, tool_choice):
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", backend)
    if api_override:
        monkeypatch.setenv("AGENTARK_KAGGLE_LITELLM_API", api_override)
    notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                    model=wire_model, base_url="https://proxy.example/genai",
                                    max_completion_tokens=30000)
    config = yaml.safe_load(notebook["CONFIG_PATH"].read_text())
    model = config["models"][0]
    assert model["model"] == wire_model and model["base_url"] == "https://proxy.example/openapi"
    assert model["api_key_env"] == "MODEL_PROXY_API_KEY"
    assert model["temperature"] == temperature and model["max_retries"] == (0 if backend == "litellm" else 1)
    assert model["simple_agent"]["tools"]["enabled"] is True
    assert model["simple_agent"]["tool_choice"] == tool_choice
    if backend == "litellm":
        assert model["simple_agent"]["backend"] == "litellm"
        assert model["simple_agent"]["litellm"] == {"model": "openai/" + wire_model, "api": expected_api}
    else:
        assert "backend" not in model["simple_agent"] and "litellm" not in model["simple_agent"]
    assert "MODEL_PROXY_API_KEY=" not in notebook["CONFIG_PATH"].read_text()



def test_imported_backend_constant_is_editable_and_environment_override_wins(notebook, monkeypatch):
    assert notebook["DEFAULT_MODEL_BACKEND"] == "litellm"
    notebook["DEFAULT_MODEL_BACKEND"] = "native"
    assert notebook["_selected_model_backend"]() == "native"
    assert notebook["_selected_litellm_api"]("unknown-model") == "chat_completions"
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", "litellm")
    assert notebook["_selected_model_backend"]() == "litellm"


@pytest.mark.parametrize("wire_model", [
    "claude-opus-5-5", "claude-opus-5-5-default", "claude-opus-5-5@default",
    "claude-opus-5.5", "claude-opus-5.5-default", "claude-opus-5.5@default",
    "anthropic/claude-opus-5-5@default", "anthropic/claude-opus-5.5",
])
def test_tool_choice_policy_matches_only_bounded_opus_5_5_aliases_in_litellm(notebook, wire_model):
    assert notebook["_selected_tool_choice"](wire_model, "litellm") == "auto"
    assert notebook["_selected_tool_choice"](wire_model, "native") == "required"


@pytest.mark.parametrize("wire_model", [
    "claude-opus-5-6", "claude-opus-4-5", "claude-sonnet-5-5", "opus-5-5",
    "claude-opus-5-50", "my-claude-opus-5-5", "claude-opus-5-5-thinking",
    "claude-opus-5-5@preview", "claude-opus-5-5-default-extra", "claude-opus-5-5/other",
    "openai/gpt-6-astra", "google/gemini-2.5-pro", "selected-model",
])
def test_unknown_model_tool_choice_policy_remains_required(notebook, wire_model):
    assert notebook["_selected_tool_choice"](wire_model, "litellm") == "required"


@pytest.mark.parametrize("backend,wire_model,default,override,expected", [
    ("litellm", "anthropic/claude-opus-5-5@default", "required", None, "required"),
    ("litellm", "anthropic/claude-opus-5-5@default", "auto", "required", "required"),
    ("litellm", "selected-model", None, "auto", "auto"),
    ("native", "selected-model", "auto", None, "auto"),
    ("native", "selected-model", "required", "auto", "auto"),
])
def test_tool_choice_explicit_default_and_environment_override_are_authored_before_request(
        notebook, monkeypatch, backend, wire_model, default, override, expected):
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", backend)
    notebook["DEFAULT_TOOL_CHOICE"] = default
    if override is not None:
        monkeypatch.setenv("AGENTARK_KAGGLE_TOOL_CHOICE", override)
    notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                    model=wire_model, base_url="https://proxy.example",
                                    max_completion_tokens=30000)
    recipe = yaml.safe_load(notebook["CONFIG_PATH"].read_text())["models"][0]
    assert recipe["simple_agent"]["tool_choice"] == expected


@pytest.mark.parametrize("source", ["default", "environment"])
@pytest.mark.parametrize("invalid", ["", "any", "none", "automatic"])
def test_invalid_tool_choice_is_rejected_before_config_output(notebook, monkeypatch, source, invalid):
    if source == "default":
        notebook["DEFAULT_TOOL_CHOICE"] = invalid
    else:
        monkeypatch.setenv("AGENTARK_KAGGLE_TOOL_CHOICE", invalid)
    with pytest.raises(ValueError, match="TOOL_CHOICE.*required or auto"):
        notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                        model="anthropic/claude-opus-5-5@default", base_url="https://proxy.example",
                                        max_completion_tokens=30000)
    assert not notebook["CONFIG_PATH"].exists()



@pytest.mark.parametrize("api,suffix,expected", [
    ("responses", "/genai", "/openapi"),
    ("responses", "", "/openapi"),
    ("responses", "/openapi", "/openapi"),
    ("responses", "/openapi/v1", "/openapi"),
    ("chat_completions", "/openapi/v1", "/openapi/v1"),
])
def test_litellm_responses_canonicalizes_only_its_proxy_endpoint(notebook, monkeypatch, api, suffix, expected):
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", "litellm")
    monkeypatch.setenv("AGENTARK_KAGGLE_LITELLM_API", api)
    notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                    model="selected-model", base_url="https://proxy.example/models" + suffix,
                                    max_completion_tokens=30000)
    model = yaml.safe_load(notebook["CONFIG_PATH"].read_text())["models"][0]
    assert model["base_url"] == "https://proxy.example/models" + expected
    assert model["simple_agent"]["litellm"]["api"] == api



@pytest.mark.parametrize("backend", ["native", "litellm"])
def test_python_install_selects_optional_extra_only_for_explicit_litellm(notebook, monkeypatch, tmp_path, backend):
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", backend)
    root = tmp_path / "wrapper checkout with spaces"
    notebook["AGENTARK_ROOT"] = root
    if backend == "litellm":
        feature = root / "src/agent_ark/agent/simple_agent/litellm_model.py"
        feature.parent.mkdir(parents=True)
        feature.touch()
    notebook["VENV_PY"].parent.mkdir(parents=True, exist_ok=True)
    notebook["VENV_PY"].touch()
    notebook["_python_version"] = lambda command: (3, 10, 12)
    notebook["_ensure_venv_pip"] = lambda: None
    commands = []
    notebook["_run"] = lambda command, **kwargs: commands.append(command)
    notebook["_ensure_python_env"]()
    target = str(root) + "[litellm]" if backend == "litellm" else root
    assert commands == [[notebook["VENV_PY"], "-m", "pip", "install", "-q", "-e", target]]



def test_old_wrapper_pin_rejects_opt_in_sdk_before_package_install(notebook, monkeypatch):
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", "litellm")
    notebook["VENV_PY"].parent.mkdir(parents=True, exist_ok=True)
    notebook["VENV_PY"].touch()
    notebook["_python_version"] = lambda command: (3, 10, 12)
    notebook["_ensure_venv_pip"] = lambda: None
    notebook["_run"] = lambda *args, **kwargs: pytest.fail("Old wrapper pin reached optional package installation")
    with pytest.raises(RuntimeError, match="does not include.*LiteLLM.*wrapper.*--wrapper-commit"):
        notebook["_ensure_python_env"]()



@pytest.mark.parametrize("setting,value", [("BACKEND", "auto"), ("LITELLM_API", "automatic")])
def test_invalid_backend_or_protocol_is_rejected_during_config_authoring(notebook, monkeypatch, setting, value):
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", "litellm")
    monkeypatch.setenv("AGENTARK_KAGGLE_" + setting, value)
    with pytest.raises(ValueError, match="AGENTARK_KAGGLE_" + setting):
        notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                        model="selected-model", base_url="https://proxy.example",
                                        max_completion_tokens=30000)



@pytest.mark.parametrize("wire_model,expected_temperature,override,tool_choice,error", [
    ("openai/gpt-6-astra", None, None, "required", "temperature is unsupported"),
    ("selected-model", 1.0, None, "required", "temperature is unsupported"),
    ("anthropic/claude-opus-5-5@default", 1.0, None, "auto", "tool choice is unsupported"),
    ("anthropic/claude-opus-5-5@default", 1.0, "required", "required", "tool_choice type tool and any not supported"),
])
def test_litellm_authors_first_request_policy_without_parameter_fallback(
        notebook, monkeypatch, wire_model, expected_temperature, override, tool_choice, error):
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", "litellm")
    if override is not None:
        monkeypatch.setenv("AGENTARK_KAGGLE_TOOL_CHOICE", override)
    path = notebook["RESULT_PATH"]
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = []
    real_write = notebook["_write_eval_config"]

    def write_config(**kwargs):
        real_write(**kwargs)
        configs.append(yaml.safe_load(notebook["CONFIG_PATH"].read_text())["models"][0])

    def run(*args, **kwargs):
        rows = [{"status": "error", "requested_group_seed": seed, "error": error}
                for seed in [1, 2]]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return SimpleNamespace(returncode=1, stdout=error)

    notebook["_write_eval_config"] = write_config
    notebook["subprocess"] = SimpleNamespace(run=run, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
    results = notebook["_run_agentark_eval"]({}, task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                            max_attempts=2, model=wire_model,
                                            base_url="https://proxy.example/openapi", max_completion_tokens=30000)
    assert len(configs) == 1
    assert configs[0]["temperature"] == expected_temperature and configs[0]["max_retries"] == 0
    assert configs[0]["simple_agent"]["tool_choice"] == tool_choice
    assert not notebook["_ok_seeds"](results)
    assert not list(notebook["EXPORTED_RESULT_PATH"].parent.glob("*_before_temperature_retry.jsonl"))
