"""The formal LiteLLM notebook imports and runs offline against local fakes."""
from __future__ import annotations

import importlib.util
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

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
    assert notebook["DEFAULT_LITELLM_API"] == "responses"
    assert notebook["DEFAULT_TEMPERATURE"] == "default"
    assert notebook["DEFAULT_TASK_NAME"] == "MarbleStop"
    assert notebook["DEFAULT_SEED_START"] == 1 and notebook["DEFAULT_SEED_END"] == 10
    assert notebook["DEFAULT_TASK_SCORE_SCALE_MIN"] == 0 and notebook["DEFAULT_TASK_SCORE_SCALE_MAX"] == 1
    assert notebook["DEFAULT_BENCHMARK_TASK_NAME"].endswith("-litellm-responses-auto")
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


@pytest.mark.parametrize("wire_model,response_shape", [
    ("openai/gpt-6.1-sol", "reasoning_phase"),
    ("anthropic/claude-opus-5-5@default", "message"),
    ("google/gemini-2.5-flash", "function_only"),
    ("xai/grok-4.6", "message"),
])
def test_uniform_responses_recipe_uses_real_sdk_images_host_continuation_and_submission(
        notebook, monkeypatch, wire_model, response_shape):
    """Synthetic vendor-shaped replies prove serialization, not cloud support."""
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
            output = []
            if response_shape == "reasoning_phase":
                output.append({"type": "reasoning", "id": "reason-" + str(len(received)),
                               "summary": [], "encrypted_content": "opaque-synthetic-reasoning"})
            if response_shape != "function_only":
                message = {"type": "message", "id": "message-" + str(len(received)),
                           "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": "Inspect visible state." if first else "Wait one second.", "annotations": []}]}
                if response_shape == "reasoning_phase":
                    message["phase"] = "commentary"
                output.append(message)
            output.append({"type": "function_call", "id": "fc-" + str(len(received)),
                           "call_id": "call-" + str(len(received)), "name": name,
                           "arguments": json.dumps(arguments), "status": "completed"})
            payload = {"id": "response-" + str(len(received)), "object": "response", "created_at": 0,
                       "model": wire_model, "status": "completed", "output": output,
                       "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}}
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
        assert agent.tool_choice == recipe["simple_agent"]["tool_choice"] == "auto"
        assert recipe["simple_agent"]["litellm"] == {"model": "openai/" + wire_model, "api": "responses"}
        assert runtime["evaluation_identity"] == build_evaluation_identity(recipe)
        other_choice = dict(recipe, simple_agent=dict(recipe["simple_agent"], tool_choice="required"))
        assert runtime["evaluation_identity"] != build_evaluation_identity(other_choice)
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "green").save(image, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
        content = [{"type": "text", "text": "Visible door state."},
                   {"type": "image_url", "image_url": {"url": image_url}}]
        actions, traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": content}]}})
        assert json.loads(actions[0]["action"][11:-12]) == action
        next_content = [{"type": "text", "text": "Wait action completed. Updated visible door state."}, content[1]]
        next_actions, next_traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": next_content}]}})
        assert json.loads(next_actions[0]["action"][11:-12]) == action
        assert len(received) == 3 and all(path == "/models/openapi/responses" and body["model"] == wire_model
                                          for path, body in received)
        assert all(body["tool_choice"] == "auto" for path, body in received)
        assert traces[0]["usage"]["total_tokens"] == 24
        assert next_traces[0]["usage"]["total_tokens"] == 12
        assert all(body["store"] is False and "temperature" not in body and "reasoning" not in body
                   and "reasoning_effort" not in body and body["max_output_tokens"] == 30000
                   for path, body in received)
        assert all(tool["strict"] is False for tool in received[0][1]["tools"])
        assert any(item.get("type") == "function_call_output" and item["call_id"] == "call-1"
                   for item in received[1][1]["input"])
        assert any(part.get("type") == "input_image" and part["image_url"] == image_url
                   for item in received[0][1]["input"] for part in item.get("content", []))
        assert any(part.get("type") == "input_image" for item in received[2][1]["input"]
                   for part in item.get("content", []))
        if response_shape == "reasoning_phase":
            assert any(item.get("type") == "reasoning" and item["encrypted_content"] == "opaque-synthetic-reasoning"
                       for item in received[1][1]["input"])
            assert any(item.get("type") == "message" and item.get("phase") == "commentary"
                       for item in received[1][1]["input"])
    finally:
        if agent is not None:
            agent.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("backend,wire_model,api_override,expected_api,temperature,tool_choice", [
    ("native", "google/gemini-2.5-pro", None, None, 1.0, "required"),
    ("native", "anthropic/claude-opus-5-5@default", None, None, 1.0, "required"),
    ("litellm", "openai/gpt-6.1-sol", None, "responses", None, "auto"),
    ("litellm", "gpt-6-astra", "chat_completions", "chat_completions", None, "auto"),
    ("litellm", "google/gemini-2.5-flash", None, "responses", None, "auto"),
    ("litellm", "selected-model", "responses", "responses", None, "auto"),
    ("litellm", "anthropic/claude-opus-5-5@default", None, "responses", None, "auto"),
    ("litellm", "xai/grok-4.6", None, "responses", None, "auto"),
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
    assert notebook["_selected_litellm_api"]() == "responses"
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", "litellm")
    assert notebook["_selected_model_backend"]() == "litellm"


@pytest.mark.parametrize("wire_model", [
    "model-never-seen-before", "future-vendor/future-model@preview",
    "openai/gpt-6.1-sol", "anthropic/claude-opus-5-5@default",
    "google/gemini-2.5-flash", "xai/grok-4.6", "arbitrary/claude-opus-5-50",
])
def test_first_request_policy_is_independent_of_wire_model_name(notebook, monkeypatch, wire_model):
    notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                    model=wire_model, base_url="https://proxy.example",
                                    max_completion_tokens=30000)
    recipe = yaml.safe_load(notebook["CONFIG_PATH"].read_text())["models"][0]
    assert recipe["model"] == wire_model
    assert recipe["simple_agent"]["litellm"] == {"model": "openai/" + wire_model, "api": "responses"}
    assert recipe["temperature"] is None and recipe["simple_agent"]["tool_choice"] == "auto"
    assert "reasoning_effort" not in recipe and "reasoning" not in recipe
    assert notebook["_selected_tool_choice"]("native") == "required"


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


@pytest.mark.parametrize("default,override,expected", [
    ("chat_completions", None, "chat_completions"),
    ("chat_completions", "responses", "responses"),
    ("responses", "chat_completions", "chat_completions"),
])
def test_api_constant_and_environment_override_choose_first_request(notebook, monkeypatch, default, override, expected):
    notebook["DEFAULT_LITELLM_API"] = default
    if override is not None:
        monkeypatch.setenv("AGENTARK_KAGGLE_LITELLM_API", override)
    notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                    model="arbitrary-future-model", base_url="https://proxy.example",
                                    max_completion_tokens=30000)
    recipe = yaml.safe_load(notebook["CONFIG_PATH"].read_text())["models"][0]
    assert recipe["simple_agent"]["litellm"]["api"] == expected


@pytest.mark.parametrize("entry_point", ["writer", "runner"])
@pytest.mark.parametrize("backend,default,override,explicit,expected", [
    ("litellm", "default", None, {}, None),
    ("native", "default", None, {}, 1.0),
    ("litellm", 0.25, None, {}, 0.25),
    ("litellm", None, None, {}, None),
    ("litellm", 0.25, "default", {}, None),
    ("native", 0.25, "default", {}, 1.0),
    ("litellm", 0.25, "null", {}, None),
    ("native", 1.0, "none", {}, None),
    ("litellm", 0.25, "0.75", {}, 0.75),
    ("litellm", 0.25, "0.75", {"temperature": None}, None),
    ("litellm", 0.25, "0.75", {"temperature": 0.5}, 0.5),
    ("native", 1.0, "0.75", {"temperature": None}, None),
])
def test_temperature_resolution_is_shared_and_explicit_none_wins(
        notebook, monkeypatch, entry_point, backend, default, override, explicit, expected):
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", backend)
    notebook["DEFAULT_TEMPERATURE"] = default
    if override is not None:
        monkeypatch.setenv("AGENTARK_KAGGLE_TEMPERATURE", override)
    kwargs = dict(task_name="Snake", seed_start=1, seed_end=1, env_id=0,
                  model="arbitrary-future-model", base_url="https://proxy.example", max_completion_tokens=30000)
    if entry_point == "writer":
        notebook["_write_eval_config"](**kwargs, **explicit)
    else:
        def run(*args, **kwargs):
            notebook["RESULT_PATH"].write_text(json.dumps({"status": "ok", "requested_group_seed": 1}) + "\n")
            return SimpleNamespace(returncode=0, stdout="")
        notebook["subprocess"] = SimpleNamespace(run=run, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
        notebook["_run_agentark_eval"]({}, max_attempts=1, **kwargs, **explicit)
    recipe = yaml.safe_load(notebook["CONFIG_PATH"].read_text())["models"][0]
    assert recipe["temperature"] == expected


@pytest.mark.parametrize("source", ["default", "environment", "argument"])
@pytest.mark.parametrize("invalid", ["", "true", "nan", "inf", "-Infinity", "automatic"])
def test_invalid_temperature_fails_before_config_output(notebook, monkeypatch, source, invalid):
    explicit = {}
    if source == "default":
        notebook["DEFAULT_TEMPERATURE"] = invalid
    elif source == "environment":
        monkeypatch.setenv("AGENTARK_KAGGLE_TEMPERATURE", invalid)
    else:
        explicit["temperature"] = invalid
    with pytest.raises(ValueError, match="TEMPERATURE.*finite number"):
        notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                        model="arbitrary-model", base_url="https://proxy.example",
                                        max_completion_tokens=30000, **explicit)
    assert not notebook["CONFIG_PATH"].exists()


@pytest.mark.parametrize("invalid", [True, False, float("nan"), float("inf"), {}, []])
def test_temperature_rejects_boolean_nonfinite_and_non_numeric_python_values(notebook, invalid):
    with pytest.raises(ValueError, match="TEMPERATURE.*finite number"):
        notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                        model="arbitrary-model", base_url="https://proxy.example",
                                        max_completion_tokens=30000, temperature=invalid)
    assert not notebook["CONFIG_PATH"].exists()


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


@pytest.mark.parametrize("source", ["default", "environment"])
@pytest.mark.parametrize("invalid", ["", "automatic", "chat", "none"])
def test_invalid_api_has_no_implicit_route(notebook, monkeypatch, source, invalid):
    if source == "default":
        notebook["DEFAULT_LITELLM_API"] = invalid
    else:
        monkeypatch.setenv("AGENTARK_KAGGLE_LITELLM_API", invalid)
    with pytest.raises(ValueError, match="LITELLM_API.*chat_completions or responses"):
        notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                        model="arbitrary-model", base_url="https://proxy.example",
                                        max_completion_tokens=30000)
    assert not notebook["CONFIG_PATH"].exists()



@pytest.mark.parametrize("wire_model,expected_temperature,override,tool_choice,error", [
    ("openai/gpt-6.1-sol", None, None, "auto", "temperature is unsupported"),
    ("selected-model", 0.6, None, "auto", "temperature is unsupported"),
    ("selected-model", None, None, "auto", "Responses API is unsupported"),
    ("google/gemini-2.5-flash", None, None, "auto", "Multiple tools are supported only when they are all search tools"),
    ("anthropic/claude-opus-5-5@default", None, "required", "required", "tool_choice type tool and any not supported"),
])
def test_litellm_authors_first_request_policy_without_parameter_fallback(
        notebook, monkeypatch, wire_model, expected_temperature, override, tool_choice, error):
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", "litellm")
    if expected_temperature is not None:
        monkeypatch.setenv("AGENTARK_KAGGLE_TEMPERATURE", str(expected_temperature))
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
    assert configs[0]["simple_agent"]["litellm"]["api"] == "responses"
    assert not notebook["_ok_seeds"](results)
    assert not list(notebook["EXPORTED_RESULT_PATH"].parent.glob("*_before_temperature_retry.jsonl"))


def test_native_selection_keeps_its_existing_unsupported_temperature_retry(notebook, monkeypatch):
    monkeypatch.setenv("AGENTARK_KAGGLE_BACKEND", "native")
    # A native retry's explicit None must win over this environment setting.
    monkeypatch.setenv("AGENTARK_KAGGLE_TEMPERATURE", "1")
    configs = []
    real_write = notebook["_write_eval_config"]

    def write_config(**kwargs):
        real_write(**kwargs)
        configs.append(yaml.safe_load(notebook["CONFIG_PATH"].read_text())["models"][0])

    def run(*args, **kwargs):
        rows = [{"status": "error", "requested_group_seed": 1, "error": "temperature is unsupported"}]
        if len(configs) == 2:
            rows = [{"status": "ok", "requested_group_seed": 1, "score_reward": 1.0}]
        notebook["RESULT_PATH"].write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return SimpleNamespace(returncode=0 if len(configs) == 2 else 1, stdout="")

    notebook["_write_eval_config"] = write_config
    notebook["subprocess"] = SimpleNamespace(run=run, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
    results = notebook["_run_agentark_eval"]({}, task_name="Snake", seed_start=1, seed_end=1, env_id=0,
                                            max_attempts=2, model="arbitrary-model",
                                            base_url="https://proxy.example/openapi", max_completion_tokens=30000)
    assert [config["temperature"] for config in configs] == [1.0, None]
    assert all(config["simple_agent"]["tool_choice"] == "required" and config["max_retries"] == 1
               and "backend" not in config["simple_agent"] for config in configs)
    assert notebook["_ok_seeds"](results) == {1}
    assert len(list(notebook["EXPORTED_RESULT_PATH"].parent.glob("*_before_temperature_retry.jsonl"))) == 1
