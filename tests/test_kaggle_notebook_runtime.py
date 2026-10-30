"""Execute the importable Kaggle notebook locally without Kaggle or paid models."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from PIL import Image

from agent_ark.agent.simple_agent import SimpleAgent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "benchmarks/kaggle/new-benchmark-task-f8fdd.ipynb"


@pytest.fixture
def notebook(tmp_path, monkeypatch):
    """Run every original code cell, including its lightweight creation entry."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("AGENTARK_KAGGLE_") or key == "BENCHMARK_MODE":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENTARK_KAGGLE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("AGENTARK_KAGGLE_SEED_END", "2")
    monkeypatch.setenv("AGENTARK_KAGGLE_FULL_EVAL", "0")
    creation_returns = []

    def task(**options):
        def decorate(function):
            function.run = lambda llm: creation_returns.append(function(llm))
            return function
        return decorate

    def assert_in(expected, actual, **kwargs):
        assert expected in actual

    module = ModuleType("kaggle_benchmarks")
    module.task = task
    module.llm = object()
    module.assertions = SimpleNamespace(assert_in=assert_in)
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", module)
    namespace = {"__name__": "__notebook__"}
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] == "code":
            assert not cell.get("outputs") and cell.get("execution_count") is None
            exec(compile("".join(cell["source"]), f"{NOTEBOOK}:cell{index}", "exec"), namespace)
    assert creation_returns == [0.0]
    assert json.loads(namespace["EXPORTED_SUMMARY_PATH"].read_text())["evaluation_status"] == "creation_smoke"
    return namespace


def prepare_full_run(namespace, rows):
    namespace.update({
        "_should_run_full_eval": lambda: True,
        "_require_model_proxy_env": lambda: ("https://proxy.example/openapi", "dummy", "selected-model"),
        "_install_and_configure_agentark": lambda: {},
        "_download_task": lambda name: name,
        "_run_agentark_eval": lambda *args, **kwargs: rows,
    })


@pytest.mark.parametrize("errors", [1, 2])
def test_failed_full_notebook_exports_diagnostics_then_raises(notebook, errors):
    rows = [{"requested_group_seed": seed, "status": "error" if seed <= errors else "ok",
             "score_reward": 0.0, "turns": 0 if seed <= errors else 6,
             "error_type": "BadRequestError" if seed <= errors else None,
             "error": "tool support rejected" if seed <= errors else None}
            for seed in [1, 2]]
    prepare_full_run(notebook, rows)
    with pytest.raises(RuntimeError):
        notebook["agentark_task_seeds_1_10"](None)
    summary = json.loads(notebook["EXPORTED_SUMMARY_PATH"].read_text())
    assert summary["evaluation_status"] == "failed"
    assert summary["error_count"] == errors
    exported = [json.loads(line) for line in notebook["EXPORTED_RESULT_PATH"].read_text().splitlines()]
    assert exported == rows


def test_legitimate_zero_score_still_completes(notebook):
    rows = [{"requested_group_seed": seed, "status": "ok", "score_reward": 0.0, "turns": 6}
            for seed in [1, 2]]
    prepare_full_run(notebook, rows)
    assert notebook["agentark_task_seeds_1_10"](None) == 0.0
    summary = json.loads(notebook["EXPORTED_SUMMARY_PATH"].read_text())
    assert summary["evaluation_status"] == "complete" and summary["error_count"] == 0


def test_same_count_of_wrong_seeds_is_not_complete(notebook):
    rows = [{"requested_group_seed": seed, "status": "ok", "score_reward": 0.0}
            for seed in [1, 3]]
    with pytest.raises(AssertionError, match="seed"):
        notebook["_results_summary"](rows, seed_start=1, seed_end=2,
                                      task_score_scale_min=0, task_score_scale_max=1)


@pytest.mark.parametrize("suffix, expected", [
    ("", "/openapi"), ("/genai/", "/openapi"),
    ("/openapi/", "/openapi"), ("/openapi/v1/", "/openapi/v1"),
])
def test_config_accepts_proxy_url_forms(notebook, suffix, expected):
    notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                    model="selected-model", base_url="https://proxy.example" + suffix,
                                    max_completion_tokens=30000)
    config = yaml.safe_load(notebook["CONFIG_PATH"].read_text())
    model = config["models"][0]
    assert model["base_url"] == "https://proxy.example" + expected
    assert model["agent_type"] == "simple" and model["simple_agent"]["tools"]["enabled"]
    assert config["eval"]["group_seeds"] == {"start": 1, "end": 2}


def test_env106_wrapped_archive_extracts_without_an_extra_directory(notebook, tmp_path):
    archive = tmp_path / "published-layout-fixture.zip"
    root = notebook["ENV_ROOT"]
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(root.name + "/AgentArk.x86_64", b"fixture executable")
        stream.writestr(root.name + "/AgentArk_Data/Resources/Mods/config.yaml.bak", b"baseline")
    downloads = []

    def download(url, target, digest):
        assert url.endswith("/artifacts/envs/1.0.6/linux/AgentArk-env-1.0.6-linux.zip")
        assert digest == "f32061282f51bbc2ab36a4617d7af5eb6b4d349a944dd861171aba95c00ccdcc"
        downloads.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        notebook["shutil"].copyfile(archive, target)

    notebook["_download_verified"] = download
    notebook["_download_runtime"]()
    assert (root / "AgentArk.x86_64").read_bytes() == b"fixture executable"
    assert (root / "AgentArk_Data/Resources/Mods/config.yaml.bak").read_bytes() == b"baseline"
    assert not (root / root.name).exists()
    notebook["_download_runtime"]()
    assert len(downloads) == 1


def test_xvfb_executable_is_distinct_from_package(notebook, monkeypatch):
    probes, installs = [], []
    def which(name):
        probes.append(name)
        return "/usr/bin/" + name if name in ["git", "Xvfb"] else None
    monkeypatch.setattr(notebook["shutil"], "which", which)
    notebook["_run"] = lambda *args, **kwargs: installs.append(args)
    notebook["_ensure_os_packages"]()
    assert "Xvfb" in probes and not installs
    monkeypatch.setattr(notebook["shutil"], "which", lambda name: "/usr/bin/git" if name == "git" else None)
    notebook["_ensure_os_packages"]()
    assert "xvfb" in installs[-1][0]


@pytest.mark.parametrize("version, accepted", [
    ((3, 10, 0), False), ((3, 10, 1), True), ((3, 10, 12), True),
    ((3, 10, 13), False), ((3, 11, 0), False),
])
def test_auto_python_selection_respects_pinned_wrapper(notebook, monkeypatch, version, accepted):
    monkeypatch.setattr(notebook["shutil"], "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(notebook["subprocess"], "check_output", lambda *args, **kwargs: ".".join(map(str, version)) + "\n")
    monkeypatch.setattr(notebook["subprocess"], "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    selected = notebook["_python310_command"]()
    assert bool(selected) == accepted


def test_incompatible_explicit_python_is_rejected_early(notebook, monkeypatch):
    monkeypatch.setenv("AGENTARK_KAGGLE_PYTHON_BIN", "/custom/python")
    notebook["_python_version"] = lambda command: (3, 10, 13)
    with pytest.raises(RuntimeError, match="AGENTARK_KAGGLE_PYTHON_BIN"):
        notebook["_python310_command"]()


def test_incompatible_cached_environment_never_attempts_pip(notebook):
    notebook["VENV_PY"].parent.mkdir(parents=True, exist_ok=True)
    notebook["VENV_PY"].touch()  # A presence-only test fixture; never executed.
    notebook["_python_version"] = lambda command: (3, 10, 13)
    def unexpected_install(*args, **kwargs):
        pytest.fail("An incompatible cached interpreter reached installation")
    notebook["_run"] = unexpected_install
    notebook["_ensure_venv_pip"] = unexpected_install
    with pytest.raises(RuntimeError, match="Cached AgentArk environment"):
        notebook["_ensure_python_env"]()


def test_temperature_retry_cannot_reuse_old_success(notebook):
    path = notebook["RESULT_PATH"]
    path.parent.mkdir(parents=True, exist_ok=True)
    attempted_temperatures, configs = [], []
    real_write = notebook["_write_eval_config"]

    def write_config(**kwargs):
        attempted_temperatures.append(kwargs["temperature"])
        real_write(**kwargs)
        configs.append(yaml.safe_load(notebook["CONFIG_PATH"].read_text()))

    def run(*args, **kwargs):
        attempt = len(configs)
        existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        if attempt == 1:
            rows = [{"status": "ok", "requested_group_seed": 1, "score_reward": 1.0, "temperature": 1},
                    {"status": "error", "requested_group_seed": 2, "error": "temperature is unsupported"}]
        else:
            assert existing == [], "Old configuration success leaked into the fallback run"
            rows = [{"status": "error", "requested_group_seed": 1, "error": "different failure"},
                    {"status": "ok", "requested_group_seed": 2, "score_reward": 1.0, "temperature": None}]
        with path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        return SimpleNamespace(returncode=0, stdout="")

    notebook["_write_eval_config"] = write_config
    notebook["subprocess"] = SimpleNamespace(run=run, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
    results = notebook["_run_agentark_eval"]({}, task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                            max_attempts=2, model="selected-model",
                                            base_url="https://proxy.example/openapi", max_completion_tokens=30000)
    assert attempted_temperatures == [1.0, None]
    assert notebook["_ok_seeds"](results) == {2}
    archived = list(notebook["EXPORTED_RESULT_PATH"].parent.glob("*_before_temperature_retry.jsonl"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text().splitlines()[0])["requested_group_seed"] == 1


def test_generated_model_recipe_makes_real_sdk_function_requests(notebook):
    """Exercise HTTP serialization, host-tool continuation and submit_action."""
    received = []
    action = {"name": "ExecutePlan", "arguments": {"plan": "U1,L9"}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body))
            if len(received) == 1:
                name, arguments = "list_files", {"path": "records", "offset": 0, "limit": 10}
            else:
                name, arguments = "submit_action", {"format": "tool_call", "content": json.dumps(action)}
            response = {"id": "mock-response", "object": "chat.completion", "created": 0,
                        "model": "selected-model", "choices": [{"index": 0, "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": "Move toward food.", "tool_calls": [
                            {"id": f"call-{len(received)}", "type": "function", "function": {
                                "name": name, "arguments": json.dumps(arguments)}}]}}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
            encoded = json.dumps(response).encode()
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
        notebook["_write_eval_config"](task_name="Snake", seed_start=1, seed_end=2, env_id=0,
                                        model="selected-model", base_url=f"http://127.0.0.1:{server.server_port}",
                                        max_completion_tokens=30000)
        model = yaml.safe_load(notebook["CONFIG_PATH"].read_text())["models"][0]
        agent = SimpleAgent(api_key="dummy", base_url=model["base_url"], provider=model["provider"],
                            model=model["model"], temperature=model["temperature"], max_retries=0,
                            max_completion_tokens=model["max_completion_tokens"],
                            simple_agent_config=model["simple_agent"])
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "green").save(image, format="PNG")
        import base64
        observation = {0: {"messages": [{"role": "system", "content": "Control the snake."},
                       {"role": "user", "content": [{"type": "text", "text": "Find food."},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()}}]}]}}
        responses, traces = agent.forward_with_trace(observation)
        assert json.loads(responses[0]["action"].removeprefix("<tool_call>").removesuffix("</tool_call>")) == action
        assert len(received) == 2
        assert all(path == "/openapi/chat/completions" for path, body in received)
        assert all(body["tool_choice"] == "required" and "max_completion_tokens" in body for path, body in received)
        assert {tool["function"]["name"] for tool in received[0][1]["tools"]} >= {"list_files", "submit_action"}
        assert any(message["role"] == "tool" for message in received[1][1]["messages"])
        assert any(part.get("type") == "image_url" for message in received[0][1]["messages"]
                   if isinstance(message.get("content"), list) for part in message["content"])
        assert traces[0]["simple_agent"]["native_action_submission"] is True
    finally:
        if agent is not None:
            agent.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
