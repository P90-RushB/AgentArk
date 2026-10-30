"""Offline contracts for new evaluation recipes and generated Kaggle tasks."""
import ast
import importlib.util
import json
import os
import textwrap
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import yaml

from agent_ark.agent.agent_registry import configure_agent_environment, resolve_agent_type
from agent_ark.agent.simple_agent import SimpleRuntimeConfig


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "benchmarks" / "kaggle"


def generator():
    spec = importlib.util.spec_from_file_location("kaggle_generator", KAGGLE / "generate_agentark_kaggle_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename", [
    "eval_seed1.example.yaml", "eval_seeds_1_n.example.yaml",
    "parallel_api_eval.example.yaml", "simple_agent.example.yaml",
])
def test_new_evaluation_examples_enable_simple_native_tools(filename):
    cfg = yaml.safe_load((ROOT / "config" / "ark_env" / filename).read_text(encoding="utf-8"))
    for model in cfg["models"]:
        assert resolve_agent_type(model).name == "simple"
        assert model["simple_agent"]["tools"]["enabled"] is True
        assert model["simple_agent"]["tools"]["workspace_parent"].startswith("tmp/")
        budget = SimpleRuntimeConfig(**model["simple_agent"].get("context", {}))
        assert budget.output_reserve_tokens >= model["max_completion_tokens"]
        effective = configure_agent_environment(cfg["env_cfg"], model)
        assert effective["env_config_overrides"]["env_wrapper_cfg"]["context_manager"]["messages"]["return_mode"] == "delta"


def test_generated_task_uses_generic_simple_template_and_immutable_pin(tmp_path):
    module = generator()
    output = tmp_path / "robot_painting.py"
    pin = "a" * 40
    module.generate(task_name="RobotPainting", seed_start=2, seed_end=4, env_id=7,
                    output=output, wrapper_commit=pin)
    source = output.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert f'AGENTARK_COMMIT = "{pin}"' in source
    assert '"RobotPainting"' in source
    assert 'DEFAULT_SEED_START = 2' in source and 'DEFAULT_SEED_END = 4' in source
    assert 'DEFAULT_ENV_ID = 7' in source
    assert 'def agentark_robotpainting(llm)' in source
    assert 'agentark_robotpainting.run(kbench.llm)' in source
    assert '"agentark-robotpainting"' in source
    assert 'agent_type: simple' in source
    assert 'ENV_VERSION = "1.0.6"' in source
    assert 'HF_REVISION = "6a16c98e262abd760b09ef007a49118d0b1b0cb9"' in source
    assert 'ENV_ZIP_SHA256 = "f32061282f51bbc2ab36a4617d7af5eb6b4d349a944dd861171aba95c00ccdcc"' in source

    # Execute only the real config author, with no Kaggle import, downloads,
    # subprocesses, environment creation or model calls.
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_write_eval_config")
    namespace = {"Path": Path, "os": os, "json": json, "textwrap": textwrap, "Optional": __import__("typing").Optional,
                 "AGENTARK_ROOT": tmp_path, "RESULT_PATH": tmp_path / "tmp" / "simple.jsonl",
                 "ENV_ROOT": tmp_path / "env", "MOD_PATH": tmp_path / "Mods", "CONFIG_PATH": tmp_path / "eval.yaml"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(output), "exec"), namespace)
    namespace["_write_eval_config"](task_name="RobotPainting", seed_start=2, seed_end=4, env_id=7,
                                   model="selected-model", base_url="https://proxy.example/genai",
                                   max_completion_tokens=30000)
    cfg = yaml.safe_load(namespace["CONFIG_PATH"].read_text(encoding="utf-8"))
    model = cfg["models"][0]
    assert model["agent_type"] == "simple"
    assert model["base_url"] == "https://proxy.example/openapi"
    assert model["simple_agent"]["tools"]["enabled"] is True
    assert model["simple_agent"]["context"]["output_reserve_tokens"] == 30000
    assert cfg["eval"]["group_seeds"] == {"start": 2, "end": 4}


@pytest.mark.parametrize("pin", ["main", "54ab7cb", "a" * 39, "g" * 40])
def test_generator_rejects_moving_or_invalid_pin(tmp_path, pin):
    with pytest.raises(ValueError, match="immutable"):
        generator().generate(task_name="RobotPainting", seed_start=1, seed_end=2, env_id=0,
                             output=tmp_path / "out.py", wrapper_commit=pin)


def test_generator_protects_existing_benchmark_and_validates_identifiers(tmp_path):
    output = tmp_path / "existing.py"
    output.write_text("historical benchmark", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generator().generate(task_name="RobotPainting", seed_start=1, seed_end=2, env_id=0, output=output)
    assert output.read_text(encoding="utf-8") == "historical benchmark"
    with pytest.raises(ValueError, match="identifier"):
        generator().generate(task_name='bad"name', seed_start=1, seed_end=2, env_id=0,
                             output=tmp_path / "out.py")


def test_kaggle_notebook_and_generic_template_stay_in_sync():
    assert generator().DEFAULT_WRAPPER_COMMIT == "a76159828fca187e4cacb98dc0f934e803d0f67f"
    notebook = json.loads((KAGGLE / "new-benchmark-task-f8fdd.ipynb").read_text(encoding="utf-8"))
    expected = "\n\n".join("# %%\n" + "".join(cell["source"]).rstrip()
                              for cell in notebook["cells"] if cell["cell_type"] == "code") + "\n"
    # The generator keeps its historical raw mean metric, while the notebook
    # keeps its pre-existing normalized metric. All other source stays aligned.
    expected = "# Generated-task template: preserve the generator's raw mean score; the companion notebook uses normalized scores.\n" + expected
    expected = expected.replace('return float(summary["normalized_mean_score_reward"])', 'return float(summary["mean_score_reward"])')
    expected = expected.replace('"metric": "normalized_mean_score_reward"', '"metric": "mean_score_reward"')
    expected = expected.replace('Mean of per-seed AgentArk score_reward normalized from the configured task score min/max range to 0-1.',
                                'Mean of per-seed AgentArk final-attempt score_reward; normalized values are diagnostics only.')
    expected = expected.replace('Formal benchmark runs normalize each AgentArk score_reward from the configured task score min/max range to 0-1, then average across seeds.',
                                'Formal benchmark runs average raw per-seed AgentArk final-attempt score_reward; normalized values are diagnostics only.')
    assert 'AGENTARK_COMMIT = "a76159828fca187e4cacb98dc0f934e803d0f67f"' in expected
    assert (KAGGLE / "agentark_simple_task_template.py").read_text(encoding="utf-8") == expected
    ast.parse(expected)


@pytest.mark.parametrize("notebook, expected_score", [(False, 20.0), (True, 0.2)])
def test_kaggle_entry_points_preserve_their_score_metrics(tmp_path, monkeypatch, notebook, expected_score):
    if notebook:
        data = json.loads((KAGGLE / "new-benchmark-task-f8fdd.ipynb").read_text(encoding="utf-8"))
        source = "\n\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")
    else:
        source = (KAGGLE / "agentark_simple_task_template.py").read_text(encoding="utf-8")
    names = {"_dedupe_results_by_seed", "_score_value", "_result_seed", "_results_summary", "agentark_task_seeds_1_10"}
    functions = [deepcopy(node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    rows = [{"status": "ok", "requested_group_seed": index, "score_reward": score}
            for index, score in [(1, 10.0), (2, 30.0)]]
    for key in ("TASK_NAME", "TASK_SCORE_SCALE_MIN", "TASK_SCORE_SCALE_MAX", "SEED_START", "SEED_END", "EVAL_ATTEMPTS", "MAX_COMPLETION_TOKENS", "ENV_ID"):
        monkeypatch.delenv("AGENTARK_KAGGLE_" + key, raising=False)
    summaries = []
    def assert_in(expected, actual, **kwargs):
        assert expected in actual
    namespace = {"Any": Any, "Dict": Dict, "List": List, "os": os, "time": __import__("time"),
                 "AGENTARK_COMMIT": "a" * 40, "ENV_VERSION": "1.0.6", "HF_REVISION": "b" * 40,
                 "TASK_ARTIFACT": {}, "RESULT_PATH": tmp_path / "results.jsonl", "EXPORTED_AGENTARK_LOG_PATH": tmp_path / "agent.log",
                 "EXPORTED_SUMMARY_PATH": tmp_path / "summary.json", "DEFAULT_TASK_NAME": "Task", "DEFAULT_TASK_SCORE_SCALE_MIN": 0,
                 "DEFAULT_TASK_SCORE_SCALE_MAX": 100, "DEFAULT_SEED_START": 1, "DEFAULT_SEED_END": 2,
                 "DEFAULT_EVAL_ATTEMPTS": 3, "DEFAULT_MAX_COMPLETION_TOKENS": 30000, "DEFAULT_ENV_ID": 0,
                 "_should_run_full_eval": lambda: True, "_require_model_proxy_env": lambda: ("https://proxy.example", "unused", "model"),
                 "_install_and_configure_agentark": lambda: {}, "_download_task": lambda task: task,
                 "_run_agentark_eval": lambda *args, **kwargs: rows, "_export_result_jsonl": lambda rows: "results.jsonl",
                 "_export_summary_json": summaries.append, "kbench": SimpleNamespace(assertions=SimpleNamespace(assert_in=assert_in))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "kaggle_scoring", "exec"), namespace)
    assert namespace["agentark_task_seeds_1_10"](None) == expected_score
    assert summaries[-1]["mean_score_reward"] == 20.0
    assert summaries[-1]["normalized_mean_score_reward"] == 0.2


def test_colab_model_list_explicitly_keeps_simple_tools():
    notebook = json.loads((ROOT / "docs/tutorials/03_online_evaluation_tutorial.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "conf.models = [model_config]" in "".join(cell["source"]))
    ast.parse(source)
    assert '"agent_type": "simple"' in source
    assert '"tools": {"enabled": True' in source
