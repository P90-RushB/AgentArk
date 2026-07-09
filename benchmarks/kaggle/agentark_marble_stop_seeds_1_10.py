# %%
"""Kaggle Benchmark task for AgentArk MarbleStop seeds 1-10.

This file is intentionally self-contained because Kaggle Benchmark tasks run as
notebooks on Kaggle infrastructure. It installs a pinned AgentArk checkout,
downloads the public Unity runtime, evaluates the Kaggle-selected model through
the Kaggle Model Proxy, and reports AgentArk's final-attempt score metric.
"""

# %%
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import kaggle_benchmarks as kbench


AGENTARK_REPO_URL = "https://github.com/P90-RushB/AgentArk.git"
AGENTARK_COMMIT = "31a97e7800861e9c1d5b382f323b64392328d042"
ENV_ZIP_URL = (
    "https://huggingface.co/datasets/P90-RushB/AgentArk/resolve/main/"
    "artifacts/envs/1.0.1/linux/AgentArk-env-1.0.1-linux.zip?download=true"
)

WORK_ROOT = Path(os.getenv("AGENTARK_KAGGLE_WORK_ROOT", "/tmp/agentark_kaggle_work")).resolve()
AGENTARK_ROOT = WORK_ROOT / "AgentArk"
VENV_DIR = WORK_ROOT / "agentark_py310"
VENV_PY = VENV_DIR / "bin" / "python"
ENV_ZIP = WORK_ROOT / "AgentArk-env-1.0.1-linux.zip"
ENV_ROOT = WORK_ROOT / "AgentArk-env-1.0.1-linux"
MOD_PATH = ENV_ROOT / "AgentArk_Data" / "Resources" / "Mods"
TASK_STORE_PATH = MOD_PATH / "all_tasks"
RUNTIME_POOL_ROOT = Path("/tmp/agentark_runtime_pool")
CONFIG_PATH = WORK_ROOT / "agentark_kaggle_eval.yaml"
RESULT_PATH = AGENTARK_ROOT / "tmp" / "kaggle_agentark_marble_stop_seeds_1_10.jsonl"
EXPORTED_RESULT_PATH = Path.cwd() / "agentark_marble_stop_seeds_1_10_results.jsonl"
EXPORTED_SUMMARY_PATH = Path.cwd() / "agentark_marble_stop_seeds_1_10_summary.json"
EXPORTED_AGENTARK_LOG_PATH = Path.cwd() / "agentark_marble_stop_seeds_1_10_agentark.log"

DEFAULT_TASK_NAME = "MarbleStop"
DEFAULT_SEED_START = 1
DEFAULT_SEED_END = 10
DEFAULT_EVAL_ATTEMPTS = 100
DEFAULT_ENV_ID = 0
DEFAULT_PYTHON_VERSION = "3.10.12"


# %%
def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    candidates = [Path.cwd() / ".env"]
    source_file = globals().get("__file__")
    if source_file:
        candidates.extend(parent / ".env" for parent in Path(source_file).resolve().parents)
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            load_dotenv(path, override=False)


def _run(
    cmd: List[Union[str, Path]],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    printable = " ".join(str(part) for part in cmd)
    print("+", printable, flush=True)
    subprocess.run([str(part) for part in cmd], cwd=str(cwd) if cwd else None, env=env, check=True)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _ensure_os_packages() -> None:
    if _bool_env("AGENTARK_KAGGLE_SKIP_APT", default=False):
        return
    missing = [name for name in ("git", "xvfb") if shutil.which(name) is None]
    if not missing:
        return
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    _run(["apt-get", "update", "-qq"], env=env)
    _run(["apt-get", "install", "-y", "-qq", *missing], env=env)


def _python310_command() -> Optional[str]:
    explicit = os.getenv("AGENTARK_KAGGLE_PYTHON_BIN", "").strip()
    if explicit:
        return explicit
    for name in ("python3.10", "python3"):
        found = shutil.which(name)
        if not found:
            continue
        version = subprocess.check_output([found, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True).strip()
        if version == "3.10":
            probe = subprocess.run(
                [found, "-m", "venv", "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if probe.returncode != 0:
                continue
            return found
    return None


def _ensure_uv() -> None:
    probe = subprocess.run(
        [sys.executable, "-m", "uv", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return
    _run([sys.executable, "-m", "pip", "install", "-q", "uv"])


def _ensure_venv_pip() -> None:
    probe = subprocess.run(
        [VENV_PY, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return
    _ensure_uv()
    _run([sys.executable, "-m", "uv", "pip", "install", "--python", VENV_PY, "pip"])


def _ensure_agentark_checkout() -> None:
    if not AGENTARK_ROOT.exists():
        _run(["git", "clone", AGENTARK_REPO_URL, AGENTARK_ROOT])
    else:
        _run(["git", "fetch", "origin"], cwd=AGENTARK_ROOT)
    _run(["git", "checkout", "--detach", AGENTARK_COMMIT], cwd=AGENTARK_ROOT)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(AGENTARK_ROOT), text=True).strip()
    if current != AGENTARK_COMMIT:
        raise RuntimeError(f"AgentArk checkout mismatch: expected {AGENTARK_COMMIT}, got {current}")


def _ensure_python_env() -> None:
    if not VENV_PY.exists():
        python_cmd = _python310_command()
        if python_cmd:
            _run([python_cmd, "-m", "venv", VENV_DIR])
        else:
            python_version = os.getenv("AGENTARK_KAGGLE_PYTHON_VERSION", DEFAULT_PYTHON_VERSION)
            _ensure_uv()
            _run([sys.executable, "-m", "uv", "python", "install", python_version])
            _run([sys.executable, "-m", "uv", "venv", "--seed", "--python", python_version, VENV_DIR])
    _ensure_venv_pip()
    _run([VENV_PY, "-m", "pip", "install", "-q", "-e", AGENTARK_ROOT])


def _download_runtime() -> None:
    if (ENV_ROOT / "AgentArk.x86_64").exists():
        return
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    if ENV_ZIP.exists():
        try:
            with zipfile.ZipFile(ENV_ZIP) as zf:
                bad_member = zf.testzip()
                if bad_member is None:
                    print(f"Using existing AgentArk runtime zip: {ENV_ZIP}", flush=True)
                else:
                    raise RuntimeError(f"Corrupt runtime zip member: {bad_member}")
            with zipfile.ZipFile(ENV_ZIP) as zf:
                zf.extractall(WORK_ROOT)
            exe = ENV_ROOT / "AgentArk.x86_64"
            if not exe.exists():
                raise FileNotFoundError(f"Expected Unity executable at {exe}")
            exe.chmod(0o755)
            return
        except Exception:
            ENV_ZIP.unlink()

    for attempt in range(1, 4):
        try:
            print(f"Downloading AgentArk runtime to {ENV_ZIP} (attempt {attempt}/3)", flush=True)
            if ENV_ZIP.exists():
                ENV_ZIP.unlink()
            urllib.request.urlretrieve(ENV_ZIP_URL, ENV_ZIP)
            with zipfile.ZipFile(ENV_ZIP) as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    raise RuntimeError(f"Corrupt runtime zip member: {bad_member}")
            break
        except Exception:
            if ENV_ZIP.exists():
                ENV_ZIP.unlink()
            if attempt >= 3:
                raise
            time.sleep(5 * attempt)
    with zipfile.ZipFile(ENV_ZIP) as zf:
        zf.extractall(WORK_ROOT)
    exe = ENV_ROOT / "AgentArk.x86_64"
    if not exe.exists():
        raise FileNotFoundError(f"Expected Unity executable at {exe}")
    exe.chmod(0o755)


def _runtime_env() -> Dict[str, str]:
    values = {
        "AGENTARK_ENV_PATH": str(ENV_ROOT / "AgentArk.x86_64"),
        "AGENTARK_MOD_PATH": str(MOD_PATH),
        "AGENTARK_TASK_STORE_PATH": str(TASK_STORE_PATH),
        "AGENTARK_RUNTIME_TEMPLATE_ROOT": str(ENV_ROOT),
        "AGENTARK_RUNTIME_POOL_ROOT": str(RUNTIME_POOL_ROOT),
        "MLAGENTS_PYTHON_BIN": str(VENV_PY),
    }
    env = os.environ.copy()
    env.update(values)
    env.setdefault("GALLIUM_NUM_THREADS", "16")
    (AGENTARK_ROOT / ".env").write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    return env


def _enable_virtual_display(env: Dict[str, str]) -> None:
    mod_config = MOD_PATH / "config.yaml"
    if not mod_config.exists():
        print(f"Warning: mod config not found: {mod_config}", flush=True)
        return
    script = (
        "from omegaconf import OmegaConf\n"
        f"path = {str(mod_config)!r}\n"
        "conf = OmegaConf.load(path)\n"
        "conf.virtual_display = True\n"
        "OmegaConf.save(config=conf, f=path)\n"
    )
    _run([VENV_PY, "-c", script], env=env)


def _require_model_proxy_env() -> Tuple[str, str, str]:
    _load_dotenv_if_available()
    base_url = os.getenv("MODEL_PROXY_URL", "").strip()
    api_key = os.getenv("MODEL_PROXY_API_KEY", "").strip()
    model = (
        os.getenv("AGENTARK_KAGGLE_MODEL")
        or os.getenv("LLM_DEFAULT")
        or os.getenv("LLM_DEFAULT_EVAL")
        or ""
    ).strip()
    if not base_url:
        raise RuntimeError("MODEL_PROXY_URL is not set. Run `kaggle b init -y` or execute through Kaggle Benchmarks.")
    if not api_key:
        raise RuntimeError("MODEL_PROXY_API_KEY is not set. Run `kaggle b auth -y` if local credentials expired.")
    if not model:
        raise RuntimeError("LLM_DEFAULT is not set and AGENTARK_KAGGLE_MODEL was not provided.")
    return base_url, api_key, model


def _write_eval_config(
    *,
    task_name: str,
    seed_start: int,
    seed_end: int,
    env_id: int,
    model: str,
    base_url: str,
) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    openai_base_url = base_url.rstrip("/")
    if openai_base_url.endswith("/genai"):
        openai_base_url = openai_base_url[: -len("/genai")]
    if not openai_base_url.endswith("/openapi"):
        openai_base_url = openai_base_url + "/openapi"
    config = f"""
env_cfg:
  env_path: "{ENV_ROOT / "AgentArk.x86_64"}"
  mod_path: "{MOD_PATH}"
  task_type: RLTask
  env_config_overrides:
    override_by_task: true
    num_parallel_envs: 1
    virtual_display: true
    virtual_display_render_env:
      GALLIUM_NUM_THREADS: 16

eval:
  output_path: "{RESULT_PATH.relative_to(AGENTARK_ROOT)}"
  stop_on_error: false
  skip_existing_results: true
  fixed_env_id: {int(env_id)}
  task_names:
    - "{task_name}"
  group_seeds:
    start: {int(seed_start)}
    end: {int(seed_end)}

hooks:
  visualization:
    enabled: false
    host: 127.0.0.1
    port: 18181
    open_browser: false
    keep_open_on_end: false
  human_interaction:
    enabled: false
    name: human-local
    timeout_s: null

models:
  - name: "kaggle-model-proxy"
    provider: openai
    model: "{model}"
    base_url: "{openai_base_url}"
    api_key_env: MODEL_PROXY_API_KEY
    temperature: 0.0
    timeout_s: {float(os.getenv("AGENTARK_KAGGLE_REQUEST_TIMEOUT_S", "600"))}
    max_retries: 2
"""
    CONFIG_PATH.write_text(textwrap.dedent(config).strip() + "\n", encoding="utf-8")


def _read_results() -> List[Dict[str, Any]]:
    if not RESULT_PATH.exists():
        return []
    lines = [line for line in RESULT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: List[Dict[str, Any]] = []
    for index, line in enumerate(lines):
        result = json.loads(line)
        if not isinstance(result, dict):
            raise RuntimeError(f"AgentArk result line {index + 1} is not a JSON object")
        results.append(result)
    return results


def _is_notebook_progress_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    prefixes = (
        "Loaded ",
        "Resuming from ",
        "Results will be written",
        "[case ",
        "[model ",
        "=== Evaluation Summary",
        "model=",
    )
    return stripped.startswith(prefixes)


def _print_filtered_agentark_output(output: str) -> None:
    for line in output.splitlines():
        if _is_notebook_progress_line(line):
            print(line, flush=True)


def _expected_seeds(seed_start: int, seed_end: int) -> List[int]:
    step = 1 if int(seed_start) <= int(seed_end) else -1
    return list(range(int(seed_start), int(seed_end) + step, step))


def _ok_seeds(results: List[Dict[str, Any]]) -> set[int]:
    return {_result_seed(result) for result in results if result.get("status") == "ok"}


def _missing_seed_records(
    results: List[Dict[str, Any]],
    *,
    task_name: str,
    seed_start: int,
    seed_end: int,
    env_id: int,
) -> List[Dict[str, Any]]:
    present = {_result_seed(result) for result in results}
    records: List[Dict[str, Any]] = []
    for seed in _expected_seeds(seed_start, seed_end):
        if seed in present:
            continue
        records.append(
            {
                "status": "error",
                "case_id": f"{task_name}-seed-{seed:04d}",
                "requested_task_name": task_name,
                "requested_group_seed": seed,
                "requested_env_id": int(env_id),
                "error_type": "MissingResult",
                "error": "AgentArk did not produce a result record for this seed after retry attempts.",
            }
        )
    return records


def _install_and_configure_agentark() -> Dict[str, str]:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_os_packages()
    _ensure_agentark_checkout()
    _ensure_python_env()
    _download_runtime()
    env = _runtime_env()
    _enable_virtual_display(env)
    return env


def _run_agentark_eval(
    env: Dict[str, str],
    *,
    task_name: str,
    seed_start: int,
    seed_end: int,
    env_id: int,
    max_attempts: int,
) -> List[Dict[str, Any]]:
    if RESULT_PATH.exists():
        RESULT_PATH.unlink()
    expected = set(_expected_seeds(seed_start, seed_end))
    last_returncode = 0
    EXPORTED_AGENTARK_LOG_PATH.write_text("", encoding="utf-8")
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        print(f"AgentArk evaluation attempt {attempt}/{max_attempts}", flush=True)
        proc = subprocess.run(
            [str(VENV_PY), "-m", "agent_ark.ark_eval.run_api_agent", "--config", str(CONFIG_PATH)],
            cwd=str(AGENTARK_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = proc.stdout or ""
        with EXPORTED_AGENTARK_LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(f"\n===== AgentArk evaluation attempt {attempt}/{max_attempts} =====\n")
            log.write(output)
            if output and not output.endswith("\n"):
                log.write("\n")
        _print_filtered_agentark_output(output)
        last_returncode = int(proc.returncode)
        results = _read_results()
        ok = _ok_seeds(results)
        missing_ok = sorted(expected - ok)
        print(
            f"AgentArk attempt {attempt}/{max_attempts}: ok={len(ok)}/{len(expected)} "
            f"pending_or_error_seeds={missing_ok}",
            flush=True,
        )
        if expected.issubset(ok):
            return results
        if last_returncode != 0 and not results:
            break
    results = _read_results()
    if last_returncode != 0 and not results:
        raise RuntimeError(f"AgentArk evaluation failed before writing any result records; exit={last_returncode}")
    return results + _missing_seed_records(
        results,
        task_name=task_name,
        seed_start=seed_start,
        seed_end=seed_end,
        env_id=env_id,
    )


def _dedupe_results_by_seed(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_seed: Dict[int, Dict[str, Any]] = {}
    for result in results:
        seed = _result_seed(result)
        previous = by_seed.get(seed)
        if previous is None or previous.get("status") != "ok" or result.get("status") == "ok":
            by_seed[seed] = result
    return [by_seed[seed] for seed in sorted(by_seed)]


def _score_value(result: Dict[str, Any]) -> float:
    if result.get("status") != "ok":
        return 0.0
    return float(result.get("score_reward", result.get("last_attempt_reward", 0.0)) or 0.0)


def _result_seed(result: Dict[str, Any]) -> int:
    return int(result.get("actual_rollout_group_seed", result.get("requested_group_seed", 0)) or 0)


def _results_summary(
    results: List[Dict[str, Any]],
    *,
    seed_start: int,
    seed_end: int,
) -> Dict[str, Any]:
    expected_count = abs(int(seed_end) - int(seed_start)) + 1
    results = _dedupe_results_by_seed(results)
    if len(results) != expected_count:
        raise AssertionError(f"Expected {expected_count} AgentArk result records, got {len(results)}")

    sorted_results = sorted(results, key=_result_seed)
    scores = [_score_value(result) for result in sorted_results]
    score_sum = sum(scores)
    mean_score = score_sum / len(scores) if scores else 0.0
    per_seed = [
        {
            "seed": _result_seed(result),
            "case_id": result.get("case_id"),
            "status": result.get("status"),
            "score_reward": _score_value(result),
            "last_attempt_reward": float(result.get("last_attempt_reward", _score_value(result)) or 0.0),
            "best_attempt_reward": float(result.get("best_attempt_reward", 0.0) or 0.0),
            "rollout_success": bool(result.get("rollout_success", False)),
            "ever_attempt_success": bool(result.get("ever_attempt_success", False)),
            "turns": int(result.get("turns", 0) or 0),
            "attempt_count": int(result.get("attempt_count", 0) or 0),
            "error_type": result.get("error_type"),
            "error": result.get("error"),
        }
        for result in sorted_results
    ]

    first = sorted_results[0]
    return {
        "agentark_commit": AGENTARK_COMMIT,
        "evaluation_status": "complete",
        "task_name": first.get("actual_task_name", first.get("requested_task_name")),
        "seed_start": int(seed_start),
        "seed_end": int(seed_end),
        "seed_count": len(sorted_results),
        "env_id": first.get("actual_env_id", first.get("requested_env_id")),
        "model": first.get("model"),
        "metric": "mean_score_reward",
        "metric_definition": "Mean of AgentArk score_reward across seeds; score_reward is the final attempt reward.",
        "mean_score_reward": mean_score,
        "score_reward_sum": score_sum,
        "min_score_reward": min(scores) if scores else 0.0,
        "max_score_reward": max(scores) if scores else 0.0,
        "error_count": sum(1 for result in sorted_results if result.get("status") != "ok"),
        "final_success_count": sum(1 for result in sorted_results if result.get("rollout_success")),
        "ever_success_count": sum(1 for result in sorted_results if result.get("ever_attempt_success", False)),
        "total_turns": sum(int(result.get("turns", 0) or 0) for result in sorted_results),
        "max_attempts": first.get("max_attempts"),
        "max_steps_per_attempt": first.get("max_steps_per_attempt"),
        "per_seed": per_seed,
        "result_path": str(RESULT_PATH),
        "agentark_log_path": str(EXPORTED_AGENTARK_LOG_PATH),
    }


def _export_result_jsonl(results: List[Dict[str, Any]]) -> str:
    EXPORTED_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_results = _dedupe_results_by_seed(results)
    with EXPORTED_RESULT_PATH.open("w", encoding="utf-8") as f:
        for result in final_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return str(EXPORTED_RESULT_PATH)


def _export_summary_json(summary: Dict[str, Any]) -> str:
    EXPORTED_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORTED_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(EXPORTED_SUMMARY_PATH)


def _should_run_full_eval() -> bool:
    explicit = os.getenv("AGENTARK_KAGGLE_FULL_EVAL")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "y", "on"}
    return os.getenv("BENCHMARK_MODE", "").strip().upper() in {"RUN", "TRIAL"}


def _creation_smoke_summary() -> Dict[str, Any]:
    return {
        "agentark_commit": AGENTARK_COMMIT,
        "evaluation_status": "creation_smoke",
        "task_name": DEFAULT_TASK_NAME,
        "seed_start": DEFAULT_SEED_START,
        "seed_end": DEFAULT_SEED_END,
        "seed_count": DEFAULT_SEED_END - DEFAULT_SEED_START + 1,
        "metric": "mean_score_reward",
        "metric_definition": "Formal benchmark runs compute mean AgentArk score_reward across seeds.",
        "mean_score_reward": 0.0,
        "score_reward_sum": 0.0,
        "error_count": 0,
        "max_attempts": DEFAULT_EVAL_ATTEMPTS,
        "full_eval": False,
        "benchmark_mode": os.getenv("BENCHMARK_MODE", ""),
    }


# %%
@kbench.task(name="agentark-marble-stop-seeds-1-10")
def agentark_marble_stop_seeds_1_10(llm) -> float:
    """Evaluate the Kaggle-selected model on AgentArk MarbleStop seeds 1-10."""
    del llm  # The AgentArk subprocess consumes the same Kaggle Model Proxy env.
    started = time.time()

    if not _should_run_full_eval():
        summary = _creation_smoke_summary()
        summary["wall_time_s"] = round(time.time() - started, 3)
        summary["exported_summary_path"] = str(EXPORTED_SUMMARY_PATH)
        _export_summary_json(summary)
        kbench.assertions.assert_in(
            "creation_smoke",
            summary["evaluation_status"],
            expectation="Kaggle task creation should produce a lightweight run output.",
        )
        return float(summary["mean_score_reward"])

    task_name = os.getenv("AGENTARK_KAGGLE_TASK_NAME", DEFAULT_TASK_NAME).strip() or DEFAULT_TASK_NAME
    seed_start = int(os.getenv("AGENTARK_KAGGLE_SEED_START", str(DEFAULT_SEED_START)))
    seed_end = int(os.getenv("AGENTARK_KAGGLE_SEED_END", str(DEFAULT_SEED_END)))
    max_attempts = int(os.getenv("AGENTARK_KAGGLE_EVAL_ATTEMPTS", str(DEFAULT_EVAL_ATTEMPTS)))
    env_id = int(os.getenv("AGENTARK_KAGGLE_ENV_ID", str(DEFAULT_ENV_ID)))
    base_url, api_key, model = _require_model_proxy_env()

    env = _install_and_configure_agentark()
    env["MODEL_PROXY_API_KEY"] = api_key
    _write_eval_config(
        task_name=task_name,
        seed_start=seed_start,
        seed_end=seed_end,
        env_id=env_id,
        model=model,
        base_url=base_url,
    )
    results = _run_agentark_eval(
        env,
        task_name=task_name,
        seed_start=seed_start,
        seed_end=seed_end,
        env_id=env_id,
        max_attempts=max_attempts,
    )
    summary = _results_summary(results, seed_start=seed_start, seed_end=seed_end)
    summary["exported_result_path"] = _export_result_jsonl(results)
    summary["wall_time_s"] = round(time.time() - started, 3)
    summary["exported_summary_path"] = str(EXPORTED_SUMMARY_PATH)
    _export_summary_json(summary)

    kbench.assertions.assert_in(
        "complete",
        summary["evaluation_status"],
        expectation="AgentArk should produce a complete score summary for every requested seed.",
    )
    return float(summary["mean_score_reward"])


agentark_marble_stop_seeds_1_10.run(kbench.llm)
