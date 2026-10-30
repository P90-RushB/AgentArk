# Generated-task template: preserve the generator's raw mean score; the companion notebook uses normalized scores.
# %%
"""Kaggle Benchmark task for AgentArk tasks over a configured seed range.

This file is intentionally self-contained because Kaggle Benchmark tasks run as
notebooks on Kaggle infrastructure. It installs a pinned AgentArk checkout,
downloads the public Unity runtime, evaluates the Kaggle-selected model through
the Kaggle Model Proxy, and reports AgentArk's final-attempt score metric.
"""

# %%
import hashlib
import json
import os
import re
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
AGENTARK_COMMIT = "71a9cc305e3e87c0c5ed416086586e24db35210b"
ENV_VERSION = "1.0.6"
HF_REVISION = "6a16c98e262abd760b09ef007a49118d0b1b0cb9"
HF_BASE_URL = f"https://huggingface.co/datasets/P90-RushB/AgentArk/resolve/{HF_REVISION}"
ENV_ZIP_URL = f"{HF_BASE_URL}/artifacts/envs/{ENV_VERSION}/linux/AgentArk-env-{ENV_VERSION}-linux.zip"
ENV_ZIP_SHA256 = "f32061282f51bbc2ab36a4617d7af5eb6b4d349a944dd861171aba95c00ccdcc"
TASK_ARTIFACT = None


def _task_slug(task_name: str) -> str:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[0-9])(?=[A-Z][a-z])", "_", task_name.strip())
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", spaced).strip("_").lower()
    return normalized or "task"


# Edit the task name when importing this notebook to create another benchmark.
DEFAULT_TASK_NAME = os.getenv("AGENTARK_KAGGLE_TASK_NAME", "MarbleStop").strip() or "MarbleStop"
DEFAULT_TASK_SLUG = os.getenv("AGENTARK_KAGGLE_TASK_SLUG", _task_slug(DEFAULT_TASK_NAME)).strip() or _task_slug(DEFAULT_TASK_NAME)
# Set the task's actual score range: MarbleStop 0..1; Snake 0..6.
# The notebook leaderboard normalizes with this range; the Python generator keeps raw scores.
DEFAULT_TASK_SCORE_SCALE_MIN = float(os.getenv("AGENTARK_KAGGLE_TASK_SCORE_SCALE_MIN", "0.0"))
DEFAULT_TASK_SCORE_SCALE_MAX = float(os.getenv("AGENTARK_KAGGLE_TASK_SCORE_SCALE_MAX", "1.0"))
# For a quick real test, use seeds 1..2; restore the intended range before publishing.
DEFAULT_SEED_START = 1
DEFAULT_SEED_END = 10
DEFAULT_EVAL_ATTEMPTS = 3
DEFAULT_MAX_COMPLETION_TOKENS = 30000
DEFAULT_ENV_ID = 0
DEFAULT_PYTHON_VERSION = "3.10.12"
# Use a new benchmark name for new evaluation conditions, preserving existing tasks.
DEFAULT_BENCHMARK_TASK_NAME_FALLBACK = f"agentark-{DEFAULT_TASK_SLUG.replace('_', '-')}"
DEFAULT_BENCHMARK_TASK_NAME = os.getenv("AGENTARK_KAGGLE_BENCHMARK_NAME", DEFAULT_BENCHMARK_TASK_NAME_FALLBACK).strip() or DEFAULT_BENCHMARK_TASK_NAME_FALLBACK

WORK_ROOT = Path(os.getenv("AGENTARK_KAGGLE_WORK_ROOT", "/tmp/agentark_kaggle_work")).resolve()
AGENTARK_ROOT = WORK_ROOT / "AgentArk"
VENV_DIR = WORK_ROOT / "agentark_py310"
VENV_PY = VENV_DIR / "bin" / "python"
ENV_ZIP = WORK_ROOT / f"AgentArk-env-{ENV_VERSION}-linux.zip"
ENV_ROOT = WORK_ROOT / f"AgentArk-env-{ENV_VERSION}-linux"
MOD_PATH = ENV_ROOT / "AgentArk_Data" / "Resources" / "Mods"
TASK_STORE_PATH = MOD_PATH / "all_tasks"
RUNTIME_POOL_ROOT = WORK_ROOT / f"runtime_pool_{ENV_VERSION}"
CONFIG_PATH = WORK_ROOT / "agentark_kaggle_eval.yaml"
RESULT_STEM = f"agentark_{DEFAULT_TASK_SLUG}_simple_seeds_{DEFAULT_SEED_START}_{DEFAULT_SEED_END}"
RESULT_PATH = AGENTARK_ROOT / "tmp" / f"kaggle_{RESULT_STEM}.jsonl"
EXPORTED_RESULT_PATH = Path.cwd() / f"{RESULT_STEM}_results.jsonl"
EXPORTED_SUMMARY_PATH = Path.cwd() / f"{RESULT_STEM}_summary.json"
EXPORTED_AGENTARK_LOG_PATH = Path.cwd() / f"{RESULT_STEM}_agentark.log"

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
    missing = [package for package, executable in (("git", "git"), ("xvfb", "Xvfb"))
               if shutil.which(executable) is None]
    if not missing:
        return
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    _run(["apt-get", "update", "-qq"], env=env)
    _run(["apt-get", "install", "-y", "-qq", *missing], env=env)


def _python_version(command: Union[str, Path]) -> Tuple[int, ...]:
    try:
        text = subprocess.check_output(
            [str(command), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        version = tuple(int(part) for part in text.split("."))
        if len(version) != 3:
            raise ValueError("expected a three-part Python version")
        return version
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f"Cannot determine Python version for {command}") from exc


def _python310_command() -> Optional[str]:
    explicit = os.getenv("AGENTARK_KAGGLE_PYTHON_BIN", "").strip()
    if explicit:
        version = _python_version(explicit)
        if not (3, 10, 1) <= version < (3, 10, 13):
            raise RuntimeError(
                "AGENTARK_KAGGLE_PYTHON_BIN must select Python >=3.10.1,<3.10.13; "
                f"got {'.'.join(map(str, version))}. Use Python {DEFAULT_PYTHON_VERSION}."
            )
        return explicit
    for name in ("python3.10", "python3"):
        found = shutil.which(name)
        if not found:
            continue
        try:
            version = _python_version(found)
        except RuntimeError:
            continue
        if (3, 10, 1) <= version < (3, 10, 13):
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
            python_version = os.getenv("AGENTARK_KAGGLE_PYTHON_VERSION", DEFAULT_PYTHON_VERSION).strip()
            try:
                requested_version = tuple(int(part) for part in python_version.split("."))
            except ValueError:
                requested_version = ()
            if len(requested_version) != 3 or not (3, 10, 1) <= requested_version < (3, 10, 13):
                raise RuntimeError(
                    "AGENTARK_KAGGLE_PYTHON_VERSION must name a Python >=3.10.1,<3.10.13 patch; "
                    f"use {DEFAULT_PYTHON_VERSION}."
                )
            _ensure_uv()
            _run([sys.executable, "-m", "uv", "python", "install", python_version])
            _run([sys.executable, "-m", "uv", "venv", "--seed", "--python", python_version, VENV_DIR])
    version = _python_version(VENV_PY)
    if not (3, 10, 1) <= version < (3, 10, 13):
        raise RuntimeError(
            f"Cached AgentArk environment {VENV_DIR} uses Python {'.'.join(map(str, version))}; "
            "the pinned wrapper requires >=3.10.1,<3.10.13. "
            f"Choose a fresh AGENTARK_KAGGLE_WORK_ROOT and Python {DEFAULT_PYTHON_VERSION}."
        )
    _ensure_venv_pip()
    _run([VENV_PY, "-m", "pip", "install", "-q", "-e", AGENTARK_ROOT])


def _download_verified(url: str, target: Path, sha256: str) -> None:
    def matches() -> bool:
        if not target.is_file():
            return False
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == sha256

    if matches():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            urllib.request.urlretrieve(url, target)
            if not matches():
                raise RuntimeError(f"SHA-256 mismatch: {target.name}")
            return
        except Exception:
            target.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(5 * attempt)


def _extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not (root / name).resolve().is_relative_to(root):
                raise RuntimeError(f"Archive member escapes destination: {name}")
        zf.extractall(root)


def _download_runtime() -> None:
    if (ENV_ROOT / "AgentArk.x86_64").exists():
        return
    _download_verified(ENV_ZIP_URL, ENV_ZIP, ENV_ZIP_SHA256)
    # The 1.0.6 starter ZIP contains the enclosing ENV_ROOT directory.
    _extract_archive(ENV_ZIP, ENV_ROOT.parent)
    exe = ENV_ROOT / "AgentArk.x86_64"
    if not exe.is_file():
        raise FileNotFoundError(f"Expected Unity executable at {exe}")
    exe.chmod(0o755)


def _version_key(value: str) -> Tuple[int, ...]:
    parts = [int(part) for part in value.removeprefix("v").split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _select_task_artifact(rows: List[Dict[str, Any]], task_name: str) -> Dict[str, Any]:
    candidates = [
        row for row in rows
        if row["platform"] == "linux"
        and task_name in [row["task_name"], row.get("task_config_task_name"), *row.get("legacy_names", [])]
        and _version_key(row.get("min_env_version", row["env_version"])) <= _version_key(ENV_VERSION)
    ]
    if not candidates:
        raise ValueError(f"No compatible Linux artifact for {task_name} on env {ENV_VERSION}")
    if len({row["task_id"] for row in candidates}) != 1:
        raise ValueError(f"Ambiguous task name: {task_name}")
    return max(candidates, key=lambda row: (
        _version_key(row["task_version"]),
        _version_key(row.get("min_env_version", row["env_version"])),
        row["path"],
    ))


def _download_task(task_name: str) -> str:
    global TASK_ARTIFACT
    with urllib.request.urlopen(f"{HF_BASE_URL}/registry/tasks.jsonl", timeout=60) as response:
        rows = [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line.strip()]
    row = _select_task_artifact(rows, task_name)
    name = row["task_name"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid task folder name: {name}")
    required_commit = row.get("requirements", {}).get("python_wrapper_git_commit")
    if required_commit:
        _run(["git", "merge-base", "--is-ancestor", required_commit, AGENTARK_COMMIT], cwd=AGENTARK_ROOT)
    archive = WORK_ROOT / "task_archives" / f"{row['sha256']}.zip"
    _download_verified(f"{HF_BASE_URL}/{row['path']}", archive, row["sha256"])
    # Stage the selected immutable package to avoid retaining older catalog/bundle files.
    staging = WORK_ROOT / "task_staging" / name
    if staging.exists():
        shutil.rmtree(staging)
    _extract_archive(archive, staging)
    if not (staging / "cfg" / "task_config.yaml").is_file():
        raise RuntimeError(f"Task archive is missing cfg/task_config.yaml: {name}")
    destination = TASK_STORE_PATH / name
    TASK_STORE_PATH.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(staging), str(destination))
    TASK_ARTIFACT = {key: row[key] for key in ("task_id", "task_name", "task_version", "env_version", "path", "sha256")}
    return name


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
    max_completion_tokens: int,
    temperature: Optional[float] = 1.0,
) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    openai_base_url = base_url.rstrip("/")
    if openai_base_url.endswith("/genai"):
        openai_base_url = openai_base_url[: -len("/genai")]
    if not openai_base_url.endswith(("/openapi", "/openapi/v1")):
        openai_base_url = openai_base_url + "/openapi"
    temperature_yaml = "null" if temperature is None else f"{float(temperature):g}"
    config = f"""
env_cfg:
  env_path: {json.dumps(str(ENV_ROOT / "AgentArk.x86_64"))}
  mod_path: {json.dumps(str(MOD_PATH))}
  task_type: RLTask
  env_config_overrides:
    override_by_task: true
    num_parallel_envs: 1
    virtual_display: true
    virtual_display_render_env:
      GALLIUM_NUM_THREADS: 16

eval:
  output_path: {json.dumps(str(RESULT_PATH.relative_to(AGENTARK_ROOT)))}
  stop_on_error: false
  skip_existing_results: true
  fixed_env_id: {int(env_id)}
  task_names:
    - {json.dumps(task_name)}
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
  - name: "kaggle-simple-model-proxy"
    # agent_type options (harness); provider selects the model backend:
    # simple: SimpleAgent, recommended; context compaction and optional host tools
    #         over HTTP or the Codex SDK.
    # api: APIAgent, the legacy stateless HTTP harness.
    # codex: CodexAgent, the SDK harness (use provider: codex).
    # Omitted: provider: codex selects CodexAgent; HTTP providers select APIAgent.
    # Keep the simple_agent settings below only when agent_type is simple.
    agent_type: simple
    simple_agent:
      tool_choice: required
      output_token_parameter: max_completion_tokens
      context:
        # Tune this estimate for the selected proxy model before publishing.
        max_context_tokens: 262144
        output_reserve_tokens: {int(max_completion_tokens)}
      tools:
        enabled: true
        workspace_parent: tmp/kaggle_simple_agent_workspaces
    provider: openai
    model: {json.dumps(model)}
    base_url: {json.dumps(openai_base_url)}
    api_key_env: MODEL_PROXY_API_KEY
    temperature: {temperature_yaml}
    max_completion_tokens: {int(max_completion_tokens)}
    timeout_s: {float(os.getenv("AGENTARK_KAGGLE_REQUEST_TIMEOUT_S", "600"))}
    max_retries: 1
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


def _is_temperature_unsupported_text(text: str) -> bool:
    lowered = (text or "").lower()
    if "temperature" not in lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "unsupported",
            "not supported",
            "does not support",
            "only the default",
            "unsupported_value",
            "deprecated",
        )
    )


def _results_have_temperature_unsupported_error(results: List[Dict[str, Any]]) -> bool:
    for result in results:
        text = "\n".join(
            str(result.get(key, "") or "")
            for key in ("error", "error_type", "status", "model_name", "model")
        )
        if _is_temperature_unsupported_text(text):
            return True
    return False


def _is_quota_error_text(text: str) -> bool:
    lowered = (text or "").lower()
    return (
        "exceeds your available quota" in lowered
        or ("permissiondeniederror" in lowered and "quota" in lowered)
    )


def _results_have_quota_error(results: List[Dict[str, Any]]) -> bool:
    for result in results:
        text = "\n".join(
            str(result.get(key, "") or "")
            for key in ("error", "error_type", "status")
        )
        if _is_quota_error_text(text):
            return True
    return False


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
    model: str,
    base_url: str,
    max_completion_tokens: int,
) -> List[Dict[str, Any]]:
    if RESULT_PATH.exists():
        RESULT_PATH.unlink()
    expected = set(_expected_seeds(seed_start, seed_end))
    last_returncode = 0
    temperature: Optional[float] = 1.0
    retried_without_temperature = False
    _write_eval_config(
        task_name=task_name,
        seed_start=seed_start,
        seed_end=seed_end,
        env_id=env_id,
        model=model,
        base_url=base_url,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )
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
        if _is_quota_error_text(output) or _results_have_quota_error(results):
            print(
                "Kaggle quota cannot admit another request; stopping instead of "
                "retrying unchanged requests.",
                flush=True,
            )
            break
        if (
            not retried_without_temperature
            and temperature is not None
            and (
                _is_temperature_unsupported_text(output)
                or _results_have_temperature_unsupported_error(results)
            )
        ):
            if RESULT_PATH.exists():
                diagnostic_path = EXPORTED_RESULT_PATH.with_name(
                    EXPORTED_RESULT_PATH.stem + "_before_temperature_retry.jsonl"
                )
                diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(RESULT_PATH, diagnostic_path)
                RESULT_PATH.unlink()
                print(f"Saved pre-retry results to {diagnostic_path}", flush=True)
            temperature = None
            retried_without_temperature = True
            print(
                "Model rejected the temperature parameter; retrying remaining AgentArk "
                "eval attempts without temperature. All seeds will use the new configuration.",
                flush=True,
            )
            _write_eval_config(
                task_name=task_name,
                seed_start=seed_start,
                seed_end=seed_end,
                env_id=env_id,
                model=model,
                base_url=base_url,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
            continue
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
    task_score_scale_min: float,
    task_score_scale_max: float,
) -> Dict[str, Any]:
    task_score_scale_min = float(task_score_scale_min)
    task_score_scale_max = float(task_score_scale_max)
    if task_score_scale_max <= task_score_scale_min:
        raise ValueError(
            "AGENTARK_KAGGLE_TASK_SCORE_SCALE_MAX must be greater than "
            f"AGENTARK_KAGGLE_TASK_SCORE_SCALE_MIN, got min={task_score_scale_min}, max={task_score_scale_max}"
        )
    task_score_scale_range = task_score_scale_max - task_score_scale_min
    expected_count = abs(int(seed_end) - int(seed_start)) + 1
    results = _dedupe_results_by_seed(results)
    if len(results) != expected_count:
        raise AssertionError(f"Expected {expected_count} AgentArk result records, got {len(results)}")
    expected_seeds = set(range(min(int(seed_start), int(seed_end)), max(int(seed_start), int(seed_end)) + 1))
    if {_result_seed(result) for result in results} != expected_seeds:
        raise AssertionError("AgentArk result seeds do not match the requested seed range")

    sorted_results = sorted(results, key=_result_seed)
    scores = [_score_value(result) for result in sorted_results]
    score_sum = sum(scores)
    mean_score = score_sum / len(scores) if scores else 0.0
    normalized_scores = [(score - task_score_scale_min) / task_score_scale_range for score in scores]
    normalized_score_sum = sum(normalized_scores)
    normalized_mean_score = (mean_score - task_score_scale_min) / task_score_scale_range
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
    error_count = sum(1 for result in sorted_results if result.get("status") != "ok")
    return {
        "agentark_commit": AGENTARK_COMMIT,
        "agent_type": "simple",
        "env_version": ENV_VERSION,
        "hf_revision": HF_REVISION,
        "task_artifact": TASK_ARTIFACT,
        "evaluation_status": "complete" if error_count == 0 else "failed",
        "task_name": first.get("actual_task_name", first.get("requested_task_name")),
        "seed_start": int(seed_start),
        "seed_end": int(seed_end),
        "seed_count": len(sorted_results),
        "env_id": first.get("actual_env_id", first.get("requested_env_id")),
        "model": first.get("model"),
        "metric": "mean_score_reward",
        "metric_definition": "Mean of per-seed AgentArk final-attempt score_reward; normalized values are diagnostics only.",
        "task_score_scale_min": task_score_scale_min,
        "task_score_scale_max": task_score_scale_max,
        "mean_score_reward": mean_score,
        "normalized_mean_score_reward": normalized_mean_score,
        "score_reward_sum": score_sum,
        "normalized_score_reward_sum": normalized_score_sum,
        "min_score_reward": min(scores) if scores else 0.0,
        "max_score_reward": max(scores) if scores else 0.0,
        "min_normalized_score_reward": min(normalized_scores) if normalized_scores else 0.0,
        "max_normalized_score_reward": max(normalized_scores) if normalized_scores else 0.0,
        "error_count": error_count,
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
        "agent_type": "simple",
        "env_version": ENV_VERSION,
        "hf_revision": HF_REVISION,
        "task_artifact": TASK_ARTIFACT,
        "evaluation_status": "creation_smoke",
        "task_name": DEFAULT_TASK_NAME,
        "seed_start": DEFAULT_SEED_START,
        "seed_end": DEFAULT_SEED_END,
        "seed_count": DEFAULT_SEED_END - DEFAULT_SEED_START + 1,
        "metric": "mean_score_reward",
        "metric_definition": "Formal benchmark runs average raw per-seed AgentArk final-attempt score_reward; normalized values are diagnostics only.",
        "task_score_scale_min": DEFAULT_TASK_SCORE_SCALE_MIN,
        "task_score_scale_max": DEFAULT_TASK_SCORE_SCALE_MAX,
        "mean_score_reward": 0.0,
        "normalized_mean_score_reward": 0.0,
        "score_reward_sum": 0.0,
        "normalized_score_reward_sum": 0.0,
        "error_count": 0,
        "max_attempts": DEFAULT_EVAL_ATTEMPTS,
        "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "full_eval": False,
        "benchmark_mode": os.getenv("BENCHMARK_MODE", ""),
    }

# %%
@kbench.task(name=DEFAULT_BENCHMARK_TASK_NAME)
def agentark_task_seeds_1_10(llm) -> float:
    """Evaluate the Kaggle-selected model on an AgentArk task over the configured seeds."""
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
    task_score_scale_min = float(os.getenv("AGENTARK_KAGGLE_TASK_SCORE_SCALE_MIN", str(DEFAULT_TASK_SCORE_SCALE_MIN)))
    task_score_scale_max = float(os.getenv("AGENTARK_KAGGLE_TASK_SCORE_SCALE_MAX", str(DEFAULT_TASK_SCORE_SCALE_MAX)))
    seed_start = int(os.getenv("AGENTARK_KAGGLE_SEED_START", str(DEFAULT_SEED_START)))
    seed_end = int(os.getenv("AGENTARK_KAGGLE_SEED_END", str(DEFAULT_SEED_END)))
    max_attempts = int(os.getenv("AGENTARK_KAGGLE_EVAL_ATTEMPTS", str(DEFAULT_EVAL_ATTEMPTS)))
    max_completion_tokens = int(os.getenv("AGENTARK_KAGGLE_MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_COMPLETION_TOKENS)))
    if max_completion_tokens <= 0:
        raise ValueError("AGENTARK_KAGGLE_MAX_COMPLETION_TOKENS must be a positive integer")
    env_id = int(os.getenv("AGENTARK_KAGGLE_ENV_ID", str(DEFAULT_ENV_ID)))
    base_url, api_key, model = _require_model_proxy_env()

    env = _install_and_configure_agentark()
    task_name = _download_task(task_name)
    env["MODEL_PROXY_API_KEY"] = api_key
    results = _run_agentark_eval(
        env,
        task_name=task_name,
        seed_start=seed_start,
        seed_end=seed_end,
        env_id=env_id,
        max_attempts=max_attempts,
        model=model,
        base_url=base_url,
        max_completion_tokens=max_completion_tokens,
    )
    summary = _results_summary(
        results,
        seed_start=seed_start,
        seed_end=seed_end,
        task_score_scale_min=task_score_scale_min,
        task_score_scale_max=task_score_scale_max,
    )
    summary["max_completion_tokens"] = max_completion_tokens
    summary["exported_result_path"] = _export_result_jsonl(results)
    summary["wall_time_s"] = round(time.time() - started, 3)
    summary["exported_summary_path"] = str(EXPORTED_SUMMARY_PATH)
    _export_summary_json(summary)

    print(
        f"AgentArk evaluation_status={summary['evaluation_status']} "
        f"error_count={summary['error_count']} total_turns={summary['total_turns']}; "
        f"summary={EXPORTED_SUMMARY_PATH} log={EXPORTED_AGENTARK_LOG_PATH}",
        flush=True,
    )
    if summary["evaluation_status"] != "complete":
        failed_seeds = [row["seed"] for row in summary["per_seed"] if row["status"] != "ok"]
        raise RuntimeError(
            f"AgentArk evaluation failed for seeds {failed_seeds}; "
            "no leaderboard score was returned. "
            f"Inspect {EXPORTED_SUMMARY_PATH} and {EXPORTED_AGENTARK_LOG_PATH}."
        )

    kbench.assertions.assert_in(
        "complete",
        summary["evaluation_status"],
        expectation="AgentArk should produce a complete score summary for every requested seed.",
    )
    return float(summary["mean_score_reward"])


agentark_task_seeds_1_10.run(kbench.llm)
