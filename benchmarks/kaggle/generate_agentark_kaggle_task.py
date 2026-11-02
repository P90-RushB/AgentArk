#!/usr/bin/env python3
"""Generate a new SimpleAgent Kaggle task without changing historical tasks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "agentark_simple_task_template.py"
NATIVE_NOTEBOOK_PATH = ROOT / "new-benchmark-task-f8fdd.ipynb"
DEFAULT_WRAPPER_COMMIT = "b37efe0b692b29a16c42e31b19aecbd3472271cf"
# Published optional SDK source; native generation keeps its original pin.
DEFAULT_LITELLM_WRAPPER_COMMIT = "3465fc850634204da24f82813a22541e7cf7eb00"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError(f"Unable to derive slug from {value!r}")
    return slug


def _function_name(slug: str) -> str:
    return slug.replace("-", "_")


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Template token not found: {old!r}")
    return text.replace(old, new, 1)


def _backend_template(backend: str, wrapper_commit: str | None) -> str:
    if backend not in {"native", "litellm"}:
        raise ValueError("backend must be native or litellm")
    pin = wrapper_commit if wrapper_commit is not None else (DEFAULT_LITELLM_WRAPPER_COMMIT if backend == "litellm" else DEFAULT_WRAPPER_COMMIT)
    if pin is None:
        raise ValueError("LiteLLM default wrapper pin awaits publication; provide a full published immutable wrapper_commit")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", pin):
        raise ValueError("wrapper_commit must be a full immutable 40-character Git commit")
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    text = _replace_once(text, f'AGENTARK_COMMIT = "{DEFAULT_WRAPPER_COMMIT}"',
                         f'AGENTARK_COMMIT = "{pin.lower()}"')
    text = _replace_once(text, 'DEFAULT_MODEL_BACKEND = "native"', f'DEFAULT_MODEL_BACKEND = "{backend}"')
    if backend == "litellm":
        text = _replace_once(text, "# The pinned native default remains usable without the optional SDK.",
                             "# This benchmark explicitly selects and installs the optional LiteLLM SDK.")
        text = _replace_once(text, '"/tmp/agentark_kaggle_work"', '"/tmp/agentark_kaggle_litellm_work"')
        text = _replace_once(text,
            'RESULT_STEM = f"agentark_{DEFAULT_TASK_SLUG}_simple_seeds_{DEFAULT_SEED_START}_{DEFAULT_SEED_END}"',
            'RESULT_STEM = f"agentark_{DEFAULT_TASK_SLUG}_simple_litellm_seeds_{DEFAULT_SEED_START}_{DEFAULT_SEED_END}"')
    return text


def build_litellm_notebook(*, wrapper_commit: str | None = None) -> dict:
    """Derive the importable notebook from the one shared Python template."""
    text = _backend_template("litellm", wrapper_commit)
    text = _replace_once(text,
        'DEFAULT_BENCHMARK_TASK_NAME_FALLBACK = f"agentark-{DEFAULT_TASK_SLUG.replace(\'_\', \'-\')}"',
        'DEFAULT_BENCHMARK_TASK_NAME_FALLBACK = f"agentark-{DEFAULT_TASK_SLUG.replace(\'_\', \'-\')}-litellm"')
    notebook = json.loads(NATIVE_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    _, *chunks = text.split("# %%\n")
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if len(cells) != len(chunks):
        raise RuntimeError("Shared template and notebook cell structure differ")
    for cell, chunk in zip(cells, chunks):
        source = chunk.rstrip() + "\n"
        source = source.replace('return float(summary["mean_score_reward"])', 'return float(summary["normalized_mean_score_reward"])')
        source = source.replace('"metric": "mean_score_reward"', '"metric": "normalized_mean_score_reward"')
        source = source.replace('Mean of per-seed AgentArk final-attempt score_reward; normalized values are diagnostics only.',
                                'Mean of per-seed AgentArk score_reward normalized from the configured task score min/max range to 0-1.')
        source = source.replace('Formal benchmark runs average raw per-seed AgentArk final-attempt score_reward; normalized values are diagnostics only.',
                                'Formal benchmark runs normalize each AgentArk score_reward from the configured task score min/max range to 0-1, then average across seeds.')
        cell.update(source=source.splitlines(keepends=True), execution_count=None, outputs=[])
    introduction = """# AgentArk — LiteLLM Kaggle benchmark

Import this notebook as a **new** Kaggle Benchmark task. It defaults to
`DEFAULT_MODEL_BACKEND = "litellm"`, native function tools, and context
compaction. The immutable `AGENTARK_COMMIT` in the configuration cell selects
the published wrapper with optional LiteLLM support. The separate native
notebook and existing benchmark definitions retain their original conditions.

The generic defaults evaluate **MarbleStop seeds 1–10**, normalizing its 0–1
final-attempt reward. The selected Model Proxy model remains unchanged. Verified
`openai/gpt-6-astra` / `gpt-6-astra` models default to the Responses API with
`temperature: null`; other models default to Chat Completions. Set
`DEFAULT_LITELLM_API` or `AGENTARK_KAGGLE_LITELLM_API` to choose either API
explicitly. Only LiteLLM Responses canonicalizes `/openapi/v1` to `/openapi`.

`DEFAULT_TOOL_CHOICE = None` selects `auto` for the known Kaggle Opus 5.5
aliases (last model segment `claude-opus-5-5` or `claude-opus-5.5`, optionally
ending in `-default` or `@default`) in LiteLLM mode. Other models and native mode keep
`required`. Set `DEFAULT_TOOL_CHOICE` or `AGENTARK_KAGGLE_TOOL_CHOICE` to
`required` or `auto` to override this before the first request. This mapping is
limited to the Kaggle template; the wrapper pin and core defaults are unchanged.
`auto` enables function tools without forcing a call. The model is still
instructed to finish with `submit_action`; a prose-only response ends the
decision under the existing runtime behavior and can be an invalid task action.
There is no new retry or exception-based fallback. Tool choice is already part
of the evaluation fingerprint. For a manual YAML evaluation on any compatible
endpoint, explicitly set `simple_agent.tool_choice: auto` when required tool
choice is unsupported.

## Reproduce the DoorSpin3Turns Astra evaluation

Before running the configuration cell, edit these defaults (or use the matching
`AGENTARK_KAGGLE_` environment overrides) and select Astra in Kaggle:

| Setting | Value |
| --- | --- |
| `DEFAULT_TASK_NAME` | `DoorSpin3Turns` |
| `DEFAULT_SEED_START` / `DEFAULT_SEED_END` | `1` / `2` |
| `DEFAULT_TASK_SCORE_SCALE_MIN` / `DEFAULT_TASK_SCORE_SCALE_MAX` | `-1.0` / `1.0` |
| `DEFAULT_BENCHMARK_TASK_NAME_FALLBACK` | `agentark-door-spin-3-turns-litellm-astra-s1-2` |

For the tested Door/Astra conditions, copy this block to the beginning of the
configuration cell, before its defaults are evaluated:

```python
import os
os.environ.update({
    "AGENTARK_KAGGLE_TASK_NAME": "DoorSpin3Turns",
    "AGENTARK_KAGGLE_SEED_START": "1",
    "AGENTARK_KAGGLE_SEED_END": "2",
    "AGENTARK_KAGGLE_TASK_SCORE_SCALE_MIN": "-1.0",
    "AGENTARK_KAGGLE_TASK_SCORE_SCALE_MAX": "1.0",
    "AGENTARK_KAGGLE_BENCHMARK_NAME": "agentark-door-spin-3-turns-litellm-astra-s1-2",
    "AGENTARK_KAGGLE_MODEL": "openai/gpt-6-astra",
})
```

Use a new benchmark name for new conditions. When choosing the model through an
environment override, use `AGENTARK_KAGGLE_MODEL=openai/gpt-6-astra`. This recipe
does not change other tasks or their scoring rules.

With `BENCHMARK_MODE=CREATE` (and full evaluation disabled), importing and running
the notebook produces `evaluation_status: creation_smoke` without installing
AgentArk, launching Unity, or calling a model. `RUN` / `TRIAL` perform the full
evaluation. For an interactive full evaluation, explicitly set
`AGENTARK_KAGGLE_FULL_EVAL=1` before the final cell. Failed seeds export diagnostics
and fail the run. Inspect the summary, results JSONL, and AgentArk log; this
notebook returns normalized mean score, while generated Python tasks keep raw
mean score. No source overlays or diagnostic monkeypatches are embedded.
"""
    notebook["cells"][0]["source"] = introduction.splitlines(keepends=True)
    return notebook


def generate(
    *,
    task_name: str,
    seed_start: int,
    seed_end: int,
    env_id: int,
    output: Path | None,
    wrapper_commit: str | None = None,
    backend: str = "native",
) -> Path:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task_name):
        raise ValueError("task_name must be an AgentArk task identifier")
    if seed_start < 0 or seed_end < seed_start or env_id < 0:
        raise ValueError("require 0 <= seed_start <= seed_end and env_id >= 0")
    text = _backend_template(backend, wrapper_commit)
    task_slug = _slugify(task_name)
    slug = f"agentark-{task_slug}" + ("-litellm" if backend == "litellm" else "")
    func_name = _function_name(slug)
    output_path = output or ROOT / f"{func_name}.py"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing benchmark: {output_path}")

    text = text.replace("agentark_task_seeds_1_10", func_name)
    text = _replace_once(text,
        'DEFAULT_TASK_NAME = os.getenv("AGENTARK_KAGGLE_TASK_NAME", "MarbleStop").strip() or "MarbleStop"',
        f'DEFAULT_TASK_NAME = os.getenv("AGENTARK_KAGGLE_TASK_NAME", {json.dumps(task_name)}).strip() or {json.dumps(task_name)}')
    text = _replace_once(text, "DEFAULT_SEED_START = 1", f"DEFAULT_SEED_START = {int(seed_start)}")
    text = _replace_once(text, "DEFAULT_SEED_END = 10", f"DEFAULT_SEED_END = {int(seed_end)}")
    text = _replace_once(text, "DEFAULT_ENV_ID = 0", f"DEFAULT_ENV_ID = {int(env_id)}")
    text = _replace_once(text,
        'DEFAULT_BENCHMARK_TASK_NAME_FALLBACK = f"agentark-{DEFAULT_TASK_SLUG.replace(\'_\', \'-\')}"',
        f'DEFAULT_BENCHMARK_TASK_NAME_FALLBACK = {json.dumps(slug)}')

    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="AgentArk task name, e.g. MarbleStop")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--backend", choices=("native", "litellm"), default="native")
    parser.add_argument("--wrapper-commit", default=None,
                        help="Override the backend-specific full immutable wrapper commit; never a moving branch")
    args = parser.parse_args()

    output_path = generate(
        task_name=args.task_name,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        env_id=args.env_id,
        output=args.output,
        wrapper_commit=args.wrapper_commit,
        backend=args.backend,
    )
    print(output_path)


if __name__ == "__main__":
    main()
