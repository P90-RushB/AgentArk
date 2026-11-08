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
# Fresh native and LiteLLM tasks share the published compaction/output-budget fix.
# The historical native notebook and generated benchmarks remain frozen separately.
DEFAULT_WRAPPER_COMMIT = "b6cdf375f32a02c875cecc9258484b4b606c2dbe"
DEFAULT_LITELLM_WRAPPER_COMMIT = DEFAULT_WRAPPER_COMMIT


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
        'DEFAULT_BENCHMARK_TASK_NAME_FALLBACK = f"agentark-{DEFAULT_TASK_SLUG.replace(\'_\', \'-\')}-litellm-responses-auto"')
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
the published wrapper with generic tool-round compaction and separate output
budgets. Fresh native and LiteLLM generation share this revision. The frozen
native notebook and existing benchmark definitions retain their original bytes
and conditions.

The generic defaults evaluate **MarbleStop seeds 1–10**, normalizing its 0–1
final-attempt reward. Every LiteLLM model uses the same first-request policy:
**Responses API, `tool_choice: auto`, omitted temperature and optional reasoning**.
The selected Model Proxy model remains unchanged; there is no model-name routing.
Only LiteLLM Responses canonicalizes `/openapi/v1` to `/openapi`.

Edit `DEFAULT_LITELLM_API = "responses"` (`chat_completions` is also supported),
`DEFAULT_TOOL_CHOICE = None` (`auto` for LiteLLM, `required` for native), or
`DEFAULT_TEMPERATURE = "default"` (omitted for LiteLLM, `1.0` for native).
Matching `AGENTARK_KAGGLE_LITELLM_API`, `AGENTARK_KAGGLE_TOOL_CHOICE`, and
`AGENTARK_KAGGLE_TEMPERATURE` environment settings override those constants.
Temperature accepts `default`, `null`/`none`, or a finite number; explicit Python
`None` also omits it. An explicit `temperature=` argument to the config writer or
evaluation runner takes precedence over the environment, including `None`.
Native HTTP transport retains its existing parameter defaults and
unsupported-temperature retry.

The authored configuration explicitly uses `compaction_policy: tool_rounds_v1`
and `output_budget_policy: separate_native_v1`. Complete tool batches can become
ordinary visible history at a fresh context boundary; older visible content may
then be summarized. Raw IDs and native signatures stay intact in records.
Visible replies and complete tool arguments use action-output limits, while
replay data has `max_protocol_output_bytes: 1048576`. Replay data still counts
toward context size, local token estimates, compaction thresholds, and archive
limits. `legacy_v1` selects the previous combined output-budget behavior;
`message_chunks_v1` and `whole_turn_v1` select older compaction policies.

Tool exploration remains bounded per decision: `tools.limits.max_model_rounds: 16`
counts main-model requests, including host-tool selection and the final
`submit_action`; summary requests do not consume those rounds. `max_calls: 32`
limits host calls. `model tool round budget exhausted` means this exploration
budget ended before a final action was selected; it is distinct from an ID,
signature, or context-size error. Edit these explicit YAML limits in the config
writer if required, and use a new benchmark name for changed evaluation conditions.

The new wrapper pin and authored compaction/output-budget policies define new
evaluation conditions. This notebook does not guarantee that every current or
future proxy model supports its transport settings; check cloud compatibility
for the selected model and endpoint.

Known limitation: `google/gemini-2.5-flash` rejects Responses requests containing
multiple function tools with HTTP 400 (`Multiple tools are supported only when
they are all search tools`). For that model, explicitly choose
`DEFAULT_LITELLM_API = "chat_completions"` or
`AGENTARK_KAGGLE_LITELLM_API=chat_completions` before running. No automatic
model-name rule or API fallback is applied; use a new benchmark name for Chat.
The tested `xai/grok-4.6` Responses route also returned HTTP 404 (`requested
model not found`). Chat is available as the same explicit API alternative; this
probe did not live-validate Grok Chat support.

`auto` enables function tools without forcing a call. The model is still
instructed to finish with `submit_action`; a prose-only response ends the
decision under the existing runtime behavior and can be an invalid task action.
There is no API or parameter fallback for LiteLLM. Tool choice is already part
of the evaluation fingerprint. For a manual YAML evaluation on any compatible
endpoint, explicitly set `simple_agent.tool_choice: auto` when required tool
choice is unsupported.

## Configure DoorSpin3Turns with Astra

These task/model/score settings come from an earlier Astra evaluation. This
notebook uses the newly pinned compaction/output-budget conditions; exact
historical reproduction requires the original notebook and source revision.
Before running the configuration cell, edit these defaults (or use matching
`AGENTARK_KAGGLE_` environment overrides), select Astra, and use a new task name:

| Setting | Value |
| --- | --- |
| `DEFAULT_TASK_NAME` | `DoorSpin3Turns` |
| `DEFAULT_SEED_START` / `DEFAULT_SEED_END` | `1` / `2` |
| `DEFAULT_TASK_SCORE_SCALE_MIN` / `DEFAULT_TASK_SCORE_SCALE_MAX` | `-1.0` / `1.0` |
| `DEFAULT_BENCHMARK_TASK_NAME_FALLBACK` | `agentark-door-spin-3-turns-litellm-astra-s1-2` |
| API / tool choice / temperature | `responses` / `required` / `None` |

For this Door/Astra task/model selection, copy this block to the beginning of the
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
    "AGENTARK_KAGGLE_LITELLM_API": "responses",
    "AGENTARK_KAGGLE_TOOL_CHOICE": "required",
    "AGENTARK_KAGGLE_TEMPERATURE": "null",
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
    slug = f"agentark-{task_slug}" + ("-litellm-responses-auto" if backend == "litellm" else "")
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
