# AgentArk Kaggle Benchmark Tasks

New benchmarks use `agentark_simple_task_template.py` or
`new-benchmark-task-f8fdd.ipynb`. Both explicitly select **SimpleAgent** with native
host tools and context compaction. They pin wrapper commit
`71a9cc305e3e87c0c5ed416086586e24db35210b`, HF revision
`0e360716cff54bb759e8d228ded1df967727ccbe`, and the env-1.0.4 Linux starter hash.
The selected task package is fetched from that immutable registry and its hash
and minimum wrapper commit are checked. A newer task may need a newer compatible
HF/runtime pin; edit those constants before creating its benchmark.

These are new evaluation conditions. Existing hosted task definitions and the
historical `agentark_marble_stop_seeds_1_10.py` remain unchanged. An old task's
immutable wrapper commit continues to select its original harness. Its saved
actions can still be replayed independently of the model harness.

The pinned wrapper includes the SimpleAgent defaults, trace/resume and replay
compatibility improvements, compaction failure handling, and HumanView updates.
For later wrapper revisions, generate with
`--wrapper-commit <full-published-commit>`. Never replace a formal benchmark pin
with `main` or an unpublished local commit.

## Importing the Notebook to Create a New Task

Import `new-benchmark-task-f8fdd.ipynb` into the Kaggle Benchmark task editor.
Its first Markdown cell explains the setup. Before running its configuration
cell, edit the defaults for the new task:

| Setting | Snake smoke example | Purpose |
| --- | --- | --- |
| `DEFAULT_TASK_NAME` | `Snake` | Selects the AgentArk task package |
| `DEFAULT_BENCHMARK_TASK_NAME_FALLBACK` | `agentark-snake-simple` | Gives the new evaluation conditions their own benchmark name |
| `DEFAULT_SEED_START` / `DEFAULT_SEED_END` | `1` / `2` | Keeps the first real test short; restore the formal range afterward |
| `DEFAULT_TASK_SCORE_SCALE_MIN` / `DEFAULT_TASK_SCORE_SCALE_MAX` | `0` / `6` | Normalizes Snake's final-attempt reward for the notebook leaderboard |

The task and benchmark names serve different purposes. A new benchmark name
avoids replacing the existing APIAgent task. Environment variables with the
matching `AGENTARK_KAGGLE_` names can override these defaults.

Creation checks intentionally return `evaluation_status: creation_smoke` without
starting Unity or calling a model. They validate notebook creation, rather than
the SimpleAgent integration. A formal Kaggle `RUN` or `TRIAL` executes the real
evaluation. To test from an interactive notebook instead, set
`os.environ["AGENTARK_KAGGLE_FULL_EVAL"] = "1"` before the final run cell; remove
that temporary override before saving the formal task.

For a real test, inspect the exported summary, results JSONL, and AgentArk log:

- `evaluation_status` must be `complete`, with `error_count: 0` and an `ok` row
  for every requested seed. A valid zero reward remains a valid model result.
- `total_turns` and each seed's `turns` show that environment decisions ran.
  The JSONL records `agent_type: simple` and the actual requests, including
  native `submit_action` calls; host-tool calls and summaries appear when used.
- A failed seed makes the run fail after exporting its diagnostics. Failed
  requests are not silently returned as a zero leaderboard score. If a model
  rejects temperature, earlier results are archived as
  `*_before_temperature_retry.jsonl`, and all seeds rerun with the new setting.

The notebook still returns `normalized_mean_score_reward`; the generated Python
template still returns raw `mean_score_reward`. Neither score formula has changed.
The selected Model Proxy model must support the declared native tools. Tool
errors remain visible rather than silently changing to another harness.

## Historical MarbleStop Benchmark

This directory contains Kaggle Benchmark task files for AgentArk. The first task,
`agentark_marble_stop_seeds_1_10.py`, mirrors the online evaluation notebook flow:

- installs Python 3.10 and the AgentArk package in a temporary runtime directory,
- checks out AgentArk at commit `3cc987164c591f7548a542de334d1e9dab7de725`,
- downloads the public `env-1.0.1` Linux Unity runtime from Hugging Face,
- routes AgentArk's OpenAI-compatible evaluator through Kaggle's Model Proxy,
- returns `mean_score_reward` across `MarbleStop` seeds 1-10 as the Kaggle
  leaderboard score.

The task slug is `agentark-marble-stop`.

AgentArk's generic benchmark metric is `score_reward`, which the repository
defines as the final attempt reward for a case. `rollout_success` and
`ever_attempt_success` are returned as diagnostics only.

Kaggle's leaderboard score comes from the task function's return value. AgentArk
details are exported as artifacts for auditing, but the JSONL is not what Kaggle
uses for ranking. This task therefore returns `float(mean_score_reward)` and
also writes a full summary JSON.

The Python environments are intentionally split:

- Run this Kaggle task file with the Python environment that has
  `kaggle-benchmarks` installed, for example `/root/kaggle_env`.
- The task file then creates `/tmp/agentark_kaggle_work/agentark_py310` and installs
  AgentArk there using Python `3.10.12`, matching the Colab tutorial flow.
  If Kaggle's system image does not provide `python3.10`, the task installs
  that interpreter with `uv` instead of relying on apt packages. The inner
  venv is seeded with pip but does not force-upgrade pip during task creation.
- AgentArk checkout, the Python 3.10 venv, and the downloaded Unity runtime zip
  live under `/tmp/agentark_kaggle_work` by default, so Kaggle output only keeps
  benchmark artifacts instead of the runtime cache.
- Remote headless execution follows the Colab path: the task enables
  `virtual_display: true` and Xvfb/llvmpipe rendering. It does not force
  ML-Agents `no_graphics`, because visual observations are part of AgentArk.
- The detailed AgentArk JSONL is exported to
  `agentark_marble_stop_seeds_1_10_results.jsonl` in the Kaggle run output.
  If earlier retry loops produced error rows that later succeeded, the exported
  JSONL keeps the final record per seed, matching the benchmark metric summary.
- The aggregate summary is exported to
  `agentark_marble_stop_seeds_1_10_summary.json`; this mirrors the leaderboard
  value plus per-seed diagnostics.
  Use `kaggle b t download ...` to fetch it. Kaggle Benchmark `-d/--kaggle-dataset`
  attaches input datasets to a task; publishing downloaded outputs as a Kaggle
  Dataset is a separate dataset-versioning step.
- Notebook output is filtered to show only high-signal AgentArk progress lines.
  Full Unity/AgentArk subprocess output, including `UnityMemory` and
  `[ArkSubEnv]` startup logs, is saved as
  `agentark_marble_stop_seeds_1_10_agentark.log` in the Kaggle run output.
- Kaggle task creation (`push`) runs the notebook once to verify that it emits a
  run output. To keep that creation step within Kaggle's time window, non-`RUN`
  modes return a lightweight `creation_smoke` result. Formal benchmark runs use
  `BENCHMARK_MODE=RUN`/`TRIAL` and execute the full AgentArk evaluation.

## Adding Another AgentArk Task

Generate a task file from the modern SimpleAgent template, then inspect the generated
constants before pushing:

```bash
python3 benchmarks/kaggle/generate_agentark_kaggle_task.py \
  --task-name YourTaskName \
  --seed-start 1 \
  --seed-end 10
```

This creates `benchmarks/kaggle/agentark_yourtaskname.py` with a
matching Kaggle task slug. The generated file follows the same contract:

- fixed AgentArk commit for reproducibility,
- explicit `agent_type: simple`, with `list_files`, `read_file`, `view_image`,
  `write_note`, and native `submit_action`,
- separate SimpleAgent result filenames, so older APIAgent results are not reused,
- seeds 1-10 by default,
- `stop_on_error: false`,
- up to 3 full eval loops while any seed lacks an ok result,
- `virtual_display: true` for headless Kaggle execution,
- `float(mean_score_reward)` returned for the leaderboard, preserving the
  generator's raw final-attempt reward metric; normalized diagnostics are retained,
- JSONL, summary JSON, and AgentArk subprocess log exported as artifacts.

The companion notebook retains its existing `normalized_mean_score_reward`
leaderboard metric. Choose the intended score definition before creating a new
benchmark; the harness migration does not change either entry point's metric.

The generator refuses to overwrite existing benchmark files. `--wrapper-commit`
accepts a full 40-character commit; it must contain SimpleAgent and satisfy the
task artifact's wrapper requirements. Review the model's context window, output
budget and native tool support before running. The 262144-token context estimate
is a local budget, not automatic detection of the selected model's limit.
If the proxy requires `max_tokens` or `tool_choice: auto`, adjust the generated
model recipe before publishing; the template does not silently switch harnesses.

For a new formal task, push and run the generated file:

```bash
kaggle b t push agentark-yourtaskname -f benchmarks/kaggle/agentark_yourtaskname.py --wait
kaggle b t run agentark-yourtaskname -m gemini-3.5-flash --wait
```

Local smoke validation, after `kaggle b init -y` or `kaggle b auth -y`:

```bash
python3 benchmarks/kaggle/agentark_marble_stop_seeds_1_10.py
ls -1 *.run.json
```

Local full validation:

```bash
AGENTARK_KAGGLE_FULL_EVAL=1 python3 benchmarks/kaggle/agentark_marble_stop_seeds_1_10.py
```

Some GPT/Gemini/reasoning model proxy endpoints recommend or require the
provider default sampling temperature. The task starts every model with
`temperature: 1.0`. If AgentArk sees a temperature-specific error such as
"temperature is deprecated" or "temperature is unsupported", the same task
rewrites its eval config to `temperature: null` and retries the remaining
AgentArk eval attempts without sending the temperature parameter.

Remote Kaggle Benchmark runs are different: they are not launched with
`python3 benchmarks/kaggle/...py`. Push the task file, then run it remotely with
the Kaggle CLI:

```bash
kaggle b t push agentark-marble-stop -f benchmarks/kaggle/agentark_marble_stop_seeds_1_10.py --wait
kaggle b t run agentark-marble-stop -m gpt-5.5 --wait
```

The `kaggle b t run ... -m ...` command selects the remote model. The
temperature retry behavior above runs inside the remote task, so all models use
the same pushed task file and leaderboard definition.

For the historical MarbleStop Python task, the model selector maps
`gemini-3-flash-preview` to
`gemini-3.5-flash` for AgentArk evaluation. Override this with
`AGENTARK_KAGGLE_CREATION_MODEL` if Kaggle's default changes or if a cheaper
creation-check model is preferred. Formal `run -m ...` selections still take
priority unless the chosen model is exactly Kaggle's push-default alias; use
`AGENTARK_KAGGLE_MODEL` for an explicit override.
The modern SimpleAgent notebook/template uses the selected model unchanged and
does not implement that historical creation-model alias.

Push and run, one checkpoint at a time:

```bash
kaggle b t push agentark-marble-stop -f benchmarks/kaggle/agentark_marble_stop_seeds_1_10.py --wait
kaggle b t run agentark-marble-stop -m gemini-3.5-flash --wait
kaggle b t status agentark-marble-stop
kaggle b t download agentark-marble-stop -o ./results -s
```

Publish after the task and at least one model run are satisfactory:

```bash
kaggle b t publish agentark-marble-stop
```

Useful environment overrides for the modern SimpleAgent notebook/template:

- `AGENTARK_KAGGLE_TASK_NAME`, default `MarbleStop`
- `AGENTARK_KAGGLE_SEED_START`, default `1`
- `AGENTARK_KAGGLE_SEED_END`, default `10`
- `AGENTARK_KAGGLE_TASK_SLUG`, optional artifact filename slug
- `AGENTARK_KAGGLE_BENCHMARK_NAME`, optional benchmark name override
- `AGENTARK_KAGGLE_TASK_SCORE_SCALE_MIN` / `AGENTARK_KAGGLE_TASK_SCORE_SCALE_MAX`, default `0.0` / `1.0`; set the task's score range for notebook normalization
- `AGENTARK_KAGGLE_EVAL_ATTEMPTS`, default `3`; reruns the full AgentArk eval loop while any requested seed lacks an ok result, then stops when all seeds are ok or the loop limit is reached
- `AGENTARK_KAGGLE_ENV_ID`, default `0`
- `AGENTARK_KAGGLE_MODEL`, explicit model override for local or remote runs
- `AGENTARK_KAGGLE_MAX_COMPLETION_TOKENS`, default `30000`
- `AGENTARK_KAGGLE_REQUEST_TIMEOUT_S`, default `600`
- `AGENTARK_KAGGLE_FULL_EVAL=1`, force a full AgentArk evaluation outside
  Kaggle `BENCHMARK_MODE=RUN`
- `AGENTARK_KAGGLE_PYTHON_VERSION`, default `3.10.12`; must satisfy the pinned wrapper's `>=3.10.1,<3.10.13` requirement
- `AGENTARK_KAGGLE_PYTHON_BIN`, optional explicit compatible Python executable
- `AGENTARK_KAGGLE_WORK_ROOT`, default `/tmp/agentark_kaggle_work`
- `AGENTARK_KAGGLE_SKIP_APT=1`, for images that already contain Python 3.10, git, and Xvfb
