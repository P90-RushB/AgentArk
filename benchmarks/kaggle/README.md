# AgentArk Kaggle Benchmark Tasks

This directory contains Kaggle Benchmark task files for AgentArk. The first task,
`agentark_marble_stop_seeds_1_10.py`, mirrors the online evaluation notebook flow:

- installs Python 3.10 and the AgentArk package in a temporary runtime directory,
- checks out AgentArk at commit `31a97e7800861e9c1d5b382f323b64392328d042`,
- downloads the public `env-1.0.1` Linux Unity runtime from Hugging Face,
- routes AgentArk's OpenAI-compatible evaluator through Kaggle's Model Proxy,
- returns `mean_score_reward` across `MarbleStop` seeds 1-10 as the Kaggle
  leaderboard score.

The task slug is `agentark-marble-stop-seeds-1-10`.

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

Generate a task file from the MarbleStop template, then inspect the generated
constants before pushing:

```bash
python3 benchmarks/kaggle/generate_agentark_kaggle_task.py \
  --task-name YourTaskName \
  --seed-start 1 \
  --seed-end 10
```

This creates `benchmarks/kaggle/agentark_yourtaskname_seeds_1_10.py` with a
matching Kaggle task slug. The generated file follows the same contract:

- fixed AgentArk commit for reproducibility,
- seeds 1-10 by default,
- `stop_on_error: false`,
- up to 100 full eval loops while any seed lacks an ok result,
- `virtual_display: true` for headless Kaggle execution,
- `float(mean_score_reward)` returned for the leaderboard,
- JSONL, summary JSON, and AgentArk subprocess log exported as artifacts.

For a new formal task, push and run the generated file:

```bash
kaggle b t push agentark-yourtaskname-seeds-1-10 -f benchmarks/kaggle/agentark_yourtaskname_seeds_1_10.py --wait
kaggle b t run agentark-yourtaskname-seeds-1-10 -m gemini-3.5-flash --wait
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
kaggle b t push agentark-marble-stop-seeds-1-10 -f benchmarks/kaggle/agentark_marble_stop_seeds_1_10.py --wait
kaggle b t run agentark-marble-stop-seeds-1-10 -m gpt-5.5 --wait
```

The `kaggle b t run ... -m ...` command selects the remote model. The
temperature retry behavior above runs inside the remote task, so all models use
the same pushed task file and leaderboard definition.

Push and run, one checkpoint at a time:

```bash
kaggle b t push agentark-marble-stop-seeds-1-10 -f benchmarks/kaggle/agentark_marble_stop_seeds_1_10.py --wait
kaggle b t run agentark-marble-stop-seeds-1-10 -m gemini-3.5-flash --wait
kaggle b t status agentark-marble-stop-seeds-1-10
kaggle b t download agentark-marble-stop-seeds-1-10 -o ./results -s
```

Publish after the task and at least one model run are satisfactory:

```bash
kaggle b t publish agentark-marble-stop-seeds-1-10
```

Useful environment overrides while developing copies of the task:

- `AGENTARK_KAGGLE_TASK_NAME`, default `MarbleStop`
- `AGENTARK_KAGGLE_SEED_START`, default `1`
- `AGENTARK_KAGGLE_SEED_END`, default `10`
- `AGENTARK_KAGGLE_EVAL_ATTEMPTS`, default `100`; reruns the full AgentArk eval loop while any requested seed lacks an ok result, then stops when all seeds are ok or the loop limit is reached
- `AGENTARK_KAGGLE_ENV_ID`, default `0`
- `AGENTARK_KAGGLE_MODEL`, default `LLM_DEFAULT` or `LLM_DEFAULT_EVAL`
- `AGENTARK_KAGGLE_FULL_EVAL=1`, force a full AgentArk evaluation outside
  Kaggle `BENCHMARK_MODE=RUN`
- `AGENTARK_KAGGLE_PYTHON_VERSION`, default `3.10.12`
- `AGENTARK_KAGGLE_PYTHON_BIN`, optional explicit Python 3.10 executable
- `AGENTARK_KAGGLE_WORK_ROOT`, default `/tmp/agentark_kaggle_work`
- `AGENTARK_KAGGLE_SKIP_APT=1`, for images that already contain Python 3.10, git, and Xvfb
