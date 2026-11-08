# AgentArk Kaggle Benchmark Tasks

There are two importable notebooks. Both select **SimpleAgent**, native function
tools, and context compaction, with generic MarbleStop seeds 1–10 defaults:

| Notebook | Default backend | Wrapper pin |
| --- | --- | --- |
| `new-benchmark-task-f8fdd.ipynb` | Native HTTP | Frozen `b37efe0b692b29a16c42e31b19aecbd3472271cf` |
| `agentark_litellm_task.ipynb` | LiteLLM SDK | `7822a18b60d2798572d16cab844dce7295181efe` |

The native notebook is retained byte-for-byte for its existing evaluation
conditions. The LiteLLM notebook uses separate benchmark names, result names,
and a work directory containing `litellm`. Both retain HF revision
`6a16c98e262abd760b09ef007a49118d0b1b0cb9` (`env-1.0.6`) and the env-1.0.6 Linux starter hash.
The selected task package is fetched from that immutable registry and its hash
and minimum wrapper commit are checked. A newer task may need a newer compatible
HF/runtime pin; edit those constants before creating its benchmark.

The 1.0.6 starter is named `AgentArk-env-1.0.6-linux.zip` (138,660,417 bytes)
and contains an `AgentArk-env-1.0.6-linux/` top-level directory. Its SHA-256 is
`f32061282f51bbc2ab36a4617d7af5eb6b4d349a944dd861171aba95c00ccdcc`.
The notebook handles that layout. The runtime includes the 1.0.6 memory cleanup
fixes and keeps the legacy environment task-selection menu hidden while
preserving task-owned UI and configuration-driven task loading. Only newly
imported or generated templates get these defaults; existing hosted or imported
task definitions keep their own immutable pins.

These are new evaluation conditions. Existing hosted task definitions and the
historical `agentark_marble_stop_seeds_1_10.py` remain unchanged. An old task's
immutable wrapper commit continues to select its original harness. Its saved
actions can still be replayed independently of the model harness.

The native wrapper pin includes the SimpleAgent defaults, trace/resume and replay
compatibility improvements, compaction failure handling, and HumanView updates.
It also accepts opaque provider tool-call IDs up to 64 KiB of UTF-8 without
rewriting them, fixing the prior 256-character rejection on Kaggle Gemini calls.
The current LiteLLM notebook, shared Python template and both fresh generator
backends pin `7822a18b60d2798572d16cab844dce7295181efe`, explicitly using
`compaction_policy: tool_rounds_v1` and
`output_budget_policy: separate_native_v1`, with
`round_budget_policy: visible_v1`. Closed tool batches may restart
context within a decision; visible arguments/results remain available and older
history may be summarized. Opaque signatures leave working context only at a
fresh-context boundary, while immutable records retain their originals.

Visible text/full tool arguments and protocol replay have separate single-output
limits. `max_protocol_output_bytes: 1048576` bounds the aggregate replay data;
the existing per-ID 64 KiB and per-native-state 1 MiB limits remain. All replay
data still counts toward context characters, bytes, token estimates and archive
limits. This does not make encrypted state free or provide an exact count of
server-restored tokens. To reproduce pre-separation checks, explicitly select
`output_budget_policy: legacy_v1` together with the historical compaction policy.
For evaluations predating visible round-budget snapshots, also select
`round_budget_policy: legacy_v1`; retain the output-budget and compaction
policies recorded for that evaluation when reproducing its identity. Saved
records remain replayable without changing their historical prompts or actions.
The frozen native notebook retains its `message_chunks_v1` conditions and old pin.
Change strategy and budgets in `_write_eval_config` before creating a new version.
For later wrapper revisions, generate with
`--wrapper-commit <full-published-commit>`. Never replace a formal benchmark pin
with `main` or an unpublished local commit.

## LiteLLM Notebook and Generated Tasks

Import `agentark_litellm_task.ipynb` for a new LiteLLM benchmark. It sets
`DEFAULT_MODEL_BACKEND = "litellm"`, installs `AgentArk[litellm]`, and authors
`simple_agent.backend: litellm`. Its first Markdown cell gives a DoorSpin3Turns
recipe: seeds 1–2, score range −1–1, and a new benchmark name. Historical Astra
and Opus validation conditions are recorded separately in the
[LiteLLM guide](../../docs/litellm-evaluation.md#validation-scope).
Its runnable defaults remain generic MarbleStop seeds 1–10 with score range 0–1.

The shared Python template `agentark_simple_task_template.py` still defaults to
native HTTP, including Gemini. Both fresh generator backends select the current
immutable wrapper pin; `--backend litellm` gives the default generated filename
and Python function a `_litellm_responses_auto` suffix and the benchmark slug
a `-litellm-responses-auto` suffix. For example:

```bash
python benchmarks/kaggle/generate_agentark_kaggle_task.py --backend litellm \
  --task-name Snake --seed-start 1 --seed-end 2
```

This produces `benchmarks/kaggle/agentark_snake_litellm_responses_auto.py` with
function `agentark_snake_litellm_responses_auto` and benchmark slug
`agentark-snake-litellm-responses-auto`. The CLI discovers that function
identifier when loading the generated task. An explicit `--output` may choose
another filename. Push and run the new condition with:

```bash
kaggle b t push agentark-snake-litellm-responses-auto -f benchmarks/kaggle/agentark_snake_litellm_responses_auto.py --wait
kaggle b t run agentark-snake-litellm-responses-auto -m your-model-slug --wait
```

Omitting `--backend` preserves native generation with the current pin. An
explicit `--wrapper-commit <full-published-commit>` overrides the selected pin.
The frozen notebook's older native pin does not contain LiteLLM; setup rejects a missing backend
feature instead of silently changing transport. Environment overrides
`AGENTARK_KAGGLE_BACKEND`, `AGENTARK_KAGGLE_LITELLM_API`,
`AGENTARK_KAGGLE_TOOL_CHOICE` and `AGENTARK_KAGGLE_TEMPERATURE` remain available in
the shared template and new notebook. The frozen native notebook keeps its
original settings.

Only one Python implementation is maintained. The generator's
`build_litellm_notebook()` derives the LiteLLM notebook from that shared template;
tests verify the generated notebook, empty outputs, immutable pin, and offline
import. The new notebook contains no source overlays or diagnostic patches.

LiteLLM mode uses the same default for every selected model: Responses,
`tool_choice: auto`, omitted temperature, and omitted optional reasoning
settings. Model names do not choose these settings. The selected wire model
remains unchanged in `model`; the LiteLLM
route prefixes it with `openai/` to use the same Model Proxy `/openapi` endpoint.
The Responses mode removes a trailing `/v1` from `/openapi/v1`, using the proxy's
supported `/openapi/responses` route; native and LiteLLM Chat keep their existing URL handling.
For example, `openai/gpt-6-astra` becomes
`simple_agent.litellm: {model: openai/openai/gpt-6-astra, api: responses}`.

| Constant | Environment override | Values and default |
| --- | --- | --- |
| `DEFAULT_LITELLM_API = "responses"` | `AGENTARK_KAGGLE_LITELLM_API` | `responses` or `chat_completions` |
| `DEFAULT_TOOL_CHOICE = None` | `AGENTARK_KAGGLE_TOOL_CHOICE` | `auto` or `required`; `None` selects `auto` for LiteLLM and `required` for native |
| `DEFAULT_TEMPERATURE = "default"` | `AGENTARK_KAGGLE_TEMPERATURE` | `default` selects omitted temperature for LiteLLM and `1.0` for native; `null`/`none` omits it; a finite number sets it explicitly |

Environment overrides take precedence over constants. An explicit `temperature`
argument to `_write_eval_config` or `_run_agentark_eval`, including `None`, takes
precedence over the temperature environment override. Python constants also
accept `None` or a finite numeric temperature; booleans and nonfinite values are
rejected. Invalid settings fail during config authoring. The selected settings
apply to the first request; no API or parameter fallback is triggered by errors.
Omitted temperature is recorded as `temperature: null`; reasoning remains
optional, with no forced `reasoning_effort: none`.

The authored YAML also shows the existing host-tool budgets explicitly:
`simple_agent.tools.limits.max_model_rounds: 16` and `max_calls: 32` per decision.
Action-generation requests, including host-tool selection and final submission,
consume rounds; summary requests do not. Parallel host calls consume their
individual host-call slots within one main-model round. With explicit
`round_budget_policy: visible_v1`, every main request receives an append-only
snapshot of its current/total/remaining rounds, including that response, and
the separate remaining host-call slots. The last available main response is
prompted to submit the final action through the currently exposed schema.
Earlier snapshots are historical; old request prefixes are preserved, and
compaction does not reset the counters. No tools are removed or tool choice
forced. Ignoring the final-round hint still reaches the same hard stop.
`round_budget_policy: legacy_v1` disables these snapshots for historical
request/identity reproduction. `model tool round budget exhausted`
means the model kept exploring without submitting within that allowance. It is
different from an ID, output or context limit. Change the YAML only for a new
evaluation condition; compaction keeps the current counters rather than resetting them.

The targeted Gemini 3.8 Flash validation accepted three native submissions above
the old 120k action cap and continued through five context restarts. Both seeds
ultimately stopped at the unchanged 16-round tool budget after 5/3 environment
steps, so this is not a full-case pass. See the
[English validation scope](../../docs/litellm-evaluation.md#validation-scope) or
[中文验证范围](../../docs/litellm-evaluation.zh-CN.md#验证范围) for the evidence and limits.

This is a Kaggle notebook/shared-template policy. Core LiteLLM configs continue
to default to Chat Completions and `required`. Native generation retains Chat
Completions, `required`, temperature `1.0`, and its existing temperature-error
retry. The frozen native notebook remains unchanged. Older imported LiteLLM
notebooks keep their original model-name rules: manually selecting Responses and
null temperature does not change their old Opus-only automatic tool-choice
mapping. Choose tool choice explicitly or import the new generated notebook.

`auto` continues to expose function tools and instruct the model to
`submit_action`, without forcing a tool call. A prose-only response ends the
decision under existing SimpleAgent behavior and can become an invalid
environment action. The round-budget hint does not add a retry for a prose-only
response or an exception-based fallback. Tool choice and the round-budget policy
are included in the evaluation fingerprint.
Local manual YAML can also set `tool_choice: auto` under `simple_agent`.

LiteLLM mode uses zero SDK retries and keeps unsupported-parameter errors
visible. Choose a new benchmark name and output path for these evaluation
conditions; the new default name ends in `-litellm-responses-auto`. This uniform
policy does not guarantee that every current or future model supports Responses,
images, tools, and the chosen parameters. Historical Astra Responses/required and
Opus Chat/auto results do not certify the new policy across vendors. Existing
tasks, seeds, artifacts, and score formulas remain intact.

Kaggle Gemini 2.5 Flash (`google/gemini-2.5-flash`) has a known Responses
limitation: the production adapter's request with two function tools returned
HTTP 400 `invalid_prompt`: "Multiple tools are supported only when they are all
search tools". Select `DEFAULT_LITELLM_API = "chat_completions"` or
`AGENTARK_KAGGLE_LITELLM_API=chat_completions` explicitly for this configuration.
There is no model-name routing or automatic API fallback.

The tested Grok 4.6 Responses route (`xai/grok-4.6`) returned HTTP 404
"requested model not found" on its first production-adapter request. The same
explicit Chat override is a configurable alternative, but Chat has not been
live-verified here. The child receipt establishes this failure; the outer
benchmark's completion status did not establish success.

| Evidence | Established scope |
| --- | --- |
| Synthetic SDK / local HTTP loopback | Controlled image, tool continuation, state, and configuration behavior; no provider certification. |
| Production adapter v2 / Responses | Sol 6.1 (`openai/gpt-6.1-sol`, run 6231754), Opus 5.5 (`anthropic/claude-opus-5-5@default`, run 6231755), and Gemini 3 Flash Preview (`google/gemini-3-flash-preview`, run 6231756) passed three actual production `aresponses` calls each: two image turns with a host echo result, native continuation and action submission, then a third tool-free image summary. All used auto, omitted temperature/reasoning effort, and zero retries. These are small protocol smokes, not Unity/full Door evaluations. Gemini 2.5 Flash returned the two-function-tool HTTP 400 above; Grok 4.6 (`xai/grok-4.6`) returned HTTP 404 "requested model not found" on its first request. Chat alternatives remain unverified here. |
| Raw OpenAI SDK / Gemini 3 Flash Preview | `/models/openapi/responses` passed two tools and two turns, with 2,700 tokens reported; `/models/openapi/v1/responses` returned HTTP 405. This is transport-only, not production-adapter proof. |
| Historical packaged `DoorSpin3Turns` evaluation | Astra used Responses/required/null temperature; Opus used Chat/auto/temperature 1.0. Full results and quota limits remain in the [historical validation matrix](../../docs/litellm-evaluation.md#validation-scope). |

The local-only [v2 probe verification report](../../../local_save/task_validation/SimpleAgent/unified_kaggle_20261105/verified_v2_probes.json)
summarizes receipts under
`local_save/task_validation/SimpleAgent/unified_kaggle_20261105/v2_outputs/`
(relative to the outer repository root). These ignored probe artifacts are
unavailable in a fresh clone.

## Importing the Notebook to Create a New Task

Import the native `new-benchmark-task-f8fdd.ipynb` or LiteLLM
`agentark_litellm_task.ipynb` into the Kaggle Benchmark task editor.
Its first Markdown cell explains the setup. Before running its configuration
cell, edit the defaults for the new task:

| Setting | Snake smoke example | Purpose |
| --- | --- | --- |
| `DEFAULT_TASK_NAME` | `Snake` | Selects the AgentArk task package |
| `DEFAULT_BENCHMARK_TASK_NAME_FALLBACK` | `agentark-snake-litellm-responses-auto` | Gives the LiteLLM evaluation conditions their own benchmark name; choose a separate name for native conditions |
| `DEFAULT_SEED_START` / `DEFAULT_SEED_END` | `1` / `2` | Keeps the first real test short; restore the formal range afterward |
| `DEFAULT_TASK_SCORE_SCALE_MIN` / `DEFAULT_TASK_SCORE_SCALE_MAX` | `0` / `6` | Normalizes Snake's final-attempt reward for the notebook leaderboard |

The task and benchmark names serve different purposes. A new benchmark name
avoids replacing the existing APIAgent task. Environment variables with the
matching `AGENTARK_KAGGLE_` names can override these defaults.

With `BENCHMARK_MODE=CREATE` or another mode outside `RUN`/`TRIAL`, and full
evaluation disabled, the notebook returns `evaluation_status: creation_smoke`
without starting Unity or calling a model. That checks notebook import and output
generation. `RUN` or `TRIAL` executes the real evaluation, including when Kaggle
injects `RUN` during a task `push`; pushing does not guarantee a lightweight run.
To test from an interactive notebook, set
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
  rejects temperature in the default native backend, earlier results are archived as
  `*_before_temperature_retry.jsonl`, and all seeds rerun with the new setting.

The notebook still returns `normalized_mean_score_reward`; the generated Python
template still returns raw `mean_score_reward`. Neither score formula has changed.
The selected Model Proxy model must support the declared native tools. Tool
errors remain visible rather than silently changing to another harness.
Kaggle's Gemini `/openapi` route supports native tools. Its tool-call IDs can
contain opaque thought signatures such as a `__mpthought__` suffix; the entire
original ID must be returned unchanged in both the assistant tool call and its
matching tool result. The new wrapper pin preserves that continuation state.

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
  run output. With full evaluation disabled, modes outside `RUN`/`TRIAL` return
  a lightweight `creation_smoke` result. `BENCHMARK_MODE=RUN`/`TRIAL` executes
  the full AgentArk evaluation, even when Kaggle injects that mode during `push`.

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
The template explicitly allows 229376 estimated summary-input tokens plus
32768 summary-output tokens within that same 262144-token budget. Its 524288
summary-input character cap matches the working-context cap. This Kaggle override
can reduce summary requests for video history; the core's default remains 48000
estimated input tokens. The message policy can split eligible old messages when
needed. Protected inputs, complete tool groups, and all image, byte, and request
limits remain enforced. Tune both action and summary budgets for the selected
model; token estimates do not guarantee the provider's actual context limit.
If the proxy requires `max_tokens`, adjust the generated model recipe before
publishing. For tool choice, use the explicit controls described above; the
template does not silently switch harnesses.

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
provider default sampling temperature. The historical native task starts every model with
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
- `AGENTARK_KAGGLE_BACKEND`, `native` or `litellm`; the shared template defaults to native and the LiteLLM notebook defaults to LiteLLM
- `AGENTARK_KAGGLE_LITELLM_API`, `responses` (LiteLLM Kaggle default) or `chat_completions`
- `AGENTARK_KAGGLE_TOOL_CHOICE`, `auto` or `required`; defaults to auto for LiteLLM and required for native
- `AGENTARK_KAGGLE_TEMPERATURE`, `default`, `null`, `none`, or a finite number; default omits temperature for LiteLLM and uses `1.0` for native
- `AGENTARK_KAGGLE_MAX_COMPLETION_TOKENS`, default `30000`
- `AGENTARK_KAGGLE_REQUEST_TIMEOUT_S`, default `600`
- `AGENTARK_KAGGLE_FULL_EVAL=1`, force a full AgentArk evaluation outside
  Kaggle `BENCHMARK_MODE=RUN`
- `AGENTARK_KAGGLE_PYTHON_VERSION`, default `3.10.12`; must satisfy the pinned wrapper's `>=3.10.1,<3.10.13` requirement
- `AGENTARK_KAGGLE_PYTHON_BIN`, optional explicit compatible Python executable
- `AGENTARK_KAGGLE_WORK_ROOT`, default `/tmp/agentark_kaggle_work`
- `AGENTARK_KAGGLE_SKIP_APT=1`, for images that already contain Python 3.10, git, and Xvfb
