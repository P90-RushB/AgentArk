---
name: write-kaggle-benchmarks
description: Write, push, run, publish, and manage Kaggle Benchmark tasks using the kaggle CLI and the kaggle-benchmarks Python SDK. Use when the user wants to create or push a benchmark task (optionally with attached Kaggle datasets), run benchmarks against LLM models, check task/run status, stream or fetch execution logs, download results and source notebooks, publish a task to make it public, or troubleshoot benchmark workflows. 
---

# Write Kaggle Benchmarks

## Keywords
Kaggle benchmarks, write a benchmark, benchmark task, kbench, push task, run task.

## Official Resources
- **SDK source & API** — https://github.com/Kaggle/kaggle-benchmarks
- **SDK auto-generated docs** — https://deepwiki.com/Kaggle/kaggle-benchmarks
- **CLI docs** — https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md

## Command Hierarchy
```
kaggle benchmarks (alias: kaggle b)
├── auth              — Fetch Model Proxy credentials
├── init              — Fetch credentials + setup local dev environment
└── tasks (alias: t)  — Manage benchmark tasks
    ├── push          — Upload a task from a .py file
    ├── run           — Run a task against model(s)
    ├── list          — List your benchmark tasks
    ├── status        — Show task details and per-model run status
    ├── download      — Download completed run outputs (and optionally source notebooks)
    ├── log (logs)    — Show execution logs for run(s) (streams live for RUNNING runs)
    ├── publish       — Make a task public (publishes the backing notebook by default)
    ├── models        — List available benchmark models
    └── delete        — Delete a task (not yet supported by server)
```

## Setup
```bash
# Full setup: credentials + .env + example_task.py + kaggle_benchmarks_reference.md
kaggle b init -y

# Credentials only (refresh MODEL_PROXY_* in .env)
kaggle b auth -y
```

Custom paths: `--env-file <FILE>` and `--example-file <FILE>` for init.

### Env vars written by init:
- `MODEL_PROXY_URL`
- `MODEL_PROXY_API_KEY`
- `MODEL_PROXY_EXPIRY_TIME`
- `LLM_DEFAULT`
- `LLM_DEFAULT_EVAL`
- `LLMS_AVAILABLE`

## Core workflow: Init → Write → Validate → Push → Run → Status → Download

### Pacing — check in at every stage
**Do NOT chain the full pipeline.** Treat each numbered step below as a checkpoint:

1. State what you are about to do for the current step (one sentence, including the exact command you intend to run).
2. Wait for the user's go-ahead before executing — including for steps that look "obvious" like `init` or `list`.
3. After the step completes, show the relevant output, then **stop**. Do not auto-advance to the next step.
4. Ask the user how they want to proceed: continue to the next documented step, change parameters, or branch off.

If the user explicitly asks for "the whole pipeline" or "do everything", you may chain, but **summarize the planned chain in advance and ask for one confirmation covering the lot**, instead of skipping the per-step checkpoints silently.

### 0. Init (once per environment, re-run when creds expire)

`init` fetches Model Proxy credentials, writes `.env`, and drops an
`example_task.py` + `kaggle_benchmarks_reference.md` next to it. Every
later step depends on the `MODEL_PROXY_*` vars it writes, so run it
before anything else — and re-run it any time `python task.py` or
`kaggle b t run` fails with an auth error (the API key is short-lived).

```bash
kaggle b init -y                      # first-time setup
kaggle b auth -y                      # creds-only refresh (no scaffolding)
```

### 1. Write a task file
A task file must:
- Import `kaggle_benchmarks as kbench`
- Define at least one function decorated with `@kbench.task(...)`
- Call `.run(kbench.llm)` (or `.evaluate(...)`) on the task function — see Gotchas
- Use `# %%` cell markers (jupytext percent format)

#### Minimal example:
```python
# %%
import kaggle_benchmarks as kbench

# %%
@kbench.task(name="my-test-task")
def my_test_task(llm):
    response = llm.prompt("What is 2 + 2?")
    kbench.assertions.assert_in("4", response, expectation="Should contain 4")

my_test_task.run(kbench.llm)
```

#### LLM resolution precedence (highest → lowest):
1. **Explicit model in code**: `task.run(llm=kbench.llms["gemini-3.5-flash"])`
2. **Default in code**: `task.run(llm=kbench.llm)` (resolves to `LLM_DEFAULT`)
3. **Env vars from .env** (`LLM_DEFAULT`, `LLMS_AVAILABLE`, `MODEL_PROXY_*`)

### 2. Validate locally
Run the task end-to-end before pushing. This catches the silent-no-op gotcha and broken prompts before the push → run → wait → download round-trip.

```bash
kaggle b init -y                     # ensure .env is current
python task.py                       # run the task directly
ls -1 *.run.json                     # confirm a run file was produced
```
If `python task.py` exits cleanly and `*.run.json` appears, the task is safe to push. If validation fails, fix and re-run before proceeding to Step 3.

### 3. Push
```bash
kaggle b t push my-task -f task.py --wait
kaggle b t push my-task -f task.py -d owner/dataset1 -d owner/dataset2   # attach datasets
```
`--wait [TIMEOUT]` blocks until server-side creation finishes (no arg = indefinite). `--poll-interval <SECONDS>` caps the polling interval (default 60s; polling starts at 5s and grows adaptively). Repeat `-d`/`--kaggle-dataset` once per dataset (do **not** space-separate; see the Gotchas).

### 4. Run
```bash
# Interactive picker
kaggle b t run my-task

# Specific model
kaggle b t run my-task -m gemini-3.5-flash

# Multiple models (repeat -m, do NOT space-separate)
kaggle b t run my-task -m gemini-3.5-flash -m claude-haiku-4-5

# Wait for completion
kaggle b t run my-task -m gemini-3.5-flash --wait
```
List available models: `kaggle b t models`.

### 5. Status
```bash
kaggle b t status my-task
kaggle b t status my-task -m gemini-3.5-flash
```
Prints task metadata (slug, version, state, created timestamp, public flag, task URL) and a per-model run table. Errored runs render their final exception line under an `Errors:` section.

### 6. Download
```bash
kaggle b t download my-task                       # all terminal runs
kaggle b t download my-task -o ./results          # custom directory
kaggle b t download my-task -m gemini-3.5-flash
kaggle b t download my-task -s                    # also fetch source notebooks
kaggle b t download my-task -f                    # force re-download (overwrite)
```
Output layout: `<output>/<task>/<version>/<model>/<run_id>/....` Already-downloaded runs are skipped unless `--force`/`-f` is passed. With `--include-source`/`-s`, each run's directory also contains `__notebook__.ipynb` and `__notebook_source__.ipynb` alongside the regular outputs (useful for debugging the kernel session).

### 7. Log
```bash
kaggle b t log my-task                            # logs for every run of the task
kaggle b t log my-task -m gemini-3.5-flash # filter to one model
kaggle b t log my-task -m model-a -m model-b      # multiple models, sequential
```
`RUNNING` runs stream live via SSE; `COMPLETED`/`ERRORED` runs print the persisted log in one shot; `QUEUED` runs print `(No logs available — server returned 404)` and continue.

### 8. Publish
```bash
kaggle b t publish my-task                              # publish task + backing notebook (default)
kaggle b t publish my-task --no-publish-backing-notebook  # publish task only, keep notebook private
```
Publishes both the task and the backing notebook by default. If the task is already public the command is a no-op for the task itself but will still publish the notebook unless `--no-publish-backing-notebook` is passed.

## Quick Recipes
**Reminder**: these are reference snippets, not invocations to chain automatically. Per the "Pacing" section above, run them one at a time with user confirmation between each, unless the user explicitly asks you to chain them.

```bash
# Push → run → download (run one command at a time, confirm between)
kaggle b t push my-task -f task.py --wait
kaggle b t run my-task -m gemini-3.5-flash --wait
kaggle b t download my-task -o ./results

# List tasks, filtered
kaggle b t list --name-regex "^math" --status errored

# Debug an errored run: pull logs first, then download source notebook
kaggle b t log my-task -m gemini-3.5-flash
kaggle b t download my-task -m gemini-3.5-flash -s -f
```

## Gotchas
Most of these are silent failures the agent will not detect on its own — review before generating any task file or CLI invocation.
- **No `.run()` call → silent no-op**. The push will succeed even if the file has no `.run()` (push validation only checks for `@task` decorators). The task will then execute on the server and produce no `.run.json`, so nothing is recorded. Every task function must end with `task_fn.run(kbench.llm)` (or `.evaluate(...)`).
- **`MODEL_PROXY_API_KEY` is short-lived**. If `python task.py` fails with an auth error, re-run `kaggle b auth -y` (or `kaggle b init -y`) to refresh.
- **`init` / `auth` append to the env file**. Loaded via `dotenv` so last-wins makes re-running safe, but the file accumulates duplicate entries over time.
- **Task slug must match a `@task` decorator**. `kaggle b t push <SLUG> -f file.py` fails if `<SLUG>` doesn't match the slugified name of some `@kbench.task(name=...)` (or function name) in the file. Names are normalized: `My Task` → `my-task`, `my_task` → `my-task`.
- **Use bare canonical model slugs** (e.g. `gemini-3.5-flash`, `claude-haiku-4-5`). The CLI auto-normalizes input by stripping any provider prefix (`google/`, `anthropic/`) and replacing `@` with `-`, so `google/gemini-3.5-flash`, `anthropic/claude-haiku-4-5@20251001`, and `claude-haiku-4-5-6@default` all work — but tables, error logs, and download directories always display the canonical form (e.g. `claude-haiku-4-5-20251001`). Standardize on the bare slug everywhere to keep commands, output, and paths consistent.
- **Re-pushing without `-d` detaches previous datasets.** If a prior version of a task had Kaggle datasets attached, re-pushing without repeating the `-d` flags prints a yellow warning to stderr and silently detaches them. Always repeat the full `-d` list on each push.
- **`-f -s` together to backfill source notebooks.** If you originally downloaded without `-s`, a follow-up `download -s` will skip cached runs and print a tip. Re-run with `-f -s` to overwrite cached runs and fetch their source notebooks.
- **`log` is sequential, not interleaved.** When multiple runs are active, the CLI blocks on the first run's stream until it terminates before moving to the next. This prevents log interleaving but means you'll wait on one run at a time.
- **`delete` is not implemented server-side**. The command exists but currently prints `Delete is not supported by the server yet.`
- **Repeated flags, not space-separated**. For multi-value flags (`-m`, `-d`/`--kaggle-dataset`), pass the flag once per value: `-m a -m b`, not `-m a b`. Space-separated form is **not** supported and will error.
- **CLI scope is tasks only, not benchmarks**. A *benchmark* is a curated collection of tasks. The CLI lets you create, push, and run individual tasks, but creating or managing benchmarks (collections) must be done on the Kaggle web UI.

## SDK features beyond the basics

These are higher-leverage `kaggle_benchmarks` patterns from the [task-writing skill](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/skills/kaggle-benchmarks/SKILL.md). Reach for them when the minimal example isn't enough — and consult the upstream skill for full worked code.

### Task definition rules
- **`@kbench.benchmark` is an exact alias** for `@kbench.task`. Use whichever reads better.
- **Return-type annotation is mandatory if the task returns a value.** `def score(llm) -> float:` — without `-> float` / `-> bool` / `-> dict` etc., the run object's `.result` typing breaks.
- **`store_task=False` for sub-tasks.** If a `@kbench.task` is called from inside another task (e.g. a per-row scorer in a dataset eval), set `store_task=False` on the inner one to avoid persisting it as a top-level run.

### Run object
After `run = task_fn.run(...)`:
- `run.passed` — `bool`, `True` if the result *and* every assertion passed.
- `run.result` — the returned value (typed by the task's return annotation).
- `run.assertion_results` — `list[AssertionResult]`, every recorded assertion.

### Reasoning and temperature on `llm.prompt()`
```python
response = llm.prompt("Solve: 127 * 53?", reasoning="high")   # "low" | "medium" | "high"
traces   = kbench.last_reasoning_traces()                     # str | None — may be None if disabled/unsupported
response = llm.prompt("Write a creative story.", temperature=0.7)   # default 0
```
Always null-check `last_reasoning_traces()` before using it.

### Multimodal inputs
```python
from kaggle_benchmarks.content_types import images, videos, audios

# Images: from_url / from_path / from_base64
img = images.from_url("https://example.com/cat.jpg")     # llm.prompt auto-downloads + base64-encodes
img = images.from_path("local/chart.png")                # read + base64 immediately
img = images.from_base64(b64_str, format="png")          # default format="jpeg"

# Videos: YouTube URLs only (no local files, no non-YouTube URLs); select Gemini models only
video = videos.from_url("https://www.youtube.com/watch?v=...")

# Audio
audio = audios.from_path("speech.mp3")

response = llm.prompt("Describe this", image=img)
```
- `llm.prompt(...)` auto-downloads image URLs to base64 (safe for non-URL-capable models). `user.send(...)` passes the raw URL through — use it to test native URL fetching, but it'll fail on models that don't support it.
- Video input is **YouTube-only** and limited to specific Gemini models; local files or non-YouTube URLs will error.

### Branching and isolating chat histories
- `chats.fork("name")` — copy current history into a temporary branch; the original chat is untouched on exit.
- `contexts.enter(chat=...)` — low-level swap of the active chat for fully isolated agent histories. Use this only when `ChatRoom` doesn't fit (e.g. agents that must not see each other).

### Multi-agent: `ChatRoom` + `Participant` (Pattern I.5)
For debate / hidden-role / negotiation flows, prefer `kbench.ChatRoom` over hand-rolled `contexts.enter()` wiring. Each participant sees a perspective-projected transcript; the same `kbench.llm` can back many participants.
```python
room = kbench.ChatRoom(system_prompt="A structured 2-turn debate.")
pro  = room.add_participant(llm, name="Pro", system_prompt="Argue FOR.")
con  = room.add_participant(llm, name="Con", system_prompt="Argue AGAINST.")
pro.reply()
con.reply()
```
See [`docs/chatroom/rooms_walkthrough.md`](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/docs/chatroom/rooms_walkthrough.md) in the kaggle-benchmarks repo for a full walkthrough.

### Custom assertions
- `@assertion_handler()` — decorate a function to register it as a custom assertion (returns/raises an `AssertionResult`). Use `@assertion_handler(raises_assertion_error=True)` to fail loudly on mismatch.
- `kbench.assertions.assert_tool_was_invoked(fn)` — verify the model actually called a registered tool during the prompt.
