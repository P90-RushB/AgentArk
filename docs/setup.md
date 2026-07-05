# Environment Setup

This guide installs the Python wrapper and a packaged AgentArk Unity runtime.
The Python package lives in this repository. Runtime builds, task mods, and
records live in the Hugging Face dataset:

`https://huggingface.co/datasets/P90-RushB/AgentArk`

The dataset exposes a versioned `manifest.json` plus `registry/*.jsonl`. Prefer
reading those registry files instead of hard-coding directory scans. Current
release: `env-1.0.1`.

## Python

AgentArk runtime, evaluation, replay, and env server processes require Python
3.10.12 or an earlier Python 3.10 patch version. Python 3.10.12 is recommended.

Linux/macOS shell:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

For source-tree debugging without installing the package, set `PYTHONPATH=src`.

## Runtime Download

The current packaged runtime includes 32 starter tasks. Download the runtime for
your OS with the Hugging Face CLI command `hf download`. Install the CLI first
if `hf` is not available.

Linux:

```bash
hf download P90-RushB/AgentArk \
  --type dataset \
  --include artifacts/envs/1.0.1/linux/AgentArk-env-1.0.1-linux.zip \
  --local-dir downloads/agentark-assets
```

Windows PowerShell:

```powershell
hf download P90-RushB/AgentArk `
  --type dataset `
  --include artifacts/envs/1.0.1/windows/AgentArk-env-1.0.1-windows.zip `
  --local-dir downloads/agentark-assets
```

Direct links:

- Linux: `https://huggingface.co/datasets/P90-RushB/AgentArk/resolve/main/artifacts/envs/1.0.1/linux/AgentArk-env-1.0.1-linux.zip`
- Windows: `https://huggingface.co/datasets/P90-RushB/AgentArk/resolve/main/artifacts/envs/1.0.1/windows/AgentArk-env-1.0.1-windows.zip`

Extract the zip. On Linux, make the Unity executable runnable if needed:

```bash
chmod +x /path/to/AgentArk-env-1.0.1-linux/AgentArk.x86_64
```

## Local Paths

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Linux example:

```dotenv
AGENTARK_ENV_PATH=/path/to/AgentArk-env-1.0.1-linux/AgentArk.x86_64
AGENTARK_MOD_PATH=/path/to/AgentArk-env-1.0.1-linux/AgentArk_Data/Resources/Mods
AGENTARK_TASK_STORE_PATH=${AGENTARK_MOD_PATH}/all_tasks
AGENTARK_RUNTIME_TEMPLATE_ROOT=/path/to/AgentArk-env-1.0.1-linux
AGENTARK_RUNTIME_POOL_ROOT=/tmp/agentark_runtime_pool
MLAGENTS_PYTHON_BIN=/path/to/agentark/.venv/bin/python
```

Windows example:

```dotenv
AGENTARK_ENV_PATH=C:\path\to\AgentArk-env-1.0.1-windows
AGENTARK_MOD_PATH=C:\path\to\AgentArk-env-1.0.1-windows\AgentArk_Data\Resources\Mods
AGENTARK_TASK_STORE_PATH=${AGENTARK_MOD_PATH}\all_tasks
AGENTARK_RUNTIME_TEMPLATE_ROOT=C:\path\to\AgentArk-env-1.0.1-windows
AGENTARK_RUNTIME_POOL_ROOT=C:\path\to\agentark-runtime-pool
MLAGENTS_PYTHON_BIN=C:\path\to\AgentArk\.venv\Scripts\python.exe
```

The Python package auto-loads the nearest `.env` from the current directory or
one of its parents. Explicit shell or CI environment variables take precedence.
Set `AGENTARK_AUTO_LOAD_DOTENV=0` to disable auto-loading.

## Headless Linux

Visual observations need a display. On a Linux server without a desktop session,
install Xvfb:

```bash
sudo apt update
sudo apt install -y xvfb
```

Then set `virtual_display: true` in the relevant AgentArk config. The example
server config already enables it under `env_cfg.env_config_overrides`.

## Smoke Test

Start with a single packaged task. `ObjectRotationMatch` is a small task that is
useful for checking runtime startup:

```bash
python -m agent_ark.ark_env.ark_sub_env \
  --task-name ObjectRotationMatch \
  --group-seed 1 \
  --env-id 0 \
  --skip-step
```

If reset succeeds, run one step with a tool call:

```bash
python -m agent_ark.ark_env.ark_sub_env \
  --task-name ObjectRotationMatch \
  --group-seed 1 \
  --env-id 0 \
  --max-steps 1 \
  --action '<tool_call>{"name":"RotateControlled","arguments":{"axis":"Y","degrees":0}}</tool_call>'
```

## Browse Tasks

[AgentArk Hub](https://p90-rushb.github.io/agentark-hub/) is the public browser
for released AgentArk tasks. It provides task cards, task detail pages, preview
media, public aggregate leaderboards, and links back to downloadable artifacts.
Use it to choose a `task_name` before editing evaluation configs.

## Extra Task Mods

The packaged runtime already includes 32 starter tasks. Additional task mod
archives are listed in `registry/tasks.jsonl`, one row per task and platform.

Download the registry:

```bash
hf download P90-RushB/AgentArk \
  --type dataset \
  --include registry/tasks.jsonl \
  --local-dir downloads/agentark-assets
```

Each task archive contains files such as `catalog_*.json`, `cfg/task_config.yaml`,
the task bundle, and task DLLs. Extract a task archive into:

```text
<runtime>/AgentArk_Data/Resources/Mods/all_tasks/<TaskName>/
```

Use the `task_name` value from the registry row when configuring evaluation or
training.

## Runtime Pool

Parallel evaluation and RL training should use a sandboxed runtime pool. Each
worker gets a private writable runtime while sharing task assets through
`Mods/all_tasks`.

You can prepare the pool explicitly:

```bash
python -m agent_ark.tools.prepare_runtime_pool \
  --runtime-platform linux \
  --template-root "$AGENTARK_RUNTIME_TEMPLATE_ROOT" \
  --template-env-path "$AGENTARK_ENV_PATH" \
  --template-mod-path "$AGENTARK_MOD_PATH" \
  --shared-task-store-path "$AGENTARK_TASK_STORE_PATH" \
  --pool-root "$AGENTARK_RUNTIME_POOL_ROOT" \
  --pool-size 8 \
  --link-mode auto
```

Most configs also support `runtime_sandbox.auto_prepare: true`, which prepares
the pool when the launcher starts.
