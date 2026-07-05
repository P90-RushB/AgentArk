# AgentArk

AgentArk is a Unity ML-Agents based environment stack for evaluating API
models and training RL agents in multimodal tasks. The Unity runtime loads task
mods at reset time. Agents receive visual observations plus task text, and their
actions are either structured tool calls or generated C# code that the Unity
side compiles and runs with Roslyn.

This repository provides the Python package for runtime control, model
evaluation, replay, environment serving, and RL training integration. Public
runtime builds, task mods, and example replay/evaluation records are available
at [P90-RushB/AgentArk on Hugging Face](https://huggingface.co/datasets/P90-RushB/AgentArk).

<p align="center">
  <img src="docs/figures/agentark-figure1.png" alt="AgentArk overview" width="900">
</p>

## Documentation Map

- Environment setup: [docs/setup.md](docs/setup.md)
- Colab tutorials: [docs/tutorials](docs/tutorials)
- Model evaluation and replay: [docs/evaluation-guide.md](docs/evaluation-guide.md)
- RL training with verl: [docs/rl-training.md](docs/rl-training.md)
- Runtime sandbox details: [docs/runtime-sandbox-migration.md](docs/runtime-sandbox-migration.md)

To browse the currently released task set before downloading the runtime, visit
[AgentArk Hub](https://p90-rushb.github.io/agentark-hub/). It is the public task
catalog and aggregate leaderboard site for AgentArk tasks, with task pages,
preview media, scoreboards, and links back to released artifacts.

## 1. Environment Setup

AgentArk currently uses Python 3.10.12 or an earlier Python 3.10 patch version
for the runtime wrapper, evaluation, replay, and env server processes. Python
3.10.12 is recommended.

```bash
git clone https://github.com/P90-RushB/AgentArk.git
cd AgentArk
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

On Windows PowerShell:

```powershell
git clone https://github.com/P90-RushB/AgentArk.git
cd AgentArk
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

Download a matching AgentArk runtime from Hugging Face. Current release:
`env-1.0.1`. The packaged runtime includes 32 starter tasks.

```bash
# Linux
conda run -n hf hf download P90-RushB/AgentArk \
  --type dataset \
  --include artifacts/envs/1.0.1/linux/AgentArk-env-1.0.1-linux.zip \
  --local-dir downloads/agentark-assets

# Windows
conda run -n hf hf download P90-RushB/AgentArk \
  --type dataset \
  --include artifacts/envs/1.0.1/windows/AgentArk-env-1.0.1-windows.zip \
  --local-dir downloads/agentark-assets
```

You can also download directly from:

- Linux: `https://huggingface.co/datasets/P90-RushB/AgentArk/resolve/main/artifacts/envs/1.0.1/linux/AgentArk-env-1.0.1-linux.zip`
- Windows: `https://huggingface.co/datasets/P90-RushB/AgentArk/resolve/main/artifacts/envs/1.0.1/windows/AgentArk-env-1.0.1-windows.zip`

Copy `.env.example` to `.env` and point it at the extracted runtime:

```bash
cp .env.example .env
```

Important variables:

```dotenv
AGENTARK_ENV_PATH=/path/to/AgentArk-env-1.0.1-linux/AgentArk.x86_64
AGENTARK_MOD_PATH=/path/to/AgentArk-env-1.0.1-linux/AgentArk_Data/Resources/Mods
AGENTARK_TASK_STORE_PATH=${AGENTARK_MOD_PATH}/all_tasks
AGENTARK_RUNTIME_TEMPLATE_ROOT=/path/to/AgentArk-env-1.0.1-linux
AGENTARK_RUNTIME_POOL_ROOT=/tmp/agentark_runtime_pool
MLAGENTS_PYTHON_BIN=/path/to/python3.10
```

On Windows, `AGENTARK_ENV_PATH` may point to either the runtime directory or
the `.exe`; `AGENTARK_MOD_PATH` should point to
`AgentArk_Data\Resources\Mods`.

For headless Linux servers, install Xvfb before running visual tasks:

```bash
sudo apt update
sudo apt install -y xvfb
```

See [docs/setup.md](docs/setup.md) for platform-specific extraction commands,
task mod installation, and a smoke test.

## 2. Model Evaluation

Set an API key for the provider used by your eval config. For example, if you
use the default OpenRouter-style example config:

```bash
export OPENROUTER_API_KEY=...
```

For other OpenAI-compatible providers, change `models[*].provider`,
`models[*].base_url`, and `models[*].api_key_env` or set `models[*].api_key`
directly in your local config.

Edit [config/ark_env/eval_seed1.example.yaml](config/ark_env/eval_seed1.example.yaml)
so `eval.cases[*].task_name` exists in your runtime and `models[*]` matches your
OpenAI-compatible provider. Then run:

```bash
python -m agent_ark.ark_eval.run_api_agent \
  --config config/ark_env/eval_seed1.example.yaml
```

For multiple seeds:

```bash
python -m agent_ark.ark_eval.run_api_agent \
  --config config/ark_env/eval_seeds_1_n.example.yaml
```

For parallel model/seed evaluation across multiple isolated Unity runtimes:

```bash
python -m agent_ark.ark_eval.run_parallel_api_eval \
  --config config/ark_env/parallel_api_eval.example.yaml
```

When `eval.max_parallel_envs > 1`, keep
`env_cfg.runtime_sandbox.enabled: true`. Each worker gets a private writable
runtime while sharing task assets through `Mods/all_tasks`.

Saved JSONL records can be replayed without calling a model:

```bash
python -m agent_ark.ark_eval.run_replay \
  --config config/ark_env/replay.example.yaml \
  --records tmp/DelayTrain_seed1_5.jsonl \
  --index 0
```

AgentArk Hub is also useful after an eval run: it shows the public task catalog
and aggregate scoreboards, while this repository stores your local JSONL results.
The evaluation guide covers model configs, browser visualization, human
interaction, scoring fields, trajectory save/load, and replay:
[docs/evaluation-guide.md](docs/evaluation-guide.md).

## 3. RL Training

AgentArk provides the env server side. The verl GRPO integration lives in the
public fork [P90-RushB/verl](https://github.com/P90-RushB/verl), branch
`agentark_rl`, under
[`agentark_recipe/agentark_env_agent`](https://github.com/P90-RushB/verl/tree/agentark_rl/agentark_recipe/agentark_env_agent).

The runtime wrapper and env server run in the AgentArk Python 3.10.12
environment. The verl trainer can run in its own Python environment because it
talks to AgentArk over HTTP.

Start the env server from the AgentArk checkout:

```bash
bash scripts/run_env_server_mlagents.sh
```

Warm up the pool:

```bash
python -m agent_ark.ark_env.serving.warmup_envs \
  --config config/ark_env/agentark_runtime_config.example.yaml \
  --output tmp/warmup_snapshot.json
```

Check health:

```bash
curl http://127.0.0.1:18080/health
curl http://127.0.0.1:18080/v1/envs
```

Then follow the verl integration guide to generate the dataset and launch GRPO
training:
[docs/rl-training.md](docs/rl-training.md).

## Package Layout

- `agent_ark.ark_env`: Unity runtime lifecycle, task reset/step protocol,
  runtime sandboxing, env server, warmup, and HTTP client utilities.
- `agent_ark.ark_eval`: API model evaluation, parallel evaluation, replay, and
  trajectory save/load.
- `agent_ark.ark_rl`: reserved namespace for future in-package RL adapters; the
  current RL training implementation is in the verl fork.
- `agent_ark.interaction`: local browser viewer and human-interaction hooks.

## Licensing

The Python package in this repository is Apache-2.0. Runtime builds, task mods,
and records on Hugging Face are distributed under the license stated on the
dataset card, currently CC BY-NC 4.0 unless otherwise noted.
