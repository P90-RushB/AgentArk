# AgentArk

English | [简体中文](README.zh-CN.md)

<p align="center">
  <img src="docs/figures/agentark-task-diversity.png" alt="A growing gallery of diverse AgentArk tasks supported by one runtime for evaluation and reinforcement learning" width="900">
</p>

AgentArk is an open environment framework for multimodal agents: models can see
interactive tasks, write actions as code or tool calls, receive verifiable
feedback, and improve through evaluation, replay, or RL.

The goal is not to freeze one benchmark. AgentArk is built as infrastructure for
continuously growing interactive tasks. Its base environment can load arbitrary
task mods, while each mod defines its own scene, prompt, observations, actions,
scoring rules, and termination conditions. Coding agents can help turn new task
ideas into verified mods; the same tasks can then be used for multimodal model
evaluation, trace replay, and reinforcement learning.

This repository provides the Python package for runtime control, model
evaluation, replay, environment serving, and RL training integration.

## What AgentArk Enables

<p align="center">
  <img src="docs/figures/agentark-figure1.png" alt="AgentArk architecture and workflow overview" width="900">
</p>

- **Task scaling with coding agents.** New environments are packaged as task
  mods, so designers, builders, and reviewers can expand the task library
  without changing the core runtime.
- **Multimodal task evaluation.** Models interact with visual and textual state,
  receive score and error feedback, and produce replayable traces for analysis.
- **Multimodal agent training.** The same runtime and task definitions can be
  served over HTTP for RL frameworks, including ms-swift and verl GRPO
  integrations.
- **A broad task surface.** AgentArk is designed for 2D and 3D scenes, physics
  calibration, timing control, path planning, video-level observation,
  mini-games, GUI-like tasks, and future task families that can be expressed as
  loadable mods with verifiable scoring.

## Try AgentArk First

You do not need to install the local runtime before seeing what AgentArk can do.
Start with the Hub and the Colab tutorials:

| Entry | What it is for |
| --- | --- |
| [AgentArk Hub](https://p90-rushb.github.io/agentark-hub/) | Browse released tasks, preview media, public scoreboards, model results, and artifact links. |
| [AgentArk Bench on Kaggle](https://www.kaggle.com/benchmarks/xunyiljg/agentark-bench) | Run and compare a growing selection of AgentArk tasks on Kaggle Benchmarks. |
| [01_human_play_tutorial.ipynb](https://colab.research.google.com/drive/1OdGgcjtNUO5V4W935Qzm1760mO5V_vF1?usp=drive_link) | Play and debug AgentArk tasks manually from Colab. |
| [02_model_replay_tutorial.ipynb](https://colab.research.google.com/drive/12rypa1bzmtErXMZ1GJAI8qGzCYAfViQI?usp=drive_link) | Replay saved model actions without calling a model API again. |
| [03_online_evaluation_tutorial.ipynb](https://colab.research.google.com/drive/1hP1OxjbboxEa5rvySwo5UWLT_Wn-PxsK?usp=drive_link) | Run online API evaluation against AgentArk tasks. |
| [04_rl_training_tutorial.ipynb](https://colab.research.google.com/drive/1ktAtXJLyi99FteZpdwnBcF6AiCSvOn4i?usp=drive_link) | Launch the RL training workflow around the AgentArk env server. |
| [Hugging Face artifacts](https://huggingface.co/datasets/P90-RushB/AgentArk) | Download runtime builds, task mods, replay records, and registries. |

Kaggle evaluations use its OpenAI-compatible Model Proxy. Their leaderboard
runs use `temperature: 1.0` when accepted, or omit the parameter for models that
do not allow it. The original local scoreboards on AgentArk Hub were evaluated
with `temperature: 0.0`, so results from the two sites use different settings
and should not be compared as identical runs.

## Documentation Map

- Environment setup: [docs/setup.md](docs/setup.md)
- System paper: [docs/paper/AgentArk.pdf](docs/paper/AgentArk.pdf)
- Colab tutorials: [docs/tutorials](docs/tutorials)
- Model evaluation and replay: [docs/evaluation-guide.md](docs/evaluation-guide.md)
- RL training with ms-swift or verl: [docs/rl-training.md](docs/rl-training.md)
- Runtime sandbox details: [docs/runtime-sandbox-migration.md](docs/runtime-sandbox-migration.md)

## 1. Setup

AgentArk local evaluation, replay, and env serving use the Python package in
this repository plus a matching packaged Unity runtime from Hugging Face.
Current runtime release: `env-1.0.1`, with 32 starter tasks.

See [docs/setup.md](docs/setup.md) for installation, runtime download, local
path configuration, and smoke-test instructions.

## 2. Model Evaluation

For OpenAI-compatible HTTP providers, set an API key for the provider used by
your eval config. For example, if you use the default OpenRouter-style example
config:

```bash
export OPENROUTER_API_KEY=...
```

For other OpenAI-compatible HTTP providers, change `models[*].provider`,
`models[*].base_url`, and `models[*].api_key_env` or set `models[*].api_key`
directly in your local config.

Codex SDK evaluation is also supported through `provider: codex`. See
[docs/evaluation-guide.md](docs/evaluation-guide.md) for the Codex install,
model config, and message-context settings.

Edit [config/ark_env/eval_seed1.example.yaml](config/ark_env/eval_seed1.example.yaml)
so `eval.cases[*].task_name` exists in your runtime and `models[*]` matches your
selected provider. Then run:

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

AgentArk serves its Unity runtime pool over HTTP for multi-turn, multimodal RL.
The runtime wrapper and Env Server use the AgentArk Python 3.10.12 environment,
while each trainer keeps its own Python environment and dependency stack.

Two GRPO integrations are available:

- [ms-swift](integrations/ms_swift/README.md), maintained in this repository
  with a Chinese runbook.
- [VERL](integrations/verl/README.md), with its trainer-side implementation in
  the public `agentark_rl` fork.

Start with the [RL integrations index](integrations/README.md) to choose a
framework and follow its end-to-end runbook. Shared architecture, grouping and
task-selection semantics are described in the
[RL training guide](docs/rl-training.md).

## Future Development

The long-term goal of AgentArk is model-environment co-evolution: agents find
their own capability gaps, propose new tasks, implement and verify task modules,
train on those environments, and then generate harder tasks from their failures.

Near-term development will focus on:

- **1k+ task scale in 2026.** Grow the public task store from the current
  starter suite to more than one thousand reproducible, trainable task mods.
- **Dynamic curriculum.** Select tasks based on model success rates, error
  types, task parameters, and capability coverage.
- **Long-horizon memory.** Compress observations, actions, scores, and error
  analysis for tasks with long interaction histories.
- **Richer environment sources.** Combine generated assets, 3D generation, and
  world models with AgentArk's verifiable task logic.
- **Stronger Hub artifacts.** Improve task/runtime versioning, trace artifacts,
  download links, and public model reports.

## Package Layout

- `agent_ark.ark_env`: Unity runtime lifecycle, task reset/step protocol,
  runtime sandboxing, env server, warmup, and HTTP client utilities.
- `agent_ark.ark_eval`: API model evaluation, parallel evaluation, replay, and
  trajectory save/load.
- `integrations`: framework-specific RL adapters and runbooks; the VERL
  trainer-side recipe remains in its public fork.
- `agent_ark.interaction`: local browser viewer and human-interaction hooks.

## Licensing

The Python package in this repository is Apache-2.0. Runtime builds, task mods,
and records on Hugging Face are distributed under the license stated on the
dataset card, currently CC BY-NC 4.0 unless otherwise noted.
