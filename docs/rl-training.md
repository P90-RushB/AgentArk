# RL Training

English | [简体中文](rl-training.zh-CN.md)

AgentArk provides the Unity environment server. The current GRPO training
integration lives in the public verl fork:

`https://github.com/P90-RushB/verl/tree/agentark_rl/agentark_recipe/agentark_env_agent`

The design deliberately separates Python environments:

- AgentArk env server: Python 3.10.12, this repository.
- verl training process: the Python version and dependency stack required by
  verl.

The trainer does not need to import the `agent_ark` package. It talks to the env
server over HTTP.

## 1. Prepare AgentArk

Follow [docs/setup.md](setup.md), including runtime download, `.env`, and Xvfb
on headless Linux.

For training, keep runtime sandboxing enabled. The example config is:

`config/ark_env/agentark_runtime_config.example.yaml`

Size the warmup pool for the maximum number of concurrent rollout environments.
For example, if `TRAIN_BATCH_SIZE=2` and `ROLLOUT_N=8`, warm up at least 16
envs.

## 2. Start The Env Server

From this repository:

```bash
bash scripts/run_env_server_mlagents.sh
```

The script loads `.env`, sets `PYTHONPATH=src`, and starts:

```bash
python -m agent_ark.ark_env.serving.run_server --host 127.0.0.1 --port 18080
```

Warm up envs:

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

## 3. Install The verl Integration

Clone the public verl fork and switch to the AgentArk branch:

```bash
git clone https://github.com/P90-RushB/verl.git
cd verl
git switch agentark_rl
```

The AgentArk integration is under:

```text
agentark_recipe/agentark_env_agent/
```

Use that directory's README as the source of truth for verl-side environment
setup, configs, dataset generation, and training launch details. The older
`recipe/agentark_env_agent` path is not used in this fork.

## 4. Generate Dataset Rows

Run this in the verl environment:

```bash
DATA_DIR=/path/to/agentark_data

python agentark_recipe/agentark_env_agent/generate_agentark_dataset.py \
  --local-save-dir "${DATA_DIR}" \
  --num-train 1000 \
  --num-test 200 \
  --seed 1234
```

By default, dataset rows leave `task_name=None`. The env server then maps the
GRPO group id (`uid`) deterministically to a task and seed. All samples in the
same GRPO group use the same task and seed, while different groups spread across
the configured task list.

To pin one task:

```bash
python agentark_recipe/agentark_env_agent/generate_agentark_dataset.py \
  --local-save-dir "${DATA_DIR}" \
  --task-name Snake \
  --num-train 1000 \
  --num-test 200
```

## 5. Launch GRPO

Example:

```bash
PYTHON=python \
MODEL_PATH=/path/to/model \
DATA_DIR=/path/to/agentark_data \
NNODES=1 NGPUS_PER_NODE=8 \
ROLLOUT_TP=8 ROLLOUT_N=16 \
TRAIN_BATCH_SIZE=1 PPO_MINI_BATCH_SIZE=1 \
AGENT_NUM_WORKERS=8 \
TOTAL_EPOCHS=1 \
TOTAL_TRAINING_STEPS=1000 \
SAVE_FREQ=250 TEST_FREQ=-1 \
TRAINER_LOGGER='["console","tensorboard"]' \
TENSORBOARD_DIR=/path/to/tensorboard/agentark \
bash agentark_recipe/agentark_env_agent/run_qwen3_5_9b_agentark_env_grpo.sh
```

Common knobs:

- `TOTAL_TRAINING_STEPS`: set to a positive integer for a fixed run, or `-1`
  for a full epoch.
- `ROLLOUT_N`: number of samples per prompt / GRPO group.
- `AGENT_NUM_WORKERS`: number of concurrent agent-loop workers.
- `SAVE_FREQ`: checkpoint interval, or `-1` to disable.
- `TRAINER_LOGGER`: include `tensorboard` to write TensorBoard logs.

Avoid running unrelated GPU occupancy scripts during training; they can starve
Unity reset/step calls and trigger timeouts.

## Task Selection Semantics

Training requests can be pinned or server-managed.

Pinned task:

- The dataset row sets `extra_info.task_name`.
- Optional `group_seed` and `unity_env_id` are forwarded to the server.
- The env resets to exactly that task.

Server-managed task:

- The dataset row leaves `task_name=None`.
- The trainer forwards `uid`.
- `EnvSessionManager` maps `uid -> (task_name, group_seed)` through a
  deterministic `TaskSelector`.
- All samples sharing one `uid` use the same task and seed.

This keeps task curriculum policy on the AgentArk side, so RL framework code can
stay stable while task selection evolves.

## Env Resilience

Long training runs repeatedly reset Unity and execute dynamically compiled code.
The env server includes several guards:

- Hard timeouts for blocking Unity `reset`, `step`, and `close` calls.
- Broken runtimes are discarded and recreated.
- HTTP client retries transient failures with backoff.
- The verl agent loop converts environment failures into valid failed rollouts
  instead of aborting the whole training step.

Use the soak test to stress the env server:

```bash
python -m agent_ark.tools.env_soak_test \
  --workers 6 \
  --rounds 150 \
  --fault-mode gentle
```
