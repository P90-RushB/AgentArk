# AgentArk VERL GRPO Integration

English | [简体中文](README.zh-CN.md)

This directory is AgentArk's versioned entry point for VERL. This repository
provides the Unity runtime, sandbox pool, Env Server, and HTTP v1 API. The
dataset, `AgentArkEnvAgentLoop`, VERL configuration, and trainer launcher remain
in the public
[`P90-RushB/verl` `agentark_rl` branch](https://github.com/P90-RushB/verl/tree/agentark_rl/agentark_recipe/agentark_env_agent).
The VERL Python environment communicates over HTTP and does not import
`agent_ark`.

The local bridge checks the external checkout offline, renders the exact
`env_cfg.yaml` that its agent loop sends, warms the matching v1 pool, and applies
launch guards around the external recipe. The reviewed history baseline and
contract are recorded in [compatibility.json](compatibility.json). This
structural check does not replace a real Unity rollout.

## 1. Prerequisites and checkout

Complete one real Unity evaluation using the [AgentArk setup](../../docs/setup.md)
and its Python 3.10.12 environment. Install VERL, Ray, and the selected rollout
backend in a separate VERL Python environment.

```bash
git clone --branch agentark_rl --single-branch \
  https://github.com/P90-RushB/verl.git /absolute/path/to/verl

export AGENTARK_REPO_ROOT=/absolute/path/to/AgentArk
export VERL_ROOT=/absolute/path/to/verl
cd "$AGENTARK_REPO_ROOT"

python integrations/verl/check_compatibility.py --checkout "$VERL_ROOT"
```

Continue after `[COMPATIBLE]`. Use `--strict` for release/CI provenance checks.
The default mode accepts descendants of the reviewed commit and warns when
review-sensitive files changed.

## 2. Capacity

For the current recipe, peak v1 leases in one generation batch are:

```text
ENV_CONCURRENCY = TRAIN_BATCH_SIZE * ROLLOUT_N
ENV_CONCURRENCY % AGENT_NUM_WORKERS == 0
```

`AGENT_NUM_WORKERS` partitions work among Ray actors; it is not the environment
concurrency limit. A first smoke can use `TRAIN_BATCH_SIZE=1`, `ROLLOUT_N=8`,
and `AGENT_NUM_WORKERS=4`, requiring eight warmed environments.

The external `runtime_sandbox.pool_size` is an initial preparation size, not a
hard limit. With `auto_prepare`, AgentArk expands for new worker indices. The
bridge deliberately preserves that value because changing any semantic
`env_cfg` field creates a different server-pool fingerprint.

## 3. Terminal A: Env Server

Run this in the AgentArk Python environment and keep it running:

```bash
export AGENTARK_REPO_ROOT=/absolute/path/to/AgentArk
cd "$AGENTARK_REPO_ROOT"
./integrations/verl/scripts/run_agentark_server.sh
```

One Env Server process on a host/port manages multiple Unity processes through
the existing runtime sandbox pool. Do not launch multiple uvicorn workers. The
reviewed recipe connects to `127.0.0.1:18080`, so this path is single-node.

## 4. Terminal B: exact v1 warmup

Open another AgentArk Python terminal:

```bash
export AGENTARK_REPO_ROOT=/absolute/path/to/AgentArk
export VERL_ROOT=/absolute/path/to/verl
cd "$AGENTARK_REPO_ROOT"

./integrations/verl/scripts/warmup_agentark_v1.sh \
  --verl-root "$VERL_ROOT" \
  --num-envs 8
```

The script resolves the external recipe's `${oc.env:...}` values with
OmegaConf, writes ignored `tmp/verl_v1_runtime_config.json`, and explicitly
warms protocol v1. This prevents a generic AgentArk config from warming a pool
whose fingerprint the VERL request cannot reuse.

```bash
curl -fsS http://127.0.0.1:18080/health
curl -fsS http://127.0.0.1:18080/v1/envs
```

Verify that the snapshot says `protocol_version="v1"`, health reports
`"ok":true`, and the planned number of entries are `started=true`,
`in_use=false`, and `protocol_namespace="v1"`. Protocol v1 and v2 pool
namespaces are isolated.

## 5. Terminal C: dataset

Use the separate VERL environment from the VERL repository root:

```bash
cd "$VERL_ROOT"
export DATA_DIR=/absolute/path/to/agentark_data

python agentark_recipe/agentark_env_agent/generate_agentark_dataset.py \
  --local-save-dir "$DATA_DIR" \
  --num-train 1000 \
  --num-test 200 \
  --seed 1234
```

`test.parquet` must exist and be non-empty even with `TEST_FREQ=-1`, because the
current trainer still constructs its validation dataloader. For a fixed step
limit, size the train split to at least:

```text
num_train >= TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE
```

In the default server-managed data, VERL generates a `uid` each time a prompt
group is consumed and AgentArk uses it to select the task. The dataset row's
explicit `group_seed` selects the seed. Sibling trajectories share both; when
the same row is consumed in another epoch, its new uid may select another task
while its row seed remains stable.

## 6. Launch training

Use the external recipe
[README and launcher](https://github.com/P90-RushB/verl/tree/agentark_rl/agentark_recipe/agentark_env_agent)
as the source of truth for model, GPU, tensor-parallel, FSDP, and vLLM settings.
The current named launcher targets Qwen3.5-9B/FSDP2/vLLM; validate VERL-side
configuration when changing model or topology.

Invoke it through the repository-local safety wrapper:

```bash
export AGENTARK_REPO_ROOT=/absolute/path/to/AgentArk
export VERL_ROOT=/absolute/path/to/verl
export MODEL_PATH=/absolute/path/to/model
export DATA_DIR=/absolute/path/to/agentark_data
export CKPT_DIR=/absolute/path/to/checkpoints/agentark_verl

cd "$AGENTARK_REPO_ROOT"
PYTHON="/absolute/path/to/verl-env/bin/python" \
NNODES=1 NGPUS_PER_NODE=8 \
ROLLOUT_NAME=vllm ROLLOUT_TP=8 \
TRAIN_BATCH_SIZE=1 ROLLOUT_N=8 AGENT_NUM_WORKERS=4 \
PPO_MINI_BATCH_SIZE=1 \
TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=10 \
SAVE_FREQ=5 TEST_FREQ=-1 \
TRAINER_LOGGER='["console","tensorboard"]' \
TENSORBOARD_DIR=/absolute/path/to/tensorboard/agentark_verl \
./integrations/verl/scripts/run_verl_training.sh \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode=disable \
2>&1 | tee /tmp/agentark_verl_train.log
```

The wrapper enforces the single-node and rollout-partition constraints, checks
both parquet files, imports only the five required `AGENTARK_*` runtime paths,
and prevents unrelated API keys in AgentArk's `.env` from reaching the external
launcher's xtrace. It translates `TOTAL_TRAINING_STEPS=-1` to Hydra `null`, so
VERL derives the step count from the dataloader and epochs; positive limits are
preserved.

Pass `--preflight-only` as the wrapper's first argument to validate the
checkout, datasets, runtime variables, and parameter relationships without
starting training.

A successful first run reports the expected total steps, completes at least one
multimodal rollout, reward computation, and actor update, and does not stop at
global step 1. At the save interval, `CKPT_DIR` should contain a
`global_step_*` checkpoint.

## 7. Interruption and resume

An ordinary exit attempts to release v1 envs in `finally`. A hard trainer
termination can leave v1 leases behind because v1 has neither v2 lease TTL nor
operation-ID replay safety. Restart Terminal A, repeat Terminal B, then relaunch
with the original model/batch/rollout/GPU settings and either:

```bash
./integrations/verl/scripts/run_verl_training.sh \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode=auto
```

or an explicit checkpoint:

```bash
./integrations/verl/scripts/run_verl_training.sh \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CKPT_DIR/global_step_5"
```

Confirm that the log resumes from the selected global step instead of starting
a new step-zero run.

## 8. Ownership and diagnostics

- Unity reset/step, sandbox, pool, task selection, and Env Server issues belong
  to this repository.
- VERL import, Ray, vLLM, FSDP, Hydra, dataset, and trainer issues belong to the
  external fork.
- `check_compatibility.py` is offline and read-only: it never fetches, checks
  out, imports, or executes external code. Exit codes are `0` for structurally
  compatible, `1` for incompatible, and `2` for indeterminate (for example, a
  shallow checkout missing the baseline commit).

See the [RL training guide](../../docs/rl-training.md) for shared architecture,
GRPO grouping semantics, and the v1/v2 comparison.
