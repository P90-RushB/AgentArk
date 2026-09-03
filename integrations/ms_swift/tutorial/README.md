# Training AgentArk Snake with Official ms-swift

English | [简体中文](README.zh-CN.md)

This tutorial gives an end-to-end path that has been run successfully: prepare 16
concurrent Unity runtimes and use official ms-swift to perform eight-GPU,
full-parameter GRPO training of Qwen3.5-9B. The goal is to verify AgentArk's
multimodal rollout, Unity interaction, reward, backward, and checkpoint path. You do
not need to reproduce any particular reward or runtime.

[Snake](https://p90-rushb.github.io/agentark-hub/tasks/snake/) is a 2D grid task
hosted on AgentArk Hub. The agent controls a snake from visual observations, moves
toward food, and avoids collisions with walls and its own body. The rules are simple,
but the task still tests planning with position, body geometry, and future collision
risk. The Hub page provides the task description, Windows and Linux task packages,
and public evaluation results.

The published task uses a `20×20` grid by default. This tutorial temporarily changes
the logical grid to `8×8` to reduce early exploration, produce useful rewards sooner,
and validate the training path with a shorter run. This is not the official Snake
evaluation specification. After integration works, restore the default grid or move
to another task.

If AgentArk is new to you, complete the [one-step GRPO quick start](../README.md)
before scaling to this example. This tutorial assumes that you have used ms-swift and
can run `swift rlhf` on Linux with NVIDIA GPUs. Replace every path, port, and
hardware-dependent value for your machine.

## 1. Working example configuration

The following is one known-good resource configuration, not a benchmark that must be
reproduced exactly:

| Item | Example value |
| --- | --- |
| AgentArk Unity package | `AgentArk-env-1.0.3-linux` |
| ms-swift | Official repository (`4.6.0.dev0` when tested) |
| PyTorch / vLLM | `2.10.0+cu128` / `0.19.0` |
| Task | Snake with an `8×8` logical grid |
| Model | Local BF16 Qwen3.5-9B checkpoint |
| GPU | 8 × NVIDIA H800 80 GB |
| Training | Full parameter, ordinary eight-GPU DDP |
| Optimizer | Adafactor |
| DeepSpeed / ZeRO | Disabled |
| CPU offload | Disabled |
| Rollout | Colocated vLLM, TP=1 |
| vLLM memory utilization | `0.35` |
| vLLM maximum context | `16384` |
| Per-device training batch | `1` |
| Gradient accumulation | `2` |
| Effective optimizer batch | `1 × 8 × 2 = 16` |
| `generation_batch_size` | `16` |
| `num_generations` | `16` |
| Maximum turns per trajectory | `6` |
| Tickets | 600, seeds `1234..1833` |
| Optimizer steps | 600 |
| Unity runtimes | 16 |

The 600-step run is useful for observing reward and stability after the pipeline
works; it is not the minimum requirement. Integration acceptance starts with the
one-step smoke test in section 8. With fewer GPUs or less memory, begin with the LoRA
quick start in the parent README and reduce the model and generation batch.

For this topology:

```text
global micro-batch         = 1 × 8 = 8
effective optimizer batch = 8 × 2 = 16
groups per rollout batch  = 16 / 16 = 1
```

Each rollout batch therefore contains one GRPO group with 16 sibling trajectories.
They share the group UID, task, and seed, but each has a distinct request ID and Unity
lease. The Unity pool follows rollout concurrency and must contain 16 runtimes; eight
runtimes are not enough merely because training uses eight GPUs.

## 2. Prerequisites and paths

Prepare:

- an AgentArk checkout;
- the AgentArk Linux Unity environment package;
- a Python environment that can start AgentArk and ML-Agents;
- an official ms-swift environment that can run `swift rlhf`;
- a local Qwen3.5-9B checkpoint;
- eight approximately 80 GB NVIDIA GPUs for this full-training topology;
- Linux graphics dependencies and Xvfb.

The Server and trainer may use separate environments. The working example used
Python 3.10 for AgentArk and Python 3.12 for ms-swift.

### 2.1 Obtain official ms-swift

The exact token-in/token-out fix from PR
[#10012](https://github.com/modelscope/ms-swift/pull/10012) is upstream, so use the
official repository directly:

```bash
export SWIFT_ROOT=/absolute/path/to/official-ms-swift
git clone https://github.com/modelscope/ms-swift.git "$SWIFT_ROOT"
```

The old temporary fork is no longer needed. Follow the parent
[runbook](../README.md) to install the trainer dependencies and AgentArk adapter.

### 2.2 Define machine-local paths

Replace every value below with an absolute path on the current machine:

```bash
export AGENTARK_ROOT=/absolute/path/to/AgentArk
export UNITY_SOURCE=/absolute/path/to/AgentArk-env-1.0.3-linux
export UNITY_ROOT=/absolute/path/to/AgentArk-env-1.0.3-linux-snake-8x8
export MODEL_PATH=/absolute/path/to/Qwen3.5-9B

export AGENTARK_PYTHON_BIN=/absolute/path/to/agentark-env/bin/python
export SWIFT_PYTHON_BIN=/absolute/path/to/swift-env/bin/python
export SWIFT_BIN=/absolute/path/to/swift-env/bin/swift

export RUN_ROOT=/absolute/path/to/agentark-runs/snake-8x8
export DATASET_PATH="$RUN_ROOT/snake-8x8-600.jsonl"
export AGENTARK_RUNTIME_CONFIG="$AGENTARK_ROOT/integrations/ms_swift/tutorial/config/agentark_runtime_config.snake.local.yaml"
export AGENTARK_SERVER_URL=http://127.0.0.1:28182
```

Create the run directory and confirm that both Python environments import the
intended source trees:

```bash
mkdir -p "$RUN_ROOT"

PYTHONPATH="$AGENTARK_ROOT/src" \
"$AGENTARK_PYTHON_BIN" -c 'import agent_ark; print(agent_ark.__file__)'

PYTHONPATH="$SWIFT_ROOT" \
"$SWIFT_PYTHON_BIN" -c \
  'import importlib.metadata as m, swift; print(m.version("ms-swift")); print(swift.__file__)'

"$SWIFT_BIN" --help >/dev/null
```

The second command must print the active ms-swift version and a source path inside
the official checkout. Do not let global site-packages or stale `PYTHONPATH` entries
override it. Run the remaining commands from the AgentArk root:

```bash
cd "$AGENTARK_ROOT"
```

## 3. Prepare a Snake-only Unity package

Copy the Unity package before changing it. `UNITY_SOURCE` is the original package;
`UNITY_ROOT` is a dedicated training copy. If a dedicated copy already exists, point
`UNITY_ROOT` to it and skip the copy. Finish these changes before starting the Server.

```bash
test -d "$UNITY_SOURCE" \
  && test ! -e "$UNITY_ROOT" \
  && mkdir -p "$(dirname "$UNITY_ROOT")" \
  && cp -a "$UNITY_SOURCE" "$UNITY_ROOT" \
  && chmod +x "$UNITY_ROOT/AgentArk.x86_64"
```

### 3.1 Keep only Snake

The task store is:

```text
$UNITY_ROOT/AgentArk_Data/Resources/Mods/all_tasks
```

To make this package strictly Snake-only, move other tasks into a backup directory
outside `all_tasks`. Run this only against the new copy, never `UNITY_SOURCE`:

```bash
export MODS_DIR="$UNITY_ROOT/AgentArk_Data/Resources/Mods"
export TASK_STORE="$MODS_DIR/all_tasks"
export DISABLED_TASKS="$MODS_DIR/disabled_tasks_backup"

mkdir -p "$DISABLED_TASKS"
for task_dir in "$TASK_STORE"/*; do
  [[ -d "$task_dir" ]] || continue
  [[ "${task_dir##*/}" == "Snake" ]] || mv -- "$task_dir" "$DISABLED_TASKS/"
done
```

Afterward, `all_tasks` should contain only `Snake`. The other tasks remain recoverable
from `disabled_tasks_backup`.

### 3.2 Select Snake as the default task

Update both root configurations so Unity does not request a task that was moved:

```text
$MODS_DIR/config.yaml
$MODS_DIR/config.json
```

The YAML must contain:

```yaml
task_name: Snake
```

Change only the existing `task_name` field in JSON to `Snake`; do not replace the
entire configuration with a one-line example.

### 3.3 Reduce the logical grid to 8×8

Edit `$TASK_STORE/Snake/cfg/task_config.yaml`:

```yaml
task_params:
  initialSize: 3
  gridWidth: 8
  gridHeight: 8
  maxStepsWithoutFood: 2
  foodRestoresLife: false
```

Allow six environment steps:

```yaml
max_attempts: 1
max_steps_per_attempt: 6
```

`gridWidth` and `gridHeight` control game logic, not visual observation resolution.
Keep Snake at `width: 480` and `height: 320`; only the exploration space is reduced.

## 4. Configure 16 Unity runtimes

Copy the Snake template without overwriting an existing local experiment:

```bash
test ! -e "$AGENTARK_RUNTIME_CONFIG" \
  && cp integrations/ms_swift/tutorial/config/agentark_runtime_config.snake.example.yaml \
    "$AGENTARK_RUNTIME_CONFIG"
```

Edit the local YAML and replace at least:

```text
env_cfg.env_path
env_cfg.mod_path
env_cfg.runtime_sandbox.template_root
env_cfg.runtime_sandbox.template_env_path
env_cfg.runtime_sandbox.template_mod_path
env_cfg.runtime_sandbox.shared_task_store_path
env_cfg.runtime_sandbox.pool_root
```

The template already sets Server port `28182`, Unity ports `5200..5215`, 16 warm
runtimes, a sandbox pool of 16, one Unity env per runtime, Xvfb,
`GALLIUM_NUM_THREADS=1`, and protocol-v2 sandbox isolation.

If ports conflict, update the Server URL, YAML, and trainer variables together. Use
a new idle `pool_root`; never reuse a pool owned by another live experiment.

## 5. Generate and validate 600 Snake tickets

Each row represents one GRPO prompt group. Generate 600 rows with seeds 1234 through
1833:

```bash
"$SWIFT_PYTHON_BIN" \
  integrations/ms_swift/scripts/generate_tickets.py \
  --output "$DATASET_PATH" \
  --run-id snake-8x8-600 \
  --count 600 \
  --task-name Snake \
  --group-seed-base 1234
```

The generator refuses to overwrite an existing file. Use a new dataset path or run
ID for a new experiment. Validate capacity before training:

```bash
"$SWIFT_PYTHON_BIN" \
  integrations/ms_swift/scripts/check_ticket_capacity.py \
  --dataset "$DATASET_PATH" \
  --max-steps 600 \
  --per-device-train-batch-size 1 \
  --world-size 8 \
  --gradient-accumulation-steps 2 \
  --num-generations 16 \
  --generation-batch-size 16 \
  --num-iterations 1 \
  --reserve-percent 0
```

The important fields are:

```json
{
  "generation_batch_size": 16,
  "global_train_batch_size": 8,
  "steps_per_generation": 2,
  "generation_calls": 600,
  "groups_per_batch": 1,
  "required_rows": 600,
  "dataset_rows": 600,
  "unique_group_uids": 600,
  "ok": true
}
```

## 6. Start the Server and warm runtimes

### 6.1 Start the Env Server

In terminal 1, keep the following process alive throughout training:

```bash
export AGENTARK_REPO_ROOT="$AGENTARK_ROOT"
export AGENTARK_SERVER_URL
export AGENTARK_RUNTIME_CONFIG

AGENTARK_PYTHON_BIN="$AGENTARK_PYTHON_BIN" \
PYTHONPATH="$AGENTARK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
bash integrations/ms_swift/scripts/run_agentark_server.sh
```

Check health:

```bash
curl -sS "$AGENTARK_SERVER_URL/health"
```

A new Server normally reports `env_count: 0` before warmup.

### 6.2 Warm 16 protocol-v2 runtimes

In terminal 2:

```bash
PYTHONPATH="$AGENTARK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
"$AGENTARK_PYTHON_BIN" \
  -m agent_ark.ark_env.serving.warmup_envs \
  --config "$AGENTARK_RUNTIME_CONFIG" \
  --protocol-version v2 \
  --num-envs 16
```

Validate capacity:

```bash
"$AGENTARK_PYTHON_BIN" \
  integrations/ms_swift/scripts/check_agentark_server.py \
  --server-url "$AGENTARK_SERVER_URL" \
  --protocol-version v2 \
  --required-idle 16
```

Require `healthy: true`, `idle_started_envs: 16`, `in_use_envs: 0`, and
`required_idle: 16`. After changing the task package or runtime YAML, stop the old
Server, choose a new pool path, restart, and warm again. Old health output does not
prove that a new configuration was loaded.

## 7. Set training parameters

In terminal 3:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29583
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SWIFT_SINGLE_DEVICE_MODE

export AGENTARK_REPO_ROOT="$AGENTARK_ROOT"
export AGENTARK_SWIFT_PYTHON="$SWIFT_PYTHON_BIN"
export AGENTARK_SWIFT_BIN="$SWIFT_BIN"
export AGENTARK_MODEL="$MODEL_PATH"
export AGENTARK_TICKET_DATASET="$DATASET_PATH"
export AGENTARK_SERVER_URL
export AGENTARK_RUNTIME_CONFIG
export AGENTARK_PROTOCOL_VERSION=v2
export PYTHONPATH="$SWIFT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Full BF16 training without DeepSpeed, ZeRO, or CPU offload.
export AGENTARK_TUNER_TYPE=full
export AGENTARK_TORCH_DTYPE=bfloat16
export AGENTARK_FREEZE_LLM=false
export AGENTARK_FREEZE_VIT=false
export AGENTARK_FREEZE_ALIGNER=false
export AGENTARK_OPTIM=adafactor
export AGENTARK_LEARNING_RATE=1e-6
export AGENTARK_LR_SCHEDULER_TYPE=cosine
export AGENTARK_GRADIENT_CHECKPOINTING=true

# Eight trainer processes and one 16-sibling GRPO group.
export AGENTARK_WORLD_SIZE=8
export AGENTARK_PER_DEVICE_TRAIN_BATCH_SIZE=1
export AGENTARK_GRADIENT_ACCUMULATION_STEPS=2
export AGENTARK_NUM_GENERATIONS=16
export AGENTARK_GENERATION_BATCH_SIZE=16
export AGENTARK_NUM_ITERATIONS=1

# Six-round multimodal rollout.
export AGENTARK_MAX_TURNS=6
export AGENTARK_MAX_LENGTH=6144
export AGENTARK_MAX_COMPLETION_LENGTH=512
export AGENTARK_ENABLE_THINKING=true
export AGENTARK_ASSISTANT_LOSS_SCOPE=all_turns

# Colocate vLLM with the training model.
export AGENTARK_USE_VLLM=true
export AGENTARK_VLLM_GPU_MEMORY_UTILIZATION=0.35
export AGENTARK_VLLM_TENSOR_PARALLEL_SIZE=1
export AGENTARK_VLLM_MAX_MODEL_LEN=16384
export AGENTARK_VLLM_MM_PROCESSOR_CACHE_GB=0
export AGENTARK_SLEEP_LEVEL=1

export AGENTARK_TICKET_RESERVE_PERCENT=0
export AGENTARK_LOGGING_STEPS=1
export AGENTARK_REPORT_TO=none
```

`vllm_gpu_memory_utilization=0.35` controls only vLLM's allocation; it is not the
total process memory fraction. Full model weights, gradients, optimizer state, and
colocated inference consume substantially more memory.

With `AGENTARK_REPORT_TO=none`, metrics go to `logging.jsonl`. To produce TensorBoard
events from the start, use:

```bash
export AGENTARK_REPORT_TO=tensorboard
```

## 8. Run a one-step smoke test first

Do not begin with 600 steps. Validate rollout, reward, backward, and saving with the
same topology:

```bash
export AGENTARK_RUN_ID=snake-8x8-full-vllm-8gpu-smoke
export AGENTARK_OUTPUT_DIR="$RUN_ROOT/smoke"
export AGENTARK_MAX_STEPS=1
export AGENTARK_SAVE_ONLY_MODEL=true
export AGENTARK_SAVE_STEPS=1
export AGENTARK_SAVE_TOTAL_LIMIT=1

bash integrations/ms_swift/scripts/run_agentark_grpo.sh \
  --completion_length_limit_scope per_round
```

Minimum acceptance criteria:

- process exit code is zero;
- logs reach `global_step/max_steps: 1/1`;
- `grad_norm` is finite;
- reward and `completions.jsonl` are written;
- `checkpoint-1` exists;
- Env Server `active_v2_leases` returns to zero;
- all 16 runtimes are idle again.

No exact reward, mean length, or mean turn count is required. `max_turns=6` is an
upper bound, and Snake may terminate trajectories earlier.

## 9. Start the 600-step run

After smoke succeeds and all leases are released:

```bash
export AGENTARK_RUN_ID=snake-8x8-full-vllm-8gpu-600
export AGENTARK_OUTPUT_DIR="$RUN_ROOT/train-600"
export AGENTARK_MAX_STEPS=600

# Preserve resumable state every 100 steps and keep the latest two checkpoints.
export AGENTARK_SAVE_ONLY_MODEL=false
export AGENTARK_SAVE_STEPS=100
export AGENTARK_SAVE_TOTAL_LIMIT=2

bash integrations/ms_swift/scripts/run_agentark_grpo.sh \
  --completion_length_limit_scope per_round
```

Do not append `--deepspeed` if you intend to use this topology. It uses ordinary
eight-GPU DDP and Adafactor with no ZeRO or CPU offload. Swift creates a timestamped
`v0-*` directory below the output directory containing `logging.jsonl`,
`completions.jsonl`, and periodic checkpoints. `save_total_limit=2` retains only the
latest two.

## 10. Acceptance after training

Inspect the final `trainer_state.json`. A completed 600-step example should record:

```json
{
  "global_step": 600,
  "epoch": 1.0
}
```

Check the Server again:

```bash
"$AGENTARK_PYTHON_BIN" \
  integrations/ms_swift/scripts/check_agentark_server.py \
  --server-url "$AGENTARK_SERVER_URL" \
  --protocol-version v2 \
  --required-idle 16
```

All 16 runtimes should be idle and no runtime should be in use. With 16 siblings per
optimizer step, 600 steps perform 9,600 trajectory rollouts. A GRPO step loss may be
very close to zero; use finite nonzero gradient norms, within-group reward variance,
reward trends, and completion contents to judge whether the training signal is real.

## 11. Troubleshooting

### OOM from an oversized per-device batch

Do not confuse `generation_batch_size=16` with
`per_device_train_batch_size=16`. The example uses batch 1 per GPU, eight GPUs, and
gradient accumulation 2. A single-GPU full-training batch of 16 will generally fail
during backward.

### vLLM conflicts with `device_map`

Confirm that eight GPUs are visible, `NPROC_PER_NODE=8`, `AGENTARK_WORLD_SIZE=8`, and
`SWIFT_SINGLE_DEVICE_MODE` is unset.

### vLLM cannot allocate a cache block

Carefully increase `AGENTARK_VLLM_GPU_MEMORY_UTILIZATION`, shorten context, reduce
model size, or change parallelism. The threshold depends on model, GPU, and vLLM
version and must be established with a one-step smoke test.

### DeepSpeed CPUAdam reports a CUDA mismatch

The system CUDA toolkit, PyTorch CUDA wheel, and DeepSpeed extension are incompatible.
The working topology avoids DeepSpeed and CPU offload. Align those versions and rerun
smoke before enabling ZeRO or offload.

### Triton reports `Python.h: No such file or directory`

Locate the actual headers for the active Python. If they are under
`/usr/local/include/python3.12`, for example:

```bash
export CPATH=/usr/local/include/python3.12
export C_INCLUDE_PATH=/usr/local/include/python3.12
export CPLUS_INCLUDE_PATH=/usr/local/include/python3.12
```

Do not copy this path when the machine uses a different Python installation.

### Runtime shortage or stale leases

- `generation_batch_size=16` requires at least 16 idle runtimes;
- normal terminal and exception paths release leases;
- after OOM, `SIGKILL`, or host failure, protocol v2 reclaims leases after TTL;
- restart the Server and warm a new pool after changing Unity or runtime config.

### Unity fails after other tasks are removed

Confirm that both `Mods/config.yaml` and `Mods/config.json` select `Snake`. Unity may
fail during task-store loading when the root configuration still names a removed task.

## 12. Execution gates for coding agents

An automated agent should advance through these gates instead of starting 600-step
training immediately:

1. read this tutorial, the parent runbook, runtime template, and launcher;
2. inspect the AgentArk checkout, official ms-swift checkout, both Python environments,
   Unity package, model, dataset, and output paths without changing them;
3. confirm that `swift.__file__` resolves to the official checkout and record the
   active package version;
4. modify only a dedicated Unity package copy, never the source package;
5. generate tickets and pass capacity validation;
6. start the Server, warm runtimes, and require 16 idle environments;
7. pass the one-step smoke test, including reward, finite gradients, checkpoint, and
   zero active leases;
8. start long-running training only with explicit user authorization;
9. never overwrite output, kill unrelated processes, or delete checkpoints;
10. after training, inspect trainer state, completions, reward trends, and runtime
    reclamation.

The repository-local [agent execution guide](SKILL.md) contains the complete safety
rules.
