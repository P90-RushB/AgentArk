# AgentArk × ms-swift GRPO

English | [简体中文](README.zh-CN.md)

This integration registers AgentArk as an ms-swift Gym environment. It uses the
AgentArk Env Server to schedule concurrent Unity runtimes and sends model-generated
code or tool calls to those runtimes as actions. A trajectory may contain multiple
rounds of text and visual observations, while Unity returns the reward directly.

The adapter is model-agnostic. A model can use the same training path when the
selected ms-swift and vLLM versions can load it, it can consume AgentArk's OpenAI-style
multimodal messages, and it can emit the code or tool-call format requested by the
task prompt.

AgentArk can train across multiple tasks. For a complete single-task Snake example,
see the [Snake training tutorial](tutorial/README.md).

## Quick start

The shortest complete path is:

```text
run an AgentArk evaluation
→ install the Swift trainer
→ create local configuration
→ start the Env Server
→ run the two-runtime Unity smoke test
→ run one GRPO step
```

Run all commands from the AgentArk repository root.

### 1. Verify AgentArk evaluation

Follow [the setup guide](../../docs/setup.md) and complete one real Unity evaluation
before configuring training.

### 2. Install the Swift trainer

The AgentArk Env Server and Swift trainer use separate Python environments. The
adapter has been validated with ms-swift `4.4.1`, `4.5.0.dev0`, and `4.6.0.dev0`.
For a new environment, use the official
[modelscope/ms-swift repository](https://github.com/modelscope/ms-swift). PR
[#10012](https://github.com/modelscope/ms-swift/pull/10012) upstreamed the exact
token-in/token-out fix required by AgentArk's multi-step agentic rollouts.

```bash
export SWIFT_ROOT=/path/to/official-ms-swift
git clone https://github.com/modelscope/ms-swift.git "$SWIFT_ROOT"

python -m pip install -U uv
uv venv .venv-swift --python 3.12
source .venv-swift/bin/activate
uv pip install -e "$SWIFT_ROOT" -e integrations/ms_swift \
  "torch==2.10.0" \
  "vllm==0.19.0" \
  "transformers==5.14.1" \
  "trl==0.29.1" \
  "peft==0.19.1" \
  "accelerate==1.14.0" \
  "datasets==4.8.4" \
  --torch-backend=auto
```

Do not omit `-e "$SWIFT_ROOT"`, or Python may load another Swift installation.
These package versions target Linux, NVIDIA CUDA, and colocated vLLM. If your
hardware requires different Torch or vLLM wheels, keep ms-swift within a version
accepted by the launcher, install a compatible CUDA stack, and rerun the smoke test.

### 3. Create local configuration

Copy the two templates:

```bash
cp config/ark_env/agentark_runtime_config.example.yaml \
  config/ark_env/agentark_runtime_config.local.yaml
cp integrations/ms_swift/configs/agentark_grpo.env.example \
  integrations/ms_swift/configs/agentark_grpo.env.local
```

Both `*.local` files are ignored by Git. At minimum, set these values in
`integrations/ms_swift/configs/agentark_grpo.env.local`:

```bash
AGENTARK_SWIFT_PYTHON="$PWD/.venv-swift/bin/python"
AGENTARK_MODEL=/path/to/local-model
# A Swift-supported model ID is also valid:
# AGENTARK_MODEL=organization/model-name
AGENTARK_PYTHON_BIN=/path/to/agentark-python/bin/python
AGENTARK_RUNTIME_CONFIG="$PWD/config/ark_env/agentark_runtime_config.local.yaml"
```

`AGENTARK_MODEL` may be a local directory or a model ID supported by the installed
Swift version. A local checkpoint is easier for the first run because it separates
download failures from environment-integration failures.

Load the configuration in every server, smoke, and trainer terminal:

```bash
set -a
source integrations/ms_swift/configs/agentark_grpo.env.local
set +a
```

### 4. Start the Env Server

Terminal 1:

```bash
set -a
source integrations/ms_swift/configs/agentark_grpo.env.local
set +a
./integrations/ms_swift/scripts/run_agentark_server.sh
```

Keep this process alive throughout training. One Server process manages multiple
runtime sandboxes and Unity child processes. The script derives its default bind
host and port from `AGENTARK_SERVER_URL`; advanced deployments may override `HOST`
and `PORT`, provided the trainer can still reach the configured URL.

### 5. Run the two-runtime Unity smoke test

Terminal 2:

```bash
set -a
source integrations/ms_swift/configs/agentark_grpo.env.local
set +a
./integrations/ms_swift/scripts/smoke_agentark_unity.sh
```

The smoke test concurrently resets two Unity runtimes and verifies that:

- sibling trajectories receive different `env_id` values;
- one GRPO group receives identical initial messages and, when exposed by the
  Server, the same task and seed;
- visual tasks return an inline image;
- both leases are released at the end.

For a text-only task, set `AGENTARK_SMOKE_ALLOW_NO_IMAGE=1`. To pin the smoke test
to one task and seed:

```bash
AGENTARK_SMOKE_TASK_NAME=Pushbox \
AGENTARK_SMOKE_GROUP_SEED=1234 \
./integrations/ms_swift/scripts/smoke_agentark_unity.sh
```

### 6. Run one GRPO step

The default template uses one host, one optimizer step, `G=2`, and LoRA. It is
intended to validate the full path before scaling up:

```bash
./integrations/ms_swift/scripts/run_agentark_grpo.sh
```

Before loading the model, the launcher:

1. checks Swift, the plugin, runtime configuration, and Env Server;
2. computes the unique GRPO ticket capacity required by the run;
3. generates or validates the ticket dataset;
4. checks that idle Unity capacity covers the generation batch;
5. starts `swift rlhf`.

A successful output directory contains Swift logs, `completions.jsonl`, and a
checkpoint. A one-step smoke test validates Unity interaction, multimodal rollout,
reward collection, GRPO backward, and saving; it does not mean that the model has
learned the task.

## Choosing a model and training mode

### Model requirements

The model must:

- have a usable template and processor in the selected ms-swift version;
- support generation in the selected vLLM version, unless Transformers rollout is used;
- accept the text and visual messages used by the task;
- emit the `<code>` or tool-call action specified by the AgentArk prompt.

The repository regression uses Qwen3.5-0.8B because it fits the test machine. This
is not a restriction on architecture, parameter count, or training method. After
changing models, recalibrate context length, visual tokens, dtype, tensor parallelism,
and memory allocation.

### LoRA

LoRA is the default quick-start mode:

```bash
export AGENTARK_TUNER_TYPE=lora
export AGENTARK_LEARNING_RATE=1e-5
export AGENTARK_LORA_RANK=8
export AGENTARK_LORA_ALPHA=16
```

Append model-specific Swift options to the launcher command when special target
modules are required. Set capacity-related values through `AGENTARK_*` variables so
that preflight and the final Swift command use identical values.

### Full-parameter training

Switch to full training with:

```bash
export AGENTARK_TUNER_TYPE=full
export AGENTARK_LEARNING_RATE=1e-6
export AGENTARK_OPTIM=adafactor
export AGENTARK_GRADIENT_CHECKPOINTING=true
```

Swift exposes independent freeze switches for multimodal components. To train every
component:

```bash
export AGENTARK_FREEZE_LLM=false
export AGENTARK_FREEZE_VIT=false
export AGENTARK_FREEZE_ALIGNER=false
```

Keep `AGENTARK_FREEZE_VIT=true` and `AGENTARK_FREEZE_ALIGNER=true` to train only the
language component. Full training must account for optimizer state, checkpoint
storage, and colocated vLLM memory. `AGENTARK_SAVE_ONLY_MODEL=true` reduces checkpoint
size, but such a checkpoint cannot exactly resume optimizer state.

### Common model variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENTARK_MODEL` | required | Local model directory or Swift model ID |
| `AGENTARK_TUNER_TYPE` | `lora` | `lora` or `full` |
| `AGENTARK_TORCH_DTYPE` | `bfloat16` | Training dtype |
| `AGENTARK_ENABLE_THINKING` | unset | Forwarded to the model template when set |
| `AGENTARK_FREEZE_LLM` | unset | Freeze the language model |
| `AGENTARK_FREEZE_VIT` | unset | Freeze the vision/audio encoder |
| `AGENTARK_FREEZE_ALIGNER` | unset | Freeze the multimodal projector/aligner |
| `AGENTARK_GRADIENT_CHECKPOINTING` | `true` for full | Trade compute for memory |
| `AGENTARK_USE_VLLM` | `true` | Use vLLM; `false` selects Transformers rollout |
| `AGENTARK_VLLM_TENSOR_PARALLEL_SIZE` | `1` | Single-host vLLM tensor parallel size |
| `AGENTARK_VLLM_GPU_MEMORY_UTILIZATION` | `0.30` | Colocated vLLM GPU/KV-cache budget |
| `AGENTARK_VLLM_MM_PROCESSOR_CACHE_GB` | `0` | Multimodal processor cache; `0` disables it |

The launcher explicitly forwards `AGENTARK_VLLM_MM_PROCESSOR_CACHE_GB` as
`--vllm_mm_processor_cache_gb`. Disabling the cache is the validated safe default for
image-heavy, multi-turn AgentArk rollout. With affected vLLM versions, enabling it
can desynchronize P0/P1 cache lifetimes across sleep/offload or subsequent requests
and produce `Expected a cached item for mm_hash=...`.

If a newer vLLM version has been validated, set a nonzero capacity explicitly:

```bash
export AGENTARK_VLLM_MM_PROCESSOR_CACHE_GB=4
```

This is distinct from `AGENTARK_VLLM_GPU_MEMORY_UTILIZATION`: the former caches image
processor or visual features, while the latter controls the vLLM GPU/KV-cache budget.

## Scaling to a real run

### 1. Calculate Unity concurrency

Let `D` be Swift's generation batch:

```text
D = generation_batch_size

when generation_batch_size is omitted:
D = per_device_train_batch_size
  × AGENTARK_WORLD_SIZE
  × gradient_accumulation_steps
```

The required idle Unity runtime count is `D`. `G=num_generations` determines the
number of sibling trajectories in each prompt group. Increasing only `G` does not
increase Unity concurrency while `D` stays constant, but `D % G` must equal zero.

### 2. Expand the runtime pool

Stop the old Server and set both values in the final runtime configuration:

```yaml
warmup:
  num_envs: 8

env_cfg:
  runtime_sandbox:
    pool_size: 8
```

Restart the Server, then warm the same configuration with the AgentArk Python:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
"$AGENTARK_PYTHON_BIN" -m agent_ark.ark_env.serving.warmup_envs \
  --config "$AGENTARK_RUNTIME_CONFIG" \
  --num-envs 8 \
  --protocol-version v2
```

The smoke script prepares only its minimum two runtimes. Prepare the full training
pool with the command above. Keep one final runtime configuration per Server process
to avoid mixing incompatible pools.

### 3. Configure the real run

```bash
export AGENTARK_RUN_ID=run-001
export AGENTARK_GENERATED_DATA_DIR=/persistent/path/agentark-data
export AGENTARK_OUTPUT_DIR=/persistent/path/run-001
export AGENTARK_MAX_STEPS=1000
export AGENTARK_PER_DEVICE_TRAIN_BATCH_SIZE=4
export AGENTARK_GRADIENT_ACCUMULATION_STEPS=2
export AGENTARK_NUM_GENERATIONS=4
export AGENTARK_NUM_ITERATIONS=1

./integrations/ms_swift/scripts/run_agentark_grpo.sh
```

When `AGENTARK_TICKET_DATASET` is unset, the launcher generates enough unique tickets
and adds a 10% reserve by default. More optimizer steps require more tickets. Larger
batches, more trainer processes, or more gradient accumulation usually increase `D`
and therefore require a larger Unity pool.

`AGENTARK_WORLD_SIZE` is the number of local trainer processes and is mapped to
`NPROC_PER_NODE`. Visible GPUs must cover those processes, and
`AGENTARK_VLLM_TENSOR_PARALLEL_SIZE` must divide `AGENTARK_WORLD_SIZE`.

## Task distribution and tickets

Each JSONL row is a GRPO group ticket. The real system, user, and visual messages are
injected after Unity reset. Swift repeats one ticket `G=num_generations` times;
siblings use the same task and seed while leasing distinct Unity runtimes.

By default, the AgentArk task selector chooses a task from `group_uid`. To pin a task:

```bash
export AGENTARK_TASK_NAME=Pushbox
```

The launcher derives a stable, distinct seed for each group. Seeds may also be set
explicitly:

```bash
export AGENTARK_GROUP_SEED=1234
# Or assign base+i to ticket i:
export AGENTARK_GROUP_SEED_BASE=100000
```

Use a new `AGENTARK_RUN_ID` for an independent experiment. Reuse the original ticket
dataset when resuming the same run.

## Assistant policy loss

By default, policy loss covers every assistant round in a trajectory:

```bash
export AGENTARK_ASSISTANT_LOSS_SCOPE=all_turns
```

To train only the final assistant round:

```bash
export AGENTARK_ASSISTANT_LOSS_SCOPE=last_round
```

Environment observations are excluded from policy loss. The adapter returns a
per-token loss mask and preserves the assistant token IDs produced during inference.

## Resume and shutdown

Reuse the checkpoint, run ID, ticket dataset, and output directory together:

```bash
export AGENTARK_RUN_ID=run-001
export AGENTARK_TICKET_DATASET=/persistent/path/agentark-data/run-001.jsonl
export AGENTARK_OUTPUT_DIR=/persistent/path/run-001

./integrations/ms_swift/scripts/run_agentark_grpo.sh \
  --resume_from_checkpoint /persistent/path/run-001/v0-*/checkpoint-N
```

Replace the wildcard with the actual checkpoint path. For a normal shutdown, send
Ctrl-C to the trainer, wait for trajectory cleanup, and then stop the Env Server.
Inspect state with:

```bash
curl -s http://127.0.0.1:18080/health
"$AGENTARK_SWIFT_PYTHON" integrations/ms_swift/scripts/check_agentark_server.py \
  --server-url "$AGENTARK_SERVER_URL" \
  --protocol-version v2
```

`active_v2_leases` should return to zero. If OOM, `SIGKILL`, or a host failure prevents
cleanup, the Server reclaims the runtime after the lease TTL expires.

## Common training variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENTARK_MAX_STEPS` | `1` | Optimizer steps |
| `AGENTARK_PER_DEVICE_TRAIN_BATCH_SIZE` | `2` | Batch per local trainer process |
| `AGENTARK_WORLD_SIZE` | `1` | Local trainer process count |
| `AGENTARK_GRADIENT_ACCUMULATION_STEPS` | `1` | Gradient accumulation |
| `AGENTARK_GENERATION_BATCH_SIZE` | automatic | Explicit Swift generation batch |
| `AGENTARK_NUM_GENERATIONS` | `2` | Sibling trajectories per GRPO group |
| `AGENTARK_NUM_ITERATIONS` | `1` | Policy updates reusing a rollout batch |
| `AGENTARK_MAX_TURNS` | `2` | Maximum assistant rounds per trajectory |
| `AGENTARK_MAX_LENGTH` | `6144` | Maximum Swift training sequence length |
| `AGENTARK_MAX_COMPLETION_LENGTH` | `512` | Completion limit per rollout round |
| `AGENTARK_VLLM_MAX_MODEL_LEN` | sum of lengths | vLLM maximum context |
| `AGENTARK_TICKET_RESERVE_PERCENT` | `10` | Extra ticket capacity |
| `AGENTARK_RUN_ID` | UTC time + PID | Run and ticket identity |
| `AGENTARK_OUTPUT_DIR` | generated runs | Checkpoint output directory |

Image count, visual tokens, text length, and step latency differ greatly between
tasks. Tune lengths, memory, lease TTL, and Unity concurrency from real rollout logs.

## Troubleshooting

- `server is not healthy`: keep the Env Server terminal alive and use the same URL
  in every terminal.
- Insufficient `idle started envs`: enlarge the final sandbox pool, restart the
  Server, and warm the v2 pool again.
- Unity reset timeout: inspect the Unity executable, Mods, task store, sandbox,
  Xvfb, CPU, and memory.
- CUDA OOM: reduce the training or generation batch, context length, or vLLM memory
  budget. Full training may also use gradient checkpointing or more parallelism.
- No image in smoke: fix observation configuration for visual tasks, or set
  `AGENTARK_SMOKE_ALLOW_NO_IMAGE=1` for a text-only task.
- HTTP 409/410: the operation ID or lease is stale; start a new trajectory.
- Swift version rejected: recreate the trainer environment with a supported version.

## Compatibility and security

- The adapter and rollout cleanup are validated with ms-swift `4.4.1`,
  `4.5.0.dev0`, and `4.6.0.dev0`; new agentic runs should use the official repository.
- The validated platform is Linux with NVIDIA CUDA and colocated vLLM.
- The launcher supports single-host multi-GPU configuration and preflight. The
  repository smoke baseline is single-GPU, while the Snake tutorial records a
  complete eight-GPU path. Multi-node training and vLLM server mode need additional
  capacity and routing work.
- The supported multi-turn path uses concurrent environment I/O within a rollout
  batch, while rollout and optimizer batches still alternate synchronously.
- One uvicorn worker manages all leases; this does not limit the number of Unity
  runtimes controlled by that process.
- `scripts/compat/sitecustomize.py` enables a PyTorch causal-conv1d fallback for the
  validated Torch stack. Point `AGENTARK_SWIFT_COMPAT_DIR` to an empty directory only
  after validating the native extension with the selected model.
- Unity/Roslyn executes model-generated code or tool actions. The Server defaults to
  `127.0.0.1` and does not provide authentication or TLS. Use a trusted network,
  firewall, and authenticated proxy for remote deployment.

## Further reading

- Rollout, token, ticket, lease, failure recovery, and VERL comparison:
  [architecture and implementation semantics](ARCHITECTURE.md)
- AgentArk RL overview: [RL training](../../docs/rl-training.md)
- Swift Gym interface:
  [official ms-swift documentation](https://swift.readthedocs.io/en/latest/Instruction/GRPO/DeveloperGuide/gym_env.html)
  (the launcher's version check defines this adapter's concrete compatibility)

After changing or releasing the adapter, run:

```bash
PYTHONPATH=src "$AGENTARK_PYTHON_BIN" -m unittest discover -s tests -q
PYTHONPATH=integrations/ms_swift/src "$AGENTARK_SWIFT_PYTHON" \
  -m unittest discover -s integrations/ms_swift/tests -t integrations/ms_swift -q
```
