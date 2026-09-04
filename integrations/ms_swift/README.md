# AgentArk × ms-swift GRPO

English | [简体中文](README.zh-CN.md)

This directory connects AgentArk environments to ms-swift GRPO. The trainer
generates code or tool calls, AgentArk executes them in Unity, and the task returns
the observations and reward used to continue or train the trajectory.

The integration supports multimodal, multi-turn rollouts and concurrent Unity
runtimes. It does not bind AgentArk to one model or one task.

## Start here

If this is your first AgentArk training run, follow the
[Snake training tutorial](tutorial/README.md). It is the executable, end-to-end
example for this integration: preparing Snake, configuring Unity runtimes,
installing official ms-swift, running a one-step smoke test, and starting a longer
run.

This README intentionally does not repeat those commands. Use these documents for
the corresponding purpose:

| Goal | Document |
| --- | --- |
| Run the complete training example | [Snake training tutorial](tutorial/README.md) |
| Install AgentArk and pass an evaluation first | [AgentArk setup](../../docs/setup.md) |
| Understand rollout, token, ticket, and lease behavior | [Architecture](ARCHITECTURE.md) |
| Understand the repository-wide RL interfaces | [RL training overview](../../docs/rl-training.md) |
| Automate the tutorial with a coding agent | [Tutorial execution guide](tutorial/SKILL.md) |

The tutorial is the source of truth for concrete shell commands and a known-good
configuration. This README is an orientation guide for adapting that example.

## How the integration fits together

```text
ms-swift trainer
    │  ticket / generated action
    ▼
AgentArk Swift adapter ──HTTP──► AgentArk Env Server
                                  │
                                  ├── Unity runtime 1
                                  ├── Unity runtime 2
                                  └── Unity runtime N
    ▲                                  │
    └──── messages, images, reward ────┘
```

The Swift trainer and AgentArk Env Server run in separate Python environments.
The adapter is installed in the trainer environment and talks to the Server over
protocol v2. The Server owns the Unity processes and leases one runtime to each
active trajectory.

Five terms are useful when reading the tutorial and logs:

| Term | Meaning |
| --- | --- |
| **Runtime** | One independent Unity environment process. |
| **Trajectory** | One model-environment conversation, possibly with multiple turns. |
| **GRPO group** | Sibling trajectories that start from the same task and seed. |
| **Ticket** | A lightweight dataset row that identifies one GRPO group; the real prompt and images arrive from Unity after reset. |
| **Lease** | The Server's temporary assignment of one runtime to one trajectory. |

## Adapting the Snake example

First run the tutorial's one-step smoke test without changing its training topology.
That separates installation and environment problems from experiment-design
problems. Once it passes, change one dimension at a time.

### Use another model

`AGENTARK_MODEL` may be a local model directory or a model ID supported by the
installed Swift stack. The model must:

- have a compatible ms-swift template and processor;
- support the task's text and visual messages;
- work with the selected vLLM or Transformers rollout backend; and
- produce the `<code>` or tool-call action format requested by the task prompt.

Model changes commonly require new context-length, dtype, tensor-parallel, and GPU
memory settings. LoRA is the lower-memory starting point; full-parameter training
needs substantially more GPU memory and checkpoint storage.

### Use another task

The Unity package, runtime configuration, and optional `AGENTARK_TASK_NAME` must all
refer to an available task. With no fixed task name, AgentArk can select tasks per
group. Within a GRPO group, sibling trajectories share the task and initial seed but
use different Unity runtimes.

### Change training scale

The number of idle Unity runtimes must cover Swift's generation batch. Increasing
the trainer batch, process count, or gradient accumulation can therefore require a
larger runtime pool. `num_generations` controls the number of sibling trajectories
per group and must divide the generation batch.

Keep `warmup.num_envs` and `env_cfg.runtime_sandbox.pool_size` aligned in the runtime
configuration. See the [architecture document](ARCHITECTURE.md) for ticket and
runtime capacity details before scaling beyond the tutorial.

### Resume a run

A resumable experiment consists of more than a checkpoint. Preserve the run ID,
ticket dataset, output directory, and checkpoint together. Reusing the same tickets
keeps group identity and seed derivation stable. The
[architecture document](ARCHITECTURE.md) explains why ticket identity matters.

## Configuration entry points

You normally edit or invoke only these files:

| Path | Purpose |
| --- | --- |
| [`tutorial/README.md`](tutorial/README.md) | Complete commands and the tested Snake workflow. |
| [`configs/agentark_grpo.env.example`](configs/agentark_grpo.env.example) | Generic model, trainer, rollout, and output settings. Copy values into a machine-local configuration; do not commit local paths. |
| [`tutorial/config/agentark_runtime_config.snake.example.yaml`](tutorial/config/agentark_runtime_config.snake.example.yaml) | Runtime-pool template used by the Snake example. |
| [`tutorial/config/deepspeed_zero2_adafactor.json`](tutorial/config/deepspeed_zero2_adafactor.json) | ZeRO-2 configuration used by the verified long-sequence full-parameter Snake run, without CPU optimizer offload. |
| [`scripts/run_agentark_server.sh`](scripts/run_agentark_server.sh) | Starts the AgentArk Env Server. |
| [`scripts/run_agentark_grpo.sh`](scripts/run_agentark_grpo.sh) | Validates capacity and launches `swift rlhf`. |
| [`scripts/smoke_agentark_unity.sh`](scripts/smoke_agentark_unity.sh) | Checks two concurrent Unity trajectories and lease cleanup. |

The launcher derives and validates ticket capacity from the training settings, so
prefer its `AGENTARK_*` inputs over adding unrelated Swift flags. The environment
template documents the generic options; the tutorial provides one complete set of
values that was actually used together.

## Troubleshooting direction

- **The Server is unhealthy:** confirm that it is still running and that the trainer
  uses the same URL and protocol version.
- **There are not enough idle runtimes:** stop the run, enlarge the final runtime
  configuration, restart the Server, and warm that same pool.
- **Unity reset times out:** check the Unity build, task/Mods content, sandbox paths,
  Xvfb, CPU, and host memory.
- **CUDA OOM:** reduce the training or generation batch, sequence lengths, or vLLM
  memory share. For full training, also reconsider gradient checkpointing and tensor
  parallelism.
- **A visual task returns no image:** verify its observation configuration. The smoke
  test may allow no image only for intentionally text-only tasks.
- **HTTP 409/410 or stale leases:** start a new trajectory. After an unclean trainer
  exit, the Server reclaims leases when their TTL expires.

The [tutorial troubleshooting section](tutorial/README.md#11-troubleshooting) covers
the errors encountered while running the example. The
[architecture document](ARCHITECTURE.md) explains lifecycle and recovery behavior.

## Compatibility and safety

- The adapter and rollout cleanup currently accept ms-swift `4.4.1`, `4.5.0.dev0`,
  and `4.6.0.dev0`. Use the official
  [modelscope/ms-swift repository](https://github.com/modelscope/ms-swift); the
  token-consistency fix required for AgentArk's agentic rollouts was merged upstream
  in [PR #10012](https://github.com/modelscope/ms-swift/pull/10012).
- The validated deployment is Linux with NVIDIA CUDA and single-host colocated
  rollout. Multi-node training and vLLM server mode require additional routing and
  capacity work.
- ms-swift 4.4.1 `async_generate` is not compatible with this multi-turn scheduler.
  The provided launcher uses concurrent environments within a batch and synchronous
  rollout/update phases.
- The Env Server defaults to `127.0.0.1` and does not provide authentication or TLS.
  Use a trusted network, firewall, and authenticated proxy for remote deployment.
- Unity/Roslyn may execute model-generated code or tool actions. Run training tasks
  only inside appropriately isolated environments.
