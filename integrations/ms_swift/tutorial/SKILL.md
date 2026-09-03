---
name: agentark-snake-grpo
description: Run AgentArk Snake GRPO with official ms-swift and help prepare Python, Unity, tickets, the runtime server, a one-step smoke test, full training, and acceptance checks. Audit real paths, the official checkout, and versions before starting training.
---

# AgentArk Snake GRPO execution skill

English | [简体中文](SKILL.zh-CN.md)

## Goal

Help a user run AgentArk Snake GRPO on any machine that satisfies the dependencies.
Never assume shared storage, `/tmp`, a home directory, ports, GPU count, model path,
or dataset size.

## Execution order

1. Read the Snake tutorial and the repository's AgentArk/ms-swift runbook.
2. Obtain or confirm the real AgentArk, ms-swift, Python, Unity, model, dataset,
   Server, and output paths.
3. Perform read-only inspection before creating training directories.
4. Verify that both Python interpreters execute. Inspect `pyvenv.cfg` and the
   `bin/python` link when a virtual environment was copied between machines.
5. Confirm that `agent_ark.__file__` and `swift.__file__` resolve to the intended
   checkouts, and record the installed package versions.
6. Validate the Unity executable, Mods, task store, runtime configuration, and pool.
7. Generate or validate Snake tickets, including task name, seed range, row count,
   and unique group UIDs.
8. Pass ticket-capacity preflight.
9. Start the Server, warm runtimes, and run the one-step smoke test.
10. Require both trainer output and lease reclamation to pass acceptance.
11. Start long-running training only with explicit user authorization.

## Python environment rules

A Python virtual environment does not contain its base interpreter. Do not assume a
copied venv still works merely because `bin/python` exists. If an absolute symlink
points to another machine, either recreate the venv with the matching Python on the
current host or ask the user for the intended base interpreter and repair both the
link and `pyvenv.cfg`.

Never connect a venv to a different Python minor version.

## Training configuration

Unless the user already provided them, confirm:

- task name and environment version;
- dataset path, group count, and seed policy;
- `max_steps`, `max_turns`, `num_generations`, and `generation_batch_size`;
- full-parameter or LoRA training;
- dtype, DeepSpeed, vLLM, and GPU topology;
- completion-length limit scope;
- output directory and TensorBoard reporting.

Tutorial values are examples, not mandatory settings for the current experiment.

## Official ms-swift checkout

- Use the official `modelscope/ms-swift` repository, not the old temporary fork.
- The exact token-in/token-out work from PR #10012 is upstream.
- Before launch, print the resolved checkout path, `swift.__file__`, and package version.
- Inspect `PYTHONPATH`, the console script, and worker command lines so stale
  site-packages cannot override the intended checkout.

## Safety boundaries

- Do not start training without explicit user authorization.
- Do not delete checkpoints.
- Do not stop, kill, or restart an existing user job without permission.
- Do not overwrite an existing output directory.
- Do not use `pkill`, `killall`, or broad process-name matching for cleanup.
- Dry runs and audits must not leave training workers running.
- Stop and report stale Python links, wrong checkouts, invalid paths, or insufficient
  capacity.

## Reference files

- `scripts/generate_snake_tickets.sh`: parameterized Snake ticket generation;
- `config/agentark_runtime_config.example.yaml`: generic runtime template;
- `config/agentark_runtime_config.snake.example.yaml`: 8×8 Snake, 16-runtime example;
- `config/deepspeed_zero2_cpu.json`: generic ZeRO-2 CPU optimizer-offload example.

Use the launcher, CLI help, and configuration definitions in the active AgentArk and
ms-swift checkouts as the source of truth for concrete launch arguments.
