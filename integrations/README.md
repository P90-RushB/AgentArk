# AgentArk RL Integrations

English | [简体中文](README.zh-CN.md)

AgentArk exposes its Unity runtime pool as an HTTP environment service so RL
trainers can run multi-turn, multimodal rollouts in their own Python
environments. Start by completing the [AgentArk setup](../docs/setup.md) and one
real Unity evaluation, then choose an integration below.

## Choose an integration

| Integration | Adapter location | Env protocol | Start here |
| --- | --- | --- | --- |
| ms-swift GRPO | Maintained in this repository under `integrations/ms_swift` | v2 | [ms-swift runbook (Chinese)](ms_swift/README.md) |
| VERL GRPO | Maintained in the public [`agentark_rl` fork](https://github.com/P90-RushB/verl/tree/agentark_rl/agentark_recipe/agentark_env_agent) | Legacy v1, as used by the current recipe | [VERL integration guide](verl/README.md) |

The ms-swift runbook is the complete installation and training path for the
repository-local adapter. The VERL guide prepares and verifies the AgentArk
side, then hands off to the external fork for its dataset, agent-loop, and
trainer configuration.

## Shared process layout

```text
RL trainer (framework-specific Python environment)
                       |
                      HTTP
                       |
AgentArk Env Server (AgentArk Python 3.10.12 environment)
                       |
            runtime sandbox pool
                       |
             multiple Unity processes
```

The Env Server is one service process for a host and port, but it can manage
multiple runtime sandboxes and Unity child processes concurrently. Keep the
AgentArk environment and the trainer environment separate so each can use its
own dependency stack.

Protocol v1 and v2 use isolated pool namespaces. Warm up the namespace named by
the selected integration guide; environments prepared for one namespace are
not available to the other.

## First-run checkpoints

Before increasing training scale, confirm that:

1. A real AgentArk Unity evaluation completes successfully.
2. The selected runbook's server-health and runtime-pool checks pass.
3. Its smoke run or one-step training run completes a rollout, reports reward,
   and reaches the success markers documented by that integration.

For shared architecture, grouping and task-selection semantics, and operational
background, see the [RL training guide](../docs/rl-training.md). The
[ms-swift architecture note (Chinese)](ms_swift/ARCHITECTURE.zh-CN.md) describes
that adapter's data flow and compares it with the VERL implementation.
