# Model Evaluation And Replay

This guide covers the local model-evaluation path in this repository. AgentArk
runtime setup is covered in [setup.md](setup.md). Public task pages and aggregate
leaderboards are available at [AgentArk Hub](https://p90-rushb.github.io/agentark-hub/).

## Evaluation Entrypoints

Use one of these scripts:

| Goal | Command |
| --- | --- |
| One task / one or more cases in one local env | `python -m agent_ark.ark_eval.run_api_agent --config config/ark_env/eval_seed1.example.yaml` |
| One task / many seed-model jobs across multiple envs | `python -m agent_ark.ark_eval.run_parallel_api_eval --config config/ark_env/parallel_api_eval.example.yaml` |
| Replay saved model actions without API calls | `python -m agent_ark.ark_eval.run_replay --config config/ark_env/replay.example.yaml` |

Start with `run_api_agent`. Move to parallel evaluation only after a single env
can reset and step reliably.

## Configuration Layers

AgentArk eval YAML files use four main sections:

- `env_cfg`: runtime paths, task type, and optional runtime sandbox config.
- `env_cfg.env_config_overrides`: temporary overrides for the runtime
  `Mods/config.yaml`, such as `num_parallel_envs`, `virtual_display`, or history
  settings.
- `eval`: case selection, seed ranges, output path, and parallelism.
- `models`: OpenAI-compatible model providers.

For evaluation, keep `env_cfg.env_config_overrides.num_parallel_envs: 1`.
Parallel scoring is done by launching multiple independent `ArkEnv` instances,
not by increasing sub-env count inside one `ArkEnv`.

## Single-Task Evaluation

Edit [config/ark_env/eval_seed1.example.yaml](../config/ark_env/eval_seed1.example.yaml):

```yaml
eval:
  output_path: tmp/marble_stop_seed1.jsonl
  cases:
    - case_id: marble-stop-seed-0001
      task_name: MarbleStop
      group_seed: 1
      env_id: 0

models:
  - name: openrouter-model
    provider: openrouter
    model: replace-with-openrouter-model-id
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    temperature: 0.0
```

This model block is only an OpenRouter example. Replace `provider`, `model`,
`base_url`, and `api_key_env` for any OpenAI-compatible endpoint you want to
evaluate.

Then run:

```bash
python -m agent_ark.ark_eval.run_api_agent \
  --config config/ark_env/eval_seed1.example.yaml
```

`task_name` must match a folder under `Mods/all_tasks`. The packaged runtime
includes 32 starter tasks, such as `MarbleStop`, `Snake`, `Pushbox`,
`ObjectRotationMatch`, and `StarterRouteJump3D`.

## Multiple Seeds

Use [config/ark_env/eval_seeds_1_n.example.yaml](../config/ark_env/eval_seeds_1_n.example.yaml):

```yaml
eval:
  task_names:
    - MarbleStop
  group_seeds:
    start: 1
    end: 5
```

Run:

```bash
python -m agent_ark.ark_eval.run_api_agent \
  --config config/ark_env/eval_seeds_1_n.example.yaml
```

You can also use an explicit list:

```yaml
eval:
  task_names:
    - MarbleStop
  group_seeds: [1, 2, 3, 10, 20]
```

If `eval.cases` is non-empty, it has the highest priority and each case can
specify exact `task_name`, `group_seed`, and `env_id`.

## Parallel API Evaluation

Parallel evaluation fans out many independent seed/model jobs across multiple
Unity runtimes:

```bash
python -m agent_ark.ark_eval.run_parallel_api_eval \
  --config config/ark_env/parallel_api_eval.example.yaml
```

Important fields:

- `eval.max_parallel_envs`: maximum number of live `ArkEnv` instances.
- `eval.worker_index_base`: worker index assigned to slot 0.
- `eval.task_names` or `eval.cases`: current runner expects one task per eval
  run.
- `models`: each model is evaluated on each selected seed/case.
- `env_cfg.runtime_sandbox`: keep enabled when `max_parallel_envs > 1`.

The runtime sandbox gives each worker a private writable runtime/Mods directory
while sharing task assets through `Mods/all_tasks`. This avoids races when Unity
or Python rewrites active config and bundle files.

## Browser Viewer And Human Actions

Enable the local chat viewer:

```yaml
hooks:
  visualization:
    enabled: true
    host: 127.0.0.1
    port: 18181
    open_browser: true
    keep_open_on_end: true
```

The viewer shows the messages sent to the model, images from observations, and
assistant responses. It is useful for debugging prompts and task behavior.

For manual task debugging, also enable:

```yaml
hooks:
  human_interaction:
    enabled: true
```

In this mode the script does not call the model API. You type actions in the
browser. Use the action format expected by the task, for example:

```xml
<tool_call>{"name":"RotateControlled","arguments":{"axis":"Y","degrees":20}}</tool_call>
```

Human interaction is intended for single-env debugging, not unattended parallel
evaluation.

## Action Modes

AgentArk tasks choose their action mode in runtime config.

`action_mode: func` expects structured tool calls:

```xml
<tool_call>{"name":"ExecutePlan","arguments":{"plan":"L4,U7"}}</tool_call>
```

The task prompt exposes tool documentation. Python validates the tool name and
arguments, renders a minimal C# call, and sends it to Unity/Roslyn.

`action_mode: code` expects a full C# script:

```xml
<code>
using UnityEngine;

public class ArkAct_Step0 : MonoBehaviour
{
    void Start()
    {
        var router = GetComponent<ActRouter>();
        router.Call("ExecutePlan", "L4,U7");
    }
}
</code>
```

Use tool-call mode when possible. Full-code mode is for tasks that require
cross-frame control, richer program logic, or multiple task API calls.

## Scoring

Evaluation runs through the final attempt allowed by the effective task runtime
config. An earlier successful attempt does not stop the rollout.

Key result fields:

- `score_reward`: official score for the case; equal to the last attempt reward.
- `best_attempt_reward`: diagnostic best reward across attempts.
- `ever_attempt_success`: whether any attempt succeeded.
- `final_attempt_success` / `rollout_success`: whether the final attempt
  succeeded.
- `first_success_attempt_index`: first successful attempt, if any.
- `rollout_truncated`: true when the rollout exhausted the configured budget.

`total_reward` is retained for debugging but is not the primary score.

## Replay

Replay reuses saved actions from JSONL evaluation records and reruns them in the
environment. It does not call a model API.

```bash
python -m agent_ark.ark_eval.run_replay \
  --config config/ark_env/replay.example.yaml \
  --records tmp/delay_train_seed1_5.jsonl \
  --index 0
```

Useful replay config:

```yaml
replay:
  records_path: tmp/delay_train_seed1_5.jsonl
  record_index: 0
  step_delay_s: 0.5
  use_record_max_attempts: true
  require_match: true
```

Common selectors include `record_index`, `record_indices`, `record_lines`,
`case_id`, `model_name`, `task_name`, `group_seed`, `env_id`, and `limit`.

Replay is most useful with the browser viewer enabled because images are
regenerated from the runtime instead of stored in the original eval record.

## Trajectory Save And Load

Evaluation can save attempt prefixes into a trajectory bank:

```yaml
eval:
  trajectory_save:
    enabled: true
    output_path: tmp/trajectory_bank.jsonl
    condition: ever_success
    prefix_attempts: 4
    include_images: true
```

Load prefixes later:

```yaml
eval:
  trajectory_load:
    enabled: true
    path: tmp/trajectory_bank.jsonl
    prefix_attempts: 4
```

This restores attempt-boundary history context. It does not restore half-step
physics state.
