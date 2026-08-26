# Model Evaluation And Replay

English | [简体中文](evaluation-guide.zh-CN.md)

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
- `models`: OpenAI-compatible model providers, or `provider: codex` for local
  Codex SDK evaluation.

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
```

Then choose one model provider style below.

### OpenAI-Compatible Providers

Use this path for OpenAI, OpenRouter, DashScope-compatible, or other
OpenAI-compatible HTTP endpoints. These entries are run by `APIAgent` through
chat-completions style requests.

```yaml
models:
  - name: openrouter-model
    provider: openrouter
    model: replace-with-openrouter-model-id
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    thinking:
      type: enabled
    reasoning_effort: high
    max_completion_tokens: 30000
    temperature: 0.0
```

Replace `provider`, `model`, `base_url`, and `api_key_env` for your endpoint.
You can also set `api_key` directly in a private local config instead of using
`api_key_env`.
If a model or proxy reports that `temperature` is deprecated or unsupported,
set `temperature: null` to omit the parameter from API requests.
`max_completion_tokens` defaults to `30000` for HTTP API models and limits
reasoning plus visible output tokens. Set it to `null` to omit the parameter.

HTTP models share the `thinking.type` (`enabled` or `disabled`) and
`reasoning_effort` configuration. AgentArk maps them by provider:

| Provider | Request fields sent |
| --- | --- |
| `volcengine` | `thinking.type` and top-level `reasoning_effort` |
| `openrouter` | `reasoning.enabled` and `reasoning.effort` |
| `dashscope` / `qwen` | `enable_thinking`; no `reasoning_effort` mapping |
| `openai` / `generic` | Top-level `reasoning_effort`; no `thinking` mapping |

AgentArk raises a configuration error before the request if a provider cannot
map a setting unambiguously. Omitting the settings preserves the existing
provider defaults.

#### Volcengine Ark Online Inference

Volcengine Ark online inference is compatible with the OpenAI SDK. Use
`provider: volcengine` so AgentArk sends Chat Completions requests with Ark's
deep-thinking fields:

```yaml
models:
  - name: doubao-seed-2-1-pro-260628
    provider: volcengine
    model: doubao-seed-2-1-pro-260628
    base_url: https://ark.cn-beijing.volces.com/api/v3
    api_key_env: ARK_API_KEY
    thinking:
      type: enabled
    reasoning_effort: high
    max_completion_tokens: 30000
    timeout_s: 180
    temperature: 1.0
```

Set the API key before running:

```bash
export ARK_API_KEY="your-api-key"
```

This versioned model ID and `/api/v3` base URL select pay-as-you-go online
inference. Do not mix them with the Coding Plan `/api/coding/v3` base URL and
the `doubao-seed-2.0-pro` model name. AgentArk currently calls the Chat
Completions API. `thinking.type: enabled` explicitly enables deep thinking,
`reasoning_effort: high` selects the reasoning level,
`max_completion_tokens: 30000` caps reasoning plus visible output tokens, and
`temperature: 1.0` sets the sampling temperature explicitly. Visual requests at
`high` can exceed the example's 180-second timeout; increase `timeout_s` when
that latency is acceptable.

For stateless HTTP providers, prefer full message context:

```yaml
env_cfg:
  env_config_overrides:
    env_wrapper_cfg:
      context_manager:
        messages:
          enabled: true
          only_return_messages: true
          append_only: true
          return_mode: full
```

### Codex SDK Provider

Use this path for local Codex SDK evaluation. It is not an OpenAI-compatible
HTTP endpoint: `provider: codex` does not use `base_url`, `api_key`, or
`api_key_env`.

Install AgentArk with the `codex` extra:

```bash
python -m pip install -U pip
python -m pip install -e ".[codex]"
```

If your package index does not mirror the beta Codex SDK yet, install from PyPI
explicitly:

```bash
python -m pip install -i https://pypi.org/simple openai-codex
```

```yaml
models:
  - name: codex-gpt55
    provider: codex
    model: gpt-5.5
    sandbox: read_only
    timeout_s: 600
    # Choices: none, minimal, low, medium, high, xhigh.
    # Omit to use the Codex SDK/model default. low is the safer cheap setting;
    # minimal can be rejected when Codex tools are enabled.
    reasoning_effort: medium
    thread_mode: per_agent
    codex_context_mode: lean
    # Add any other MCP server names from your personal Codex configuration.
    codex_disabled_mcp_servers: [node_repl, unityMCP]
```

AgentArk converts the OpenAI-style evaluation messages into Codex text and
image inputs. Data-URI image observations are passed directly as Codex
`ImageInput` items. Use `reasoning_effort` to override the Codex turn reasoning
depth when you want cheaper or faster model turns.

AgentArk's `provider: codex` defaults to `codex_context_mode: lean`. The runner
passes this profile only to the SDK thread it creates: it does not modify
`~/.codex/config.toml` or affect Codex app, CLI, IDE, or other SDK sessions. A
lean thread uses empty Codex base/developer instructions, deny-all approvals, an
ephemeral thread, and an empty temporary cwd. It also disables web/image/shell
tools, apps, plugins, multi-agent, and related features. The common
`node_repl` and `unityMCP` servers are explicitly disabled by default.

Codex MCP configuration maps merge by server name, so an empty map cannot
override every personal MCP definition. Add any other configured server names
to `codex_disabled_mcp_servers`. For a control run that retains standard Codex
context, set `codex_context_mode: default`. Both modes remain local to this
AgentArk SDK evaluation. Result JSONL records the effective context mode and MCP
disable list. The currently used `openai-codex==0.1.0b3` already supports these
thread parameters; no upgrade is required for this feature.

`thread_mode` defaults to `per_agent`. This is recommended because Codex SDK
turns include Codex-side runtime, system, and tool context in addition to
AgentArk's visible `request_messages`. That context is not shown by the
environment, but it is counted in Codex token usage. In local MarbleStop checks,
a first turn with only a few thousand visible text characters plus one image
reported tens of thousands of input tokens, and `per_agent` runs showed much
higher cached-token reuse on later turns. Keeping a persistent thread lets
Codex reuse its own setup plus the reset/base prompt; the environment can then
send only step deltas.

For the recommended stateful Codex mode, use delta message context:

```yaml
env_cfg:
  env_config_overrides:
    env_wrapper_cfg:
      context_manager:
        messages:
          enabled: true
          only_return_messages: true
          append_only: true
          return_mode: delta
          history_only_on_attempt_start: true
```

Use `return_mode: full` if you set Codex `thread_mode: per_turn`. Use
`append_only: true` plus `return_mode: delta` with `thread_mode: per_agent` so
the persistent Codex thread receives only the new step delta after the
reset/base prompt.

Delta delivery is controlled by AgentArk's `messages.return_mode`; it is not
unconditionally generated by `CodexAgent`. When messages are enabled, the code
default is `delta`: reset still returns the complete base message once, and
later step/attempt updates return only new messages. Keep `append_only: true`,
`return_mode: delta`, and `thread_mode: per_agent` explicit in batch eval
configs for reproducibility.

Then run:

```bash
python -m agent_ark.ark_eval.run_api_agent \
  --config config/ark_env/eval_seed1.example.yaml
```

`task_name` must match a folder under `Mods/all_tasks`. The packaged runtime
includes 51 starter tasks, such as `MarbleStop`, `Snake`, `Pushbox`,
`ObjectRotationMatch`, and `StarterRouteJump3D`.

### Black-Box Player Feedback

Use [player_feedback.example.yaml](../config/ark_env/player_feedback.example.yaml)
to let the same Codex SDK thread play a packaged task and then report observable
task-quality problems after the rollout. The player receives the normal task
prompt, image observations, visible/history messages, and action results. It is
not injected with task source, hidden state, oracle actions, design notes, or
reviewer evidence.

Use the SDK player for the actual play loop rather than a normal workflow
subagent. The SDK adapter already consumes AgentArk's multimodal messages as
native image inputs, preserves one player thread across steps and attempts, and
records the exact task, seed, actions, outcomes, trajectory, and replay evidence.
A workflow subagent normally inherits repository and design/review context, so
it is appropriate for launching the run and checking its artifacts, but not for
choosing black-box player actions.

```bash
python -m agent_ark.ark_eval.run_api_agent \
  --config config/ark_env/player_feedback.example.yaml
```

The model entry must use `provider: codex`, `thread_mode: per_agent`, and:

```yaml
player_feedback:
  enabled: true
```

The default lean player mode adds these safeguards automatically:

- The action prompt tells Codex to use only player-visible messages and images.
- The Codex thread starts in a new empty temporary cwd instead of the source
  repository; that directory is removed when the agent closes.
- Codex base/developer instructions are emptied, and tools, apps, plugins,
  multi-agent, and the listed MCP servers are disabled at thread scope.

After the terminal observation, `run_api_agent` asks that same thread for one
structured `<player_feedback>` JSON report. The report separates concrete task
defects from non-defects such as failing to solve the task, difficulty, player
mistakes, or a legitimate need to explore. The result record stores the report
under `player_feedback`. Enable `trajectory_save` with `condition: all` and
`include_images: true` so every reported issue can be replayed even when the
player did not succeed. If the terminal report is missing, fails, or does not
match the required JSON schema, the evaluation result is marked `status: error`
so resume/retry logic does not silently treat the player gate as complete; the
rollout and trajectory evidence are still retained.

The report also contains `information_reveal_assessment` with one of four
player-side classifications: `complete_initially`, `intentional_exploration`,
`suspected_missing_information_defect`, or `unclear`. It records the visible
evidence and the attempt indices the player actually considered. `ArkEnv` may
intentionally use an early attempt for discovery and expose consequences or
configured history at later attempt boundaries. Initial uncertainty is therefore
not a defect when progressive discovery is intentional and works. A suspected
defect requires visible evidence that promised or necessary information never
became discoverable, or that the observed reveal behavior contradicted the
player-facing contract.

The player's classification is evidence, not the final verdict, and it never
directly requests a source change. The workflow sends the parsed report and
image-inclusive replay to the task reviewer. The reviewer independently compares
it with the approved information-reveal design, task description, effective
`max_attempts`, seed/history semantics, and packaged replay, then classifies each
candidate as `confirmed defect`, `non-defect`, or `inconclusive`. Only confirmed
defects reach the builder; an inconclusive finding may receive one fresh targeted
run, and intentional exploration remains a recorded non-defect.

Both serial and parallel runners reject player-feedback configs unless
`trajectory_save.enabled: true`, `condition: all`, `include_images: true`, and
an output path are present. They also reject human-interaction replacement,
non-Codex providers, non-read-only sandboxes, explicit source cwd values, and
stateless Codex threads for this mode.

The isolated cwd and prompt are contamination-reduction measures, not a hard
security boundary: the current Codex SDK adapter does not disable every
filesystem or shell tool. Do not describe a run as strict source-proof black-box
testing unless the deployed SDK/runtime also enforces that stronger tool
restriction.

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

An explicit `action_mode: code` opts into the restricted Code Runtime. The task
prompt supplies a capability manifest, and the model returns one plain managed
`AgentAction` source file:

```xml
<code>
public sealed class AgentAction : CodeAction
{
    public override void Start()
    {
        int target = Context.Read("read_target").AsInt32();
        Context.Call("apply_target", CodeValue.FromInt32(target));
        Context.CompleteStep();
        Context.StopSelf();
    }
}
</code>
```

The member names above are illustrative; generated code may use only members in
the current task's manifest. It cannot access Unity, the scene hierarchy,
`ActRouter`, reflection, files, networking, or threads directly. Use Func Mode
when bounded tool calls are sufficient. Use Code Mode when the evaluation needs
branching, loops, persistent program state, or trusted capability calls across
multiple host-driven `Tick`/`FixedTick` callbacks.

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
