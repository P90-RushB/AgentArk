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

Then choose one model provider style below. Keep `agent_type: simple` and its
`simple_agent` settings when replacing the provider fields in a new evaluation.
Provider-only snippets below describe transport settings and retain legacy
selection when copied without an explicit harness.

### OpenAI-Compatible Providers

Use this path for OpenAI, OpenRouter, DashScope-compatible, or other
OpenAI-compatible HTTP endpoints. New evaluation templates explicitly select
`SimpleAgent`, which uses Chat Completions with native function tools.

```yaml
models:
  - name: simple-openrouter-model
    agent_type: simple
    simple_agent:
      output_token_parameter: max_tokens
      context:
        max_context_tokens: 262144
        output_reserve_tokens: 30000
      tools:
        enabled: true
        workspace_parent: tmp/simple_agent_workspaces
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

For the legacy stateless `agent_type: api` harness, prefer full message context
(the SimpleAgent evaluator sets its own delta contract):

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

### Recommended SimpleAgent: Context Compaction and Host Tools

`APIAgent`, `SimpleAgent`, and `CodexAgent` are separate agents. The optional
`models[*].agent_type` selects `api`, `simple`, or `codex`; `provider` continues
to select the model backend. Without `agent_type`, the existing behavior is
unchanged: `provider: codex` selects CodexAgent, other providers select APIAgent.
The existing agents retain their implementation and configuration defaults. New
seed, parallel and Colab examples explicitly choose SimpleAgent with host tools
enabled; no global missing-key default or task configuration is rewritten. To
retain the baseline, explicitly select `agent_type: api` and omit `simple_agent`.

Evaluation results describe the model plus its harness configuration. Keep APIAgent
and SimpleAgent scores as separate conditions. Resume checks distinguish harness
identity/configuration; historical records without identity retain legacy inference.
An older SimpleAgent result whose configuration cannot be established may be
rerun instead of silently reused; its saved action playback is unaffected.
Use a new output path for a new evaluation condition. Action playback uses
`ReplayAgent`, consumes saved final actions, and does not call either model harness.
Exact replay still requires compatible task/runtime artifacts, seeds and parameters.

Run [simple_agent.example.yaml](../config/ark_env/simple_agent.example.yaml):

```bash
python -m agent_ark.ark_eval.run_api_agent --config config/ark_env/simple_agent.example.yaml
```

Selecting `agent_type: simple` makes the evaluator enable append-only delta
messages in that agent's private environment configuration. SimpleAgent owns
history, recent complete turns, image selection, and model-generated summaries
under `simple_agent.context` budgets. Stateless HTTP requests still receive the
complete assembled working context. Original bounded history is kept separately;
summaries are labelled model memory, not environment facts. Instructions, current
observations, necessary feedback, and complete code are not silently shortened.
Token counting uses a configurable image/text estimate alongside encoded-byte,
image-count, and raw-image-byte limits.
Working-context and summary-input encoded bytes each default to 16,000,000,
including base64 images, leaving room for the API request envelope.

The shared defaults allow 262,144 (256 * 1024) estimated context tokens, including a
20,480 action-output reserve, up to 256 working images, and two protected recent inputs
(including the current input). Summary requests accept up to 48,000 estimated
input tokens and have a separate 32,768-token generation budget. Only their final
`content` becomes memory; separate readable reasoning remains in the raw trace
and still counts toward provider output usage. Empty final content, output-limit
truncation, overlong summaries and summaries that do not reduce the source are
recoverable generation failures. The next summary request receives a short
failure explanation and asks for a corrected final summary. These retries share
the existing per-decision `max_summary_calls` limit (default 4), lifetime call
budget and decision deadline; successful compactions count toward the same
limit. Invalid summaries never replace working history. There is no separate
summary-length target.
These are configurable upper bounds, not a fixed compression ratio or a guarantee
that a model's native context window is large enough.

`context.compaction_policy` selects `message_chunks_v1` (new-run default) or
`whole_turn_v1` (legacy). The new policy keeps normal complete-turn summaries
when they fit, and can split an oversized old ordinary text/multi-image message
into bounded summary requests before merging the notes. Whole images retain
their bytes and order; text fragments carry source ranges. Closed tool-call
batches and their results remain indivisible, and an oversized tool group fails
explicitly. Current/protected inputs and in-progress native tool prefixes are
never split. Leaves, merges and invalid-output retries all share the existing
four-call default; a larger history is not guaranteed to fit that finite budget.
Working history changes only after the complete plan succeeds, while original
archives remain immutable. Summary request/event traces include the policy,
stage and fragment provenance. All numeric defaults above are unchanged.

The new policy is a distinct evaluation condition and gets a new fingerprint.
Set `compaction_policy: whole_turn_v1` to retain the historical SimpleAgent
fingerprint and completed-case resume behavior. Saved-action replay does not
run compaction. Loading a trajectory history prefix is not restoration of the
private model working context or its compaction state. Existing Kaggle tasks
keep their immutable wrapper pin and evaluation conditions.

SimpleAgent's HTTP backend preserves the supported OpenRouter reasoning fields
and Volcengine's original `reasoning_content` / `encrypted_content` pair in
Chat Completions history. Response content and native reasoning are stored
separately; response fields select the supported replay family without a new
configuration option. Other endpoint contracts are not guaranteed compatible.
See [native reasoning history](simple-agent-native-reasoning.md) for replay,
compaction and record behavior.

SimpleAgent uses the HTTP API transport by default. To select an existing
ChatGPT subscription instead, use this model entry for any task:

```yaml
models:
  - name: simple-codex
    agent_type: simple
    provider: codex
    model: gpt-6.1-sol
    reasoning_effort: medium
```

Omitted `simple_agent` settings inherit the shared defaults. Environment, case
and workspace settings follow `config/ark_env/simple_agent.example.yaml`.
This backend requires the optional `openai-codex` dependency
and existing ChatGPT file credentials; use `AGENTARK_CODEX_BIN` when the SDK's
pinned CLI does not support the requested model.
SimpleAgent still owns memory, compaction and host tools. Each model request
starts a fresh isolated SDK thread; a strict JSON envelope transports declared
tool calls back to SimpleRuntime. These are not Codex native function tools.
SDK reasoning, cached-token details and usage remain audit metadata. The SDK
does not expose a hard output-token cap: requested action/summary budgets are
prompt guidance, while runtime text/byte and summary-estimate checks still
apply. Traces mark this limitation explicitly. A single SDK turn is not a
guarantee of a single underlying HTTP request.
The trace field `simple_agent.sdk_max_retries` describes HTTP backend retries;
it does not control Codex CLI network recovery. Codex error notifications are
audited while the same turn continues to its terminal result, bounded by eight
notifications and the request
deadline; this adapter does not issue a fresh outer request retry.

Set `simple_agent.tools.enabled: true` and a writable `workspace_parent` to
enable `list_files`, ranged UTF-8 `read_file`, actual-image `view_image`, and
`write_note`. Each case/agent has private notes and immutable records of its own
permitted observations and outputs. Auto-reset attempts retain case memory;
new cases clear it, and close removes the private working directories. Host tools
do not receive project directories, hidden state, source, credentials, a shell,
or a game client. Their bounded inner loop returns one final action in the task's
existing format. Tool and summary calls do not allocate environment steps.
When a host tool registry is present, SimpleAgent also declares the reserved
`submit_action` control function. It accepts `format` (`tool_call`, `params`,
`code`, or `raw`) and a string `content`. Supply the inner action body without
XML wrappers; the adapter translates a single standalone submission into the
existing environment action. Code source is preserved; `params` keeps the
existing extractor's whitespace normalization. Payloads must omit action-envelope
tags (use C# Unicode escapes for such string literals in code).
Submission never executes a host tool or advances the game. This separates
task XML output recipes from model-native tool calls.
Future context keeps the native call and a labelled selection acknowledgement;
authentic provider text may also contain a plaintext action. The separate
environment action echo is deduplicated; a native submission takes precedence
over accompanying provider prose. The following environment observation supplies
the actual outcome. Summary
requests receive source-labelled historical tool transcripts and original images,
without executable tools. Traces distinguish provider output from action
translation. Mixed host/submission calls and malformed arguments fail explicitly.
Native tool arguments have a 128 KiB UTF-8 limit, including JSON escaping;
decoded actions also obey the configured output bounds. Plain assistant output
remains a compatibility path with `simple_agent.tool_choice: auto`, and
tools-disabled agents keep that path. Native action rounds default to
`simple_agent.tool_choice: required`: the model chooses a host tool or terminal
submission. A missing native call then fails explicitly. Set `auto` for endpoints
that do not support required tool selection; there is no implicit parameter
downgrade or retry. Summary requests and requests without native tools omit this
parameter. See [OpenRouter's tool choice documentation](https://openrouter.ai/blog/tutorials/tool-calling/)
for the endpoint-dependent support of stricter choices.
Python callers can also supply `SimpleAgent(..., tool_setup=callback)` to
register trusted `ToolSpec`/function pairs on each fresh case's `ToolRegistry`.
This hook is not loaded from YAML and registries are not shared between cases.
ToolRegistry validates a bounded JSON schema subset (objects, arrays, strings,
integers, and booleans) before invoking a registered function.
The agent defaults to 512 published records and a 512 MiB published-byte cap;
explicitly size `tools.limits` for longer tasks. Original archive total/image
byte caps are each 512 MiB, with 32 Mi characters of archived text. These limits
are cumulative and are not reclaimed by summarizing working context. Record
overflow stops the decision and retains failure evidence.

Working context, original archive, model calls, tool rounds/results, records and
notes have finite bounds. Failed compaction or exhausted budgets fail explicitly.
New agent HTTP retries default to zero; an explicit `max_retries` controls client
retries. Action output is capped by the smaller of `max_completion_tokens` and
`output_reserve_tokens`; summaries use their independent `summary_output_tokens`
budget (default 32,768), without the action `max_completion_tokens` cap. This is not
unbounded history, disk-backed conversation resume, or an operating-system
sandbox against hostile local processes.
If summary recovery is exhausted, the evaluator records the case error and its
request/usage trace. The general example uses `eval.stop_on_error: false` to
continue other cases; it does not replay the failed case. Authentication,
parameter, refusal, cancellation, timeout and archive-capacity failures do not
receive automatic summary-generation retries. An unsuccessful summary does not
allocate an environment step or submit a placeholder action.
Compaction summarizes complete older turns. If a complete turn exceeds the
summary input bounds and the remaining suffix provably exceeds action hard
bounds, it reports the limiting capacities immediately instead of repeatedly
rewriting the same notes. Configured budgets are not increased automatically.
On rollout exceptions, error results add `rollout_complete: false` and a
versioned `partial_rollout`, preserving completed environment steps, image
assets and the failed decision. An action whose `env.step` raised has unknown
execution status and is stored separately. Raw initial/current observations
are encoded separately; intermediate model-visible observations remain in the
actual request traces and image assets. Incomplete rollouts stay in result
JSONL, outside the trajectory bank's completed attempts and training prefixes.
Reasoning tokens share each request's completion cap. Summary requests keep
the same reasoning/thinking parameters as action requests, including the Codex
SDK backend. Only final assistant content becomes summary memory; reasoning
fields remain audit data. Separating these fields does not reserve output tokens
for content, so the independent summary budget must allow room for both.
The first summary prompt includes the token budget (including reasoning) and
the final-content character/UTF-8 byte limits, and requests only concise notes.
A response ending with `length` fails without submitting a partial action or
accepting partial notes. Size both action caps and the independent summary
budget for the chosen profile. The Qwen example selects low reasoning effort.
The current budgets require validation
with the chosen model; earlier bounded Player checks do not validate these new
defaults. Direct construction keeps file tools opt-in and requires a writable
`workspace_parent` when enabled; the example enables them explicitly.

Each step's `raw_response_by_agent` retains `model_requests`, `agent_events`,
`agent_usage`, and safe `simple_agent` configuration metadata. These distinguish
actual action/summary requests and tool continuations. Request images use
`agentark-trace-image:<sha256>` references; `result.model_trace_images` stores each
referenced image once per case as `sha256 -> {mime_type, data_url}`. Image hashes,
byte sizes and known usage from failed calls are retained, including preceding
successful summary and tool work.

HumanView keeps the environment chat and separately displays actual model calls,
including prompts, tools, parameters, assistant text, calls, results and summaries.
Request previews are labelled separately from post-compaction calls. Replay
passes through saved calls/events without executing host tools; an older record
without those fields is labelled unavailable, not reconstructed as an exact request.
The Codex transport audit captures application-provided SDK turn inputs, not
hidden SDK or model-internal context. Returned opaque reasoning remains opaque.
The viewer builds details when expanded and shows the latest 200 decisions;
older calls remain in saved records. The viewer server's image cache preserves the active case's
images for later tool rereads and expires prior-case images when buffered calls
no longer reference them.

`eval.trajectory_save.include_agent_trace` defaults to `true`. Saved trajectories
add optional `agent_trace: {version: 1, steps, model_trace_images}` outside
`history_snapshot`; it is never restored into the model prompt. A prefix export
filters out later attempts. `include_images: false` omits trace image assets, and
`include_agent_trace: false` retains the legacy trajectory storage shape. Historical
records remain readable; information that was never saved cannot be recovered.
The main conversation separates public assistant content from the selected
environment action; subsequent environment feedback reports execution. The
`Model instructions / Harness` panel shows system/developer messages from actual
action requests, including the saved Codex outer JSON protocol. Original
environment prompts are labelled separately because the harness may adapt them.
Summary steps show notes actually used by the action request; an attempted
summary alone is not presented as committed new memory.
Environment replay still replays final actions without re-running models or
restoring tools/notes. RL rollout integration is separate: selecting this agent
does not change Swift/VERL training trajectories.

Implementation lives in `src/agent_ark/agent/simple_agent/`; API and Codex retain
their own modules. Trusted Python extensions can register an `AgentType` through
`agent/agent_registry.py`, with a separate module, options keyword, and message
contract. Eval YAML cannot import arbitrary modules.

See [SimpleAgent validation](simple-agent-validation.md) for the pre-commit
regression and legacy-agent compatibility checks.

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
    # GPT-6 Astra choices: low, medium, high, xhigh, max.
    # Omit to use the Codex SDK/model default.
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
disable list. Current GPT-6-compatible installations should use
`openai-codex>=0.147.0`. Set `AGENTARK_CODEX_BIN` to the current Codex App/CLI
executable when the SDK-bundled CLI does not yet support the selected model.

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

Replay selects `status: ok` by default. To debug a new partial error record,
explicitly configure:

```yaml
replay:
  records_path: tmp/results.jsonl
  status: error
  allow_partial: true
  require_match: true
```

Partial replay executes only completed environment steps and compares their
actions, rewards, done flags and attempt/rollout metadata. It does not execute
the failed decision or an action with unknown execution status, and does not
rerun models or host tools. Its result remains `status: error` with
`rollout_complete: false` and no complete evaluation score. Missing steps in
older error records cannot be recovered. Existing complete records retain the
normal replay path.

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
