# SimpleAgent evaluation with LiteLLM

English | [简体中文](litellm-evaluation.zh-CN.md)

LiteLLM is a supported, optional model backend for `SimpleAgent`. It uses the
same evaluation runners, environment observations, host tools, context budgets,
compaction and action submission as the native backend. Existing configurations
keep `simple_agent.backend: native`; installing the extra does not select LiteLLM.
For runtime installation and paths, see [setup](setup.md). For the full harness,
parallel evaluation and replay, see [the evaluation guide](evaluation-guide.md).

## Select the harness, backend and API separately

| Setting | Meaning |
| --- | --- |
| `agent_type: simple` | Selects the SimpleAgent harness. Omitting it still selects the historical APIAgent/CodexAgent behavior. |
| `simple_agent.backend` | `native` by default; set `litellm` to use the optional SDK. |
| `simple_agent.litellm.model` | Required explicit LiteLLM `provider/model` route; this selects the SDK vendor. |
| `simple_agent.litellm.api` | `chat_completions` by default (`acompletion`), or `responses` (`aresponses`). |
| Top-level `provider` | Retains the evaluator's provider label and participates in evaluation identity. It does not select the LiteLLM vendor. `codex` cannot be combined with this backend. |
| Top-level `model` | Display/evaluation label recorded in results. Examples use the provider's wire model here, especially for Kaggle; the nested LiteLLM model controls SDK dispatch. |
| Top-level `base_url` | Optional explicit provider/proxy API base. Omission selects the SDK provider's default, rather than the native backend's historical DashScope default. |

Chat Completions can use the pinned SDK's supported Chat providers, subject to
the chosen model's vision, function-tool and parameter support. Responses
currently requires an `openai/` route, including an OpenAI-compatible proxy.
AgentArk calls native `aresponses`; it does not translate Chat responses into
Responses items. The backend does not infer a vendor or API mode from a URL or
display model.

For `api: responses`, `openai/responses/X` is accepted as an alias for `openai/X`
and normalized before dispatch and fingerprinting. It still uses native
`aresponses`. That spelling is rejected with `api: chat_completions`; prefer the
explicit API setting and ordinary `openai/X` spelling in new configs.

## Install and run

From the AgentArk Python package root, use the existing Python 3.10 environment
(supported range: 3.10.1–3.10.12):

```bash
python -m pip install -e ".[litellm]"
python -m pip check
```

The extra pins `litellm[mcp]==1.106.6` and `tenacity==9.1.2`. MCP supplies the SDK's
tool-request imports; tenacity supplies its retry runtime. The install can upgrade
shared dependencies; the validated environment upgraded OpenAI SDK 2.8.1 to
2.54.0. Record dependency versions with evaluation conditions. Only explicitly
selected LiteLLM configurations use the new backend.

Set `AGENTARK_ENV_PATH`, `AGENTARK_MOD_PATH` and `OPENAI_API_KEY` in the current
process as described in [setup](setup.md). A complete direct-OpenAI example is:

```yaml
env_cfg:
  env_path: "${AGENTARK_ENV_PATH}"
  mod_path: "${AGENTARK_MOD_PATH}"
  task_type: RLTask
  env_config_overrides:
    override_by_task: true
    num_parallel_envs: 1

eval:
  output_path: tmp/simple_agent_litellm_results.jsonl
  stop_on_error: false
  fixed_env_id: 0
  task_names: [MarbleStop]
  group_seeds:
    start: 1
    end: 2

models:
  - name: simple-astra-litellm
    agent_type: simple
    provider: openai
    model: gpt-6-astra
    api_key_env: OPENAI_API_KEY
    temperature: null
    reasoning_effort: medium
    max_completion_tokens: 30000
    max_retries: 0
    timeout_s: 600
    simple_agent:
      backend: litellm
      litellm:
        model: openai/gpt-6-astra
        api: responses
      tool_choice: required
      context:
        compaction_policy: tool_rounds_v1
        output_reserve_tokens: 30000
      tools:
        enabled: true
        workspace_parent: tmp/simple_agent_litellm_workspaces
```

This is also available as
[simple_agent_litellm.example.yaml](../config/ark_env/simple_agent_litellm.example.yaml).
Run it with:

```bash
python -m agent_ark.ark_eval.run_api_agent --config config/ark_env/simple_agent_litellm.example.yaml
```

Use a task present in your packaged runtime. Environment overrides, seed ranges,
trajectory saving and parallel jobs use the ordinary evaluation schema; no
LiteLLM-specific runner or runtime build is needed. This recipe configures
direct OpenAI; the real-provider smoke below used Kaggle's compatible proxy.

## OpenRouter Chat Completions

Keep the preceding `env_cfg` and `eval`; replace `models` with the following.
Set `OPENROUTER_API_KEY` in the process. The SDK can select OpenRouter's endpoint
without `base_url`; an explicit compatible endpoint may also be supplied.

```yaml
models:
  - name: simple-qwen-litellm
    agent_type: simple
    provider: openrouter
    model: qwen/qwen3.7-plus
    api_key_env: OPENROUTER_API_KEY
    temperature: null
    max_completion_tokens: 30000
    max_retries: 0
    timeout_s: 600
    simple_agent:
      backend: litellm
      litellm:
        model: openrouter/qwen/qwen3.7-plus
        api: chat_completions
      output_token_parameter: max_tokens
      tool_choice: required
      context:
        output_reserve_tokens: 30000
      tools:
        enabled: true
        workspace_parent: tmp/simple_agent_litellm_workspaces
```

This is a routing recipe, not a live certification of that model or all
OpenRouter reasoning modes. Add common `reasoning_effort` only if the selected
model/SDK route supports it. The existing native OpenRouter example and its
`thinking` mapping remain available separately in the
[evaluation guide](evaluation-guide.md#native-openai-compatible-providers).

Other Chat vendors use the same rule. For example, this routing fragment uses
an Anthropic-qualified model while retaining the evaluator's neutral `auto`
label; replace `your-claude-model-id` with the intended supported model and keep
the environment, budgets and tool settings from the full configuration:

```yaml
provider: auto
model: your-claude-model-id
api_key_env: ANTHROPIC_API_KEY
simple_agent:
  backend: litellm
  litellm:
    model: anthropic/your-claude-model-id
    api: chat_completions
```

## Kaggle model proxy and notebook

Use [agentark_litellm_task.ipynb](../benchmarks/kaggle/agentark_litellm_task.ipynb)
for a separate LiteLLM notebook. Its generic default is MarbleStop seeds 1–10;
the opening instructions also provide a Door (`DoorSpin3Turns`) seeds 1–2 recipe and its
score range of −1 to 1. The environment remains `env-1.0.6`. The notebook pins
the AgentArk wrapper source to its fixed commit; retain that pin with published
benchmark conditions. Existing native notebooks retain their own pins and behavior.
See [Kaggle benchmark instructions](../benchmarks/kaggle/README.md) for execution.

The current notebook and shared Kaggle template use one LiteLLM policy for every
selected model: Responses, `tool_choice: auto`, omitted temperature, and omitted
optional reasoning settings. Model names do not choose an API or parameters.
This policy belongs to the Kaggle helpers; the core LiteLLM configuration still
defaults to Chat Completions and `required` when those fields are omitted.

| Notebook/template constant | Environment override | Values and default |
| --- | --- | --- |
| `DEFAULT_LITELLM_API = "responses"` | `AGENTARK_KAGGLE_LITELLM_API` | `responses` or `chat_completions` |
| `DEFAULT_TOOL_CHOICE = None` | `AGENTARK_KAGGLE_TOOL_CHOICE` | `auto` or `required`; `None` selects `auto` for LiteLLM and `required` for native |
| `DEFAULT_TEMPERATURE = "default"` | `AGENTARK_KAGGLE_TEMPERATURE` | `default` selects omitted temperature for LiteLLM and `1.0` for native; `null`/`none` omits it; a finite number sets it explicitly |

Environment overrides take precedence over constants. An explicit `temperature`
argument to `_write_eval_config` or `_run_agentark_eval`, including `None`, takes
precedence over the temperature environment override. Python constants also
accept `None` to omit temperature or a finite numeric temperature; booleans and
nonfinite values are rejected. Invalid settings fail before the first request. These controls
author that first request, with no exception-driven API or parameter fallback.
The model config records omitted temperature as `temperature: null`. Reasoning
remains optional; the helper does not force `reasoning_effort: none`.

Native generation retains Chat Completions, `required`, temperature `1.0`, and
its existing temperature-error retry. LiteLLM uses zero SDK retries and leaves
unsupported-parameter errors visible. Older imported notebooks retain their
original model-name rules: manually choosing Responses and null temperature
does not change their old Opus-only automatic tool-choice mapping. Set the tool
choice explicitly or import the newly generated notebook for the uniform policy.

`auto` still exposes function tools and the model remains instructed to use
`submit_action`, but it does not force a tool call. Existing SimpleAgent behavior
ends a decision on a prose-only response, which can become an invalid environment
action. This change adds no reminder, retry or exception-based fallback. The
selected API and tool choice participate in the evaluation fingerprint. The new
default benchmark name ends in `-litellm-responses-auto`; use a new benchmark name
and output path when changing these conditions. A single policy reduces hidden
differences between models, but does not guarantee that every current or future
model supports the chosen API, images, tools, and parameters.

For example, generating `Snake` with `--backend litellm` creates
`agentark_snake_litellm_responses_auto.py`, defines the CLI-discovered function
`agentark_snake_litellm_responses_auto`, and uses benchmark slug
`agentark-snake-litellm-responses-auto`. An explicit `--output` may override the
filename; the generated function and benchmark identity still identify the new
conditions.

One known limitation is Kaggle Gemini 2.5 Flash (`google/gemini-2.5-flash`): a
production-adapter Responses request containing two function tools returned
HTTP 400 `invalid_prompt`: "Multiple tools are supported only when they are all
search tools". Select Chat Completions explicitly with
`DEFAULT_LITELLM_API = "chat_completions"` or
`AGENTARK_KAGGLE_LITELLM_API=chat_completions` for that configuration. This is a
manual evaluation condition; the helper does not recognize model names or retry
with another API. The successful production-adapter Gemini 3 Flash Preview
probe does not establish support for Gemini 2.5 Flash.

The tested Grok 4.6 Responses route (`xai/grok-4.6`) returned HTTP 404
"requested model not found" on its first production-adapter request. Chat
Completions is also an explicit configurable alternative for this route, but
has not been live-verified here. The child probe receipt, rather than the outer
benchmark's completion status, determines the probe outcome.

The notebook returns `normalized_mean_score_reward`: each case's raw final-attempt
`score_reward` is normalized as `(score - min) / (max - min)` and averaged across
seeds. Generated Python tasks return raw `mean_score_reward`. Set the task's
actual score range before creating a benchmark; neither metric is cumulative
episode reward or a success rate. Selecting LiteLLM does not change these formulas.

For a manual evaluation, use Kaggle's OpenAPI base ending in `/models/openapi`
and the existing proxy credentials. Replace the example hostname with the real
base URL; the notebook derives this URL from Kaggle's proxy environment.

```yaml
models:
  - name: kaggle-astra-litellm
    agent_type: simple
    provider: openai
    model: openai/gpt-6-astra
    base_url: https://your-proxy-host/models/openapi
    api_key_env: MODEL_PROXY_API_KEY
    temperature: null
    max_completion_tokens: 30000
    max_retries: 0
    timeout_s: 600
    simple_agent:
      backend: litellm
      litellm:
        model: openai/openai/gpt-6-astra
        api: responses
      tool_choice: auto
      context:
        output_reserve_tokens: 30000
      tools:
        enabled: true
        workspace_parent: tmp/simple_agent_litellm_workspaces
```

The first `openai/` in the nested model selects LiteLLM's OpenAI adapter. The
remaining `openai/gpt-6-astra` is Kaggle's wire model, so the doubled prefix is
intentional. Direct OpenAI instead uses top-level `gpt-6-astra` and nested
`openai/gpt-6-astra`. The Kaggle policy omits temperature for every model; common
`reasoning_effort` is optional and omitted by default. Do not send the proxy's `/genai` endpoint to this
OpenAPI route or append `/chat/completions` to a Responses base URL.

## Credentials, parameters and budgets

- Prefer an explicit `api_key_env` to identify the intended credential. An
  explicit `api_key` is also accepted in a private config. If `api_key_env` is
  named but unset, evaluation fails before the request. If neither is given,
  LiteLLM uses its provider-specific environment credentials; AgentArk does not
  apply the native backend's cross-provider API-key fallback.
- `thinking` must be omitted or `null`, including when `provider: openrouter`
  or `volcengine` is retained as a label. Native `thinking` mappings do not run.
  Optional common `reasoning_effort` is sent through the SDK; its accepted
  values and behavior depend on the selected route.
- `temperature: null` omits temperature. For Chat, `output_token_parameter`
  selects `max_completion_tokens` (default) or `max_tokens`; Responses always
  sends `max_output_tokens`. Unsupported parameters fail rather than being
  silently dropped or retried with different settings.
- `max_completion_tokens` caps action generation together with the runtime's
  action budget. Summary generation keeps its own `summary_output_tokens`
  budget (default 32,768); an action cap of 30,000 does not reduce it to 30,000.
  Reasoning and visible output share the provider's output allowance. Responses
  rejects an effective output budget below 16 before the model call.
- SimpleAgent model retries default to zero. `max_retries: N` requests at most
  N SDK retries using one retry layer. The SDK's process-wide `drop_params`,
  `modify_params`, fallback models and nonzero global retry settings must remain
  disabled; AgentArk rejects them instead of changing evaluation conditions.
  Request timeout and the SimpleAgent decision/call budgets still apply.
- Host tools and `submit_action` keep their existing SimpleAgent semantics.
  For any endpoint that supports function tools but rejects required tool choice,
  a local manual YAML config can explicitly set `tool_choice: auto` under
  `simple_agent`.
  This permits tools without forcing a call and is a separate evaluation
  condition. A prose-only response ends the decision and can be an invalid
  environment action; no extra retry is added. Summaries have no executable
  tools. Read the
  [harness limits](evaluation-guide.md#recommended-simpleagent-context-compaction-and-host-tools)
  before raising budgets.

## Reasoning history, compaction and traces

The LiteLLM adapter has its own bounded, typed assistant state, scoped to the
backend, API mode, endpoint and normalized SDK model. It leaves the native
HTTP backend's `NativeAssistantState` contract unchanged.

For Chat, supported state includes `reasoning_items`, `thinking_blocks`,
`reasoning_content` and `provider_specific_fields`, including supported
tool-call signature fields. For Responses, supported ordered output items
retain assistant messages and phases, reasoning/encrypted content, and function
calls with their `call_id`. Calls remain paired with their tool results. These
are the SDK-returned items: LiteLLM may normalize provider data, omit null fields
or correct foreign call IDs. This is not an all-provider losslessness guarantee.

SimpleAgent continues to own history. Summary inputs use the portable text/image
and tool-history projection, excluding opaque native state; only final summary
content becomes model memory. Successful compaction clears **all working native
state**, including state attached to retained recent messages. Immutable archives
keep the original bounded state. This existing policy can remove reasoning
before its first replay; it does not imply an unexplained transport loss.
With explicit `whole_turn_v1` or `message_chunks_v1`, an unfinished native
host-tool continuation protects its prefix from rewriting; if it cannot fit,
the decision fails within the existing limits.

The default `simple_agent.context.compaction_policy: tool_rounds_v1` can restart
inside that continuation at a closed assistant call batch and all its results,
without waiting for `submit_action`. It first preserves visible tool history,
including original arguments, result text and images, while retiring opaque
provider state with an audit event. The latest unread batch reaches the next main
model request intact; only older eligible history may be summarized. A restart
may require no model-summary call. Protected inputs/events plus new results that cannot
fit fail explicitly. This evaluation condition leaves the action-output,
native-state and byte caps unchanged; an individually oversized response or call
ID can still fail. See the
[three-policy comparison and configuration snippet](evaluation-guide.md#recommended-simpleagent-context-compaction-and-host-tools)
for the session, budget and immutable-record guarantees.
The provider evidence below does not validate this new policy on cloud Gemini
or a full packaged task.

Saved model requests identify `request_transport.kind: logical_sdk_request` and the
`acompletion` or `aresponses` method. They show what AgentArk supplied to LiteLLM,
not a capture of the final provider HTTP payload. Response metadata and native
state remain separate from public assistant content and the submitted task
action. HumanView/replay can show the saved calls, summaries and events; they
do not reconstruct missing upstream traffic or reveal opaque reasoning.

## Resume, replay and existing evaluations

Choosing LiteLLM creates a distinct evaluation condition. Its backend, normalized
SDK model, API mode and endpoint participate in the result fingerprint alongside
the existing harness settings. Use a new output path when changing conditions;
native results cannot silently satisfy LiteLLM resume checks. An accepted
Responses alias normalizes to the same SDK-model identity.

Omitting `backend` and explicitly selecting `backend: native` retain the native
transport. SimpleAgent's new `tool_rounds_v1` default intentionally changes its
fingerprint; explicitly select `message_chunks_v1` or `whole_turn_v1` to reproduce
the corresponding older behavior and identity. APIAgent, CodexAgent,
SimpleAgent's native OpenRouter handling and RL integration retain their
existing behavior.
Historical records remain readable. Saved-action replay uses `ReplayAgent` and
calls no model backend; it does not require the LiteLLM extra or restore private
model working state. Compatible task/runtime artifacts, seeds and parameters are
still required for replay. Dependency upgrades alone do not prove identical
remote model behavior or scores.

## Validation scope

The evidence types below establish different scopes. Historical packaged runs
predate the uniform Kaggle Responses/auto policy and do not certify that new
policy across vendors. A provider probe is not a packaged task evaluation.

| Check | Established scope |
| --- | --- |
| Synthetic: offline fake-SDK and local HTTP loopback | Chat and Responses requests, multimodal/tool continuation, signatures/state validation, ordered Responses items including phases, retries, limits, compaction, trace and evaluation-identity behavior. These controlled responses do not certify provider support. |
| Production-adapter v2 probes / Responses | Sol 6.1 (`openai/gpt-6.1-sol`, run 6231754), Opus 5.5 (`anthropic/claude-opus-5-5@default`, run 6231755), and Gemini 3 Flash Preview (`google/gemini-3-flash-preview`, run 6231756) passed. Each probe made three actual production `aresponses` calls: two image turns covering a host echo result, native continuation and action submission, then a third tool-free image summary. All used `auto`, omitted temperature and reasoning effort, and zero retries. These are small protocol smokes, not Unity or full `DoorSpin3Turns` evaluations. |
| Production-adapter probe: Kaggle Gemini 2.5 Flash (`google/gemini-2.5-flash`) / Responses | A request with two function tools returned HTTP 400 `invalid_prompt`: "Multiple tools are supported only when they are all search tools". Use the explicit Chat API override when configuring this route; no automatic fallback is added. |
| Production-adapter v2 probe: Kaggle Grok 4.6 (`xai/grok-4.6`) / Responses | The first request returned HTTP 404 "requested model not found". The child receipt records failure even though the outer benchmark reported completion. Chat is an explicit configurable alternative, not a live-verified result. |
| Raw OpenAI SDK probe: Kaggle Gemini 3 Flash Preview / Responses | A direct SDK call to `/models/openapi/responses` passed two tools and two turns, with 2,700 tokens reported. `/models/openapi/v1/responses` returned HTTP 405. This is transport-only evidence, not a production-adapter or packaged-task pass. |
| Historical real Kaggle Astra / Responses / `tool_choice: required` / `temperature: null` / packaged `DoorSpin3Turns`, seeds 1–2 | Both cases completed 10 Unity actions with raw final-attempt reward 1. Total: 20 action calls, 14 summary calls, zero model-call errors and zero SDK retries. |
| Reasoning audit in that real run | Seven returned reasoning items: four were replayed in six later requests; three were cleared by existing compaction before first replay. No additional unexplained omission was found. |
| Limits of the real run | Returned message phases were absent; phase preservation and intermediate host-tool continuations were covered by controlled loopback, not that run. Every real action directly submitted `submit_action`. |
| Historical real Kaggle Opus 5.5 / Chat Completions / `tool_choice: auto` / `temperature: 1.0` / `DoorSpin3Turns`, run 5800062 | Seed 1 completed 10 environment turns with raw final-attempt score 1.0 and final success. Seed 2 completed 4 environment turns, then an APIError stopped the run because the next request's estimated $0.647248 exceeded remaining Kaggle quota. The original forced-tool-choice HTTP 400 was resolved; this is partial two-seed validation interrupted by quota, not a task code error. The raw-output audit matched all 14 accepted native `submit_action` calls to environment actions; two mixed batches were rejected and corrected through existing behavior. |
| Unverified claims | No direct native-Responses A/B comparison, performance-parity result, all-vendor signature validation or blanket live certification of the provider recipes above. |

Evidence paths relative to the outer repository root are
`local_save/task_validation/SimpleAgent/litellm_door_20261102/`:
[execution report](../../local_save/task_validation/SimpleAgent/litellm_door_20261102/execution_report.json),
[reasoning continuation audit](../../local_save/task_validation/SimpleAgent/litellm_door_20261102/readonly_reasoning_continuation_audit.json)
and [end-to-end audit](../../local_save/task_validation/SimpleAgent/litellm_door_20261102/readonly_e2e_audit.json).
These reports and raw records, images and logs are local-only ignored evidence,
unavailable in a fresh clone. Their absence does not turn logical SDK traces
into wire captures.

The current v2 protocol-smoke receipts are summarized in the local-only
[verified probe report](../../local_save/task_validation/SimpleAgent/unified_kaggle_20261105/verified_v2_probes.json).
Per-run receipts are under
`local_save/task_validation/SimpleAgent/unified_kaggle_20261105/v2_outputs/`,
relative to the outer repository root. These are also unavailable in a fresh
clone and establish only the probe conditions listed above.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing LiteLLM / MCP / tenacity import | Install the pinned `.[litellm]` extra with the same Python executable used to run evaluation, then run `python -m pip check`. |
| Function-tool HTTP 400 | Verify the selected model and API route support function tools. The current LiteLLM Kaggle helper defaults every model to `auto`; manual YAML can set `tool_choice: auto` under `simple_agent`. Check explicit overrides and old imported notebook versions. Do not enable global parameter dropping. |
| Wrong model or model-not-found | Check the required nested provider prefix, top-level wire model, API mode and endpoint. Kaggle Astra needs nested `openai/openai/gpt-6-astra`; direct OpenAI needs `openai/gpt-6-astra`. |
| `openai/responses/` configuration error | Select `api: responses`; use the ordinary `openai/` route for new configs. Responses currently rejects other SDK vendor prefixes. |
| Unsupported temperature / thinking | Use `temperature: null` and remove native `thinking`. The current LiteLLM Kaggle helper already omits temperature and optional reasoning. Check explicit overrides; add `reasoning_effort` only where supported. |
| Unexpected credential or endpoint | Set `api_key_env` explicitly and inspect the logical trace's endpoint. Omitted `base_url` means the SDK provider default; an unset named key is an error. |
| Global modification/fallback error | Clear SDK-wide automatic dropping, modification, fallback and retry settings in the calling process; choose conditions explicitly in AgentArk config. |
| Reasoning absent after a summary | Inspect compaction events and archived native state; successful compaction intentionally clears all working native state. |
| Old results are not skipped | Check backend/API/model/harness fingerprints and use a separate output path for the intended condition. Saved-action replay remains available. |
