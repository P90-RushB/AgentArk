# SimpleAgent native reasoning history

SimpleAgent's HTTP backend uses one fixed contract: OpenRouter Chat
Completions messages and function tools, with OpenRouter reasoning extensions.
The existing `provider: openai` setting selects this HTTP backend; it does not
promise compatibility with every endpoint that describes itself as
OpenAI-compatible. No vendor-specific dispatch or automatic fallback is added.
Endpoints that reject these fields can fail the request. Successful HTTP
acceptance alone does not prove that an endpoint uses the reasoning fields.

## Messages and audit records

Native action instructions require a brief, user-facing action rationale (one or
two sentences) alongside the final `submit_action` of every environment step.
Use the HTTP assistant message's `content`, or the Codex transport envelope's
`text`, for this explanation of the action and its purpose. It is separate from
private model reasoning and uses no `<think>` wrapper. The `submit_action`
argument also named `content` contains only the task action payload. This is a
prompt requirement; it does not add a hard rejection of empty response content.

- `ModelCompletion.text` and assistant text parts contain ordinary response
  content only. Provider reasoning is never promoted to reply content or used
  as an action when content and tool calls are absent.
- A detached, immutable `NativeAssistantState` retains the original assistant
  content, including null and whitespace, and exact function-call IDs and
  argument strings. Its endpoint/model scope must match the HTTP backend.
- Nonempty `reasoning_details` takes precedence and is replayed without
  editing its objects or their order. Otherwise a nonempty `reasoning` string
  is replayed; the input alias `reasoning_content` maps to `reasoning`.
  Missing, null, or empty details do not create native replay state.
- Portable message encoding, summary input and Codex SDK input exclude this
  API-only state. The SDK transport continues to use its own protocol.
- Request traces use the actual HTTP message projection. Response traces
  preserve native assistant state separately from ordinary content. Enabled
  record tools publish `native_assistants.json` alongside `assistant.txt` and
  `action.txt` when a decision has native responses. The native file is an
  audit record, not an instruction to copy opaque data into reply text.
  Existing record file-count and byte quotas apply to it.

## Limits and compaction

Native state has a 1 MiB per-message hard bound and participates in action
output, working-context and immutable-archive limits. Token accounting includes
a conservative allowance for this API-only state, including when a portable
custom counter is supplied; the request then reports its count as estimated.

Successful compaction starts a new working prefix and removes native replay
state from retained messages as well as summarized messages. The immutable
originals remain available in the archive and audit records. Failed or
cancelled summaries do not commit this change. Summaries retain only their
ordinary response content as model notes.

Once an unfinished host-tool turn has produced native state, the runtime keeps
that turn's prefix unchanged through its continuation. If it exceeds the
context budget, it raises a bounded limit error instead of rewriting a prefix
that an opaque signature may depend on. A new case/reset clears working state;
an attempt that carries an unchanged history can retain native state.

APIAgent's existing response parsing and history behavior are unchanged.
Offline fake-client tests check HTTP request fields, tool continuation,
submission, records, compaction and budget boundaries. They do not establish
compatibility with every remote provider or validate remote signature checks.

Protocol reference:
[OpenRouter reasoning preservation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning).
