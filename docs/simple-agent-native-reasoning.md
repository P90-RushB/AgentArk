# SimpleAgent native reasoning history

SimpleAgent's HTTP backend uses Chat Completions messages and function tools,
with closed OpenRouter and Volcengine native reasoning field families.
The existing `provider: openai` setting selects this HTTP backend; it does not
promise compatibility with every endpoint that describes itself as
OpenAI-compatible. Response fields select the supported family automatically;
no new platform setting, URL-based dispatch or automatic request fallback is added.
Endpoints that reject these fields can fail the request. Successful HTTP
acceptance alone does not prove that an endpoint uses the reasoning fields.

## Messages and audit records

Native action instructions require a brief, user-facing action rationale (one or
two sentences) alongside the final `submit_action` of every environment step.
Use the HTTP assistant message's `content`, or the Codex transport envelope's
`text`, for this explanation of the action and its purpose. It is separate from
private model reasoning and uses ordinary prose without explanation tags. The
`submit_action` argument also named `content` contains only the task action
payload. This is a prompt requirement; it does not add a hard rejection of empty
response content.
SimpleAgent's own transport instructions describe this rule without literal
reasoning tags. Supplied task instructions and literal task payloads remain
unchanged, including legacy response-format text.

- `ModelCompletion.text` and assistant text parts contain ordinary response
  content only. Provider reasoning is never promoted to reply content or used
  as an action when content and tool calls are absent.
- A detached, immutable `NativeAssistantState` retains the original assistant
  content, including null and whitespace, and exact function-call IDs and
  argument strings. Its endpoint/model scope must match the HTTP backend.
- Nonempty OpenRouter `reasoning_details` retains first priority and is replayed
  without editing its objects or their order. Foreign `encrypted_content` fields
  do not affect this existing path, including when their shape is unsupported.
- Otherwise, an explicitly supplied `encrypted_content` selects the Volcengine
  family. The assistant replays that field and any explicitly supplied
  `reasoning_content` under their original names and with unchanged string or
  null values. Ciphertext is never decoded, trimmed, truncated or summarized.
  The summary is never renamed to `reasoning` or copied into assistant content.
  A ciphertext-only response creates native state. Missing, null and empty
  summary fields remain distinct, as do null and empty ciphertext fields; even
  an explicitly empty family retains native state and its conservative bounds.
- One compatibility exception preserves existing OpenRouter behavior: when
  `encrypted_content` is null or empty, `reasoning_content` is absent, and a
  nonempty `reasoning` string exists, the original `reasoning` path wins.
  Without an explicit `encrypted_content`, the previous nonempty `reasoning`
  path and `reasoning_content`-to-`reasoning` alias remain unchanged. Missing,
  null or empty details alone still do not create native replay state.
- Native state accepts exactly one of these field families: `reasoning_details`,
  `reasoning`, or `encrypted_content` with optional `reasoning_content`. Mixed
  families and unknown replay fields are rejected. A selected encrypted family
  requires string or null values; malformed state fails before host tools run.
  SDK optional defaults are distinguished from explicit fields using field-set
  and extra-field metadata. SDK message extras are never copied wholesale.
- Portable message encoding, summary input and Codex SDK input exclude this
  API-only state. The SDK transport continues to use its own protocol.
- Request traces use the actual HTTP message projection. Response traces
  preserve native assistant state separately from ordinary content. Enabled
  record tools publish `native_assistants.json` alongside `assistant.txt` and
  `action.txt` when a decision has native responses. The native file is an
  audit record, not an instruction to copy opaque data into reply text.
  Existing record file-count and byte quotas apply to it.
- The original assistant function calls and their IDs remain paired with the
  runtime's tool results. Native state is replayed on host-tool continuations
  and subsequent environment steps only within its original endpoint/model
  scope. The existing scope format is unchanged.

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

Protocol references:
[OpenRouter reasoning preservation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning),
[Volcengine deep thinking](https://docs.volcengine.com/docs/ark/deep-thinking?lang=zh),
[Volcengine Chat API](https://docs.volcengine.com/docs/ark/chat-api?lang=en).
