# SimpleAgent validation status

Updated 2026-10-16 for nested AgentArk commit `47df8b3` (dedicated native prompts),
following native reasoning support in `3bd9d2c`. Historical runs retain their actual
model, effort and budgets; they do not establish results for other models or `high` effort.

## Current offline regression

Fake-model/local-service regression: `715 passed, 2 skipped, 7 warnings, 176 subtests passed`.
Skips concern Windows filesystem-link fixtures; warnings exercise missing immutable-backup fixtures.
Coverage includes memory/compaction, summary recovery, cancellation/limits, native
submission, field replay, records, isolation, evaluators and legacy agents. All eight
prompt resources were also tested in a real installed wheel.

At `47df8b3`, the separate legacy compatibility validator passed **9 unittest tests
and 160 differential cases** against reference `95c4138`: default/explicit
`legacy_think`, task/language/action parsing, errors and wrapper selection matched.
Its submission behavior functions remain AST-equal; native prompts intentionally
changed. It claims neither whole native-request equality nor identical model behavior.
APIAgent, CodexAgent and shared environment/template implementations are unchanged.

## Current source defaults

These values are read from `SimpleRuntimeConfig`; task overrides remain independent:

| Setting | Current default |
| --- | ---: |
| Estimated context tokens / action reserve | 262,144 / 20,480 |
| Working-context / summary-input encoded bytes | 16,000,000 / 16,000,000 |
| Working images / protected recent inputs | 256 / 2 |
| Summary input tokens / characters | 48,000 / 196,608 |
| Summary output tokens / characters / bytes | 32,768 / 32,000 / 128,000 |
| Summary calls per decision | 4 |
| Archive total bytes / image bytes | 512 MiB / 512 MiB |

Image count, estimated tokens, text and bytes are separate compaction triggers.
Encoded-byte budgets include base64 image expansion and leave room for the API
request envelope; raw-image bytes alone do not bound the HTTP payload.
Boundary tests retain 256 images, compact at 257 and archive every original image.
With the reserve, 241,664 input tokens fit; one more is rejected before a model
call when no prior turn can compact. These are local estimates, not a provider's
native context-window guarantee. Codex SDK output-token requests are guidance,
not an SDK-enforced cap; runtime text/byte/summary checks still apply.

Summary memory uses final response content only; reasoning stays in audit metadata.
Summary generation inherits the action reasoning/thinking parameters, with its own
output-token budget that is not reduced by the action `max_completion_tokens` cap.
Reasoning and content still share the provider's completion-token allowance. The
first summary prompt states that total budget and the final-content bounds.
Empty, truncated, excessive or non-reducing summaries receive bounded feedback within
the same call/deadline budgets. History commits only after successful compaction.
Direct file tools remain opt-in; the general example enables a 512-record bound.

## Native prompts and provider history — 2026-10-16

Commit `47df8b3` loads the main `native_system_prompt.txt` and seven mode-specific
TXT resources from SimpleAgent's own `info/` directory. Exact known wrapper pins
are adapted for native requests, previews and readable records; task rules/functions,
images, unknown/custom messages and nonnative paths are preserved. Final submission
uses native `submit_action`; Unity functions are payload choices, separate from host tools.

OpenRouter nonempty `reasoning_details` keeps precedence; existing plaintext
`reasoning` and its legacy alias remain compatible. Volcengine replays opaque
`encrypted_content` and any `reasoning_content` under their original names, including ciphertext-only
responses and missing/null/empty distinctions. Renaming the summary to `reasoning`
or putting either field into assistant content would lose that native contract.
The fixed families retain scope, content/tool-ID validation, sidecar budgets and
unfinished-tool protection. This does not certify every OpenAI-compatible endpoint
or remote signatures; see [native history/prompt boundaries](simple-agent-native-reasoning.md).

## Latest real API smoke — 2026-10-16

The packaged Task107 smoke at `47df8b3` used `doubao-seed-2-1-pro-260628`, Volcengine,
`reasoning_effort=medium`, thinking enabled, temperature 1.0, `tool_choice=auto`,
`brief_rationale` and seed 107. It completed **3 API calls and 3 environment steps**:
`unreachable_waypoint`, `unreachable_waypoint`, then `motion_complete`.
All rewards were 0, `done` remained false and no summary was requested. The runtime
audit passed 18/18 checks; native history matched in three replayed assistant entries.
The final response had no later request, so its replay was not verified. This is
interface smoke, not full-episode/artwork evidence or a Codex `high` effort result.

## Historical packaged SDK run — 2026-10-14/15

SimpleAgent `provider: codex`, **gpt-6.1-sol / medium**, seed 107, fixed brush/board,
two 960×720 workspace cameras plus an attempt-start image, completed three
same-seed attempts with a 3×128-step budget. This run used the then-current
**64-image threshold**, not today's 256-image default. It submitted after 32, 32
and 31 actions: **95 actions, 99 logical requests and four first-try summaries**,
with no host-tool continuations. Reward measures submission integrity; art quality is external.

| Attempt | Steps | Input tokens | Output tokens | Summary requests |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32 | 1,536,283 | 28,329 | 1 |
| 2 | 32 | 2,375,250 | 32,676 | 1 |
| 3 | 31 | 2,392,989 | 35,512 | 2 |
| Total | 95 | 6,304,522 | 96,517 | 4 |

The actual input peak was 96,504 tokens versus local estimate 86,827. Compaction
at 65 images reduced next action inputs to 28, 32, 31 and 32 images; summaries
consumed 207,101 input and 9,141 output tokens.
Cached input 63,104 and reasoning output 7,479 are subsets already in the totals.
No SDK-native compaction or built-in tool execution was observed; effective
YAML/JSON equality and the immutable Mods backup hash remained valid.

A recoverable SDK notification originally interrupted the process after 29 actions.
After the transport fix, those recorded completions were replayed without new
model calls through actual Player transitions. All 29 complete text/tool/image
fingerprints and the first fresh request matched. This reconstructs visible
inputs in a new Player, not a live process snapshot. Original paid replies count
once; replay callbacks add no model usage. One failed original request has unknown
usage. A discarded diagnostic used 85,995 tokens and a READY probe used 5,883,
separate from the 6,401,039-token logical drawing run. Known whole-session usage
is **6,492,917 tokens plus one unknown request**; SDK retries are not extra logical
requests, and their underlying HTTP-attempt count was not reported.

## Local evidence

Paths are relative to the outer repository root; raw evidence is ignored and
unavailable in a fresh clone. Earlier offline checkpoints remain under
`local_save/task_validation/SimpleAgent/{20261014_commit,20261014_summary_recovery,20261015_defaults_256k}/`.
Task107 evidence is under `local_save/task_validation/Task107_RobotPainting/`:
`20261016_seed_native_prompt/` holds the latest report, audits and pytest log;
its `legacy_think/compatibility_evidence.json` records the current 9-test/160-case check.
`20261014_simple_codex64/real_run/analysis/session_report.md` and its
`recovered_logical/report.md` retain complete usage/replay scopes. Media is in
`20261014_simple_codex64/recovered_run/delivery/`: inference waits are omitted and
brush-follow recordings are human-only. This update runs no model/Unity or new performance test.
