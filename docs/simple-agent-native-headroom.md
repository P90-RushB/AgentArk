# Native fresh-decision headroom

`SimpleRuntime` now attempts compaction before starting a native tool decision
when its working context exceeds 75% of the existing character or encoded-byte
cap. Tool declarations and opaque native replay state count toward that target.
The headroom watermark itself retains the token, image, summary, action and
model/tool-call budgets and adds no configuration field. Tools-disabled
operation keeps its prior context-admission policy. The separate output-budget
policy described below is an independent later change.

The watermark reserves best-effort headroom. It cannot guarantee that every tool
round or arbitrarily large batch will fit. Protected inputs, instructions,
required feedback or the existing summary allowance may prevent reaching it;
if the context still fits every hard bound, the new decision proceeds. A native
continuation that later exceeds a hard bound still stops without compacting its
signature-bound prefix under explicit `whole_turn_v1` or `message_chunks_v1`.
Compaction uses the existing native-state invalidation and immutable-archive
rules.

The default `tool_rounds_v1` policy can restart at closed tool batches during an
unfinished decision without waiting for `submit_action`. It preserves visible
tool history before considering summaries of older eligible history. The latest
unread results keep their original text/images for the next main model request;
protected inputs/events plus new results that cannot fit fail explicitly. Retiring
opaque provider state alone may require no model-summary call. Single-response
admission now separates semantic action output from native replay using
`output_budget_policy: separate_native_v1`; ID and native-state bounds still
apply, with a separate bounded aggregate protocol allowance. All replay data
continues to count toward context/token estimates and archive limits. A response
exceeding its applicable admission bound still fails before compaction. See the
[policy comparison and configuration snippet](evaluation-guide.md#recommended-simpleagent-context-compaction-and-host-tools)
for context restart and record/session guarantees.
The historical offline results below describe the headroom change, not cloud
Gemini or full packaged-task validation of `tool_rounds_v1`.

For offline restoration of a recorded old-policy prefix only, an instance may
set `_fresh_decision_fits = runtime._fits`. Delete that instance override before
starting the fresh decision on the new policy. Restore complete successful
inputs/actions, excluding the abandoned decision's partial tool prefix. Mocked
summaries used for verification are disposable test data and must not become
production resume state.
When constructing `SimpleRuntimeConfig` from a historical saved context that
predates output-budget separation, explicitly add
`output_budget_policy="legacy_v1"` for exact old admission behavior. Also set
`round_budget_policy="legacy_v1"` when reconstructing requests from before runtime
round notices were added. These settings are only for offline reconstruction;
normal new evaluations keep the new defaults.

## Offline verification

`tests/test_simple_agent_native_headroom.py` covers exact synthetic size
regressions: 489,182 + 63,660 = 552,842 characters and 15,943,648 + 58,543 =
16,002,191 bytes. The prior hard-only policy stops inside the native turn; the
new policy summarizes before opening it and completes the same valid host
rounds. Tests also check unchanged continuation prefixes, inclusive watermarks
and hard bounds, protected-input/summary-budget fallback, tools-disabled policy,
wire-message equality below pressure, soft-watermark crossings inside a native
continuation, and successful-summary fallback after the summary allowance is
exhausted.

The SimpleAgent suite, native reasoning/Volcengine, submission recovery, summary,
Codex SDK adapter and agent-selection tests were run offline: 618 passed, one
skipped, and 31 subtests passed in the existing Linux environment. The wheel
resource test requires `pip`, absent from that interpreter; the same test passed
separately in the existing Windows Python environment (one test, 8.37 seconds).
No environment or dependencies were installed. Two additional policy regressions
were then added and the focused headroom suite passed all 14 cases.

A separate local-only proof decoded and hash-checked the actual Opus and DeepSeek
audits. Both original fresh-decision snapshots fit hard bounds and exceed the
new watermark. One mocked summary of the exact Opus working prefix reduced it
from 489,182 characters / 5,039,209 bytes to 18,962 / 855,704. It preserved original
archive/native records and the full hard-limit configuration. External socket,
DNS, HTTP and process-launch guards were active; no model or environment ran.
Raw snapshots and receipts remain outside source control and are unavailable in
a fresh clone.
