# SimpleAgent pre-commit validation

Validated on 2026-10-14 against HEAD `d78f707` plus the pending SimpleAgent
implementation, evaluator integration, documentation and tests.

## Automated regression

The complete Python test directory passed in a clean archive overlaid with only
the candidate files:

```text
python -m pytest tests -q --maxfail=1
495 passed, 3 skipped, 130 subtests passed
```

The existing Code Runtime tests also require the outer repository's unchanged
task-config audit script; it was copied as a read-only fixture alongside the
snapshot. The three skips concern filesystem link fixtures unavailable in this
Windows environment. Seven existing warnings intentionally exercise missing
immutable-backup fallback fixtures.

Coverage includes bounded memory/compaction, cancellation and limits, native
submission, controlled file/image/note tools, case isolation, agent selection,
legacy API/Codex behavior, parallel evaluation, hooks, trajectory/replay and
the separately committed attempt-start image mechanism. API calls use test
doubles or local test services. No model, Unity Editor or Player was run for
this pre-commit check.

## Independent compatibility review

Read-only HEAD/current comparisons passed for 16 legacy API/Codex model
configurations, including constructor arguments, runtime metadata and invalid
configuration errors. Legacy trace/error normalization and mocked parallel
setup, results, trajectory calls and cleanup matched for the auto, OpenRouter,
Codex and human paths.

APIAgent/CodexAgent implementation files and environment code are unchanged
by this commit. Without selecting a new agent, provider-based selection and
environment message settings retain their prior behavior. SimpleAgent-specific
delta settings apply only to its private environment config. Importing the
evaluator does not load SimpleAgent; the core/registry import without provider,
imaging or Unity dependencies. No actionable compatibility issue remained.

`git diff --check` and inspection for machine-specific paths, cached bytecode,
secrets and generated evidence passed. Only source, documentation, an example
configuration, license notice and tests belong to this commit.

Local-only snapshot/provenance is under
`local_save/task_validation/SimpleAgent/20261014_commit/`, relative to the outer
repository root, and is unavailable in a fresh clone. Earlier paid Qwen/Player
validation belongs to the separate 2026-10-13 report; no new full-episode or
model-performance result is claimed here.

## Summary memory, recovery and shared defaults — 2026-10-14

Summary requests now use only the provider's final `content` as memory. Separate
readable reasoning remains in raw response metadata and provider usage, including
when a reasoning-only summary fails. Action response handling is unchanged. No
new summary-length setting was added. Empty content, output truncation, excessive
summary content or a non-reducing summary now receives bounded corrective
feedback on the next summary request. Failed and successful summary attempts
share the existing four-call per-decision cap, lifetime call budget and deadline.
Feedback is included in input budgets; only complete source turns are selected.
Working history is committed only after the entire compaction succeeds.

The shared runtime defaults and general example now use a 128,000-token local
context estimate, 20,480-token action reserve, 32 images and two protected recent
inputs. Summary input/output caps are 48,000/10,000 tokens, with a 196,608-character
summary input cap. Archive total/image
byte caps are each 512 MiB, with 32 Mi characters of archived text; SimpleAgent's
effective record limit is 512. File tools remain explicit opt-in for direct
construction and are enabled by the general example.

The complete test directory passed with fake models and local test services:

```text
python -m pytest tests -q --maxfail=1
543 passed, 2 skipped, 130 subtests passed
```

The two skips are filesystem link fixtures. Seven existing warnings intentionally
exercise missing immutable-backup fallback fixtures. The new provider-response
tests cover final-content isolation, subsequent action inputs, raw reasoning and
usage retention, recovery from empty or incomplete summaries, retry feedback
budgets, rollback, exhaustion, cancellation, non-recoverable errors and action
compatibility. Generic injected chat backends also retain failed-attempt usage.
The existing evaluator's error recording and both stop-on-error policies were
tested without modifying evaluator code. The general example and Task107 profile
use `eval.stop_on_error: false` to continue other cases after recovery is exhausted;
no placeholder environment action is submitted for a failed summary.

An offline constructor/parameter check confirmed that both examples match shared
context defaults and the effective action/summary output caps are 20,480/10,000.
No HTTP client was created by that check. Local-only test logs are under
`local_save/task_validation/SimpleAgent/20261014_summary_recovery/`, relative to the
outer repository root, and are unavailable in a fresh clone.

APIAgent, CodexAgent and environment implementations are unchanged. No real
model, Unity Editor or Player was run for this revision. Current defaults remain
local budget estimates, not a model-native context-window guarantee.
