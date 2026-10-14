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
