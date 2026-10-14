# Attempt initial images: validation

Validated on 2026-10-14. This change adds an optional Python-wrapper image
attachment; Unity task code, camera sensors, task configs, and the immutable
Mods-root `config.yaml.bak` were not changed.

## Regression results

- Feature, rollout, config, function-mode, camera, HTTP protocol, trajectory,
  and interaction-hook suites: **204 passed; 91 subtests passed**.
- API and Codex client contracts use test doubles; no model requests are made.
- Independent review reproduced and verified fixes for CMYK JPEG transport,
  late-appearing agents, and delayed attachments with current history disabled.
  Default-off message comparisons against the original modules were identical.
- `git diff --check` passed.

Before the separate feature commit, the same 204-test/91-subtest gate passed
against a clean HEAD archive overlaid with only the 15 feature files. The
existing Code Runtime tests additionally use the unchanged outer repository's
config-audit script as a test fixture. No uncommitted SimpleAgent, evaluator,
registry, or trajectory-code changes were included in that snapshot.

## Actual packaged Player comparison

The reusable runner is
[`manual_attempt_initial_images_runtime.py`](../tests/manual_attempt_initial_images_runtime.py).
It accepts existing runtime, Mods, source, pool, and output paths as arguments;
it uses a copied Windows Player and scripted actions, with no model requests.
The comparison source uses the original versions of changed modules while
retaining unrelated pre-existing edits elsewhere.

| Existing task | Seed | Actions | Final reward | Done | Truncated |
| --- | ---: | ---: | ---: | --- | --- |
| ConnectionCounterplay2D | 17 | 4 | 1 | true | false |
| LocalSeatShortageCertificate2D | 2 | 14 | 1 | true | false |
| OneItemRewind2D | 17 | 1 (GiveUp) | 0 | true | true |

All 22 observations and 19 action results matched between original and updated
sources: camera pixel hashes, message hashes, observation text, rewards, done,
and truncated. Native-size reset and final frames were visually inspected.
GiveUp's existing truncated outcome was preserved.

An enabled-image probe ran three attempts, including two automatic resets.
Each attempt received both configured images before its first action; ordinary
steps received no new attachment. In the same Player process, switching to an
existing task that omits the option cleared both images and effective config.
The old source images had been deleted before that switch. Effective YAML,
JSON, and runtime configuration matched; the backup hash stayed unchanged.

Local evidence, relative to the outer repository root:
`local_save/task_validation/attempt_initial_images/20261014_generic/` contains
`baseline_final/report.json`, `updated/report.json`, `comparison.json`, and
native-size frames. These ignored local outputs are unavailable in a fresh
clone. The runner and automated tests are retained for reproduction.

Task107 adoption and any model painting run are separate from this generic
wrapper validation.
