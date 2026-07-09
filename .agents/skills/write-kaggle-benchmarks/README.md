# Write Kaggle Benchmarks Skill

This skill teaches AI coding agents how to write, push, run, and manage Kaggle Benchmark tasks using the `kaggle` CLI and the `kaggle-benchmarks` Python SDK.

## What this skill enables

With this skill installed, an agent can go from a blank file to a deployed benchmark in a single conversation:

```bash
# Write a task → validate locally → push → run → download results
kaggle b init -y
python task.py
kaggle b t push my-task -f task.py --wait
kaggle b t run my-task -m google/gemini-3.5-flash --wait
kaggle b t download my-task -o ./results
```

## Prerequisites

- Python 3.11+
- `kaggle` CLI installed (`pip install kaggle`)
- `kaggle-benchmarks` SDK installed (`pip install kaggle-benchmarks`)
- Valid Kaggle credentials (`kaggle auth login`, `KAGGLE_API_TOKEN` env var, or `~/.kaggle/access_token`)

## Structure

```
write-kaggle-benchmarks/
├── SKILL.md    # Complete skill file — quick reference, CLI docs, and core workflow
└── README.md   # This file
```

## Installation

Point your AI coding agent to read `SKILL.md`.

**Gemini CLI** — Copy to the auto-discovery location:
```bash
cp -r write-kaggle-benchmarks ~/.gemini/skills/write-kaggle-benchmarks
```

**Other tools** — Ask your agent to read `write-kaggle-benchmarks/SKILL.md`.

## Official resources

- [kaggle-benchmarks SDK](https://github.com/Kaggle/kaggle-benchmarks) — full source, API reference, and examples
- [SDK auto-generated docs](https://deepwiki.com/Kaggle/kaggle-benchmarks)
- [CLI benchmarks docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md)

## Advanced use cases

`SKILL.md` is a focused recipe for the common write-push-run-download loop. For deeper SDK and CLI surface area not covered here, point your agent at the upstream skill files:

- [`kaggle-benchmarks/SKILL.md`](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/skills/kaggle-benchmarks/SKILL.md) — complete SDK reference for benchmarks: every class, method, and pattern beyond the core workflow shown here.
- [`kaggle-cli/skills/references/benchmarks.md`](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/benchmarks.md) — complete CLI reference for benchmarks: every subcommand, flag, and option beyond the core workflow shown here.

## Version

`0.2`
