#!/usr/bin/env python3
"""Generate an AgentArk Kaggle Benchmark task from the MarbleStop template."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "agentark_marble_stop_seeds_1_10.py"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError(f"Unable to derive slug from {value!r}")
    return slug


def _function_name(slug: str) -> str:
    return slug.replace("-", "_")


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Template token not found: {old!r}")
    return text.replace(old, new, 1)


def generate(
    *,
    task_name: str,
    seed_start: int,
    seed_end: int,
    env_id: int,
    output: Path | None,
) -> Path:
    task_slug = _slugify(task_name)
    slug = f"agentark-{task_slug}-seeds-{seed_start}-{seed_end}"
    func_name = _function_name(slug)
    output_path = output or ROOT / f"{func_name}.py"

    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    text = _replace_once(text, "AgentArk MarbleStop seeds 1-10", f"AgentArk {task_name} seeds {seed_start}-{seed_end}")
    text = text.replace("agentark_marble_stop_seeds_1_10", func_name)
    text = text.replace("kaggle_agentark_marble_stop_seeds_1_10", f"kaggle_{func_name}")
    text = _replace_once(text, 'DEFAULT_TASK_NAME = "MarbleStop"', f'DEFAULT_TASK_NAME = "{task_name}"')
    text = _replace_once(text, "DEFAULT_SEED_START = 1", f"DEFAULT_SEED_START = {int(seed_start)}")
    text = _replace_once(text, "DEFAULT_SEED_END = 10", f"DEFAULT_SEED_END = {int(seed_end)}")
    text = _replace_once(text, "DEFAULT_ENV_ID = 0", f"DEFAULT_ENV_ID = {int(env_id)}")
    text = text.replace('@kbench.task(name="agentark-marble-stop-seeds-1-10")', f'@kbench.task(name="{slug}")')
    text = text.replace(
        "Evaluate the Kaggle-selected model on AgentArk MarbleStop seeds 1-10.",
        f"Evaluate the Kaggle-selected model on AgentArk {task_name} seeds {seed_start}-{seed_end}.",
    )

    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="AgentArk task name, e.g. MarbleStop")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = generate(
        task_name=args.task_name,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        env_id=args.env_id,
        output=args.output,
    )
    print(output_path)


if __name__ == "__main__":
    main()
