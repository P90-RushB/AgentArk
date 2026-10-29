#!/usr/bin/env python3
"""Generate a new SimpleAgent Kaggle task without changing historical tasks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "agentark_simple_task_template.py"
DEFAULT_WRAPPER_COMMIT = "71a9cc305e3e87c0c5ed416086586e24db35210b"


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
    wrapper_commit: str = DEFAULT_WRAPPER_COMMIT,
) -> Path:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task_name):
        raise ValueError("task_name must be an AgentArk task identifier")
    if seed_start < 0 or seed_end < seed_start or env_id < 0:
        raise ValueError("require 0 <= seed_start <= seed_end and env_id >= 0")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", wrapper_commit):
        raise ValueError("wrapper_commit must be a full immutable 40-character Git commit")
    task_slug = _slugify(task_name)
    slug = f"agentark-{task_slug}"
    func_name = _function_name(slug)
    output_path = output or ROOT / f"{func_name}.py"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing benchmark: {output_path}")

    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    text = text.replace("agentark_task_seeds_1_10", func_name)
    text = _replace_once(text,
        'DEFAULT_TASK_NAME = os.getenv("AGENTARK_KAGGLE_TASK_NAME", "MarbleStop").strip() or "MarbleStop"',
        f'DEFAULT_TASK_NAME = os.getenv("AGENTARK_KAGGLE_TASK_NAME", {json.dumps(task_name)}).strip() or {json.dumps(task_name)}')
    text = _replace_once(text, f'AGENTARK_COMMIT = "{DEFAULT_WRAPPER_COMMIT}"',
                         f'AGENTARK_COMMIT = "{wrapper_commit.lower()}"')
    text = _replace_once(text, "DEFAULT_SEED_START = 1", f"DEFAULT_SEED_START = {int(seed_start)}")
    text = _replace_once(text, "DEFAULT_SEED_END = 10", f"DEFAULT_SEED_END = {int(seed_end)}")
    text = _replace_once(text, "DEFAULT_ENV_ID = 0", f"DEFAULT_ENV_ID = {int(env_id)}")
    text = _replace_once(text,
        'DEFAULT_BENCHMARK_TASK_NAME_FALLBACK = f"agentark-{DEFAULT_TASK_SLUG.replace(\'_\', \'-\')}"',
        f'DEFAULT_BENCHMARK_TASK_NAME_FALLBACK = {json.dumps(slug)}')

    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="AgentArk task name, e.g. MarbleStop")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--wrapper-commit", default=DEFAULT_WRAPPER_COMMIT,
                        help="Full immutable SimpleAgent-capable wrapper commit; never a moving branch")
    args = parser.parse_args()

    output_path = generate(
        task_name=args.task_name,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        env_id=args.env_id,
        output=args.output,
        wrapper_commit=args.wrapper_commit,
    )
    print(output_path)


if __name__ == "__main__":
    main()
