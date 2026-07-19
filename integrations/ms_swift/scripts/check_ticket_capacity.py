#!/usr/bin/env python3
"""Validate unique AgentArk ticket capacity for an ms-swift GRPO run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and validate unique group rows from ms-swift 4.4.1's "
            "generation-batch reuse schedule."
        )
    )
    parser.add_argument("--dataset", type=Path, help="Ticket JSONL to validate")
    parser.add_argument("--max-steps", type=_positive_int, required=True)
    parser.add_argument("--per-device-train-batch-size", type=_positive_int, required=True)
    parser.add_argument("--world-size", type=_positive_int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, required=True)
    parser.add_argument("--num-generations", type=_positive_int, required=True)
    parser.add_argument(
        "--num-iterations",
        type=_positive_int,
        default=1,
        help="Number of policy updates that reuse each generated batch",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=_positive_int,
        default=None,
        help=(
            "Explicit Swift generation_batch_size. When omitted, use "
            "per_device_train_batch_size * world_size * gradient_accumulation_steps."
        ),
    )
    parser.add_argument(
        "--reserve-percent",
        type=float,
        default=0.0,
        help="Extra unique rows to reserve, rounded up to a full prompt-group batch",
    )
    parser.add_argument(
        "--print-required",
        action="store_true",
        help="Print only the aligned required row count; --dataset is then optional",
    )
    return parser.parse_args()


def _capacity(args: argparse.Namespace) -> dict[str, int | float]:
    if args.num_generations < 2:
        raise SystemExit("GRPO requires --num-generations >= 2")
    if args.reserve_percent < 0:
        raise SystemExit("--reserve-percent must be non-negative")

    default_generation_batch_size = (
        args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
    )
    generation_batch_size = args.generation_batch_size or default_generation_batch_size
    global_train_batch_size = args.per_device_train_batch_size * args.world_size
    if args.generation_batch_size is not None and generation_batch_size % global_train_batch_size != 0:
        raise SystemExit(
            "Explicit generation_batch_size must be divisible by the global train batch size: "
            f"{generation_batch_size} % {global_train_batch_size} != 0"
        )
    if generation_batch_size % args.num_generations != 0:
        raise SystemExit(
            "generation_batch_size must be divisible by num_generations: "
            f"{generation_batch_size} % {args.num_generations} != 0"
        )

    steps_per_generation = generation_batch_size // global_train_batch_size
    reuse_micro_steps = steps_per_generation * args.num_iterations
    training_micro_steps = args.max_steps * args.gradient_accumulation_steps
    generation_calls = (training_micro_steps + reuse_micro_steps - 1) // reuse_micro_steps
    groups_per_batch = generation_batch_size // args.num_generations
    base_required = generation_calls * groups_per_batch
    with_reserve = math.ceil(base_required * (1.0 + args.reserve_percent / 100.0))
    aligned_required = math.ceil(with_reserve / groups_per_batch) * groups_per_batch
    return {
        "generation_batch_size": generation_batch_size,
        "default_generation_batch_size": default_generation_batch_size,
        "global_train_batch_size": global_train_batch_size,
        "steps_per_generation": steps_per_generation,
        "num_iterations": args.num_iterations,
        "training_micro_steps": training_micro_steps,
        "generation_reuse_micro_steps": reuse_micro_steps,
        "generation_calls": generation_calls,
        "groups_per_batch": groups_per_batch,
        "base_required_rows": base_required,
        "reserve_percent": args.reserve_percent,
        "required_rows": aligned_required,
    }


def _load_dataset(path: Path) -> tuple[int, set[str], list[str]]:
    if not path.is_file():
        raise SystemExit(f"Ticket dataset not found: {path}")

    group_uids: set[str] = set()
    errors: list[str] = []
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            rows += 1
            try:
                row: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_number}: row must be an object")
                continue
            env_config = row.get("env_config")
            if not isinstance(env_config, dict):
                errors.append(f"line {line_number}: missing object env_config")
                continue
            if env_config.get("name") != "agentark":
                errors.append(f"line {line_number}: env_config.name must be 'agentark'")
            group_uid = env_config.get("group_uid")
            if not isinstance(group_uid, str) or not group_uid.strip():
                errors.append(f"line {line_number}: missing non-empty env_config.group_uid")
                continue
            if group_uid != group_uid.strip():
                errors.append(f"line {line_number}: group_uid must not have leading or trailing whitespace")
            if group_uid in group_uids:
                errors.append(f"line {line_number}: duplicate group_uid {group_uid!r}")
            group_uids.add(group_uid)

            task_name = env_config.get("task_name")
            group_seed = env_config.get("group_seed")
            if task_name is not None and (not isinstance(task_name, str) or not task_name.strip()):
                errors.append(f"line {line_number}: task_name must be a non-empty string when provided")
            elif isinstance(task_name, str) and task_name != task_name.strip():
                errors.append(f"line {line_number}: task_name must not have leading or trailing whitespace")
            if group_seed is not None and (
                isinstance(group_seed, bool)
                or not isinstance(group_seed, int)
                or not 1 <= group_seed <= 2**31 - 2
            ):
                errors.append(f"line {line_number}: group_seed must be an integer in [1, {2**31 - 2}]")
            if isinstance(task_name, str) and task_name.strip() and group_seed is None:
                errors.append(
                    f"line {line_number}: pinned task_name requires group_seed so all G trajectories "
                    "reset to the same task state"
                )
            if env_config.get("env_id") is not None:
                errors.append(
                    f"line {line_number}: env_id must not be pinned because Swift repeats one ticket G times"
                )

            for media_field in ("images", "audios", "videos"):
                if row.get(media_field):
                    errors.append(
                        f"line {line_number}: top-level {media_field} is not supported; "
                        "AgentArk supplies inline media after reset"
                    )

            messages = row.get("messages")
            expected_placeholder = f"<agentark-ticket:{group_uid}>"
            if not (
                isinstance(messages, list)
                and len(messages) == 1
                and isinstance(messages[0], dict)
                and messages[0].get("role") == "user"
                and messages[0].get("content") == expected_placeholder
            ):
                errors.append(
                    f"line {line_number}: messages must contain the unique placeholder "
                    f"{expected_placeholder!r}"
                )
    return rows, group_uids, errors


def main() -> None:
    args = _parse_args()
    capacity = _capacity(args)
    if args.print_required:
        print(capacity["required_rows"])
        return
    if args.dataset is None:
        raise SystemExit("--dataset is required unless --print-required is used")

    rows, group_uids, errors = _load_dataset(args.dataset)
    groups_per_batch = int(capacity["groups_per_batch"])
    required_rows = int(capacity["required_rows"])
    if rows < required_rows:
        errors.append(f"dataset has {rows} rows but this run requires at least {required_rows}")
    if rows % groups_per_batch != 0:
        errors.append(
            f"dataset rows ({rows}) must be a multiple of the prompt-group batch "
            f"({groups_per_batch}); RepeatSampler drops the tail"
        )
    if len(group_uids) != rows:
        errors.append(f"dataset has {rows} rows but only {len(group_uids)} unique group_uid values")

    result = {
        **capacity,
        "dataset": str(args.dataset),
        "dataset_rows": rows,
        "unique_group_uids": len(group_uids),
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
