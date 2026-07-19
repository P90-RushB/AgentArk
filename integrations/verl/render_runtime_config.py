#!/usr/bin/env python3
"""Render the external VERL recipe's resolved env config for AgentArk warmup."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from omegaconf import OmegaConf
except ImportError as exc:  # pragma: no cover - exercised by the CLI environment
    raise SystemExit("OmegaConf is required; run this with the AgentArk Python environment") from exc


DEFAULT_ENV_CONFIG = Path("agentark_recipe/agentark_env_agent/config/env_cfg.yaml")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must resolve to a mapping")
    return value


def _source_path(ver_root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("--env-config must be a safe path relative to --verl-root")
    root = ver_root.expanduser().resolve(strict=True)
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError(f"VERL env config must not be a symlink: {unresolved}")
    source = unresolved.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError("VERL env config escapes --verl-root") from exc
    if not source.is_file():
        raise ValueError(f"VERL env config is not a regular file: {source}")
    return source


def render(ver_root: Path, env_config: Path, num_envs: int | None) -> dict[str, Any]:
    source = _source_path(ver_root, env_config)
    loaded = OmegaConf.load(source)
    resolved = OmegaConf.to_container(loaded, resolve=True)
    root = _mapping(resolved, "VERL env config")
    server = _mapping(root.get("server"), "server")
    env_cfg = _mapping(root.get("env_cfg"), "env_cfg")

    sandbox = env_cfg.get("runtime_sandbox", {})
    sandbox = sandbox if isinstance(sandbox, dict) else {}
    pool_size_raw = sandbox.get("pool_size")
    try:
        pool_size = int(pool_size_raw) if pool_size_raw is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("env_cfg.runtime_sandbox.pool_size must be an integer") from exc

    if num_envs is None:
        if pool_size is None or pool_size <= 0:
            raise ValueError(
                "--num-envs is required when env_cfg.runtime_sandbox.pool_size is not positive"
            )
        num_envs = pool_size
    if num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    return {
        "server": server,
        "warmup": {
            "num_envs": num_envs,
            "step_once": False,
            "close_envs_after_warmup": False,
            "env_id_prefix": "verl-v1-warmup-env",
            "port_assignment": {
                "assign_worker_index": True,
                "assign_env_id": False,
                "assign_base_port": False,
                "base_port_start": 5005,
                "base_port_stride": 1,
                "base_port_plan": [],
            },
        },
        "env_cfg": env_cfg,
        "_agentark_verl_bridge": {
            "source": os.fspath(source),
            "protocol_version": "v1",
            "configured_sandbox_pool_size": pool_size,
            "sandbox_auto_expands_for_worker_index": bool(
                sandbox.get("enabled") and sandbox.get("auto_prepare") and pool_size is not None
                and num_envs > pool_size
            ),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verl-root",
        type=Path,
        default=Path(os.environ["VERL_ROOT"]) if os.environ.get("VERL_ROOT") else None,
        help="VERL checkout root (or set VERL_ROOT)",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=DEFAULT_ENV_CONFIG,
        help="Path relative to --verl-root",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.verl_root is None:
        parser.error("--verl-root is required when VERL_ROOT is not set")
    try:
        rendered = render(args.verl_root, args.env_config, args.num_envs)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(rendered, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[ERROR] could not render VERL runtime config: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": os.fspath(output),
                "num_envs": rendered["warmup"]["num_envs"],
                "server": rendered["server"],
                "source": rendered["_agentark_verl_bridge"]["source"],
                "sandbox_auto_expands_for_worker_index": rendered[
                    "_agentark_verl_bridge"
                ]["sandbox_auto_expands_for_worker_index"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
