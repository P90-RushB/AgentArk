from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from agent_ark.ark_env.serving.env_client import EnvHttpClient
from agent_ark.ark_env.runtime_config import load_runtime_config


def build_env_cfg_for_index(base_env_cfg: Dict[str, Any], warm_cfg: Dict[str, Any], index: int) -> Dict[str, Any]:
    cfg = dict(base_env_cfg)
    port_cfg = warm_cfg.get("port_assignment", {}) or {}

    if bool(port_cfg.get("assign_worker_index", True)):
        cfg["worker_index"] = int(index)

    if bool(port_cfg.get("assign_env_id", False)):
        cfg["env_id"] = int(index)

    if bool(port_cfg.get("assign_base_port", False)):
        plan = port_cfg.get("base_port_plan", []) or []
        if index < len(plan):
            cfg["base_port"] = int(plan[index])
        else:
            start = int(port_cfg.get("base_port_start", 5005))
            stride = max(1, int(port_cfg.get("base_port_stride", 1)))
            cfg["base_port"] = int(start + index * stride)
        cfg["base_port_stride"] = max(1, int(port_cfg.get("base_port_stride", 1)))

    return cfg


def _to_jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(type(value).__name__)


def _obs_summary(started: Dict[str, Any]) -> Dict[str, Any]:
    obs = started.get("obs", {}) if isinstance(started, dict) else {}
    vis = obs.get("vis", []) if isinstance(obs, dict) else []

    cam_num = len(vis) if isinstance(vis, list) else 0
    frame_counts = []
    if isinstance(vis, list):
        for frames in vis:
            frame_counts.append(len(frames) if isinstance(frames, list) else 0)

    return {
        "env_id": started.get("env_id"),
        "unity_id": started.get("unity_id"),
        "obs_summary": {
            "keys": sorted(list(obs.keys())) if isinstance(obs, dict) else [],
            "camera_count": cam_num,
            "frames_per_camera": frame_counts,
        },
        "info": _to_jsonable(started.get("info", {})),
    }


async def warmup(config: Dict[str, Any], output: str | None = None):
    server = config["server"]
    warm_cfg = config["warmup"]
    env_cfg = config["env_cfg"]

    client = EnvHttpClient(base_url=f"{server['host']}:{server['port']}", timeout=float(server["timeout"]))

    num_envs = int(warm_cfg.get("num_envs", 1))
    step_once = bool(warm_cfg.get("step_once", True))
    close_envs_after_warmup = bool(warm_cfg.get("close_envs_after_warmup", False))

    created_envs: List[Dict[str, Any]] = []
    started_envs: List[Dict[str, Any]] = []

    env_id_prefix = str(warm_cfg.get("env_id_prefix", "warmup-env"))

    for i in range(num_envs):
        env_id = f"{env_id_prefix}-{i:04d}"
        env_cfg_i = build_env_cfg_for_index(env_cfg, warm_cfg, i)
        created = await client.acreate_env(cfg=env_cfg_i, env_id=env_id)
        created_envs.append(created)
        print(
            f"[warmup] created env={env_id}, cfg_override={{'worker_index': {env_cfg_i.get('worker_index')}, "
            f"'env_id': {env_cfg_i.get('env_id')}, 'base_port': {env_cfg_i.get('base_port')}}}"
        )

    for env_meta in created_envs:
        env_id = env_meta["env_id"]
        started = await client.astart_env(env_id)
        started_envs.append(_obs_summary(started))
        print(f"[warmup] started env={env_id}, unity_id={started.get('unity_id')}")

        if step_once:
            stepped = await client.astep_env(env_id, action=None)
            print(
                f"[warmup] step env={env_id} done={stepped.get('done')} reward={stepped.get('reward')}"
            )

        if not close_envs_after_warmup:
            await client.arelease_env(env_id)
            print(f"[warmup] released env={env_id}")

    if close_envs_after_warmup:
        for env_meta in created_envs:
            await client.aclose_env(env_meta["env_id"])
            print(f"[warmup] closed env={env_meta['env_id']}")

    snapshot = {
        "server": _to_jsonable(server),
        "created_envs": _to_jsonable(created_envs),
        "started_envs": _to_jsonable(started_envs),
    }

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        print(f"[warmup] saved snapshot: {out}")

    client.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm up multiple EnvWrapper instances for training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/ark_env/agentark_runtime_config.example.yaml",
        help="Path to warmup config (.yaml/.json)",
    )
    parser.add_argument("--num-envs", type=int, default=-1, help="Override warmup.num_envs when > 0")
    parser.add_argument("--output", type=str, default="", help="Optional output json path")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    cfg = load_runtime_config(args.config)
    if args.num_envs > 0:
        cfg["warmup"]["num_envs"] = int(args.num_envs)
    asyncio.run(warmup(cfg, output=(args.output or None)))
