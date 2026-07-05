"""Framework-free parallel client demo for the AgentArk env server.

This is a small tutorial-oriented harness. It drives the public env server HTTP
API through ``EnvHttpClient`` and intentionally avoids importing any RL trainer
framework. It is useful in Colab and local smoke tests to demonstrate the server
side of RL training: multiple rollouts can concurrently lease envs, reset them,
take a few steps, and release them back to the pool.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_ark.ark_env.runtime_config import load_runtime_config
from agent_ark.ark_env.serving.env_client import EnvHttpClient


def _server_url(config: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override.rstrip("/")
    server = config.get("server", {}) or {}
    host = str(server.get("host", "http://127.0.0.1")).rstrip("/")
    port = int(server.get("port", 18080))
    return f"{host}:{port}"


def _obs_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    obs = payload.get("obs", {}) if isinstance(payload, dict) else {}
    vis = obs.get("vis", []) if isinstance(obs, dict) else []
    step_msg = obs.get("step_msg", "") if isinstance(obs, dict) else ""
    if isinstance(step_msg, str) and len(step_msg) > 240:
        step_msg = step_msg[:240] + "..."

    return {
        "keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
        "camera_count": len(vis) if isinstance(vis, list) else 0,
        "frames_per_camera": [len(frames) if isinstance(frames, list) else 0 for frames in vis]
        if isinstance(vis, list)
        else [],
        "step_msg": step_msg,
    }


def _safe_name(value: Optional[str]) -> str:
    text = str(value or "server-managed")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:80] or "item"


def _save_obs_images(
    payload: Dict[str, Any],
    *,
    image_dir: Optional[Path],
    rollout_index: int,
    task_name: Optional[str],
    phase: str,
    max_images_per_observation: int,
) -> List[str]:
    if image_dir is None or max_images_per_observation <= 0:
        return []

    obs = payload.get("obs", {}) if isinstance(payload, dict) else {}
    vis = obs.get("vis", []) if isinstance(obs, dict) else []
    if not isinstance(vis, list):
        return []

    image_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    task_part = _safe_name(task_name)
    remaining = int(max_images_per_observation)

    for cam_index, frames in enumerate(vis):
        if remaining <= 0:
            break
        if not isinstance(frames, list) or not frames:
            continue
        selected = frames[-remaining:]
        for frame_offset, frame in enumerate(selected):
            if remaining <= 0:
                break
            if not hasattr(frame, "save"):
                continue
            path = image_dir / (
                f"rollout_{rollout_index:04d}_{task_part}_{phase}_"
                f"cam{cam_index}_img{frame_offset}.png"
            )
            frame.save(path)
            paths.append(str(path))
            remaining -= 1

    return paths


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)


async def _rollout(
    rollout_index: int,
    base_url: str,
    timeout: float,
    env_cfg: Dict[str, Any],
    *,
    task_name: Optional[str],
    group_seed: Optional[int],
    unity_env_id: Optional[int],
    uid_prefix: str,
    max_steps: int,
    action: str,
    semaphore: asyncio.Semaphore,
    image_dir: Optional[Path],
    max_images_per_observation: int,
) -> Dict[str, Any]:
    async with semaphore:
        client = EnvHttpClient(base_url=base_url, timeout=timeout)
        uid = f"{uid_prefix}-{rollout_index:04d}"
        env_id: Optional[str] = None
        record: Dict[str, Any] = {
            "rollout_index": rollout_index,
            "requested_task_name": task_name,
            "uid": uid,
            "steps": [],
            "released": False,
        }

        try:
            started = await client.aacquire_start_env(
                env_cfg,
                task_name=task_name,
                group_seed=group_seed,
                unity_env_id=unity_env_id,
                uid=None if task_name else uid,
            )
            env_id = str(started.get("env_id", "") or "")
            start_image_paths = _save_obs_images(
                started,
                image_dir=image_dir,
                rollout_index=rollout_index,
                task_name=task_name,
                phase="start",
                max_images_per_observation=max_images_per_observation,
            )
            record.update(
                {
                    "env_id": env_id,
                    "unity_id": started.get("unity_id"),
                    "start_info": _jsonable(started.get("info", {})),
                    "start_obs": _obs_summary(started),
                    "start_images": start_image_paths,
                }
            )

            done = False
            total_reward = 0.0
            for step_index in range(max(0, max_steps)):
                if done:
                    break
                stepped = await client.astep_env(env_id, action=action, assistant=action)
                reward = float(stepped.get("reward", 0.0) or 0.0)
                done = bool(stepped.get("done", False))
                total_reward += reward
                step_image_paths = _save_obs_images(
                    stepped,
                    image_dir=image_dir,
                    rollout_index=rollout_index,
                    task_name=task_name,
                    phase=f"step{step_index:02d}",
                    max_images_per_observation=max_images_per_observation,
                )
                record["steps"].append(
                    {
                        "step_index": step_index,
                        "reward": reward,
                        "done": done,
                        "obs": _obs_summary(stepped),
                        "images": step_image_paths,
                        "info": _jsonable(stepped.get("info", {})),
                    }
                )

            record["total_reward"] = total_reward
            record["done"] = done
            record["ok"] = True
        except Exception as exc:
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if env_id:
                try:
                    await client.arelease_env(env_id)
                    record["released"] = True
                except Exception as exc:
                    record["release_error"] = f"{type(exc).__name__}: {exc}"
            client.close()
        return record


async def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_runtime_config(args.config)
    env_cfg = dict(config.get("env_cfg", {}) or {})
    server = config.get("server", {}) or {}
    base_url = _server_url(config, args.base_url)
    timeout = float(args.timeout if args.timeout is not None else server.get("timeout", 1200))

    client = EnvHttpClient(base_url=base_url, timeout=timeout)
    semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))
    image_dir = None if args.no_save_images else Path(args.image_dir)

    try:
        preflight = await client.avalidate_env_cfg(env_cfg)
    except Exception as exc:
        preflight = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"], "warnings": []}

    if args.require_valid_config and not bool(preflight.get("ok", False)):
        raise RuntimeError(f"Env config validation failed: {preflight}")

    task_names = [item.strip() for item in args.task_names.split(",") if item.strip()]
    if not task_names and args.task_name.strip():
        task_names = [args.task_name.strip()]
    action = args.action
    tasks = [
        _rollout(
            i,
            base_url,
            timeout,
            env_cfg,
            task_name=task_names[i % len(task_names)] if task_names else None,
            group_seed=args.group_seed,
            unity_env_id=args.unity_env_id,
            uid_prefix=args.uid_prefix,
            max_steps=args.max_steps,
            action=action,
            semaphore=semaphore,
            image_dir=image_dir,
            max_images_per_observation=args.max_images_per_observation,
        )
        for i in range(max(1, int(args.rollouts)))
    ]
    results = await asyncio.gather(*tasks)

    try:
        final_pool = await client.alist_envs()
    except Exception as exc:
        final_pool = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        client.close()

    return {
        "base_url": base_url,
        "config_path": args.config,
        "preflight": _jsonable(preflight),
        "requested": {
            "rollouts": args.rollouts,
            "concurrency": args.concurrency,
            "max_steps": args.max_steps,
            "task_names": task_names,
            "group_seed": args.group_seed,
            "unity_env_id": args.unity_env_id,
            "server_managed_tasks": not task_names,
            "image_dir": str(image_dir) if image_dir is not None else None,
            "max_images_per_observation": args.max_images_per_observation,
        },
        "ok_count": sum(1 for item in results if item.get("ok")),
        "error_count": sum(1 for item in results if not item.get("ok")),
        "rollouts": _jsonable(results),
        "final_pool": _jsonable(final_pool),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Framework-free parallel AgentArk env server demo")
    parser.add_argument("--config", default="config/ark_env/agentark_runtime_config.example.yaml")
    parser.add_argument("--base-url", default="", help="Override config server URL, e.g. http://127.0.0.1:18080")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--task-name", default="", help="Pin a task. Leave empty to use server-managed uid selection.")
    parser.add_argument(
        "--task-names",
        default="",
        help="Comma-separated pinned tasks. Rollouts cycle through this list. Overrides --task-name.",
    )
    parser.add_argument("--group-seed", type=int, default=None)
    parser.add_argument("--unity-env-id", type=int, default=None)
    parser.add_argument("--uid-prefix", default="env-client-demo")
    parser.add_argument("--action", default="", help="Action text sent on every demo step. Empty string is a no-op.")
    parser.add_argument("--output", default="tmp/env_parallel_client_demo.json")
    parser.add_argument("--image-dir", default="tmp/env_parallel_client_demo_images")
    parser.add_argument("--max-images-per-observation", type=int, default=1)
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return non-zero when one or more rollouts fail. By default the demo writes a report and exits 0.",
    )
    parser.add_argument(
        "--require-valid-config",
        action="store_true",
        help="Fail before rollout if /v1/envs/validate reports an invalid env_cfg.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    report = asyncio.run(run_demo(args))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[env-client-demo] ok={report['ok_count']} error={report['error_count']}")
    print(f"[env-client-demo] server_managed_tasks={report['requested']['server_managed_tasks']}")
    if report["requested"].get("image_dir"):
        print(f"[env-client-demo] images saved under: {report['requested']['image_dir']}")
    print(f"[env-client-demo] report saved: {out}")
    if report["error_count"]:
        for item in report["rollouts"]:
            if not item.get("ok"):
                print(f"[env-client-demo] rollout {item.get('rollout_index')} error: {item.get('error')}")
    return 1 if args.fail_on_error and report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
