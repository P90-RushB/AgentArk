#!/usr/bin/env python3
"""Reset multiple real Unity envs with one group UID and verify group parity."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _post(session: requests.Session, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = session.post(url, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"POST {url} failed: {response.status_code}: {response.text}") from exc
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"POST {url} returned a non-object JSON payload")
    return value


def _get(session: requests.Session, url: str, timeout: float) -> dict[str, Any]:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"GET {url} returned a non-object JSON payload")
    return value


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _count_inline_images(value: Any) -> int:
    if isinstance(value, dict):
        own = int(value.get("type") == "image_url" and isinstance(value.get("image_url"), dict))
        return own + sum(_count_inline_images(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_inline_images(item) for item in value)
    return 0


def _find_first(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key in keys:
            if value.get(key) not in (None, ""):
                return value[key]
        for item in value.values():
            found = _find_first(item, keys)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_first(item, keys)
            if found not in (None, ""):
                return found
    return None


async def _acquire(
    base_url: str,
    env_cfg: dict[str, Any],
    group_uid: str,
    task_name: str | None,
    group_seed: int | None,
    timeout: float,
    protocol_version: str,
    trajectory_index: int,
) -> tuple[requests.Session, dict[str, Any]]:
    session = requests.Session()
    payload = {
        "cfg": env_cfg,
        "env_id": None,
        "task_name": task_name,
        "group_seed": group_seed,
        "unity_env_id": None,
        "history_snapshot": None,
        "start_attempt_index": None,
        "uid": group_uid,
    }
    path = "/v1/envs/acquire_start"
    if protocol_version == "v2":
        payload.update(
            {
                "acquire_request_id": f"smoke-{uuid.uuid4().hex}-{trajectory_index}",
                "client_id": "agentark-ms-swift-unity-smoke",
            }
        )
        path = "/v2/envs/acquire_start"
    try:
        result = await asyncio.to_thread(
            _post,
            session,
            f"{base_url}{path}",
            payload,
            timeout,
        )
        return session, result
    except Exception:
        session.close()
        raise


async def _release(
    base_url: str,
    session: requests.Session,
    started: dict[str, Any],
    timeout: float,
    protocol_version: str,
) -> None:
    try:
        env_id = str(started["env_id"])
        path = f"/v1/envs/{env_id}/release"
        payload: dict[str, Any] = {}
        if protocol_version == "v2":
            path = f"/v2/envs/{env_id}/release"
            payload = {
                "server_epoch": started["server_epoch"],
                "lease_id": started["lease_id"],
                "lease_generation": started["lease_generation"],
                "release_request_id": f"smoke-release-{uuid.uuid4().hex}",
            }
        await asyncio.to_thread(_post, session, f"{base_url}{path}", payload, timeout)
    finally:
        session.close()


async def _run(args: argparse.Namespace) -> None:
    # Import only for execution so --help remains usable outside the AgentArk
    # Python 3.10 environment.
    from agent_ark.ark_env.runtime_config import load_runtime_config

    config = load_runtime_config(str(args.runtime_config))
    env_cfg = dict(config["env_cfg"])
    config_server = config.get("server", {})
    base_url = (
        args.server_url
        or f"{str(config_server.get('host', 'http://127.0.0.1')).rstrip('/')}:{config_server.get('port', 18080)}"
    ).rstrip("/")
    group_uid = args.group_uid or f"unity-smoke:{int(time.time())}:{os.getpid()}"

    acquired: list[tuple[requests.Session, dict[str, Any]]] = []
    summary: dict[str, Any] | None = None
    try:
        results = await asyncio.gather(
            *[
                _acquire(
                    base_url,
                    env_cfg,
                    group_uid,
                    args.task_name,
                    args.group_seed,
                    args.timeout,
                    args.protocol_version,
                    index,
                )
                for index in range(args.copies)
            ],
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        acquired = [result for result in results if not isinstance(result, BaseException)]
        if failures:
            details = "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures)
            raise RuntimeError(f"{len(failures)} acquire/reset calls failed: {details}")

        env_ids = [str(result["env_id"]) for _, result in acquired]
        if len(set(env_ids)) != args.copies:
            raise RuntimeError(f"Expected {args.copies} distinct env_id values, got {env_ids}")

        observations = [result.get("obs") for _, result in acquired]
        messages = [obs.get("messages") if isinstance(obs, dict) else None for obs in observations]
        if any(not isinstance(item, list) or not item for item in messages):
            raise RuntimeError("Every reset must return a non-empty obs.messages list")
        message_hashes = [_canonical_hash(item) for item in messages]
        if len(set(message_hashes)) != 1:
            raise RuntimeError(f"Same group UID produced different initial messages: {message_hashes}")

        image_counts = [_count_inline_images(item) for item in messages]
        if not args.allow_no_image and any(count <= 0 for count in image_counts):
            raise RuntimeError(f"Expected inline image_url observations, got counts {image_counts}")

        tasks = [
            _find_first(result.get("info"), {"task_name", "actual_task_name", "folder_name"})
            for _, result in acquired
        ]
        seeds = [
            _find_first(result.get("info"), {"group_seed", "rollout_group_seed", "actual_group_seed"})
            for _, result in acquired
        ]
        present_tasks = [task for task in tasks if task not in (None, "")]
        present_seeds = [seed for seed in seeds if seed not in (None, "")]
        if present_tasks and len(set(map(str, present_tasks))) != 1:
            raise RuntimeError(f"Same group UID resolved to different tasks: {tasks}")
        if present_seeds and len(set(map(str, present_seeds))) != 1:
            raise RuntimeError(f"Same group UID resolved to different seeds: {seeds}")

        if args.step_action is not None:
            def _step_payload(result: dict[str, Any], index: int) -> tuple[str, dict[str, Any]]:
                env_id = str(result["env_id"])
                if args.protocol_version == "v1":
                    return (
                        f"/v1/envs/{env_id}/step",
                        {"action": args.step_action, "assistant": args.step_action},
                    )
                return (
                    f"/v2/envs/{env_id}/step",
                    {
                        "server_epoch": result["server_epoch"],
                        "lease_id": result["lease_id"],
                        "lease_generation": result["lease_generation"],
                        "action_id": f"smoke-action-{result['acquire_request_id']}-{index}",
                        "turn_index": 1,
                        "action": args.step_action,
                        "assistant": args.step_action,
                    },
                )

            step_calls = []
            for index, (session, result) in enumerate(acquired):
                path, payload = _step_payload(result, index)
                step_calls.append(
                    asyncio.to_thread(
                        _post,
                        session,
                        f"{base_url}{path}",
                        payload,
                        args.timeout,
                    )
                )
            step_results = await asyncio.gather(*step_calls)
            if any("reward" not in result or "done" not in result for result in step_results):
                raise RuntimeError("A step response is missing reward or done")

        summary = {
            "ok": True,
            "server_url": base_url,
            "protocol_version": args.protocol_version,
            "group_uid": group_uid,
            "env_ids": env_ids,
            "task_values": tasks,
            "group_seed_values": seeds,
            "initial_messages_sha256": message_hashes[0],
            "inline_image_counts": image_counts,
            "step_executed": args.step_action is not None,
        }
    finally:
        release_results = await asyncio.gather(
            *[
                _release(
                    base_url,
                    session,
                    result,
                    min(args.timeout, 30.0),
                    args.protocol_version,
                )
                for session, result in acquired
                if result.get("env_id")
            ],
            return_exceptions=True,
        )
        release_failures = [result for result in release_results if isinstance(result, BaseException)]
        if release_failures:
            details = "; ".join(f"{type(exc).__name__}: {exc}" for exc in release_failures)
            raise RuntimeError(f"Failed to release one or more smoke leases: {details}")

        if acquired:
            check_session = requests.Session()
            try:
                payload = await asyncio.to_thread(_get, check_session, f"{base_url}/v1/envs", 10.0)
            finally:
                check_session.close()
            states = {
                str(item.get("env_id")): bool(item.get("in_use"))
                for item in payload.get("items", [])
                if isinstance(item, dict)
            }
            leaked = [
                str(result["env_id"])
                for _, result in acquired
                if states.get(str(result["env_id"]), False)
            ]
            if leaked:
                raise RuntimeError(f"Smoke leases remain in_use after release: {leaked}")

    if summary is not None:
        summary["leases_released"] = True
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test same-group resets against real Unity envs")
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--server-url", default=None)
    parser.add_argument(
        "--protocol-version",
        choices=("v1", "v2"),
        default=os.getenv("AGENTARK_PROTOCOL_VERSION", "v2"),
    )
    parser.add_argument("--copies", type=_positive_int, default=2)
    parser.add_argument("--group-uid", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--group-seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--allow-no-image", action="store_true")
    parser.add_argument(
        "--step-action",
        default=None,
        help="Optional raw action/assistant text to send once to every env",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
