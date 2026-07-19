#!/usr/bin/env python3
"""Read-only health and idle-pool preflight for an AgentArk env server."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"AgentArk request failed: GET {url}: {type(exc).__name__}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Check AgentArk server health and idle Unity capacity")
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument(
        "--protocol-version",
        choices=("v1", "v2"),
        default=os.getenv("AGENTARK_PROTOCOL_VERSION", "v2"),
        help="Only count pool runtimes owned by this protocol namespace",
    )
    parser.add_argument("--required-idle", type=_positive_int, default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--allow-unstarted",
        action="store_true",
        help="Count released but not-yet-started envs as available",
    )
    args = parser.parse_args()
    base_url = args.server_url.rstrip("/")

    health = _get_json(f"{base_url}/health", args.timeout)
    if health.get("ok") is not True:
        raise SystemExit(f"AgentArk health check returned an unhealthy payload: {health!r}")
    env_payload = _get_json(f"{base_url}/v1/envs", args.timeout)
    items = env_payload.get("items")
    if not isinstance(items, list):
        raise SystemExit(f"AgentArk /v1/envs response has no items list: {env_payload!r}")

    protocol_items = [
        item
        for item in items
        if isinstance(item, dict)
        and str(item.get("protocol_namespace", "v1")) == args.protocol_version
    ]
    idle_items = [
        item
        for item in protocol_items
        if not bool(item.get("in_use"))
        and (args.allow_unstarted or bool(item.get("started")))
    ]
    result = {
        "server_url": base_url,
        "healthy": True,
        "protocol_version": args.protocol_version,
        "env_count": len(protocol_items),
        "total_env_count": len(items),
        "idle_started_envs": len(idle_items),
        "in_use_envs": sum(bool(item.get("in_use")) for item in protocol_items),
        "required_idle": args.required_idle,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.required_idle is not None and len(idle_items) < args.required_idle:
        raise SystemExit(
            f"AgentArk has {len(idle_items)} idle started envs; this rollout requires "
            f"at least {args.required_idle}"
        )


if __name__ == "__main__":
    main()
