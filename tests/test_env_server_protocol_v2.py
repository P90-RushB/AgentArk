from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_ark.ark_env.serving import env_server  # noqa: E402
from agent_ark.ark_env.serving.lease_protocol import (  # noqa: E402
    LeaseConflict,
    LeaseOperationInProgress,
)


IDENTITY = {
    "server_epoch": "server-epoch-1",
    "lease_id": "lease-1",
    "lease_generation": 3,
}


class FakeManager:
    def __init__(self) -> None:
        self.start_reaper_calls = 0
        self.shutdown_calls = 0
        self.calls: list[tuple[Any, ...]] = []

    def start_reaper(self) -> bool:
        self.start_reaper_calls += 1
        return True

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def list_envs(self) -> list[dict[str, Any]]:
        return []

    def protocol_status(self) -> dict[str, Any]:
        return {
            "server_epoch": "server-epoch-1",
            "protocol_versions": ["v1", "v2"],
            "active_v2_leases": 0,
            "starting_v2_leases": 0,
            "lease_ttl_s": 300.0,
            "reaper_running": True,
        }

    def acquire_start_env_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("acquire", kwargs))
        return {
            "env_id": "env-1",
            "unity_id": 0,
            "obs": {},
            **IDENTITY,
        }

    def step_env_v2(self, env_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("step", env_id, kwargs))
        if kwargs["action_id"] == "busy-action":
            raise LeaseOperationInProgress("synthetic step is still running")
        return {
            "env_id": env_id,
            "action_id": kwargs["action_id"],
            "turn_index": kwargs["turn_index"],
            "obs": {},
            "reward": 1.0,
            "done": False,
        }

    def heartbeat_env_v2(self, env_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("heartbeat", env_id, kwargs))
        return {"env_id": env_id, "ok": True}

    def heartbeat_many_v2(self, leases: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append(("heartbeat_many", leases))
        return {
            "server_epoch": "server-epoch-1",
            "items": [{"env_id": item["env_id"], "ok": True} for item in leases],
        }

    def release_env_v2(self, env_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("release", env_id, kwargs))
        return {"env_id": env_id, "ok": True}

    def release_env(self, env_id: str) -> bool:
        self.calls.append(("legacy_release", env_id))
        raise LeaseConflict("legacy release cannot mutate a v2 lease")


class EnvServerProtocolV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.original_manager = env_server.manager
        self.manager = FakeManager()
        env_server.manager = self.manager

    def tearDown(self) -> None:
        env_server.manager = self.original_manager

    def test_capabilities(self) -> None:
        with TestClient(env_server.app) as client:
            response = client.get("/v2/capabilities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.manager.protocol_status())

    def test_v2_acquire_step_and_release_payload_routing(self) -> None:
        acquire_body = {
            "cfg": {"env_path": "/fake/env", "mod_path": "/fake/mod"},
            "acquire_request_id": "acquire-1",
            "client_id": "trainer-rank-0",
            "env_id": None,
            "task_name": "TaskA",
            "group_seed": 7,
            "unity_env_id": 2,
            "history_snapshot": {"0": []},
            "start_attempt_index": 4,
            "uid": "group-1",
        }
        step_body = {
            **IDENTITY,
            "action_id": "action-1",
            "turn_index": 1,
            "action": "MoveForward();",
            "assistant": "```csharp\nMoveForward();\n```",
        }
        release_body = {
            **IDENTITY,
            "release_request_id": "release-1",
        }

        with TestClient(env_server.app) as client:
            acquire_response = client.post("/v2/envs/acquire_start", json=acquire_body)
            step_response = client.post("/v2/envs/env-1/step", json=step_body)
            release_response = client.post("/v2/envs/env-1/release", json=release_body)

        self.assertEqual(acquire_response.status_code, 200, acquire_response.text)
        self.assertEqual(step_response.status_code, 200, step_response.text)
        self.assertEqual(release_response.status_code, 200, release_response.text)

        acquire_call, step_call, release_call = self.manager.calls
        self.assertEqual(acquire_call[0], "acquire")
        self.assertEqual(acquire_call[1], acquire_body)
        self.assertEqual(step_call, ("step", "env-1", step_body))
        self.assertEqual(release_call, ("release", "env-1", release_body))

    def test_retryable_lease_error_has_stable_detail_status_and_header(self) -> None:
        with TestClient(env_server.app) as client:
            response = client.post(
                "/v2/envs/env-1/step",
                json={
                    **IDENTITY,
                    "action_id": "busy-action",
                    "turn_index": 1,
                    "action": "MoveForward();",
                },
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.headers.get("Retry-After"), "1")
        self.assertEqual(
            response.json(),
            {
                "detail": {
                    "code": "operation_in_progress",
                    "message": "synthetic step is still running",
                    "retryable": True,
                }
            },
        )

    def test_batch_heartbeat_schema_and_payload(self) -> None:
        leases = [
            {
                "env_id": "env-1",
                **IDENTITY,
                "heartbeat_id": "heartbeat-1",
            },
            {
                "env_id": "env-2",
                "server_epoch": "server-epoch-1",
                "lease_id": "lease-2",
                "lease_generation": 9,
                "heartbeat_id": "heartbeat-2",
            },
        ]
        with TestClient(env_server.app) as client:
            response = client.post("/v2/leases/heartbeat", json={"leases": leases})
            invalid_response = client.post(
                "/v2/leases/heartbeat",
                json={"leases": [{key: value for key, value in leases[0].items() if key != "heartbeat_id"}]},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["items"], [
            {"env_id": "env-1", "ok": True},
            {"env_id": "env-2", "ok": True},
        ])
        self.assertEqual(self.manager.calls, [("heartbeat_many", leases)])
        self.assertEqual(invalid_response.status_code, 422)

    def test_legacy_lease_conflict_uses_v2_stable_error(self) -> None:
        with TestClient(env_server.app) as client:
            response = client.post("/v1/envs/env-1/release", json={})

        self.assertEqual(response.status_code, 409)
        self.assertIsNone(response.headers.get("Retry-After"))
        self.assertEqual(
            response.json(),
            {
                "detail": {
                    "code": "lease_conflict",
                    "message": "legacy release cannot mutate a v2 lease",
                    "retryable": False,
                }
            },
        )

    def test_lifespan_starts_and_stops_manager_once(self) -> None:
        self.assertEqual(self.manager.start_reaper_calls, 0)
        self.assertEqual(self.manager.shutdown_calls, 0)

        with TestClient(env_server.app):
            self.assertEqual(self.manager.start_reaper_calls, 1)
            self.assertEqual(self.manager.shutdown_calls, 0)

        self.assertEqual(self.manager.start_reaper_calls, 1)
        self.assertEqual(self.manager.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
