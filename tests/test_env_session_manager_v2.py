from __future__ import annotations

import threading
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

from agent_ark.ark_env.serving import session_manager as session_manager_module
from agent_ark.ark_env.serving.lease_protocol import (
    IdempotencyConflict,
    IdempotencyResultGone,
    LeaseConflict,
    LeaseGone,
    LeaseOperationInProgress,
)
from agent_ark.ark_env.serving.session_manager import EnvSessionManager


ENV_CFG = {
    "env_path": "/fake/AgentArk.x86_64",
    "mod_path": "/fake/mod",
    "task_type": "RLTask",
    "acquire_start_max_retries": 0,
}


class FakeClock:
    def __init__(self, initial: float = 100.0):
        self._value = float(initial)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> float:
        with self._lock:
            self._value += float(seconds)
            return self._value


class FakeRuntime:
    instances: list["FakeRuntime"] = []

    def __init__(self, env_id: str, cfg: dict[str, Any]):
        self.env_id = env_id
        self.cfg = deepcopy(cfg)
        self.broken = False
        self.started = False
        self.completed_interactions = 0
        self.max_interactions = int(cfg.get("max_interactions_per_runtime", 0) or 0)
        self.start_calls: list[dict[str, Any]] = []
        self.step_calls: list[dict[str, Any]] = []
        self.close_calls = 0
        self.block_steps = False
        self.step_entered = threading.Event()
        self.allow_step_to_finish = threading.Event()
        type(self).instances.append(self)

    def start_interaction(self, **kwargs: Any) -> dict[str, Any]:
        self.start_calls.append(deepcopy(kwargs))
        self.started = True
        return {
            "env_id": self.env_id,
            "unity_id": 0,
            "obs": {
                "messages": [
                    {"role": "system", "content": "fake system"},
                    {
                        "role": "user",
                        "content": f"task={kwargs.get('task_name')} seed={kwargs.get('group_seed')}",
                    },
                ]
            },
            "info": {
                "task_name": kwargs.get("task_name"),
                "rollout_group_seed": kwargs.get("group_seed"),
            },
        }

    def step(self, action: str | None = None, assistant: str | None = None) -> dict[str, Any]:
        call = {"action": action, "assistant": assistant}
        self.step_calls.append(call)
        ordinal = len(self.step_calls)
        self.step_entered.set()
        if self.block_steps:
            if not self.allow_step_to_finish.wait(timeout=5):
                raise AssertionError("test did not release the blocked fake step")
        return {
            "unity_id": 0,
            "obs": {"messages": [{"role": "user", "content": f"observation-{ordinal}"}]},
            "reward": float(ordinal),
            "done": False,
            "info": {"step_ordinal": ordinal},
        }

    def should_recycle_after_release(self) -> bool:
        return self.max_interactions > 0 and self.completed_interactions >= self.max_interactions

    def close(self) -> None:
        self.close_calls += 1
        self.started = False


def acquire(
    manager: EnvSessionManager,
    request_id: str,
    *,
    uid: str = "shared-group",
    task_name: str = "TaskA",
    group_seed: int = 7,
    env_id: str | None = None,
) -> dict[str, Any]:
    return manager.acquire_start_env_v2(
        deepcopy(ENV_CFG),
        acquire_request_id=request_id,
        client_id="pytest-client",
        env_id=env_id,
        task_name=task_name,
        group_seed=group_seed,
        uid=uid,
    )


def identity_kwargs(started: dict[str, Any]) -> dict[str, Any]:
    return {
        "server_epoch": started["server_epoch"],
        "lease_id": started["lease_id"],
        "lease_generation": started["lease_generation"],
    }


def step(
    manager: EnvSessionManager,
    started: dict[str, Any],
    *,
    action_id: str,
    turn_index: int,
    action: str,
    assistant: str | None = None,
) -> dict[str, Any]:
    return manager.step_env_v2(
        started["env_id"],
        **identity_kwargs(started),
        action_id=action_id,
        turn_index=turn_index,
        action=action,
        assistant=assistant if assistant is not None else action,
    )


def release(
    manager: EnvSessionManager,
    started: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    return manager.release_env_v2(
        started["env_id"],
        **identity_kwargs(started),
        release_request_id=request_id,
    )


def heartbeat(
    manager: EnvSessionManager,
    started: dict[str, Any],
    heartbeat_id: str,
) -> dict[str, Any]:
    return manager.heartbeat_env_v2(
        started["env_id"],
        **identity_kwargs(started),
        heartbeat_id=heartbeat_id,
    )


def without_replay_flag(payload: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(payload)
    result.pop("replayed", None)
    result.pop("already_released", None)
    return result


def assert_raises(expected_exception):
    """Return unittest's exception context for the implementation checks below."""

    return unittest.TestCase().assertRaises(expected_exception)


def _check_acquire_replay_resets_once_and_payload_conflict_is_rejected(manager_factory):
    manager, _clock = manager_factory()

    first = acquire(manager, "acquire-1")
    replayed = acquire(manager, "acquire-1")
    runtime = manager.envs[first["env_id"]]

    assert len(runtime.start_calls) == 1
    assert first["replayed"] is False
    assert replayed["replayed"] is True
    assert without_replay_flag(replayed) == without_replay_flag(first)

    with assert_raises(IdempotencyConflict):
        acquire(manager, "acquire-1", group_seed=8)
    assert len(runtime.start_calls) == 1
    assert manager._v2_leases[first["env_id"]].identity.token == first["lease_id"]


def _check_same_grpo_group_uses_distinct_envs_and_leases(manager_factory):
    manager, _clock = manager_factory()

    first = acquire(manager, "trajectory-a", uid="group-42")
    second = acquire(manager, "trajectory-b", uid="group-42")

    assert first["env_id"] != second["env_id"]
    assert first["lease_id"] != second["lease_id"]
    assert first["info"] == second["info"] == {
        "task_name": "TaskA",
        "rollout_group_seed": 7,
    }
    assert len(manager.envs[first["env_id"]].start_calls) == 1
    assert len(manager.envs[second["env_id"]].start_calls) == 1


def _check_step_replay_executes_once_and_payload_conflict_is_rejected(manager_factory):
    manager, _clock = manager_factory()
    started = acquire(manager, "acquire-step-replay")
    runtime = manager.envs[started["env_id"]]

    first = step(manager, started, action_id="action-1", turn_index=1, action="R1")
    replayed = step(manager, started, action_id="action-1", turn_index=1, action="R1")

    assert len(runtime.step_calls) == 1
    assert first["replayed"] is False
    assert replayed["replayed"] is True
    assert without_replay_flag(replayed) == without_replay_flag(first)

    with assert_raises(IdempotencyConflict):
        step(manager, started, action_id="action-1", turn_index=1, action="R2")
    assert len(runtime.step_calls) == 1


def _check_turn_indices_must_be_strictly_sequential(manager_factory):
    manager, _clock = manager_factory()
    started = acquire(manager, "acquire-turn-order")
    runtime = manager.envs[started["env_id"]]

    with assert_raises(LeaseConflict):
        step(manager, started, action_id="action-2", turn_index=2, action="U1")
    assert runtime.step_calls == []

    step(manager, started, action_id="action-1", turn_index=1, action="R1")
    with assert_raises(LeaseConflict):
        step(manager, started, action_id="action-3", turn_index=3, action="L1")
    assert len(runtime.step_calls) == 1


def _check_evicted_step_result_returns_gone_and_never_reexecutes(manager_factory):
    manager, _clock = manager_factory(step_cache_size=1)
    started = acquire(manager, "acquire-cache-eviction")
    runtime = manager.envs[started["env_id"]]

    step(manager, started, action_id="action-1", turn_index=1, action="R1")
    step(manager, started, action_id="action-2", turn_index=2, action="U1")
    assert len(runtime.step_calls) == 2

    with assert_raises(IdempotencyResultGone):
        step(manager, started, action_id="action-1", turn_index=1, action="R1")
    assert len(runtime.step_calls) == 2


def _check_release_replay_counts_interaction_once(manager_factory):
    manager, _clock = manager_factory()
    started = acquire(manager, "acquire-release")
    runtime = manager.envs[started["env_id"]]

    first = release(manager, started, "release-1")
    replayed = release(manager, started, "release-1")

    assert first["replayed"] is False
    assert first["already_released"] is False
    assert replayed["replayed"] is True
    assert replayed["already_released"] is True
    assert runtime.completed_interactions == 1
    assert manager.in_use[started["env_id"]] is False

    with assert_raises(LeaseGone):
        release(manager, started, "different-release-request")
    assert runtime.completed_interactions == 1


def _check_old_generation_cannot_step_heartbeat_or_release_new_lease(manager_factory):
    manager, _clock = manager_factory()
    old = acquire(manager, "acquire-old")
    runtime = manager.envs[old["env_id"]]
    release(manager, old, "release-old")
    current = acquire(manager, "acquire-current")

    assert current["env_id"] == old["env_id"]
    assert current["lease_generation"] == old["lease_generation"] + 1
    assert current["lease_id"] != old["lease_id"]

    with assert_raises(LeaseGone):
        step(manager, old, action_id="stale-action", turn_index=1, action="R1")
    with assert_raises((LeaseConflict, LeaseGone)):
        heartbeat(manager, old, "stale-heartbeat")
    with assert_raises(LeaseGone):
        release(manager, old, "stale-release")

    assert manager.in_use[current["env_id"]] is True
    assert manager._v2_leases[current["env_id"]].identity.token == current["lease_id"]
    assert runtime.completed_interactions == 1
    assert runtime.step_calls == []

    step(manager, current, action_id="current-action", turn_index=1, action="U1")
    assert len(runtime.step_calls) == 1


def _check_heartbeat_extends_ttl_replay_does_not_extend_again_and_expiry_is_final(manager_factory):
    manager, clock = manager_factory(ttl=10.0)
    started = acquire(manager, "acquire-heartbeat")
    env_id = started["env_id"]
    initial_deadline = manager._v2_leases[env_id].expires_at

    clock.advance(9.0)
    assert manager.reap_expired_leases() == {"expired": [], "expire_pending": []}
    first = heartbeat(manager, started, "heartbeat-1")
    extended_deadline = manager._v2_leases[env_id].expires_at
    assert extended_deadline == initial_deadline + 9.0
    assert first["replayed"] is False

    clock.advance(1.0)
    replayed = heartbeat(manager, started, "heartbeat-1")
    assert replayed["replayed"] is True
    assert manager._v2_leases[env_id].expires_at == extended_deadline

    clock.advance(9.0)
    reaped = manager.reap_expired_leases()
    assert reaped == {"expired": [env_id], "expire_pending": []}
    assert manager.in_use[env_id] is False
    assert manager.envs[env_id].completed_interactions == 1
    with assert_raises(LeaseGone):
        heartbeat(manager, started, "heartbeat-after-expiry")


def _check_ttl_during_step_marks_pending_never_releases_slot_and_discards_after_step(manager_factory):
    manager, clock = manager_factory(ttl=5.0)
    started = acquire(manager, "acquire-blocked-step")
    env_id = started["env_id"]
    runtime = manager.envs[env_id]
    runtime.block_steps = True
    result: dict[str, Any] = {}
    failure: list[BaseException] = []

    def run_step() -> None:
        try:
            result.update(
                step(
                    manager,
                    started,
                    action_id="blocked-action",
                    turn_index=1,
                    action="R1",
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            failure.append(exc)

    thread = threading.Thread(target=run_step, name="protocol-v2-blocked-step")
    thread.start()
    assert runtime.step_entered.wait(timeout=2), "fake runtime step did not start"

    clock.advance(6.0)
    assert manager.reap_expired_leases() == {
        "expired": [],
        "expire_pending": [env_id],
    }
    assert manager.in_use[env_id] is True
    assert manager._v2_leases[env_id].expire_pending is True

    sibling = acquire(manager, "acquire-while-old-step-runs")
    assert sibling["env_id"] != env_id
    assert manager.in_use[env_id] is True

    with assert_raises(LeaseOperationInProgress):
        release(manager, started, "release-during-step")

    runtime.allow_step_to_finish.set()
    thread.join(timeout=2)
    assert not thread.is_alive(), "blocked protocol-v2 step did not finish"
    assert failure == []
    assert result["lease_expired_after_step"] is True
    assert len(runtime.step_calls) == 1
    assert runtime.completed_interactions == 1
    assert runtime.close_calls == 1
    assert env_id not in manager.envs
    assert env_id not in manager.in_use
    assert env_id not in manager._v2_leases


def _check_v1_step_release_and_delete_cannot_bypass_v2_lease(manager_factory):
    manager, _clock = manager_factory()
    started = acquire(manager, "acquire-v1-bypass")
    env_id = started["env_id"]
    runtime = manager.envs[env_id]

    with assert_raises(LeaseConflict):
        manager.step_env(env_id, action="unsafe-v1-action", assistant="unsafe-v1-action")
    with assert_raises(LeaseConflict):
        manager.release_env(env_id)
    with assert_raises(LeaseConflict):
        manager.close_env(env_id)

    assert runtime.step_calls == []
    assert runtime.completed_interactions == 0
    assert runtime.close_calls == 0
    assert manager.in_use[env_id] is True
    assert manager._v2_leases[env_id].identity.token == started["lease_id"]


class EnvSessionManagerV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        FakeRuntime.instances = []
        self._managers: list[EnvSessionManager] = []
        self._runtime_patch = patch.object(session_manager_module, "EnvRuntime", FakeRuntime)
        self._validation_patch = patch.object(
            EnvSessionManager,
            "validate_env_cfg",
            new=staticmethod(lambda _cfg: {"ok": True, "errors": [], "warnings": []}),
        )
        self._runtime_patch.start()
        self._validation_patch.start()

    def tearDown(self) -> None:
        for manager in self._managers:
            manager.shutdown()
        self._validation_patch.stop()
        self._runtime_patch.stop()

    def manager_factory(
        self,
        *,
        ttl: float = 10.0,
        step_cache_size: int = 8,
        clock: FakeClock | None = None,
    ) -> tuple[EnvSessionManager, FakeClock]:
        resolved_clock = clock or FakeClock()
        manager = EnvSessionManager(
            lease_ttl_s=ttl,
            lease_reaper_interval_s=1.0,
            idempotency_retention_s=1000.0,
            step_replay_cache_size=step_cache_size,
            heartbeat_replay_cache_size=8,
            clock=resolved_clock,
        )
        self._managers.append(manager)
        return manager, resolved_clock

    def test_acquire_replay_resets_once_and_payload_conflict_is_rejected(self):
        _check_acquire_replay_resets_once_and_payload_conflict_is_rejected(self.manager_factory)

    def test_same_grpo_group_uses_distinct_envs_and_leases(self):
        _check_same_grpo_group_uses_distinct_envs_and_leases(self.manager_factory)

    def test_step_replay_executes_once_and_payload_conflict_is_rejected(self):
        _check_step_replay_executes_once_and_payload_conflict_is_rejected(self.manager_factory)

    def test_turn_indices_must_be_strictly_sequential(self):
        _check_turn_indices_must_be_strictly_sequential(self.manager_factory)

    def test_evicted_step_result_returns_gone_and_never_reexecutes(self):
        _check_evicted_step_result_returns_gone_and_never_reexecutes(self.manager_factory)

    def test_release_replay_counts_interaction_once(self):
        _check_release_replay_counts_interaction_once(self.manager_factory)

    def test_old_generation_cannot_step_heartbeat_or_release_new_lease(self):
        _check_old_generation_cannot_step_heartbeat_or_release_new_lease(self.manager_factory)

    def test_heartbeat_extends_ttl_replay_does_not_extend_again_and_expiry_is_final(self):
        _check_heartbeat_extends_ttl_replay_does_not_extend_again_and_expiry_is_final(
            self.manager_factory
        )

    def test_ttl_during_step_marks_pending_never_releases_slot_and_discards_after_step(self):
        _check_ttl_during_step_marks_pending_never_releases_slot_and_discards_after_step(
            self.manager_factory
        )

    def test_v1_step_release_and_delete_cannot_bypass_v2_lease(self):
        _check_v1_step_release_and_delete_cannot_bypass_v2_lease(self.manager_factory)

    def test_pool_reuse_matches_protocol_namespace_cfg_and_unpinned_worker(self):
        manager, _clock = self.manager_factory()
        prewarmed = manager.create_env(
            {**deepcopy(ENV_CFG), "worker_index": 7},
            env_id="prewarmed-v2",
            _protocol_namespace="v2",
        )

        started_v2 = acquire(manager, "acquire-prewarmed-v2")
        self.assertEqual(started_v2["env_id"], prewarmed["env_id"])
        release(manager, started_v2, "release-prewarmed-v2")

        started_v1 = manager.acquire_start_env(
            deepcopy(ENV_CFG),
            task_name="TaskA",
            group_seed=7,
        )
        self.assertNotEqual(started_v1["env_id"], started_v2["env_id"])
        self.assertEqual(manager._protocol_by_env[started_v1["env_id"]], "v1")
        self.assertTrue(manager.release_env(started_v1["env_id"]))


if __name__ == "__main__":
    unittest.main()
