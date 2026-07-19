from __future__ import annotations

import os
import importlib
import threading
import time
import uuid
import traceback
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_ark.ark_env.ark_env import ArkEnv
from agent_ark.ark_env.direct_env import EnvInfoManager
from agent_ark.ark_env.op_timeout import OperationTimeout, run_with_timeout
from agent_ark.ark_env.serving.task_selector import get_default_selector, resolve_task_for_group
from agent_ark.ark_env.serving.protocol import EnvStartPayload, EnvStepPayload, as_json_dict, encode_obs
from agent_ark.ark_env.serving.lease_protocol import (
    AcquireReplay,
    CachedOperationFailure,
    HeartbeatReplay,
    IdempotencyConflict,
    IdempotencyResultGone,
    LeaseConflict,
    LeaseGone,
    LeaseIdentity,
    LeaseNotFound,
    LeaseOperationInProgress,
    LeaseRecord,
    LeaseTombstone,
    StepReplay,
    request_fingerprint,
)
from agent_ark.interaction.hooks import ensure_hook_manager
from agent_ark.interaction.serialization import serialize_action_details, serialize_obs_map


def _decode_history_snapshot(history_snapshot: Optional[Dict[int, list]]) -> Dict[int, list]:
    trajectory_io = importlib.import_module('agent_ark.ark_eval.trajectory_io')
    return trajectory_io.decode_history_snapshot(history_snapshot)


class EnvRuntime:
    def __init__(self, env_id: str, cfg: Dict[str, Any]):
        self.env_id = env_id
        self.cfg = dict(cfg)
        self.env = ArkEnv(self.cfg)
        self.lock = threading.RLock()
        self.hooks = ensure_hook_manager(self.cfg.get('hook_manager', self.cfg.get('_hook_manager', None)))
        hook_cfg = self.cfg.get('hooks', {}) if isinstance(self.cfg.get('hooks', {}), dict) else {}
        visualization_cfg = hook_cfg.get('visualization', {}) if isinstance(hook_cfg.get('visualization', {}), dict) else {}
        self._hook_text_max_chars = int(visualization_cfg.get('text_max_chars', 6000) or 6000)
        self._hook_max_images_per_observation = int(visualization_cfg.get('max_images_per_observation', 4) or 4)

        # Hard timeouts (seconds) for the blocking Unity calls, which have no
        # internal timeout. 0/None disables a guard. reset/step include Unity
        # process (re)start, so reset gets a larger budget than step.
        self.reset_timeout_s = float(self.cfg.get("reset_timeout_s", 600.0) or 0.0)
        self.step_timeout_s = float(self.cfg.get("step_timeout_s", 120.0) or 0.0)
        self.close_timeout_s = float(self.cfg.get("close_timeout_s", 60.0) or 0.0)
        # Set when a guarded op times out / the env is deemed unusable. A broken
        # runtime must be discarded (close + recreate) rather than reused.
        self.broken = False

        self.active_unity_id: Optional[int] = None
        self.expected_unity_ids: List[int] = []
        self.started = False
        self.completed_interactions = 0
        self.max_interactions = int(self.cfg.get("max_interactions_per_runtime", 0) or 0)
        self._unity_runtime_identity: Optional[int] = None


    def _emit_env_event(self, event: str, payload: Dict[str, Any], *, phase: Optional[str] = None):
        if not getattr(self.hooks, 'enabled', False):
            return
        self.hooks.emit(event, payload, source='EnvRuntime', phase=phase, env_id=self.env_id)

    @staticmethod
    def _pick_single_unity_agent(obs: Dict[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        if not isinstance(obs, dict) or not obs:
            raise RuntimeError("Env reset returned empty obs")

        # Current training path assumes one agent per env interaction.
        unity_id = sorted(obs.keys())[0]
        return int(unity_id), obs[unity_id]

    def start_interaction(
        self,
        *,
        task_name: Optional[str] = None,
        group_seed: Optional[int] = None,
        unity_env_id: Optional[int] = None,
        history_snapshot: Optional[Dict[int, list]] = None,
        start_attempt_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            try:
                obs, info = run_with_timeout(
                    self.env.reset,
                    task_name=task_name,
                    group_seed=group_seed,
                    env_id=unity_env_id,
                    history_snapshot=_decode_history_snapshot(history_snapshot),
                    start_attempt_index=start_attempt_index,
                    timeout_s=getattr(self, "reset_timeout_s", 600.0),
                    op="reset",
                )
            except OperationTimeout:
                self.broken = True
                raise
            except Exception:
                # A failed reset may leave the underlying Unity runtime in an
                # inconsistent state; treat the runtime as broken so the manager
                # rebuilds it instead of reusing a half-reset env.
                self.broken = True
                raise
            unity_id, single_obs = self._pick_single_unity_agent(obs)
            current_identity = self._get_unity_runtime_identity()
            previous_identity = getattr(self, "_unity_runtime_identity", None)
            if previous_identity is not None and current_identity != previous_identity:
                self.completed_interactions = 0
            self._unity_runtime_identity = current_identity
            self.active_unity_id = unity_id
            self.expected_unity_ids = sorted(int(k) for k in obs.keys())
            self.started = True
            payload = EnvStartPayload(
                env_id=self.env_id,
                unity_id=unity_id,
                obs=encode_obs(single_obs),
                info=info or {},
            )
            self._emit_env_event(
                'env_reset',
                {
                    'env_id': self.env_id,
                    'unity_id': unity_id,
                    'obs': serialize_obs_map(
                        {unity_id: single_obs},
                        text_max_chars=self._hook_text_max_chars,
                        max_images_per_observation=self._hook_max_images_per_observation,
                    ),
                    'info': info or {},
                },
            )
            return as_json_dict(payload)

    def step(self, action: Optional[str] = None, assistant: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            if not self.started:
                raise RuntimeError("start_interaction must be called before step")
            if self.active_unity_id is None:
                raise RuntimeError("No active unity_id in runtime")

            action_text = "" if action is None else str(action)
            assistant_text = None if assistant is None else str(assistant)

            # ArkEnv.step expects actions for all current Unity agents.
            # We train one active agent but still pad other agents with empty action.
            target_unity_ids = list(self.expected_unity_ids)
            if not target_unity_ids:
                target_unity_ids = [int(self.active_unity_id)]
            if int(self.active_unity_id) not in target_unity_ids:
                target_unity_ids.append(int(self.active_unity_id))
            code_act = {int(unity_id): {"action": "", "assistant": None} for unity_id in target_unity_ids}
            code_act[int(self.active_unity_id)] = {
                "action": action_text,
                "assistant": assistant_text if assistant_text is not None else action_text,
            }

            try:
                next_obs, reward, done, info = run_with_timeout(
                    self.env.step, code_act, timeout_s=getattr(self, "step_timeout_s", 120.0), op="step"
                )
            except OperationTimeout as e:
                # Unity is unresponsive; mark broken so the manager rebuilds it.
                self.broken = True
                raise RuntimeError(
                    f"Env step timed out: env_id={self.env_id}, unity_id={self.active_unity_id}, "
                    f"timeout_s={getattr(self, 'step_timeout_s', 120.0)}"
                ) from e
            except Exception as e:
                self.broken = True
                action_preview = action_text[:200] if isinstance(action_text, str) else str(type(action_text).__name__)
                tb = traceback.format_exc(limit=8)
                raise RuntimeError(
                    f"Env step failed: env_id={self.env_id}, unity_id={self.active_unity_id}, "
                    f"target_unity_ids={target_unity_ids}, "
                    f"action_len={len(action_text)}, action_preview={action_preview!r}, "
                    f"error={type(e).__name__}: {e}. traceback={tb}"
                ) from e
            unity_id = self.active_unity_id

            if isinstance(next_obs, dict) and next_obs:
                self.expected_unity_ids = sorted(int(k) for k in next_obs.keys())

            # After one step, the selected unity_id may be absent when the agent is done.
            if isinstance(next_obs, dict) and unity_id in next_obs:
                single_obs = next_obs[unity_id]
            elif isinstance(next_obs, dict) and next_obs:
                # Fallback for compatibility with future env behavior.
                fallback_unity_id = sorted(next_obs.keys())[0]
                self.active_unity_id = int(fallback_unity_id)
                single_obs = next_obs[fallback_unity_id]
            else:
                single_obs = {}

            reward_map = reward if isinstance(reward, dict) else {}
            done_map = done if isinstance(done, dict) else {}
            step_done = bool(done_map.get(unity_id, False) or done_map.get("__all__", False))
            payload = EnvStepPayload(
                unity_id=unity_id,
                obs=encode_obs(single_obs),
                reward=float(reward_map.get(unity_id, 0.0)),
                done=step_done,
                info=info or {},
            )
            self._emit_env_event(
                'env_step',
                {
                    'env_id': self.env_id,
                    'unity_id': unity_id,
                    'actions': serialize_action_details({unity_id: code_act.get(int(unity_id), {})}),
                    'next_obs': serialize_obs_map(
                        {unity_id: single_obs},
                        text_max_chars=self._hook_text_max_chars,
                        max_images_per_observation=self._hook_max_images_per_observation,
                    ),
                    'reward': float(reward_map.get(unity_id, 0.0)),
                    'done': step_done,
                    'info': info or {},
                },
            )
            return as_json_dict(payload)

    def _get_unity_runtime_identity(self) -> Optional[int]:
        sub_env = getattr(getattr(self.env, "sub_env", None), "env", None)
        return id(sub_env) if sub_env is not None else None

    def should_recycle_after_release(self) -> bool:
        return self.max_interactions > 0 and self.completed_interactions >= self.max_interactions

    def close(self):
        with self.lock:
            try:
                run_with_timeout(self.env.close, timeout_s=getattr(self, "close_timeout_s", 60.0), op="close")
            except OperationTimeout:
                # close() itself hung. Don't block forever: best-effort kill the
                # underlying Unity process so resources are reclaimed, then move on.
                self.broken = True
                self._force_kill_unity()
            finally:
                self.started = False
                self.active_unity_id = None

    def _force_kill_unity(self):
        """Best-effort SIGKILL of the underlying Unity subprocess after a hung close.

        Tries a few common handle locations without assuming a specific
        mlagents/runtime version; any failure is swallowed since this is a
        last-resort cleanup path.
        """
        candidates = []
        env = getattr(self, "env", None)
        for attr_chain in (
            ("proc",), ("process",), ("_proc",), ("_process",),
            ("sub_env", "proc"), ("sub_env", "process"),
            ("sub_env", "env", "_process"),
            ("env", "proc"), ("env", "process"), ("env", "_process"),
        ):
            obj = env
            ok = True
            for attr in attr_chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    ok = False
                    break
            if ok and obj is not None:
                candidates.append(obj)
        for proc in candidates:
            try:
                if hasattr(proc, "kill"):
                    proc.kill()
                elif hasattr(proc, "pid"):
                    import os
                    import signal
                    os.kill(int(proc.pid), signal.SIGKILL)
            except Exception:
                pass


class EnvSessionManager:
    """Own AgentArk runtimes and both generations of the HTTP protocol.

    Protocol v1 retains its historical pool semantics. Protocol v2 adds a
    capability lease around a runtime, monotonic generations, bounded
    idempotency records, and TTL reclamation. The two protocols deliberately do
    not share an active runtime: a v1 call cannot bypass a v2 lease token.
    """

    def __init__(
        self,
        *,
        lease_ttl_s: Optional[float] = None,
        lease_reaper_interval_s: Optional[float] = None,
        idempotency_retention_s: Optional[float] = None,
        step_replay_cache_size: Optional[int] = None,
        heartbeat_replay_cache_size: Optional[int] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.lock = threading.RLock()
        self.envs: Dict[str, EnvRuntime] = {}
        self.env_order: List[str] = []
        self.in_use: Dict[str, bool] = {}
        self._task_selector = get_default_selector()
        # Cache of task lists keyed by mod_path. The task store does not change
        # during a training run, so caching avoids re-scanning the filesystem on
        # every rollout. Entries are (mutable) plain lists produced by
        # EnvInfoManager.get_task_list (already stably sorted).
        self._task_list_cache: Dict[str, List[Dict[str, Any]]] = {}
        # Per-env unique worker_index. mlagents' UnityEnvironment binds a socket
        # at base_port + worker_index, so concurrently created envs MUST get
        # distinct indices; otherwise they all collide on worker 0 (base_port
        # 5005), race on the bind, and unlucky envs retry until the Unity startup
        # handshake times out (reset_timeout). Each live env owns one index;
        # indices are returned to the free pool when the env is discarded/closed.
        # worker_index also selects the per-worker runtime sandbox.
        self._worker_index_by_env: Dict[str, int] = {}
        self._used_worker_indices: set[int] = set()
        self._creating_env_ids: set[str] = set()
        self._pool_key_by_env: Dict[str, str] = {}
        self._protocol_by_env: Dict[str, str] = {}

        self.lease_ttl_s = self._resolve_nonnegative_float(
            lease_ttl_s,
            env_name="AGENTARK_LEASE_TTL_S",
            default=300.0,
        )
        self.lease_reaper_interval_s = self._resolve_positive_float(
            lease_reaper_interval_s,
            env_name="AGENTARK_LEASE_REAPER_INTERVAL_S",
            default=5.0,
        )
        self.idempotency_retention_s = self._resolve_positive_float(
            idempotency_retention_s,
            env_name="AGENTARK_IDEMPOTENCY_RETENTION_S",
            default=3600.0,
        )
        self.step_replay_cache_size = self._resolve_positive_int(
            step_replay_cache_size,
            env_name="AGENTARK_STEP_REPLAY_CACHE_SIZE",
            default=8,
        )
        self.heartbeat_replay_cache_size = self._resolve_positive_int(
            heartbeat_replay_cache_size,
            env_name="AGENTARK_HEARTBEAT_REPLAY_CACHE_SIZE",
            default=8,
        )
        self._clock = clock or time.monotonic
        self.server_epoch = uuid.uuid4().hex
        self._lease_generations: Dict[str, int] = {}
        self._v2_leases: Dict[str, LeaseRecord] = {}
        self._v2_starting_envs: Dict[str, str] = {}
        self._v2_acquires: Dict[str, AcquireReplay] = {}
        self._v2_tombstones: Dict[Tuple[str, int, str], LeaseTombstone] = {}
        self._reaper_stop = threading.Event()
        self._reaper_thread: Optional[threading.Thread] = None

    @staticmethod
    def _resolve_nonnegative_float(value: Optional[float], *, env_name: str, default: float) -> float:
        resolved = os.getenv(env_name) if value is None else value
        number = default if resolved in (None, "") else float(resolved)
        if number < 0:
            raise ValueError(f"{env_name} must be non-negative")
        return number

    @staticmethod
    def _resolve_positive_float(value: Optional[float], *, env_name: str, default: float) -> float:
        resolved = os.getenv(env_name) if value is None else value
        number = default if resolved in (None, "") else float(resolved)
        if number <= 0:
            raise ValueError(f"{env_name} must be positive")
        return number

    @staticmethod
    def _resolve_positive_int(value: Optional[int], *, env_name: str, default: int) -> int:
        resolved = os.getenv(env_name) if value is None else value
        number = default if resolved in (None, "") else int(resolved)
        if number <= 0:
            raise ValueError(f"{env_name} must be positive")
        return number

    def _alloc_worker_index(self) -> int:
        """Return the smallest non-negative integer not currently in use.

        Caller must hold ``self.lock``.
        """
        idx = 0
        while idx in self._used_worker_indices:
            idx += 1
        self._used_worker_indices.add(idx)
        return idx

    def _release_worker_index(self, env_id: str) -> None:
        """Return an env's worker_index to the free pool.

        Caller must hold ``self.lock``.
        """
        idx = self._worker_index_by_env.pop(env_id, None)
        if idx is not None:
            self._used_worker_indices.discard(idx)

    def _get_task_list(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        mod_path = str(cfg.get("mod_path", "") or "")
        if not mod_path:
            raise ValueError("cfg.mod_path is required to enumerate tasks for uid-based selection")
        cached = self._task_list_cache.get(mod_path)
        if cached is None:
            cached = EnvInfoManager.get_task_list(mod_path)
            self._task_list_cache[mod_path] = cached
        return cached

    def _resolve_task_for_uid(
        self,
        cfg: Dict[str, Any],
        *,
        uid: Optional[str],
        task_name: Optional[str],
        group_seed: Optional[int],
    ) -> Tuple[Optional[str], Optional[int]]:
        """Translate an RL group id (``uid``) into a concrete (task_name, group_seed).

        ``uid`` is an RL-framework concept (e.g. verl's GRPO group id) and is
        consumed *here*, at the server boundary, so it never leaks into the env
        runtime. Only when the caller did not pin an explicit ``task_name`` and a
        ``uid`` is provided do we deterministically pick a task; all samples that
        share the same ``uid`` resolve to the same task and seed, keeping a whole
        GRPO group on one task without any shared mutable state.
        """
        if task_name is not None or not uid:
            return task_name, group_seed

        task_list = self._get_task_list(cfg)
        sel_task, sel_seed = resolve_task_for_group(uid, task_list, self._task_selector)
        # Respect an explicitly provided group_seed if any; otherwise use the
        # uid-derived seed so the whole group shares identical env randomness.
        return sel_task, (group_seed if group_seed is not None else sel_seed)

    @staticmethod
    def _resolve_effective_env_cfg(
        cfg: Dict[str, Any],
        *,
        task_name: Optional[str],
        group_seed: Optional[int],
        unity_env_id: Optional[int],
    ) -> Dict[str, Any]:
        info_mgr = EnvInfoManager(dict(cfg))
        info_mgr.reset(
            task_name=task_name,
            group_seed=group_seed,
            env_id=unity_env_id,
        )
        return dict(getattr(info_mgr, "env_config", {}) or {})

    @staticmethod
    def validate_env_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        required = ["env_path", "mod_path", "task_type"]
        missing = [k for k in required if k not in cfg or cfg.get(k) in (None, "")]
        if missing:
            return {
                "ok": False,
                "errors": [f"Missing required cfg keys: {missing}"],
                "warnings": [],
            }

        errors: List[str] = []
        warnings: List[str] = []

        env_path = str(cfg.get("env_path"))
        mod_path = str(cfg.get("mod_path"))
        task_type = str(cfg.get("task_type"))
        sandbox_cfg = dict(cfg.get("runtime_sandbox", {}) or {})
        sandbox_enabled = bool(sandbox_cfg.get("enabled", False))

        if not os.path.exists(env_path):
            errors.append(f"env_path not found: {env_path}")
        elif not os.path.isfile(env_path) and not os.path.isdir(env_path):
            errors.append(f"env_path is neither file nor directory: {env_path}")

        if not os.path.exists(mod_path):
            errors.append(f"mod_path not found: {mod_path}")
        elif not os.path.isdir(mod_path):
            errors.append(f"mod_path is not a directory: {mod_path}")

        if sandbox_enabled:
            shared_task_store_path = str(sandbox_cfg.get("shared_task_store_path", "") or "").strip()
            if not shared_task_store_path:
                errors.append(
                    "runtime_sandbox.shared_task_store_path is required when runtime_sandbox.enabled=true"
                )
            elif not os.path.exists(shared_task_store_path):
                errors.append(f"runtime_sandbox.shared_task_store_path not found: {shared_task_store_path}")
            elif not os.path.isdir(shared_task_store_path):
                errors.append(
                    f"runtime_sandbox.shared_task_store_path is not a directory: {shared_task_store_path}"
                )
        elif os.path.isdir(mod_path):
            task_store_path = EnvInfoManager.resolve_task_store_path(mod_path)
            if not os.path.exists(task_store_path):
                errors.append(f"task store not found under mod_path: {task_store_path}")
            elif not os.path.isdir(task_store_path):
                errors.append(f"task store path is not a directory: {task_store_path}")

        if task_type not in ("RLTask", "Create"):
            warnings.append(f"task_type is unusual: {task_type} (expected RLTask/Create)")

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    @staticmethod
    def _pool_key(cfg: Dict[str, Any]) -> str:
        # worker_index is pool topology, not environment semantics. A request
        # that does not pin it may reuse any compatible pre-warmed worker; an
        # explicitly pinned request is checked separately below.
        semantic_cfg = dict(cfg)
        semantic_cfg.pop("worker_index", None)
        return request_fingerprint({"cfg": semantic_cfg})

    @staticmethod
    def _worker_index_matches(runtime: EnvRuntime, cfg: Dict[str, Any]) -> bool:
        requested = cfg.get("worker_index")
        return requested is None or int(runtime.cfg.get("worker_index", -1)) == int(requested)

    def create_env(
        self,
        cfg: Dict[str, Any],
        env_id: Optional[str] = None,
        *,
        _protocol_namespace: str = "v1",
    ) -> Dict[str, Any]:
        preflight = self.validate_env_cfg(cfg)
        if not preflight["ok"]:
            raise ValueError(f"Invalid env cfg: {preflight}")

        protocol_namespace = str(_protocol_namespace)
        if protocol_namespace not in {"v1", "v2"}:
            raise ValueError(f"unknown protocol namespace: {protocol_namespace!r}")
        pool_key = self._pool_key(cfg)
        with self.lock:
            env_id = env_id or str(uuid.uuid4())
            if env_id in self.envs or env_id in self._creating_env_ids:
                raise ValueError(f"env_id already exists: {env_id}")
            self._creating_env_ids.add(env_id)
            runtime_cfg = dict(cfg)
            # Assign a unique worker_index unless the caller pinned one. This
            # isolates each env's mlagents base_port (base_port + worker_index)
            # and selects its per-worker runtime sandbox, preventing the port
            # contention that makes concurrent env startup time out.
            if runtime_cfg.get("worker_index") is None:
                worker_index = self._alloc_worker_index()
                runtime_cfg["worker_index"] = worker_index
            else:
                worker_index = int(runtime_cfg["worker_index"])
                if worker_index in self._used_worker_indices:
                    self._creating_env_ids.discard(env_id)
                    raise ValueError(f"worker_index already in use: {worker_index}")
                self._used_worker_indices.add(worker_index)
            self._worker_index_by_env[env_id] = worker_index

        # ArkEnv construction may touch filesystem/runtime state and must not
        # block every unrelated lease behind the manager lock.
        try:
            runtime = EnvRuntime(env_id=env_id, cfg=runtime_cfg)
        except Exception:
            with self.lock:
                self._creating_env_ids.discard(env_id)
                self._release_worker_index(env_id)
            raise
        with self.lock:
            self._creating_env_ids.discard(env_id)
            if env_id in self.envs:
                self._release_worker_index(env_id)
                collision = True
            else:
                collision = False
                self.envs[env_id] = runtime
                self.env_order.append(env_id)
                self.in_use[env_id] = False
                self._pool_key_by_env[env_id] = pool_key
                self._protocol_by_env[env_id] = protocol_namespace
        if collision:
            self._close_runtime_best_effort(runtime)
            raise ValueError(f"env_id already exists: {env_id}")
        return {"env_id": env_id}

    def _detach_env_locked(self, env_id: str) -> Optional[EnvRuntime]:
        """Atomically remove a runtime from every pool index.

        The caller holds ``self.lock``. Closing is intentionally separate: a
        slow Unity close must never serialize unrelated leases behind the
        manager-wide lock.
        """

        runtime = self.envs.pop(env_id, None)
        if env_id in self.env_order:
            self.env_order.remove(env_id)
        self.in_use.pop(env_id, None)
        self._pool_key_by_env.pop(env_id, None)
        self._protocol_by_env.pop(env_id, None)
        self._release_worker_index(env_id)
        return runtime

    @staticmethod
    def _close_runtime_best_effort(runtime: Optional[EnvRuntime]) -> None:
        if runtime is not None:
            try:
                # ``run_with_timeout`` cannot stop its worker thread. Once an
                # operation times out, kill the Unity subprocess before asking
                # the same ArkEnv object to close so the orphan call cannot
                # continue mutating a live process concurrently with cleanup.
                if getattr(runtime, "broken", False):
                    force_kill = getattr(runtime, "_force_kill_unity", None)
                    if callable(force_kill):
                        force_kill()
                runtime.close()
            except Exception:
                pass

    def _discard_env(self, env_id: str) -> None:
        """Remove a broken env from the pool and close it (best-effort)."""

        with self.lock:
            runtime = self._detach_env_locked(env_id)
        self._close_runtime_best_effort(runtime)

    def _lease_env(
        self,
        cfg: Dict[str, Any],
        env_id: Optional[str],
        *,
        protocol_namespace: str = "v1",
    ) -> Tuple[str, EnvRuntime]:
        pool_key = self._pool_key(cfg)
        requested_env_id = str(env_id) if env_id else None
        with self.lock:
            if requested_env_id and requested_env_id in self.envs:
                if self._protocol_by_env.get(requested_env_id) != protocol_namespace:
                    raise ValueError(
                        f"env_id {requested_env_id!r} belongs to protocol namespace "
                        f"{self._protocol_by_env.get(requested_env_id)!r}, not {protocol_namespace!r}"
                    )
                if self._pool_key_by_env.get(requested_env_id) != pool_key:
                    raise ValueError(f"env_id {requested_env_id!r} was created with a different cfg")
                if self.in_use.get(requested_env_id, False):
                    raise ValueError(f"env_id already in use: {requested_env_id}")
                runtime = self.envs[requested_env_id]
                if not self._worker_index_matches(runtime, cfg):
                    raise ValueError(
                        f"env_id {requested_env_id!r} uses worker_index "
                        f"{runtime.cfg.get('worker_index')}, not {cfg.get('worker_index')}"
                    )
                if runtime.broken:
                    raise ValueError(f"env_id is broken: {requested_env_id}")
                self.in_use[requested_env_id] = True
                return requested_env_id, runtime

            if not requested_env_id:
                for candidate in self.env_order:
                    runtime = self.envs.get(candidate)
                    if (
                        runtime is not None
                        and not self.in_use.get(candidate, False)
                        and not runtime.broken
                        and self._protocol_by_env.get(candidate) == protocol_namespace
                        and self._pool_key_by_env.get(candidate) == pool_key
                        and self._worker_index_matches(runtime, cfg)
                    ):
                        self.in_use[candidate] = True
                        return candidate, runtime

        created = self.create_env(
            cfg=cfg,
            env_id=requested_env_id,
            _protocol_namespace=protocol_namespace,
        )
        target_env_id = str(created["env_id"])
        with self.lock:
            runtime = self.envs.get(target_env_id)
            if runtime is None:
                raise KeyError(f"Unknown env_id: {target_env_id}")
            self.in_use[target_env_id] = True
            return target_env_id, runtime

    def _unlease_after_failed_start(self, env_id: str, runtime: EnvRuntime) -> None:
        if getattr(runtime, "broken", False):
            self._discard_env(env_id)
        else:
            with self.lock:
                self.in_use[env_id] = False

    def acquire_start_env(
        self,
        cfg: Dict[str, Any],
        env_id: Optional[str] = None,
        *,
        task_name: Optional[str] = None,
        group_seed: Optional[int] = None,
        unity_env_id: Optional[int] = None,
        history_snapshot: Optional[Dict[int, list]] = None,
        start_attempt_index: Optional[int] = None,
        uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        task_name, group_seed = self._resolve_task_for_uid(
            cfg, uid=uid, task_name=task_name, group_seed=group_seed
        )
        max_retries = int(cfg.get("acquire_start_max_retries", 2) or 0)

        attempts = 1 if env_id else max(1, max_retries + 1)
        last_error: Optional[Exception] = None

        for _ in range(attempts):
            target_env_id, runtime = self._lease_env(cfg, env_id)
            try:
                return runtime.start_interaction(
                    task_name=task_name,
                    group_seed=group_seed,
                    unity_env_id=unity_env_id,
                    history_snapshot=history_snapshot,
                    start_attempt_index=start_attempt_index,
                )
            except Exception as exc:
                last_error = exc
                self._unlease_after_failed_start(target_env_id, runtime)
                if env_id:
                    break

        if last_error is not None:
            raise last_error
        raise RuntimeError("acquire_start_env failed without an exception")


    def release_env(self, env_id: str) -> bool:
        recycle = False
        with self.lock:
            if env_id in self._v2_leases or env_id in self._v2_starting_envs:
                raise LeaseConflict(
                    f"env_id {env_id!r} has an active protocol-v2 lease; "
                    "use the token-authenticated v2 release endpoint"
                )
            runtime = self.envs.get(env_id)
            if runtime is None:
                return False
            runtime.completed_interactions += 1
            self.in_use[env_id] = False
            recycle = runtime.should_recycle_after_release()

        if recycle:
            self._discard_env(env_id)
        return True

    def list_envs(self) -> List[Dict[str, Any]]:
        with self.lock:
            now = self._clock()
            out = []
            for env_id in self.env_order:
                runtime = self.envs.get(env_id)
                if runtime is None:
                    continue
                item: Dict[str, Any] = {
                    "env_id": env_id,
                    "started": runtime.started,
                    "in_use": self.in_use.get(env_id, False),
                    "protocol_namespace": self._protocol_by_env.get(env_id, "v1"),
                }
                lease = self._v2_leases.get(env_id)
                if lease is not None:
                    item.update(
                        {
                            "lease_protocol": "v2",
                            "lease_generation": lease.identity.generation,
                            "lease_active_operations": lease.active_operations,
                            "lease_expire_pending": lease.expire_pending,
                            "lease_expires_in_s": self._expires_in(lease, now),
                        }
                    )
                out.append(item)
            return out

    def start_env(
        self,
        env_id: str,
        *,
        task_name: Optional[str] = None,
        group_seed: Optional[int] = None,
        unity_env_id: Optional[int] = None,
        history_snapshot: Optional[Dict[int, list]] = None,
        start_attempt_index: Optional[int] = None,
        uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            runtime = self.envs.get(env_id)
            if runtime is None:
                raise KeyError(f"Unknown env_id: {env_id}")
            if env_id in self._v2_leases or env_id in self._v2_starting_envs:
                raise LeaseConflict(
                    f"env_id {env_id!r} has an active protocol-v2 lease; v1 start is forbidden"
                )
            if self.in_use.get(env_id, False):
                raise ValueError(f"env_id already in use: {env_id}")
            self.in_use[env_id] = True

        # Translate the RL group id into a concrete task. Uses the env's own cfg
        # (so the task list comes from the same mod_path the runtime was built with).
        task_name, group_seed = self._resolve_task_for_uid(
            runtime.cfg, uid=uid, task_name=task_name, group_seed=group_seed
        )

        try:
            return runtime.start_interaction(
                task_name=task_name,
                group_seed=group_seed,
                unity_env_id=unity_env_id,
                history_snapshot=history_snapshot,
                start_attempt_index=start_attempt_index,
            )
        except Exception:
            if getattr(runtime, "broken", False):
                self._discard_env(env_id)
            else:
                with self.lock:
                    self.in_use[env_id] = False
            raise

    def step_env(self, env_id: str, action: Optional[str], assistant: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            runtime = self.envs.get(env_id)
            if runtime is None:
                raise KeyError(f"Unknown env_id: {env_id}")
            if env_id in self._v2_leases or env_id in self._v2_starting_envs:
                raise LeaseConflict(
                    f"env_id {env_id!r} has an active protocol-v2 lease; "
                    "use the token-authenticated v2 step endpoint"
                )
        try:
            return runtime.step(action=action, assistant=assistant)
        except Exception:
            # A step timeout/error marks the runtime broken. Drop it from the pool
            # and release the lease so it is never reused; the env will be
            # recreated on demand. The failing rollout is handled (failed) by the
            # caller / agent loop.
            if getattr(runtime, "broken", False):
                self._discard_env(env_id)
            else:
                with self.lock:
                    self.in_use[env_id] = False
            raise

    # ------------------------------------------------------------------
    # Protocol v2: owned leases, idempotent operations, and TTL recovery.

    @staticmethod
    def _validate_operation_id(value: str, *, field: str) -> str:
        resolved = str(value or "").strip()
        if not resolved:
            raise ValueError(f"{field} must not be empty")
        if len(resolved) > 256:
            raise ValueError(f"{field} must be at most 256 characters")
        return resolved

    def _deadline(self, now: float) -> float:
        return float("inf") if self.lease_ttl_s == 0 else now + self.lease_ttl_s

    def _expires_in(self, lease: LeaseRecord, now: float) -> Optional[float]:
        if self.lease_ttl_s == 0:
            return None
        return max(0.0, lease.expires_at - now)

    def _touch_lease_locked(self, lease: LeaseRecord, now: float) -> None:
        lease.touched_at = now
        lease.expires_at = self._deadline(now)

    def _identity_fields(self, identity: LeaseIdentity) -> Dict[str, Any]:
        return {
            "server_epoch": self.server_epoch,
            "env_id": identity.env_id,
            "lease_id": identity.token,
            "lease_generation": identity.generation,
        }

    def _record_tombstone_locked(
        self,
        lease: LeaseRecord,
        *,
        state: str,
        error: Optional[str] = None,
        release_response: Optional[Dict[str, Any]] = None,
        release_request_id: Optional[str] = None,
    ) -> LeaseTombstone:
        now = self._clock()
        tombstone = LeaseTombstone(
            identity=lease.identity,
            state=state,
            updated_at=now,
            acquire_request_id=lease.acquire_request_id,
            step_replays=lease.step_replays,
            release_response=deepcopy(release_response),
            release_request_id=release_request_id,
            error=error,
        )
        self._v2_tombstones[lease.identity.key] = tombstone
        acquire = self._v2_acquires.get(lease.acquire_request_id)
        if acquire is not None:
            acquire.state = state
            acquire.updated_at = now
            acquire.response = None
            acquire.error = error
        return tombstone

    def _terminate_lease_locked(
        self,
        lease: LeaseRecord,
        *,
        state: str,
        discard_runtime: bool,
        error: Optional[str] = None,
        release_response: Optional[Dict[str, Any]] = None,
        release_request_id: Optional[str] = None,
    ) -> Optional[EnvRuntime]:
        current = self._v2_leases.get(lease.identity.env_id)
        if current is not lease:
            return None
        self._v2_leases.pop(lease.identity.env_id, None)
        runtime = self.envs.get(lease.identity.env_id)
        if runtime is not None:
            runtime.completed_interactions += 1
            self.in_use[lease.identity.env_id] = False
            discard_runtime = bool(
                discard_runtime or runtime.broken or runtime.should_recycle_after_release()
            )
        self._record_tombstone_locked(
            lease,
            state=state,
            error=error,
            release_response=release_response,
            release_request_id=release_request_id,
        )
        if discard_runtime:
            return self._detach_env_locked(lease.identity.env_id)
        return None

    def _require_active_lease_locked(
        self,
        *,
        env_id: str,
        lease_id: str,
        lease_generation: int,
        server_epoch: str,
    ) -> LeaseRecord:
        if server_epoch != self.server_epoch:
            raise LeaseGone(
                "server_epoch does not match this env-server process; the previous lease cannot be resumed"
            )
        identity = LeaseIdentity(env_id=env_id, generation=int(lease_generation), token=str(lease_id))
        lease = self._v2_leases.get(env_id)
        if lease is not None:
            if lease.identity != identity:
                raise LeaseConflict(
                    f"lease identity does not own env_id {env_id!r}; it may belong to an older generation"
                )
            if self.lease_ttl_s > 0 and lease.expires_at <= self._clock():
                # Fence the lease immediately at the authentication
                # linearization point. The reaper performs the actual pool
                # transition/close outside this call's critical section.
                lease.expire_pending = True
                if lease.active_operations > 0:
                    raise LeaseOperationInProgress(
                        "lease expired while an operation is still draining"
                    )
                raise LeaseGone("lease heartbeat deadline elapsed")
            return lease
        tombstone = self._v2_tombstones.get(identity.key)
        if tombstone is not None:
            raise LeaseGone(f"lease is no longer active (state={tombstone.state})")
        if env_id not in self.envs:
            raise LeaseNotFound(f"unknown env_id or lease: {env_id!r}")
        raise LeaseGone(f"env_id {env_id!r} has no active protocol-v2 lease")

    def _purge_idempotency_locked(self, now: float) -> None:
        cutoff = now - self.idempotency_retention_s
        for key, tombstone in list(self._v2_tombstones.items()):
            if tombstone.updated_at < cutoff:
                self._v2_tombstones.pop(key, None)
        for request_id, replay in list(self._v2_acquires.items()):
            if replay.state not in {"in_progress", "active"} and replay.updated_at < cutoff:
                self._v2_acquires.pop(request_id, None)

    def reap_expired_leases(self) -> Dict[str, Any]:
        """Reclaim idle expired leases without ever racing an active Unity call."""

        now = self._clock()
        to_close: List[EnvRuntime] = []
        expired: List[str] = []
        pending: List[str] = []
        with self.lock:
            if self.lease_ttl_s > 0:
                for env_id, lease in list(self._v2_leases.items()):
                    if lease.expires_at > now:
                        continue
                    if lease.active_operations > 0:
                        lease.expire_pending = True
                        pending.append(env_id)
                        continue
                    runtime = self._terminate_lease_locked(
                        lease,
                        state="expired",
                        discard_runtime=False,
                        error="lease heartbeat deadline elapsed",
                    )
                    if runtime is not None:
                        to_close.append(runtime)
                    expired.append(env_id)
            self._purge_idempotency_locked(now)
        for runtime in to_close:
            self._close_runtime_best_effort(runtime)
        return {"expired": expired, "expire_pending": pending}

    def start_reaper(self) -> bool:
        """Start one process-local TTL reaper thread; safe to call repeatedly."""

        if self.lease_ttl_s == 0:
            return False
        with self.lock:
            if self._reaper_thread is not None and self._reaper_thread.is_alive():
                return False
            self._reaper_stop.clear()

            def _run() -> None:
                while not self._reaper_stop.wait(self.lease_reaper_interval_s):
                    try:
                        self.reap_expired_leases()
                    except Exception:
                        traceback.print_exc()

            thread = threading.Thread(
                target=_run,
                name=f"agentark-lease-reaper-{self.server_epoch[:8]}",
                daemon=True,
            )
            self._reaper_thread = thread
            thread.start()
            return True

    def stop_reaper(self, timeout_s: float = 5.0) -> bool:
        with self.lock:
            thread = self._reaper_thread
            self._reaper_stop.set()
        if thread is None:
            return False
        if thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout_s))
        with self.lock:
            if self._reaper_thread is thread and not thread.is_alive():
                self._reaper_thread = None
        return not thread.is_alive()

    def shutdown(self) -> None:
        """Stop background work and close all runtimes outside the global lock."""

        self.stop_reaper()
        with self.lock:
            runtimes = [self._detach_env_locked(env_id) for env_id in list(self.env_order)]
            for lease in list(self._v2_leases.values()):
                self._v2_leases.pop(lease.identity.env_id, None)
                self._record_tombstone_locked(lease, state="shutdown", error="env server shutdown")
            self._v2_starting_envs.clear()
        for runtime in runtimes:
            self._close_runtime_best_effort(runtime)

    def protocol_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "server_epoch": self.server_epoch,
                "protocol_versions": ["v1", "v2"],
                "single_process_required": True,
                "idempotency_scope": "server_process_lifetime",
                "active_v2_leases": len(self._v2_leases),
                "starting_v2_leases": len(self._v2_starting_envs),
                "lease_ttl_s": self.lease_ttl_s,
                "reaper_running": bool(
                    self._reaper_thread is not None and self._reaper_thread.is_alive()
                ),
            }

    def acquire_start_env_v2(
        self,
        cfg: Dict[str, Any],
        *,
        acquire_request_id: str,
        client_id: Optional[str] = None,
        env_id: Optional[str] = None,
        task_name: Optional[str] = None,
        group_seed: Optional[int] = None,
        unity_env_id: Optional[int] = None,
        history_snapshot: Optional[Dict[int, list]] = None,
        start_attempt_index: Optional[int] = None,
        uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reset and lease one runtime, safely replayable by request id."""

        request_id = self._validate_operation_id(acquire_request_id, field="acquire_request_id")
        fingerprint = request_fingerprint(
            {
                "client_id": client_id,
                "cfg": cfg,
                "env_id": env_id,
                "task_name": task_name,
                "group_seed": group_seed,
                "unity_env_id": unity_env_id,
                "history_snapshot": history_snapshot,
                "start_attempt_index": start_attempt_index,
                "uid": uid,
            }
        )
        self.reap_expired_leases()
        now = self._clock()
        with self.lock:
            existing = self._v2_acquires.get(request_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise IdempotencyConflict(
                        "acquire_request_id was already used with a different request payload"
                    )
                if existing.state == "in_progress":
                    raise LeaseOperationInProgress("the original acquire request is still in progress")
                if existing.state == "active" and existing.identity is not None and existing.response is not None:
                    current = self._v2_leases.get(existing.identity.env_id)
                    if current is not None and current.identity == existing.identity:
                        if self.lease_ttl_s > 0 and current.expires_at <= self._clock():
                            current.expire_pending = True
                            raise LeaseGone("the acquired lease heartbeat deadline elapsed")
                        result = deepcopy(existing.response)
                        result["replayed"] = True
                        return result
                if existing.state == "failed":
                    raise CachedOperationFailure(existing.error or "the original acquire request failed")
                raise LeaseGone(f"the acquired lease is no longer active (state={existing.state})")
            self._v2_acquires[request_id] = AcquireReplay(
                fingerprint=fingerprint,
                state="in_progress",
                updated_at=now,
            )

        target_env_id: Optional[str] = None
        runtime: Optional[EnvRuntime] = None
        lease_committed = False
        try:
            task_name, group_seed = self._resolve_task_for_uid(
                cfg, uid=uid, task_name=task_name, group_seed=group_seed
            )
            max_retries = int(cfg.get("acquire_start_max_retries", 2) or 0)
            attempts = 1 if env_id else max(1, max_retries + 1)
            last_error: Optional[Exception] = None
            start_payload: Optional[Dict[str, Any]] = None
            for _ in range(attempts):
                target_env_id, runtime = self._lease_env(
                    cfg, env_id, protocol_namespace="v2"
                )
                with self.lock:
                    self._v2_starting_envs[target_env_id] = request_id
                try:
                    start_payload = runtime.start_interaction(
                        task_name=task_name,
                        group_seed=group_seed,
                        unity_env_id=unity_env_id,
                        history_snapshot=history_snapshot,
                        start_attempt_index=start_attempt_index,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    with self.lock:
                        self._v2_starting_envs.pop(target_env_id, None)
                    self._unlease_after_failed_start(target_env_id, runtime)
                    target_env_id = None
                    runtime = None
                    if env_id:
                        break
            if start_payload is None or target_env_id is None or runtime is None:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("acquire_start_env_v2 failed without an exception")

            now = self._clock()
            with self.lock:
                self._v2_starting_envs.pop(target_env_id, None)
                if self.envs.get(target_env_id) is not runtime:
                    raise LeaseGone("runtime was closed while its v2 acquire was starting")
                generation = self._lease_generations.get(target_env_id, 0) + 1
                self._lease_generations[target_env_id] = generation
                identity = LeaseIdentity(
                    env_id=target_env_id,
                    generation=generation,
                    token=uuid.uuid4().hex,
                )
                response = {
                    **deepcopy(start_payload),
                    **self._identity_fields(identity),
                    "acquire_request_id": request_id,
                    "lease_ttl_s": self.lease_ttl_s,
                    "heartbeat_interval_s": None if self.lease_ttl_s == 0 else self.lease_ttl_s / 3.0,
                    "lease_expires_in_s": None if self.lease_ttl_s == 0 else self.lease_ttl_s,
                    "next_turn_index": 1,
                    "replayed": False,
                }
                lease = LeaseRecord(
                    identity=identity,
                    acquire_request_id=request_id,
                    acquire_fingerprint=fingerprint,
                    created_at=now,
                    touched_at=now,
                    expires_at=self._deadline(now),
                    acquire_response=deepcopy(response),
                )
                self._v2_leases[target_env_id] = lease
                replay = self._v2_acquires[request_id]
                replay.state = "active"
                replay.updated_at = now
                replay.identity = identity
                replay.response = deepcopy(response)
                lease_committed = True
                return response
        except BaseException as exc:
            if target_env_id is not None:
                with self.lock:
                    self._v2_starting_envs.pop(target_env_id, None)
                if not lease_committed:
                    # Reset may already have mutated Unity even though the
                    # ownership record could not be committed. Never expose
                    # that ambiguous runtime as idle.
                    self._discard_env(target_env_id)
            with self.lock:
                replay = self._v2_acquires.get(request_id)
                if replay is not None and replay.state == "in_progress":
                    replay.state = "failed"
                    replay.updated_at = self._clock()
                    replay.error = f"{type(exc).__name__}: {exc}"
            raise

    def _cached_step_from_tombstone_locked(
        self,
        identity: LeaseIdentity,
        *,
        action_id: str,
        fingerprint: str,
    ) -> Optional[Dict[str, Any]]:
        tombstone = self._v2_tombstones.get(identity.key)
        if tombstone is None:
            return None
        replay = tombstone.step_replays.get(action_id)
        if replay is None:
            raise LeaseGone(f"lease is no longer active (state={tombstone.state})")
        if replay.fingerprint != fingerprint:
            raise IdempotencyConflict("action_id was already used with a different step payload")
        if replay.state == "succeeded" and replay.response is not None:
            response = deepcopy(replay.response)
            response["replayed"] = True
            return response
        if replay.state == "failed":
            raise CachedOperationFailure(replay.error or "the original step failed")
        raise IdempotencyResultGone("the original step result is no longer retained")

    def _evict_step_responses_locked(self, lease: LeaseRecord) -> None:
        """Bound full response bodies while retaining action-id fingerprints.

        The lightweight ``evicted`` marker is what prevents an old action from
        ever becoming executable again after its response body ages out.
        """

        retained = sum(
            replay.state == "succeeded" and replay.response is not None
            for replay in lease.step_replays.values()
        )
        if retained <= self.step_replay_cache_size:
            return
        for replay in lease.step_replays.values():
            if replay.state == "succeeded" and replay.response is not None:
                replay.response = None
                replay.state = "evicted"
                retained -= 1
                if retained <= self.step_replay_cache_size:
                    return

    def step_env_v2(
        self,
        env_id: str,
        *,
        server_epoch: str,
        lease_id: str,
        lease_generation: int,
        action_id: str,
        turn_index: int,
        action: Optional[str],
        assistant: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute at most one Unity action for an ``action_id``."""

        operation_id = self._validate_operation_id(action_id, field="action_id")
        turn = int(turn_index)
        if turn <= 0:
            raise ValueError("turn_index must be positive")
        identity = LeaseIdentity(env_id=env_id, generation=int(lease_generation), token=str(lease_id))
        fingerprint = request_fingerprint(
            {"turn_index": turn, "action": action, "assistant": assistant}
        )
        self.reap_expired_leases()
        now = self._clock()
        with self.lock:
            if server_epoch != self.server_epoch:
                raise LeaseGone("server_epoch does not match this env-server process")
            tombstone_response = self._cached_step_from_tombstone_locked(
                identity, action_id=operation_id, fingerprint=fingerprint
            )
            if tombstone_response is not None:
                return tombstone_response
            lease = self._require_active_lease_locked(
                env_id=env_id,
                lease_id=lease_id,
                lease_generation=lease_generation,
                server_epoch=server_epoch,
            )
            now = self._clock()
            existing = lease.step_replays.get(operation_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise IdempotencyConflict("action_id was already used with a different step payload")
                if existing.state == "in_progress":
                    raise LeaseOperationInProgress("the original step is still in progress")
                if existing.state == "succeeded" and existing.response is not None:
                    response = deepcopy(existing.response)
                    response["replayed"] = True
                    return response
                if existing.state == "failed":
                    raise CachedOperationFailure(existing.error or "the original step failed")
                raise IdempotencyResultGone("the original step result is no longer retained")
            if lease.expire_pending:
                raise LeaseGone("lease expired while an earlier operation was in progress")
            if lease.active_operations > 0:
                raise LeaseOperationInProgress("another operation is already running for this lease")
            if turn < lease.next_turn_index:
                raise IdempotencyResultGone(
                    f"turn_index {turn} was already consumed; its cached result is no longer retained"
                )
            if turn > lease.next_turn_index:
                raise LeaseConflict(
                    f"out-of-order turn_index {turn}; expected {lease.next_turn_index}"
                )
            runtime = self.envs.get(env_id)
            if runtime is None:
                raise LeaseNotFound(f"runtime for env_id {env_id!r} no longer exists")
            replay = StepReplay(fingerprint=fingerprint, state="in_progress", updated_at=now)
            lease.step_replays[operation_id] = replay
            lease.active_operations += 1
            self._touch_lease_locked(lease, now)

        try:
            result = runtime.step(action=action, assistant=assistant)
        except BaseException as exc:
            to_close: Optional[EnvRuntime] = None
            with self.lock:
                current = self._v2_leases.get(env_id)
                if current is lease:
                    replay.state = "failed"
                    replay.error = f"{type(exc).__name__}: {exc}"
                    replay.updated_at = self._clock()
                    lease.active_operations = max(0, lease.active_operations - 1)
                    to_close = self._terminate_lease_locked(
                        lease,
                        state="failed",
                        discard_runtime=True,
                        error=replay.error,
                    )
            self._close_runtime_best_effort(to_close)
            raise

        to_close = None
        with self.lock:
            current = self._v2_leases.get(env_id)
            if current is not lease:
                raise LeaseGone("lease was invalidated while its step was running")
            now = self._clock()
            response = {
                **deepcopy(result),
                **self._identity_fields(identity),
                "action_id": operation_id,
                "turn_index": turn,
                "next_turn_index": turn + 1,
                "replayed": False,
            }
            replay.state = "succeeded"
            replay.response = deepcopy(response)
            replay.updated_at = now
            lease.active_operations = max(0, lease.active_operations - 1)
            lease.next_turn_index += 1
            self._evict_step_responses_locked(lease)
            expired_during_step = bool(
                lease.expire_pending
                or (self.lease_ttl_s > 0 and lease.expires_at <= now)
            )
            if expired_during_step:
                response["lease_expired_after_step"] = True
                replay.response = deepcopy(response)
                to_close = self._terminate_lease_locked(
                    lease,
                    state="expired",
                    discard_runtime=True,
                    error="lease expired while step was in progress",
                )
            else:
                self._touch_lease_locked(lease, now)
                response["lease_expires_in_s"] = self._expires_in(lease, now)
                replay.response = deepcopy(response)
        self._close_runtime_best_effort(to_close)
        return response

    def release_env_v2(
        self,
        env_id: str,
        *,
        server_epoch: str,
        lease_id: str,
        lease_generation: int,
        release_request_id: str,
    ) -> Dict[str, Any]:
        """Terminate one exact lease; response-loss retries never count twice."""

        request_id = self._validate_operation_id(release_request_id, field="release_request_id")
        identity = LeaseIdentity(env_id=env_id, generation=int(lease_generation), token=str(lease_id))
        self.reap_expired_leases()
        with self.lock:
            if server_epoch != self.server_epoch:
                raise LeaseGone("server_epoch does not match this env-server process")
            tombstone = self._v2_tombstones.get(identity.key)
            if tombstone is not None:
                if tombstone.state == "released":
                    if tombstone.release_request_id != request_id:
                        raise LeaseGone("lease was already released by a different request")
                    result = deepcopy(tombstone.release_response or {"ok": True})
                    result["already_released"] = True
                    result["replayed"] = True
                    return result
                return {
                    **self._identity_fields(identity),
                    "release_request_id": request_id,
                    "ok": True,
                    "already_released": True,
                    "termination_state": tombstone.state,
                    "replayed": True,
                }
            lease = self._require_active_lease_locked(
                env_id=env_id,
                lease_id=lease_id,
                lease_generation=lease_generation,
                server_epoch=server_epoch,
            )
            if lease.active_operations > 0:
                raise LeaseOperationInProgress("cannot release a lease while its step is in progress")
            response = {
                **self._identity_fields(identity),
                "release_request_id": request_id,
                "ok": True,
                "already_released": False,
                "replayed": False,
            }
            to_close = self._terminate_lease_locked(
                lease,
                state="released",
                discard_runtime=False,
                release_response=response,
                release_request_id=request_id,
            )
        self._close_runtime_best_effort(to_close)
        return response

    def heartbeat_env_v2(
        self,
        env_id: str,
        *,
        server_epoch: str,
        lease_id: str,
        lease_generation: int,
        heartbeat_id: str,
    ) -> Dict[str, Any]:
        """Extend one lease; duplicate heartbeat ids return the original deadline."""

        operation_id = self._validate_operation_id(heartbeat_id, field="heartbeat_id")
        fingerprint = request_fingerprint(
            {
                "server_epoch": server_epoch,
                "env_id": env_id,
                "lease_id": lease_id,
                "lease_generation": int(lease_generation),
            }
        )
        self.reap_expired_leases()
        now = self._clock()
        with self.lock:
            lease = self._require_active_lease_locked(
                env_id=env_id,
                lease_id=lease_id,
                lease_generation=lease_generation,
                server_epoch=server_epoch,
            )
            now = self._clock()
            if lease.expire_pending:
                raise LeaseGone("lease is pending expiry and cannot be revived")
            existing = lease.heartbeat_replays.get(operation_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise IdempotencyConflict(
                        "heartbeat_id was already used with a different lease payload"
                    )
                response = deepcopy(existing.response)
                response["replayed"] = True
                return response
            self._touch_lease_locked(lease, now)
            response = {
                **self._identity_fields(lease.identity),
                "heartbeat_id": operation_id,
                "ok": True,
                "lease_ttl_s": self.lease_ttl_s,
                "lease_expires_in_s": self._expires_in(lease, now),
                "replayed": False,
            }
            lease.heartbeat_replays[operation_id] = HeartbeatReplay(
                fingerprint=fingerprint,
                response=deepcopy(response),
                updated_at=now,
            )
            while len(lease.heartbeat_replays) > self.heartbeat_replay_cache_size:
                lease.heartbeat_replays.popitem(last=False)
            return response

    def heartbeat_many_v2(self, leases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Heartbeat many leases with per-item results for a shared client thread."""

        items: List[Dict[str, Any]] = []
        for item in leases:
            try:
                result = self.heartbeat_env_v2(
                    str(item.get("env_id", "")),
                    server_epoch=str(item.get("server_epoch", "")),
                    lease_id=str(item.get("lease_id", "")),
                    lease_generation=int(item.get("lease_generation", 0)),
                    heartbeat_id=str(item.get("heartbeat_id", "")),
                )
                items.append(result)
            except (ValueError, TypeError) as exc:
                items.append(
                    {
                        "env_id": item.get("env_id"),
                        "ok": False,
                        "error": {
                            "code": "invalid_request",
                            "message": str(exc),
                            "retryable": False,
                        },
                    }
                )
            except Exception as exc:
                detail = exc.as_detail() if hasattr(exc, "as_detail") else {
                    "code": "heartbeat_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                    "retryable": False,
                }
                items.append({"env_id": item.get("env_id"), "ok": False, "error": detail})
        return {"server_epoch": self.server_epoch, "items": items}

    def close_env(self, env_id: str) -> bool:
        with self.lock:
            if env_id in self._v2_leases or env_id in self._v2_starting_envs:
                raise LeaseConflict(
                    f"env_id {env_id!r} is owned by protocol v2; "
                    "the legacy DELETE endpoint cannot invalidate its lease"
                )
            runtime = self._detach_env_locked(env_id)
            if runtime is None:
                return False

        runtime.close()
        return True
