from __future__ import annotations

import os
import importlib
import threading
import uuid
import traceback
from typing import Any, Dict, List, Optional, Tuple

from agent_ark.ark_env.ark_env import ArkEnv
from agent_ark.ark_env.direct_env import EnvInfoManager
from agent_ark.ark_env.op_timeout import OperationTimeout, run_with_timeout
from agent_ark.ark_env.serving.task_selector import get_default_selector, resolve_task_for_group
from agent_ark.ark_env.serving.protocol import EnvStartPayload, EnvStepPayload, as_json_dict, encode_obs
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
    def __init__(self):
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

    def create_env(self, cfg: Dict[str, Any], env_id: Optional[str] = None) -> Dict[str, Any]:
        preflight = self.validate_env_cfg(cfg)
        if not preflight["ok"]:
            raise ValueError(f"Invalid env cfg: {preflight}")

        with self.lock:
            env_id = env_id or str(uuid.uuid4())
            if env_id in self.envs:
                raise ValueError(f"env_id already exists: {env_id}")
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
                self._used_worker_indices.add(worker_index)
            self._worker_index_by_env[env_id] = worker_index
            try:
                runtime = EnvRuntime(env_id=env_id, cfg=runtime_cfg)
            except Exception:
                # Building the runtime failed; reclaim the index so it is not leaked.
                self._release_worker_index(env_id)
                raise
            self.envs[env_id] = runtime
            self.env_order.append(env_id)
            self.in_use[env_id] = False
            return {"env_id": env_id}

    def _discard_env(self, env_id: str) -> None:
        """Remove a broken env from the pool and close it (best-effort).

        Used when an env times out / errors so it is never leased again. close()
        is guarded by its own timeout and force-kills the Unity process if needed,
        so this will not block the caller indefinitely.
        """
        with self.lock:
            runtime = self.envs.pop(env_id, None)
            if env_id in self.env_order:
                self.env_order.remove(env_id)
            self.in_use.pop(env_id, None)
            self._release_worker_index(env_id)
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass

    def _lease_env(self, cfg: Dict[str, Any], env_id: Optional[str]) -> Tuple[str, EnvRuntime]:
        with self.lock:
            target_env_id: Optional[str] = None

            if env_id:
                if env_id in self.envs:
                    if self.in_use.get(env_id, False):
                        raise ValueError(f"env_id already in use: {env_id}")
                    target_env_id = env_id
                else:
                    created = self.create_env(cfg=cfg, env_id=env_id)
                    target_env_id = str(created["env_id"])
            else:
                for candidate in self.env_order:
                    runtime = self.envs.get(candidate)
                    if runtime is not None and not self.in_use.get(candidate, False) and not runtime.broken:
                        target_env_id = candidate
                        break

                if target_env_id is None:
                    created = self.create_env(cfg=cfg)
                    target_env_id = str(created["env_id"])

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
            out = []
            for env_id in self.env_order:
                runtime = self.envs.get(env_id)
                if runtime is None:
                    continue
                out.append(
                    {
                        "env_id": env_id,
                        "started": runtime.started,
                        "in_use": self.in_use.get(env_id, False),
                    }
                )
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

    def close_env(self, env_id: str) -> bool:
        with self.lock:
            runtime = self.envs.pop(env_id, None)
            if runtime is None:
                return False
            if env_id in self.env_order:
                self.env_order.remove(env_id)
            self.in_use.pop(env_id, None)
            self._release_worker_index(env_id)

        runtime.close()
        return True
