import argparse
import hashlib
import os
from copy import deepcopy
from typing import Any, Dict, Optional

from agent_ark.ark_env.ark_sub_env import (
    ArkSubEnv,
    _normalize_optional_path,
    _resolve_cli_action_payloads,
    _resolve_cli_action_step_count,
)
from agent_ark.ark_env.coordination import (
    SharedEpisodeStore,
    compute_history_bucket_key,
    get_history_cfg,
    get_history_retention_cfg,
)
from . import derive_rollout_step_budget, resolve_max_steps_per_attempt
from agent_ark.ark_env.direct_env import EnvInfoManager
from agent_ark.ark_env.context_manager import HistoryContext, MessageContext
from agent_ark.interaction.hooks import ensure_hook_manager
from agent_ark.interaction.serialization import serialize_action_details, serialize_obs_map


class ArkRolloutContext:
    """Rollout-level context manager built on top of stable unity_id keys."""

    def __init__(self):
        self.history_ctx = HistoryContext({})
        self.msg_ctx = MessageContext({})
        self._last_obs_by_uid: Dict[int, dict] = {}
        self._rollout_prompts_by_uid: Dict[int, dict] = {}
        self._pending_rollout_end_by_uid: Dict[int, bool] = {}
        self._msg_state_by_uid: Dict[int, dict] = {}
        self._pending_transitions_by_uid: Dict[int, list] = {}

    def configure(self, env_cfg: Optional[dict]):
        wrapper_cfg = (env_cfg or {}).get('env_wrapper_cfg', {}) or {}
        cfg = wrapper_cfg.get('context_manager', {}) or {}

        history_cfg = (cfg.get('history', {}) if isinstance(cfg, dict) else {}) or {}
        self.history_ctx.configure(history_cfg)

        msg_cfg = (cfg.get('messages', {}) if isinstance(cfg, dict) else {}) or {}
        self.msg_ctx.configure(msg_cfg, history_cfg=history_cfg)

    @staticmethod
    def _snapshot(obs: Optional[dict]):
        if not isinstance(obs, dict):
            return {}
        snap = deepcopy(obs)
        snap.pop('history', None)
        snap.pop('messages', None)
        snap.pop('previous_attempt_terminal_obs', None)
        return snap

    def on_reset(
        self,
        env_cfg: Optional[dict],
        obs: Dict[int, dict],
        history_snapshot: Optional[Dict[int, list]] = None,
    ):
        self.configure(env_cfg)
        self._last_obs_by_uid = {}
        self._rollout_prompts_by_uid = {}
        self._pending_rollout_end_by_uid = {}
        self._msg_state_by_uid = {}
        self._pending_transitions_by_uid = {}

        if self.msg_ctx.enabled:
            for unity_id, obs_dict in (obs or {}).items():
                if not isinstance(obs_dict, dict) or obs_dict.get('skip_infer'):
                    continue
                try:
                    sys_prompt, task_prompt = self.msg_ctx._try_parse_system_task(obs_dict.get('step_msg', ''))
                except Exception:
                    sys_prompt, task_prompt = None, None
                if sys_prompt or task_prompt:
                    self._rollout_prompts_by_uid[int(unity_id)] = {
                        'system_prompt': sys_prompt,
                        'task_prompt': task_prompt,
                    }

        if self.history_ctx.enabled:
            normalized_snapshot = {}
            if isinstance(history_snapshot, dict):
                normalized_snapshot = {
                    int(unity_id): deepcopy(history_snapshot.get(unity_id, []))
                    for unity_id in obs.keys()
                }
            self.history_ctx.start_episode(list(obs.keys()), history_snapshot=normalized_snapshot)
            obs = self._attach_history(obs)

        raw_snaps = {
            int(unity_id): self._snapshot(obs_dict)
            for unity_id, obs_dict in (obs or {}).items()
            if isinstance(obs_dict, dict)
        }

        obs = self._attach_messages(obs, is_reset=True)
        self._last_obs_by_uid = raw_snaps
        return obs

    def _attach_history(self, obs: Dict[int, dict]):
        if not self.history_ctx.enabled:
            return obs
        samples = self.history_ctx.sample_batch(list(obs.keys()))
        for unity_id in obs.keys():
            obs[unity_id]['history'] = samples.get(unity_id, [])
        return obs

    def record_transition(
        self,
        next_obs: Dict[int, dict],
        code_act: Any,
        reward: Dict[int, float],
        done: Dict[int, bool],
        info: Optional[Dict[str, Any]] = None,
    ):
        need_transition = bool(self.msg_ctx.enabled and self.msg_ctx.append_only)
        if not need_transition and not self.history_ctx.enabled:
            return

        for unity_id, obs_dict in next_obs.items():
            if not isinstance(obs_dict, dict):
                continue
            is_done = bool(done.get(unity_id, False)) if isinstance(done, dict) else False
            if obs_dict.get('skip_infer') and not is_done:
                continue

            prev_obs = self._last_obs_by_uid.get(unity_id)
            if prev_obs is None:
                continue

            payload = code_act.get(unity_id) if isinstance(code_act, dict) else None
            if isinstance(payload, dict):
                action_text = payload.get('action', None)
                assistant_text = payload.get('assistant', None)
            else:
                action_text = payload
                assistant_text = None
            self._pending_rollout_end_by_uid[unity_id] = is_done
            func_render_errors = info.get('func_render_errors', {}) if isinstance(info, dict) else {}
            has_func_render_error = False
            if isinstance(func_render_errors, dict):
                has_func_render_error = unity_id in func_render_errors or str(unity_id) in func_render_errors

            transition = {
                'obs': prev_obs,
                'next_obs': self._snapshot(obs_dict),
                'action': action_text,
                'assistant': assistant_text,
                'reward': reward.get(unity_id),
                'done': is_done,
            }
            if has_func_render_error:
                transition['omit_next_obs_images'] = True
                transition['next_obs_image_omitted_reason'] = (
                    'omitted because the previous assistant tool/function-call was invalid and no Unity action was executed; '
                    'the visual state is unchanged from the previous observation.'
                )
            self._pending_transitions_by_uid.setdefault(unity_id, []).append(transition)

            if self.history_ctx.enabled:
                self.history_ctx.record_with_next_obs(
                    agent_id=unity_id,
                    obs=prev_obs,
                    next_obs=self._snapshot(obs_dict),
                    action=action_text,
                    assistant=assistant_text,
                    reward=reward.get(unity_id),
                    done=is_done,
                    omit_next_obs_images=bool(transition.get('omit_next_obs_images', False)),
                    next_obs_image_omitted_reason=transition.get('next_obs_image_omitted_reason', None),
                    finalize=False,
                )

    def finalize_attempt(self, unity_ids):
        if self.history_ctx.enabled:
            for unity_id in unity_ids:
                try:
                    self.history_ctx.finalize_episode(int(unity_id))
                except Exception:
                    pass

    def on_attempt_reset(
        self,
        env_cfg: Optional[dict],
        obs: Dict[int, dict],
        history_snapshot: Optional[Dict[int, list]] = None,
        *,
        current_attempt_index: Optional[int] = None,
        next_attempt_index: Optional[int] = None,
        omit_terminal_obs_images_by_uid: Optional[Dict[int, bool]] = None,
        terminal_obs_image_omitted_reason_by_uid: Optional[Dict[int, str]] = None,
        omit_reset_obs_images: bool = False,
        reset_obs_image_omitted_reason: Optional[str] = None,
    ):
        self.configure(env_cfg)

        if self.msg_ctx.enabled:
            for unity_id, obs_dict in (obs or {}).items():
                if not isinstance(obs_dict, dict) or obs_dict.get('skip_infer'):
                    continue
                try:
                    sys_prompt, task_prompt = self.msg_ctx._try_parse_system_task(obs_dict.get('step_msg', ''))
                except Exception:
                    sys_prompt, task_prompt = None, None
                if sys_prompt or task_prompt:
                    self._rollout_prompts_by_uid[int(unity_id)] = {
                        'system_prompt': sys_prompt,
                        'task_prompt': task_prompt,
                    }

        if self.history_ctx.enabled:
            normalized_snapshot = {}
            if isinstance(history_snapshot, dict):
                normalized_snapshot = {
                    int(unity_id): deepcopy(history_snapshot.get(unity_id, []))
                    for unity_id in obs.keys()
                }
            self.history_ctx.start_episode(list(obs.keys()), history_snapshot=normalized_snapshot)
            obs = self._attach_history(obs)

        raw_snaps = {
            int(unity_id): self._snapshot(obs_dict)
            for unity_id, obs_dict in (obs or {}).items()
            if isinstance(obs_dict, dict)
        }

        if self.msg_ctx.enabled and bool(getattr(self.msg_ctx, 'append_only', False)):
            for unity_id, obs_dict in obs.items():
                if not isinstance(obs_dict, dict) or obs_dict.get('skip_infer'):
                    continue

                pending = self._pending_transitions_by_uid.get(unity_id, []) or []
                transition = pending[-1] if pending else {}
                cached = self._rollout_prompts_by_uid.get(unity_id, {}) if isinstance(self._rollout_prompts_by_uid, dict) else {}
                fallback_system = cached.get('system_prompt') if isinstance(cached, dict) else None
                fallback_task = cached.get('task_prompt') if isinstance(cached, dict) else None
                omit_terminal_images = bool((omit_terminal_obs_images_by_uid or {}).get(unity_id, False))
                terminal_image_reason = None
                if isinstance(terminal_obs_image_omitted_reason_by_uid, dict):
                    terminal_image_reason = terminal_obs_image_omitted_reason_by_uid.get(unity_id, None)
                delta_messages = self.msg_ctx.build_auto_reset_messages(
                    transition,
                    raw_snaps.get(unity_id, {}),
                    current_attempt_index=current_attempt_index,
                    next_attempt_index=next_attempt_index,
                    fallback_system_prompt=fallback_system,
                    fallback_task_prompt=fallback_task,
                    omit_terminal_obs_images=omit_terminal_images,
                    terminal_obs_image_omitted_reason=terminal_image_reason,
                    omit_reset_obs_images=omit_reset_obs_images,
                    reset_obs_image_omitted_reason=reset_obs_image_omitted_reason,
                )

                state = self._msg_state_by_uid.get(unity_id, {})
                full_messages = state.get('messages') if isinstance(state, dict) else None
                if not isinstance(full_messages, list):
                    full_messages = []
                if not full_messages:
                    history_episodes = obs_dict.get('history', []) if isinstance(obs_dict, dict) else []
                    full_messages = self.msg_ctx.build_chat_messages(
                        current_obs=obs_dict,
                        history_episodes=history_episodes if isinstance(history_episodes, list) else [],
                        current_episode_steps=[],
                        fallback_system_prompt=fallback_system,
                        fallback_task_prompt=fallback_task,
                    )
                else:
                    full_messages = deepcopy(full_messages)
                    full_messages.extend(deepcopy(delta_messages))

                self._msg_state_by_uid[unity_id] = {'messages': full_messages}
                messages_out = delta_messages if self.msg_ctx.return_mode == 'delta' else deepcopy(full_messages)
                if self.msg_ctx.only_return_messages:
                    obs[unity_id] = {'messages': messages_out, 'skip_infer': bool(obs_dict.get('skip_infer', False))}
                else:
                    obs_dict['messages'] = messages_out
        else:
            obs = self._attach_messages(obs, is_reset=True)

        for unity_id in obs.keys():
            self._pending_transitions_by_uid[unity_id] = []
        self._last_obs_by_uid = raw_snaps
        self._pending_rollout_end_by_uid = {}
        return obs

    def finalize_obs(self, obs: Dict[int, dict], rollout_done: Optional[Dict[int, bool]] = None):
        raw_snaps = {
            int(unity_id): self._snapshot(obs_dict)
            for unity_id, obs_dict in (obs or {}).items()
            if isinstance(obs_dict, dict)
        }

        if self.history_ctx.enabled:
            obs = self._attach_history(obs)

        obs = self._attach_messages(obs, is_reset=False)

        finalized_unity_ids = []
        for unity_id in obs.keys():
            if rollout_done is not None:
                is_done = bool((rollout_done or {}).get(unity_id, False))
            else:
                is_done = bool(self._pending_rollout_end_by_uid.get(unity_id, False))

            if is_done:
                self._last_obs_by_uid.pop(unity_id, None)
                finalized_unity_ids.append(unity_id)
                continue

            snap = raw_snaps.get(unity_id)
            if isinstance(snap, dict):
                self._last_obs_by_uid[unity_id] = snap

        if self.history_ctx.enabled:
            for unity_id in finalized_unity_ids:
                try:
                    self.history_ctx.finalize_episode(unity_id)
                except Exception:
                    pass

        self._pending_rollout_end_by_uid = {}
        return obs

    def _attach_messages(self, obs: Dict[int, dict], *, is_reset: bool):
        if not self.msg_ctx.enabled:
            return obs

        if bool(getattr(self.msg_ctx, 'append_only', False)):
            for unity_id, obs_dict in obs.items():
                if not isinstance(obs_dict, dict) or obs_dict.get('skip_infer'):
                    continue

                cached = self._rollout_prompts_by_uid.get(unity_id, {}) if isinstance(self._rollout_prompts_by_uid, dict) else {}
                fallback_system = cached.get('system_prompt') if isinstance(cached, dict) else None
                fallback_task = cached.get('task_prompt') if isinstance(cached, dict) else None

                if is_reset or unity_id not in self._msg_state_by_uid:
                    history_episodes = obs_dict.get('history', []) if isinstance(obs_dict, dict) else []
                    base_messages = self.msg_ctx.build_chat_messages(
                        current_obs=obs_dict,
                        history_episodes=history_episodes if isinstance(history_episodes, list) else [],
                        current_episode_steps=[],
                        fallback_system_prompt=fallback_system,
                        fallback_task_prompt=fallback_task,
                    )

                    state = {
                        'messages': deepcopy(base_messages) if isinstance(base_messages, list) else [],
                    }
                    self._msg_state_by_uid[unity_id] = state
                    messages_out = base_messages
                else:
                    state = self._msg_state_by_uid.get(unity_id, {})
                    pending = self._pending_transitions_by_uid.get(unity_id, [])
                    new_steps = pending if isinstance(pending, list) else []

                    delta_messages = []
                    for transition in new_steps:
                        delta_messages.extend(self.msg_ctx.build_step_messages(transition))
                    self._pending_transitions_by_uid[unity_id] = []

                    if self.msg_ctx.return_mode == 'delta':
                        messages_out = delta_messages
                    else:
                        full_messages = state.get('messages')
                        if not isinstance(full_messages, list):
                            full_messages = []
                        if delta_messages:
                            full_messages.extend(deepcopy(delta_messages))
                            state['messages'] = full_messages
                            self._msg_state_by_uid[unity_id] = state
                        messages_out = deepcopy(state.get('messages') or [])

                if self.msg_ctx.only_return_messages:
                    obs[unity_id] = {'messages': messages_out, 'skip_infer': bool(obs_dict.get('skip_infer', False))}
                else:
                    obs_dict['messages'] = messages_out
            return obs

        for unity_id, obs_dict in obs.items():
            if not isinstance(obs_dict, dict) or obs_dict.get('skip_infer'):
                continue

            history_episodes = obs_dict.get('history', []) if isinstance(obs_dict, dict) else []
            current_rollout_steps = self.history_ctx.current_episode(unity_id) if self.history_ctx.enabled else []

            cached = self._rollout_prompts_by_uid.get(unity_id, {}) if isinstance(self._rollout_prompts_by_uid, dict) else {}
            fallback_system = cached.get('system_prompt') if isinstance(cached, dict) else None
            fallback_task = cached.get('task_prompt') if isinstance(cached, dict) else None

            messages = self.msg_ctx.build_chat_messages(
                current_obs=obs_dict,
                history_episodes=history_episodes if isinstance(history_episodes, list) else [],
                current_episode_steps=current_rollout_steps,
                fallback_system_prompt=fallback_system,
                fallback_task_prompt=fallback_task,
            )
            if self.msg_ctx.only_return_messages:
                obs[unity_id] = {'messages': messages, 'skip_infer': obs_dict.get('skip_infer', False)}
            else:
                obs_dict['messages'] = messages
        return obs

    def take_finalized_attempts(self) -> Dict[int, list]:
        if not self.history_ctx:
            return {}
        return self.history_ctx.take_finalized_episodes()


class ArkEnv(object):
    """Rollout-level env that chains multiple ArkSubEnv attempts under one task/group seed.

    The public ArkEnv `group_seed` is rollout-scoped identity. Each ArkSubEnv reset may reuse
    that seed or deterministically derive a per-attempt seed from it, depending on
    reroll_group_seed_on_same_task.
    """

    _MAX_GROUP_SEED = 2**31 - 2

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.sub_env = ArkSubEnv(self.cfg)
        self.rollout_ctx = ArkRolloutContext()
        self.hooks = ensure_hook_manager(self.cfg.get('hook_manager', self.cfg.get('_hook_manager', None)))
        hook_cfg = self.cfg.get('hooks', {}) if isinstance(self.cfg.get('hooks', {}), dict) else {}
        visualization_cfg = hook_cfg.get('visualization', {}) if isinstance(hook_cfg.get('visualization', {}), dict) else {}
        self._hook_text_max_chars = int(visualization_cfg.get('text_max_chars', 6000) or 6000)
        self._hook_max_images_per_observation = int(visualization_cfg.get('max_images_per_observation', 4) or 4)
        self.local_history_store = SharedEpisodeStore()

        self.rollout_started = False
        self.current_attempt_index = 0
        self.max_attempts = 1
        self.max_steps_per_attempt: Optional[int] = None

        self._selected_task_name = None
        self._selected_rollout_group_seed = None
        self._current_attempt_group_seed = None
        self._attempt_group_seed_history: list[int] = []
        self._reroll_group_seed_on_same_task = False
        self._selected_env_id = None
        self._selected_env_cfg: Dict[str, Any] = {}
        self._last_reset_plan: Dict[str, Any] = {}
        self._active_history_bucket_key: Optional[str] = None
        self._rollout_finalized_attempts: Dict[int, list] = {}

    def _emit_env_event(self, event: str, payload: Dict[str, Any], *, phase: Optional[str] = None):
        if not getattr(self.hooks, 'enabled', False):
            return
        self.hooks.emit(event, payload, source='ArkEnv', phase=phase)

    @staticmethod
    def _coerce_int(value, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @classmethod
    def _coerce_group_seed(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            raw = int(value)
        except Exception:
            return None

        if raw == 0:
            return 1
        if raw < 0:
            raw = abs(raw)
        return ((raw - 1) % cls._MAX_GROUP_SEED) + 1

    @classmethod
    def _derive_attempt_group_seed(
        cls,
        rollout_group_seed: Optional[int],
        task_name: Optional[str],
        attempt_index: int,
    ) -> Optional[int]:
        normalized_rollout_seed = cls._coerce_group_seed(rollout_group_seed)
        if normalized_rollout_seed is None:
            return None
        if int(attempt_index) <= 1:
            return normalized_rollout_seed

        material = f"{normalized_rollout_seed}::{str(task_name or '').strip()}::{int(attempt_index)}"
        digest = hashlib.sha256(material.encode('utf-8')).digest()
        derived = int.from_bytes(digest[:8], 'big')
        return (derived % cls._MAX_GROUP_SEED) + 1

    def _attempt_group_seed_for_index(self, attempt_index: int) -> Optional[int]:
        if not bool(self._reroll_group_seed_on_same_task):
            return self._coerce_group_seed(self._selected_rollout_group_seed)
        return self._derive_attempt_group_seed(
            self._selected_rollout_group_seed,
            self._selected_task_name,
            attempt_index,
        )

    def _record_attempt_group_seed(self, attempt_index: int, attempt_group_seed: Optional[int]):
        normalized_attempt_index = max(1, int(attempt_index))
        normalized_seed = self._coerce_group_seed(attempt_group_seed)
        self._current_attempt_group_seed = normalized_seed

        while len(self._attempt_group_seed_history) < normalized_attempt_index:
            self._attempt_group_seed_history.append(0)
        self._attempt_group_seed_history[normalized_attempt_index - 1] = int(normalized_seed or 0)

    def _seed_attempt_group_seed_history(self, through_attempt_index: int):
        target_index = max(1, int(through_attempt_index))
        while len(self._attempt_group_seed_history) < target_index:
            next_index = len(self._attempt_group_seed_history) + 1
            seed = self._attempt_group_seed_for_index(next_index)
            self._attempt_group_seed_history.append(int(self._coerce_group_seed(seed) or 0))

    def _seed_state_payload(self) -> Dict[str, Any]:
        return {
            'rollout_group_seed': self._coerce_group_seed(self._selected_rollout_group_seed),
            'current_attempt_group_seed': self._coerce_group_seed(self._current_attempt_group_seed),
            'attempt_group_seed_history': [
                int(seed) for seed in self._attempt_group_seed_history if int(seed or 0) > 0
            ],
            'reroll_group_seed_on_same_task': bool(self._reroll_group_seed_on_same_task),
        }

    def _build_selected_env_cfg(self, env_cfg: Optional[dict], *, attempt_index: int) -> Dict[str, Any]:
        snapshot = deepcopy(env_cfg) if isinstance(env_cfg, dict) else {}
        snapshot.update(self._seed_state_payload())
        snapshot['attempt_group_seed'] = self._coerce_group_seed(self._current_attempt_group_seed)
        snapshot['current_attempt_index'] = max(1, int(attempt_index))
        return snapshot

    def _resolve_max_attempts(self, explicit_max_attempts: Optional[int] = None) -> int:
        if explicit_max_attempts is not None:
            return max(1, self._coerce_int(explicit_max_attempts, 1))

        env_cfg = self.sub_env.env_info_mgr.env_config or {}
        wrapper_cfg = env_cfg.get('env_wrapper_cfg', {}) if isinstance(env_cfg, dict) else {}
        ark_cfg = wrapper_cfg.get('ark_env', {}) if isinstance(wrapper_cfg, dict) else {}

        for candidate in (
            self.cfg.get('max_attempts', None),
            self.cfg.get('rollout_max_attempts', None),
            env_cfg.get('max_attempts', None),
            env_cfg.get('rollout_max_attempts', None),
            wrapper_cfg.get('max_attempts', None) if isinstance(wrapper_cfg, dict) else None,
            ark_cfg.get('max_attempts', None) if isinstance(ark_cfg, dict) else None,
        ):
            if candidate is not None:
                return max(1, self._coerce_int(candidate, 1))

        return 1

    @staticmethod
    def _resolve_max_steps_per_attempt(env_cfg: Optional[dict]) -> Optional[int]:
        return resolve_max_steps_per_attempt(env_cfg, default=None)

    def get_rollout_step_budget(self) -> Optional[int]:
        return derive_rollout_step_budget(self.max_attempts, self.max_steps_per_attempt)

    @staticmethod
    def _effective_num_parallel_envs(env_cfg: Optional[dict]) -> int:
        if not isinstance(env_cfg, dict):
            return 1
        try:
            return max(1, int(env_cfg.get('num_parallel_envs', 1) or 1))
        except Exception:
            return 1

    def _expected_unity_agent_ids(self, env_cfg: Optional[dict] = None) -> list[int]:
        unity_ids = sorted({int(unity_id) for unity_id in (self.sub_env.ml_unity_id_map or {}).values()})
        if unity_ids:
            return unity_ids
        return list(range(self._effective_num_parallel_envs(env_cfg)))

    def _resolve_effective_env_cfg(
        self,
        *,
        task_name: Optional[str],
        group_seed: Optional[int],
        env_id: Optional[int],
    ) -> Dict[str, Any]:
        info_mgr = EnvInfoManager(dict(self.cfg))
        info_mgr.reset(task_name=task_name, group_seed=group_seed, env_id=env_id)
        return dict(getattr(info_mgr, 'env_config', {}) or {})

    def _prepare_reset_plan(
        self,
        *,
        env_cfg: dict,
        unity_agent_ids: list[int],
        task_name: Optional[str],
        group_seed: Optional[int],
        history_snapshot: Optional[Dict[int, list]],
    ) -> Dict[str, Any]:
        history_cfg = get_history_cfg(env_cfg)
        retention_cfg = get_history_retention_cfg(env_cfg)
        bucket_key = compute_history_bucket_key(
            env_cfg,
            task_name=task_name,
            group_seed=group_seed,
            override_bucket_id=None,
        )
        plan = {
            'history_bucket_key': bucket_key,
            'history_snapshot': self.local_history_store.sample_snapshot(
                bucket_key,
                unity_agent_ids,
                history_cfg,
                retention_cfg=retention_cfg,
            ),
        }
        if isinstance(history_snapshot, dict):
            plan['history_snapshot'] = deepcopy(history_snapshot)

        self._last_reset_plan = deepcopy(plan)
        self._active_history_bucket_key = plan.get('history_bucket_key', None)
        return plan

    @staticmethod
    def _merge_history_snapshot(
        history_snapshot: Optional[Dict[int, list]],
        finalized_attempts: Optional[Dict[int, list]],
    ) -> Dict[int, list]:
        merged = deepcopy(history_snapshot) if isinstance(history_snapshot, dict) else {}
        for unity_id, attempts in (finalized_attempts or {}).items():
            if not isinstance(attempts, list) or not attempts:
                continue
            existing = merged.get(int(unity_id), [])
            if not isinstance(existing, list):
                existing = []
            for attempt in attempts:
                if not isinstance(attempt, list):
                    continue
                if existing and existing[-1] == attempt:
                    continue
                existing.append(deepcopy(attempt))
            merged[int(unity_id)] = existing
        return merged

    def _publish_finalized_attempts(self, finalized_attempts: Optional[Dict[int, list]] = None):
        history_cfg = get_history_cfg(self._selected_env_cfg or self.sub_env.env_info_mgr.env_config or {})
        retention_cfg = get_history_retention_cfg(self._selected_env_cfg or self.sub_env.env_info_mgr.env_config or {})
        if finalized_attempts is None:
            finalized_attempts = self.rollout_ctx.take_finalized_attempts()
        if not finalized_attempts:
            return

        for unity_id, attempts in finalized_attempts.items():
            if not isinstance(attempts, list):
                continue
            export_attempts = self._rollout_finalized_attempts.setdefault(int(unity_id), [])
            for attempt in attempts:
                export_attempts.append(deepcopy(attempt))
                self.local_history_store.publish_episode(
                    self._active_history_bucket_key,
                    int(unity_id),
                    attempt,
                    history_cfg,
                    retention_cfg=retention_cfg,
                )

    def _sample_history_snapshot(self, agent_ids: list[int]) -> Dict[int, list]:
        history_cfg = get_history_cfg(self._selected_env_cfg or self.sub_env.env_info_mgr.env_config or {})
        retention_cfg = get_history_retention_cfg(self._selected_env_cfg or self.sub_env.env_info_mgr.env_config or {})
        return self.local_history_store.sample_snapshot(
            self._active_history_bucket_key,
            [int(agent_id) for agent_id in agent_ids],
            history_cfg,
            retention_cfg=retention_cfg,
        )

    def export_finalized_attempts(self, *, prefix_attempts: Optional[int] = None) -> Dict[int, list]:
        snapshot = deepcopy(self._rollout_finalized_attempts)
        if prefix_attempts is None:
            return snapshot
        limit = max(0, int(prefix_attempts))
        return {int(unity_id): attempts[:limit] for unity_id, attempts in snapshot.items()}

    def _current_unity_to_ml_id_map(self) -> Dict[int, int]:
        return {int(unity_id): int(ml_id) for ml_id, unity_id in (self.sub_env.ml_unity_id_map or {}).items()}

    def _obs_to_unity_id(self, obs: Dict[int, dict]) -> Dict[int, dict]:
        unity_obs = {}
        for ml_id, obs_dict in (obs or {}).items():
            unity_id = self.sub_env.ml_unity_id_map.get(ml_id, ml_id)
            unity_obs[int(unity_id)] = deepcopy(obs_dict)
        return unity_obs

    @staticmethod
    def _mapping_to_unity_id(mapping: Dict[Any, Any], ml_to_unity_id_map: Dict[int, int]) -> Dict[Any, Any]:
        out = {}
        for key, value in (mapping or {}).items():
            if key == '__all__':
                out[key] = value
                continue
            unity_id = ml_to_unity_id_map.get(int(key), int(key))
            out[int(unity_id)] = value
        return out

    @staticmethod
    def _rollout_done_dict(agent_ids, value: bool) -> Dict[Any, bool]:
        done = {int(agent_id): bool(value) for agent_id in agent_ids}
        done['__all__'] = bool(value)
        return done

    @staticmethod
    def _rollout_truncated_dict(agent_ids, value: bool) -> Dict[Any, bool]:
        truncated = {int(agent_id): bool(value) for agent_id in agent_ids}
        truncated['__all__'] = bool(value)
        return truncated

    def _validate_action_keys(self, code_act: Dict[int, Any]):
        current_unity_to_ml = self._current_unity_to_ml_id_map()
        unknown = sorted(int(unity_id) for unity_id in code_act.keys() if int(unity_id) not in current_unity_to_ml)
        if unknown:
            raise KeyError(f'Unknown unity agent ids for current attempt: {unknown}')

    @staticmethod
    def _normalize_action_payload(value: Any) -> Dict[str, Optional[str]]:
        if isinstance(value, dict):
            action_value = value.get('action', None)
            assistant_value = value.get('assistant', None)
        else:
            action_value = value
            assistant_value = None

        action_text = None if action_value is None else str(action_value)
        assistant_text = None if assistant_value is None else str(assistant_value)
        return {
            'action': action_text,
            'assistant': assistant_text,
        }

    def _build_subenv_action(self, code_act: Dict[int, Any]) -> tuple[Dict[int, Dict[str, Optional[str]]], Dict[int, str]]:
        self._validate_action_keys(code_act)
        current_unity_to_ml = self._current_unity_to_ml_id_map()
        unity_actions = {}
        ml_actions = {}

        for unity_id, ml_id in current_unity_to_ml.items():
            payload = self._normalize_action_payload(code_act.get(unity_id, None))
            unity_actions[int(unity_id)] = payload
            ml_actions[int(ml_id)] = payload.get('action', '') or ''

        return unity_actions, ml_actions

    @staticmethod
    def _attempt_success(done_by_unity_id: Dict[Any, Any], reward_by_unity_id: Dict[Any, Any]) -> bool:
        if not bool(done_by_unity_id.get('__all__', False)):
            return False

        agent_ids = [agent_id for agent_id in done_by_unity_id.keys() if agent_id != '__all__']
        if not agent_ids:
            return False

        for unity_id in agent_ids:
            if not bool(done_by_unity_id.get(unity_id, False)):
                return False
            if float(reward_by_unity_id.get(unity_id, 0.0)) < 1.0:
                return False
        return True

    def _reset_sub_env(self, *, attempt_index: Optional[int] = None):
        target_attempt_index = max(1, int(attempt_index if attempt_index is not None else self.current_attempt_index))
        target_group_seed = self._attempt_group_seed_for_index(target_attempt_index)

        sub_obs, sub_info = self.sub_env.reset(
            task_name=self._selected_task_name,
            group_seed=target_group_seed,
            env_id=self._selected_env_id,
        )

        env_cfg = self.sub_env.env_info_mgr.env_config or {}
        actual_task_name = env_cfg.get('task_name', self._selected_task_name)
        if self._selected_task_name is not None and actual_task_name != self._selected_task_name:
            raise RuntimeError(
                'ArkEnv does not support switching task_name within one rollout. '
                f'Expected {self._selected_task_name!r}, got {actual_task_name!r} '
                f'at attempt_index={target_attempt_index}.'
            )
        actual_attempt_group_seed = env_cfg.get('group_seed', target_group_seed)
        self._record_attempt_group_seed(target_attempt_index, actual_attempt_group_seed)
        self._selected_env_cfg = self._build_selected_env_cfg(env_cfg, attempt_index=target_attempt_index)
        return sub_obs, sub_info

    def _build_reset_info(self, sub_info: Dict[str, Any], history_snapshot: Optional[Dict[int, list]] = None):
        rollout_step_budget = self.get_rollout_step_budget()
        return {
            'truncated': self._rollout_truncated_dict(self._current_unity_to_ml_id_map().keys(), False),
            'attempt': {
                'index': self.current_attempt_index,
                'group_seed': self._coerce_group_seed(self._current_attempt_group_seed),
                'attempt_group_seed': self._coerce_group_seed(self._current_attempt_group_seed),
                'max_attempts': self.max_attempts,
                'max_steps_per_attempt': self.max_steps_per_attempt,
                'auto_reset': False,
                'done': False,
                'success': False,
            },
            'rollout': {
                'started': True,
                'success': False,
                'truncated': False,
                'task_name': self._selected_task_name,
                'group_seed': self._coerce_group_seed(self._selected_rollout_group_seed),
                'rollout_group_seed': self._coerce_group_seed(self._selected_rollout_group_seed),
                'current_attempt_group_seed': self._coerce_group_seed(self._current_attempt_group_seed),
                'attempt_group_seed_history': deepcopy(self._attempt_group_seed_history),
                'reroll_group_seed_on_same_task': bool(self._reroll_group_seed_on_same_task),
                'env_id': self._selected_env_id,
                'current_attempt_index': self.current_attempt_index,
                'max_attempts': self.max_attempts,
                'max_steps_per_attempt': self.max_steps_per_attempt,
                'max_rollout_steps': rollout_step_budget,
                'history_snapshot_seeded': bool(history_snapshot),
            },
            'sub_env': {
                'reset': deepcopy(sub_info) if isinstance(sub_info, dict) else {},
            },
        }

    def reset(
        self,
        task_name=None,
        group_seed=None,
        env_id=None,
        *,
        history_snapshot: Optional[Dict[int, list]] = None,
        max_attempts: Optional[int] = None,
        start_attempt_index: Optional[int] = None,
    ):
        sub_obs, sub_info = self.sub_env.reset(
            task_name=task_name,
            group_seed=group_seed,
            env_id=env_id,
        )

        env_cfg = self.sub_env.env_info_mgr.env_config or {}
        self._selected_task_name = env_cfg.get('task_name', task_name)
        initial_attempt_group_seed = env_cfg.get('group_seed', group_seed)
        self._selected_rollout_group_seed = self._coerce_group_seed(initial_attempt_group_seed)
        self._selected_env_id = self._coerce_int(env_cfg.get('env_id', env_id if env_id is not None else 0), 0)
        self._reroll_group_seed_on_same_task = bool(env_cfg.get('reroll_group_seed_on_same_task', False))

        target_attempt_index = max(1, self._coerce_int(start_attempt_index, 1))
        self._attempt_group_seed_history = []
        self.max_attempts = self._resolve_max_attempts(max_attempts)
        if target_attempt_index > self.max_attempts:
            raise ValueError(
                f'start_attempt_index={target_attempt_index} exceeds max_attempts={self.max_attempts}'
            )
        self.max_steps_per_attempt = self._resolve_max_steps_per_attempt(env_cfg)
        self.rollout_started = True
        self._rollout_finalized_attempts = {}

        if target_attempt_index > 1:
            self._seed_attempt_group_seed_history(target_attempt_index - 1)
            sub_obs, sub_info = self._reset_sub_env(attempt_index=target_attempt_index)
            env_cfg = self.sub_env.env_info_mgr.env_config or env_cfg
            self.current_attempt_index = target_attempt_index
            self._seed_attempt_group_seed_history(target_attempt_index)
            self._selected_env_cfg = self._build_selected_env_cfg(env_cfg, attempt_index=self.current_attempt_index)
        else:
            self.current_attempt_index = 1
            self._record_attempt_group_seed(self.current_attempt_index, initial_attempt_group_seed)
            self._selected_env_cfg = self._build_selected_env_cfg(env_cfg, attempt_index=self.current_attempt_index)

        unity_obs = self._obs_to_unity_id(sub_obs)
        reset_plan = self._prepare_reset_plan(
            env_cfg=env_cfg,
            unity_agent_ids=sorted(unity_obs.keys()),
            task_name=self._selected_task_name,
            group_seed=self._selected_rollout_group_seed,
            history_snapshot=history_snapshot,
        )
        rollout_obs = self.rollout_ctx.on_reset(
            env_cfg=self._selected_env_cfg or env_cfg,
            obs=unity_obs,
            history_snapshot=reset_plan.get('history_snapshot', {}),
        )
        info = self._build_reset_info(sub_info, history_snapshot=reset_plan.get('history_snapshot', {}))
        info['history'] = {
            'bucket_key': reset_plan.get('history_bucket_key', None),
            'history_bucket_key': reset_plan.get('history_bucket_key', None),
        }
        self._emit_env_event(
            'env_reset',
            {
                'obs': serialize_obs_map(
                    rollout_obs,
                    text_max_chars=self._hook_text_max_chars,
                    max_images_per_observation=self._hook_max_images_per_observation,
                ),
                'info': info,
                'task_name': self._selected_task_name,
                'rollout_group_seed': self._coerce_group_seed(self._selected_rollout_group_seed),
                'env_id': self._selected_env_id,
                'attempt_index': self.current_attempt_index,
            },
        )
        return rollout_obs, info

    def step(self, code_act: Dict[int, Any], info: Optional[Dict[str, Any]] = None):
        if not self.rollout_started:
            raise RuntimeError('ArkEnv.reset must be called before step')

        unity_actions, subenv_actions = self._build_subenv_action(code_act or {})
        sub_next_obs, sub_reward, sub_done, sub_info = self.sub_env.step(subenv_actions, info=info)

        ml_to_unity_id_map = {int(ml_id): int(unity_id) for ml_id, unity_id in (self.sub_env.ml_unity_id_map or {}).items()}
        next_obs_by_unity_id = self._obs_to_unity_id(sub_next_obs)
        reward_by_unity_id = self._mapping_to_unity_id(sub_reward, ml_to_unity_id_map)
        attempt_done_by_unity_id = self._mapping_to_unity_id(sub_done, ml_to_unity_id_map)
        sub_truncated_by_unity_id = self._mapping_to_unity_id((sub_info or {}).get('truncated', {}), ml_to_unity_id_map)
        func_render_errors_by_unity_id = self._mapping_to_unity_id(
            (sub_info or {}).get('func_render_errors', {}),
            ml_to_unity_id_map,
        )
        transition_info = {}
        if func_render_errors_by_unity_id:
            transition_info['func_render_errors'] = func_render_errors_by_unity_id

        self.rollout_ctx.record_transition(
            next_obs=next_obs_by_unity_id,
            code_act=unity_actions,
            reward=reward_by_unity_id,
            done=attempt_done_by_unity_id,
            info=transition_info,
        )

        attempt_done = bool(attempt_done_by_unity_id.get('__all__', False))
        attempt_success = self._attempt_success(attempt_done_by_unity_id, reward_by_unity_id)
        final_attempt = attempt_done and self.current_attempt_index >= self.max_attempts
        rollout_success = final_attempt and attempt_success
        rollout_truncated = final_attempt and not attempt_success
        auto_reset = attempt_done and not final_attempt

        attempt_agent_ids = sorted(unity_id for unity_id in next_obs_by_unity_id.keys())

        if auto_reset:
            self.rollout_ctx.finalize_attempt(attempt_agent_ids)
            finalized_attempts = self.rollout_ctx.take_finalized_attempts()
            self._publish_finalized_attempts(finalized_attempts)

            previous_attempt_index = self.current_attempt_index
            previous_attempt_group_seed = self._coerce_group_seed(self._current_attempt_group_seed)
            next_attempt_index = previous_attempt_index + 1

            reset_sub_obs, reset_sub_info = self._reset_sub_env(attempt_index=next_attempt_index)
            reset_obs_by_unity_id = self._obs_to_unity_id(reset_sub_obs)

            self.current_attempt_index = next_attempt_index
            next_attempt_group_seed = self._coerce_group_seed(self._current_attempt_group_seed)
            first_attempt_group_seed = None
            if self._attempt_group_seed_history:
                first_attempt_group_seed = self._coerce_group_seed(self._attempt_group_seed_history[0])
            omit_reset_obs_images = (
                next_attempt_index > 1
                and first_attempt_group_seed is not None
                and next_attempt_group_seed == first_attempt_group_seed
            )
            reset_obs_image_omitted_reason = None
            if omit_reset_obs_images:
                reset_obs_image_omitted_reason = (
                    f'omitted because attempt {next_attempt_index} uses the same task seed as attempt 1; '
                    'its initial view is the same as the first attempt initial observation.'
                )
            omit_terminal_obs_images_by_uid = {int(unity_id): True for unity_id in func_render_errors_by_unity_id.keys()}
            terminal_obs_image_omitted_reason_by_uid = {
                int(unity_id): (
                    'omitted because the previous assistant tool/function-call was invalid and no Unity action was executed; '
                    'the visual state is unchanged from the previous observation.'
                )
                for unity_id in func_render_errors_by_unity_id.keys()
            }
            history_snapshot = self._sample_history_snapshot(sorted(reset_obs_by_unity_id.keys()))
            history_snapshot = self._merge_history_snapshot(history_snapshot, finalized_attempts)
            self._last_reset_plan['history_snapshot'] = deepcopy(history_snapshot)
            rollout_obs = self.rollout_ctx.on_attempt_reset(
                env_cfg=self._selected_env_cfg or self.sub_env.env_info_mgr.env_config or {},
                obs=reset_obs_by_unity_id,
                history_snapshot=history_snapshot,
                current_attempt_index=previous_attempt_index,
                next_attempt_index=self.current_attempt_index,
                omit_terminal_obs_images_by_uid=omit_terminal_obs_images_by_uid,
                terminal_obs_image_omitted_reason_by_uid=terminal_obs_image_omitted_reason_by_uid,
                omit_reset_obs_images=omit_reset_obs_images,
                reset_obs_image_omitted_reason=reset_obs_image_omitted_reason,
            )

            rollout_done_by_unity_id = self._rollout_done_dict(rollout_obs.keys(), False)
            rollout_truncated_by_unity_id = self._rollout_truncated_dict(rollout_obs.keys(), False)
            info_out = {
                'truncated': rollout_truncated_by_unity_id,
                'attempt': {
                    'index': previous_attempt_index,
                    'next_index': self.current_attempt_index,
                    'group_seed': previous_attempt_group_seed,
                    'attempt_group_seed': previous_attempt_group_seed,
                    'next_attempt_group_seed': self._coerce_group_seed(self._current_attempt_group_seed),
                    'max_attempts': self.max_attempts,
                    'max_steps_per_attempt': self.max_steps_per_attempt,
                    'done': True,
                    'success': attempt_success,
                    'auto_reset': True,
                },
                'rollout': {
                    'success': False,
                    'truncated': False,
                    'task_name': self._selected_task_name,
                    'group_seed': self._coerce_group_seed(self._selected_rollout_group_seed),
                    'rollout_group_seed': self._coerce_group_seed(self._selected_rollout_group_seed),
                    'current_attempt_group_seed': self._coerce_group_seed(self._current_attempt_group_seed),
                    'attempt_group_seed_history': deepcopy(self._attempt_group_seed_history),
                    'reroll_group_seed_on_same_task': bool(self._reroll_group_seed_on_same_task),
                    'env_id': self._selected_env_id,
                    'current_attempt_index': self.current_attempt_index,
                    'max_attempts': self.max_attempts,
                    'max_steps_per_attempt': self.max_steps_per_attempt,
                    'max_rollout_steps': self.get_rollout_step_budget(),
                },
                'sub_env': {
                    'step': deepcopy(sub_info) if isinstance(sub_info, dict) else {},
                    'reset': deepcopy(reset_sub_info) if isinstance(reset_sub_info, dict) else {},
                    'attempt_done': attempt_done_by_unity_id,
                    'attempt_truncated': sub_truncated_by_unity_id,
                },
                'history': {
                    'bucket_key': self._last_reset_plan.get('history_bucket_key', None),
                    'history_bucket_key': self._last_reset_plan.get('history_bucket_key', None),
                },
            }
            self._emit_env_event(
                'env_step',
                {
                    'actions': serialize_action_details(unity_actions),
                    'next_obs': serialize_obs_map(
                        rollout_obs,
                        text_max_chars=self._hook_text_max_chars,
                        max_images_per_observation=self._hook_max_images_per_observation,
                    ),
                    'reward': reward_by_unity_id,
                    'done': rollout_done_by_unity_id,
                    'info': info_out,
                    'auto_reset': True,
                },
            )
            return rollout_obs, reward_by_unity_id, rollout_done_by_unity_id, info_out

        rollout_done_flags = {
            int(unity_id): bool(rollout_success or rollout_truncated)
            for unity_id in attempt_agent_ids
        }
        rollout_obs = self.rollout_ctx.finalize_obs(next_obs_by_unity_id, rollout_done=rollout_done_flags)

        if rollout_success or rollout_truncated:
            self.rollout_started = False
            self._publish_finalized_attempts()

        rollout_done_by_unity_id = dict(rollout_done_flags)
        rollout_done_by_unity_id['__all__'] = bool(rollout_success or rollout_truncated)
        rollout_truncated_by_unity_id = self._rollout_truncated_dict(attempt_agent_ids, rollout_truncated)

        info_out = {
            'truncated': rollout_truncated_by_unity_id,
            'attempt': {
                'index': self.current_attempt_index,
                'group_seed': self._coerce_group_seed(self._current_attempt_group_seed),
                'attempt_group_seed': self._coerce_group_seed(self._current_attempt_group_seed),
                'max_attempts': self.max_attempts,
                'max_steps_per_attempt': self.max_steps_per_attempt,
                'done': attempt_done,
                'success': attempt_success,
                'auto_reset': False,
            },
            'rollout': {
                'success': rollout_success,
                'truncated': rollout_truncated,
                'task_name': self._selected_task_name,
                'group_seed': self._coerce_group_seed(self._selected_rollout_group_seed),
                'rollout_group_seed': self._coerce_group_seed(self._selected_rollout_group_seed),
                'current_attempt_group_seed': self._coerce_group_seed(self._current_attempt_group_seed),
                'attempt_group_seed_history': deepcopy(self._attempt_group_seed_history),
                'reroll_group_seed_on_same_task': bool(self._reroll_group_seed_on_same_task),
                'env_id': self._selected_env_id,
                'current_attempt_index': self.current_attempt_index,
                'max_attempts': self.max_attempts,
                'max_steps_per_attempt': self.max_steps_per_attempt,
                'max_rollout_steps': self.get_rollout_step_budget(),
            },
            'sub_env': {
                'step': deepcopy(sub_info) if isinstance(sub_info, dict) else {},
                'attempt_done': attempt_done_by_unity_id,
                'attempt_truncated': sub_truncated_by_unity_id,
            },
            'history': {
                'bucket_key': self._last_reset_plan.get('history_bucket_key', None),
                'history_bucket_key': self._last_reset_plan.get('history_bucket_key', None),
            },
        }
        self._emit_env_event(
            'env_step',
            {
                'actions': serialize_action_details(unity_actions),
                'next_obs': serialize_obs_map(
                    rollout_obs,
                    text_max_chars=self._hook_text_max_chars,
                    max_images_per_observation=self._hook_max_images_per_observation,
                ),
                'reward': reward_by_unity_id,
                'done': rollout_done_by_unity_id,
                'info': info_out,
                'auto_reset': False,
            },
        )
        return rollout_obs, reward_by_unity_id, rollout_done_by_unity_id, info_out

    def close(self):
        self.sub_env.close()
        self.rollout_started = False


def _preview_text(value, limit: int = 240) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + '...'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Debug ArkEnv with repeated real Unity attempts')
    parser.add_argument(
        '--env-path',
        default=os.environ.get('AGENTARK_ENV_PATH', None),
        help='Path to the Unity executable. Defaults to AGENTARK_ENV_PATH when set.',
    )
    parser.add_argument(
        '--mod-path',
        default=os.environ.get('AGENTARK_MOD_PATH', None),
        help='Path to the Unity Mods directory. Defaults to AGENTARK_MOD_PATH when set.',
    )
    parser.add_argument('--task-type', default='RLTask', help='Task type passed to EnvInfoManager')
    parser.add_argument('--task-name', default='MarbleStop', help='Task folder name or task identifier')
    parser.add_argument('--group-seed', type=int, default=123, help='Group seed used during reset')
    parser.add_argument('--env-id', type=int, default=0, help='Unity env_id used during reset')
    parser.add_argument('--num-parallel-envs', type=int, default=1, help='Override num_parallel_envs for local debugging')
    parser.add_argument('--max-attempts', type=int, default=2, help='Maximum attempts inside one ArkEnv rollout')
    parser.add_argument('--max-rollout-steps', type=int, default=None, help='Maximum external ArkEnv.step calls to run; defaults to the action sequence length, or 12 without a sequence')
    parser.add_argument(
        '--action',
        default='<params>{"plan":"U7,L7"}</params>',
        help='Action sent on each external step',
    )
    parser.add_argument(
        '--action-sequence',
        default=None,
        help='Optional action trajectory. Pass an inline JSON array/object with actions, or a path to .json, .jsonl, or text file. Each item is sent on one external rollout step.',
    )
    args = parser.parse_args(argv)
    action_payloads = _resolve_cli_action_payloads(args.action, args.action_sequence)
    max_rollout_steps = _resolve_cli_action_step_count(
        default_steps=12,
        requested_steps=args.max_rollout_steps,
        action_payloads=action_payloads,
        has_action_sequence=args.action_sequence is not None,
        option_name='--max-rollout-steps',
    )

    cfg = {
        'env_path': _normalize_optional_path(args.env_path),
        'mod_path': _normalize_optional_path(args.mod_path),
        'task_type': args.task_type,
        'env_config_overrides': {
            'num_parallel_envs': args.num_parallel_envs,
        },
        'max_attempts': args.max_attempts,
    }

    env = ArkEnv(cfg)
    try:
        obs, info = env.reset(
            task_name=args.task_name,
            group_seed=args.group_seed,
            env_id=args.env_id,
            max_attempts=args.max_attempts,
        )
        print('reset_ok=True')
        print(f'obs_keys={sorted(obs.keys())}')
        print(f'rollout_info={info.get("rollout", {})}')

        print(f'max_rollout_steps={max_rollout_steps}')
        print(f'action_count={len(action_payloads)}')

        for rollout_step_idx in range(1, max_rollout_steps + 1):
            if not obs:
                print('empty_obs=True')
                return 1

            unity_id = sorted(obs.keys())[0]
            step_action = action_payloads[min(rollout_step_idx - 1, len(action_payloads) - 1)]
            next_obs, reward, done, step_info = env.step({unity_id: step_action})
            next_item = next_obs.get(unity_id, {}) if isinstance(next_obs, dict) else {}
            print(f'rollout_step={rollout_step_idx}')
            print(f'unity_id={unity_id}')
            print(f'action={step_action}')
            print(f'reward={reward}')
            print(f'done={done}')
            print(f'truncated={step_info.get("truncated", {})}')
            print(f'attempt={step_info.get("attempt", {})}')
            print(f'rollout={step_info.get("rollout", {})}')
            print(f'next_step_msg_preview={_preview_text(next_item.get("step_msg", ""))}')

            if done.get('__all__', False) or step_info.get('truncated', {}).get('__all__', False):
                break
            obs = next_obs

        return 0
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
