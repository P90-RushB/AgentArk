from __future__ import annotations

import random
import threading
import time
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional


def _cfg_value(cfg: Optional[dict], primary_key: str, legacy_key: str, default=None):
    if not isinstance(cfg, dict):
        return default
    if primary_key in cfg:
        return cfg.get(primary_key)
    if legacy_key in cfg:
        return cfg.get(legacy_key)
    return default


def _cfg_int(cfg: Optional[dict], primary_key: str, legacy_key: str, default: int = 0) -> int:
    raw = _cfg_value(cfg, primary_key, legacy_key, default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _deep_merge(dst: dict, src: dict) -> dict:
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = deepcopy(value)
    return dst


def get_wrapper_cfg(env_cfg: Optional[dict]) -> dict:
    if not isinstance(env_cfg, dict):
        return {}
    wrapper_cfg = env_cfg.get('env_wrapper_cfg', {})
    return wrapper_cfg if isinstance(wrapper_cfg, dict) else {}


def get_history_cfg(env_cfg: Optional[dict]) -> dict:
    wrapper_cfg = get_wrapper_cfg(env_cfg)
    ctx_cfg = wrapper_cfg.get('context_manager', {})
    if not isinstance(ctx_cfg, dict):
        return {}
    history_cfg = ctx_cfg.get('history', {})
    return history_cfg if isinstance(history_cfg, dict) else {}


def history_enabled(history_cfg: Optional[dict]) -> bool:
    if not isinstance(history_cfg, dict):
        return False
    return _cfg_int(history_cfg, 'max_history_attempts', 'max_episodes', 0) > 0


def get_history_bucket_mode(env_cfg: Optional[dict]) -> str:
    history_cfg = get_history_cfg(env_cfg)
    mode = history_cfg.get('bucket_mode', 'none') if isinstance(history_cfg, dict) else 'none'
    mode = str(mode or 'none').strip().lower()
    if mode not in ('none', 'task', 'task_group_seed'):
        return 'none'
    return mode


def compute_history_bucket_key(
    env_cfg: Optional[dict],
    *,
    task_name: Optional[str],
    group_seed: Optional[int],
    override_bucket_id: Optional[str] = None,
) -> Optional[str]:
    if override_bucket_id is not None:
        bucket_id = str(override_bucket_id).strip()
        return bucket_id or None

    history_cfg = get_history_cfg(env_cfg)
    if not history_enabled(history_cfg):
        return None

    mode = get_history_bucket_mode(env_cfg)
    task_key = str(task_name or '').strip()
    if not task_key:
        return None

    if mode == 'task':
        return f'task::{task_key}'
    if mode == 'task_group_seed':
        if group_seed is None:
            return None
        return f'task_group_seed::{task_key}::{int(group_seed)}'
    return None


def get_history_retention_cfg(env_cfg: Optional[dict]) -> Dict[str, Any]:
    history_cfg = get_history_cfg(env_cfg)
    history_cfg = history_cfg if isinstance(history_cfg, dict) else {}

    bucket_mode = get_history_bucket_mode(env_cfg)

    raw_max_bucket_count = history_cfg.get('max_bucket_count', None)
    if raw_max_bucket_count is None:
        max_bucket_count = 2048 if bucket_mode == 'task_group_seed' else 0
    else:
        try:
            max_bucket_count = max(0, int(raw_max_bucket_count))
        except Exception:
            max_bucket_count = 0

    raw_bucket_ttl_s = history_cfg.get('bucket_ttl_s', None)
    if raw_bucket_ttl_s is None:
        bucket_ttl_s = 0.0
    else:
        try:
            bucket_ttl_s = max(0.0, float(raw_bucket_ttl_s))
        except Exception:
            bucket_ttl_s = 0.0

    return {
        'max_bucket_count': int(max_bucket_count),
        'bucket_ttl_s': float(bucket_ttl_s),
    }


class SharedEpisodeStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._episodes: Dict[str, Dict[int, deque]] = {}
        self._bucket_last_touched: Dict[str, float] = {}

    def _drop_bucket_locked(self, bucket_key: str):
        self._episodes.pop(bucket_key, None)
        self._bucket_last_touched.pop(bucket_key, None)

    def _touch_bucket_locked(self, bucket_key: Optional[str], now: Optional[float] = None):
        if not bucket_key:
            return
        self._bucket_last_touched[str(bucket_key)] = time.monotonic() if now is None else float(now)

    def _cleanup_locked(self, retention_cfg: Optional[dict]):
        retention_cfg = retention_cfg if isinstance(retention_cfg, dict) else {}
        ttl_s = max(0.0, float(retention_cfg.get('bucket_ttl_s', 0.0) or 0.0))
        max_bucket_count = max(0, int(retention_cfg.get('max_bucket_count', 0) or 0))
        now = time.monotonic()

        if ttl_s > 0:
            expired_keys = [
                bucket_key
                for bucket_key, last_touched in list(self._bucket_last_touched.items())
                if now - float(last_touched) >= ttl_s
            ]
            for bucket_key in expired_keys:
                self._drop_bucket_locked(bucket_key)

        if max_bucket_count > 0 and len(self._episodes) > max_bucket_count:
            ordered_keys = sorted(
                list(self._episodes.keys()),
                key=lambda bucket_key: self._bucket_last_touched.get(bucket_key, 0.0),
            )
            excess = len(self._episodes) - max_bucket_count
            for bucket_key in ordered_keys[:excess]:
                self._drop_bucket_locked(bucket_key)

    @staticmethod
    def _trim_episode(episode: List[dict], history_cfg: dict) -> List[dict]:
        if not isinstance(episode, list):
            return []
        max_steps = max(0, _cfg_int(history_cfg, 'max_history_steps_per_attempt', 'max_steps_per_episode', 0))
        trimmed = deepcopy(episode)
        if max_steps > 0 and len(trimmed) > max_steps:
            trimmed = trimmed[-max_steps:]
        return trimmed

    @staticmethod
    def _sample_episodes(pool: deque, history_cfg: dict) -> List[List[dict]]:
        if not pool:
            return []
        mode = str(history_cfg.get('sample_mode', 'random') or 'random').strip().lower()
        sample_size = max(1, int(history_cfg.get('sample_size', 1) or 1))
        count = min(sample_size, len(pool))

        if mode == 'latest':
            selected = list(pool)[-count:]
        else:
            selected = random.sample(list(pool), count)
        return deepcopy(selected)

    def publish_episode(
        self,
        bucket_key: Optional[str],
        agent_id: int,
        episode: List[dict],
        history_cfg: dict,
        retention_cfg: Optional[dict] = None,
    ):
        if not bucket_key or not history_enabled(history_cfg):
            return

        trimmed = self._trim_episode(episode, history_cfg)
        if not trimmed:
            return

        max_attempts = max(1, _cfg_int(history_cfg, 'max_history_attempts', 'max_episodes', 1))
        with self._lock:
            self._cleanup_locked(retention_cfg)
            by_agent = self._episodes.setdefault(str(bucket_key), {})
            pool = by_agent.setdefault(int(agent_id), deque())
            pool.append(trimmed)
            while len(pool) > max_attempts:
                pool.popleft()
            self._touch_bucket_locked(bucket_key)
            self._cleanup_locked(retention_cfg)

    def sample_snapshot(
        self,
        bucket_key: Optional[str],
        agent_ids: List[int],
        history_cfg: dict,
        retention_cfg: Optional[dict] = None,
    ) -> Dict[int, List[List[dict]]]:
        agent_ids = [int(agent_id) for agent_id in agent_ids]
        empty = {agent_id: [] for agent_id in agent_ids}
        if not bucket_key or not history_enabled(history_cfg):
            return empty

        with self._lock:
            self._cleanup_locked(retention_cfg)
            by_agent = self._episodes.get(str(bucket_key), {})
            if not by_agent:
                return empty
            self._touch_bucket_locked(bucket_key)

            if bool(history_cfg.get('share_across_agents', False)):
                merged = deque()
                for pool in by_agent.values():
                    merged.extend(list(pool))
                shared = self._sample_episodes(merged, history_cfg) if merged else []
                return {agent_id: deepcopy(shared) for agent_id in agent_ids}

            return {
                agent_id: self._sample_episodes(by_agent.get(agent_id, deque()), history_cfg)
                for agent_id in agent_ids
            }

    def get_episode_counts(self, bucket_key: Optional[str], retention_cfg: Optional[dict] = None) -> Dict[int, int]:
        if not bucket_key:
            return {}
        with self._lock:
            self._cleanup_locked(retention_cfg)
            by_agent = self._episodes.get(str(bucket_key), {})
            if by_agent:
                self._touch_bucket_locked(bucket_key)
            return {
                int(agent_id): len(pool)
                for agent_id, pool in by_agent.items()
            }

    def get_bucket_count(self, retention_cfg: Optional[dict] = None) -> int:
        with self._lock:
            self._cleanup_locked(retention_cfg)
            return len(self._episodes)
