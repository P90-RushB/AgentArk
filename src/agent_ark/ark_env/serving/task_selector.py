"""Deterministic task selection for RL training.

When an RL rollout requests an env with ``task_name=None``, the server picks a
task on its behalf. To keep a whole GRPO group on the *same* task (all ``n``
samples of one prompt share the same group id / ``uid``), selection must be a
pure deterministic function of that group id -- no shared state, no locks.

``HashTaskSelector`` maps a group id string to a ``(task_name, group_seed)``
pair by hashing the id and indexing into the (stably sorted) task list. The same
group id always resolves to the same task and seed, on any process or machine,
as long as the task list is identical (see ``EnvInfoManager.get_task_list``,
which sorts its folders for exactly this reason).

The ``TaskSelector`` interface is intentionally small so a future selector
(e.g. one that samples from a difficulty-weighted, success-rate-driven subset of
tasks) can replace ``HashTaskSelector`` without touching callers.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Protocol, Tuple


# group_seed must fit the env's accepted range: a positive 31-bit int.
_MAX_GROUP_SEED = 2**31 - 2


def _stable_hash(text: str) -> int:
    """Process-independent hash of a string (Python's hash() is salted)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def task_name_of(task_info: Dict[str, Any]) -> str:
    """Return the canonical task name (mod folder name) for a task list entry."""
    return str(task_info.get("folder_name", "") or "")


class TaskSelector(Protocol):
    """Maps a group id to a concrete (task_name, group_seed)."""

    def select(self, group_id: str, task_list: List[Dict[str, Any]]) -> Tuple[str, int]:
        ...


class HashTaskSelector:
    """Uniform, stateless task selection by hashing the group id.

    - ``task_name`` = ``task_list[hash(group_id) % len(task_list)]`` -> uniform.
    - ``group_seed`` is derived from a separate hash of the group id so that all
      samples in one group also share the same env seed (identical environment
      except for sampling randomness).
    """

    def select(self, group_id: str, task_list: List[Dict[str, Any]]) -> Tuple[str, int]:
        if not task_list:
            raise ValueError("task_list is empty; cannot select a task")
        if not group_id:
            raise ValueError("group_id is required for deterministic task selection")

        idx = _stable_hash(group_id) % len(task_list)
        task_name = task_name_of(task_list[idx])
        group_seed = (_stable_hash(group_id + "::seed") % _MAX_GROUP_SEED) + 1
        return task_name, group_seed


_DEFAULT_SELECTOR: TaskSelector = HashTaskSelector()


def get_default_selector() -> TaskSelector:
    return _DEFAULT_SELECTOR


def resolve_task_for_group(
    group_id: Optional[str],
    task_list: List[Dict[str, Any]],
    selector: Optional[TaskSelector] = None,
) -> Tuple[str, int]:
    """Resolve (task_name, group_seed) for a group id using the given selector."""
    selector = selector or _DEFAULT_SELECTOR
    return selector.select(str(group_id), task_list)
