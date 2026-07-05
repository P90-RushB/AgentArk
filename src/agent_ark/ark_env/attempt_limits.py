from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def resolve_max_steps_per_attempt(env_cfg: Optional[Dict[str, Any]], default: Optional[int] = None) -> Optional[int]:
    if not isinstance(env_cfg, dict):
        return default

    value = _coerce_int(env_cfg.get('max_steps_per_attempt', None))
    if value is not None and value > 0:
        return int(value)

    return default


def normalize_max_steps_per_attempt(env_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = dict(env_cfg or {})

    value = resolve_max_steps_per_attempt(normalized, default=None)
    if value is None:
        return normalized

    normalized['max_steps_per_attempt'] = int(value)
    return normalized


def derive_rollout_step_budget(max_attempts: Any, max_steps_per_attempt: Any) -> Optional[int]:
    attempts = _coerce_int(max_attempts)
    steps = _coerce_int(max_steps_per_attempt)
    if attempts is None or steps is None or attempts <= 0 or steps <= 0:
        return None
    return int(attempts * steps)


def require_rollout_step_budget(
    *,
    max_attempts: Any,
    max_steps_per_attempt: Any,
    context: str,
) -> Tuple[int, int, int]:
    attempts = _coerce_int(max_attempts)
    steps = _coerce_int(max_steps_per_attempt)

    if attempts is None or attempts <= 0:
        raise ValueError(f'{context} requires a positive max_attempts, got {max_attempts!r}')
    if steps is None or steps <= 0:
        raise ValueError(
            f'{context} requires env_cfg.max_steps_per_attempt '
            f'to be set to a positive integer. Got {max_steps_per_attempt!r}'
        )

    return int(attempts), int(steps), int(attempts * steps)
