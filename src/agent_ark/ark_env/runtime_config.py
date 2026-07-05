from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:
    yaml = None

DEFAULT_RUNTIME_CONFIG: Dict[str, Any] = {
    "server": {
        "host": "http://127.0.0.1",
        "port": 18080,
        "timeout": 1200,
    },
    "interaction": {
        "acquire_env": True,
        "release_env_on_finalize": True,
        "auto_create_env": False,
        "close_env_on_finalize": False,
        "start_options": {},
    },
    "warmup": {
        "num_envs": 2,
        "step_once": False,
        "close_envs_after_warmup": False,
        "env_id_prefix": "warmup-env",
        "port_assignment": {
            "assign_worker_index": True,
            "assign_env_id": False,
            "assign_base_port": False,
            "base_port_start": 5005,
            "base_port_stride": 1,
            "base_port_plan": [],
        },
    },
    "env_cfg": {
        "env_path": "${AGENTARK_ENV_PATH}",
        "mod_path": "${AGENTARK_MOD_PATH}",
        "task_type": "RLTask",
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _expand_env_vars(value: str) -> str:
    expanded = str(value)
    for _ in range(3):
        next_value = os.path.expandvars(expanded)
        if next_value == expanded:
            break
        expanded = next_value
    return os.path.expanduser(expanded)


def _expand_env_vars_in_obj(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_vars(value)
    if isinstance(value, dict):
        return {key: _expand_env_vars_in_obj(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars_in_obj(item) for item in value]
    return value


def load_runtime_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required for .yaml config")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Config must be .yaml/.yml/.json")

    cfg = _expand_env_vars_in_obj(deep_merge(DEFAULT_RUNTIME_CONFIG, data))

    interaction_cfg = dict(cfg.get("interaction", {}) or {})
    if "max_turns" in interaction_cfg:
        raise ValueError(
            "interaction.max_turns has been removed. Rollout step budget is now derived from "
            "env_cfg.max_attempts and env_cfg.max_steps_per_attempt."
        )

    return cfg
