from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None

from agent_ark.agent.api_agent import APIAgent
from agent_ark.ark_env import ArkEnv, ensure_runtime_pool_range, require_rollout_step_budget, resolve_runtime_sandbox_cfg
from agent_ark.ark_env.ark_sub_env import _normalize_optional_path, _resolve_cli_mod_path
from agent_ark.ark_env.direct_env import EnvInfoManager
from agent_ark.interaction import HookManager, HumanActionBroker, HumanInteractiveAgent, LocalViewerHook
from agent_ark.interaction.serialization import serialize_action_details, serialize_obs_map


_trajectory_io = importlib.import_module('agent_ark.ark_eval.trajectory_io')
TrajectoryJsonlWriter = _trajectory_io.TrajectoryJsonlWriter
build_eval_trajectory_record = _trajectory_io.build_eval_trajectory_record
count_history_prefix_attempts = _trajectory_io.count_history_prefix_attempts
history_snapshot_from_record = _trajectory_io.history_snapshot_from_record
load_trajectory_record = _trajectory_io.load_trajectory_record
target_attempt_index_from_record = _trajectory_io.target_attempt_index_from_record


_MAX_GROUP_SEED = 2**31 - 2
ResumeKey = Tuple[str, str, str, str]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if lowered in ('0', 'false', 'no', 'n', 'off'):
            return False
    return bool(default)


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


def normalize_eval_env_paths(env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(_expand_env_vars_in_obj(env_cfg or {}))
    out['env_path'] = _normalize_optional_path(out.get('env_path', None))

    mod_path = _normalize_optional_path(out.get('mod_path', None))
    sandbox_cfg = out.get('runtime_sandbox', {}) if isinstance(out, dict) else {}
    sandbox_enabled = _coerce_bool(
        sandbox_cfg.get('enabled', False) if isinstance(sandbox_cfg, dict) else False,
        default=False,
    )

    if sandbox_enabled:
        out['mod_path'] = mod_path
    else:
        out['mod_path'] = _resolve_cli_mod_path(mod_path, out.get('env_path', None))
    return out


def load_eval_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')

    suffix = path.suffix.lower()
    if suffix in ('.yaml', '.yml'):
        if yaml is None:
            raise RuntimeError('PyYAML is required for .yaml config files')
        with path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    elif suffix == '.json':
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError('Config must be .yaml/.yml/.json')

    if not isinstance(data, dict):
        raise ValueError('Evaluation config root must be a mapping/object')

    cfg = dict(data)

    env_cfg = dict(cfg.get('env_cfg', {}) or {})
    env_cfg_overrides = dict(env_cfg.get('env_config_overrides', {}) or {})
    env_cfg_overrides.setdefault('num_parallel_envs', 1)
    env_cfg['env_config_overrides'] = env_cfg_overrides
    env_cfg = normalize_eval_env_paths(env_cfg)
    cfg['env_cfg'] = env_cfg

    eval_cfg = dict(cfg.get('eval', {}) or {})
    if 'max_turns' in eval_cfg:
        raise ValueError(
            'eval.max_turns has been removed. Rollout step budget is now derived from '
            'env_cfg.env_config_overrides.max_attempts and the task runtime max_steps_per_attempt '
            'and no legacy step-limit aliases are accepted.'
        )
    return cfg


def _to_jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(type(value).__name__)


def _truncate_text(text: Any, max_len: int = 500) -> str:
    if not isinstance(text, str):
        text = '' if text is None else str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + '...[truncated]'


def _message_text_preview(content: Any, max_len: int = 2000) -> str:
    if isinstance(content, str):
        return _truncate_text(content, max_len=max_len)
    if isinstance(content, list):
        text_parts: List[str] = []
        image_count = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'text' and part.get('text'):
                text_parts.append(str(part.get('text')))
            elif part.get('type') in ('image', 'image_url'):
                image_count += 1
        merged = '\n'.join(text_parts)
        if image_count > 0:
            suffix = f'\n[image_parts={image_count}]'
            merged = merged + suffix if merged else suffix.lstrip()
        return _truncate_text(merged, max_len=max_len)
    return _truncate_text(content, max_len=max_len)


def summarize_obs(unity_id: int, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    history = obs_dict.get('history', []) if isinstance(obs_dict, dict) else []
    messages = obs_dict.get('messages', []) if isinstance(obs_dict, dict) else []
    history_attempt_count = len(history) if isinstance(history, list) else 0
    history_attempt_step_counts = [
        len(ep) if isinstance(ep, list) else 0
        for ep in (history[:5] if isinstance(history, list) else [])
    ]

    summary: Dict[str, Any] = {
        'unity_id': int(unity_id),
        'step_msg': _truncate_text(obs_dict.get('step_msg', '') if isinstance(obs_dict, dict) else '', max_len=2000),
        'history_attempt_count': history_attempt_count,
        'history_attempt_step_counts': history_attempt_step_counts,
        'history_episode_count': history_attempt_count,
        'history_step_counts': history_attempt_step_counts,
    }

    if isinstance(messages, list):
        preview_msgs = list(messages[:2])
        if len(messages) > 2:
            tail = messages[-2:]
            for msg in tail:
                if msg not in preview_msgs:
                    preview_msgs.append(msg)
        summary['message_count'] = len(messages)
        summary['message_roles'] = [msg.get('role') for msg in messages if isinstance(msg, dict)]
        summary['messages'] = [
            {
                'role': msg.get('role'),
                'text_preview': _message_text_preview(msg.get('content'), max_len=2000),
            }
            for msg in preview_msgs
            if isinstance(msg, dict)
        ]
    else:
        summary['message_count'] = 0
        summary['message_roles'] = []
        summary['messages'] = []

    return summary


def summarize_seed_fields(
    initial_env_cfg: Dict[str, Any] | None,
    final_env_cfg: Dict[str, Any] | None,
    requested_group_seed: int,
) -> Dict[str, Any]:
    initial_cfg = dict(initial_env_cfg or {})
    final_cfg = dict(final_env_cfg or {})

    rollout_group_seed = final_cfg.get(
        'rollout_group_seed',
        initial_cfg.get('rollout_group_seed', requested_group_seed),
    )
    initial_attempt_group_seed = initial_cfg.get(
        'attempt_group_seed',
        initial_cfg.get('group_seed', rollout_group_seed),
    )
    final_attempt_group_seed = final_cfg.get(
        'attempt_group_seed',
        final_cfg.get('group_seed', initial_attempt_group_seed),
    )

    attempt_group_seed_history = final_cfg.get(
        'attempt_group_seed_history',
        initial_cfg.get('attempt_group_seed_history', []),
    )
    if not isinstance(attempt_group_seed_history, list):
        attempt_group_seed_history = []

    return {
        'actual_group_seed': rollout_group_seed,
        'actual_rollout_group_seed': rollout_group_seed,
        'initial_attempt_group_seed': initial_attempt_group_seed,
        'final_attempt_group_seed': final_attempt_group_seed,
        'attempt_group_seed_history': attempt_group_seed_history,
        'reroll_group_seed_on_same_task': bool(
            final_cfg.get(
                'reroll_group_seed_on_same_task',
                initial_cfg.get('reroll_group_seed_on_same_task', False),
            )
        ),
    }


def _resolve_eval_env_config(env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    info_mgr = EnvInfoManager(dict(env_cfg or {}))
    base_env_config = info_mgr._read_base_env_config()
    if not isinstance(base_env_config, dict):
        base_env_config = {}

    overrides = (env_cfg or {}).get('env_config_overrides', None)
    if isinstance(overrides, dict) and overrides:
        base_env_config = EnvInfoManager._apply_env_config_overrides(base_env_config, overrides)
    return base_env_config


def _eval_env_uses_task_store(env_cfg: Dict[str, Any]) -> bool:
    return EnvInfoManager._uses_task_store(_resolve_eval_env_config(env_cfg))


def _default_prefab_task_name(env_cfg: Dict[str, Any]) -> str:
    resolved_env_config = _resolve_eval_env_config(env_cfg)
    return str(resolved_env_config.get('task_name') or 'prefab').strip() or 'prefab'


def _coerce_eval_group_seed(value: Any, *, context: str) -> int:
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{context} must be an integer group_seed') from exc

    if seed < 1 or seed > _MAX_GROUP_SEED:
        raise ValueError(f'{context} must be in [1, {_MAX_GROUP_SEED}], got {seed}')
    return seed


def _expand_group_seed_mapping(seed_cfg: Dict[str, Any], *, context: str) -> List[int]:
    start_value = seed_cfg.get('start', seed_cfg.get('from', None))
    end_value = seed_cfg.get('end', seed_cfg.get('to', None))
    if start_value is None or end_value is None:
        raise ValueError(f'{context} must include start/end')

    start = _coerce_eval_group_seed(start_value, context=f'{context}.start')
    end = _coerce_eval_group_seed(end_value, context=f'{context}.end')
    try:
        step = int(seed_cfg.get('step', 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{context}.step must be an integer') from exc

    if step == 0:
        raise ValueError(f'{context}.step must not be 0')
    if start < end and step < 0:
        raise ValueError(f'{context}.step must be positive when start < end')
    if start > end and step > 0:
        raise ValueError(f'{context}.step must be negative when start > end')

    stop = end + (1 if step > 0 else -1)
    return [
        _coerce_eval_group_seed(seed, context=f'{context} range')
        for seed in range(start, stop, step)
    ]


def expand_eval_group_seeds(eval_cfg: Dict[str, Any]) -> List[int]:
    if 'group_seeds' not in (eval_cfg or {}):
        return []

    raw_group_seeds = eval_cfg.get('group_seeds', None)
    if raw_group_seeds is None or raw_group_seeds == '':
        return []

    if isinstance(raw_group_seeds, dict):
        return _expand_group_seed_mapping(raw_group_seeds, context='eval.group_seeds')

    if isinstance(raw_group_seeds, (list, tuple)):
        seeds: List[int] = []
        for index, seed_item in enumerate(raw_group_seeds):
            context = f'eval.group_seeds[{index}]'
            if isinstance(seed_item, dict):
                seeds.extend(_expand_group_seed_mapping(seed_item, context=context))
            else:
                seeds.append(_coerce_eval_group_seed(seed_item, context=context))
        return seeds

    return [_coerce_eval_group_seed(raw_group_seeds, context='eval.group_seeds')]


def _case_id_task_slug(task_name: str) -> str:
    slug = ''.join(
        char if char.isalnum() or char in ('-', '_') else '-'
        for char in str(task_name or '').strip()
    ).strip('-')
    return slug or 'task'


def _task_info_available_names(task_infos: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for info in task_infos:
        folder_name = str(info.get('folder_name') or '').strip()
        if folder_name:
            names.append(folder_name)
        for alias in info.get('aliases') or []:
            alias_text = str(alias or '').strip()
            if alias_text:
                names.append(alias_text)
    return sorted(set(names))


def build_eval_cases(env_cfg: Dict[str, Any], eval_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    uses_task_store = _eval_env_uses_task_store(env_cfg)
    available_task_names: List[str] = []
    if uses_task_store:
        task_infos = EnvInfoManager.get_task_list(env_cfg['mod_path'])
        available_task_names = _task_info_available_names(task_infos)

    rng = random.Random(int(eval_cfg.get('global_seed', 12345)))
    fixed_env_id = eval_cfg.get('fixed_env_id', 0)
    cases_cfg = eval_cfg.get('cases', []) or []

    cases: List[Dict[str, Any]] = []
    if cases_cfg:
        task_names_in_cases = set()
        for idx, case_cfg in enumerate(cases_cfg):
            if not isinstance(case_cfg, dict):
                raise ValueError(f'eval.cases[{idx}] must be an object')

            task_name = str(case_cfg.get('task_name', '')).strip()
            if not task_name:
                raise ValueError(f'eval.cases[{idx}] is missing task_name')
            if uses_task_store and task_name not in available_task_names:
                raise ValueError(
                    f"Unknown task_name={task_name!r} in eval.cases[{idx}]. "
                    f"Available tasks: {available_task_names}"
                )
            task_names_in_cases.add(task_name)

            group_seed = case_cfg.get('group_seed', None)
            if group_seed is None:
                group_seed = int(rng.randint(1, _MAX_GROUP_SEED))
            group_seed = _coerce_eval_group_seed(group_seed, context=f'eval.cases[{idx}].group_seed')

            env_id = case_cfg.get('env_id', fixed_env_id)
            case_id = str(case_cfg.get('case_id', f'case-{idx:04d}'))

            case = {
                'case_id': case_id,
                'task_name': task_name,
                'group_seed': group_seed,
            }
            if env_id is not None:
                case['env_id'] = int(env_id)
            for optional_field in (
                'trajectory_ref',
                'trajectory_path',
                'trajectory_id',
                'trajectory_index',
                'trajectory_prefix_attempts',
                'history_snapshot',
                'start_attempt_index',
                'max_attempts',
            ):
                if optional_field in case_cfg:
                    case[optional_field] = deepcopy(case_cfg[optional_field])
            cases.append(case)

        if len(task_names_in_cases) > 1:
            raise ValueError(
                'Evaluation now expects exactly one task per run. '
                f'Got multiple task_names in eval.cases: {sorted(task_names_in_cases)}'
            )
        return cases

    requested_task_names = [str(t).strip() for t in (eval_cfg.get('task_names', []) or []) if str(t).strip()]
    if not requested_task_names:
        if uses_task_store:
            raise ValueError(
                'Evaluation now expects exactly one task per run. '
                'Set eval.task_names to a single task_name, or provide eval.cases for one task.'
            )
        requested_task_names = [_default_prefab_task_name(env_cfg)]

    invalid = sorted(set(requested_task_names) - set(available_task_names)) if uses_task_store else []
    if invalid:
        raise ValueError(f'Unknown task_names in eval.task_names: {invalid}')

    candidate_task_names = list(dict.fromkeys(requested_task_names))
    if len(candidate_task_names) != 1:
        raise ValueError(
            'Evaluation now expects exactly one task per run. '
            f'Got eval.task_names={candidate_task_names}'
        )

    task_name = candidate_task_names[0]
    group_seeds = expand_eval_group_seeds(eval_cfg)
    if group_seeds:
        task_slug = _case_id_task_slug(task_name)
        for group_seed in group_seeds:
            case = {
                'case_id': f'{task_slug}-seed-{group_seed:04d}',
                'task_name': task_name,
                'group_seed': group_seed,
            }
            if fixed_env_id is not None:
                case['env_id'] = int(fixed_env_id)
            cases.append(case)
        return cases

    for idx in range(int(eval_cfg.get('num_cases', 1))):
        case = {
            'case_id': f'case-{idx:04d}',
            'task_name': task_name,
            'group_seed': int(rng.randint(1, _MAX_GROUP_SEED)),
        }
        if fixed_env_id is not None:
            case['env_id'] = int(fixed_env_id)
        cases.append(case)
    return cases


def _hook_visualization_limits(hooks_cfg: Dict[str, Any]) -> Dict[str, int]:
    visualization_cfg = hooks_cfg.get('visualization', {}) if isinstance(hooks_cfg.get('visualization', {}), dict) else {}
    return {
        'text_max_chars': int(visualization_cfg.get('text_max_chars', 6000) or 6000),
        'max_images_per_observation': int(visualization_cfg.get('max_images_per_observation', 4) or 4),
    }


def _hook_keep_open_on_end(hooks_cfg: Dict[str, Any] | None) -> bool:
    hooks_cfg = dict(hooks_cfg or {})
    visualization_cfg = hooks_cfg.get('visualization', {}) if isinstance(hooks_cfg.get('visualization', {}), dict) else {}
    return _coerce_bool(visualization_cfg.get('keep_open_on_end', False), default=False)


def build_eval_hook_manager(hooks_cfg: Dict[str, Any]) -> tuple[HookManager, HumanActionBroker | None]:
    hooks_cfg = dict(hooks_cfg or {})
    visualization_cfg = hooks_cfg.get('visualization', {}) if isinstance(hooks_cfg.get('visualization', {}), dict) else {}
    human_cfg = hooks_cfg.get('human_interaction', {}) if isinstance(hooks_cfg.get('human_interaction', {}), dict) else {}

    visualization_enabled = _coerce_bool(visualization_cfg.get('enabled', False), default=False)
    human_enabled = _coerce_bool(human_cfg.get('enabled', False), default=False)
    manager = HookManager()
    action_broker = HumanActionBroker() if human_enabled else None

    if visualization_enabled or human_enabled:
        viewer = LocalViewerHook(
            host=str(visualization_cfg.get('host', '127.0.0.1') or '127.0.0.1'),
            port=int(visualization_cfg.get('port', 18181) or 18181),
            event_buffer_size=int(visualization_cfg.get('event_buffer_size', 500) or 500),
            open_browser=human_enabled or _coerce_bool(visualization_cfg.get('open_browser', False), default=False),
            action_broker=action_broker,
        )
        manager.add_hook(viewer)
        if action_broker is None:
            action_broker = viewer.action_broker
    return manager, action_broker


def build_model_runtimes(
    model_cfgs: List[Dict[str, Any]],
    *,
    hooks_cfg: Dict[str, Any] | None = None,
    hook_manager: HookManager | None = None,
    action_broker: HumanActionBroker | None = None,
) -> List[Dict[str, Any]]:
    hooks_cfg = dict(hooks_cfg or {})
    human_cfg = hooks_cfg.get('human_interaction', {}) if isinstance(hooks_cfg.get('human_interaction', {}), dict) else {}
    if _coerce_bool(human_cfg.get('enabled', False), default=False):
        limits = _hook_visualization_limits(hooks_cfg)
        agent = HumanInteractiveAgent(
            name=str(human_cfg.get('name', 'human-local') or 'human-local'),
            action_broker=action_broker,
            hooks=hook_manager,
            timeout_s=human_cfg.get('timeout_s', None),
            text_max_chars=limits['text_max_chars'],
            max_images_per_observation=limits['max_images_per_observation'],
        )
        return [{
            'name': agent.name,
            'model': 'human',
            'provider': 'human',
            'base_url': None,
            'api_key_env': None,
            'temperature': None,
            'agent': agent,
            'human_interaction': True,
        }]

    runtimes: List[Dict[str, Any]] = []
    for idx, model_cfg in enumerate(model_cfgs):
        if not isinstance(model_cfg, dict):
            raise ValueError(f'models[{idx}] must be an object')

        model_name = str(model_cfg.get('name', f'model-{idx:02d}')).strip()
        model_id = str(model_cfg.get('model', '')).strip()
        if not model_id:
            raise ValueError(f'models[{idx}] is missing model')

        base_url = model_cfg.get('base_url', None)
        provider = str(model_cfg.get('provider', 'auto')).strip().lower() or 'auto'
        api_key = model_cfg.get('api_key', None)
        api_key_env = str(model_cfg.get('api_key_env', '')).strip() or None
        if api_key is None and api_key_env is not None:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f'models[{idx}] expects env var {api_key_env}, but it is not set')

        temperature = float(model_cfg.get('temperature', 0.2))
        timeout_s = model_cfg.get('timeout_s', model_cfg.get('request_timeout_s', 180.0))
        if timeout_s is not None:
            timeout_s = float(timeout_s)
        max_retries = int(model_cfg.get('max_retries', 2))
        agent = APIAgent(
            name=model_name,
            api_key=api_key,
            base_url=base_url,
            model=model_id,
            temperature=temperature,
            provider=provider,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        runtimes.append({
            'name': model_name,
            'model': model_id,
            'provider': provider,
            'base_url': base_url,
            'api_key_env': api_key_env,
            'temperature': temperature,
            'timeout_s': timeout_s,
            'max_retries': max_retries,
            'agent': agent,
        })
    if not runtimes:
        raise ValueError('No models configured. Add at least one item under models.')
    return runtimes


def pick_single_unity_agent(obs: Dict[int, Dict[str, Any]]) -> int:
    if not isinstance(obs, dict) or not obs:
        raise RuntimeError('Env reset returned empty obs')
    if len(obs) != 1:
        raise RuntimeError(
            'run_api_agent currently supports one unity_id per ArkEnv for fair evaluation. '
            'Set env_cfg.env_config_overrides.num_parallel_envs=1 for now.'
        )
    return int(sorted(obs.keys())[0])


def collect_step_messages(next_obs: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for unity_id, obs_dict in (next_obs or {}).items():
        if not isinstance(obs_dict, dict):
            continue
        messages.append(summarize_obs(int(unity_id), obs_dict))
    return messages


def _summarize_raw_trace_image_value(value: Any) -> str:
    if isinstance(value, str):
        if value.startswith('data:image/'):
            return f'[omitted image data URL payload; length={len(value)} chars]'
        return _truncate_text(value, max_len=120)

    try:
        from PIL import Image as _PILImage  # local import to keep startup light
    except Exception:  # pragma: no cover
        _PILImage = None

    if _PILImage is not None and isinstance(value, _PILImage.Image):
        return f'[PIL.Image size={value.size} mode={value.mode}]'

    shape = getattr(value, 'shape', None)
    dtype = getattr(value, 'dtype', None)
    if shape is not None:
        return f'[image array shape={tuple(shape)} dtype={dtype}]'

    return f'[{type(value).__name__} image payload omitted]'


def _sanitize_raw_trace_message_part(part: Any) -> Any:
    if isinstance(part, list):
        return [_sanitize_raw_trace_message_part(item) for item in part]

    if not isinstance(part, dict):
        return part

    part_type = part.get('type', None)
    if part_type == 'image_url':
        image_url = part.get('image_url', None)
        if isinstance(image_url, dict):
            out = dict(part)
            out['image_url'] = dict(image_url)
            out['image_url']['url'] = _summarize_raw_trace_image_value(image_url.get('url', ''))
            return out

        out = dict(part)
        out['image_url'] = _summarize_raw_trace_image_value(image_url)
        return out

    if part_type == 'image':
        out = dict(part)
        if 'image_base64' in out:
            out['image_base64'] = _summarize_raw_trace_image_value(out.get('image_base64'))
        if 'image' in out:
            out['image'] = _summarize_raw_trace_image_value(out.get('image'))
        return out

    return {
        key: _sanitize_raw_trace_message_part(value)
        for key, value in part.items()
    }


def sanitize_raw_request_messages(messages: Any) -> Any:
    return _sanitize_raw_trace_message_part(messages)


def collect_raw_request_messages(agent: Any, obs: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    builder = getattr(agent, 'build_request_messages', None)
    if not callable(builder):
        return {}

    try:
        messages_by_agent = builder(obs)
    except Exception:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(messages_by_agent, dict):
        return out

    for agent_id, messages in messages_by_agent.items():
        out[str(agent_id)] = {'request_messages': _to_jsonable(sanitize_raw_request_messages(deepcopy(messages)))}
    return out


def normalize_response_trace(trace_by_agent: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(trace_by_agent, dict):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for agent_id, trace in trace_by_agent.items():
        if not isinstance(trace, dict):
            continue

        item: Dict[str, Any] = {}
        if 'assistant_raw' in trace:
            item['assistant_raw'] = _to_jsonable(trace.get('assistant_raw'))
        if 'action_extracted' in trace:
            item['action_extracted'] = _to_jsonable(trace.get('action_extracted'))
        if 'usage' in trace:
            item['usage'] = _to_jsonable(trace.get('usage'))
        if 'skipped' in trace:
            item['skipped'] = bool(trace.get('skipped', False))
        if item:
            out[str(agent_id)] = item
    return out


def forward_agent_with_trace(
    agent: Any,
    obs: Dict[int, Dict[str, Any]],
) -> tuple[Dict[int, Any], Dict[str, Dict[str, Any]]]:
    forward_with_trace = getattr(agent, 'forward_with_trace', None)
    if callable(forward_with_trace):
        code_act, trace_by_agent = forward_with_trace(obs)
        return code_act, normalize_response_trace(trace_by_agent)

    return agent.forward_with_details(obs), {}


def preview_action(code_act: Dict[int, Any]) -> str:
    if not isinstance(code_act, dict):
        return _truncate_text(code_act, max_len=400)
    for _, action_text in sorted(code_act.items(), key=lambda item: int(item[0])):
        if isinstance(action_text, dict):
            action_text = action_text.get('action', None)
        if action_text is None:
            continue
        return _truncate_text(action_text, max_len=400)
    return ''


def preview_assistant(code_act: Dict[int, Any]) -> str:
    if not isinstance(code_act, dict):
        return ''
    for _, payload in sorted(code_act.items(), key=lambda item: int(item[0])):
        assistant_text = payload.get('assistant', None) if isinstance(payload, dict) else None
        if assistant_text is None:
            continue
        return _truncate_text(assistant_text, max_len=400)
    return ''


def summarize_attempt_rewards(step_records: List[Dict[str, Any]], reset_info: Dict[str, Any] | None = None) -> Dict[str, Any]:
    attempts: Dict[int, Dict[str, Any]] = {}
    base_attempt_index = 1
    if isinstance(reset_info, dict):
        attempt_info = reset_info.get('attempt', {}) if isinstance(reset_info.get('attempt', {}), dict) else {}
        try:
            base_attempt_index = max(1, int(attempt_info.get('index', 1) or 1))
        except Exception:
            base_attempt_index = 1

    for step in step_records or []:
        if not isinstance(step, dict):
            continue
        info = step.get('info', {}) if isinstance(step.get('info', {}), dict) else {}
        attempt_info = info.get('attempt', {}) if isinstance(info.get('attempt', {}), dict) else {}
        try:
            attempt_index = max(1, int(attempt_info.get('index', base_attempt_index) or base_attempt_index))
        except Exception:
            attempt_index = base_attempt_index

        entry = attempts.setdefault(attempt_index, {
            'index': int(attempt_index),
            'reward_total': 0.0,
            'turns': 0,
            'done': False,
            'success': False,
            'auto_reset': False,
        })
        entry['reward_total'] += float(step.get('reward_total', 0.0) or 0.0)
        entry['turns'] += 1
        entry['done'] = bool(attempt_info.get('done', False) or entry['done'])
        entry['success'] = bool(attempt_info.get('success', False) or entry['success'])
        entry['auto_reset'] = bool(attempt_info.get('auto_reset', False) or entry['auto_reset'])

    if not attempts:
        attempts[base_attempt_index] = {
            'index': int(base_attempt_index),
            'reward_total': 0.0,
            'turns': 0,
            'done': False,
            'success': False,
            'auto_reset': False,
        }

    ordered_attempts = [attempts[idx] for idx in sorted(attempts.keys())]
    last_attempt = ordered_attempts[-1]
    best_attempt = max(ordered_attempts, key=lambda item: float(item.get('reward_total', 0.0)))
    successful_attempts = [item for item in ordered_attempts if bool(item.get('success', False))]
    first_success_attempt = successful_attempts[0] if successful_attempts else None
    return {
        'attempts': _to_jsonable(ordered_attempts),
        'attempt_count': len(ordered_attempts),
        'last_attempt_index': int(last_attempt['index']),
        'last_attempt_reward': float(last_attempt.get('reward_total', 0.0)),
        'best_attempt_reward': float(best_attempt.get('reward_total', 0.0)),
        'final_attempt_success': bool(last_attempt.get('success', False)),
        'ever_attempt_success': bool(successful_attempts),
        'first_success_attempt_index': (
            int(first_success_attempt['index']) if isinstance(first_success_attempt, dict) else None
        ),
        'success_attempt_count': len(successful_attempts),
    }


def _run_case_rollout(
    env: ArkEnv,
    model_runtime: Dict[str, Any],
    case: Dict[str, Any],
    *,
    max_attempts: int | None = None,
    hook_manager: HookManager | None = None,
    hooks_cfg: Dict[str, Any] | None = None,
    phase: str = 'eval',
) -> Dict[str, Any]:
    agent = model_runtime['agent']
    hooks = hook_manager or HookManager()
    limits = _hook_visualization_limits(dict(hooks_cfg or {}))
    requested_task_name = case['task_name']
    requested_group_seed = int(case['group_seed'])
    requested_env_id = case.get('env_id', None)
    history_snapshot = case.get('history_snapshot', None) if isinstance(case, dict) else None
    start_attempt_index = case.get('start_attempt_index', None) if isinstance(case, dict) else None
    case_max_attempts = case.get('max_attempts', max_attempts) if isinstance(case, dict) else max_attempts

    started_at = time.time()
    obs, info = env.reset(
        task_name=requested_task_name,
        group_seed=requested_group_seed,
        env_id=requested_env_id,
        history_snapshot=history_snapshot,
        max_attempts=case_max_attempts,
        start_attempt_index=start_attempt_index,
    )
    agent.reset()

    unity_id = pick_single_unity_agent(obs)
    initial_obs = obs[unity_id]
    initial_env_cfg = getattr(env, '_selected_env_cfg', None) or getattr(env.sub_env.env_info_mgr, 'env_config', {}) or {}
    resolved_max_attempts, max_steps_per_attempt, rollout_step_budget = require_rollout_step_budget(
        max_attempts=getattr(env, 'max_attempts', None),
        max_steps_per_attempt=getattr(env, 'max_steps_per_attempt', None),
        context=f"run_api_agent case={case['case_id']}",
    )

    step_records: List[Dict[str, Any]] = []
    total_reward = 0.0
    rollout_success = False
    rollout_truncated = False
    rollout_budget_exhausted = False

    for turn_idx in range(rollout_step_budget):
        request_messages = collect_step_messages(obs)
        raw_request_by_agent = collect_raw_request_messages(agent, obs)
        hooks.emit(
            'agent_request',
            {
                'case_id': case['case_id'],
                'model_name': model_runtime['name'],
                'turn_index': int(turn_idx),
                'obs': serialize_obs_map(
                    obs,
                    text_max_chars=limits['text_max_chars'],
                    max_images_per_observation=limits['max_images_per_observation'],
                ),
                'request_messages': request_messages,
                'raw_trace_by_agent': raw_request_by_agent,
            },
            source='run_api_agent',
            phase=phase,
        )
        code_act, raw_response_by_agent = forward_agent_with_trace(agent, obs)
        hooks.emit(
            'agent_response',
            {
                'case_id': case['case_id'],
                'model_name': model_runtime['name'],
                'turn_index': int(turn_idx),
                'actions': serialize_action_details(code_act),
                'action_preview': preview_action(code_act),
                'assistant_preview': preview_assistant(code_act),
                'raw_trace_by_agent': raw_response_by_agent,
            },
            source='run_api_agent',
            phase=phase,
        )
        next_obs, reward, done, step_info = env.step(code_act, info={'turn_index': int(turn_idx)})
        rollout_info = step_info.get('rollout', {}) if isinstance(step_info, dict) else {}

        reward_total = float(sum(float(v) for v in (reward or {}).values()))
        total_reward += reward_total
        step_records.append({
            'turn_index': int(turn_idx),
            'action_preview': preview_action(code_act),
            'assistant_preview': preview_assistant(code_act),
            'reward_total': reward_total,
            'reward_by_agent': _to_jsonable(reward or {}),
            'done': _to_jsonable(done or {}),
            'info': _to_jsonable(step_info or {}),
            'request_messages': request_messages,
            'raw_request_by_agent': raw_request_by_agent,
            'raw_response_by_agent': raw_response_by_agent,
            'step_messages': collect_step_messages(next_obs),
        })

        rollout_success = bool(rollout_info.get('success', False))
        rollout_truncated = bool(rollout_info.get('truncated', False))
        if rollout_success or rollout_truncated:
            obs = next_obs
            break
        obs = next_obs

    if not rollout_success and not rollout_truncated:
        rollout_truncated = True
        rollout_budget_exhausted = True

    final_env_cfg = getattr(env, '_selected_env_cfg', None) or getattr(env.sub_env.env_info_mgr, 'env_config', {}) or {}

    return {
        'requested_task_name': requested_task_name,
        'requested_group_seed': requested_group_seed,
        'requested_env_id': requested_env_id,
        'initial_env_cfg': initial_env_cfg,
        'final_env_cfg': final_env_cfg,
        'unity_id': unity_id,
        'initial_obs': initial_obs,
        'reset_info': info or {},
        'loaded_history_attempt_count': count_history_prefix_attempts(history_snapshot),
        'start_attempt_index': getattr(env, 'current_attempt_index', start_attempt_index or 1),
        'turns': len(step_records),
        'total_reward': total_reward,
        'max_attempts': resolved_max_attempts,
        'max_steps_per_attempt': max_steps_per_attempt,
        'rollout_step_budget': rollout_step_budget,
        'rollout_budget_exhausted': rollout_budget_exhausted,
        'rollout_success': rollout_success,
        'rollout_truncated': rollout_truncated,
        'step_records': step_records,
        'duration_s': round(time.time() - started_at, 4),
    }


def evaluate_case(
    env: ArkEnv,
    model_runtime: Dict[str, Any],
    case: Dict[str, Any],
    *,
    hook_manager: HookManager | None = None,
    hooks_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rollout = _run_case_rollout(
        env=env,
        model_runtime=model_runtime,
        case=case,
        max_attempts=None,
        hook_manager=hook_manager,
        hooks_cfg=hooks_cfg,
        phase='eval',
    )
    attempt_summary = summarize_attempt_rewards(rollout['step_records'], rollout['reset_info'])
    seed_summary = summarize_seed_fields(
        rollout.get('initial_env_cfg'),
        rollout.get('final_env_cfg'),
        rollout['requested_group_seed'],
    )

    return {
        'status': 'ok',
        'case_id': case['case_id'],
        'model_name': model_runtime['name'],
        'model': model_runtime['model'],
        'provider': model_runtime.get('provider'),
        'base_url': model_runtime['base_url'],
        'api_key_env': model_runtime['api_key_env'],
        'requested_task_name': rollout['requested_task_name'],
        'requested_group_seed': rollout['requested_group_seed'],
        'requested_env_id': rollout['requested_env_id'],
        'actual_task_name': rollout['final_env_cfg'].get('task_name', rollout['initial_env_cfg'].get('task_name')),
        'actual_group_seed': seed_summary['actual_group_seed'],
        'actual_rollout_group_seed': seed_summary['actual_rollout_group_seed'],
        'initial_attempt_group_seed': seed_summary['initial_attempt_group_seed'],
        'final_attempt_group_seed': seed_summary['final_attempt_group_seed'],
        'attempt_group_seed_history': seed_summary['attempt_group_seed_history'],
        'reroll_group_seed_on_same_task': seed_summary['reroll_group_seed_on_same_task'],
        'actual_env_id': rollout['final_env_cfg'].get('env_id', rollout['initial_env_cfg'].get('env_id')),
        'unity_id': rollout['unity_id'],
        'start_attempt_index': rollout.get('start_attempt_index', 1),
        'loaded_history_attempt_count': rollout.get('loaded_history_attempt_count', 0),
        'trajectory_ref_resolved': _to_jsonable(case.get('trajectory_ref_resolved', None)),
        'turns': rollout['turns'],
        'max_attempts': rollout['max_attempts'],
        'max_steps_per_attempt': rollout['max_steps_per_attempt'],
        'rollout_step_budget': rollout['rollout_step_budget'],
        'rollout_budget_exhausted': rollout['rollout_budget_exhausted'],
        'attempt_count': attempt_summary['attempt_count'],
        'score_reward': attempt_summary['last_attempt_reward'],
        'last_attempt_index': attempt_summary['last_attempt_index'],
        'last_attempt_reward': attempt_summary['last_attempt_reward'],
        'best_attempt_reward': attempt_summary['best_attempt_reward'],
        'final_attempt_success': attempt_summary['final_attempt_success'],
        'ever_attempt_success': attempt_summary['ever_attempt_success'],
        'first_success_attempt_index': attempt_summary['first_success_attempt_index'],
        'success_attempt_count': attempt_summary['success_attempt_count'],
        'attempt_rewards': attempt_summary['attempts'],
        'done_all': bool(rollout['rollout_success'] or rollout['rollout_truncated']),
        'truncated_all': rollout['rollout_truncated'],
        'rollout_success': rollout['rollout_success'],
        'rollout_truncated': rollout['rollout_truncated'],
        'rollout_terminal': bool(rollout['rollout_success'] or rollout['rollout_truncated']),
        'total_reward': rollout['total_reward'],
        'reset_info': _to_jsonable(rollout['reset_info']),
        'initial_step_msg_preview': _truncate_text(rollout['initial_obs'].get('step_msg', ''), max_len=1000),
        'initial_obs_summary': summarize_obs(rollout['unity_id'], rollout['initial_obs']),
        'duration_s': rollout['duration_s'],
        'steps': rollout['step_records'],
    }


def build_error_result(model_runtime: Dict[str, Any], case: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    return {
        'status': 'error',
        'case_id': case['case_id'],
        'model_name': model_runtime['name'],
        'model': model_runtime['model'],
        'provider': model_runtime.get('provider'),
        'base_url': model_runtime['base_url'],
        'api_key_env': model_runtime['api_key_env'],
        'requested_task_name': case['task_name'],
        'requested_group_seed': int(case['group_seed']),
        'requested_env_id': case.get('env_id', None),
        'error_type': type(error).__name__,
        'error': str(error),
        'traceback': traceback.format_exc(limit=8),
    }


class JsonlWriter:
    def __init__(self, path: str, *, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not append:
            self.path.unlink()

    def write(self, record: Dict[str, Any]):
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + '\n')


def _stable_resume_value(value: Any) -> str:
    if value is None:
        return ''
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _eval_case_model_resume_key(case: Dict[str, Any], model_name: str) -> ResumeKey:
    return (
        str(case.get('task_name', '') or ''),
        _stable_resume_value(case.get('group_seed', '')),
        _stable_resume_value(case.get('env_id', '')),
        str(model_name or ''),
    )


def _eval_result_resume_key(record: Dict[str, Any]) -> ResumeKey | None:
    model_name = str(record.get('model_name', '') or '')
    task_name = str(record.get('requested_task_name', record.get('actual_task_name', '')) or '')
    group_seed = record.get('requested_group_seed', record.get('actual_group_seed', None))
    env_id = record.get('requested_env_id', record.get('actual_env_id', ''))
    if not model_name or not task_name or group_seed is None:
        return None
    return (
        task_name,
        _stable_resume_value(group_seed),
        _stable_resume_value(env_id),
        model_name,
    )


def _expected_eval_resume_keys(cases: List[Dict[str, Any]], model_runtimes: List[Dict[str, Any]]) -> set[ResumeKey]:
    return {
        _eval_case_model_resume_key(case, str(model_runtime.get('name', '') or ''))
        for case in cases
        for model_runtime in model_runtimes
    }


def load_existing_eval_results(
    output_path: str,
    cases: List[Dict[str, Any]],
    model_runtimes: List[Dict[str, Any]],
) -> Dict[ResumeKey, Dict[str, Any]]:
    path = Path(output_path)
    if not path.exists():
        return {}

    expected_keys = _expected_eval_resume_keys(cases, model_runtimes)
    existing: Dict[ResumeKey, Dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except Exception as exc:
                print(f'[run_api_agent] WARNING: ignoring malformed JSONL line {path}:{line_no}: {exc}')
                continue
            if not isinstance(record, dict) or record.get('status') != 'ok':
                continue
            key = _eval_result_resume_key(record)
            if key is None or key not in expected_keys:
                continue
            existing[key] = record
    return existing


def compact_eval_results_file(
    output_path: str,
    cases: List[Dict[str, Any]],
    model_runtimes: List[Dict[str, Any]],
) -> int:
    path = Path(output_path)
    if not path.exists():
        return 0

    expected_keys = _expected_eval_resume_keys(cases, model_runtimes)
    entries: List[Dict[str, Any]] = []
    selected_by_key: Dict[ResumeKey, int] = {}

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except Exception:
                entries.append({'kind': 'raw', 'line': line.rstrip('\n'), 'key': None})
                continue

            key = _eval_result_resume_key(record) if isinstance(record, dict) else None
            entry_index = len(entries)
            entries.append({'kind': 'record', 'record': record, 'key': key})
            if key is None or key not in expected_keys:
                continue

            prev_index = selected_by_key.get(key)
            if prev_index is None:
                selected_by_key[key] = entry_index
                continue

            prev_record = entries[prev_index].get('record', {})
            prev_ok = isinstance(prev_record, dict) and prev_record.get('status') == 'ok'
            cur_ok = record.get('status') == 'ok'
            if cur_ok or not prev_ok:
                selected_by_key[key] = entry_index

    selected_indices = set(selected_by_key.values())
    duplicate_count = 0
    output_lines: List[str] = []
    for index, entry in enumerate(entries):
        key = entry.get('key')
        if key is not None and key in expected_keys and index not in selected_indices:
            duplicate_count += 1
            continue
        if entry.get('kind') == 'raw':
            output_lines.append(str(entry.get('line', '')))
        else:
            output_lines.append(json.dumps(_to_jsonable(entry.get('record', {})), ensure_ascii=False))

    if duplicate_count > 0:
        with path.open('w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
    return duplicate_count


def write_eval_result(
    writer: JsonlWriter,
    result: Dict[str, Any],
    *,
    replace_existing_by_resume_key: bool = False,
) -> None:
    if not replace_existing_by_resume_key:
        writer.write(result)
        return

    key = _eval_result_resume_key(result)
    if key is None or not writer.path.exists():
        writer.write(result)
        return

    kept_records: List[Dict[str, Any]] = []
    with writer.path.open('r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                existing = json.loads(text)
            except Exception:
                continue
            if not isinstance(existing, dict):
                continue
            if _eval_result_resume_key(existing) == key:
                continue
            kept_records.append(existing)

    kept_records.append(result)
    with writer.path.open('w', encoding='utf-8') as f:
        for item in kept_records:
            f.write(json.dumps(_to_jsonable(item), ensure_ascii=False) + '\n')


def _trajectory_cfg(eval_cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    cfg = eval_cfg.get(key, {}) if isinstance(eval_cfg, dict) else {}
    return dict(cfg or {}) if isinstance(cfg, dict) else {}


def build_trajectory_writer(eval_cfg: Dict[str, Any]) -> TrajectoryJsonlWriter | None:
    save_cfg = _trajectory_cfg(eval_cfg, 'trajectory_save')
    if not _coerce_bool(save_cfg.get('enabled', False), default=False):
        return None
    output_path = str(save_cfg.get('output_path', '') or '').strip()
    if not output_path:
        raise ValueError('eval.trajectory_save.output_path is required when trajectory_save.enabled=true')
    append = _coerce_bool(save_cfg.get('append', False), default=False)
    return TrajectoryJsonlWriter(output_path, append=append)


def _trajectory_result_matches_condition(result: Dict[str, Any], save_cfg: Dict[str, Any]) -> bool:
    condition = str(save_cfg.get('condition', 'all') or 'all').strip().lower()
    threshold = float(save_cfg.get('reward_threshold', 0.0) or 0.0)
    if condition in ('all', 'always'):
        return True
    if condition in ('final_success', 'rollout_success'):
        return bool(result.get('rollout_success', False))
    if condition == 'ever_success':
        return bool(result.get('ever_attempt_success', result.get('rollout_success', False)))
    if condition in ('last_attempt_reward_gt', 'score_reward_gt'):
        return float(result.get('last_attempt_reward', result.get('score_reward', 0.0)) or 0.0) > threshold
    if condition == 'best_attempt_reward_gt':
        return float(result.get('best_attempt_reward', 0.0) or 0.0) > threshold
    raise ValueError(f'Unsupported eval.trajectory_save.condition={condition!r}')


def maybe_save_eval_trajectory(
    *,
    env: ArkEnv,
    result: Dict[str, Any],
    case: Dict[str, Any],
    model_runtime: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    writer: TrajectoryJsonlWriter | None,
) -> Dict[str, Any] | None:
    if writer is None:
        return None
    save_cfg = _trajectory_cfg(eval_cfg, 'trajectory_save')
    if not _trajectory_result_matches_condition(result, save_cfg):
        return None

    prefix_attempts = save_cfg.get('prefix_attempts', None)
    if prefix_attempts is not None:
        prefix_attempts = max(0, int(prefix_attempts))
    include_images = _coerce_bool(save_cfg.get('include_images', True), default=True)

    history_snapshot = env.export_finalized_attempts(prefix_attempts=prefix_attempts)
    record = build_eval_trajectory_record(
        result=result,
        case=case,
        model_runtime=model_runtime,
        history_snapshot=history_snapshot,
        prefix_attempts=prefix_attempts,
        include_images=include_images,
    )
    writer.write(record)
    result['trajectory'] = {
        'saved': True,
        'trajectory_id': record.get('trajectory_id'),
        'output_path': str(writer.path),
        'prefix_attempt_count': (record.get('prefix') or {}).get('attempt_count'),
        'target_attempt_index': (record.get('prefix') or {}).get('target_attempt_index'),
    }
    return record


def _case_trajectory_ref(case: Dict[str, Any], load_cfg: Dict[str, Any]) -> Dict[str, Any]:
    ref = case.get('trajectory_ref', None)
    out: Dict[str, Any] = dict(load_cfg or {})
    if isinstance(ref, dict):
        out.update(ref)
    elif isinstance(ref, str) and ref.strip():
        ref_text = ref.strip()
        if ref_text.endswith('.jsonl') or '/' in ref_text or '\\' in ref_text:
            out['path'] = ref_text
        else:
            out['trajectory_id'] = ref_text

    if case.get('trajectory_path', None) not in (None, ''):
        out['path'] = case.get('trajectory_path')
    if case.get('trajectory_id', None) not in (None, ''):
        out['trajectory_id'] = case.get('trajectory_id')
    if case.get('trajectory_index', None) is not None:
        out['index'] = case.get('trajectory_index')
    if case.get('trajectory_prefix_attempts', None) is not None:
        out['prefix_attempts'] = case.get('trajectory_prefix_attempts')
    return out


def apply_trajectory_load_to_cases(cases: List[Dict[str, Any]], eval_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    load_cfg = _trajectory_cfg(eval_cfg, 'trajectory_load')
    load_enabled = _coerce_bool(load_cfg.get('enabled', False), default=False)
    if not load_enabled and not any(
        any(key in case for key in ('trajectory_ref', 'trajectory_path', 'trajectory_id', 'history_snapshot'))
        for case in cases
    ):
        return cases

    out: List[Dict[str, Any]] = []
    for case in cases:
        new_case = deepcopy(case)
        if isinstance(new_case.get('history_snapshot'), dict):
            out.append(new_case)
            continue

        ref_cfg = _case_trajectory_ref(new_case, load_cfg)
        path = str(ref_cfg.get('path', ref_cfg.get('input_path', '')) or '').strip()
        if not path:
            if not load_enabled:
                out.append(new_case)
                continue
            raise ValueError('eval.trajectory_load.path is required when trajectory loading is enabled')

        case_id_filter = ref_cfg.get('case_id', None)
        if case_id_filter is None and ref_cfg.get('trajectory_id', None) is None and ref_cfg.get('index', None) is None:
            case_id_filter = new_case.get('case_id')

        record = load_trajectory_record(
            path,
            trajectory_id=ref_cfg.get('trajectory_id', None),
            case_id=case_id_filter,
            model_name=ref_cfg.get('model_name', None),
            index=ref_cfg.get('index', None),
        )
        prefix_attempts = ref_cfg.get('prefix_attempts', None)
        if prefix_attempts is not None:
            prefix_attempts = max(0, int(prefix_attempts))
        history_snapshot = history_snapshot_from_record(record, prefix_attempts=prefix_attempts)
        target_attempt_index = ref_cfg.get('start_attempt_index', None)
        if target_attempt_index is None:
            if prefix_attempts is not None:
                target_attempt_index = count_history_prefix_attempts(history_snapshot) + 1
            else:
                target_attempt_index = target_attempt_index_from_record(record, history_snapshot)
        new_case['history_snapshot'] = history_snapshot
        new_case['start_attempt_index'] = max(1, int(target_attempt_index))
        new_case['trajectory_ref_resolved'] = {
            'path': path,
            'trajectory_id': record.get('trajectory_id'),
            'prefix_attempt_count': count_history_prefix_attempts(history_snapshot),
            'target_attempt_index': new_case['start_attempt_index'],
        }
        out.append(new_case)
    return out


def print_summary(results: List[Dict[str, Any]]):
    grouped: Dict[str, Dict[str, Any]] = {}
    for result in results:
        model_name = str(result.get('model_name', 'unknown'))
        stats = grouped.setdefault(model_name, {
            'runs': 0,
            'errors': 0,
            'success_runs': 0,
            'truncated_runs': 0,
            'score_reward_sum': 0.0,
            'rollout_reward_sum': 0.0,
            'ever_success_runs': 0,
        })
        stats['runs'] += 1
        if result.get('status') != 'ok':
            stats['errors'] += 1
            continue
        if result.get('rollout_success'):
            stats['success_runs'] += 1
        if result.get('ever_attempt_success', result.get('rollout_success', False)):
            stats['ever_success_runs'] += 1
        if result.get('rollout_truncated'):
            stats['truncated_runs'] += 1
        stats['score_reward_sum'] += float(result.get('score_reward', result.get('total_reward', 0.0)))
        stats['rollout_reward_sum'] += float(result.get('total_reward', 0.0))

    print('\n=== Evaluation Summary ===')
    for model_name, stats in grouped.items():
        ok_runs = max(0, stats['runs'] - stats['errors'])
        avg_score_reward = (stats['score_reward_sum'] / ok_runs) if ok_runs > 0 else 0.0
        avg_rollout_reward = (stats['rollout_reward_sum'] / ok_runs) if ok_runs > 0 else 0.0
        print(
            f"model={model_name} runs={stats['runs']} ok={ok_runs} errors={stats['errors']} "
            f"final_success={stats['success_runs']} ever_success={stats['ever_success_runs']} "
            f"truncated={stats['truncated_runs']} "
            f"avg_score_reward={avg_score_reward:.4f} avg_rollout_reward={avg_rollout_reward:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate multiple API models on the same AgentArk initial states')
    parser.add_argument(
        '--config',
        type=str,
        default='config/ark_env/eval_seed1.example.yaml',
        help='Path to evaluation config (.yaml/.yml/.json)',
    )
    return parser


def main(args):
    cfg = load_eval_config(args.config)
    env_cfg = dict(cfg['env_cfg'])
    eval_cfg = dict(cfg['eval'])
    hooks_cfg = dict(cfg.get('hooks', {}) or {})

    ensure_runtime_pool_range(
        env_cfg,
        worker_index_base=int(env_cfg.get('worker_index', 0) or 0),
        worker_count=1,
    )

    env_cfg = resolve_runtime_sandbox_cfg(env_cfg)

    hook_manager, action_broker = build_eval_hook_manager(hooks_cfg)
    model_runtimes = build_model_runtimes(
        list(cfg.get('models', []) or []),
        hooks_cfg=hooks_cfg,
        hook_manager=hook_manager,
        action_broker=action_broker,
    )
    cases = apply_trajectory_load_to_cases(build_eval_cases(env_cfg, eval_cfg), eval_cfg)
    skip_existing_results = _coerce_bool(eval_cfg.get('skip_existing_results', False), default=False)
    compacted_existing_results = 0
    if skip_existing_results:
        compacted_existing_results = compact_eval_results_file(
            str(eval_cfg['output_path']),
            cases,
            model_runtimes,
        )
        existing_results_by_key = load_existing_eval_results(
            str(eval_cfg['output_path']),
            cases,
            model_runtimes,
        )
    else:
        existing_results_by_key = {}

    writer = JsonlWriter(str(eval_cfg['output_path']), append=skip_existing_results)
    trajectory_eval_cfg = deepcopy(eval_cfg)
    if skip_existing_results:
        trajectory_save_cfg = trajectory_eval_cfg.get('trajectory_save', {})
        if isinstance(trajectory_save_cfg, dict):
            trajectory_save_cfg = dict(trajectory_save_cfg)
            trajectory_save_cfg['append'] = True
            trajectory_eval_cfg['trajectory_save'] = trajectory_save_cfg
    trajectory_writer = build_trajectory_writer(trajectory_eval_cfg)

    print(f"Loaded {len(model_runtimes)} models and {len(cases)} evaluation cases")
    if skip_existing_results:
        expected_job_count = len(model_runtimes) * len(cases)
        print(
            f"Resuming from {writer.path}: skipped {len(existing_results_by_key)} existing ok results, "
            f"pending {expected_job_count - len(existing_results_by_key)} jobs"
        )
        if compacted_existing_results:
            print(f"Compacted {compacted_existing_results} superseded result records from {writer.path}")
    print(f"Results will be written to {writer.path}")
    if trajectory_writer is not None:
        print(f"Trajectory records will be written to {trajectory_writer.path}")

    results: List[Dict[str, Any]] = list(existing_results_by_key.values())
    stop_on_error = bool(eval_cfg.get('stop_on_error', False))

    hook_manager.start()
    hook_manager.emit(
        'run_start',
        {
            'config_path': args.config,
            'case_count': len(cases),
            'model_count': len(model_runtimes),
            'skipped_existing_job_count': len(existing_results_by_key),
            'compacted_existing_result_count': int(compacted_existing_results),
        },
        source='run_api_agent',
    )
    completed_main_loop = False
    keep_open_on_end = _hook_keep_open_on_end(hooks_cfg)
    try:
        for case in cases:
            print(
                f"\n[case {case['case_id']}] task={case['task_name']} "
                f"group_seed={case['group_seed']} env_id={case.get('env_id', None)}"
            )
            hook_manager.emit('case_start', case, source='run_api_agent')
            for model_runtime in model_runtimes:
                resume_key = _eval_case_model_resume_key(case, str(model_runtime.get('name', '') or ''))
                if skip_existing_results and resume_key in existing_results_by_key:
                    print(
                        f"[model {model_runtime['name']}] skip existing ok result "
                        f"case={case['case_id']} group_seed={case['group_seed']}"
                    )
                    continue

                print(
                    f"[model {model_runtime['name']}] start "
                    f"model={model_runtime['model']} provider={model_runtime.get('provider')} "
                    f"base_url={model_runtime['base_url']}"
                )
                hook_manager.emit(
                    'model_start',
                    {
                        'case_id': case['case_id'],
                        'model_name': model_runtime['name'],
                        'model': model_runtime['model'],
                        'provider': model_runtime.get('provider'),
                    },
                    source='run_api_agent',
                )
                runtime_env_cfg = deepcopy(env_cfg)
                runtime_env_cfg['hook_manager'] = hook_manager
                runtime_env_cfg['hooks'] = hooks_cfg
                env = ArkEnv(runtime_env_cfg)
                try:
                    result = evaluate_case(
                        env=env,
                        model_runtime=model_runtime,
                        case=case,
                        hook_manager=hook_manager,
                        hooks_cfg=hooks_cfg,
                    )
                    maybe_save_eval_trajectory(
                        env=env,
                        result=result,
                        case=case,
                        model_runtime=model_runtime,
                        eval_cfg=eval_cfg,
                        writer=trajectory_writer,
                    )
                except Exception as e:
                    result = build_error_result(model_runtime, case, e)
                    write_eval_result(
                        writer,
                        result,
                        replace_existing_by_resume_key=skip_existing_results,
                    )
                    results.append(result)
                    hook_manager.emit('error', result, source='run_api_agent')
                    print(f"[model {model_runtime['name']}] error: {type(e).__name__}: {e}")
                    if stop_on_error:
                        raise
                    continue
                finally:
                    env.close()

                write_eval_result(
                    writer,
                    result,
                    replace_existing_by_resume_key=skip_existing_results,
                )
                results.append(result)
                hook_manager.emit(
                    'model_end',
                    {
                        'case_id': case['case_id'],
                        'model_name': model_runtime['name'],
                        'turns': result['turns'],
                        'rollout_success': result['rollout_success'],
                        'rollout_truncated': result['rollout_truncated'],
                        'total_reward': result['total_reward'],
                    },
                    source='run_api_agent',
                )
                print(
                    f"[model {model_runtime['name']}] done turns={result['turns']} "
                    f"final_success={result['rollout_success']} "
                    f"ever_success={result.get('ever_attempt_success', result['rollout_success'])} "
                    f"truncated={result['rollout_truncated']} "
                    f"score_reward={result.get('score_reward', result['total_reward'])} "
                    f"total_reward={result['total_reward']}"
                )
        hook_manager.emit(
            'run_end',
            {
                'result_count': len(results),
                'new_result_count': len(results) - len(existing_results_by_key),
                'skipped_existing_job_count': len(existing_results_by_key),
                'compacted_existing_result_count': int(compacted_existing_results),
            },
            source='run_api_agent',
        )
        completed_main_loop = True
    finally:
        if not (completed_main_loop and keep_open_on_end and hook_manager.enabled):
            hook_manager.close()

    print_summary(results)
    if keep_open_on_end and hook_manager.enabled:
        try:
            print('\n[AgentArk viewer] keep_open_on_end=true; press Ctrl+C to stop the viewer.')
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print('\n[AgentArk viewer] stopping')
        finally:
            hook_manager.close()


if __name__ == '__main__':
    main(build_parser().parse_args())
