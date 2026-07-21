import uuid
import io
import re
import socket
import atexit
import shutil
import subprocess
import threading
import time
import warnings
from contextlib import nullcontext
try:
    from PIL import Image  # Pillow for PNG decoding
except ImportError:
    Image = None  # Graceful degradation if Pillow not installed
try:
    import yaml  # For YAML config management
except ImportError:
    yaml = None
from mlagents_envs.environment import UnityEnvironment
import os
import json
import random
import uuid
import importlib
import numpy as np
from typing import Any, Dict, List, Optional
from copy import deepcopy
from agent_ark.ark_env import SharedEpisodeStore, compute_history_bucket_key, get_history_cfg, get_history_retention_cfg
from .attempt_limits import normalize_max_steps_per_attempt, require_rollout_step_budget
from agent_ark.ark_env.context_manager import ContextManager

from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.raw_bytes_channel import RawBytesChannel
from mlagents_envs.exception import UnityCommunicationException, UnityTimeOutException, UnityWorkerInUseException

from agent_ark.utils.image_utils import _deal_frame_array
from agent_ark.side_channels.agent_raw_bytes_channel import AgentRawBytesChannel
from agent_ark.side_channels.image_frames_channel import ImageFramesChannel
from agent_ark.utils.image_utils import env_arr_to_pil_image
from agent_ark.utils.parse_utils import (
    build_llm_visible_prompt,
    extract_tag_content,
    parse_task_prompt_payload,
    render_tool_call_to_csharp,
)


def _patch_unity_environment_partial_init_close() -> None:
    """Avoid noisy ML-Agents atexit errors after early constructor failures."""
    original_close = getattr(UnityEnvironment, '_close', None)
    if original_close is None or getattr(original_close, '_agentark_partial_init_safe', False):
        return

    def _agentark_partial_init_safe_close(self, timeout=None):
        if getattr(self, '_communicator', None) is None:
            self._loaded = False
            return
        return original_close(self, timeout)

    _agentark_partial_init_safe_close._agentark_partial_init_safe = True
    UnityEnvironment._close = _agentark_partial_init_safe_close


_patch_unity_environment_partial_init_close()


_EDITOR_BASE_PORT_ENV = 'AGENTARK_EDITOR_BASE_PORT'
_PLAYER_BASE_PORT_ENV = 'AGENTARK_PLAYER_BASE_PORT'


def _is_unity_editor_env_path(env_path: Any) -> bool:
    """Return whether an env path represents an Editor connection."""
    if env_path is None:
        return True
    if isinstance(env_path, str):
        return env_path.strip().lower() in ('', 'none', 'null', '~')
    return False


def _coerce_base_port(value: Any, *, source: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{source} must be an integer Unity ML-Agents base port; got {value!r}') from exc


def _resolve_configured_unity_base_port(
    cfg: Optional[Dict[str, Any]],
    env_cfg: Optional[Dict[str, Any]],
    *,
    environ: Optional[Dict[str, str]] = None,
) -> int:
    """Resolve the ML-Agents base port for every AgentArk runtime entry point.

    ``cfg`` is the caller-owned Python mapping passed to ArkEnv/ArkSubEnv; it is
    not the Mods config and AgentArk does not add a default ``cfg['base_port']``.
    That key only exists when a caller explicitly supplies it (for example via
    CLI ``--base-port`` or evaluation ``env_cfg.base_port``).

    Explicit caller configuration wins. The two environment variables are
    optional local-development overrides: Editor connections read
    AGENTARK_EDITOR_BASE_PORT and packaged Players read
    AGENTARK_PLAYER_BASE_PORT. When they are unset, the legacy effective Mods
    config and default-port behavior is preserved, so existing users do not
    need to change their configuration.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    env_cfg = env_cfg if isinstance(env_cfg, dict) else {}
    wrapper_cfg = env_cfg.get('env_wrapper_cfg', {})
    wrapper_cfg = wrapper_cfg if isinstance(wrapper_cfg, dict) else {}
    overrides = cfg.get('env_config_overrides', {})
    overrides = overrides if isinstance(overrides, dict) else {}
    environ = os.environ if environ is None else environ

    # These are explicit caller overrides, not defaults loaded from Mods.
    for source, value in (
        ("cfg['base_port']", cfg.get('base_port')),
        ("cfg['env_config_overrides']['base_port']", overrides.get('base_port')),
    ):
        if value is not None and str(value).strip():
            return _coerce_base_port(value, source=source)

    # Opt-in worktree isolation only. Do not require these variables from normal
    # AgentArk users or from existing deployment/evaluation configurations.
    env_name = (
        _EDITOR_BASE_PORT_ENV
        if _is_unity_editor_env_path(cfg.get('env_path'))
        else _PLAYER_BASE_PORT_ENV
    )
    env_value = environ.get(env_name, '')
    if str(env_value).strip():
        return _coerce_base_port(env_value, source=env_name)

    for source, value in (
        ("env_config['base_port']", env_cfg.get('base_port')),
        ("env_config['env_wrapper_cfg']['base_port']", wrapper_cfg.get('base_port')),
    ):
        if value is not None and str(value).strip():
            return _coerce_base_port(value, source=source)

    return 5005


class _SharedXvfbManager:
    """Process-wide Xvfb manager for Linux headless rendering.

    A single Xvfb display is reused by all EnvWrapper instances in this process.
    """

    _lock = threading.Lock()
    _proc = None
    _display = None
    _cleanup_registered = False

    @classmethod
    def _register_cleanup(cls):
        if not cls._cleanup_registered:
            atexit.register(cls.stop)
            cls._cleanup_registered = True

    @classmethod
    def stop(cls):
        with cls._lock:
            proc = cls._proc
            cls._proc = None
            cls._display = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @classmethod
    def _apply_render_env(cls, config: Dict[str, Any]) -> None:
        """Apply software-rendering env vars that only matter under Xvfb on Linux.

        Mesa's llvmpipe software renderer spawns one worker thread per CPU core
        when Unity creates its GL context. On large-core hosts this thread count
        crashes Unity's PlayerMain (SIGSEGV). Pinning GALLIUM_NUM_THREADS keeps
        headless software rendering stable. Driven entirely by virtual_display
        config; an env var already set by the operator always wins.
        """
        for key, raw in (config.get('render_env') or {}).items():
            if raw is None:
                continue
            if not os.environ.get(key):
                os.environ[key] = str(raw)

    @classmethod
    def ensure_started(cls, config: Dict[str, Any]) -> str:
        # Windows/macOS are not supported by this Xvfb path.
        if os.name == 'nt' or not os.path.exists('/proc'):
            raise RuntimeError("virtual_display is enabled, but Xvfb auto-start is only supported on Linux")

        cls._apply_render_env(config)

        force_raw = config.get('force', False)
        if isinstance(force_raw, str):
            force = force_raw.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        else:
            force = bool(force_raw)

        with cls._lock:
            if cls._proc is not None and cls._proc.poll() is None and cls._display:
                os.environ['DISPLAY'] = cls._display
                return cls._display

        existing_display = os.environ.get('DISPLAY', '').strip()
        if existing_display and not force:
            return existing_display

        xvfb_bin = shutil.which('Xvfb')
        if not xvfb_bin:
            raise RuntimeError(
                "virtual_display=true but Xvfb is not installed or not in PATH. "
                "Please install xvfb (e.g., apt-get install -y xvfb)."
            )

        start_display = int(config.get('display_num', 99))
        max_tries = max(1, int(config.get('display_max_tries', 20)))
        width = max(1, int(config.get('width', 1024)))
        height = max(1, int(config.get('height', 768)))
        depth = max(1, int(config.get('color_depth', 24)))
        startup_wait_s = max(0.0, float(config.get('startup_wait_s', 0.2)))

        last_error = ""
        with cls._lock:
            if cls._proc is not None and cls._proc.poll() is None and cls._display:
                os.environ['DISPLAY'] = cls._display
                return cls._display

            for offset in range(max_tries):
                disp_num = start_display + offset
                display = f":{disp_num}"
                cmd = [
                    xvfb_bin,
                    display,
                    '-screen',
                    '0',
                    f"{width}x{height}x{depth}",
                    '-ac',
                    '+extension',
                    'GLX',
                    '+render',
                    '-noreset',
                ]
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                except Exception as e:
                    last_error = str(e)
                    continue

                time.sleep(startup_wait_s)
                if proc.poll() is None:
                    cls._proc = proc
                    cls._display = display
                    cls._register_cleanup()
                    os.environ['DISPLAY'] = display
                    return display

                # Child already exited: always reap it to avoid zombie processes.
                try:
                    proc.wait(timeout=0.2)
                except Exception:
                    pass

                try:
                    err_text = proc.stderr.read().decode('utf-8', errors='ignore') if proc.stderr else ''
                except Exception:
                    err_text = ''
                last_error = err_text.strip() or f"Xvfb exited with code {proc.returncode}"

        raise RuntimeError(
            f"Failed to start Xvfb virtual display. start_display=:{start_display}, "
            f"tries={max_tries}, last_error={last_error}"
        )


def _resolve_runtime_cfg(cfg):
    resolver = importlib.import_module('agent_ark.ark_env.runtime_sandbox').resolve_runtime_sandbox_cfg
    return resolver(cfg)


class EnvInfoManager(object):
    '''负责更改（在env启动前），持有环境信息
    '''
    def __init__(self, cfg):
        self.cfg = _resolve_runtime_cfg(cfg)

        self.mod_path = self.cfg['mod_path']
        self.task_store_path = self.resolve_task_store_path(self.mod_path)
        self.task_config = {}

        # Track last selected task and group seed across resets.
        # Used to decide whether to keep or reroll group_seed when task repeats.
        self._last_task_name = None
        self._last_group_seed = None

    # def get_task_prompt(self):
    #     if self.cfg['task_type'] == 'RLTask':
    #         return self.task_config['task_prompt']
    #     return self.cfg['task_prompt']

    @staticmethod
    def resolve_task_store_path(mod_path):
        return os.path.join(str(mod_path), 'all_tasks')

    def reset(self, task_name=None, group_seed=None, env_id=None):
        self.task_store_path = self.resolve_task_store_path(self.mod_path)
        base_env_config = self._read_base_env_config()
        if not self._uses_task_store(base_env_config):
            self._set_prefab_task(base_env_config, task_name=task_name, group_seed=group_seed, env_id=env_id)
            self.system_prompt = self._get_system_prompt(action_mode=self.env_config.get('action_mode', None))
            return

        self.task_list = self.get_task_list(self.mod_path)
        self._set_task(task_name=task_name, group_seed=group_seed, env_id=env_id)

        # Select system prompt after env_config is available.
        # Priority: env_config.action_mode > cfg.action_mode > default('code')
        action_mode = None
        if isinstance(getattr(self, 'env_config', None), dict):
            action_mode = self.env_config.get('action_mode', None)
        if action_mode is None and isinstance(getattr(self, 'cfg', None), dict):
            action_mode = self.cfg.get('action_mode', None)
        self.system_prompt = self._get_system_prompt(action_mode=action_mode)

    def _read_base_env_config(self):
        env_config = self._read_yaml(self.mod_path, 'config.yaml')
        if env_config is None:
            env_config = self._read_config(self.mod_path, 'config.json') or {}
        return env_config if isinstance(env_config, dict) else {}

    def _read_clean_base_env_config(self, active_env_config=None):
        """Load the immutable root template used to materialize one task.

        config.yaml is intentionally human-readable effective runtime state.
        Reusing it as the next task's merge base would carry task-local values
        forward.  config.yaml.bak is the clean root template; only operator and
        mode/selection fields are preserved from the active runtime config.
        """
        clean = self._read_yaml(self.mod_path, 'config.yaml.bak')
        if not isinstance(clean, dict):
            warnings.warn(
                f"config.yaml.bak is missing or invalid under mod_path={self.mod_path}; "
                "falling back to the mutable active root config, so cross-task "
                "config isolation cannot be guaranteed.",
                RuntimeWarning,
                stacklevel=2,
            )
            if isinstance(active_env_config, dict):
                return deepcopy(active_env_config)
            return self._read_base_env_config()

        active = active_env_config if isinstance(active_env_config, dict) else self._read_base_env_config()
        # These values describe the current runtime/operator rather than a task.
        # Keep them when resetting from the shared template so Editor/package
        # port, graphics, and startup policy are not accidentally replaced.
        preserve_keys = (
            'override_by_task',
            'load_mod_mode',
            'task_name',
            'no_graphics',
            'unity_start_timeout_s',
            'unity_start_max_attempts',
            'unity_start_retry_wait_s',
            'serialize_unity_startup',
            'base_port',
            'auto_port_scan',
            'virtual_display',
            'virtual_display_force',
            'virtual_display_num',
            'virtual_display_max_tries',
            'virtual_display_width',
            'virtual_display_height',
            'virtual_display_color_depth',
            'virtual_display_startup_wait_s',
            'virtual_display_render_env',
        )
        for key in preserve_keys:
            if key in active:
                clean[key] = deepcopy(active[key])
        return clean

    @staticmethod
    def _uses_task_store(env_config):
        if not isinstance(env_config, dict):
            return True
        raw = env_config.get('override_by_task', True)
        if isinstance(raw, str):
            return raw.strip().lower() not in ('0', 'false', 'no', 'n', 'off')
        return bool(raw)

    def _set_prefab_task(self, env_config, task_name=None, group_seed=None, env_id=None):
        selected_task_name = str(task_name or env_config.get('task_name') or 'prefab').strip() or 'prefab'
        self.task_list = []
        self.now_task_info = {'folder_name': selected_task_name, 'json_name': ''}
        self.env_config = self._read_clean_base_env_config(env_config)

        self.task_cfg_path, self.task_config = self._find_prefab_task_config(selected_task_name)
        if isinstance(self.task_config, dict) and self.task_config:
            # Prefab mode still runs the selected task's authored observation,
            # timing, resolution, action, and error contract.  Start from the
            # stable root YAML, overlay the local task config, then force only
            # the prefab-loading fields below.  This avoids inheriting the
            # previous task's effective top-level values from config.json.
            self.env_config = self._overlay_env_with_task_config(self.env_config, self.task_config)
        else:
            self.env_config.pop('task_params', None)
        self.env_config = self._merge_python_only_task_config(self.env_config, self.task_config)

        env_cfg_overrides = self.cfg.get('env_config_overrides', None) if isinstance(self.cfg, dict) else None
        if isinstance(env_cfg_overrides, dict) and env_cfg_overrides:
            self.env_config = self._apply_env_config_overrides(self.env_config, env_cfg_overrides)

        self.env_config = normalize_max_steps_per_attempt(self.env_config)
        self.env_config['override_by_task'] = False
        self.env_config['load_mod_mode'] = 'none'
        self.env_config['task_name'] = selected_task_name

        if 'engine_para' not in self.env_config or not isinstance(self.env_config.get('engine_para'), dict):
            self.env_config['engine_para'] = {}
        if 'group_same_init' not in self.env_config:
            self.env_config['group_same_init'] = False
        if 'env_id' not in self.env_config:
            self.env_config['env_id'] = 0
        if env_id is not None:
            self.env_config['env_id'] = int(env_id)

        self.env_config = self._require_rollout_budget_config(
            self.env_config,
            context=f"prefab task {selected_task_name}",
        )

        if group_seed is not None:
            self._last_group_seed = int(group_seed)
        elif self._last_group_seed is None or self._last_task_name != selected_task_name:
            self._last_group_seed = int(random.randint(1, 2**31 - 2))
        self.env_config['group_seed'] = int(self._last_group_seed)
        self._last_task_name = selected_task_name

        # Keep a human-readable effective YAML beside the JSON that Unity reads.
        # The next reset starts from config.yaml.bak, never from this output.
        if not self._save_yaml(self.env_config, self.mod_path, 'config.yaml'):
            raise OSError(f"Failed to write effective config.yaml under mod_path={self.mod_path}")
        if not self._save_config(self.env_config, self.mod_path, 'config.json'):
            raise OSError(f"Failed to write runtime config.json under mod_path={self.mod_path}")

    @staticmethod
    def get_task_list(mod_path):
        """
        遍历 Mods/all_tasks 下的所有一级子文件夹，检查每个子文件夹中的 JSON 文件数量。

        如果任何子文件夹包含超过1个.json后缀的文件，抛出异常
        否则，返回所有JSON文件的名称列表，格式为['子文件夹名/文件名.json', ...]

        返回:
            list: 所有JSON文件的名称列表，包含子文件夹路径

        异常:
            ValueError: 当任何子文件夹包含超过1个JSON文件时抛出
        """
        task_path = EnvInfoManager.resolve_task_store_path(mod_path)
        if not os.path.exists(task_path):
            raise FileNotFoundError(
                f"Task store not found under Mods/all_tasks. mod_path={mod_path}, task_store_path={task_path}"
            )
        if not os.path.isdir(task_path):
            raise NotADirectoryError(
                f"Task store path is not a directory. mod_path={mod_path}, task_store_path={task_path}"
            )

        # 获取当前路径下的所有一级子文件夹。
        # 排序保证 task_list 顺序在任何进程/机器上稳定一致，这是上层基于
        # 索引（如 hash(uid) % len）做确定性 task 选择的正确性前提。
        subfolders = sorted(
            f for f in os.listdir(task_path)
            if os.path.isdir(os.path.join(task_path, f))
        )

        if not subfolders:
            raise ValueError(f"Task store is empty: {task_path}")

        all_json_names = []

        # 遍历每个子文件夹
        for folder in subfolders:
            folder_path = os.path.join(task_path, folder)

            # 获取文件夹中所有.json后缀的文件
            json_files = [
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.json')
            ]

            # 检查JSON文件数量
            if len(json_files) > 1:
                raise ValueError(f"子文件夹 '{folder}' 包含超过一个JSON文件，共发现 {len(json_files)} 个")

            task_cfg_path = os.path.join(folder_path, 'cfg')
            task_cfg = EnvInfoManager._read_yaml(task_cfg_path, 'task_config.yaml')
            if task_cfg is None:
                task_cfg = EnvInfoManager._read_config(task_cfg_path, 'task_config.json') or {}
            task_info = task_cfg.get('task_info', {}) if isinstance(task_cfg, dict) else {}
            if not isinstance(task_info, dict):
                task_info = {}
            legacy_names = task_info.get('legacy_names', [])
            if isinstance(legacy_names, str):
                legacy_names = [legacy_names]
            elif not isinstance(legacy_names, list):
                legacy_names = []
            aliases = []
            public_name = str(task_info.get('name') or '').strip()
            if public_name:
                aliases.append(public_name)
            aliases.extend(str(item).strip() for item in legacy_names if str(item).strip())
            config_task_name = str(task_cfg.get('task_name') or '').strip() if isinstance(task_cfg, dict) else ''
            if config_task_name:
                aliases.append(config_task_name)
            aliases = list(dict.fromkeys(alias for alias in aliases if alias and alias != folder))

            # 如果有JSON文件，添加到结果列表
            for json_file in json_files:

                ## 文件夹和文件名可以不同。文件名为addressables打包时的名字，而文件夹名可随意更改
                # all_json_names.append(f"{folder}/{json_file}")
                all_json_names.append({
                    'folder_name': folder,
                    'json_name': json_file,
                    'task_info': task_info,
                    'aliases': aliases,
                })

        if not all_json_names:
            raise ValueError(
                f"No task metadata JSON files were found under task store: {task_path}"
            )

        return all_json_names

    def _resolve_task_info(self, task_name=None):
        if task_name is None:
            return random.choice(self.task_list)

        normalized = str(task_name).strip()
        if not normalized:
            raise ValueError('task_name cannot be empty')

        by_folder = [info for info in self.task_list if info.get('folder_name') == normalized]
        if len(by_folder) == 1:
            return by_folder[0]
        if len(by_folder) > 1:
            raise ValueError(f"task_name is ambiguous by folder_name: {normalized}")

        by_full_name = [
            info for info in self.task_list
            if f"{info.get('folder_name')}/{info.get('json_name')}" == normalized
        ]
        if len(by_full_name) == 1:
            return by_full_name[0]
        if len(by_full_name) > 1:
            raise ValueError(f"task_name is ambiguous by full task path: {normalized}")

        by_json_name = [info for info in self.task_list if info.get('json_name') == normalized]
        if len(by_json_name) == 1:
            return by_json_name[0]
        if len(by_json_name) > 1:
            raise ValueError(
                f"task_name={normalized!r} matches multiple json_name entries; use folder_name or folder/json_name"
            )

        by_alias = [
            info for info in self.task_list
            if normalized in (info.get('aliases') or [])
        ]
        if len(by_alias) == 1:
            return by_alias[0]
        if len(by_alias) > 1:
            raise ValueError(f"task_name={normalized!r} matches multiple task aliases; use folder_name")

        available_names = []
        for info in self.task_list:
            available_names.append(info.get('folder_name', ''))
            available_names.extend(info.get('aliases') or [])
        available = ', '.join(sorted(set(name for name in available_names if name)))
        raise ValueError(f"Unknown task_name={normalized!r}. Available task names: {available}")

    def _set_task(self, task_name=None, group_seed=None, env_id=None):
        '''config.yaml 中 load_mod_mode 设置为随机选择的 task
        - Base config now uses YAML (config.yaml). Falls back to legacy JSON if YAML missing.
        - Per-task metadata file inside task root remains JSON (unchanged).
        - task_config in cfg/ now uses YAML (task_config.yaml) with JSON fallback.
        '''
        # Python 侧以 Mods/config.yaml 为主（缺失时退回 config.json），在此合并 task_config；Unity 运行时最终只消费 Mods 根目录的 config.json。
        self.now_task_info = self._resolve_task_info(task_name=task_name)

        # Selected task name is the mod folder name.
        # Unity ModManager uses folder-backed mod names, and this value is
        # written back to config.json for runtime loading.
        selected_task_name = self.now_task_info['folder_name']

        # Read the active config for operator/mode fields, but always materialize
        # the selected task from the immutable root template.  The effective
        # config.yaml left by the previous task is output, not the next input.
        active_env_config = self._read_base_env_config()
        self.env_config = self._read_clean_base_env_config(active_env_config)

        # Read per-task cfg task_config (YAML preferred) so Python-side history config
        # can be picked up even when it only exists in task_config.
        self.task_cfg_path = os.path.join(self.task_store_path, self.now_task_info['folder_name'], 'cfg')
        self.task_config = self._read_yaml(self.task_cfg_path, 'task_config.yaml')
        if self.task_config is None:
            self.task_config = self._read_config(self.task_cfg_path, 'task_config.json') or {}

        if self.env_config.get('override_by_task', True):
            # Overlay only overlapping keys (ignore task-only extras)
            self.env_config = self._overlay_env_with_task_config(self.env_config, self.task_config)

        # Python-only wrapper settings such as context/history are allowed to be
        # task-specific without requiring placeholders in the base env config.
        self.env_config = self._merge_python_only_task_config(self.env_config, self.task_config)

        env_cfg_overrides = self.cfg.get('env_config_overrides', None) if isinstance(self.cfg, dict) else None
        if isinstance(env_cfg_overrides, dict) and env_cfg_overrides:
            self.env_config = self._apply_env_config_overrides(self.env_config, env_cfg_overrides)

        self.env_config = normalize_max_steps_per_attempt(self.env_config)

        self.env_config = self._require_rollout_budget_config(
            self.env_config,
            context=f"task {selected_task_name}",
        )

        # Always update runtime selection fields after overlay so they're authoritative.
        # (Even when override_by_task is false, we still want reset-time selection to take effect.)
        self.env_config['load_mod_mode'] = 'task_name'
        self.env_config['task_name'] = selected_task_name

        # Ensure deterministic RNG-related keys exist
        # - group_same_init: if true, Unity forces all envId=env_id (usually 0)
        # - env_id: base env id (offset) or fixed id when group_same_init
        # - group_seed: group seed for StatelessRng (shared across the group)
        if 'group_same_init' not in self.env_config:
            self.env_config['group_same_init'] = False
        if 'env_id' not in self.env_config:
            self.env_config['env_id'] = 0
        if env_id is not None:
            self.env_config['env_id'] = int(env_id)

        # Decide group_seed policy.
        # Rule:
        # - If task changes vs last reset => always reroll group_seed.
        # - If task is same => keep or reroll depending on reroll_group_seed_on_same_task.
        if group_seed is not None:
            self._last_group_seed = int(group_seed)
        else:
            same_task_as_last = (self._last_task_name == selected_task_name)
            reroll_on_same = bool(self.env_config.get('reroll_group_seed_on_same_task', False))

            need_reroll = (not same_task_as_last) or reroll_on_same or (self._last_group_seed is None)
            if need_reroll:
                # Use 31-bit positive int to stay safe for Unity int and JSON.
                self._last_group_seed = int(random.randint(1, 2**31 - 2))
        self.env_config['group_seed'] = int(self._last_group_seed)
        self._last_task_name = selected_task_name

        # Write matching human-readable YAML and Unity-consumed JSON.  Both are
        # effective outputs; config.yaml.bak remains the clean next-run input.
        if self.env_config.get('override_by_task', False):
            if not self._save_yaml(self.env_config, self.mod_path, 'config.yaml'):
                raise OSError(f"Failed to write effective config.yaml under mod_path={self.mod_path}")
            if not self._save_config(self.env_config, self.mod_path, 'config.json'):
                raise OSError(f"Failed to write runtime config.json under mod_path={self.mod_path}")

            # 注：task 专用参数（只在 task_config 中存在，而不在全局 config.yaml 中声明的键）
            # 目前不会写回到 env_config，而是单独通过 raw_bytes 传给 Unity。
            # 这样 RLAgentManager / RLAgent 仍然只关心全局参数，不需要感知每个 task 的细节。

    def random_set_task(self):
        self._set_task()

    def set_env_para(self, env_channel):
        can_set_paras = ['width', 'height', 'num_parallel_envs']
        for key in can_set_paras:
            if key not in self.env_config:
                continue
            val = self.env_config[key]
            env_channel.set_float_parameter(key, float(val))

    def set_engine_para(self, engine_channel):
        # 图像size在此处设置不会生效，因为unity侧load task时才临时生成cam_sersors并设置size，mlagent-envs
        # 的设置就不会生效。 放到set_env_para里，并unity的RLAgentManager中读取env参数设置
        engine_keys = ['quality_level', 'time_scale', 'target_frame_rate', 'capture_frame_rate']
        engine_para = self.env_config.get('engine_para', {})
        if not isinstance(engine_para, dict):
            engine_para = {}
        set_paras = {}
        for engine_key in engine_keys:
            if engine_key in engine_para:
                set_paras[engine_key] = engine_para[engine_key]
        engine_channel.set_configuration_parameters(**set_paras)

    @staticmethod
    def _read_config(config_path, config_name):
        """
        检查指定路径下是否存在config_name文件并读取其内容

        参数:
            config_path: 字符串，指定要检查的路径

        返回:
            如果文件存在且读取成功，返回解析后的JSON数据（字典或列表）
            如果文件不存在或读取失败，返回None
        """
        # 拼接完整的文件路径
        file_path = os.path.join(config_path, config_name)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            return None

        # 检查是否是文件（不是目录）
        if not os.path.isfile(file_path):
            return None

        try:
            # 读取并解析JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return config_data
        except json.JSONDecodeError:
            # JSON格式错误
            return None
        except Exception as e:
            # 其他可能的错误（如权限问题等）
            print(f"读取config.json时发生错误: {e}")
            return None

    @staticmethod
    def _read_yaml(config_path, config_name):
        """读取 YAML 配置 (返回 dict 或 None)"""
        file_path = os.path.join(config_path, config_name)
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return None
        if yaml is None:
            print('PyYAML 未安装，无法读取 YAML 文件。')
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return data
        except Exception as e:
            print(f"读取 {config_name} 时发生错误: {e}")
            return None

    @staticmethod
    def _get_prefab_task_search_root(mod_path):
        env_root = os.environ.get('AGENTARK_PREFAB_TASK_ROOT')
        if env_root:
            env_root = os.path.abspath(os.path.expanduser(os.path.expandvars(env_root)))
            if os.path.isdir(env_root):
                return env_root

        current = os.path.abspath(str(mod_path)) if mod_path is not None else ''
        while current:
            if os.path.basename(current) == 'Assets':
                candidates = [
                    os.path.join(current, 'AgentArk', 'Tasks'),
                    os.path.join(current, 'AgentArk', 'RLTasks'),
                    current,
                ]
                for task_root in candidates:
                    if os.path.isdir(task_root):
                        return task_root

            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

        return None

    def _find_prefab_task_config(self, selected_task_name):
        search_root = self._get_prefab_task_search_root(self.mod_path)
        if not search_root:
            return None, {}

        matches = []
        for dirpath, _, filenames in os.walk(search_root):
            task_cfg = None
            if 'task_config.yaml' in filenames:
                task_cfg = self._read_yaml(dirpath, 'task_config.yaml')
            elif 'task_config.json' in filenames:
                task_cfg = self._read_config(dirpath, 'task_config.json') or {}

            if not isinstance(task_cfg, dict) or not task_cfg:
                continue

            config_task_name = str(task_cfg.get('task_name') or '').strip()
            folder_name = os.path.basename(dirpath)
            task_info = task_cfg.get('task_info', {}) if isinstance(task_cfg, dict) else {}
            if not isinstance(task_info, dict):
                task_info = {}
            legacy_names = task_info.get('legacy_names', [])
            if isinstance(legacy_names, str):
                legacy_names = [legacy_names]
            elif not isinstance(legacy_names, list):
                legacy_names = []
            task_names = {
                folder_name,
                config_task_name,
                str(task_info.get('name') or '').strip(),
                *(str(item).strip() for item in legacy_names),
            }
            task_names.discard('')
            if selected_task_name not in task_names:
                continue

            normalized_dirpath = dirpath.replace('\\', '/')
            matches.append((
                config_task_name != selected_task_name,
                '/ValidationMods/' in normalized_dirpath,
                len(normalized_dirpath),
                dirpath,
                task_cfg,
            ))

        if not matches:
            return None, {}

        matches.sort()
        _, _, _, task_cfg_path, task_cfg = matches[0]
        return task_cfg_path, task_cfg

    def _get_system_prompt(self, sys_prompt_path: str = '', action_mode=None):
        """Load system prompt text.

        action_mode:
          - 'code' (default): expects LLM to output a full Unity C# script.
          - 'func': expects LLM to output only function/argument fills (template provided by task).
        """
        if len(sys_prompt_path) != 0:
            raise NotImplementedError('暂不支持自定义system_prompt_path')

        default_root = os.path.dirname(os.path.abspath(__file__))
        mode = (action_mode or 'code').strip().lower()
        if mode == 'func':
            sys_prompt_path = os.path.join(default_root, 'info/system_prompt_func.txt')
        else:
            # Backward-compatible default
            sys_prompt_path = os.path.join(default_root, 'info/system_prompt.txt')

        with open(sys_prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def _save_config(data, save_path, file_name):
        """
        将JSON数据保存到指定路径和文件名

        参数:
            data: 要保存的JSON数据（字典或列表）
            save_path: 保存文件的路径
            file_name: 保存的文件名（如"config.json"）

        返回:
            布尔值，保存成功返回True，失败返回False
        """
        try:
            # 确保保存路径存在，如果不存在则创建
            os.makedirs(save_path, exist_ok=True)

            # 拼接完整的文件路径
            file_path = os.path.join(save_path, file_name)

            # 保存JSON数据，使用ensure_ascii=False保证中文正常显示
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            return True
        except TypeError:
            print("错误：数据类型不是JSON可序列化的")
            return False
        except Exception as e:
            print(f"保存文件时发生错误: {e}")
            return False

    @staticmethod
    def _save_yaml(data, save_path, file_name):
        """保存字典为 YAML 文件"""
        if yaml is None:
            print('PyYAML 未安装，无法保存 YAML 文件。')
            return False
        try:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            print(f"保存 YAML 文件时发生错误: {e}")
            return False

    @staticmethod
    def _overlay_env_with_task_config(env_cfg: dict, task_cfg: dict) -> dict:
        """
        将 task_cfg 中的值覆盖到 env_cfg，但仅覆盖 env_cfg 已存在的键；
        对于嵌套字典，执行递归覆盖。返回新的字典副本。

        task_params 是 task 专属参数，不参与递归合并：
        - task_config 中存在 task_params 时，整块替换全局 task_params；
        - task_config 中不存在 task_params 时，删除全局 task_params，避免跨 task 残留。
        """
        if not isinstance(env_cfg, dict) or not isinstance(task_cfg, dict):
            return env_cfg

        result = deepcopy(env_cfg)

        if 'task_params' in task_cfg:
            result['task_params'] = deepcopy(task_cfg['task_params'])
        else:
            result.pop('task_params', None)

        def merge(base, override):
            for k, v in override.items():
                if k == 'task_params':
                    continue
                if k in base:
                    if isinstance(base[k], dict) and isinstance(v, dict):
                        merge(base[k], v)
                    else:
                        base[k] = deepcopy(v)
                # 不在 base 的键忽略（根据需求暂不处理任务特有参数）

        merge(result, task_cfg)
        return result

    @staticmethod
    def _apply_env_config_overrides(env_cfg: dict, overrides: dict) -> dict:
        """Apply user-supplied env overrides after task overlay.

        Unlike task_config overlay, this merge allows new keys so local evaluation
        runners can enforce settings such as num_parallel_envs=1.
        """
        if not isinstance(env_cfg, dict) or not isinstance(overrides, dict):
            return env_cfg

        result = deepcopy(env_cfg)

        def merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    merge(dst[k], v)
                else:
                    dst[k] = deepcopy(v)

        merge(result, overrides)
        return result

    @staticmethod
    def _merge_python_only_task_config(env_cfg: dict, task_cfg: dict) -> dict:
        if not isinstance(env_cfg, dict) or not isinstance(task_cfg, dict):
            return env_cfg

        result = deepcopy(env_cfg)
        for key in ('max_attempts', 'max_steps_per_attempt'):
            if key in task_cfg:
                result[key] = deepcopy(task_cfg[key])

        task_wrapper_cfg = task_cfg.get('env_wrapper_cfg', {})
        wrapper_cfg = result.get('env_wrapper_cfg', {})
        if not isinstance(wrapper_cfg, dict):
            wrapper_cfg = {}
        result['env_wrapper_cfg'] = wrapper_cfg

        # initial_observation is task-scoped: omission should disable it rather than
        # inherit a warmup block left behind by a previously selected task.
        if isinstance(task_wrapper_cfg, dict) and 'initial_observation' in task_wrapper_cfg:
            wrapper_cfg['initial_observation'] = deepcopy(task_wrapper_cfg['initial_observation'])
        else:
            wrapper_cfg.pop('initial_observation', None)

        if not isinstance(task_wrapper_cfg, dict) or not task_wrapper_cfg:
            return result

        def merge(dst, src):
            for key, value in src.items():
                if key == 'initial_observation':
                    continue
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    merge(dst[key], value)
                else:
                    dst[key] = deepcopy(value)

        merge(wrapper_cfg, task_wrapper_cfg)
        return result

    @staticmethod
    def _require_rollout_budget_config(env_cfg: dict, *, context: str) -> dict:
        result = deepcopy(env_cfg) if isinstance(env_cfg, dict) else {}
        max_attempts, max_steps_per_attempt, _ = require_rollout_step_budget(
            max_attempts=result.get('max_attempts', None),
            max_steps_per_attempt=result.get('max_steps_per_attempt', None),
            context=f"EnvInfoManager {context}",
        )
        result['max_attempts'] = max_attempts
        result['max_steps_per_attempt'] = max_steps_per_attempt
        # Unity currently truncates attempts via envParams.max_steps, so derive it
        # from the shared per-attempt budget instead of requiring task authors to
        # duplicate the value in task configs.
        result['max_steps'] = max_steps_per_attempt
        return result


class EnvWrapper(object):
    _unity_start_lock = threading.Lock()
    _resolve_runtime_cfg = staticmethod(_resolve_runtime_cfg)

    def __init__(self, cfg):
        self.cfg = self._resolve_runtime_cfg(cfg)

        self.env_info_mgr = EnvInfoManager(self.cfg)
        self.env_info_mgr.reset()

        self.env = None
        self._last_env_resolution = None
        self._code_wrapper_by_unity_id = {}
        self._tool_manifest_by_unity_id = {}
        self.context_mgr = ContextManager()
        self.local_history_store = SharedEpisodeStore()
        self._last_reset_plan: Dict[str, Any] = {}
        self._active_history_bucket_key: Optional[str] = None

    @staticmethod
    def _coerce_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ('1', 'true', 'yes', 'y', 'on'):
                return True
            if v in ('0', 'false', 'no', 'n', 'off'):
                return False
        return default

    def _get_action_mode(self):
        env_cfg = getattr(self.env_info_mgr, 'env_config', None) or {}
        mode = env_cfg.get('action_mode', None)
        if mode is None and isinstance(self.cfg, dict):
            mode = self.cfg.get('action_mode', None)
        return (mode or 'code').strip().lower()

    def _prepare_reset_plan(
        self,
        *,
        agent_ids: Optional[List[int]] = None,
        history_snapshot: Optional[Dict[int, list]] = None,
    ) -> Dict[str, Any]:
        env_cfg = self.env_info_mgr.env_config or {}
        num_parallel_envs = int(env_cfg.get('num_parallel_envs', 1) or 1)
        task_name = env_cfg.get('task_name', None)
        group_seed = env_cfg.get('group_seed', None)
        resolved_agent_ids = [int(agent_id) for agent_id in agent_ids] if agent_ids else list(range(max(1, num_parallel_envs)))
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
                resolved_agent_ids,
                history_cfg,
                retention_cfg=retention_cfg,
            ),
        }
        if isinstance(history_snapshot, dict):
            plan['history_snapshot'] = deepcopy(history_snapshot)

        self._last_reset_plan = deepcopy(plan)
        self._active_history_bucket_key = plan.get('history_bucket_key', None)
        return plan

    def _publish_finalized_history(self):
        history_cfg = get_history_cfg(self.env_info_mgr.env_config or {})
        retention_cfg = get_history_retention_cfg(self.env_info_mgr.env_config or {})
        finalized_episodes = self.context_mgr.take_finalized_episodes()
        if not finalized_episodes:
            return
        for agent_id, episodes in finalized_episodes.items():
            if not isinstance(episodes, list):
                continue
            for episode in episodes:
                self.local_history_store.publish_episode(
                    self._active_history_bucket_key,
                    int(agent_id),
                    episode,
                    history_cfg,
                    retention_cfg=retention_cfg,
                )

    def _refresh_code_wrappers_from_task_prompt(self):
        """Cache per-unity-id code wrappers and tool manifests from reset payloads."""
        self._code_wrapper_by_unity_id = {}
        self._tool_manifest_by_unity_id = {}
        tp = getattr(self.env_info_mgr, 'task_prompt', None)
        if not isinstance(tp, dict):
            return
        for unity_id, text in tp.items():
            payload = parse_task_prompt_payload(text, prefer_english_lang=self._preferred_language_tag() != 'chinese_ver')
            if payload.get('code_wrapper'):
                self._code_wrapper_by_unity_id[unity_id] = payload['code_wrapper']
            if payload.get('tool_manifest'):
                self._tool_manifest_by_unity_id[unity_id] = payload['tool_manifest']

    @staticmethod
    def _to_csharp_literal(val):
        if val is None:
            return 'null'
        if isinstance(val, bool):
            return 'true' if val else 'false'
        if isinstance(val, (int, float)):
            return ("%s" % val)
        # Strings: emit a valid C# string literal so the model can output plain text (e.g. "R8").
        # Using JSON escaping here yields a C#-compatible escaped string (\n, \", \\ , \uXXXX).
        if isinstance(val, str):
            return json.dumps(val, ensure_ascii=False)

        # Fallback: stringify as a C# token
        return str(val)

    def _render_func_wrapper(self, unity_id: int, params_text: str) -> str:
        raw = (params_text or '').strip()
        if extract_tag_content(raw, tag='tool_call'):
            manifest = self._tool_manifest_by_unity_id.get(unity_id)
            return render_tool_call_to_csharp(raw, manifest)

        wrapper = self._code_wrapper_by_unity_id.get(unity_id)
        if not wrapper:
            raise ValueError(f"No <code_wrapper> found for unity_id={unity_id} in task prompt")

        blocks = extract_tag_content(raw, tag='params')
        if blocks:
            if len(blocks) != 1:
                raise ValueError('Expected exactly one <params> block')
            raw = blocks[0].strip()

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError('Func-mode <params> JSON must be an object')

        rendered = wrapper
        for k, v in data.items():
            token = "{{" + str(k) + "}}"
            rendered = rendered.replace(token, self._to_csharp_literal(v))
        return rendered

    def _get_current_resolution(self):
        env_cfg = self.env_info_mgr.env_config or {}
        return (env_cfg.get('width'), env_cfg.get('height'))

    def _get_no_graphics(self) -> bool:
        env_cfg = self.env_info_mgr.env_config or {}
        wrapper_cfg = env_cfg.get('env_wrapper_cfg', {}) if isinstance(env_cfg, dict) else {}

        raw = self.cfg.get('no_graphics', None)
        if raw is None:
            raw = env_cfg.get('no_graphics', None)
        if raw is None:
            raw = wrapper_cfg.get('no_graphics', None)
        return self._coerce_bool(raw, default=False)

    def _get_unity_start_config(self) -> Dict[str, Any]:
        env_cfg = self.env_info_mgr.env_config or {}
        wrapper_cfg = env_cfg.get('env_wrapper_cfg', {}) if isinstance(env_cfg, dict) else {}

        timeout_wait_s = self.cfg.get('unity_start_timeout_s', None)
        if timeout_wait_s is None:
            timeout_wait_s = env_cfg.get('unity_start_timeout_s', None)
        if timeout_wait_s is None:
            timeout_wait_s = wrapper_cfg.get('unity_start_timeout_s', 600.0)

        max_attempts = self.cfg.get('unity_start_max_attempts', None)
        if max_attempts is None:
            max_attempts = env_cfg.get('unity_start_max_attempts', None)
        if max_attempts is None:
            max_attempts = wrapper_cfg.get('unity_start_max_attempts', 0)

        retry_wait_s = self.cfg.get('unity_start_retry_wait_s', None)
        if retry_wait_s is None:
            retry_wait_s = env_cfg.get('unity_start_retry_wait_s', None)
        if retry_wait_s is None:
            retry_wait_s = wrapper_cfg.get('unity_start_retry_wait_s', 0.5)

        return {
            'timeout_wait_s': float(timeout_wait_s),
            'max_attempts': max(0, int(max_attempts)),
            'retry_wait_s': max(0.0, float(retry_wait_s)),
        }

    @staticmethod
    def _format_func_render_error(error: Exception) -> str:
        return (
            '[compile] Error: Invalid assistant tool/function-call action before Unity execution. '
            'The previous assistant output could not be converted into executable Unity code, so no environment action was run. '
            'Fix the assistant tool_call format, tool name, or arguments; this is not a Unity environment rendering/image problem. '
            f'Details: {type(error).__name__}: {error}'
        )

    @staticmethod
    def _merge_step_message_parts(*parts):
        messages = []
        for part in parts:
            if part is None or part == '':
                continue
            if isinstance(part, list):
                messages.extend(str(item) for item in part if item is not None and str(item) != '')
            else:
                messages.append(str(part))
        if not messages:
            return ''
        if len(messages) == 1:
            return messages[0]
        return messages

    @staticmethod
    def _step_message_indicates_script_error(step_msg) -> bool:
        if step_msg is None or step_msg == '':
            return False
        if isinstance(step_msg, list):
            messages = [str(item).lstrip() for item in step_msg if item is not None and str(item) != '']
        else:
            messages = [str(step_msg).lstrip()]
        return any(msg.startswith('[compile]') or msg.startswith('[runtime]') for msg in messages)

    @staticmethod
    def _positive_int_or_none(value):
        try:
            value = int(value)
        except Exception:
            return None
        return value if value > 0 else None

    @classmethod
    def _build_rollout_budget_prompt(cls, env_cfg: dict, *, prefer_english: bool = True) -> str:
        if not isinstance(env_cfg, dict):
            return ''

        max_attempts = cls._positive_int_or_none(env_cfg.get('max_attempts', None))
        max_steps_per_attempt = cls._positive_int_or_none(env_cfg.get('max_steps_per_attempt', None))
        if max_attempts is None and max_steps_per_attempt is None:
            return ''

        if prefer_english:
            lines = ['Play budget for this task:']
            if max_attempts is not None:
                lines.append(f'- You can play up to {max_attempts} game round(s).')
            if max_steps_per_attempt is not None:
                lines.append(f'- In each game round, you can take up to {max_steps_per_attempt} operation step(s).')
            if max_attempts is not None and max_steps_per_attempt is not None:
                lines.append(
                    f'- Across the whole task, you can take at most {max_attempts * max_steps_per_attempt} '
                    'operation step(s). Earlier successful game rounds are recorded, but the rollout continues '
                    'until the final game round. Only the score from the final game round is used as the final '
                    'evaluation score.'
                )
            return '\n'.join(lines)

        lines = ['本次任务的可操作次数：']
        if max_attempts is not None:
            lines.append(f'- 最多可以玩 {max_attempts} 个游戏回合。')
        if max_steps_per_attempt is not None:
            lines.append(f'- 每个游戏回合最多可以操作 {max_steps_per_attempt} 步。')
        if max_attempts is not None and max_steps_per_attempt is not None:
            lines.append(
                f'- 整个任务最多可以操作 {max_attempts * max_steps_per_attempt} 步；'
                '中间游戏回合即使成功，也会记录下来并继续到最后一个游戏回合；最终成绩以最后一个游戏回合的得分为准。'
            )
        return '\n'.join(lines)

    def _render_func_code_actions(self, exec_code_act: Dict[int, Any], log_prefix: str = 'EnvWrapper.step'):
        if self._get_action_mode() != 'func' or not exec_code_act:
            return exec_code_act, {}

        rendered = {}
        render_errors = {}
        for ml_id, params_text in exec_code_act.items():
            unity_id = self.ml_unity_id_map.get(ml_id)
            try:
                if unity_id is None:
                    raise KeyError(f'No unity_id mapping found for ml_id={ml_id}')
                rendered[ml_id] = self._render_func_wrapper(unity_id, params_text)
            except Exception as e:
                print(
                    f"[{log_prefix}] Failed to render func wrapper for "
                    f"ml_id={ml_id}, unity_id={unity_id}: {e}"
                )
                render_errors[ml_id] = self._format_func_render_error(e)
                rendered[ml_id] = None
        return {k: v for k, v in rendered.items() if v is not None}, render_errors

    def _get_virtual_display_config(self) -> Dict[str, Any]:
        env_cfg = self.env_info_mgr.env_config or {}
        wrapper_cfg = env_cfg.get('env_wrapper_cfg', {}) if isinstance(env_cfg, dict) else {}

        enabled_raw = self.cfg.get('virtual_display', None)
        if enabled_raw is None:
            enabled_raw = env_cfg.get('virtual_display', None)
        if enabled_raw is None:
            enabled_raw = wrapper_cfg.get('virtual_display', None)

        force_raw = self.cfg.get('virtual_display_force', None)
        if force_raw is None:
            force_raw = env_cfg.get('virtual_display_force', None)
        if force_raw is None:
            force_raw = wrapper_cfg.get('virtual_display_force', None)

        display_num = self.cfg.get('virtual_display_num', None)
        if display_num is None:
            display_num = env_cfg.get('virtual_display_num', None)
        if display_num is None:
            display_num = wrapper_cfg.get('virtual_display_num', 99)

        display_max_tries = self.cfg.get('virtual_display_max_tries', None)
        if display_max_tries is None:
            display_max_tries = env_cfg.get('virtual_display_max_tries', None)
        if display_max_tries is None:
            display_max_tries = wrapper_cfg.get('virtual_display_max_tries', 20)

        width = self.cfg.get('virtual_display_width', None)
        if width is None:
            width = env_cfg.get('virtual_display_width', None)
        if width is None:
            width = wrapper_cfg.get('virtual_display_width', 1024)

        height = self.cfg.get('virtual_display_height', None)
        if height is None:
            height = env_cfg.get('virtual_display_height', None)
        if height is None:
            height = wrapper_cfg.get('virtual_display_height', 768)

        color_depth = self.cfg.get('virtual_display_color_depth', None)
        if color_depth is None:
            color_depth = env_cfg.get('virtual_display_color_depth', None)
        if color_depth is None:
            color_depth = wrapper_cfg.get('virtual_display_color_depth', 24)

        startup_wait_s = self.cfg.get('virtual_display_startup_wait_s', None)
        if startup_wait_s is None:
            startup_wait_s = env_cfg.get('virtual_display_startup_wait_s', None)
        if startup_wait_s is None:
            startup_wait_s = wrapper_cfg.get('virtual_display_startup_wait_s', 0.2)

        render_env = self.cfg.get('virtual_display_render_env', None)
        if render_env is None:
            render_env = env_cfg.get('virtual_display_render_env', None)
        if render_env is None:
            render_env = wrapper_cfg.get('virtual_display_render_env', None)
        if not isinstance(render_env, dict):
            render_env = {}

        return {
            'enabled': self._coerce_bool(enabled_raw, default=False),
            'force': self._coerce_bool(force_raw, default=False),
            'display_num': int(display_num),
            'display_max_tries': int(display_max_tries),
            'width': int(width),
            'height': int(height),
            'color_depth': int(color_depth),
            'startup_wait_s': float(startup_wait_s),
            'render_env': render_env,
        }

    def _get_unity_start_serialize(self, alloc: Optional[Dict[str, Any]] = None) -> bool:
        env_cfg = self.env_info_mgr.env_config or {}
        wrapper_cfg = env_cfg.get('env_wrapper_cfg', {}) if isinstance(env_cfg, dict) else {}

        raw = self.cfg.get('serialize_unity_startup', None)
        if raw is None:
            raw = env_cfg.get('serialize_unity_startup', None)
        if raw is None:
            raw = wrapper_cfg.get('serialize_unity_startup', None)

        if isinstance(raw, str) and raw.strip().lower() == 'auto':
            raw = None

        if raw is not None:
            return self._coerce_bool(raw, default=True)

        if alloc is None:
            alloc = self._get_port_alloc_config()

        if bool(alloc.get('auto_scan', False)):
            return True

        return self.cfg.get('worker_index', None) is None

    def _should_recreate_env(self):
        if self.env is None:
            return True
        return self._get_current_resolution() != self._last_env_resolution

    @staticmethod
    def _can_bind_tcp_port(port: int, host: str = '127.0.0.1') -> bool:
        """Return True if host:port is bindable (likely free).

        Keep this check strict and cross-platform: do not set SO_REUSEADDR here,
        otherwise Linux may produce false positives in some scenarios.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, int(port)))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _get_port_alloc_config(self) -> Dict[str, Any]:
        env_cfg = self.env_info_mgr.env_config or {}
        wrapper_cfg = env_cfg.get('env_wrapper_cfg', {}) if isinstance(env_cfg, dict) else {}

        base_port = _resolve_configured_unity_base_port(self.cfg, env_cfg)
        worker_index = int(self.cfg.get('worker_index', 0) or 0)
        env_id_offset = int(env_cfg.get('env_id', 0) or 0)
        port_stride = int(
            self.cfg.get('base_port_stride',
                         env_cfg.get('base_port_stride',
                                     wrapper_cfg.get('base_port_stride', 1)))
        )
        port_stride = max(1, port_stride)
        start_port = base_port + (worker_index + env_id_offset) * port_stride

        auto_scan = bool(
            self.cfg.get('auto_port_scan',
                         env_cfg.get('auto_port_scan',
                                     wrapper_cfg.get('auto_port_scan', True)))
        )
        max_tries = int(
            self.cfg.get('base_port_max_tries',
                         env_cfg.get('base_port_max_tries',
                                     wrapper_cfg.get('base_port_max_tries', 256)))
        )
        max_tries = max(1, max_tries)

        bind_host = str(
            self.cfg.get('base_port_bind_host',
                         env_cfg.get('base_port_bind_host',
                                     wrapper_cfg.get('base_port_bind_host', '127.0.0.1')))
        )

        return {
            'start_port': int(start_port),
            'port_stride': int(port_stride),
            'auto_scan': bool(auto_scan),
            'max_tries': int(max_tries),
            'bind_host': bind_host,
        }

    def _resolve_unity_base_port(self) -> int:
        """Pick Unity ML-Agents base_port with optional auto-scan.

        Priority for base port source:
          1) cfg['base_port']
          2) cfg['env_config_overrides']['base_port']
          3) AGENTARK_EDITOR_BASE_PORT or AGENTARK_PLAYER_BASE_PORT
          4) env_config['base_port']
          5) env_config['env_wrapper_cfg']['base_port']
          6) default 5005
        """
        alloc = self._get_port_alloc_config()
        start_port = alloc['start_port']
        port_stride = alloc['port_stride']
        auto_scan = alloc['auto_scan']
        max_tries = alloc['max_tries']
        bind_host = alloc['bind_host']

        if not auto_scan:
            return int(start_port)

        for i in range(max_tries):
            candidate = int(start_port + i * port_stride)
            if self._can_bind_tcp_port(candidate, host=bind_host):
                return candidate

        raise RuntimeError(
            f"No free Unity base_port found from {start_port} with stride={port_stride}, tries={max_tries}"
        )

    def start_unity_env(self):
        ### side channels
        # 1. 内置EngineConfigurationChannel
        self.engine_channel = EngineConfigurationChannel()
        # 2. 内置EnvironmentParametersChannel, 通用环境设置
        self.env_channel = EnvironmentParametersChannel()
        # 3. raw byte channel, 由于只有该类有get_and_clear_received_messages函数，因此需要。
        # 当python侧决定'reload_scene'后，只有用此channel来获取并clear，才不会导致unity多次重复reload(因为其他channel的msg会多次被重复获取). uuid写死即可
        self.raw_byte_channel = RawBytesChannel(uuid.UUID("621f0a70-4f87-11ea-a6bf-999999999999"))

        # 4. code channel
        self.env_num = self.env_info_mgr.env_config.get('num_parallel_envs', 1)
        self.code_act_channels = self.get_code_act_channels()
        self.image_channels = self.get_image_channels()
        # self.script_channel = StringLogChannel()

        side_channels = [self.engine_channel, self.env_channel, self.raw_byte_channel, *self.code_act_channels, *self.image_channels]

        self.env_info_mgr.set_engine_para(self.engine_channel)
        self.env_info_mgr.set_env_para(self.env_channel)

        alloc = self._get_port_alloc_config()

        vd_cfg = self._get_virtual_display_config()
        if vd_cfg['enabled']:
            display = _SharedXvfbManager.ensure_started(vd_cfg)
            print(f"[EnvWrapper] virtual_display enabled, using DISPLAY={display}")

        last_error = None
        no_graphics = self._get_no_graphics()
        additional_args = self.cfg.get('additional_args', None)
        if additional_args is not None:
            if not isinstance(additional_args, (list, tuple)):
                raise ValueError('env_cfg.additional_args must be a list of Unity Player command-line arguments')
            additional_args = [str(value) for value in additional_args]
        start_cfg = self._get_unity_start_config()
        serialize_startup = self._get_unity_start_serialize(alloc)
        startup_failures = 0
        # Unity ML-Agents startup is not concurrency-safe when multiple wrappers do auto port scan
        # at the same time. For fixed per-worker ports we can skip the global lock to avoid
        # head-of-line blocking when one Unity startup stalls.
        startup_lock = self._unity_start_lock if serialize_startup else nullcontext()
        with startup_lock:
            if alloc['auto_scan']:
                first_port = self._resolve_unity_base_port()
            else:
                first_port = alloc['start_port']

            port_attempt_index = 0
            candidate_port = int(first_port)
            while port_attempt_index < alloc['max_tries']:
                try:
                    self.env = UnityEnvironment(
                        file_name=self.cfg['env_path'],
                        seed=1,
                        side_channels=side_channels,
                        timeout_wait=start_cfg['timeout_wait_s'],
                        base_port=candidate_port,
                        no_graphics=no_graphics,
                        additional_args=additional_args,
                    )
                    self._unity_base_port = candidate_port
                    print(
                        f"[EnvWrapper] UnityEnvironment started at base_port={self._unity_base_port}, "
                        f"no_graphics={no_graphics}"
                    )
                    last_error = None
                    break
                except UnityWorkerInUseException as e:
                    self.env = None
                    last_error = e
                    print(
                        f"[EnvWrapper] Unity worker in use at base_port={candidate_port}; retrying next port. "
                        f"{type(e).__name__}: {e}"
                    )
                    port_attempt_index += 1
                    candidate_port = int(first_port + port_attempt_index * alloc['port_stride'])
                    continue
                except (UnityTimeOutException, UnityCommunicationException) as e:
                    self.env = None
                    last_error = e
                    startup_failures += 1
                    print(
                        f"[EnvWrapper] Unity start failed at base_port={candidate_port} "
                        f"after timeout_wait={start_cfg['timeout_wait_s']}s; "
                        f"startup_failure={startup_failures}: {type(e).__name__}: {e}"
                    )
                    if start_cfg['max_attempts'] and startup_failures >= start_cfg['max_attempts']:
                        break
                    if start_cfg['retry_wait_s'] > 0:
                        time.sleep(start_cfg['retry_wait_s'])
                    continue
                except Exception as e:
                    self.env = None
                    last_error = e
                    startup_failures += 1
                    print(
                        f"[EnvWrapper] Unity start raised {type(e).__name__} at base_port={candidate_port}; "
                        f"startup_failure={startup_failures}: {e}"
                    )
                    if start_cfg['max_attempts'] and startup_failures >= start_cfg['max_attempts']:
                        break
                    if start_cfg['retry_wait_s'] > 0:
                        time.sleep(start_cfg['retry_wait_s'])
                    continue

        if self.env is None:
            raise RuntimeError(
                f"Failed to start UnityEnvironment after {port_attempt_index + 1} base_port attempt(s). "
                f"start_port={first_port}, stride={alloc['port_stride']}, "
                f"timeout_wait={start_cfg['timeout_wait_s']}, "
                f"startup_failures={startup_failures}, "
                f"startup_failure_limit={start_cfg['max_attempts'] or 'unlimited'}, "
                f"last_error={last_error}"
            )

        self._last_env_resolution = self._get_current_resolution()

        # self.system_prompt = self.env_info_mgr.system_prompt
        # 废弃，task_prompt由环境通过side_channel传递
        # self.task_prompt = self.env_info_mgr.get_task_prompt() # self.cfg['task_prompt']

        # 保存spec
        # self.script_channel.send_bool_and_str(False, '')
        self.send_code_act(agent_id=-1, code_act={})

        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.env_spec = self.env.behavior_specs[self.behavior_name]

        self.env_info_mgr.task_prompt = self._get_task_prompt_after_reset()
        self._refresh_code_wrappers_from_task_prompt()

        # self.agent_index_list = sorted(list(decision_steps.agent_id_to_index.values()))

    def reset(
        self,
        task_name=None,
        group_seed=None,
        env_id=None,
        *,
        history_snapshot: Optional[Dict[int, list]] = None,
        start_attempt_index: Optional[int] = None,
    ):

        # set task
        self.env_info_mgr.reset(task_name=task_name, group_seed=group_seed, env_id=env_id)

        if self._should_recreate_env():
            if self.env is not None:
                self.close_unity_env()
            self.start_unity_env()

        # 持有该 global episode（所有agent经历完一轮）的 agent_id信息

        _, info = self.soft_reset()

        # dummy act设为1，触发hardreset（确保运行时生成的脚本被清除）
        dummy_act = self.env_spec.action_spec.empty_action(self.env_num)
        self.env.set_actions(self.behavior_name, dummy_act)

        # AgentArk hard-reset mechanism.
        # 1) 通过 raw_bytes_channel 发送 reload_scene 指令
        # 2) 若 task_config.json/yaml 中存在 "task_params" 字段，则原样通过 raw_bytes_channel 发送 JSON

        # 先通过 raw_bytes 发送 env_params，支持非浮点/任意字段的灵活配置
        self._send_env_params_via_raw_bytes()

        # 触发场景重载
        ### 注意，send_raw_data时，reload_scene必须最后发送，否则若先接收reload，其他信息
        # 会被unity侧跳过接收（比如task params接收）而直接reload scene，导致信息丢失
        self.raw_byte_channel.send_raw_data(b'reload_scene')

        self.env.step()

        # hard reset了环境后，此处decision_steps为空（若只有一个agent），需再次reset才为重置完成
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        env_cfg = self.env_info_mgr.env_config or {}
        task_params = env_cfg.get("task_params", None)
        if task_params is not None:
            try:
                # 只发送 params 部分的 JSON，并在前面加上固定前缀，便于 Unity 识别
                payload_bytes = ("[task_params]" + json.dumps(task_params)).encode("utf-8")
                self.raw_byte_channel.send_raw_data(payload_bytes)
            except Exception as e:
                print(f"[EnvWrapper.reset] Failed to send task_params via raw_bytes: {e}")
        self.env.step()

        self.env.reset()

        self.env_info_mgr.task_prompt = self._get_task_prompt_after_reset()
        self._refresh_code_wrappers_from_task_prompt()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        current_init_prompt = {
            unity_id: self._build_reset_obs_payload(prompt)
            for unity_id, prompt in self.env_info_mgr.task_prompt.items()
        }
        unity_id_obs = self.get_empty_obs(current_init_prompt)

        self.ml_unity_id_map = self._get_agent_id_map(decision_steps)
        self._prepare_reset_plan(
            agent_ids=sorted(int(unity_id) for unity_id in self.ml_unity_id_map.values()),
            history_snapshot=history_snapshot,
        )

        # reset agent_idx done
        assert len(terminal_steps.agent_id_to_index) == 0
        self.episode_agent_id_to_index = deepcopy(decision_steps.agent_id_to_index)
        self.agent_done_dict = {k: False for k in self.episode_agent_id_to_index}

        env_obs = decision_steps.obs

        # for idx in self.agent_index_list:
        #     obs[idx]['env_msg'] = ''
        #     obs[idx]['vis'] = [env_obs[0][i]]
        # map ml_id to unity_id
        obs = {}
        for ml_id, unity_id in self.ml_unity_id_map.items():
            obs[ml_id] = unity_id_obs[unity_id]
            obs[ml_id]['vis'] = [env_obs[0][decision_steps.agent_id_to_index[ml_id]]]

        obs, info = self._apply_initial_observation_warmup(obs, info, current_init_prompt)

        obs = self.post_process_obs(obs)
        obs = self.context_mgr.on_reset(
            self.env_info_mgr.env_config,
            self.ml_unity_id_map,
            obs,
            history_snapshot=self._last_reset_plan.get('history_snapshot', {}),
        )
        resolved_start_attempt_index = self._resolve_start_attempt_index(
            start_attempt_index,
            self._last_reset_plan.get('history_snapshot', {}),
        )
        info['history'] = {
            'bucket_key': self._last_reset_plan.get('history_bucket_key', None),
            'history_bucket_key': self._last_reset_plan.get('history_bucket_key', None),
        }
        info['attempt'] = {
            'index': resolved_start_attempt_index,
            'history_prefix_attempt_count': max(0, resolved_start_attempt_index - 1),
        }
        return obs, info

    @staticmethod
    def _count_history_prefix_attempts(history_snapshot: Optional[Dict[int, list]]) -> int:
        if not isinstance(history_snapshot, dict):
            return 0
        counts = [len(attempts) for attempts in history_snapshot.values() if isinstance(attempts, list)]
        return max(counts) if counts else 0

    @classmethod
    def _resolve_start_attempt_index(
        cls,
        start_attempt_index: Optional[int],
        history_snapshot: Optional[Dict[int, list]],
    ) -> int:
        if start_attempt_index is None:
            return cls._count_history_prefix_attempts(history_snapshot) + 1
        return max(1, int(start_attempt_index))

    def _send_env_params_via_raw_bytes(self):
        """将 env_config 中的通用环境参数通过 raw_bytes 发送给 Unity。

        使用 JSON 承载，前缀标记为 "[env_params]"，字段与 Unity 侧 EnvironmentParams 对应。
        若字段缺失则跳过，尽量不覆盖 Unity 侧的默认值。
        """
        env_cfg = self.env_info_mgr.env_config or {}
        payload = self._build_unity_env_params_payload(env_cfg)

        if not payload:
            return

        try:
            serialized = json.dumps(payload)
            self.raw_byte_channel.send_raw_data(f"[env_params]{serialized}".encode('utf-8'))
        except Exception as e:
            print(f"[EnvWrapper] Failed to send env_params via raw_bytes: {e}")

    @staticmethod
    def _normalize_obs_mode(raw_mode) -> str:
        if isinstance(raw_mode, (int, float)):
            return 'video' if int(raw_mode) == 1 else 'decision'

        mode = str(raw_mode or '').strip().lower()
        if mode in ('video', '1'):
            return 'video'
        return 'decision'

    @staticmethod
    def _obs_mode_is_video(env_cfg: dict) -> bool:
        if not isinstance(env_cfg, dict):
            return False

        return EnvWrapper._normalize_obs_mode(env_cfg.get('obs_mode', None)) == 'video'

    @staticmethod
    def _get_video_frame_selection(env_cfg: dict) -> str:
        wrapper_cfg = env_cfg.get('env_wrapper_cfg', {}) if isinstance(env_cfg, dict) else {}
        if not isinstance(wrapper_cfg, dict):
            wrapper_cfg = {}

        raw_selection = wrapper_cfg.get('video_frame_selection', 'decision_only')
        selection = str(raw_selection or '').strip().lower().replace('-', '_')
        valid_selections = {
            'decision_only',
            'transition_and_decision',
            'transition_only',
        }
        if selection not in valid_selections:
            raise ValueError(
                'env_wrapper_cfg.video_frame_selection must be one of: '
                'decision_only, transition_and_decision, transition_only'
            )
        return selection

    @staticmethod
    def _coerce_non_negative_int(value, default: int = 0) -> int:
        try:
            return max(0, int(value))
        except Exception:
            return max(0, int(default))

    @staticmethod
    def _coerce_positive_float_or_none(value):
        if value is None:
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _get_initial_observation_cfg(env_cfg: dict) -> dict:
        wrapper_cfg = env_cfg.get('env_wrapper_cfg', {}) if isinstance(env_cfg, dict) else {}
        if not isinstance(wrapper_cfg, dict):
            wrapper_cfg = {}
        raw_cfg = wrapper_cfg.get('initial_observation', {})
        if not isinstance(raw_cfg, dict):
            raw_cfg = {}

        enabled = EnvWrapper._coerce_bool(raw_cfg.get('enabled', False), default=False)
        raw_steps = raw_cfg.get('no_action_decision_steps', raw_cfg.get('no_action_steps', None))
        steps = EnvWrapper._coerce_non_negative_int(raw_steps, default=1 if enabled else 0)
        if enabled and steps <= 0:
            steps = 1
        if not enabled:
            steps = 0

        duration = EnvWrapper._coerce_positive_float_or_none(
            raw_cfg.get('empty_step_duration_seconds', raw_cfg.get('empty_step_duration_s', None))
        )
        min_frames = EnvWrapper._coerce_non_negative_int(raw_cfg.get('require_min_frames_per_camera', 0), default=0)

        return {
            'enabled': bool(enabled and steps > 0),
            'no_action_decision_steps': int(steps),
            'empty_step_duration_seconds': duration,
            'require_min_frames_per_camera': int(min_frames),
        }

    @staticmethod
    def _validate_initial_observation_cfg(env_cfg: dict, initial_cfg: dict):
        if not isinstance(initial_cfg, dict) or not initial_cfg.get('enabled'):
            return
        if not EnvWrapper._obs_mode_is_video(env_cfg):
            raise ValueError(
                'env_wrapper_cfg.initial_observation requires obs_mode=video '
                'so Unity creates ImageFramesSideChannel payloads.'
            )

    @staticmethod
    def _build_unity_env_params_payload(env_cfg: dict) -> dict:
        env_cfg = env_cfg if isinstance(env_cfg, dict) else {}
        payload = {
            k: v for k, v in env_cfg.items()
            if k not in ('task_params', 'env_wrapper_cfg')
        }
        if 'obs_mode' in payload:
            payload['obs_mode'] = EnvWrapper._normalize_obs_mode(payload['obs_mode'])

        initial_cfg = EnvWrapper._get_initial_observation_cfg(env_cfg)
        payload['initial_observation_no_action_steps'] = 0
        payload['initial_observation_empty_step_duration_seconds'] = 0.0

        if initial_cfg.get('enabled'):
            EnvWrapper._validate_initial_observation_cfg(env_cfg, initial_cfg)
            warmup_steps = int(initial_cfg.get('no_action_decision_steps', 0) or 0)
            payload['initial_observation_no_action_steps'] = warmup_steps
            duration = initial_cfg.get('empty_step_duration_seconds', None)
            if duration is not None:
                payload['initial_observation_empty_step_duration_seconds'] = float(duration)

            max_steps = EnvWrapper._positive_int_or_none(payload.get('max_steps', None))
            if max_steps is not None:
                payload['max_steps'] = int(max_steps) + int(warmup_steps)

        return payload

    @staticmethod
    def _decode_image_payload_to_pil(payload):
        if not payload or 'error' in payload or not payload.get('cameras'):
            return None
        if Image is None:
            return {'error': 'Pillow not installed'}

        cam_dict = {}
        try:
            for cam_idx, frames in payload['cameras'].items():
                img_list = []
                for frame in frames:
                    try:
                        _, _, gray, png_bytes = frame
                        img = Image.open(io.BytesIO(png_bytes))
                        img.load()
                        if gray and img.mode != 'L':
                            img = img.convert('L')
                        img_list.append(img)
                    except Exception:
                        continue
                if img_list:
                    cam_dict[cam_idx] = img_list
            return cam_dict if cam_dict else None
        except Exception as e:
            return {'error': f'pil_decode_failed: {e}'}

    @staticmethod
    def _merge_video_pil_frames(dst: dict, src):
        if not isinstance(src, dict) or 'error' in src:
            return dst
        if not isinstance(dst, dict):
            dst = {}
        for cam_idx, frames in src.items():
            if not isinstance(frames, list):
                continue
            dst.setdefault(cam_idx, []).extend(frames)
        return dst

    def _attach_image_payloads_to_obs(self, obs: Dict[int, dict], video_payloads: Dict[int, dict]):
        for ml_id, unity_id in self.ml_unity_id_map.items():
            if ml_id not in obs or obs[ml_id].get('skip_infer'):
                continue
            payload = video_payloads.get(unity_id)
            if payload and 'error' not in payload and payload.get('cameras'):
                obs[ml_id]['video_raw'] = payload
                obs[ml_id]['video_pil'] = self._decode_image_payload_to_pil(payload)
            else:
                obs[ml_id]['video_raw'] = None
                obs[ml_id]['video_pil'] = None
        return obs

    def _clear_image_channels(self):
        for channel in getattr(self, 'image_channels', []) or []:
            try:
                channel.get_and_clear()
            except Exception:
                pass

    def _clear_code_channel_step_msgs(self):
        for channel in getattr(self, 'code_act_channels', []) or []:
            try:
                channel.clear_step_msgs()
            except Exception:
                pass

    @staticmethod
    def _initial_observation_frame_counts(obs: Dict[int, dict]) -> Dict[int, list]:
        counts = {}
        for ml_id, obs_dict in (obs or {}).items():
            if not isinstance(obs_dict, dict):
                continue
            cam_num = len(obs_dict.get('vis') or [])
            video_pil = obs_dict.get('video_pil', None)
            ml_counts = []
            for cam_idx in range(cam_num):
                side_count = 0
                if isinstance(video_pil, dict) and 'error' not in video_pil:
                    side_count = len(video_pil.get(cam_idx, []) or [])
                ml_counts.append(side_count + 1)
            counts[int(ml_id)] = ml_counts
        return counts

    def _build_obs_from_decision_steps(self, decision_steps, current_init_prompt: Dict[int, dict]):
        unity_id_obs = self.get_empty_obs(current_init_prompt)
        obs = {}
        env_obs = decision_steps.obs
        for ml_id, unity_id in self.ml_unity_id_map.items():
            if ml_id not in decision_steps.agent_id_to_index:
                continue
            obs[ml_id] = unity_id_obs[unity_id]
            obs[ml_id]['vis'] = [env_obs[0][decision_steps.agent_id_to_index[ml_id]]]
        return obs

    def _apply_initial_observation_warmup(self, obs: Dict[int, dict], info: Dict[str, Any], current_init_prompt: Dict[int, dict]):
        env_cfg = self.env_info_mgr.env_config or {}
        initial_cfg = self._get_initial_observation_cfg(env_cfg)
        if not initial_cfg.get('enabled'):
            return obs, info

        self._validate_initial_observation_cfg(env_cfg, initial_cfg)

        configured_steps = int(initial_cfg.get('no_action_decision_steps', 0) or 0)
        accumulated_video_pil: Dict[int, dict] = {}
        executed_steps = 0
        current_obs = obs
        self._clear_image_channels()

        for _ in range(configured_steps):
            dummy_act = self.env_spec.action_spec.empty_action(self.env_num)
            self.env.set_actions(self.behavior_name, dummy_act)
            self.send_code_act(agent_id=-1, code_act={})
            self.env.step()

            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
            if len(terminal_steps.agent_id_to_index) > 0:
                raise RuntimeError(
                    'Initial observation warmup ended one or more agents before the first real model action. '
                    'Increase max_steps_per_attempt, reduce initial_observation.no_action_decision_steps, '
                    'or adjust the task so hidden no-code observation steps cannot terminate the episode.'
                )

            video_payloads = {ch.agent_id: ch.get_and_clear() for ch in self.image_channels}
            for ml_id, unity_id in self.ml_unity_id_map.items():
                payload = video_payloads.get(unity_id)
                decoded = self._decode_image_payload_to_pil(payload) if payload else None
                if isinstance(decoded, dict) and 'error' not in decoded:
                    accumulated_video_pil[ml_id] = self._merge_video_pil_frames(
                        accumulated_video_pil.get(ml_id, {}),
                        decoded,
                    )

            self._clear_code_channel_step_msgs()
            current_obs = self._build_obs_from_decision_steps(decision_steps, current_init_prompt)
            executed_steps += 1

        for ml_id, cam_dict in accumulated_video_pil.items():
            if ml_id in current_obs and cam_dict:
                current_obs[ml_id]['video_pil'] = cam_dict
                current_obs[ml_id]['video_raw'] = None

        frame_counts = self._initial_observation_frame_counts(current_obs)
        required_min = int(initial_cfg.get('require_min_frames_per_camera', 0) or 0)
        if required_min > 0:
            too_short = {
                ml_id: counts
                for ml_id, counts in frame_counts.items()
                if any(int(count) < required_min for count in counts)
            }
            if too_short:
                raise RuntimeError(
                    'Initial observation warmup produced fewer frames than required: '
                    f'require_min_frames_per_camera={required_min}, frames_per_camera={too_short}'
                )

        info['initial_observation'] = {
            'enabled': True,
            'configured_no_action_decision_steps': configured_steps,
            'executed_no_action_decision_steps': executed_steps,
            'empty_step_duration_seconds': initial_cfg.get('empty_step_duration_seconds', None),
            'frames_per_camera': frame_counts,
            'unity_max_steps_offset': configured_steps,
        }
        return current_obs, info

    def soft_reset(self):
        """ML-Agents reset without the AgentArk scene-reload hard reset."""
        # self.script_channel.send_bool_and_str(False, '')
        self.send_code_act(agent_id=-1, code_act={})

        self.env.reset()
        self.env_info_mgr.task_prompt = self._get_task_prompt_after_reset()
        self._refresh_code_wrappers_from_task_prompt()

        init_prompt = {k: self._build_reset_obs_payload(v) for k, v in self.env_info_mgr.task_prompt.items()}
        unity_id_obs = self.get_empty_obs(init_prompt)
        info = {}
        return unity_id_obs, info

    def _build_llm_visible_prompt(self, raw_prompt: str) -> str:
        env_cfg = getattr(self.env_info_mgr, 'env_config', {}) or {}
        prompt_cfg = env_cfg.get('prompt') if isinstance(env_cfg.get('prompt'), dict) else {}
        include_code_wrapper = self._coerce_bool(
            prompt_cfg.get('include_code_wrapper', env_cfg.get('include_code_wrapper', False)),
            default=False,
        )
        prefer_english = self._preferred_language_tag() != 'chinese_ver'
        visible_prompt = build_llm_visible_prompt(
            raw_prompt,
            system_prompt=getattr(self.env_info_mgr, 'system_prompt', ''),
            prefer_english_lang=prefer_english,
            include_code_wrapper=include_code_wrapper,
        )
        budget_prompt = self._build_rollout_budget_prompt(env_cfg, prefer_english=prefer_english)
        if visible_prompt and budget_prompt:
            return visible_prompt.rstrip() + '\n\n' + budget_prompt
        return visible_prompt or budget_prompt

    def _build_reset_obs_payload(self, raw_prompt: str) -> dict:
        prefer_english = self._preferred_language_tag() != 'chinese_ver'
        visible_prompt = self._build_llm_visible_prompt(raw_prompt)
        payload = parse_task_prompt_payload(raw_prompt, prefer_english_lang=prefer_english)

        context_parts = []
        for key in ('reset_context', 'observation_context'):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                context_parts.append(value.strip())
        reset_context = '\n\n'.join(context_parts).strip()

        step_msg = visible_prompt
        if reset_context:
            tagged_context = f"<reset_context>\n{reset_context}\n</reset_context>"
            step_msg = (visible_prompt.rstrip() + '\n\n' + tagged_context).strip() if visible_prompt else tagged_context

        return {
            'step_msg': step_msg,
            'task_prompt': visible_prompt,
            'reset_context': reset_context,
        }

    def step(self, code_act, info={}):
        dummy_act = self.env_spec.action_spec.empty_action(len(code_act))
        self.env.set_actions(self.behavior_name, dummy_act)

        # send run script bool, and script str.
        # self.script_channel.send_bool_and_str(True, code_act[0])
        exec_code_act = {k: v for k, v in code_act.items() if v is not None}

        exec_code_act, func_render_errors = self._render_func_code_actions(exec_code_act, log_prefix='EnvWrapper.step')

        self.send_code_act(agent_id=list(exec_code_act.keys()), code_act=exec_code_act)

        self.env.step()

        # if self.env_info_mgr.cfg['task_type'] == 'Create':
        #     # inter loop, until the action is exec finish
        #     total_steps = self.cfg['human_check_steps']
        #     for t in range(total_steps):
        #         # self.script_channel.send_bool(False)
        #         self.script_channel.send_bool_and_str(False, '')
        #         self.env.step()
        #         print(t)

        # env.step后，获取信息。注意，当并行agent时，多个agent请求code_act，但step后可能只有部分agent在该step返回信息
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        # multi sub_env情况，从ds，ts获取next obs 的agent_id(ts也需要，因为如果开启auto reset on done，next_obs是下一轮的start， 有意义)
        next_obs_list = decision_steps.agent_id.tolist() + terminal_steps.agent_id.tolist()
        next_obs = self.get_empty_obs(agent_id_list=next_obs_list)

        # self.empty_step(100)
        done = {'__all__': False}
        truncated = {}

        reward = {}
        for ml_id in terminal_steps.agent_id_to_index.keys():
            if ml_id in self.agent_done_dict and self.agent_done_dict[ml_id]:
                # 在利用本轮的ds,ts前，先用上轮保存的agent_done_dict，剔除本轮之前已done（比如script错误且不reset on done情况下，会一直需要dummy act）的agent
                next_obs[ml_id]['skip_infer'] = True
                continue
            done[ml_id] = True
            self.agent_done_dict[ml_id] = True
            reward[ml_id] = terminal_steps.reward[terminal_steps.agent_id_to_index[ml_id]]
            next_obs[ml_id]['vis'] = [terminal_steps.obs[0][terminal_steps.agent_id_to_index[ml_id]]]

            index = terminal_steps.agent_id_to_index[ml_id]
            if terminal_steps.interrupted[index]:
                truncated[ml_id] = True

        for ml_id in decision_steps.agent_id_to_index.keys():
            if ml_id in self.agent_done_dict and self.agent_done_dict[ml_id]:
                # 同ts
                next_obs[ml_id]['skip_infer'] = True
                continue
            done[ml_id] = False
            assert not self.agent_done_dict[ml_id]
            reward[ml_id] = decision_steps.reward[decision_steps.agent_id_to_index[ml_id]]
            next_obs[ml_id]['vis'] = [decision_steps.obs[0][decision_steps.agent_id_to_index[ml_id]]]

        # step_msgs = self.script_channel.get_step_msgs()
        # step_msgs= {i: self.code_act_channels[self.ml_unity_id_map[i]].get_step_msgs() for i in next_obs}
        # 只获取当前需要的agent的msg
        step_msgs= {}
        for k in next_obs.keys():
            if next_obs[k]['skip_infer']:
                continue
            unity_id = self.ml_unity_id_map[k]
            channel_step_msgs = self.code_act_channels[unity_id].get_step_msgs()
            if k in func_render_errors:
                step_msgs[k] = self._merge_step_message_parts(func_render_errors[k], channel_step_msgs)
            else:
                step_msgs[k] = channel_step_msgs

        for i in next_obs.keys():
            if next_obs[i]['skip_infer']:
                continue
            if len(step_msgs[i]) > 0:
                # next_obs[i]['step_msg'] = step_msgs[i].buffer.decode('iso-8859-1')
                next_obs[i]['step_msg'] = step_msgs[i]
                # 如果不是env返回done，而是代码报错，同样视为done
                if not self.agent_done_dict[i]:
                    # 如果done_on_script_error，agent_done_dict中标记，后续忽略该agent
                    if (
                        self.env_info_mgr.env_config.get('done_on_script_error', False)
                        and self._step_message_indicates_script_error(step_msgs[i])
                    ):
                        done[i] = True
                        self.agent_done_dict[i] = True
                        reward[i] = -1.0
                        next_obs[i]['skip_infer'] = True
            else:
                next_obs[i]['step_msg'] = ''

        info = {'truncated': truncated}
        if func_render_errors:
            info['func_render_errors'] = dict(func_render_errors)

        # self.script_channel.clear_step_msgs()
        for channel in self.code_act_channels:
            channel.clear_step_msgs()

        video_payloads = {ch.agent_id: ch.get_and_clear() for ch in self.image_channels}
        self._attach_image_payloads_to_obs(next_obs, video_payloads)

        # reward 为float形式而非numpy，便于http json传输
        # reward = float(reward)
        reward = {k: float(v) for k, v in reward.items()}
        done['__all__'] = all(self.agent_done_dict.values())

        self.context_mgr.record_step(next_obs, code_act, reward, done, info=info)

        next_obs = self.post_process_obs(next_obs)
        next_obs = self.context_mgr.finalize_obs(next_obs)
        self._publish_finalized_history()
        return next_obs, reward, done, info

    def empty_step(self, step_time=100):
        '''env空跑；用于代码执行顺利执行后的人工检查等情况
        '''
        agent_alive_list = [k for k, v in self.agent_done_dict.items() if not v]

        for t in range(step_time):
            # self.script_channel.send_bool_and_str(False, '')
            self.send_code_act(agent_id=-1, code_act={})
            self.env.step()

            # step_msgs = self.script_channel.get_step_msgs()
            step_msgs= {}
            for k in agent_alive_list:
                unity_id = self.ml_unity_id_map[k]
                step_msgs[k] = self.code_act_channels[unity_id].get_step_msgs()

            next_obs = self.get_empty_obs(self.env_info_mgr.task_prompt, agent_id_list=agent_alive_list)
            # next_obs['env_msg'] = [step_msgs[i].buffer.decode('iso-8859-1') if len(step_msgs) > 0 else '' for i in code_act.keys()]
            for i in agent_alive_list:
                next_obs[i]['step_msg'] = step_msgs[i] if len(step_msgs[i]) > 0 else ''

        return 'empty step success'

    def get_empty_obs(self, step_prompt='', agent_id_list=[]):
        '''step_prompt: dict or str, if dict, 说明由step_msg有信息，以step_msg的key为准
        agent_id_list: list of int, ml_id
        若为空，则为所有agent提供empty obs
        '''
        obs_template = {
            'step_msg': '',
            'task_prompt': None,
            'reset_context': None,
            'step_context': None,
            'observation_context': None,
            'vis': None,
            # 跳过llm推理。比如当某agent已因代码错误done后，后续step不再需要llm推理。（目前无法在env中去掉因代码错误而done的agent）
            'skip_infer': False,
            # 新增：图像 sidechannel 原始数据
            'video_raw': None,
            # 新增：PIL 解码后的图像数据 {camera_index: [Image,...]} 或 None
            'video_pil': None,
            }

        if isinstance(step_prompt, dict):
            obs = {}
            for k, v in step_prompt.items():
                obs[k] = deepcopy(obs_template)
                if isinstance(v, dict):
                    obs[k].update(v)
                else:
                    obs[k]['step_msg'] = v
            return obs

        if len(agent_id_list) == 0:
        # 注意，这里的k为unity env中的agent_id（0 - self.env_num-1），用于对应side_channel
            obs = {k: deepcopy(obs_template) for k in range(self.env_num)}
        else:
            obs = {k: deepcopy(obs_template) for k in agent_id_list}
        return obs

    def get_code_act_channels(self):
        return [AgentRawBytesChannel(agent_id=i) for i in range(self.env_num)]

    def get_image_channels(self):
        return [ImageFramesChannel(agent_id=i) for i in range(self.env_num)]

    def send_code_act(self, agent_id=-1, code_act={}):
        '''agent_id: int
        code_act: list of str
        '''
        if agent_id == -1:
            for i, channel in enumerate(self.code_act_channels):
                channel.send_code_act(False, '')
        else:
            for ml_id in agent_id:
                unity_id = self.ml_unity_id_map[ml_id]
                self.code_act_channels[unity_id].send_code_act(True, code_act[ml_id])
        return

    def post_process_obs(self, obs):
        '''对obs进行后处理
        '''
        env_cfg = self.env_info_mgr.env_config
        video_frame_selection = 'decision_only'
        if EnvWrapper._obs_mode_is_video(env_cfg):
            video_frame_selection = EnvWrapper._get_video_frame_selection(env_cfg)

        for ml_id, obs_dict in obs.items():
            # 获取当前步的img_list: list形式，可能有多个camera
            cam_num = len(obs_dict['vis'])

            obs_img_list = [[] for _ in range(cam_num)]
            # 处理图像obs
            if video_frame_selection in ('transition_and_decision', 'transition_only') and obs_dict['video_pil'] is not None:
                assert cam_num == len(obs_dict['video_pil']), '决策步的cam数量与决策步之间的img数量不匹配！'
                for cam_idx, cam_frame_list in obs_dict['video_pil'].items():
                    obs_img_list[cam_idx].extend(cam_frame_list)

            if video_frame_selection in ('decision_only', 'transition_and_decision'):
                for cam_idx in range(cam_num):
                    env_arr_img = obs_dict['vis'][cam_idx]
                    pil_img = env_arr_to_pil_image(env_arr_img)
                    obs_img_list[cam_idx].append(pil_img)

            obs_dict['video_raw'] = obs_dict['video_pil'] = None
            obs_dict['vis'] = obs_img_list

        return obs

    def _get_agent_id_map(self, decision_steps):
        '''在env reset时，传入当前的decision_steps，返回一个映射：
        mlagents_env库的agent_id: unity env的agent_id(unity中设计的agent编号， 从0开始，与side_channel的uuid一一对应)
        '''
        mlagent_agent_id = decision_steps.agent_id
        # 最后一个obs，为离散的数值obs；unity中，将此obs的一号索引设计为了unity的agent_id
        obs = decision_steps.obs[-1]
        obs = obs[:, -1].astype(np.int32)

        ml_unity_id_map = {k: v for k, v in zip(mlagent_agent_id, obs)}
        return ml_unity_id_map

    def _preferred_language_tag(self):
        """Return preferred language tag name based on config (english_ver/chinese_ver)."""
        env_cfg = getattr(self.env_info_mgr, 'env_config', {}) or {}
        prompt_cfg = env_cfg.get('prompt') or {}
        lang = str(prompt_cfg.get('language', '')).strip().lower()

        if lang in ('zh', 'cn', 'chinese'):
            return 'chinese_ver'
        if lang in ('en', 'eng', 'english'):
            return 'english_ver'
        return None

    def _filter_task_prompt_language(self, msg: str) -> str:
        """Keep only the preferred language block (if available) inside <task_prompt>."""
        prefer_tag = self._preferred_language_tag()
        if prefer_tag is None:
            return msg

        drop_tag = 'english_ver' if prefer_tag == 'chinese_ver' else 'chinese_ver'

        task_blocks = extract_tag_content(msg, 'task_prompt')
        if not task_blocks:
            return msg

        task_body = task_blocks[0]

        has_prefer = f"<{prefer_tag}>" in task_body and f"</{prefer_tag}>" in task_body
        has_drop = f"<{drop_tag}>" in task_body and f"</{drop_tag}>" in task_body

        # If the preferred language block is missing, fall back to whatever is available.
        if not has_prefer:
            return msg

        cleaned_body = task_body
        if has_drop:
            pattern = rf'<{drop_tag}>.*?</{drop_tag}>'
            cleaned_body = re.sub(pattern, '', cleaned_body, flags=re.DOTALL)

        # Replace only the first task_prompt body to avoid touching other content.
        return msg.replace(task_body, cleaned_body, 1)

    def _get_task_prompt_after_reset(self):
        '''env reset后，从side_channel获取task_prompt
        '''
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        assert len(terminal_steps.agent_id_to_index) == 0
        assert len(decision_steps) == self.env_num

        task_prompt = {}
        for agent_id, unity_id in decision_steps.agent_id_to_index.items():
            step_msgs = self.code_act_channels[unity_id].get_step_msgs()
            self.code_act_channels[unity_id].clear_step_msgs()

            # env reset后，env可能会有两次msg（reload scene后，触发了两次OnEpisodeBegin，具体逻辑还没细看），取最后一次作为task_prompt
            assert len(step_msgs) in (1, 2, 3), 'env reset后，code channel应只收到1-3个(相同的)msg，即task_prompt'
            assert '<task_prompt>' in step_msgs[-1], 'env reset后，code channel收到的msg应为task_prompt'
            # 注意，这里是unity_id对应channel，外部会转换为agent_id
            filtered_msg = self._filter_task_prompt_language(step_msgs[-1])
            task_prompt[unity_id] = filtered_msg

        return task_prompt

    def close_unity_env(self):
        env = self.env
        self.env = None
        if env is None:
            return
        try:
            env.close()
        except (UnityCommunicationException, UnityTimeOutException) as e:
            print(f"[EnvWrapper.close_unity_env] Ignoring Unity close error: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"[EnvWrapper.close_unity_env] Ignoring env close error: {type(e).__name__}: {e}")

    def close(self):
        self.close_unity_env()


if __name__ == '__main__':


    ########################## custom #############################
    env_path = os.environ.get('AGENTARK_ENV_PATH')
    mod_path = os.environ.get('AGENTARK_MOD_PATH')
    # after exec one action, human check steps(unit: env steps)
    human_check_steps = 0 # n * env_step_time
    ########################## custom #############################

    task_type = 'RLTask' # RLTask, Create

    rl_task_cfg = {
        'env_path': env_path,
        'mod_path': mod_path,
        # 可选：RLTask, Create。 若是rltask，会读取指定mod中的task_prompt.txt，否则需要指定task_prompt
        'task_type': task_type,
        # 'task_prompt': task_prompt,
        # 'human_check_steps': human_check_steps,
    }
    create_task_cfg = {
        'env_path': env_path,
        'mod_path': mod_path,
        # 可选：RLTask, Create。 若是rltask，会读取指定mod中的task_prompt.txt，否则需要指定task_prompt
        'task_type': task_type,
        # 'task_prompt': task_prompt,
        'human_check_steps': human_check_steps,
    }

    # func_action_mode_act = '<params>{"forceScale":3.0}</params>'
    func_action_mode_act = '<params>{"plan":"U7,L7"}</params>'

    cfg = rl_task_cfg if task_type == 'RLTask' else create_task_cfg

    env = EnvWrapper(cfg)

    from agent_ark.agent.rule_agent import RuleAgent
    agent = RuleAgent()

    for i in range(10):
        print(i)
        obs, info = env.reset()
        done = {'__all__': False}
        for _ in range(10):

            if env.env_info_mgr.env_config['action_mode'] == 'func':
                code_act = {k: func_action_mode_act for k in obs.keys()}
            else:
                code_act = agent.forward(obs)

            next_obs, rew, done, info = env.step(code_act)

            if done['__all__']:
                print('episode {} done'.format(i))
                break

            obs = next_obs

    env.close()
