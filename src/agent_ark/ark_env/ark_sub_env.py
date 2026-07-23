import argparse
import json
import os
from pathlib import Path
import uuid
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityCommunicationException, UnityTimeOutException, UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.raw_bytes_channel import RawBytesChannel

from agent_ark.ark_env.direct_env import EnvInfoManager, EnvWrapper, _SharedXvfbManager


class ArkSubEnv(object):
    """Lightweight Unity env wrapper used by ArkEnv and local debug flows."""

    _unity_start_lock = EnvWrapper._unity_start_lock
    _coerce_bool = staticmethod(EnvWrapper._coerce_bool)
    _to_csharp_literal = staticmethod(EnvWrapper._to_csharp_literal)
    _can_bind_tcp_port = staticmethod(EnvWrapper._can_bind_tcp_port)

    _get_action_mode = EnvWrapper._get_action_mode
    _get_current_resolution = EnvWrapper._get_current_resolution
    _get_no_graphics = EnvWrapper._get_no_graphics
    _get_unity_start_config = EnvWrapper._get_unity_start_config
    _get_unity_start_serialize = EnvWrapper._get_unity_start_serialize
    _get_virtual_display_config = EnvWrapper._get_virtual_display_config
    _should_recreate_env = EnvWrapper._should_recreate_env
    _get_port_alloc_config = EnvWrapper._get_port_alloc_config
    _resolve_unity_base_port = EnvWrapper._resolve_unity_base_port
    _refresh_code_wrappers_from_task_prompt = EnvWrapper._refresh_code_wrappers_from_task_prompt
    _render_func_wrapper = EnvWrapper._render_func_wrapper
    _render_func_code_actions = EnvWrapper._render_func_code_actions
    _format_func_render_error = staticmethod(EnvWrapper._format_func_render_error)
    _merge_step_message_parts = staticmethod(EnvWrapper._merge_step_message_parts)
    _step_message_indicates_script_error = staticmethod(EnvWrapper._step_message_indicates_script_error)
    _positive_int_or_none = staticmethod(EnvWrapper._positive_int_or_none)
    _build_rollout_budget_prompt = classmethod(EnvWrapper._build_rollout_budget_prompt.__func__)
    _build_llm_visible_prompt = EnvWrapper._build_llm_visible_prompt
    _build_reset_obs_payload = EnvWrapper._build_reset_obs_payload
    _obs_mode_is_video = staticmethod(EnvWrapper._obs_mode_is_video)
    _coerce_non_negative_int = staticmethod(EnvWrapper._coerce_non_negative_int)
    _coerce_positive_float_or_none = staticmethod(EnvWrapper._coerce_positive_float_or_none)
    _get_initial_observation_cfg = staticmethod(EnvWrapper._get_initial_observation_cfg)
    _validate_initial_observation_cfg = staticmethod(EnvWrapper._validate_initial_observation_cfg)
    _build_unity_env_params_payload = staticmethod(EnvWrapper._build_unity_env_params_payload)
    _decode_image_payload_to_pil = staticmethod(EnvWrapper._decode_image_payload_to_pil)
    _merge_video_pil_frames = staticmethod(EnvWrapper._merge_video_pil_frames)
    _attach_image_payloads_to_obs = EnvWrapper._attach_image_payloads_to_obs
    _get_agent_visual_observations = EnvWrapper._get_agent_visual_observations
    _clear_image_channels = EnvWrapper._clear_image_channels
    _clear_code_channel_step_msgs = EnvWrapper._clear_code_channel_step_msgs
    _initial_observation_frame_counts = staticmethod(EnvWrapper._initial_observation_frame_counts)
    _build_obs_from_decision_steps = EnvWrapper._build_obs_from_decision_steps
    _apply_initial_observation_warmup = EnvWrapper._apply_initial_observation_warmup
    _get_agent_id_map = EnvWrapper._get_agent_id_map
    _preferred_language_tag = EnvWrapper._preferred_language_tag
    _filter_task_prompt_language = EnvWrapper._filter_task_prompt_language
    _get_task_prompt_after_reset = EnvWrapper._get_task_prompt_after_reset
    get_empty_obs = EnvWrapper.get_empty_obs
    get_code_act_channels = EnvWrapper.get_code_act_channels
    get_image_channels = EnvWrapper.get_image_channels
    send_code_act = EnvWrapper.send_code_act
    post_process_obs = EnvWrapper.post_process_obs
    soft_reset = EnvWrapper.soft_reset
    close_unity_env = EnvWrapper.close_unity_env
    close = EnvWrapper.close

    def __init__(self, cfg):
        self.cfg = EnvWrapper._resolve_runtime_cfg(cfg)

        self.env_info_mgr = EnvInfoManager(self.cfg)
        self.env_info_mgr.reset()

        self.env = None
        self._last_env_resolution = None
        self._unity_base_port = None
        self._code_wrapper_by_unity_id = {}
        self._tool_manifest_by_unity_id = {}
        self.ml_unity_id_map = {}
        self.episode_agent_id_to_index = {}
        self.agent_done_dict = {}

    def start_unity_env(self):
        self.engine_channel = EngineConfigurationChannel()
        self.env_channel = EnvironmentParametersChannel()
        self.raw_byte_channel = RawBytesChannel(uuid.UUID("621f0a70-4f87-11ea-a6bf-999999999999"))

        self.env_num = self.env_info_mgr.env_config.get('num_parallel_envs', 1)
        self.code_act_channels = self.get_code_act_channels()
        self.image_channels = self.get_image_channels()

        side_channels = [
            self.engine_channel,
            self.env_channel,
            self.raw_byte_channel,
            *self.code_act_channels,
            *self.image_channels,
        ]

        self.env_info_mgr.set_engine_para(self.engine_channel)
        self.env_info_mgr.set_env_para(self.env_channel)

        alloc = self._get_port_alloc_config()
        vd_cfg = self._get_virtual_display_config()
        if vd_cfg['enabled']:
            display = _SharedXvfbManager.ensure_started(vd_cfg)
            print(f"[ArkSubEnv] virtual_display enabled, using DISPLAY={display}")

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
                        f"[ArkSubEnv] UnityEnvironment started at base_port={self._unity_base_port}, "
                        f"no_graphics={no_graphics}"
                    )
                    last_error = None
                    break
                except UnityWorkerInUseException as e:
                    self.env = None
                    last_error = e
                    print(
                        f"[ArkSubEnv] Unity worker in use at base_port={candidate_port}; retrying next port. "
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
                        f"[ArkSubEnv] Unity start failed at base_port={candidate_port} "
                        f"after timeout_wait={start_cfg['timeout_wait_s']}s; "
                        f"startup_failure={startup_failures}: {type(e).__name__}: {e}"
                    )
                    if start_cfg['max_attempts'] and startup_failures >= start_cfg['max_attempts']:
                        break
                    if start_cfg['retry_wait_s'] > 0:
                        import time
                        time.sleep(start_cfg['retry_wait_s'])
                    continue
                except Exception as e:
                    self.env = None
                    last_error = e
                    startup_failures += 1
                    print(
                        f"[ArkSubEnv] Unity start raised {type(e).__name__} at base_port={candidate_port}; "
                        f"startup_failure={startup_failures}: {e}"
                    )
                    if start_cfg['max_attempts'] and startup_failures >= start_cfg['max_attempts']:
                        break
                    if start_cfg['retry_wait_s'] > 0:
                        import time
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

        self.send_code_act(agent_id=-1, code_act={})
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.env_spec = self.env.behavior_specs[self.behavior_name]

        self.env_info_mgr.task_prompt = self._get_task_prompt_after_reset()
        self._refresh_code_wrappers_from_task_prompt()

    def _should_recreate_env(self):
        if self.env is None:
            return True

        env_cfg = self.env_info_mgr.env_config or {}
        wrapper_cfg = env_cfg.get('env_wrapper_cfg', {}) if isinstance(env_cfg, dict) else {}
        raw = self.cfg.get('recreate_unity_on_resolution_change', None)
        if raw is None:
            raw = env_cfg.get('recreate_unity_on_resolution_change', None)
        if raw is None:
            raw = wrapper_cfg.get('recreate_unity_on_resolution_change', True)

        if not self._coerce_bool(raw, default=False):
            return False
        return self._get_current_resolution() != self._last_env_resolution

    def _prepare_unity_for_reset(self):
        if self._should_recreate_env():
            if self.env is not None:
                self.close_unity_env()
            self.start_unity_env()
        else:
            self.env_info_mgr.set_engine_para(self.engine_channel)
            self.env_info_mgr.set_env_para(self.env_channel)
            self._last_env_resolution = self._get_current_resolution()

    def _reset_started_unity_episode(self):
        _, info = self.soft_reset()

        dummy_act = self.env_spec.action_spec.empty_action(self.env_num)
        self.env.set_actions(self.behavior_name, dummy_act)

        self._send_env_params_via_raw_bytes()
        self.raw_byte_channel.send_raw_data(b'reload_scene')

        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        env_cfg = self.env_info_mgr.env_config or {}
        task_params = env_cfg.get('task_params', None)
        if task_params is not None:
            try:
                payload_bytes = ("[task_params]" + json.dumps(task_params)).encode('utf-8')
                self.raw_byte_channel.send_raw_data(payload_bytes)
            except Exception as e:
                print(f"[ArkSubEnv.reset] Failed to send task_params via raw_bytes: {e}")

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

        assert len(terminal_steps.agent_id_to_index) == 0
        self.episode_agent_id_to_index = deepcopy(decision_steps.agent_id_to_index)
        self.agent_done_dict = {k: False for k in self.episode_agent_id_to_index}

        obs = {}
        for ml_id, unity_id in self.ml_unity_id_map.items():
            obs[ml_id] = unity_id_obs[unity_id]
            obs[ml_id]['vis'] = self._get_agent_visual_observations(decision_steps, ml_id)

        obs, info = self._apply_initial_observation_warmup(obs, info, current_init_prompt)

        obs = self.post_process_obs(obs)
        return obs, info

    def reset(
        self,
        task_name=None,
        group_seed=None,
        env_id=None,
    ):
        self.env_info_mgr.reset(task_name=task_name, group_seed=group_seed, env_id=env_id)

        for attempt_index in range(2):
            try:
                if attempt_index > 0:
                    self.close_unity_env()
                self._prepare_unity_for_reset()
                return self._reset_started_unity_episode()
            except (UnityTimeOutException, UnityCommunicationException) as e:
                print(
                    f"[ArkSubEnv.reset] Unity reset failed on attempt {attempt_index + 1}/2; "
                    f"restarting Unity env: {type(e).__name__}: {e}"
                )
                self.close_unity_env()
                if attempt_index == 0:
                    continue
                raise

        raise RuntimeError("ArkSubEnv.reset exhausted retry attempts")

    def _send_env_params_via_raw_bytes(self):
        env_cfg = self.env_info_mgr.env_config or {}
        payload = self._build_unity_env_params_payload(env_cfg)

        if not payload:
            return

        try:
            serialized = json.dumps(payload)
            self.raw_byte_channel.send_raw_data(f"[env_params]{serialized}".encode('utf-8'))
        except Exception as e:
            print(f"[ArkSubEnv] Failed to send env_params via raw_bytes: {e}")

    def step(self, code_act, info: Optional[Dict[str, Any]] = None):
        step_info = dict(info) if isinstance(info, dict) else {}

        dummy_act = self.env_spec.action_spec.empty_action(len(code_act))
        self.env.set_actions(self.behavior_name, dummy_act)

        exec_code_act = {k: v for k, v in code_act.items() if v is not None}

        exec_code_act, func_render_errors = self._render_func_code_actions(exec_code_act, log_prefix='ArkSubEnv.step')

        self.send_code_act(agent_id=list(exec_code_act.keys()), code_act=exec_code_act)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        next_obs_list = decision_steps.agent_id.tolist() + terminal_steps.agent_id.tolist()
        next_obs = self.get_empty_obs(agent_id_list=next_obs_list)

        done = {'__all__': False}
        truncated = {}
        reward = {}

        for ml_id in terminal_steps.agent_id_to_index.keys():
            if ml_id in self.agent_done_dict and self.agent_done_dict[ml_id]:
                next_obs[ml_id]['skip_infer'] = True
                continue
            done[ml_id] = True
            self.agent_done_dict[ml_id] = True
            reward[ml_id] = terminal_steps.reward[terminal_steps.agent_id_to_index[ml_id]]
            next_obs[ml_id]['vis'] = self._get_agent_visual_observations(terminal_steps, ml_id)

            index = terminal_steps.agent_id_to_index[ml_id]
            if terminal_steps.interrupted[index]:
                truncated[ml_id] = True

        for ml_id in decision_steps.agent_id_to_index.keys():
            if ml_id in self.agent_done_dict and self.agent_done_dict[ml_id]:
                next_obs[ml_id]['skip_infer'] = True
                continue
            done[ml_id] = False
            assert not self.agent_done_dict[ml_id]
            reward[ml_id] = decision_steps.reward[decision_steps.agent_id_to_index[ml_id]]
            next_obs[ml_id]['vis'] = self._get_agent_visual_observations(decision_steps, ml_id)

        step_msgs = {}
        for ml_id in next_obs.keys():
            if next_obs[ml_id]['skip_infer']:
                continue
            unity_id = self.ml_unity_id_map[ml_id]
            channel_step_msgs = self.code_act_channels[unity_id].get_step_msgs()
            if ml_id in func_render_errors:
                step_msgs[ml_id] = self._merge_step_message_parts(func_render_errors[ml_id], channel_step_msgs)
            else:
                step_msgs[ml_id] = channel_step_msgs

        for ml_id in next_obs.keys():
            if next_obs[ml_id]['skip_infer']:
                continue
            if len(step_msgs[ml_id]) > 0:
                next_obs[ml_id]['step_msg'] = step_msgs[ml_id]
                if not self.agent_done_dict[ml_id]:
                    if (
                        self.env_info_mgr.env_config.get('done_on_script_error', False)
                        and self._step_message_indicates_script_error(step_msgs[ml_id])
                    ):
                        done[ml_id] = True
                        self.agent_done_dict[ml_id] = True
                        reward[ml_id] = -1.0
                        next_obs[ml_id]['skip_infer'] = True
            else:
                next_obs[ml_id]['step_msg'] = ''

        step_info['truncated'] = truncated
        if func_render_errors:
            step_info['func_render_errors'] = dict(func_render_errors)

        for channel in self.code_act_channels:
            channel.clear_step_msgs()

        video_payloads = {ch.agent_id: ch.get_and_clear() for ch in self.image_channels}
        self._attach_image_payloads_to_obs(next_obs, video_payloads)

        reward = {k: float(v) for k, v in reward.items()}
        done['__all__'] = all(self.agent_done_dict.values())

        next_obs = self.post_process_obs(next_obs)
        return next_obs, reward, done, step_info


def _preview_text(value, limit: int = 240) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + '...'


def _default_editor_mod_path() -> Optional[str]:
    candidates = []

    env_value = os.environ.get('AGENTARK_MOD_PATH') or os.environ.get('AGENT_ARK_MOD_PATH')
    if env_value:
        candidates.append(Path(_expand_path_value(env_value)))

    cwd = Path.cwd()
    project_names = ('AgentArkUnity', 'agentark-unity', 'unity')
    for root in (cwd, cwd.parent):
        for project_name in project_names:
            candidates.append(root / project_name / 'Assets' / 'Resources' / 'Mods')

    try:
        repo_root = Path(__file__).resolve().parents[4]
        for project_name in project_names:
            candidates.append(repo_root.parent / project_name / 'Assets' / 'Resources' / 'Mods')
    except IndexError:
        pass

    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return None


def _normalize_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ('none', 'null', '~'):
        return None
    return _expand_path_value(text)


def _expand_path_value(value: str) -> str:
    expanded = str(value)
    for _ in range(3):
        next_value = os.path.expandvars(expanded)
        if next_value == expanded:
            break
        expanded = next_value
    return os.path.expanduser(expanded)


def _unity_executable_base_name(executable: Path) -> str:
    name = executable.name
    lower = name.lower()
    for suffix in ('.x86_64', '.x86', '.exe'):
        if lower.endswith(suffix):
            return name[:-len(suffix)]
    return executable.stem


def _derive_mod_path_from_env_path(env_path: Optional[str]) -> Optional[str]:
    env_path = _normalize_optional_path(env_path)
    if not env_path:
        return None

    runtime_path = Path(env_path)
    if not runtime_path.exists():
        return None

    if runtime_path.is_dir():
        data_dirs = sorted(
            item for item in runtime_path.iterdir()
            if item.is_dir() and item.name.endswith('_Data')
        )
        for data_dir in data_dirs:
            mods_path = data_dir / 'Resources' / 'Mods'
            if mods_path.is_dir():
                return str(mods_path)
        return None

    executable = runtime_path
    data_dir = executable.parent / f'{_unity_executable_base_name(executable)}_Data'
    mods_path = data_dir / 'Resources' / 'Mods'
    if mods_path.is_dir():
        return str(mods_path)
    return None


def _resolve_cli_mod_path(mod_path: Optional[str], env_path: Optional[str]) -> str:
    mod_path = _normalize_optional_path(mod_path)
    env_path = _normalize_optional_path(env_path)
    if mod_path:
        return mod_path

    resolved = _derive_mod_path_from_env_path(env_path) or _default_editor_mod_path()
    if resolved:
        return resolved

    raise ValueError(
        'Could not resolve a Mods directory. Pass --mod-path explicitly, for example: '
        r'--mod-path <unity-project>\Assets\Resources\Mods'
    )


def _resolve_cli_max_steps(env_cfg: Optional[Dict[str, Any]], max_steps: Optional[int]) -> int:
    if max_steps is not None:
        if int(max_steps) <= 0:
            raise ValueError(f'--max-steps must be a positive integer, got {max_steps!r}')
        return int(max_steps)

    if isinstance(env_cfg, dict):
        resolved = EnvWrapper._positive_int_or_none(env_cfg.get('max_steps_per_attempt', None))
        if resolved is not None:
            return int(resolved)

    return 10


def _coerce_cli_action_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def _existing_cli_action_sequence_path(value: str) -> Optional[Path]:
    try:
        normalized = _normalize_optional_path(value) or value
        path = Path(normalized)
        return path if path.is_file() else None
    except (OSError, ValueError):
        return None


def _coerce_cli_action_sequence(value: Any, source: str) -> list[str]:
    if isinstance(value, dict) and 'actions' in value:
        value = value['actions']

    if not isinstance(value, list):
        raise ValueError(f'{source} must contain a JSON array, or a JSON object with an "actions" array')

    actions = [_coerce_cli_action_payload(item) for item in value]
    if not actions:
        raise ValueError(f'{source} must contain at least one action payload')
    return actions


def _load_cli_action_sequence_file(path: Path) -> list[str]:
    text = path.read_text(encoding='utf-8')
    suffix = path.suffix.lower()
    if suffix == '.json':
        return _coerce_cli_action_sequence(json.loads(text), str(path))

    actions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        if suffix == '.jsonl':
            try:
                actions.append(_coerce_cli_action_payload(json.loads(stripped)))
                continue
            except json.JSONDecodeError:
                pass

        actions.append(stripped)

    if not actions:
        raise ValueError(f'{path} must contain at least one action payload')
    return actions


def _resolve_cli_action_payloads(action: str, action_sequence: Optional[str]) -> list[str]:
    if action_sequence is None:
        return [_coerce_cli_action_payload(action)]

    stripped = action_sequence.strip()
    if not stripped:
        raise ValueError('--action-sequence must be a JSON array or an existing file path')

    sequence_path = _existing_cli_action_sequence_path(stripped)
    if sequence_path is not None:
        return _load_cli_action_sequence_file(sequence_path)

    if stripped.startswith('[') or stripped.startswith('{'):
        return _coerce_cli_action_sequence(json.loads(stripped), '--action-sequence')

    raise ValueError('--action-sequence must be a JSON array/object or an existing file path')


def _resolve_cli_action_step_count(
    default_steps: int,
    requested_steps: Optional[int],
    action_payloads: Sequence[str],
    has_action_sequence: bool,
    option_name: str,
) -> int:
    if requested_steps is not None:
        resolved = int(requested_steps)
        if resolved <= 0:
            raise ValueError(f'{option_name} must be a positive integer, got {requested_steps!r}')
        if has_action_sequence and resolved > len(action_payloads):
            raise ValueError(
                f'{option_name}={resolved} exceeds --action-sequence length {len(action_payloads)}; '
                f'provide more actions or lower {option_name}'
            )
        return resolved

    if has_action_sequence:
        return len(action_payloads)

    return int(default_steps)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Debug ArkSubEnv with a real Unity reset/step smoke test')
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
    parser.add_argument(
        '--base-port',
        type=int,
        default=None,
        help='Optional explicit Unity ML-Agents base port for local/multi-worktree development. '
             'Otherwise uses AGENTARK_EDITOR_BASE_PORT when set, then the existing Mods config.',
    )
    parser.add_argument('--task-type', default='RLTask', help='Task type passed to EnvInfoManager')
    parser.add_argument('--task-name', default=None, help='Task folder name or task identifier; omit to use config/default selection')
    parser.add_argument('--group-seed', type=int, default=123, help='Group seed used during reset')
    parser.add_argument('--env-id', type=int, default=0, help='Unity env_id used during reset')
    parser.add_argument(
        '--num-parallel-envs',
        type=int,
        default=1,
        help='Override num_parallel_envs for local smoke testing',
    )
    parser.add_argument(
        '--action',
        default='<tool_call>{"name":"ExecutePlan","arguments":{"plan":"U7,L7"}}</tool_call>',
        help='Action sent on each step to every active agent; in func mode prefer a <tool_call> payload, while code mode accepts a full C# script or <code> block',
    )
    parser.add_argument(
        '--action-sequence',
        default=None,
        help='Optional action trajectory. Pass an inline JSON array/object with actions, or a path to .json, .jsonl, or text file. Each item is sent on one external step.',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Maximum ArkSubEnv.step calls to run; defaults to env_config.max_steps_per_attempt, or 10 if unavailable',
    )
    parser.add_argument(
        '--skip-step',
        action='store_true',
        help='Only run reset and print initial observation summary',
    )
    args = parser.parse_args(argv)

    env_path = _normalize_optional_path(args.env_path)
    mod_path = _resolve_cli_mod_path(args.mod_path, env_path)
    action_payloads = _resolve_cli_action_payloads(args.action, args.action_sequence)

    cfg = {
        'env_path': env_path,
        'mod_path': mod_path,
        'task_type': args.task_type,
        'env_config_overrides': {
            'num_parallel_envs': args.num_parallel_envs,
        },
    }
    if args.base_port is not None:
        cfg['base_port'] = int(args.base_port)

    if cfg['env_path'] is None:
        print('[ArkSubEnv] env_path=None: connecting to the Unity Editor. Run this script, then press Play in Unity if it is waiting for a connection.')
    print(f'[ArkSubEnv] mod_path={cfg["mod_path"]}')

    env = ArkSubEnv(cfg)
    try:
        obs, info = env.reset(task_name=args.task_name, group_seed=args.group_seed, env_id=args.env_id)
        print('reset_ok=True')
        print(f'behavior_name={env.behavior_name}')
        print(f'task_name={env.env_info_mgr.env_config.get("task_name")}')
        print(f'group_seed={env.env_info_mgr.env_config.get("group_seed")}')
        print(f'action_mode={env.env_info_mgr.env_config.get("action_mode")}')
        print(f'obs_keys={sorted(obs.keys())}')
        print(f'info_keys={sorted(info.keys()) if isinstance(info, dict) else []}')

        if not obs:
            print('reset returned empty obs')
            return 1

        ml_id = sorted(obs.keys())[0]
        obs_item = obs[ml_id]
        unity_id = env.ml_unity_id_map.get(ml_id)
        wrapper = env._code_wrapper_by_unity_id.get(unity_id, '')
        print(f'active_ml_id={ml_id}')
        print(f'unity_id={unity_id}')
        print(f'step_msg_len={len(obs_item.get("step_msg", ""))}')
        print(f'vis_cam_count={len(obs_item.get("vis") or [])}')
        print(f'vis_frame_counts={[len(frames) if isinstance(frames, list) else 0 for frames in (obs_item.get("vis") or [])]}')
        print(f'initial_observation={info.get("initial_observation") if isinstance(info, dict) else None}')
        print(f'wrapper_preview={_preview_text(wrapper.replace(chr(10), " "))}')

        if args.skip_step:
            return 0

        max_steps = _resolve_cli_action_step_count(
            default_steps=_resolve_cli_max_steps(env.env_info_mgr.env_config, None),
            requested_steps=args.max_steps,
            action_payloads=action_payloads,
            has_action_sequence=args.action_sequence is not None,
            option_name='--max-steps',
        )
        print(f'max_steps={max_steps}')
        print(f'action_count={len(action_payloads)}')

        done = {'__all__': False}
        last_step_info = {}
        for step_idx in range(1, max_steps + 1):
            step_action = action_payloads[min(step_idx - 1, len(action_payloads) - 1)]
            active_actions = {
                active_ml_id: step_action
                for active_ml_id, obs_item in obs.items()
                if not (isinstance(obs_item, dict) and obs_item.get('skip_infer'))
            }
            if not active_actions:
                print(f'step={step_idx}')
                print('active_agent_ids=[]')
                print('no_active_agents=True')
                return 1

            next_obs, reward, done, step_info = env.step(active_actions)
            last_step_info = step_info if isinstance(step_info, dict) else {}

            preview_ml_id = next((candidate for candidate in sorted(active_actions.keys()) if candidate in next_obs), None)
            if preview_ml_id is None and next_obs:
                preview_ml_id = sorted(next_obs.keys())[0]
            next_item = next_obs.get(preview_ml_id, {}) if preview_ml_id is not None else {}
            step_msg = next_item.get('step_msg') if isinstance(next_item, dict) else ''
            if isinstance(step_msg, list):
                step_msg_preview = ' | '.join(_preview_text(part, limit=120) for part in step_msg)
            else:
                step_msg_preview = _preview_text(step_msg)

            print('step_ok=True')
            print(f'step={step_idx}')
            print(f'action_agent_ids={sorted(active_actions.keys())}')
            print(f'action={step_action}')
            print(f'reward={reward}')
            print(f'done={done}')
            print(f'step_info_keys={sorted(last_step_info.keys())}')
            print(f'next_obs_keys={sorted(next_obs.keys())}')
            print(f'next_skip_infer={next_item.get("skip_infer") if isinstance(next_item, dict) else None}')
            print(f'next_vis_cam_count={len(next_item.get("vis") or []) if isinstance(next_item, dict) else 0}')
            print(f'next_step_msg_preview={step_msg_preview}')

            obs = next_obs
            if done.get('__all__', False):
                print('episode_done=True')
                return 0

        print('episode_done=False')
        print('step_limit_reached=True')
        print(f'truncated={last_step_info.get("truncated", {})}')
        return 0
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
