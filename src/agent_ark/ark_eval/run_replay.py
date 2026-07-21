from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent_ark.agent.base_agent import BaseAgent
from agent_ark.ark_env import ArkEnv, ensure_runtime_pool_range, resolve_runtime_sandbox_cfg

from .run_api_agent import (
    JsonlWriter,
    _coerce_bool,
    build_eval_hook_manager,
    evaluate_case,
    load_eval_config,
    print_summary,
    write_eval_result,
    _hook_keep_open_on_end,
)


EnvFactory = Callable[[Dict[str, Any]], Any]


def _jsonl_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Replay records file not found: {path}')

    with p.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except Exception as exc:
                raise ValueError(f'Invalid JSONL record at {path}:{line_no}: {exc}') from exc
            if not isinstance(record, dict):
                raise ValueError(f'Replay record at {path}:{line_no} must be a JSON object')
            record = dict(record)
            record['_replay_source_path'] = str(p)
            record['_replay_source_line'] = int(line_no)
            record['_replay_record_index'] = len(records)
            records.append(record)
    return records


def _coerce_int_set(value: Any, *, context: str) -> Optional[set[int]]:
    if value is None or value == '':
        return None
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, str):
        return {int(part.strip()) for part in value.split(',') if part.strip()}
    if isinstance(value, (list, tuple)):
        return {int(item) for item in value}
    raise ValueError(f'{context} must be an integer, comma-separated string, or list of integers')


def _record_field(record: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record and record.get(name) not in (None, ''):
            return record.get(name)
    return None


def select_replay_records(records: List[Dict[str, Any]], replay_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = dict(replay_cfg or {})
    indices = _coerce_int_set(cfg.get('record_indices', cfg.get('indices', cfg.get('record_index', cfg.get('index')))), context='replay.record_indices')
    source_lines = _coerce_int_set(cfg.get('record_lines', cfg.get('lines')), context='replay.record_lines')
    status = cfg.get('status', 'ok')
    if status is not None:
        status = str(status)

    def matches(record: Dict[str, Any]) -> bool:
        if indices is not None and int(record.get('_replay_record_index', -1)) not in indices:
            return False
        if source_lines is not None and int(record.get('_replay_source_line', -1)) not in source_lines:
            return False
        if status and str(record.get('status', '')) != status:
            return False

        filters = {
            'case_id': ('case_id',),
            'job_id': ('job_id',),
            'model_name': ('model_name',),
            'task_name': ('requested_task_name', 'actual_task_name'),
            'group_seed': ('requested_group_seed', 'actual_group_seed', 'actual_rollout_group_seed'),
            'env_id': ('requested_env_id', 'actual_env_id'),
        }
        for cfg_key, field_names in filters.items():
            expected = cfg.get(cfg_key, None)
            if expected is None or expected == '':
                continue
            actual = _record_field(record, *field_names)
            if str(actual) != str(expected):
                return False
        return True

    selected = [record for record in records if matches(record)]
    limit = cfg.get('limit', None)
    if limit not in (None, ''):
        selected = selected[:max(0, int(limit))]
    return selected


def _single_saved_agent_id(step: Dict[str, Any], key: str) -> Optional[str]:
    mapping = step.get(key, {}) if isinstance(step, dict) else {}
    if not isinstance(mapping, dict) or len(mapping) != 1:
        return None
    return next(iter(mapping.keys()))


def _saved_agent_payload(step: Dict[str, Any], key: str, agent_id: Any) -> Dict[str, Any]:
    mapping = step.get(key, {}) if isinstance(step, dict) else {}
    if not isinstance(mapping, dict):
        return {}
    str_id = str(agent_id)
    if isinstance(mapping.get(str_id), dict):
        return mapping[str_id]
    single_id = _single_saved_agent_id(step, key)
    if single_id is not None and isinstance(mapping.get(single_id), dict):
        return mapping[single_id]
    return {}


def extract_replay_action(step: Dict[str, Any], agent_id: Any = 0) -> Optional[str]:
    response_payload = _saved_agent_payload(step, 'raw_response_by_agent', agent_id)
    for key in ('action_extracted', 'action', 'assistant_raw'):
        value = response_payload.get(key)
        if isinstance(value, str) and value:
            return value
    preview = step.get('action_preview') if isinstance(step, dict) else None
    return preview if isinstance(preview, str) and preview else None


def replay_case_from_record(record: Dict[str, Any], replay_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(replay_cfg or {})
    task_name = _record_field(record, 'requested_task_name', 'actual_task_name')
    group_seed = _record_field(record, 'requested_group_seed', 'actual_rollout_group_seed', 'actual_group_seed')
    if task_name in (None, ''):
        raise ValueError('Replay record is missing requested_task_name/actual_task_name')
    if group_seed in (None, ''):
        raise ValueError('Replay record is missing requested_group_seed/actual_group_seed')

    case = {
        'case_id': str(record.get('case_id') or f"replay-{int(record.get('_replay_record_index', 0)):04d}"),
        'task_name': str(task_name),
        'group_seed': int(group_seed),
    }
    env_id = _record_field(record, 'requested_env_id', 'actual_env_id')
    if env_id is not None:
        case['env_id'] = int(env_id)

    use_record_max_attempts = _coerce_bool(cfg.get('use_record_max_attempts', True), default=True)
    if use_record_max_attempts and record.get('max_attempts') is not None:
        case['max_attempts'] = int(record.get('max_attempts'))

    return case


class ReplayAgent(BaseAgent):
    def __init__(
        self,
        record: Dict[str, Any],
        *,
        name: Optional[str] = None,
        step_delay_s: float = 0.0,
    ):
        super().__init__(name or str(record.get('model_name') or 'replay-agent'))
        self.record = deepcopy(record)
        self.steps = list(record.get('steps', []) or [])
        self.step_delay_s = max(0.0, float(step_delay_s or 0.0))
        self.turn_index = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self.turn_index = 0

    def build_request_messages(self, obs: Dict[int, dict]) -> Dict[int, Optional[List[dict]]]:
        step = self._current_step()
        requests: Dict[int, Optional[List[dict]]] = {}
        for agent_idx in (obs or {}).keys():
            payload = _saved_agent_payload(step, 'raw_request_by_agent', agent_idx)
            messages = payload.get('request_messages')
            requests[int(agent_idx)] = deepcopy(messages) if isinstance(messages, list) else None
        return requests

    def forward_with_trace(
        self,
        obs: Dict[int, dict],
    ) -> tuple[Dict[int, Dict[str, Optional[str]]], Dict[int, Dict[str, Any]]]:
        step = self._current_step()
        responses: Dict[int, Dict[str, Optional[str]]] = {}
        trace_by_agent: Dict[int, Dict[str, Any]] = {}
        for agent_idx, obs_dict in (obs or {}).items():
            if isinstance(obs_dict, dict) and obs_dict.get('skip_infer'):
                responses[int(agent_idx)] = {'action': None, 'assistant': None}
                trace_by_agent[int(agent_idx)] = {
                    'skipped': True,
                    'assistant_raw': None,
                    'action_extracted': None,
                }
                continue

            response_payload = _saved_agent_payload(step, 'raw_response_by_agent', agent_idx)
            action_text = extract_replay_action(step, agent_idx)
            if action_text is None:
                raise RuntimeError(
                    f"Replay record has no saved action for turn_index={self.turn_index} agent_id={agent_idx}"
                )
            assistant_raw = response_payload.get('assistant_raw')
            if not isinstance(assistant_raw, str) or not assistant_raw:
                assistant_raw = action_text

            responses[int(agent_idx)] = {'action': action_text, 'assistant': assistant_raw}
            trace_by_agent[int(agent_idx)] = {
                'assistant_raw': assistant_raw,
                'action_extracted': action_text,
                'replay': True,
                'source_turn_index': step.get('turn_index', self.turn_index),
            }

        self.turn_index += 1
        if self.step_delay_s > 0:
            time.sleep(self.step_delay_s)
        return responses, trace_by_agent

    def forward_with_details(self, obs: Dict[int, dict]) -> Dict[int, Dict[str, Optional[str]]]:
        responses, _ = self.forward_with_trace(obs)
        return responses

    def forward(self, obs: Dict[int, dict]) -> Dict[int, Optional[str]]:
        return {
            agent_idx: payload.get('action') if isinstance(payload, dict) else None
            for agent_idx, payload in self.forward_with_details(obs).items()
        }

    def _current_step(self) -> Dict[str, Any]:
        if self.turn_index >= len(self.steps):
            raise RuntimeError(
                f'Replay record exhausted saved steps at turn_index={self.turn_index}; '
                f'available_steps={len(self.steps)}'
            )
        step = self.steps[self.turn_index]
        return step if isinstance(step, dict) else {}


def build_replay_model_runtime(record: Dict[str, Any], replay_cfg: Dict[str, Any]) -> Dict[str, Any]:
    agent = ReplayAgent(
        record,
        name=str(record.get('model_name') or 'replay-agent'),
        step_delay_s=float(replay_cfg.get('step_delay_s', replay_cfg.get('action_delay_s', 0.0)) or 0.0),
    )
    return {
        'name': agent.name,
        'model': record.get('model', 'replay'),
        'provider': 'replay',
        'base_url': record.get('base_url'),
        'api_key_env': None,
        'temperature': None,
        'agent': agent,
        'replay': True,
        'source_model_name': record.get('model_name'),
    }


def compare_replay_result(source_record: Dict[str, Any], replay_result: Dict[str, Any]) -> Dict[str, Any]:
    def close(left: Any, right: Any) -> bool:
        try:
            return abs(float(left) - float(right)) <= 1e-6
        except Exception:
            return left == right

    checks = {
        'turns_match': int(source_record.get('turns', -1)) == int(replay_result.get('turns', -2)),
        'score_reward_match': close(source_record.get('score_reward'), replay_result.get('score_reward')),
        'total_reward_match': close(source_record.get('total_reward'), replay_result.get('total_reward')),
        'rollout_success_match': bool(source_record.get('rollout_success')) == bool(replay_result.get('rollout_success')),
        'rollout_truncated_match': bool(source_record.get('rollout_truncated')) == bool(replay_result.get('rollout_truncated')),
    }
    return {
        'source_path': source_record.get('_replay_source_path'),
        'source_line': source_record.get('_replay_source_line'),
        'record_index': source_record.get('_replay_record_index'),
        'source_model_name': source_record.get('model_name'),
        'source_case_id': source_record.get('case_id'),
        'source_turns': source_record.get('turns'),
        'source_score_reward': source_record.get('score_reward'),
        'source_total_reward': source_record.get('total_reward'),
        'checks': checks,
        'match': all(checks.values()),
    }


def replay_record(
    *,
    env: Any,
    record: Dict[str, Any],
    replay_cfg: Dict[str, Any],
    hook_manager: Any = None,
    hooks_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    case = replay_case_from_record(record, replay_cfg)
    model_runtime = build_replay_model_runtime(record, replay_cfg)
    result = evaluate_case(
        env=env,
        model_runtime=model_runtime,
        case=case,
        hook_manager=hook_manager,
        hooks_cfg=hooks_cfg,
    )
    result['provider'] = 'replay'
    result['source_model_name'] = record.get('model_name')
    result['replay'] = compare_replay_result(record, result)
    if _coerce_bool(replay_cfg.get('require_match', False), default=False) and not result['replay']['match']:
        raise RuntimeError(f"Replay result mismatch for case_id={case['case_id']}: {result['replay']['checks']}")
    return result


def run_replay(
    cfg: Dict[str, Any],
    *,
    config_path: Optional[str] = None,
    env_factory: Optional[EnvFactory] = None,
    hook_manager: Any = None,
) -> List[Dict[str, Any]]:
    env_cfg = dict(cfg.get('env_cfg', {}) or {})
    replay_cfg = dict(cfg.get('replay', cfg.get('eval', {})) or {})
    hooks_cfg = dict(cfg.get('hooks', {}) or {})

    records_path = replay_cfg.get('records_path', replay_cfg.get('record_path', replay_cfg.get('input_path')))
    if not records_path:
        raise ValueError('Replay config requires replay.records_path')

    records = select_replay_records(_jsonl_records(str(records_path)), replay_cfg)
    if not records:
        raise ValueError(f'No replay records matched filters in {records_path}')

    ensure_runtime_pool_range(
        env_cfg,
        worker_index_base=int(env_cfg.get('worker_index', 0) or 0),
        worker_count=1,
    )
    env_cfg = resolve_runtime_sandbox_cfg(env_cfg)
    factory = env_factory or ArkEnv

    if hook_manager is None:
        hook_manager, _ = build_eval_hook_manager(hooks_cfg)
    writer = None
    output_path = replay_cfg.get('output_path', None)
    if output_path:
        writer = JsonlWriter(str(output_path), append=_coerce_bool(replay_cfg.get('append', False), default=False))

    print(f"Loaded {len(records)} replay records from {records_path}")
    if writer is not None:
        print(f"Replay results will be written to {writer.path}")

    results: List[Dict[str, Any]] = []
    stop_on_error = _coerce_bool(replay_cfg.get('stop_on_error', True), default=True)
    keep_open_on_end = _hook_keep_open_on_end(hooks_cfg)

    hook_manager.start()
    hook_manager.emit(
        'run_start',
        {
            'config_path': config_path,
            'record_count': len(records),
            'records_path': str(records_path),
        },
        source='run_replay',
    )
    completed = False
    try:
        for record in records:
            case = replay_case_from_record(record, replay_cfg)
            print(
                f"\n[replay {record.get('_replay_record_index')}] "
                f"line={record.get('_replay_source_line')} case={case['case_id']} "
                f"task={case['task_name']} group_seed={case['group_seed']} "
                f"model={record.get('model_name')}"
            )
            hook_manager.emit(
                'replay_record_start',
                {
                    'record_index': record.get('_replay_record_index'),
                    'source_line': record.get('_replay_source_line'),
                    'case': case,
                    'model_name': record.get('model_name'),
                    'turns': record.get('turns'),
                },
                source='run_replay',
            )

            runtime_env_cfg = deepcopy(env_cfg)
            runtime_env_cfg['hook_manager'] = hook_manager
            runtime_env_cfg['hooks'] = hooks_cfg
            env = factory(runtime_env_cfg)
            try:
                result = replay_record(
                    env=env,
                    record=record,
                    replay_cfg=replay_cfg,
                    hook_manager=hook_manager,
                    hooks_cfg=hooks_cfg,
                )
            except Exception as exc:
                result = {
                    'status': 'error',
                    'case_id': case.get('case_id'),
                    'model_name': record.get('model_name'),
                    'provider': 'replay',
                    'requested_task_name': case.get('task_name'),
                    'requested_group_seed': case.get('group_seed'),
                    'requested_env_id': case.get('env_id'),
                    'source_record_path': record.get('_replay_source_path'),
                    'source_record_line': record.get('_replay_source_line'),
                    'error_type': type(exc).__name__,
                    'error': str(exc),
                }
                results.append(result)
                if writer is not None:
                    writer.write(result)
                hook_manager.emit('error', result, source='run_replay')
                print(f"[replay {record.get('_replay_record_index')}] error: {type(exc).__name__}: {exc}")
                if stop_on_error:
                    raise
                continue
            finally:
                env.close()

            results.append(result)
            if writer is not None:
                write_eval_result(writer, result, replace_existing_by_resume_key=False)
            hook_manager.emit('replay_record_end', result, source='run_replay')
            print(
                f"[replay {record.get('_replay_record_index')}] done turns={result['turns']} "
                f"score_reward={result.get('score_reward')} total_reward={result.get('total_reward')} "
                f"match={result.get('replay', {}).get('match')}"
            )

        hook_manager.emit('run_end', {'result_count': len(results)}, source='run_replay')
        completed = True
    finally:
        if not (completed and keep_open_on_end and hook_manager.enabled):
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
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Replay saved AgentArk eval records without LLM inference')
    parser.add_argument(
        '--config',
        type=str,
        default='config/ark_env/replay.example.yaml',
        help='Path to replay config (.yaml/.yml/.json)',
    )
    parser.add_argument('--records', type=str, default=None, help='Override replay.records_path')
    parser.add_argument('--index', type=int, default=None, help='Override replay.record_index')
    parser.add_argument('--line', type=int, default=None, help='Override replay.record_lines')
    parser.add_argument('--model-name', type=str, default=None, help='Override replay.model_name filter')
    parser.add_argument('--case-id', type=str, default=None, help='Override replay.case_id filter')
    parser.add_argument('--limit', type=int, default=None, help='Override replay.limit')
    return parser


def main(args: argparse.Namespace) -> None:
    cfg = load_eval_config(args.config)
    replay_cfg = dict(cfg.get('replay', cfg.get('eval', {})) or {})
    if args.records is not None:
        replay_cfg['records_path'] = args.records
    if args.index is not None:
        replay_cfg['record_index'] = args.index
    if args.line is not None:
        replay_cfg['record_lines'] = [args.line]
    if args.model_name is not None:
        replay_cfg['model_name'] = args.model_name
    if args.case_id is not None:
        replay_cfg['case_id'] = args.case_id
    if args.limit is not None:
        replay_cfg['limit'] = args.limit
    cfg['replay'] = replay_cfg
    run_replay(cfg, config_path=args.config)


if __name__ == '__main__':
    main(build_parser().parse_args())
