from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from agent_ark.ark_env import ArkEnv, ensure_runtime_pool_range, resolve_runtime_sandbox_cfg
from agent_ark.interaction import HookManager
from agent_ark.ark_eval.run_api_agent import (
    JsonlWriter,
    _coerce_bool,
    _hook_keep_open_on_end,
    _to_jsonable,
    apply_trajectory_load_to_cases,
    build_error_result,
    build_eval_cases,
    build_eval_hook_manager,
    build_model_runtimes,
    build_trajectory_writer,
    evaluate_case,
    load_eval_config,
    maybe_save_eval_trajectory,
    print_summary,
)
from .trajectory_io import TrajectoryJsonlWriter


JobRunner = Callable[[Dict[str, Any], int, int], Dict[str, Any]]
ResultHandler = Callable[[Dict[str, Any]], None]
ResumeKey = Tuple[str, str, str, str]


def _coerce_positive_int(value: Any, *, context: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{context} must be a positive integer') from exc
    if out <= 0:
        raise ValueError(f'{context} must be a positive integer, got {value!r}')
    return out


def resolve_max_parallel_envs(eval_cfg: Dict[str, Any], *, job_count: int) -> int:
    raw = None
    for key in ('max_parallel_envs', 'max_concurrent_envs', 'env_limit', 'parallel_envs'):
        if key in (eval_cfg or {}):
            raw = eval_cfg.get(key)
            break
    if raw is None:
        raw = 1

    limit = _coerce_positive_int(raw, context='eval.max_parallel_envs')
    if int(job_count) <= 0:
        raise ValueError('Parallel eval has no jobs to run')
    return min(limit, int(job_count))


def _model_stub(model_cfg: Dict[str, Any], model_index: int) -> Dict[str, Any]:
    return {
        'name': str(model_cfg.get('name', f'model-{model_index:02d}')).strip() or f'model-{model_index:02d}',
        'model': str(model_cfg.get('model', '')).strip(),
        'provider': str(model_cfg.get('provider', 'auto')).strip().lower() or 'auto',
        'base_url': model_cfg.get('base_url', None),
        'api_key_env': str(model_cfg.get('api_key_env', '')).strip() or None,
    }


def build_parallel_eval_jobs(cases: List[Dict[str, Any]], model_cfgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cases:
        raise ValueError('Parallel eval requires at least one case')
    if not model_cfgs:
        raise ValueError('Parallel eval requires at least one model under models')

    jobs: List[Dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        case_id = str(case.get('case_id', f'case-{case_index:04d}'))
        for model_index, model_cfg in enumerate(model_cfgs):
            model_stub = _model_stub(model_cfg, model_index)
            job_index = len(jobs)
            jobs.append({
                'job_index': job_index,
                'job_id': f'{case_id}::{model_stub["name"]}',
                'case_index': case_index,
                'model_index': model_index,
                'case': deepcopy(case),
                'model_cfg': deepcopy(model_cfg),
                'model_name': model_stub['name'],
            })
    return jobs


def _stable_resume_value(value: Any) -> str:
    if value is None:
        return ''
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _parallel_job_resume_key(job: Dict[str, Any]) -> ResumeKey:
    case = job.get('case', {}) if isinstance(job.get('case', {}), dict) else {}
    return (
        str(case.get('task_name', '') or ''),
        _stable_resume_value(case.get('group_seed', '')),
        _stable_resume_value(case.get('env_id', '')),
        str(job.get('model_name', '') or ''),
    )


def _parallel_result_resume_key(record: Dict[str, Any]) -> ResumeKey | None:
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


def load_existing_parallel_results(
    output_path: str,
    jobs: List[Dict[str, Any]],
) -> Dict[ResumeKey, Dict[str, Any]]:
    path = Path(output_path)
    if not path.exists():
        return {}

    expected_keys = {_parallel_job_resume_key(job) for job in jobs}
    existing: Dict[ResumeKey, Dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except Exception as exc:
                print(f'[run_parallel_api_eval] WARNING: ignoring malformed JSONL line {path}:{line_no}: {exc}')
                continue
            if not isinstance(record, dict):
                continue
            if record.get('status') != 'ok':
                continue
            key = _parallel_result_resume_key(record)
            if key is None or key not in expected_keys:
                continue
            existing[key] = record
    return existing


def filter_existing_parallel_jobs(
    jobs: List[Dict[str, Any]],
    output_path: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    existing_by_key = load_existing_parallel_results(output_path, jobs)
    if not existing_by_key:
        return jobs, []

    pending_jobs = [
        job for job in jobs
        if _parallel_job_resume_key(job) not in existing_by_key
    ]
    existing_results = [
        existing_by_key[_parallel_job_resume_key(job)]
        for job in jobs
        if _parallel_job_resume_key(job) in existing_by_key
    ]
    return pending_jobs, existing_results


def compact_parallel_results_file(output_path: str, jobs: List[Dict[str, Any]]) -> int:
    path = Path(output_path)
    if not path.exists():
        return 0

    expected_keys = {_parallel_job_resume_key(job) for job in jobs}
    entries: List[Dict[str, Any]] = []
    selected_by_key: Dict[ResumeKey, int] = {}

    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except Exception:
                entries.append({'kind': 'raw', 'line': line.rstrip('\n'), 'key': None})
                continue

            key = _parallel_result_resume_key(record) if isinstance(record, dict) else None
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


def _attach_job_metadata(
    result: Dict[str, Any],
    job: Dict[str, Any],
    *,
    worker_index: int,
    slot_index: int,
) -> Dict[str, Any]:
    out = dict(result or {})
    out.update({
        'job_id': job['job_id'],
        'job_index': int(job['job_index']),
        'case_index': int(job['case_index']),
        'model_index': int(job['model_index']),
        'worker_index': int(worker_index),
        'parallel_slot_index': int(slot_index),
        'parallel_eval': True,
    })
    return out


def _fallback_error_result(
    job: Dict[str, Any],
    error: BaseException,
    *,
    worker_index: int,
    slot_index: int,
) -> Dict[str, Any]:
    model_runtime = _model_stub(job.get('model_cfg', {}) or {}, int(job.get('model_index', 0) or 0))
    result = build_error_result(model_runtime, job['case'], error if isinstance(error, Exception) else RuntimeError(str(error)))
    result['traceback'] = traceback.format_exc(limit=8)
    return _attach_job_metadata(result, job, worker_index=worker_index, slot_index=slot_index)


def make_parallel_job_runner(
    *,
    env_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    hooks_cfg: Dict[str, Any],
    hook_manager: HookManager,
    trajectory_writer: TrajectoryJsonlWriter | None = None,
) -> JobRunner:
    base_env_cfg = deepcopy(env_cfg)
    eval_cfg = deepcopy(eval_cfg or {})
    hooks_cfg = deepcopy(hooks_cfg or {})

    def _run_job(job: Dict[str, Any], worker_index: int, slot_index: int) -> Dict[str, Any]:
        env = None
        model_runtime = _model_stub(job.get('model_cfg', {}) or {}, int(job.get('model_index', 0) or 0))
        try:
            model_runtime = build_model_runtimes(
                [job['model_cfg']],
                hooks_cfg=hooks_cfg,
                hook_manager=hook_manager,
                action_broker=None,
            )[0]

            runtime_env_cfg = deepcopy(base_env_cfg)
            runtime_env_cfg['worker_index'] = int(worker_index)
            runtime_env_cfg['hook_manager'] = hook_manager
            runtime_env_cfg['hooks'] = hooks_cfg
            runtime_env_cfg = resolve_runtime_sandbox_cfg(runtime_env_cfg)

            env = ArkEnv(runtime_env_cfg)
            result = evaluate_case(
                env=env,
                model_runtime=model_runtime,
                case=job['case'],
                hook_manager=hook_manager,
                hooks_cfg=hooks_cfg,
            )
            maybe_save_eval_trajectory(
                env=env,
                result=result,
                case=job['case'],
                model_runtime=model_runtime,
                eval_cfg=eval_cfg,
                writer=trajectory_writer,
            )
            return _attach_job_metadata(result, job, worker_index=worker_index, slot_index=slot_index)
        except Exception as exc:
            result = build_error_result(model_runtime, job['case'], exc)
            return _attach_job_metadata(result, job, worker_index=worker_index, slot_index=slot_index)
        finally:
            close_agent = getattr(model_runtime.get('agent'), 'close', None)
            if callable(close_agent):
                close_agent()
            if env is not None:
                env.close()

    return _run_job


def execute_parallel_jobs(
    jobs: List[Dict[str, Any]],
    *,
    max_workers: int,
    worker_index_base: int,
    job_runner: JobRunner,
    on_result: ResultHandler | None = None,
    stop_on_error: bool = False,
) -> List[Dict[str, Any]]:
    if not jobs:
        return []
    max_workers = min(_coerce_positive_int(max_workers, context='max_workers'), len(jobs))

    results: List[Dict[str, Any]] = []
    next_job_index = 0
    available_slots = list(range(max_workers))
    futures: Dict[concurrent.futures.Future, Dict[str, Any]] = {}

    def submit_next(executor: concurrent.futures.ThreadPoolExecutor) -> bool:
        nonlocal next_job_index
        if next_job_index >= len(jobs) or not available_slots:
            return False
        slot_index = available_slots.pop(0)
        job = jobs[next_job_index]
        next_job_index += 1
        worker_index = int(worker_index_base) + int(slot_index)
        future = executor.submit(job_runner, job, worker_index, slot_index)
        futures[future] = {'job': job, 'slot_index': slot_index, 'worker_index': worker_index}
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while submit_next(executor):
            pass

        stop_submitting = False
        while futures:
            done, _ = concurrent.futures.wait(
                futures.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                meta = futures.pop(future)
                available_slots.append(int(meta['slot_index']))
                available_slots.sort()
                try:
                    result = future.result()
                except BaseException as exc:
                    result = _fallback_error_result(
                        meta['job'],
                        exc,
                        worker_index=int(meta['worker_index']),
                        slot_index=int(meta['slot_index']),
                    )

                results.append(result)
                if on_result is not None:
                    on_result(result)
                if stop_on_error and result.get('status') != 'ok':
                    stop_submitting = True

            if not stop_submitting:
                while submit_next(executor):
                    pass

    return results


class ThreadSafeJsonlWriter:
    def __init__(
        self,
        path: str,
        *,
        append: bool = False,
        replace_existing_by_resume_key: bool = False,
    ):
        self._writer = JsonlWriter(path, append=append)
        self._lock = threading.Lock()
        self._replace_existing_by_resume_key = bool(replace_existing_by_resume_key)

    @property
    def path(self):
        return self._writer.path

    def write(self, record: Dict[str, Any]) -> None:
        with self._lock:
            if not self._replace_existing_by_resume_key:
                self._writer.write(record)
                return

            key = _parallel_result_resume_key(record)
            if key is None or not self._writer.path.exists():
                self._writer.write(record)
                return

            kept_records: List[Dict[str, Any]] = []
            with self._writer.path.open('r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        existing = json.loads(text)
                    except Exception:
                        continue
                    if not isinstance(existing, dict):
                        continue
                    if _parallel_result_resume_key(existing) == key:
                        continue
                    kept_records.append(existing)

            kept_records.append(record)
            with self._writer.path.open('w', encoding='utf-8') as f:
                for item in kept_records:
                    f.write(json.dumps(_to_jsonable(item), ensure_ascii=False) + '\n')


def _parallel_human_enabled(hooks_cfg: Dict[str, Any]) -> bool:
    human_cfg = hooks_cfg.get('human_interaction', {}) if isinstance(hooks_cfg.get('human_interaction', {}), dict) else {}
    return _coerce_bool(human_cfg.get('enabled', False), default=False)


def _runtime_sandbox_enabled(env_cfg: Dict[str, Any]) -> bool:
    sandbox_cfg = env_cfg.get('runtime_sandbox', {}) if isinstance(env_cfg.get('runtime_sandbox', {}), dict) else {}
    return _coerce_bool(sandbox_cfg.get('enabled', False), default=False)


def validate_parallel_runtime_isolation(
    env_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    *,
    max_workers: int,
) -> None:
    if int(max_workers) <= 1:
        return
    if _runtime_sandbox_enabled(env_cfg):
        return
    if _coerce_bool(eval_cfg.get('allow_shared_runtime_parallel', False), default=False):
        print(
            '[run_parallel_api_eval] WARNING: running multiple envs against a shared runtime because '
            'eval.allow_shared_runtime_parallel=true. Use this only when each worker is already isolated or read-only.'
        )
        return
    raise ValueError(
        'run_parallel_api_eval with eval.max_parallel_envs > 1 requires '
        'env_cfg.runtime_sandbox.enabled=true so concurrent Unity workers do not share writable runtime/Mods files. '
        'Set eval.max_parallel_envs=1 for serial execution, or set eval.allow_shared_runtime_parallel=true only '
        'when the runtime is already isolated or read-only.'
    )


def run_parallel_evaluation(cfg: Dict[str, Any], *, config_path: str | None = None) -> List[Dict[str, Any]]:
    env_cfg = dict(cfg['env_cfg'])
    eval_cfg = dict(cfg['eval'])
    hooks_cfg = dict(cfg.get('hooks', {}) or {})

    if _parallel_human_enabled(hooks_cfg):
        raise ValueError('run_parallel_api_eval does not support hooks.human_interaction.enabled=true')

    model_cfgs = list(cfg.get('models', []) or [])
    cases = apply_trajectory_load_to_cases(build_eval_cases(env_cfg, eval_cfg), eval_cfg)
    jobs = build_parallel_eval_jobs(cases, model_cfgs)
    skip_existing_results = _coerce_bool(eval_cfg.get('skip_existing_results', False), default=False)
    compacted_existing_results = 0
    if skip_existing_results:
        compacted_existing_results = compact_parallel_results_file(str(eval_cfg['output_path']), jobs)
        jobs_to_run, existing_results = filter_existing_parallel_jobs(jobs, str(eval_cfg['output_path']))
    else:
        jobs_to_run, existing_results = jobs, []

    max_workers = resolve_max_parallel_envs(eval_cfg, job_count=len(jobs_to_run)) if jobs_to_run else 0
    worker_index_base = int(eval_cfg.get('worker_index_base', env_cfg.get('worker_index', 0) or 0))

    if jobs_to_run:
        validate_parallel_runtime_isolation(env_cfg, eval_cfg, max_workers=max_workers)

        ensure_runtime_pool_range(
            env_cfg,
            worker_index_base=worker_index_base,
            worker_count=max_workers,
        )

    writer = ThreadSafeJsonlWriter(
        str(eval_cfg['output_path']),
        append=skip_existing_results,
        replace_existing_by_resume_key=skip_existing_results,
    )
    trajectory_eval_cfg = deepcopy(eval_cfg)
    if skip_existing_results:
        trajectory_save_cfg = trajectory_eval_cfg.get('trajectory_save', {})
        if isinstance(trajectory_save_cfg, dict):
            trajectory_save_cfg = dict(trajectory_save_cfg)
            trajectory_save_cfg['append'] = True
            trajectory_eval_cfg['trajectory_save'] = trajectory_save_cfg
    trajectory_writer = build_trajectory_writer(trajectory_eval_cfg)
    hook_manager, _ = build_eval_hook_manager(hooks_cfg)
    job_runner = make_parallel_job_runner(
        env_cfg=env_cfg,
        eval_cfg=eval_cfg,
        hooks_cfg=hooks_cfg,
        hook_manager=hook_manager,
        trajectory_writer=trajectory_writer,
    )

    print(
        f"Loaded {len(model_cfgs)} models, {len(cases)} cases, {len(jobs)} jobs; "
        f"max_parallel_envs={max_workers} worker_index_base={worker_index_base}"
    )
    if skip_existing_results:
        print(
            f"Resuming from {writer.path}: skipped {len(existing_results)} existing jobs, "
            f"pending {len(jobs_to_run)} jobs"
        )
        if compacted_existing_results:
            print(f"Compacted {compacted_existing_results} superseded result records from {writer.path}")
    print(f"Results will be written to {writer.path}")
    if trajectory_writer is not None:
        print(f"Trajectory records will be written to {trajectory_writer.path}")

    results: List[Dict[str, Any]] = list(existing_results)
    stop_on_error = bool(eval_cfg.get('stop_on_error', False))
    keep_open_on_end = _hook_keep_open_on_end(hooks_cfg)

    def handle_result(result: Dict[str, Any]) -> None:
        writer.write(result)
        results.append(result)
        status = result.get('status', 'unknown')
        prefix = (
            f"[job {result.get('job_index')}] slot={result.get('parallel_slot_index')} "
            f"worker={result.get('worker_index')} case={result.get('case_id')} "
            f"model={result.get('model_name')}"
        )
        if status == 'ok':
            print(
                f"{prefix} done turns={result.get('turns')} "
                f"final_success={result.get('rollout_success')} "
                f"ever_success={result.get('ever_attempt_success')} "
                f"truncated={result.get('rollout_truncated')} "
                f"score_reward={result.get('score_reward')}"
            )
            hook_manager.emit('job_end', result, source='run_parallel_api_eval')
        else:
            print(f"{prefix} error: {result.get('error_type')}: {result.get('error')}")
            hook_manager.emit('error', result, source='run_parallel_api_eval')

    hook_manager.start()
    completed = False
    started_at = time.time()
    try:
        hook_manager.emit(
            'run_start',
            {
                'config_path': config_path,
                'case_count': len(cases),
                'model_count': len(model_cfgs),
                'job_count': len(jobs),
                'pending_job_count': len(jobs_to_run),
                'skipped_existing_job_count': len(existing_results),
                'compacted_existing_result_count': int(compacted_existing_results),
                'max_parallel_envs': max_workers,
                'worker_index_base': worker_index_base,
            },
            source='run_parallel_api_eval',
        )
        if jobs_to_run:
            execute_parallel_jobs(
                jobs_to_run,
                max_workers=max_workers,
                worker_index_base=worker_index_base,
                job_runner=job_runner,
                on_result=handle_result,
                stop_on_error=stop_on_error,
            )
        hook_manager.emit(
            'run_end',
            {
                'result_count': len(results),
                'new_result_count': len(results) - len(existing_results),
                'skipped_existing_job_count': len(existing_results),
                'compacted_existing_result_count': int(compacted_existing_results),
                'duration_s': round(time.time() - started_at, 4),
            },
            source='run_parallel_api_eval',
        )
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
    parser = argparse.ArgumentParser(description='Run AgentArk API evaluation jobs across multiple envs in parallel')
    parser.add_argument(
        '--config',
        type=str,
        default='config/ark_env/parallel_api_eval.example.yaml',
        help='Path to parallel evaluation config (.yaml/.yml/.json)',
    )
    return parser


def main(args):
    cfg = load_eval_config(args.config)
    run_parallel_evaluation(cfg, config_path=args.config)


if __name__ == '__main__':
    main(build_parser().parse_args())
