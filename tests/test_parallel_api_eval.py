import json
import threading
import time
import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_eval.run_parallel_api_eval import (  # noqa: E402
    build_parallel_eval_jobs,
    compact_parallel_results_file,
    execute_parallel_jobs,
    filter_existing_parallel_jobs,
    resolve_max_parallel_envs,
    validate_parallel_runtime_isolation,
)
from agent_ark.ark_eval.run_api_agent import (  # noqa: E402
    JsonlWriter,
    compact_eval_results_file,
    load_existing_eval_results,
    write_eval_result,
)


class ParallelApiEvalTest(unittest.TestCase):
    def test_build_parallel_eval_jobs_cross_product(self):
        cases = [
            {'case_id': 'seed-1', 'task_name': 'marble', 'group_seed': 1},
            {'case_id': 'seed-2', 'task_name': 'marble', 'group_seed': 2},
        ]
        models = [
            {'name': 'model-a', 'model': 'a'},
            {'name': 'model-b', 'model': 'b'},
        ]

        jobs = build_parallel_eval_jobs(cases, models)

        self.assertEqual([job['job_id'] for job in jobs], [
            'seed-1::model-a',
            'seed-1::model-b',
            'seed-2::model-a',
            'seed-2::model-b',
        ])
        self.assertEqual([job['job_index'] for job in jobs], [0, 1, 2, 3])
        self.assertEqual([job['case_index'] for job in jobs], [0, 0, 1, 1])
        self.assertEqual([job['model_index'] for job in jobs], [0, 1, 0, 1])

    def test_resolve_max_parallel_envs_caps_to_job_count(self):
        self.assertEqual(resolve_max_parallel_envs({'max_parallel_envs': 8}, job_count=3), 3)
        self.assertEqual(resolve_max_parallel_envs({'env_limit': 2}, job_count=5), 2)
        self.assertEqual(resolve_max_parallel_envs({}, job_count=5), 1)

        with self.assertRaises(ValueError):
            resolve_max_parallel_envs({'max_parallel_envs': 0}, job_count=5)

    def test_parallel_runtime_isolation_requires_sandbox_for_multiple_workers(self):
        validate_parallel_runtime_isolation({}, {}, max_workers=1)
        validate_parallel_runtime_isolation({'runtime_sandbox': {'enabled': True}}, {}, max_workers=2)
        validate_parallel_runtime_isolation({}, {'allow_shared_runtime_parallel': True}, max_workers=2)

        with self.assertRaisesRegex(ValueError, 'runtime_sandbox'):
            validate_parallel_runtime_isolation({}, {}, max_workers=2)

    def test_execute_parallel_jobs_reuses_unique_slots(self):
        jobs = [
            {'job_id': f'job-{idx}', 'job_index': idx, 'case_index': idx, 'model_index': 0, 'case': {}, 'model_cfg': {}}
            for idx in range(6)
        ]
        active_slots = set()
        max_active = 0
        lock = threading.Lock()

        def runner(job, worker_index, slot_index):
            nonlocal max_active
            with lock:
                self.assertNotIn(slot_index, active_slots)
                active_slots.add(slot_index)
                max_active = max(max_active, len(active_slots))
            time.sleep(0.01)
            with lock:
                active_slots.remove(slot_index)
            return {
                'status': 'ok',
                'job_id': job['job_id'],
                'worker_index': worker_index,
                'parallel_slot_index': slot_index,
            }

        results = execute_parallel_jobs(
            jobs,
            max_workers=2,
            worker_index_base=10,
            job_runner=runner,
        )

        self.assertEqual(len(results), 6)
        self.assertLessEqual(max_active, 2)
        self.assertEqual(sorted({result['worker_index'] for result in results}), [10, 11])
        self.assertEqual(sorted({result['parallel_slot_index'] for result in results}), [0, 1])

    def test_filter_existing_parallel_jobs_by_task_seed_env_model(self):
        cases = [
            {'case_id': 'task-seed-0001', 'task_name': 'TaskA', 'group_seed': 1, 'env_id': 0},
            {'case_id': 'task-seed-0002', 'task_name': 'TaskA', 'group_seed': 2, 'env_id': 0},
        ]
        models = [
            {'name': 'model-a', 'model': 'a'},
            {'name': 'model-b', 'model': 'b'},
        ]
        jobs = build_parallel_eval_jobs(cases, models)

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'results.jsonl'
            with output_path.open('w', encoding='utf-8') as f:
                f.write(json.dumps({
                    'status': 'ok',
                    'requested_task_name': 'TaskA',
                    'requested_group_seed': 1,
                    'requested_env_id': 0,
                    'model_name': 'model-b',
                }) + '\n')

            pending, existing = filter_existing_parallel_jobs(jobs, str(output_path))

        self.assertEqual(len(existing), 1)
        self.assertEqual(existing[0]['model_name'], 'model-b')
        self.assertEqual(
            [job['job_id'] for job in pending],
            ['task-seed-0001::model-a', 'task-seed-0002::model-a', 'task-seed-0002::model-b'],
        )

    def test_filter_existing_parallel_jobs_retries_error_results(self):
        cases = [
            {'case_id': 'task-seed-0001', 'task_name': 'TaskA', 'group_seed': 1, 'env_id': 0},
        ]
        models = [
            {'name': 'model-a', 'model': 'a'},
        ]
        jobs = build_parallel_eval_jobs(cases, models)

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'results.jsonl'
            with output_path.open('w', encoding='utf-8') as f:
                f.write(json.dumps({
                    'status': 'error',
                    'requested_task_name': 'TaskA',
                    'requested_group_seed': 1,
                    'requested_env_id': 0,
                    'model_name': 'model-a',
                    'error_type': 'APITimeoutError',
                }) + '\n')

            pending, existing = filter_existing_parallel_jobs(jobs, str(output_path))

        self.assertEqual(existing, [])
        self.assertEqual([job['job_id'] for job in pending], ['task-seed-0001::model-a'])

    def test_compact_parallel_results_prefers_ok_over_old_error(self):
        cases = [
            {'case_id': 'task-seed-0001', 'task_name': 'TaskA', 'group_seed': 1, 'env_id': 0},
        ]
        models = [
            {'name': 'model-a', 'model': 'a'},
        ]
        jobs = build_parallel_eval_jobs(cases, models)

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'results.jsonl'
            records = [
                {
                    'status': 'error',
                    'requested_task_name': 'TaskA',
                    'requested_group_seed': 1,
                    'requested_env_id': 0,
                    'model_name': 'model-a',
                    'error_type': 'APITimeoutError',
                },
                {
                    'status': 'ok',
                    'requested_task_name': 'TaskA',
                    'requested_group_seed': 1,
                    'requested_env_id': 0,
                    'model_name': 'model-a',
                    'score_reward': 1.0,
                },
            ]
            with output_path.open('w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')

            compacted = compact_parallel_results_file(str(output_path), jobs)
            remaining = [
                json.loads(line)
                for line in output_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]

        self.assertEqual(compacted, 1)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]['status'], 'ok')
        self.assertEqual(remaining[0]['score_reward'], 1.0)

    def test_serial_eval_existing_human_ok_is_loaded_for_skip(self):
        cases = [
            {'case_id': 'task-seed-0001', 'task_name': 'TaskA', 'group_seed': 1, 'env_id': 0},
            {'case_id': 'task-seed-0002', 'task_name': 'TaskA', 'group_seed': 2, 'env_id': 0},
        ]
        model_runtimes = [{'name': 'human-local', 'model': 'human', 'provider': 'human'}]

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'results.jsonl'
            with output_path.open('w', encoding='utf-8') as f:
                f.write(json.dumps({
                    'status': 'ok',
                    'requested_task_name': 'TaskA',
                    'requested_group_seed': 1,
                    'requested_env_id': 0,
                    'model_name': 'human-local',
                }) + '\n')
                f.write(json.dumps({
                    'status': 'error',
                    'requested_task_name': 'TaskA',
                    'requested_group_seed': 2,
                    'requested_env_id': 0,
                    'model_name': 'human-local',
                }) + '\n')

            existing = load_existing_eval_results(str(output_path), cases, model_runtimes)

        self.assertEqual(len(existing), 1)
        self.assertIn(('TaskA', '1', '0', 'human-local'), existing)

    def test_serial_eval_write_replaces_old_error_with_ok(self):
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'results.jsonl'
            writer = JsonlWriter(str(output_path), append=True)
            write_eval_result(writer, {
                'status': 'error',
                'requested_task_name': 'TaskA',
                'requested_group_seed': 1,
                'requested_env_id': 0,
                'model_name': 'human-local',
                'error_type': 'RuntimeError',
            })
            write_eval_result(
                writer,
                {
                    'status': 'ok',
                    'requested_task_name': 'TaskA',
                    'requested_group_seed': 1,
                    'requested_env_id': 0,
                    'model_name': 'human-local',
                    'score_reward': 1.0,
                },
                replace_existing_by_resume_key=True,
            )

            remaining = [
                json.loads(line)
                for line in output_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]

        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]['status'], 'ok')
        self.assertEqual(remaining[0]['score_reward'], 1.0)

    def test_serial_eval_compact_prefers_ok_over_old_error(self):
        cases = [
            {'case_id': 'task-seed-0001', 'task_name': 'TaskA', 'group_seed': 1, 'env_id': 0},
        ]
        model_runtimes = [{'name': 'human-local', 'model': 'human', 'provider': 'human'}]

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'results.jsonl'
            with output_path.open('w', encoding='utf-8') as f:
                f.write(json.dumps({
                    'status': 'error',
                    'requested_task_name': 'TaskA',
                    'requested_group_seed': 1,
                    'requested_env_id': 0,
                    'model_name': 'human-local',
                }) + '\n')
                f.write(json.dumps({
                    'status': 'ok',
                    'requested_task_name': 'TaskA',
                    'requested_group_seed': 1,
                    'requested_env_id': 0,
                    'model_name': 'human-local',
                    'score_reward': 1.0,
                }) + '\n')

            compacted = compact_eval_results_file(str(output_path), cases, model_runtimes)
            remaining = [
                json.loads(line)
                for line in output_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]

        self.assertEqual(compacted, 1)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
