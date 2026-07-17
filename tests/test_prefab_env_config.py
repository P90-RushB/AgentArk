import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.direct_env import EnvInfoManager, EnvWrapper  # noqa: E402
from agent_ark.ark_eval.run_api_agent import (  # noqa: E402
    _eval_env_uses_task_store,
    build_eval_cases,
    load_eval_config,
    print_summary,
)


class PrefabEnvConfigTest(unittest.TestCase):
    def _manager_for_mods(self, mods_path):
        manager = EnvInfoManager.__new__(EnvInfoManager)
        manager.cfg = {'mod_path': str(mods_path)}
        manager.mod_path = str(mods_path)
        manager.task_store_path = manager.resolve_task_store_path(manager.mod_path)
        manager.task_config = {}
        manager._last_task_name = None
        manager._last_group_seed = None
        return manager

    def _wrapper_for_video_frame_selection(self, selection):
        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper.env_info_mgr = SimpleNamespace(env_config={
            'obs_mode': 'video',
            'env_wrapper_cfg': {
                'video_frame_selection': selection,
            },
        })
        return wrapper

    def _obs_with_transition_and_decision_frames(self):
        transition_frame = Image.new('RGB', (2, 2), color=(10, 20, 30))
        decision_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        decision_frame[:, :, 0] = 40
        decision_frame[:, :, 1] = 50
        decision_frame[:, :, 2] = 60
        return {
            0: {
                'vis': [decision_frame],
                'video_raw': {'unused': True},
                'video_pil': {0: [transition_frame]},
            }
        }

    def test_override_false_skips_all_tasks_lookup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mods_path = Path(temp_dir)
            (mods_path / 'config.json').write_text(json.dumps({
                'override_by_task': False,
                'load_mod_mode': 'none',
                'task_name': 'snake_prefab',
                'action_mode': 'code',
                'engine_para': {},
                'max_attempts': 1,
                'max_steps_per_attempt': 2,
            }), encoding='utf-8')

            manager = self._manager_for_mods(mods_path)
            manager.reset()

            self.assertEqual(manager.task_list, [])
            self.assertFalse(manager.env_config['override_by_task'])
            self.assertEqual(manager.env_config['load_mod_mode'], 'none')
            self.assertEqual(manager.env_config['task_name'], 'snake_prefab')
            self.assertEqual(manager.env_config['max_attempts'], 1)
            self.assertEqual(manager.env_config['max_steps_per_attempt'], 2)

    def test_prefab_config_requires_rollout_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mods_path = Path(temp_dir)
            (mods_path / 'config.json').write_text(json.dumps({
                'override_by_task': False,
                'load_mod_mode': 'none',
                'task_name': 'snake_prefab',
                'action_mode': 'code',
                'engine_para': {},
            }), encoding='utf-8')

            manager = self._manager_for_mods(mods_path)
            with self.assertRaisesRegex(ValueError, 'max_attempts'):
                manager.reset()

    def test_prefab_task_name_loads_selected_task_params_from_unity_task_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / 'AgentArkUnity'
            mods_path = project_root / 'Assets' / 'Resources' / 'Mods'
            mods_path.mkdir(parents=True)
            task_root = project_root / 'Assets' / 'AgentArk' / 'Tasks' / 'Task6_GoldMiner2D'
            task_root.mkdir(parents=True)

            (mods_path / 'config.json').write_text(json.dumps({
                'override_by_task': False,
                'load_mod_mode': 'none',
                'task_name': 'SnakeTask',
                'action_mode': 'func',
                'obs_mode': 'video',
                'engine_para': {},
                'max_attempts': 9,
                'max_steps_per_attempt': 11,
                'task_params': {
                    'initialSize': 8,
                    'maxStepsWithoutFood': 2,
                },
            }), encoding='utf-8')
            (task_root / 'task_config.json').write_text(json.dumps({
                'task_name': 'GoldMiner2DTask',
                'max_attempts': 1,
                'max_steps_per_attempt': 10,
                'task_params': {
                    'minLaunchAngleDeg': -80.0,
                    'maxLaunchAngleDeg': 80.0,
                    'boundaryAngleMarginDeg': 8.0,
                },
                'env_wrapper_cfg': {
                    'initial_observation': {
                        'enabled': True,
                        'no_action_decision_steps': 1,
                    },
                },
            }), encoding='utf-8')

            manager = self._manager_for_mods(mods_path)
            manager.reset(task_name='GoldMiner2DTask', group_seed=1, env_id=0)

            self.assertEqual(manager.env_config['task_name'], 'GoldMiner2DTask')
            self.assertEqual(manager.env_config['obs_mode'], 'video')
            self.assertEqual(manager.env_config['task_params'], {
                'minLaunchAngleDeg': -80.0,
                'maxLaunchAngleDeg': 80.0,
                'boundaryAngleMarginDeg': 8.0,
            })
            self.assertEqual(manager.env_config['max_attempts'], 1)
            self.assertEqual(manager.env_config['max_steps_per_attempt'], 10)
            self.assertTrue(manager.env_config['env_wrapper_cfg']['initial_observation']['enabled'])

    def test_prefab_task_without_task_params_clears_stale_base_task_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / 'AgentArkUnity'
            mods_path = project_root / 'Assets' / 'Resources' / 'Mods'
            mods_path.mkdir(parents=True)
            task_root = project_root / 'Assets' / 'AgentArk' / 'Tasks' / 'NoParamsTask'
            task_root.mkdir(parents=True)

            (mods_path / 'config.json').write_text(json.dumps({
                'override_by_task': False,
                'load_mod_mode': 'none',
                'task_name': 'SnakeTask',
                'action_mode': 'func',
                'engine_para': {},
                'max_attempts': 4,
                'max_steps_per_attempt': 5,
                'task_params': {
                    'initialSize': 8,
                },
            }), encoding='utf-8')
            (task_root / 'task_config.json').write_text(json.dumps({
                'task_name': 'NoParamsTask',
                'max_attempts': 2,
                'max_steps_per_attempt': 3,
            }), encoding='utf-8')

            manager = self._manager_for_mods(mods_path)
            manager.reset(task_name='NoParamsTask', group_seed=1, env_id=0)

            self.assertEqual(manager.env_config['task_name'], 'NoParamsTask')
            self.assertNotIn('task_params', manager.env_config)
            self.assertEqual(manager.env_config['max_attempts'], 2)
            self.assertEqual(manager.env_config['max_steps_per_attempt'], 3)

    def test_prefab_task_materializes_matching_effective_yaml_and_json_from_backup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / 'AgentArkUnity'
            mods_path = project_root / 'Assets' / 'Resources' / 'Mods'
            mods_path.mkdir(parents=True)
            task_root = project_root / 'Assets' / 'AgentArk' / 'Tasks' / 'CleanPrefabTask'
            task_root.mkdir(parents=True)

            root_yaml = '\n'.join([
                'override_by_task: false',
                'load_mod_mode: none',
                'task_name: OldPrefabTask',
                'obs_mode: video',
                'capture_interval: 25',
                'early_request_act: false',
                'time_between_decisions: 40',
                'action_mode: code',
                'done_on_script_error: true',
                'num_parallel_envs: 1',
                'width: 320',
                'height: 240',
                'engine_para:',
                '  time_scale: 8',
                'max_attempts: 9',
                'max_steps_per_attempt: 11',
                'task_params:',
                '  staleFromPreviousTask: 1',
                'env_wrapper_cfg:',
                '  video_frame_selection: decision_only',
                '',
            ])
            (mods_path / 'config.yaml').write_text(root_yaml, encoding='utf-8')
            (mods_path / 'config.yaml.bak').write_text(root_yaml, encoding='utf-8')
            (mods_path / 'config.json').write_text('{}', encoding='utf-8')
            (task_root / 'task_config.yaml').write_text('\n'.join([
                'task_name: CleanPrefabTask',
                'obs_mode: decision',
                'capture_interval: 10',
                'early_request_act: true',
                'time_between_decisions: 2.5',
                'action_mode: func',
                'done_on_script_error: false',
                'num_parallel_envs: 1',
                'width: 768',
                'height: 512',
                'engine_para:',
                '  time_scale: 4',
                'max_attempts: 1',
                'max_steps_per_attempt: 6',
                'task_params:',
                '  cleanValue: 7',
                'env_wrapper_cfg:',
                '  video_frame_selection: transition_and_decision',
                '',
            ]), encoding='utf-8')

            manager = self._manager_for_mods(mods_path)
            manager.reset(task_name='CleanPrefabTask', group_seed=12, env_id=0)

            effective_yaml = yaml.safe_load((mods_path / 'config.yaml').read_text(encoding='utf-8'))
            effective = json.loads((mods_path / 'config.json').read_text(encoding='utf-8'))
            self.assertEqual(effective_yaml, effective)
            self.assertEqual((mods_path / 'config.yaml.bak').read_text(encoding='utf-8'), root_yaml)
            self.assertFalse(effective['override_by_task'])
            self.assertEqual(effective['load_mod_mode'], 'none')
            self.assertEqual(effective['task_name'], 'CleanPrefabTask')
            self.assertEqual(effective['obs_mode'], 'decision')
            self.assertEqual(effective['width'], 768)
            self.assertEqual(effective['height'], 512)
            self.assertEqual(effective['engine_para']['time_scale'], 4)
            self.assertEqual(effective['task_params'], {'cleanValue': 7})
            self.assertNotIn('staleFromPreviousTask', effective['task_params'])

    def test_override_true_still_requires_task_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mods_path = Path(temp_dir)
            (mods_path / 'config.json').write_text(json.dumps({
                'override_by_task': True,
                'load_mod_mode': 'task_name',
                'task_name': 'snake',
                'action_mode': 'code',
                'engine_para': {},
                'max_attempts': 1,
                'max_steps_per_attempt': 1,
            }), encoding='utf-8')

            manager = self._manager_for_mods(mods_path)
            with self.assertRaises(FileNotFoundError):
                manager.reset()

    def test_task_config_rollout_budget_is_merged_even_when_base_lacks_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mods_path = Path(temp_dir)
            task_root = mods_path / 'all_tasks' / 'RotateTask'
            cfg_root = task_root / 'cfg'
            cfg_root.mkdir(parents=True)
            (task_root / 'RotateTask.json').write_text('{}', encoding='utf-8')
            (mods_path / 'config.json').write_text(json.dumps({
                'override_by_task': True,
                'load_mod_mode': 'task_name',
                'task_name': 'RotateTask',
                'action_mode': 'func',
                'engine_para': {},
                'max_steps': 0,
            }), encoding='utf-8')
            (cfg_root / 'task_config.json').write_text(json.dumps({
                'max_attempts': 2,
                'max_steps_per_attempt': 3,
            }), encoding='utf-8')

            manager = self._manager_for_mods(mods_path)
            manager.reset(task_name='RotateTask')

            self.assertEqual(manager.env_config['max_attempts'], 2)
            self.assertEqual(manager.env_config['max_steps_per_attempt'], 3)

    def test_task_store_switches_tasks_from_unchanged_backup_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mods_path = Path(temp_dir)
            root_yaml = '\n'.join([
                'override_by_task: true',
                'load_mod_mode: task_name',
                'task_name: Baseline',
                'obs_mode: decision',
                'capture_interval: 10',
                'early_request_act: true',
                'time_between_decisions: 5',
                'action_mode: func',
                'done_on_script_error: false',
                'num_parallel_envs: 1',
                'width: 320',
                'height: 240',
                'engine_para:',
                '  time_scale: 4',
                'max_attempts: 1',
                'max_steps_per_attempt: 1',
                'task_params: {}',
                'env_wrapper_cfg:',
                '  video_frame_selection: decision_only',
                '  context_manager:',
                '    messages:',
                '      enabled: true',
                '',
            ])
            (mods_path / 'config.yaml').write_text(root_yaml, encoding='utf-8')
            (mods_path / 'config.yaml.bak').write_text(root_yaml, encoding='utf-8')
            (mods_path / 'config.json').write_text('{}', encoding='utf-8')

            for task_name, config_lines in {
                'TaskA': [
                    'task_name: TaskA',
                    'width: 640',
                    'height: 480',
                    'max_attempts: 1',
                    'max_steps_per_attempt: 2',
                    'task_params:',
                    '  onlyA: 1',
                    'env_wrapper_cfg:',
                    '  context_manager:',
                    '    messages:',
                    '      text_max_chars: 111',
                ],
                'TaskB': [
                    'task_name: TaskB',
                    'width: 768',
                    'height: 512',
                    'max_attempts: 1',
                    'max_steps_per_attempt: 3',
                    'task_params:',
                    '  onlyB: 2',
                    'env_wrapper_cfg:',
                    '  video_frame_selection: transition_and_decision',
                ],
            }.items():
                task_root = mods_path / 'all_tasks' / task_name
                cfg_root = task_root / 'cfg'
                cfg_root.mkdir(parents=True)
                (task_root / f'{task_name}.json').write_text('{}', encoding='utf-8')
                (cfg_root / 'task_config.yaml').write_text(
                    '\n'.join(config_lines) + '\n',
                    encoding='utf-8',
                )

            manager = self._manager_for_mods(mods_path)
            manager.reset(task_name='TaskA', group_seed=1)
            first_effective = json.loads((mods_path / 'config.json').read_text(encoding='utf-8'))
            self.assertEqual(first_effective['task_params'], {'onlyA': 1})
            self.assertEqual(
                first_effective['env_wrapper_cfg']['context_manager']['messages']['text_max_chars'],
                111,
            )

            manager.reset(task_name='TaskB', group_seed=2)
            second_effective = json.loads((mods_path / 'config.json').read_text(encoding='utf-8'))

            effective_yaml = yaml.safe_load((mods_path / 'config.yaml').read_text(encoding='utf-8'))
            self.assertEqual(effective_yaml, second_effective)
            self.assertEqual((mods_path / 'config.yaml.bak').read_text(encoding='utf-8'), root_yaml)
            self.assertEqual(second_effective['task_name'], 'TaskB')
            self.assertEqual(second_effective['width'], 768)
            self.assertEqual(second_effective['height'], 512)
            self.assertEqual(second_effective['task_params'], {'onlyB': 2})
            self.assertNotIn(
                'text_max_chars',
                second_effective['env_wrapper_cfg']['context_manager']['messages'],
            )

    def test_task_store_resolves_task_info_public_name_and_legacy_aliases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mods_path = Path(temp_dir)
            task_root = mods_path / 'all_tasks' / 'PublicRotate'
            cfg_root = task_root / 'cfg'
            cfg_root.mkdir(parents=True)
            (task_root / 'catalog_1.0.json').write_text('{}', encoding='utf-8')
            (mods_path / 'config.json').write_text(json.dumps({
                'override_by_task': True,
                'load_mod_mode': 'task_name',
                'task_name': 'PublicRotate',
                'action_mode': 'func',
                'engine_para': {},
                'max_steps': 0,
            }), encoding='utf-8')
            (cfg_root / 'task_config.json').write_text(json.dumps({
                'task_name': 'RotateTask',
                'max_attempts': 2,
                'max_steps_per_attempt': 3,
                'task_info': {
                    'id': 99,
                    'name': 'PublicRotate',
                    'version': 'v1',
                    'tags': ['3d'],
                    'legacy_names': ['Task99_RotateTask', 'RotateTask'],
                },
            }), encoding='utf-8')

            env_cfg = {'mod_path': str(mods_path)}
            eval_cfg = {'task_names': ['Task99_RotateTask'], 'group_seeds': [1]}

            cases = build_eval_cases(env_cfg, eval_cfg)
            self.assertEqual(cases[0]['task_name'], 'Task99_RotateTask')

            manager = self._manager_for_mods(mods_path)
            manager.reset(task_name='Task99_RotateTask', group_seed=1)

            self.assertEqual(manager.now_task_info['folder_name'], 'PublicRotate')
            self.assertEqual(manager.env_config['task_name'], 'PublicRotate')
            self.assertEqual(manager.env_config['max_attempts'], 2)
            self.assertEqual(manager.env_config['max_steps_per_attempt'], 3)

    def test_task_config_initial_observation_is_merged_even_when_base_lacks_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mods_path = Path(temp_dir)
            task_root = mods_path / 'all_tasks' / 'MovingBall'
            cfg_root = task_root / 'cfg'
            cfg_root.mkdir(parents=True)
            (task_root / 'MovingBall.json').write_text('{}', encoding='utf-8')
            (mods_path / 'config.json').write_text(json.dumps({
                'override_by_task': True,
                'load_mod_mode': 'task_name',
                'task_name': 'MovingBall',
                'action_mode': 'func',
                'engine_para': {},
                'obs_mode': 'video',
                'max_steps': 1,
            }), encoding='utf-8')
            (cfg_root / 'task_config.json').write_text(json.dumps({
                'max_attempts': 1,
                'max_steps_per_attempt': 1,
                'env_wrapper_cfg': {
                    'initial_observation': {
                        'enabled': True,
                        'no_action_decision_steps': 1,
                        'empty_step_duration_seconds': 0.35,
                        'require_min_frames_per_camera': 2,
                    }
                },
            }), encoding='utf-8')

            manager = self._manager_for_mods(mods_path)
            manager.reset(task_name='MovingBall')

            initial_cfg = manager.env_config['env_wrapper_cfg']['initial_observation']
            self.assertTrue(initial_cfg['enabled'])
            self.assertEqual(initial_cfg['no_action_decision_steps'], 1)
            self.assertEqual(initial_cfg['empty_step_duration_seconds'], 0.35)
            self.assertEqual(initial_cfg['require_min_frames_per_camera'], 2)

    def test_python_only_merge_removes_stale_initial_observation_when_task_omits_it(self):
        base = {
            'env_wrapper_cfg': {
                'video_frame_selection': 'decision_only',
                'initial_observation': {
                    'enabled': True,
                    'no_action_decision_steps': 1,
                    'empty_step_duration_seconds': 0.8,
                },
                'context_manager': {
                    'messages': {
                        'text_max_chars': 6000,
                    },
                },
            },
        }
        task = {
            'env_wrapper_cfg': {
                'context_manager': {
                    'messages': {
                        'text_max_chars': 4000,
                    },
                },
            },
        }

        merged = EnvInfoManager._merge_python_only_task_config(base, task)

        self.assertNotIn('initial_observation', merged['env_wrapper_cfg'])
        self.assertEqual(merged['env_wrapper_cfg']['context_manager']['messages']['text_max_chars'], 4000)
        self.assertEqual(merged['env_wrapper_cfg']['video_frame_selection'], 'decision_only')

    def test_video_frame_selection_decision_only_keeps_decision_frame(self):
        wrapper = self._wrapper_for_video_frame_selection('decision_only')

        processed = wrapper.post_process_obs(self._obs_with_transition_and_decision_frames())

        frames = processed[0]['vis'][0]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].getpixel((0, 0)), (40, 50, 60))
        self.assertIsNone(processed[0]['video_raw'])
        self.assertIsNone(processed[0]['video_pil'])

    def test_video_frame_selection_transition_and_decision_keeps_both(self):
        wrapper = self._wrapper_for_video_frame_selection('transition_and_decision')

        processed = wrapper.post_process_obs(self._obs_with_transition_and_decision_frames())

        frames = processed[0]['vis'][0]
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(frames[1].getpixel((0, 0)), (40, 50, 60))

    def test_video_frame_selection_transition_only_drops_decision_frame(self):
        wrapper = self._wrapper_for_video_frame_selection('transition_only')

        processed = wrapper.post_process_obs(self._obs_with_transition_and_decision_frames())

        frames = processed[0]['vis'][0]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].getpixel((0, 0)), (10, 20, 30))

    def test_python_only_merge_replaces_initial_observation_instead_of_deep_merging_it(self):
        base = {
            'env_wrapper_cfg': {
                'initial_observation': {
                    'enabled': True,
                    'no_action_decision_steps': 2,
                    'empty_step_duration_seconds': 1.5,
                    'require_min_frames_per_camera': 6,
                },
            },
        }
        task = {
            'env_wrapper_cfg': {
                'initial_observation': {
                    'enabled': True,
                    'no_action_decision_steps': 1,
                },
            },
        }

        merged = EnvInfoManager._merge_python_only_task_config(base, task)

        self.assertEqual(
            merged['env_wrapper_cfg']['initial_observation'],
            {
                'enabled': True,
                'no_action_decision_steps': 1,
            },
        )

    def test_initial_observation_payload_offsets_unity_max_steps_only(self):
        env_cfg = {
            'obs_mode': 'video',
            'max_attempts': 1,
            'max_steps_per_attempt': 1,
            'max_steps': 1,
            'time_between_decisions': 10.0,
            'env_wrapper_cfg': {
                'initial_observation': {
                    'enabled': True,
                    'no_action_decision_steps': 2,
                    'empty_step_duration_seconds': 0.25,
                }
            },
        }

        payload = EnvWrapper._build_unity_env_params_payload(env_cfg)

        self.assertNotIn('env_wrapper_cfg', payload)
        self.assertEqual(env_cfg['max_steps_per_attempt'], 1)
        self.assertEqual(payload['max_steps'], 3)
        self.assertEqual(payload['initial_observation_no_action_steps'], 2)
        self.assertEqual(payload['initial_observation_empty_step_duration_seconds'], 0.25)

    def test_unity_env_payload_normalizes_numeric_obs_mode(self):
        payload = EnvWrapper._build_unity_env_params_payload({
            'obs_mode': 1,
            'max_steps': 1,
            'task_params': {'unused': True},
            'env_wrapper_cfg': {},
        })

        self.assertEqual(payload['obs_mode'], 'video')
        self.assertNotIn('task_params', payload)
        self.assertNotIn('env_wrapper_cfg', payload)

    def test_initial_observation_requires_video_obs_mode(self):
        env_cfg = {
            'obs_mode': 'decision',
            'max_steps': 1,
            'env_wrapper_cfg': {
                'initial_observation': {
                    'enabled': True,
                    'no_action_decision_steps': 1,
                }
            },
        }

        with self.assertRaisesRegex(ValueError, 'obs_mode=video'):
            EnvWrapper._build_unity_env_params_payload(env_cfg)

    def test_task_params_are_replaced_by_task_config(self):
        base = {
            'override_by_task': True,
            'task_params': {
                'initialSize': 8,
                'maxStepsWithoutFood': 2,
            },
            'engine_para': {'time_scale': 1.0},
        }
        task = {
            'task_params': {
                'forceScale': 2.0,
            },
            'engine_para': {'time_scale': 4.0},
        }

        merged = EnvInfoManager._overlay_env_with_task_config(base, task)

        self.assertEqual(merged['task_params'], {'forceScale': 2.0})
        self.assertEqual(merged['engine_para'], {'time_scale': 4.0})

    def test_task_params_are_removed_when_task_config_omits_them(self):
        base = {
            'override_by_task': True,
            'task_params': {
                'initialSize': 8,
                'maxStepsWithoutFood': 2,
            },
            'engine_para': {'time_scale': 1.0},
        }
        task = {
            'engine_para': {'time_scale': 4.0},
        }

        merged = EnvInfoManager._overlay_env_with_task_config(base, task)

        self.assertNotIn('task_params', merged)
        self.assertEqual(merged['engine_para'], {'time_scale': 4.0})

    def test_llm_visible_prompt_appends_rollout_budget_from_config(self):
        env = object.__new__(EnvWrapper)
        info_mgr = type('InfoMgr', (), {})()
        info_mgr.system_prompt = 'SYS'
        info_mgr.env_config = {
            'prompt': {'language': 'en'},
            'max_attempts': 1,
            'max_steps_per_attempt': 5,
        }
        env.env_info_mgr = info_mgr

        visible = env._build_llm_visible_prompt('<task_prompt>Task body</task_prompt>')

        self.assertIn('Task body', visible)
        self.assertIn('Play budget for this task:', visible)
        self.assertIn('up to 1 game round', visible)
        self.assertIn('up to 5 operation step', visible)
        self.assertIn('at most 5 operation step', visible)
        self.assertIn('Earlier successful game rounds are recorded', visible)
        self.assertNotIn('unless you succeed earlier', visible)
        self.assertNotIn('ArkSubEnv', visible)
        self.assertNotIn('ArkEnv', visible)
        self.assertNotIn('max_attempts', visible)
        self.assertNotIn('max_steps_per_attempt', visible)
        self.assertNotIn('max_steps_per_attempt', visible)

    def test_llm_visible_prompt_uses_plain_chinese_rollout_budget(self):
        env = object.__new__(EnvWrapper)
        info_mgr = type('InfoMgr', (), {})()
        info_mgr.system_prompt = 'SYS'
        info_mgr.env_config = {
            'prompt': {'language': 'zh'},
            'max_attempts': 2,
            'max_steps_per_attempt': 3,
        }
        env.env_info_mgr = info_mgr

        visible = env._build_llm_visible_prompt('<task_prompt>任务正文</task_prompt>')

        self.assertIn('任务正文', visible)
        self.assertIn('本次任务的可操作次数：', visible)
        self.assertIn('最多可以玩 2 个游戏回合', visible)
        self.assertIn('每个游戏回合最多可以操作 3 步', visible)
        self.assertIn('整个任务最多可以操作 6 步', visible)
        self.assertIn('中间游戏回合即使成功，也会记录下来并继续到最后一个游戏回合', visible)
        self.assertNotIn('如果提前成功，任务会提前结束', visible)
        self.assertNotIn('ArkSubEnv', visible)
        self.assertNotIn('ArkEnv', visible)
        self.assertNotIn('max_attempts', visible)

    def test_reset_obs_payload_separates_static_prompt_and_context(self):
        env = object.__new__(EnvWrapper)
        info_mgr = type('InfoMgr', (), {})()
        info_mgr.system_prompt = 'SYS'
        info_mgr.env_config = {
            'prompt': {'language': 'en'},
            'max_attempts': 1,
            'max_steps_per_attempt': 5,
        }
        env.env_info_mgr = info_mgr

        payload = env._build_reset_obs_payload(
            '<task_prompt>Task body</task_prompt>\n'
            '<reset_context>Goal: red=2 yellow=1 blue=0</reset_context>'
        )

        self.assertIn('Task body', payload['task_prompt'])
        self.assertIn('Goal: red=2 yellow=1 blue=0', payload['reset_context'])
        self.assertIn('<reset_context>', payload['step_msg'])
        self.assertIn('Goal: red=2 yellow=1 blue=0', payload['step_msg'])

    def test_api_eval_prefab_config_treats_string_none_as_editor_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mods_path = root / 'Mods'
            mods_path.mkdir()
            config_path = root / 'eval.yaml'
            config_path.write_text(
                '\n'.join([
                    'env_cfg:',
                    '  env_path: None',
                    '  mod_path: None',
                    '  task_type: RLTask',
                    '  env_config_overrides:',
                    '    override_by_task: false',
                    '    num_parallel_envs: 1',
                    'eval:',
                    '  num_cases: 1',
                    '  fixed_env_id: 0',
                    '  task_names:',
                    '    - marble',
                    'models:',
                    '  - name: dummy',
                    '    model: dummy-model',
                ]),
                encoding='utf-8',
            )

            with patch.dict(os.environ, {'AGENTARK_MOD_PATH': '', 'AGENT_ARK_MOD_PATH': str(mods_path)}, clear=False):
                cfg = load_eval_config(str(config_path))

            env_cfg = cfg['env_cfg']
            self.assertIsNone(env_cfg['env_path'])
            self.assertEqual(env_cfg['mod_path'], str(mods_path))
            self.assertFalse(_eval_env_uses_task_store(env_cfg))
            self.assertEqual(build_eval_cases(env_cfg, cfg['eval'])[0]['task_name'], 'marble')

    def test_api_eval_group_seed_range_builds_explicit_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mods_path = Path(temp_dir)
            (mods_path / 'config.json').write_text(json.dumps({
                'override_by_task': False,
                'load_mod_mode': 'none',
                'task_name': 'marble',
                'action_mode': 'code',
                'engine_para': {},
                'max_attempts': 1,
                'max_steps_per_attempt': 1,
            }), encoding='utf-8')

            env_cfg = {
                'mod_path': str(mods_path),
                'env_config_overrides': {'override_by_task': False},
            }
            eval_cfg = {
                'num_cases': 99,
                'fixed_env_id': 0,
                'task_names': ['marble'],
                'group_seeds': {'start': 1, 'end': 3},
            }

            cases = build_eval_cases(env_cfg, eval_cfg)

            self.assertEqual([case['group_seed'] for case in cases], [1, 2, 3])
            self.assertEqual([case['case_id'] for case in cases], [
                'marble-seed-0001',
                'marble-seed-0002',
                'marble-seed-0003',
            ])
            self.assertEqual([case['env_id'] for case in cases], [0, 0, 0])

    def test_print_summary_labels_score_reward_average(self):
        results = [
            {
                'status': 'ok',
                'model_name': 'openrouter-model',
                'score_reward': 1.0,
                'total_reward': -2.0,
                'rollout_success': True,
                'ever_attempt_success': True,
                'rollout_truncated': False,
            },
            {
                'status': 'ok',
                'model_name': 'openrouter-model',
                'score_reward': -1.0,
                'total_reward': -5.0,
                'rollout_success': False,
                'ever_attempt_success': False,
                'rollout_truncated': True,
            },
        ]

        with patch('builtins.print') as mocked_print:
            print_summary(results)

        output = '\n'.join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
        self.assertIn('avg_score_reward=0.0000', output)
        self.assertIn('avg_rollout_reward=-3.5000', output)
        self.assertNotIn('avg_score=', output)


if __name__ == '__main__':
    unittest.main()
