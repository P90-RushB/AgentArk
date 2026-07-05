import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.ark_sub_env import (  # noqa: E402
    _derive_mod_path_from_env_path,
    _normalize_optional_path,
)
from agent_ark.ark_env.runtime_sandbox import _path_from_value  # noqa: E402
from agent_ark.ark_env.runtime_config import load_runtime_config  # noqa: E402
from agent_ark.ark_eval.run_api_agent import load_eval_config  # noqa: E402


class RuntimePathResolutionTest(unittest.TestCase):
    def _create_build(self, root: Path, executable_name: str = 'AgentArk.x86_64') -> tuple[Path, Path]:
        executable = root / executable_name
        executable.write_text('', encoding='utf-8')
        mods_path = root / 'AgentArk_Data' / 'Resources' / 'Mods'
        mods_path.mkdir(parents=True)
        return executable, mods_path

    def test_derives_mod_path_from_linux_x86_64_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executable, mods_path = self._create_build(Path(temp_dir), 'AgentArk.x86_64')

            self.assertEqual(_derive_mod_path_from_env_path(str(executable)), str(mods_path))

    def test_derives_mod_path_from_build_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            build_root = Path(temp_dir)
            _, mods_path = self._create_build(build_root, 'AgentArk.exe')

            self.assertEqual(_derive_mod_path_from_env_path(str(build_root)), str(mods_path))

    def test_expands_nested_environment_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            build_root = Path(temp_dir)
            executable, mods_path = self._create_build(build_root)
            task_store = mods_path / 'all_tasks'

            with patch.dict(os.environ, {
                'AGENTARK_ENV_PATH': str(executable),
                'AGENTARK_MOD_PATH': str(mods_path),
                'AGENTARK_TASK_STORE_PATH': '${AGENTARK_MOD_PATH}/all_tasks',
            }, clear=False):
                self.assertEqual(Path(_normalize_optional_path('${AGENTARK_ENV_PATH}')), executable)
                self.assertEqual(_path_from_value('${AGENTARK_TASK_STORE_PATH}'), task_store)

    def test_runtime_config_expands_environment_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            build_root = Path(temp_dir)
            executable, mods_path = self._create_build(build_root)
            pool_root = build_root / 'pool'
            config_path = build_root / 'runtime.yaml'
            config_path.write_text(
                '\n'.join([
                    'env_cfg:',
                    '  env_path: "${AGENTARK_ENV_PATH}"',
                    '  mod_path: "${AGENTARK_MOD_PATH}"',
                    '  runtime_sandbox:',
                    '    pool_root: "${AGENTARK_RUNTIME_POOL_ROOT}"',
                    '    shared_task_store_path: "${AGENTARK_TASK_STORE_PATH}"',
                ]),
                encoding='utf-8',
            )

            with patch.dict(os.environ, {
                'AGENTARK_ENV_PATH': str(executable),
                'AGENTARK_MOD_PATH': str(mods_path),
                'AGENTARK_TASK_STORE_PATH': '${AGENTARK_MOD_PATH}/all_tasks',
                'AGENTARK_RUNTIME_POOL_ROOT': str(pool_root),
            }, clear=False):
                cfg = load_runtime_config(str(config_path))

            self.assertEqual(cfg['env_cfg']['env_path'], str(executable))
            self.assertEqual(cfg['env_cfg']['mod_path'], str(mods_path))
            self.assertEqual(
                Path(cfg['env_cfg']['runtime_sandbox']['shared_task_store_path']),
                mods_path / 'all_tasks',
            )
            self.assertEqual(cfg['env_cfg']['runtime_sandbox']['pool_root'], str(pool_root))

    def test_eval_config_expands_nested_runtime_sandbox_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            build_root = Path(temp_dir)
            executable, mods_path = self._create_build(build_root)
            pool_root = build_root / 'pool'
            config_path = build_root / 'eval.yaml'
            config_path.write_text(
                '\n'.join([
                    'env_cfg:',
                    '  env_path: "${AGENTARK_ENV_PATH}"',
                    '  mod_path: "${AGENTARK_MOD_PATH}"',
                    '  runtime_sandbox:',
                    '    enabled: true',
                    '    template_root: "${AGENTARK_RUNTIME_TEMPLATE_ROOT}"',
                    '    template_env_path: "${AGENTARK_ENV_PATH}"',
                    '    template_mod_path: "${AGENTARK_MOD_PATH}"',
                    '    pool_root: "${AGENTARK_RUNTIME_POOL_ROOT}"',
                    '    shared_task_store_path: "${AGENTARK_TASK_STORE_PATH}"',
                    'eval:',
                    '  num_cases: 1',
                ]),
                encoding='utf-8',
            )

            with patch.dict(os.environ, {
                'AGENTARK_ENV_PATH': str(executable),
                'AGENTARK_MOD_PATH': str(mods_path),
                'AGENTARK_RUNTIME_TEMPLATE_ROOT': str(build_root),
                'AGENTARK_TASK_STORE_PATH': '${AGENTARK_MOD_PATH}/all_tasks',
                'AGENTARK_RUNTIME_POOL_ROOT': str(pool_root),
            }, clear=False):
                cfg = load_eval_config(str(config_path))

            sandbox_cfg = cfg['env_cfg']['runtime_sandbox']
            self.assertEqual(cfg['env_cfg']['env_path'], str(executable))
            self.assertEqual(sandbox_cfg['template_root'], str(build_root))
            self.assertEqual(sandbox_cfg['template_env_path'], str(executable))
            self.assertEqual(sandbox_cfg['template_mod_path'], str(mods_path))
            self.assertEqual(Path(sandbox_cfg['shared_task_store_path']), mods_path / 'all_tasks')
            self.assertEqual(sandbox_cfg['pool_root'], str(pool_root))


if __name__ == '__main__':
    unittest.main()
