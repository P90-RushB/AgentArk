import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark._dotenv import find_agentark_dotenv, load_agentark_dotenv  # noqa: E402


class DotenvLoadingTest(unittest.TestCase):
    def test_loads_dotenv_and_expands_previous_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dotenv_path = root / '.env'
            dotenv_path.write_text(
                '\n'.join([
                    '# local AgentArk paths',
                    'AGENTARK_MOD_PATH=/tmp/agentark/Mods',
                    'AGENTARK_TASK_STORE_PATH=${AGENTARK_MOD_PATH}/all_tasks',
                    'export MLAGENTS_PYTHON_BIN="/opt/ml agents/bin/python"',
                    'COMMENTED_VALUE=value # inline comment',
                ]),
                encoding='utf-8',
            )

            with patch.dict(os.environ, {}, clear=True):
                loaded = load_agentark_dotenv(dotenv_path)

                self.assertEqual(loaded, dotenv_path)
                self.assertEqual(os.environ['AGENTARK_MOD_PATH'], '/tmp/agentark/Mods')
                self.assertEqual(os.environ['AGENTARK_TASK_STORE_PATH'], '/tmp/agentark/Mods/all_tasks')
                self.assertEqual(os.environ['MLAGENTS_PYTHON_BIN'], '/opt/ml agents/bin/python')
                self.assertEqual(os.environ['COMMENTED_VALUE'], 'value')

    def test_existing_environment_wins_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = Path(temp_dir) / '.env'
            dotenv_path.write_text('AGENTARK_MOD_PATH=/from/dotenv\n', encoding='utf-8')

            with patch.dict(os.environ, {'AGENTARK_MOD_PATH': '/from/shell'}, clear=True):
                load_agentark_dotenv(dotenv_path)

                self.assertEqual(os.environ['AGENTARK_MOD_PATH'], '/from/shell')

    def test_can_find_dotenv_from_child_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dotenv_path = root / '.env'
            child = root / 'nested' / 'workdir'
            child.mkdir(parents=True)
            dotenv_path.write_text('AGENTARK_ENV_PATH=/tmp/build/AgentArk.x86_64\n', encoding='utf-8')

            self.assertEqual(find_agentark_dotenv(child), dotenv_path)

    def test_can_find_repo_dotenv_when_cwd_is_outside_repo(self):
        repo_dotenv = ROOT / '.env'
        if not repo_dotenv.exists():
            self.skipTest('repo .env is not present')

        with tempfile.TemporaryDirectory() as temp_dir:
            outside = Path(temp_dir)

            self.assertEqual(find_agentark_dotenv(outside), repo_dotenv)


if __name__ == '__main__':
    unittest.main()
