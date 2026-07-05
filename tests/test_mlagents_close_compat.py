import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mlagents_envs.environment import UnityEnvironment  # noqa: E402
from mlagents_envs.exception import UnityWorkerInUseException  # noqa: E402

import agent_ark.ark_env.direct_env  # noqa: E402,F401


class MlagentsCloseCompatTest(unittest.TestCase):
    def test_atexit_close_ignores_unity_environment_before_communicator_exists(self):
        callbacks = []

        def capture_atexit_callback(callback):
            callbacks.append(callback)

        with patch('atexit.register', side_effect=capture_atexit_callback):
            with patch.object(
                UnityEnvironment,
                '_get_communicator',
                side_effect=UnityWorkerInUseException('synthetic startup failure'),
            ):
                with self.assertRaises(UnityWorkerInUseException):
                    UnityEnvironment(file_name='/tmp/missing-AgentArk.x86_64', base_port=5005, timeout_wait=1)

        self.assertEqual(len(callbacks), 1)
        callbacks[0]()


if __name__ == '__main__':
    unittest.main()
