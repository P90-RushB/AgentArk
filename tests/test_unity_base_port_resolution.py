import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.direct_env import (  # noqa: E402
    EnvWrapper,
    _resolve_configured_unity_base_port,
)


class UnityBasePortResolutionTest(unittest.TestCase):
    def test_editor_uses_editor_environment_port_before_mods_config(self):
        with patch.dict(os.environ, {
            'AGENTARK_EDITOR_BASE_PORT': '5104',
            'AGENTARK_PLAYER_BASE_PORT': '5200',
        }, clear=False):
            port = _resolve_configured_unity_base_port(
                {'env_path': None},
                {'base_port': 5004},
            )

        self.assertEqual(port, 5104)

    def test_packaged_player_uses_player_environment_port_before_mods_config(self):
        with patch.dict(os.environ, {
            'AGENTARK_EDITOR_BASE_PORT': '5104',
            'AGENTARK_PLAYER_BASE_PORT': '5200',
        }, clear=False):
            port = _resolve_configured_unity_base_port(
                {'env_path': '/runtime/AgentArk.x86_64'},
                {'base_port': 5005},
            )

        self.assertEqual(port, 5200)

    def test_explicit_config_wins_over_environment(self):
        with patch.dict(os.environ, {'AGENTARK_PLAYER_BASE_PORT': '5200'}, clear=False):
            direct_port = _resolve_configured_unity_base_port(
                {'env_path': '/runtime/AgentArk.x86_64', 'base_port': 5300},
                {'base_port': 5005},
            )
            override_port = _resolve_configured_unity_base_port(
                {
                    'env_path': '/runtime/AgentArk.x86_64',
                    'env_config_overrides': {'base_port': 5400},
                },
                {'base_port': 5005},
            )

        self.assertEqual(direct_port, 5300)
        self.assertEqual(override_port, 5400)

    def test_mods_config_and_default_remain_fallbacks(self):
        environment = {
            'AGENTARK_EDITOR_BASE_PORT': '',
            'AGENTARK_PLAYER_BASE_PORT': '',
        }

        self.assertEqual(
            _resolve_configured_unity_base_port(
                {'env_path': None},
                {'base_port': 5004},
                environ=environment,
            ),
            5004,
        )
        self.assertEqual(
            _resolve_configured_unity_base_port(
                {'env_path': '/runtime/AgentArk.x86_64'},
                {},
                environ=environment,
            ),
            5005,
        )

    def test_worker_offset_is_applied_after_environment_port_resolution(self):
        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper.cfg = {
            'env_path': '/runtime/AgentArk.x86_64',
            'worker_index': 2,
        }
        wrapper.env_info_mgr = SimpleNamespace(env_config={'env_id': 1})

        with patch.dict(os.environ, {'AGENTARK_PLAYER_BASE_PORT': '5200'}, clear=False):
            alloc = wrapper._get_port_alloc_config()

        self.assertEqual(alloc['start_port'], 5203)

    def test_invalid_environment_port_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'AGENTARK_PLAYER_BASE_PORT'):
            _resolve_configured_unity_base_port(
                {'env_path': '/runtime/AgentArk.x86_64'},
                {},
                environ={'AGENTARK_PLAYER_BASE_PORT': 'not-a-port'},
            )


if __name__ == '__main__':
    unittest.main()
