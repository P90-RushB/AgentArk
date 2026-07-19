from __future__ import annotations

import sys
import unittest
from pathlib import Path

from ._fakes import IMAGE_URL


class MessageHelperTests(unittest.TestCase):

    def test_action_extraction_matches_existing_agentark_api_agent_semantics(self):
        from agentark_swift.messages import extract_action

        cases = [
            (
                '<think>reasoning</think>\n<tool_call>{"name":"ExecutePlan","arguments":{"plan":"R1"}}</tool_call>',
                '<tool_call>{"name":"ExecutePlan","arguments":{"plan":"R1"}}</tool_call>',
            ),
            ("prefix<params> L1, U2 </params>suffix", "<params>L1, U2</params>"),
            ("explanation<code>\nvar x = 1;\n</code>", "var x = 1;"),
            ("raw assistant action", "raw assistant action"),
            ("", None),
        ]
        for assistant, expected in cases:
            with self.subTest(assistant=assistant):
                self.assertEqual(extract_action(assistant), expected)

    def test_full_transcript_response_keeps_only_new_environment_delta(self):
        from agentark_swift.messages import new_environment_messages

        conversation = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "initial"},
            {"role": "assistant", "content": "ExecutePlan R1"},
        ]
        returned = [
            *conversation,
            {"role": "assistant", "content": "ExecutePlan R1"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "new state"},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                ],
            },
        ]

        delta = new_environment_messages(conversation, returned)

        self.assertEqual(len(delta), 1)
        self.assertEqual(delta[0]["role"], "user")
        self.assertEqual(delta[0]["content"][1]["image_url"]["url"], IMAGE_URL)


class PluginRegistrationTests(unittest.TestCase):

    def test_external_plugin_registers_env_and_scheduler(self):
        from agentark_swift.env import AgentArkEnv
        from agentark_swift.scheduler import AgentArkScheduler
        from swift.rollout.gym_env import envs
        from swift.rollout.multi_turn import multi_turns

        import agentark_swift.plugin as plugin

        plugin.register()
        self.assertIs(envs[plugin.ENV_NAME], AgentArkEnv)
        self.assertIs(multi_turns[plugin.SCHEDULER_NAME], AgentArkScheduler)

    def test_swift_import_external_file_loads_real_plugin_path(self):
        from agentark_swift.env import AgentArkEnv
        from agentark_swift.scheduler import AgentArkScheduler
        from swift.rollout.gym_env import envs
        from swift.rollout.multi_turn import multi_turns
        from swift.utils import import_external_file

        plugin_path = Path(__file__).resolve().parents[1] / "src" / "agentark_swift" / "plugin.py"
        previous_top_level_plugin = sys.modules.pop("plugin", None)
        envs.pop("agentark", None)
        multi_turns.pop("agentark_scheduler", None)
        try:
            loaded = import_external_file(str(plugin_path))
            self.assertEqual(Path(loaded.__file__).resolve(), plugin_path)
            self.assertIs(envs["agentark"], AgentArkEnv)
            self.assertIs(multi_turns["agentark_scheduler"], AgentArkScheduler)
        finally:
            sys.modules.pop("plugin", None)
            if previous_top_level_plugin is not None:
                sys.modules["plugin"] = previous_top_level_plugin


if __name__ == "__main__":
    unittest.main()
