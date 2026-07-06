import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.ark_sub_env import (  # noqa: E402
    _resolve_cli_action_payloads,
    _resolve_cli_action_step_count,
)


class CliActionSequenceTest(unittest.TestCase):
    def test_single_action_uses_default_step_count(self):
        actions = _resolve_cli_action_payloads('single', None)

        self.assertEqual(actions, ['single'])
        self.assertEqual(
            _resolve_cli_action_step_count(
                default_steps=10,
                requested_steps=None,
                action_payloads=actions,
                has_action_sequence=False,
                option_name='--max-steps',
            ),
            10,
        )

    def test_inline_json_sequence_sets_step_count_to_sequence_length(self):
        actions = _resolve_cli_action_payloads('fallback', '["a1", "a2"]')

        self.assertEqual(actions, ['a1', 'a2'])
        self.assertEqual(
            _resolve_cli_action_step_count(
                default_steps=10,
                requested_steps=None,
                action_payloads=actions,
                has_action_sequence=True,
                option_name='--max-steps',
            ),
            2,
        )

    def test_json_object_actions_field_is_supported(self):
        actions = _resolve_cli_action_payloads('fallback', '{"actions": ["a1", {"tool": "bad"}]}')

        self.assertEqual(actions, ['a1', '{"tool":"bad"}'])

    def test_jsonl_file_allows_json_and_raw_tool_call_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'actions.jsonl'
            path.write_text(
                '\n'.join([
                    '"a1"',
                    '<tool_call>{"name":"Bad","arguments":{}}</tool_call>',
                    '{"name":"Move","arguments":{"x":1}}',
                ]),
                encoding='utf-8',
            )

            actions = _resolve_cli_action_payloads('fallback', str(path))

        self.assertEqual(
            actions,
            [
                'a1',
                '<tool_call>{"name":"Bad","arguments":{}}</tool_call>',
                '{"name":"Move","arguments":{"x":1}}',
            ],
        )

    def test_explicit_step_count_cannot_exceed_sequence_length(self):
        actions = _resolve_cli_action_payloads('fallback', '["a1"]')

        with self.assertRaises(ValueError):
            _resolve_cli_action_step_count(
                default_steps=10,
                requested_steps=2,
                action_payloads=actions,
                has_action_sequence=True,
                option_name='--max-steps',
            )


if __name__ == '__main__':
    unittest.main()
