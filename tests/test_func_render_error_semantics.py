import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.direct_env import EnvWrapper  # noqa: E402


class FuncRenderErrorSemanticsTest(unittest.TestCase):
    def test_render_error_is_returned_as_compile_style_step_message(self):
        env = object.__new__(EnvWrapper)
        env.ml_unity_id_map = {1: 0}
        env._get_action_mode = lambda: 'func'

        def fail_render(unity_id, params_text):
            raise ValueError('Unknown tool: MoveLeft')

        env._render_func_wrapper = fail_render

        rendered, errors = env._render_func_code_actions(
            {1: '<tool_call>{"name":"MoveLeft","arguments":{}}</tool_call>'},
            log_prefix='test',
        )

        self.assertEqual(rendered, {})
        self.assertIn(1, errors)
        self.assertIn('[compile] Error:', errors[1])
        self.assertIn('Invalid assistant tool/function-call action', errors[1])
        self.assertIn('no environment action was run', errors[1])
        self.assertIn('not a Unity environment rendering/image problem', errors[1])
        self.assertIn('Unknown tool: MoveLeft', errors[1])

    def test_render_error_can_merge_with_unity_step_messages(self):
        merged = EnvWrapper._merge_step_message_parts(
            '[compile] Error: Python render failed',
            ['[compile] Error: Unity compile failed'],
        )

        self.assertEqual(
            merged,
            [
                '[compile] Error: Python render failed',
                '[compile] Error: Unity compile failed',
            ],
        )


if __name__ == '__main__':
    unittest.main()
