import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.direct_env import EnvWrapper  # noqa: E402
from agent_ark.ark_env.ark_sub_env import ArkSubEnv  # noqa: E402


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

    def test_missing_rendered_action_explicitly_clears_previous_unity_action(self):
        class FakeCodeActionChannel:
            def __init__(self):
                self.sent = []

            def send_code_act(self, run_flag, code_str):
                self.sent.append((run_flag, code_str))

        channel = FakeCodeActionChannel()
        env = object.__new__(EnvWrapper)
        env.ml_unity_id_map = {1: 0}
        env.code_act_channels = [channel]

        env.send_code_act(agent_id=[1], code_act={1: 'valid action A'})
        env.send_code_act(agent_id=[1], code_act={})

        self.assertEqual(
            channel.sent,
            [
                (True, 'valid action A'),
                (False, ''),
            ],
        )

    def test_mixed_agent_dispatch_clears_only_agent_without_executable_action(self):
        class FakeCodeActionChannel:
            def __init__(self):
                self.sent = []

            def send_code_act(self, run_flag, code_str):
                self.sent.append((run_flag, code_str))

        channels = [FakeCodeActionChannel(), FakeCodeActionChannel()]
        env = object.__new__(EnvWrapper)
        env.ml_unity_id_map = {10: 0, 20: 1}
        env.code_act_channels = channels

        env.send_code_act(agent_id=[10, 20], code_act={10: 'valid action'})

        self.assertEqual(channels[0].sent, [(True, 'valid action')])
        self.assertEqual(channels[1].sent, [(False, '')])

    def test_valid_action_then_malformed_action_does_not_replay_for_either_wrapper(self):
        class StopAfterSideChannelDispatch(Exception):
            pass

        class FakeCodeActionChannel:
            def __init__(self):
                self.sent = []

            def send_code_act(self, run_flag, code_str):
                self.sent.append((run_flag, code_str))

        class FakeActionSpec:
            @staticmethod
            def empty_action(count):
                return [('dummy',)] * count

        class FakeUnityEnv:
            def __init__(self, channel):
                self.channel = channel

            def set_actions(self, behavior_name, actions):
                self.behavior_name = behavior_name
                self.actions = actions

            def step(self):
                self.last_side_channel_message = self.channel.sent[-1]
                raise StopAfterSideChannelDispatch

        malformed = '<tool_call>{"name":"UnknownTool","arguments":{}}</tool_call>'

        for wrapper_type in (EnvWrapper, ArkSubEnv):
            with self.subTest(wrapper_type=wrapper_type.__name__):
                channel = FakeCodeActionChannel()
                env = object.__new__(wrapper_type)
                env.ml_unity_id_map = {1: 0}
                env.code_act_channels = [channel]
                env.env_spec = SimpleNamespace(action_spec=FakeActionSpec())
                env.env = FakeUnityEnv(channel)
                env.behavior_name = 'test'
                env._get_action_mode = lambda: 'func'
                env._render_func_wrapper = lambda unity_id, params_text: (_ for _ in ()).throw(
                    ValueError('Unknown tool: UnknownTool')
                )

                env.send_code_act(agent_id=[1], code_act={1: 'valid action A'})
                with self.assertRaises(StopAfterSideChannelDispatch):
                    wrapper_type.step(env, {1: malformed})

                self.assertEqual(
                    channel.sent,
                    [
                        (True, 'valid action A'),
                        (False, ''),
                    ],
                )
                self.assertEqual(env.env.last_side_channel_message, (False, ''))


if __name__ == '__main__':
    unittest.main()
