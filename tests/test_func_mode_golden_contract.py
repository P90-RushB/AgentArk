import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.direct_env import EnvWrapper  # noqa: E402
from agent_ark.side_channels.agent_raw_bytes_channel import AgentRawBytesChannel  # noqa: E402


class FuncModeGoldenContractTest(unittest.TestCase):
    """Byte-for-byte locks for the existing Func action transport and rendering."""

    def test_tool_call_renders_exact_csharp_utf8_bytes(self):
        env = object.__new__(EnvWrapper)
        env._tool_manifest_by_unity_id = {
            7: {
                'tools': [
                    {
                        'name': 'PlaceLabel',
                        'kind': 'method',
                        'access': 'call',
                        'arguments': [
                            {'name': 'label', 'type': 'string', 'required': True},
                            {'name': 'count', 'type': 'integer', 'required': True},
                            {'name': 'enabled', 'type': 'boolean', 'required': True},
                        ],
                    }
                ]
            }
        }
        env._code_wrapper_by_unity_id = {}

        actual = env._render_func_wrapper(
            7,
            '<tool_call>{"name":"PlaceLabel","arguments":'
            '{"enabled":true,"label":"门\\n\\\"A\\\"\\\\","count":3}}</tool_call>',
        ).encode('utf-8')

        expected = (
            'using UnityEngine;\n'
            'public class ArkAct_Step0 : MonoBehaviour\n'
            '{\n'
            '    void Start()\n'
            '    {\n'
            '        var router = GetComponent<ActRouter>();\n'
            '        router.Call("PlaceLabel", "门\\n\\\"A\\\"\\\\", 3, true);\n'
            '    }\n'
            '}\n'
        ).encode('utf-8')
        self.assertEqual(actual, expected)

        env.ml_unity_id_map = {41: 7}
        env.code_act_channels = [None] * 7 + [AgentRawBytesChannel(agent_id=7)]
        env._get_action_mode = lambda: 'func'
        action = (
            '<tool_call>{"name":"PlaceLabel","arguments":'
            '{"enabled":true,"label":"门\\n\\\"A\\\"\\\\","count":3}}</tool_call>'
        )
        rendered, errors = env._render_func_code_actions({41: action})
        env.send_code_act(agent_id=list(rendered), code_act=rendered)
        self.assertEqual(errors, {})
        self.assertEqual(
            list(env.code_act_channels[7].message_queue),
            [b'[code_act]1:' + expected],
        )

    def test_legacy_params_wrapper_renders_exact_csharp_utf8_bytes(self):
        env = object.__new__(EnvWrapper)
        env._tool_manifest_by_unity_id = {}
        env._code_wrapper_by_unity_id = {
            3: (
                'TEXT={{text}}\n'
                'FLAG={{flag}}\n'
                'NONE={{none}}\n'
                'COUNT={{count}}\n'
                'UNTOUCHED={{missing}}\n'
            )
        }

        actual = env._render_func_wrapper(
            3,
            '<params>{"text":"门\\n\\\"A\\\"\\\\","flag":true,"none":null,"count":3}</params>',
        ).encode('utf-8')

        expected = (
            'TEXT="门\\n\\\"A\\\"\\\\"\n'
            'FLAG=true\n'
            'NONE=null\n'
            'COUNT=3\n'
            'UNTOUCHED={{missing}}\n'
        ).encode('utf-8')
        self.assertEqual(actual, expected)

    def test_func_batch_render_preserves_agent_mapping_and_exact_bytes(self):
        env = object.__new__(EnvWrapper)
        env.ml_unity_id_map = {11: 101, 22: 202}
        env._get_action_mode = lambda: 'func'
        env._render_func_wrapper = lambda unity_id, payload: f'{unity_id}|{payload}'
        actions = {22: 'second', 11: 'first'}

        rendered, errors = env._render_func_code_actions(actions, log_prefix='golden')

        self.assertEqual(errors, {})
        self.assertEqual(
            [(key, value.encode('utf-8')) for key, value in rendered.items()],
            [(22, b'202|second'), (11, b'101|first')],
        )

    def test_non_func_non_code_modes_are_legacy_passthrough(self):
        for mode in ('tool', '', 'unknown'):
            with self.subTest(mode=mode):
                env = object.__new__(EnvWrapper)
                env._get_action_mode = lambda selected=mode: selected
                actions = {5: 'raw source'}
                rendered, errors = env._render_func_code_actions(actions)
                self.assertIs(rendered, actions)
                self.assertEqual(errors, {})

        env = object.__new__(EnvWrapper)
        env.env_info_mgr = SimpleNamespace(env_config={})
        env.cfg = {}
        default_actions = {6: 'default mode remains raw source'}
        self.assertEqual(env._get_action_mode(), 'code')
        rendered, errors = env._render_func_code_actions(default_actions)
        self.assertEqual(rendered, default_actions)
        self.assertEqual(errors, {})

    def test_code_action_side_channel_uses_exact_guid_and_wire_bytes(self):
        channel = AgentRawBytesChannel(agent_id=42)
        self.assertEqual(str(channel.channel_id), '621f0a70-4f87-11ea-a6bf-00000000002a')

        channel.send_code_act(True, '类 A\n{ "门"; }')
        channel.send_code_act(False, None)

        self.assertEqual(
            list(channel.message_queue),
            [
                '[code_act]1:类 A\n{ "门"; }'.encode('utf-8'),
                b'[code_act]0:',
            ],
        )

    def test_incoming_parser_preserves_colons_and_utf8_exactly(self):
        payload = '[code_act]1:public class A { string s = "门:a"; }'.encode('utf-8')
        self.assertEqual(
            AgentRawBytesChannel._parse_payload(payload),
            (True, 'public class A { string s = "门:a"; }'),
        )

    def test_func_render_failure_does_not_queue_a_sticky_field_clear(self):
        env = object.__new__(EnvWrapper)
        env.ml_unity_id_map = {9: 0}
        env.code_act_channels = [AgentRawBytesChannel(agent_id=0)]
        env._get_action_mode = lambda: 'func'

        def fail_render(unity_id, params_text):
            raise ValueError('Unknown tool: MoveLeft')

        env._render_func_wrapper = fail_render
        rendered, errors = env._render_func_code_actions(
            {9: '<tool_call>{"name":"MoveLeft","arguments":{}}</tool_call>'},
            log_prefix='golden',
        )
        env.send_code_act(agent_id=list(rendered), code_act=rendered)

        self.assertEqual(rendered, {})
        self.assertEqual(
            errors,
            {
                9: (
                    '[compile] Error: Invalid assistant tool/function-call action before Unity execution. '
                    'The previous assistant output could not be converted into executable Unity code, so no environment action was run. '
                    'Fix the assistant tool_call format, tool name, or arguments; this is not a Unity environment rendering/image problem. '
                    'Details: ValueError: Unknown tool: MoveLeft'
                )
            },
        )
        self.assertEqual(list(env.code_act_channels[0].message_queue), [])


if __name__ == '__main__':
    unittest.main()
