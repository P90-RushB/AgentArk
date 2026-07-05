import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.utils.parse_utils import (  # noqa: E402
    build_llm_visible_prompt,
    parse_system_task_prompt,
    parse_task_prompt_payload,
    render_tool_call_to_csharp,
)


MANIFEST = {
    'version': 1,
    'tools': [
        {
            'name': 'ExecutePlan',
            'kind': 'method',
            'access': 'call',
            'arguments': [
                {
                    'name': 'plan',
                    'type': 'string',
                    'required': True,
                    'pattern': r'^\s*[LRUD]\s*\d+(\s*,\s*[LRUD]\s*\d+){0,2}\s*$',
                },
            ],
        },
        {
            'name': 'SetSpeed',
            'kind': 'method',
            'access': 'call',
            'arguments': [
                {'name': 'enabled', 'type': 'bool', 'required': True},
                {'name': 'speed', 'type': 'float', 'required': True},
            ],
        },
        {
            'name': 'PushForward',
            'kind': 'method',
            'access': 'call',
            'arguments': [
                {'name': 'forceScale', 'type': 'float', 'required': False, 'default': '1'},
            ],
        },
        {
            'name': 'manualStepMode',
            'kind': 'property',
            'access': 'get/set',
            'arguments': [
                {'name': 'value', 'type': 'bool', 'required': True},
            ],
        },
    ],
}


class ToolCallPromptUtilsTest(unittest.TestCase):
    def test_parse_payload_hides_internal_blocks(self):
        raw = '''<task_prompt>
Task body
<tool_docs>Available tools...</tool_docs>
</task_prompt>
<tool_manifest>{"version":1,"tools":[]}</tool_manifest>
<code_wrapper>SECRET_TEMPLATE</code_wrapper>'''

        payload = parse_task_prompt_payload(raw)
        visible = build_llm_visible_prompt(raw, system_prompt='SYS')

        self.assertEqual(payload['tool_manifest'], {'version': 1, 'tools': []})
        self.assertEqual(payload['code_wrapper'], 'SECRET_TEMPLATE')
        self.assertIn('<tool_docs>Available tools...</tool_docs>', payload['task_prompt'])
        self.assertNotIn('SECRET_TEMPLATE', visible)
        self.assertNotIn('<tool_manifest>', visible)

    def test_parse_payload_can_include_code_wrapper_when_requested(self):
        raw = '''<system_prompt>SYS_FROM_PAYLOAD</system_prompt>
Task body
<tool_manifest>{"version":1,"tools":[]}</tool_manifest>
<code_wrapper>SECRET_TEMPLATE</code_wrapper>'''

        hidden = build_llm_visible_prompt(raw)
        visible = build_llm_visible_prompt(raw, include_code_wrapper=True)

        self.assertTrue(hidden.startswith('SYS_FROM_PAYLOAD'))
        self.assertNotIn('SECRET_TEMPLATE', hidden)
        self.assertIn('<code_wrapper>', visible)
        self.assertIn('SECRET_TEMPLATE', visible)

    def test_language_filter_preserves_tool_docs(self):
        raw = '''<system_prompt>SYS</system_prompt>
<task_prompt>
<english_ver>English task.</english_ver>
<chinese_ver>中文任务。</chinese_ver>
<tool_docs>Available tools: ExecutePlan</tool_docs>
</task_prompt>
<tool_manifest>{"version":1,"tools":[]}</tool_manifest>
<code_wrapper>SECRET_TEMPLATE</code_wrapper>'''

        system_prompt, task_prompt = parse_system_task_prompt(raw, prefer_english_lang=False)
        payload = parse_task_prompt_payload(raw, prefer_english_lang=False)
        visible = build_llm_visible_prompt(raw, prefer_english_lang=False)

        self.assertEqual(system_prompt, 'SYS')
        self.assertIn('中文任务。', task_prompt)
        self.assertNotIn('English task.', task_prompt)
        self.assertIn('<tool_docs>Available tools: ExecutePlan</tool_docs>', task_prompt)
        self.assertIn('<tool_docs>Available tools: ExecutePlan</tool_docs>', payload['task_prompt'])
        self.assertIn('<tool_docs>Available tools: ExecutePlan</tool_docs>', visible)
        self.assertNotIn('<tool_manifest>', visible)
        self.assertNotIn('SECRET_TEMPLATE', visible)

    def test_dynamic_context_is_not_static_task_prompt(self):
        raw = '''<system_prompt>SYS</system_prompt>
<task_prompt>
Task body
<reset_context>Goal inside task should not become static.</reset_context>
</task_prompt>
<reset_context>Goal: red=2 yellow=1 blue=0</reset_context>
<step_context>Door opened</step_context>'''

        payload = parse_task_prompt_payload(raw)
        visible = build_llm_visible_prompt(raw)

        self.assertIn('Task body', payload['task_prompt'])
        self.assertNotIn('Goal inside task should not become static', payload['task_prompt'])
        self.assertEqual(payload['reset_context'], 'Goal inside task should not become static.\n\nGoal: red=2 yellow=1 blue=0')
        self.assertEqual(payload['step_context'], 'Door opened')
        self.assertIn('Task body', visible)
        self.assertNotIn('Goal: red=2', visible)

    def test_render_method_tool_call_orders_arguments_from_manifest(self):
        code = render_tool_call_to_csharp(
            '<tool_call>{"name":"SetSpeed","arguments":{"speed":0.5,"enabled":true}}</tool_call>',
            MANIFEST,
        )

        self.assertIn('router.Call("SetSpeed", true, 0.5);', code)
        self.assertNotIn('0.5f', code)

    def test_render_string_method_tool_call(self):
        code = render_tool_call_to_csharp(
            '<tool_call>{"name":"ExecutePlan","arguments":{"plan":"L2,U3"}}</tool_call>',
            MANIFEST,
        )

        self.assertIn('router.Call("ExecutePlan", "L2,U3");', code)

    def test_render_optional_argument_default(self):
        code = render_tool_call_to_csharp(
            '<tool_call>{"name":"PushForward","arguments":{}}</tool_call>',
            MANIFEST,
        )

        self.assertIn('router.Call("PushForward", 1.0);', code)

    def test_render_property_set_tool_call(self):
        code = render_tool_call_to_csharp(
            '<tool_call>{"name":"manualStepMode","arguments":{"value":true}}</tool_call>',
            MANIFEST,
        )

        self.assertIn('router.Set("manualStepMode", true);', code)

    def test_rejects_unknown_arguments(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected argument'):
            render_tool_call_to_csharp(
                '<tool_call>{"name":"ExecutePlan","arguments":{"plan":"L1","extra":1}}</tool_call>',
                MANIFEST,
            )

    def test_rejects_missing_required_arguments(self):
        with self.assertRaisesRegex(ValueError, 'Required argument missing'):
            render_tool_call_to_csharp(
                '<tool_call>{"name":"ExecutePlan","arguments":{}}</tool_call>',
                MANIFEST,
            )

    def test_rejects_pattern_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'pattern'):
            render_tool_call_to_csharp(
                '<tool_call>{"name":"ExecutePlan","arguments":{"plan":"L1,U2,R3,D4"}}</tool_call>',
                MANIFEST,
            )


if __name__ == '__main__':
    unittest.main()
