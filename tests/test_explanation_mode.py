import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.agent.api_agent import APIAgent
from agent_ark.ark_env.direct_env import EnvInfoManager, EnvWrapper


INFO = ROOT / 'src' / 'agent_ark' / 'ark_env' / 'info'
SOURCE = 'public sealed class AgentAction : CodeAction { public override void Start() { Context.CompleteStep(); } }'


def manager_for_mods(mods, overrides=None):
    manager = object.__new__(EnvInfoManager)
    manager.cfg = {'mod_path': str(mods), 'env_config_overrides': overrides or {}}
    manager.mod_path = str(mods)
    manager.task_store_path = manager.resolve_task_store_path(mods)
    manager.task_config = {}
    manager._last_task_name = None
    manager._last_group_seed = None
    return manager


class ExplanationModeTest(unittest.TestCase):
    def test_omitted_and_explicit_legacy_templates_are_unchanged(self):
        for mode, filename in (
            ('func', 'system_prompt_func.txt'),
            ('code', 'system_prompt_code.txt'),
            ('tool', 'system_prompt.txt'),
            (None, 'system_prompt.txt'),
        ):
            for config in ({}, {'prompt': {'explanation_mode': 'legacy_think'}}):
                with self.subTest(mode=mode, config=config):
                    manager = object.__new__(EnvInfoManager)
                    manager.env_config = config
                    self.assertEqual(manager._get_system_prompt(action_mode=mode),
                                     (INFO / filename).read_text(encoding='utf-8'))

    def test_brief_templates_have_ordinary_explanations_and_original_action_contracts(self):
        manager = object.__new__(EnvInfoManager)
        manager.env_config = {'prompt': {'explanation_mode': 'brief_rationale'}}
        for mode, filename, contract in (
            ('func', 'system_prompt_func_brief_rationale.txt', '<tool_call>'),
            ('code', 'system_prompt_code_brief_rationale.txt', 'public sealed class AgentAction : CodeAction'),
            (None, 'system_prompt_brief_rationale.txt', 'MonoBehaviour'),
        ):
            with self.subTest(mode=mode):
                text = manager._get_system_prompt(action_mode=mode)
                self.assertEqual(text, (INFO / filename).read_text(encoding='utf-8'))
                self.assertIn(contract, text)
                self.assertNotIn('<think>', text)
                self.assertNotIn('</think>', text)
                self.assertNotIn('<rationale>', text)

    def test_invalid_explicit_mode_is_not_silently_ignored(self):
        for value in ('brief', '', None, True, [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'prompt.explanation_mode'):
                    EnvInfoManager._get_explanation_mode({'prompt': {'explanation_mode': value}})

    def test_old_baseline_accepts_only_the_new_python_prompt_field(self):
        base = {'prompt': {'language': 'en', 'include_code_wrapper': False}}
        task = {'prompt': {'explanation_mode': 'brief_rationale', 'unknown_setting': 1}}
        before = copy.deepcopy((base, task))
        merged = EnvInfoManager._overlay_env_with_task_config(base, task)
        merged = EnvInfoManager._merge_python_only_task_config(merged, task)
        self.assertEqual(merged['prompt'], {
            'language': 'en', 'include_code_wrapper': False, 'explanation_mode': 'brief_rationale',
        })
        self.assertEqual((base, task), before)

    def test_real_task_switch_materialization_and_operator_precedence(self):
        for base_prompt in ({}, {'language': 'en', 'include_code_wrapper': False}):
            with self.subTest(base_prompt=base_prompt), tempfile.TemporaryDirectory() as directory:
                mods = Path(directory)
                baseline = {
                    'override_by_task': True, 'load_mod_mode': 'task_name',
                    'task_name': 'Baseline', 'action_mode': 'func', 'engine_para': {},
                    'max_attempts': 1, 'max_steps_per_attempt': 2, 'prompt': base_prompt,
                }
                baseline_text = yaml.safe_dump(baseline)
                for name in ('config.yaml', 'config.yaml.bak'):
                    (mods / name).write_text(baseline_text, encoding='utf-8')
                for name, task_prompt in (
                    ('TaskA', {'explanation_mode': 'brief_rationale'}), ('TaskB', {}),
                ):
                    task = mods / 'all_tasks' / name
                    (task / 'cfg').mkdir(parents=True)
                    (task / (name + '.json')).write_text('{}', encoding='utf-8')
                    (task / 'cfg' / 'task_config.yaml').write_text(yaml.safe_dump({
                        'task_name': name, 'max_attempts': 1, 'max_steps_per_attempt': 2,
                        'prompt': task_prompt,
                    }), encoding='utf-8')
                manager = manager_for_mods(mods)
                for task_name, expected in (
                    ('TaskA', 'brief_rationale'), ('TaskB', 'legacy_think'),
                ):
                    manager.reset(task_name=task_name, group_seed=1)
                    effective_yaml = yaml.safe_load((mods / 'config.yaml').read_text(encoding='utf-8'))
                    effective_json = json.loads((mods / 'config.json').read_text(encoding='utf-8'))
                    self.assertEqual(effective_yaml, effective_json)
                    self.assertEqual(EnvInfoManager._get_explanation_mode(effective_yaml), expected)
                    self.assertEqual('<think>' in manager.system_prompt, expected == 'legacy_think')
                manager.cfg['env_config_overrides'] = {'prompt': {'explanation_mode': 'legacy_think'}}
                manager.reset(task_name='TaskA', group_seed=1)
                self.assertEqual(EnvInfoManager._get_explanation_mode(manager.env_config), 'legacy_think')
                self.assertEqual((mods / 'config.yaml.bak').read_text(encoding='utf-8'), baseline_text)
                before_invalid = [(mods / name).read_bytes() for name in ('config.yaml', 'config.json')]
                manager.cfg['env_config_overrides'] = {'prompt': {'explanation_mode': 'invalid'}}
                with self.assertRaisesRegex(ValueError, 'prompt.explanation_mode'):
                    manager.reset(task_name='TaskB', group_seed=1)
                self.assertEqual([(mods / name).read_bytes() for name in ('config.yaml', 'config.json')], before_invalid)

    def test_user_task_text_and_literal_tags_are_never_rewritten(self):
        manager = object.__new__(EnvInfoManager)
        manager.env_config = {'prompt': {'explanation_mode': 'brief_rationale'},
                              'max_attempts': 1, 'max_steps_per_attempt': 2}
        manager.system_prompt = manager._get_system_prompt(action_mode='func')
        env = object.__new__(EnvWrapper)
        env.env_info_mgr = manager
        task = 'Paint the text <think>literal</think> and <rationale>literal</rationale>.'
        visible = env._build_llm_visible_prompt('<task_prompt>' + task + '</task_prompt>')
        self.assertIn(task, visible)

    def test_func_action_extraction_does_not_include_the_explanation(self):
        action = '<tool_call>{"name":"MoveRight","arguments":{"steps":3}}</tool_call>'
        reply = 'I will move right toward the target.\n' + action
        self.assertEqual(APIAgent._extract_action(reply), action)

    def test_brief_exact_code_unwraps_rationale_and_leaves_raw_source_unchanged(self):
        env = object.__new__(EnvWrapper)
        env.env_info_mgr = SimpleNamespace(env_config={
            'action_mode': 'code', 'prompt': {'explanation_mode': 'brief_rationale'},
        })
        env.cfg = {}
        raw_samples = [
            SOURCE,
            '// <code> is a literal in this comment.\n' + SOURCE + '\n// </code>',
            '/* literal <code> */\n' + SOURCE + '\n/* </code> */',
            'using Example;\n' + SOURCE + '\n// literal <think> and <code> </code>',
            'using /* namespace */ System;\n' + SOURCE + '\n// literal <code> </code>',
            'using /* <code> */ System;\n' + SOURCE + '\n// literal </code>',
            '/*\n<code>\n*/\n' + SOURCE + '\n// literal </code>',
            SOURCE.replace('Context.CompleteStep();',
                           'string x = "<code>literal</code><think>literal</think>"; Context.CompleteStep();'),
        ]
        for raw in raw_samples:
            with self.subTest(raw=raw):
                rendered, errors = env._render_func_code_actions({1: raw})
                self.assertEqual(rendered[1], raw)
                self.assertEqual(errors, {})
        for prefix in ('', 'I will finish this operation.\n',
                       'using a moderate force, I will test the response.\n'):
            rendered, errors = env._render_func_code_actions({1: prefix + '<code>\n' + SOURCE + '\n</code>'})
            self.assertEqual(rendered[1], SOURCE)
            self.assertEqual(errors, {})
        rendered, errors = env._render_func_code_actions({1: 'I will finish this operation.\n<code>' + SOURCE + '</code>'})
        self.assertEqual(rendered[1], SOURCE)
        self.assertEqual(errors, {})

    def test_default_exact_code_path_does_not_accept_new_rationale_prefix(self):
        env = object.__new__(EnvWrapper)
        env.env_info_mgr = SimpleNamespace(env_config={'action_mode': 'code'})
        env.cfg = {}
        raw = 'I will finish this operation.\n<code>' + SOURCE + '</code>'
        rendered, errors = env._render_func_code_actions({1: raw})
        self.assertEqual(rendered[1], raw)
        self.assertEqual(errors, {})

    def test_new_prompt_files_are_included_in_distribution(self):
        manifest = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
        for name in ('system_prompt', 'system_prompt_func', 'system_prompt_code'):
            self.assertIn('ark_env/info/' + name + '_brief_rationale.txt', manifest)


if __name__ == '__main__':
    unittest.main()
