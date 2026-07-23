import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ArkSubEnvStaticTest(unittest.TestCase):
    def test_unity_player_additional_args_are_forwarded_by_both_wrappers(self):
        for relative_path in (
            Path('src/agent_ark/ark_env/ark_sub_env.py'),
            Path('src/agent_ark/ark_env/direct_env.py'),
        ):
            with self.subTest(path=str(relative_path)):
                tree = ast.parse((ROOT / relative_path).read_text(encoding='utf-8'))
                unity_calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'UnityEnvironment'
                ]
                self.assertTrue(unity_calls)
                for call in unity_calls:
                    keywords = {keyword.arg for keyword in call.keywords}
                    self.assertIn('additional_args', keywords)

    def test_rollout_budget_helpers_are_forwarded_from_envwrapper(self):
        tree = ast.parse((ROOT / 'src' / 'agent_ark' / 'ark_env' / 'ark_sub_env.py').read_text(encoding='utf-8'))
        ark_sub_env = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'ArkSubEnv'
        )
        assigned_names = {
            target.id
            for node in ark_sub_env.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        self.assertIn('_build_llm_visible_prompt', assigned_names)
        self.assertIn('_positive_int_or_none', assigned_names)
        self.assertIn('_build_rollout_budget_prompt', assigned_names)

    def test_initial_observation_helpers_are_forwarded_from_envwrapper(self):
        tree = ast.parse((ROOT / 'src' / 'agent_ark' / 'ark_env' / 'ark_sub_env.py').read_text(encoding='utf-8'))
        ark_sub_env = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'ArkSubEnv'
        )
        assigned_names = {
            target.id
            for node in ark_sub_env.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        self.assertIn('_get_initial_observation_cfg', assigned_names)
        self.assertIn('_build_unity_env_params_payload', assigned_names)
        self.assertIn('_apply_initial_observation_warmup', assigned_names)
        self.assertIn('_attach_image_payloads_to_obs', assigned_names)
        self.assertIn('_get_agent_visual_observations', assigned_names)


if __name__ == '__main__':
    unittest.main()
