import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.direct_env import EnvWrapper  # noqa: E402
from agent_ark.ark_env.ark_sub_env import ArkSubEnv  # noqa: E402


class FakeSteps:
    def __init__(self, agent_ids):
        self.agent_id_to_index = {
            int(agent_id): index for index, agent_id in enumerate(agent_ids)
        }

    def __len__(self):
        return len(self.agent_id_to_index)


class FakeUnityEnv:
    def __init__(self, step_batches):
        self.step_batches = list(step_batches)
        self.step_index = 0
        self.advance_count = 0

    def get_steps(self, behavior_name):
        return self.step_batches[self.step_index]

    def step(self):
        self.advance_count += 1
        if self.step_index + 1 < len(self.step_batches):
            self.step_index += 1


class FakeCodeActionChannel:
    def __init__(self, messages):
        self.messages = list(messages)
        self.clear_count = 0

    def get_step_msgs(self):
        return list(self.messages)

    def clear_step_msgs(self):
        self.clear_count += 1
        self.messages.clear()


class ResetStepBatchSemanticsTest(unittest.TestCase):
    def _build_wrapper(self, wrapper_type, *step_batches):
        wrapper = object.__new__(wrapper_type)
        wrapper.env_num = 1
        wrapper.behavior_name = 'test'
        wrapper.env = FakeUnityEnv(
            [(FakeSteps(decision_ids), FakeSteps(terminal_ids))
             for decision_ids, terminal_ids in step_batches]
        )
        wrapper.code_act_channels = [FakeCodeActionChannel(['<task_prompt>ready</task_prompt>'])]
        wrapper._filter_task_prompt_language = lambda message: message
        return wrapper

    def test_complete_decision_batch_accepts_residual_terminal_for_both_wrappers(self):
        for wrapper_type in (EnvWrapper, ArkSubEnv):
            with self.subTest(wrapper_type=wrapper_type.__name__):
                wrapper = self._build_wrapper(wrapper_type, ([101], [99]))

                prompt = wrapper_type._get_task_prompt_after_reset(wrapper)

                self.assertEqual(prompt, {0: '<task_prompt>ready</task_prompt>'})
                self.assertEqual(wrapper.code_act_channels[0].clear_count, 1)
                self.assertEqual(wrapper.env.advance_count, 0)

    def test_terminal_only_batch_is_drained_until_new_decision_arrives(self):
        for wrapper_type in (EnvWrapper, ArkSubEnv):
            with self.subTest(wrapper_type=wrapper_type.__name__):
                wrapper = self._build_wrapper(
                    wrapper_type,
                    ([], [99]),
                    ([101], []),
                )

                prompt = wrapper_type._get_task_prompt_after_reset(wrapper)

                self.assertEqual(prompt, {0: '<task_prompt>ready</task_prompt>'})
                self.assertEqual(wrapper.env.advance_count, 1)

    def test_empty_terminal_batch_keeps_existing_reset_behavior(self):
        wrapper = self._build_wrapper(EnvWrapper, ([101], []))

        prompt = EnvWrapper._get_task_prompt_after_reset(wrapper)

        self.assertEqual(prompt, {0: '<task_prompt>ready</task_prompt>'})

    def test_residual_terminal_does_not_hide_incomplete_new_decision_batch(self):
        wrapper = self._build_wrapper(EnvWrapper, ([], [99]))

        with self.assertRaisesRegex(AssertionError, 'did not produce a complete new decision batch'):
            EnvWrapper._await_reset_step_batch(wrapper, max_advance_steps=1)

        self.assertEqual(wrapper.env.advance_count, 1)


if __name__ == '__main__':
    unittest.main()
