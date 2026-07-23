import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.direct_env import EnvWrapper  # noqa: E402


class MultiCameraObservationTest(unittest.TestCase):
    @staticmethod
    def _wrapper(*shapes):
        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper.env_spec = SimpleNamespace(
            observation_specs=[SimpleNamespace(shape=shape) for shape in shapes]
        )
        return wrapper

    @staticmethod
    def _steps(*observations):
        return SimpleNamespace(
            obs=list(observations),
            agent_id_to_index={41: 1},
        )

    @staticmethod
    def _decision_wrapper():
        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper.env_info_mgr = SimpleNamespace(env_config={'obs_mode': 'decision'})
        return wrapper

    @staticmethod
    def _video_wrapper():
        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper.env_info_mgr = SimpleNamespace(env_config={
            'obs_mode': 'video',
            'env_wrapper_cfg': {
                'video_frame_selection': 'transition_and_decision',
            },
        })
        return wrapper

    def test_collects_every_visual_observation_and_ignores_non_visual_sensors(self):
        vector = np.array([[10, 0], [11, 1]], dtype=np.float32)
        camera_0 = np.full((2, 3, 4, 3), 20, dtype=np.float32)
        buffer_sensor = np.full((2, 5, 6), 30, dtype=np.float32)
        camera_1 = np.full((2, 3, 4, 3), 40, dtype=np.float32)
        wrapper = self._wrapper((2,), (3, 4, 3), (5, 6), (3, 4, 3))

        visual_observations = wrapper._get_agent_visual_observations(
            self._steps(vector, camera_0, buffer_sensor, camera_1),
            41,
        )

        self.assertEqual(len(visual_observations), 2)
        np.testing.assert_array_equal(visual_observations[0], camera_0[1])
        np.testing.assert_array_equal(visual_observations[1], camera_1[1])

    def test_single_camera_shape_remains_a_one_item_camera_list(self):
        camera = np.full((2, 2, 3, 3), 50, dtype=np.float32)
        agent_id_vector = np.array([[0], [1]], dtype=np.float32)
        wrapper = self._wrapper((2, 3, 3), (1,))

        visual_observations = wrapper._get_agent_visual_observations(
            self._steps(camera, agent_id_vector),
            41,
        )

        self.assertEqual(len(visual_observations), 1)
        np.testing.assert_array_equal(visual_observations[0], camera[1])

    def test_post_process_keeps_single_camera_and_frame_nesting(self):
        camera = np.full((2, 3, 3), 0.25, dtype=np.float32)
        obs = {
            41: {
                'vis': [camera],
                'video_raw': None,
                'video_pil': None,
            }
        }

        processed = self._decision_wrapper().post_process_obs(obs)

        self.assertEqual(len(processed[41]['vis']), 1)
        self.assertEqual(len(processed[41]['vis'][0]), 1)
        self.assertIsInstance(processed[41]['vis'][0][0], Image.Image)

    def test_post_process_keeps_each_camera_as_its_own_frame_list(self):
        camera_0 = np.full((2, 3, 3), 0.25, dtype=np.float32)
        camera_1 = np.full((2, 3, 3), 0.75, dtype=np.float32)
        obs = {
            41: {
                'vis': [camera_0, camera_1],
                'video_raw': None,
                'video_pil': None,
            }
        }

        processed = self._decision_wrapper().post_process_obs(obs)

        self.assertEqual([len(frames) for frames in processed[41]['vis']], [1, 1])
        self.assertTrue(all(isinstance(frames[0], Image.Image) for frames in processed[41]['vis']))

    def test_video_post_process_keeps_multiple_frames_grouped_by_camera(self):
        decision_0 = np.full((2, 3, 3), 0.25, dtype=np.float32)
        decision_1 = np.full((2, 3, 3), 0.75, dtype=np.float32)
        transition_0 = [Image.new('RGB', (3, 2)), Image.new('RGB', (3, 2))]
        transition_1 = [Image.new('RGB', (3, 2))]
        obs = {
            41: {
                'vis': [decision_0, decision_1],
                'video_raw': {'unused': True},
                'video_pil': {0: transition_0, 1: transition_1},
            }
        }

        processed = self._video_wrapper().post_process_obs(obs)

        self.assertEqual([len(frames) for frames in processed[41]['vis']], [3, 2])
        self.assertIs(processed[41]['vis'][0][0], transition_0[0])
        self.assertIs(processed[41]['vis'][1][0], transition_1[0])
        self.assertTrue(all(isinstance(frames[-1], Image.Image) for frames in processed[41]['vis']))

    def test_rejects_mismatched_observation_metadata(self):
        wrapper = self._wrapper((2, 3, 3), (1,))
        steps = self._steps(np.zeros((2, 2, 3, 3), dtype=np.float32))

        with self.assertRaisesRegex(RuntimeError, 'metadata does not match'):
            wrapper._get_agent_visual_observations(steps, 41)

    def test_rejects_behavior_without_a_visual_sensor(self):
        wrapper = self._wrapper((2,), (4, 5))
        steps = self._steps(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 4, 5), dtype=np.float32),
        )

        with self.assertRaisesRegex(RuntimeError, 'rank-3 visual observation'):
            wrapper._get_agent_visual_observations(steps, 41)


if __name__ == '__main__':
    unittest.main()
