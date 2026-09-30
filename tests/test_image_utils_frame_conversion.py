import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.utils.image_utils import _deal_frame_array, env_arr_to_pil_image  # noqa: E402


class FrameConversionTest(unittest.TestCase):
    def test_channel_last_single_channel_becomes_grayscale(self):
        # ML-Agents grayscale/depth cameras deliver (H, W, 1). Pillow cannot
        # build an image from (H, W, 1), so it must be squeezed to (H, W).
        arr = np.zeros((84, 84, 1), dtype=np.uint8)
        self.assertEqual(_deal_frame_array(arr).shape, (84, 84))
        img = env_arr_to_pil_image(arr)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, 'L')
        self.assertEqual(img.size, (84, 84))

    def test_channel_first_single_channel_becomes_grayscale(self):
        arr = np.zeros((1, 4, 5), dtype=np.uint8)
        self.assertEqual(_deal_frame_array(arr).shape, (4, 5))
        img = env_arr_to_pil_image(arr)
        self.assertEqual(img.mode, 'L')
        self.assertEqual(img.size, (5, 4))

    def test_float_single_channel_is_scaled_and_squeezed(self):
        arr = np.ones((6, 7, 1), dtype=np.float32)
        img = env_arr_to_pil_image(arr)
        self.assertEqual(img.mode, 'L')
        self.assertEqual(img.size, (7, 6))

    def test_rgb_channel_first_is_unchanged(self):
        arr = np.zeros((3, 4, 5), dtype=np.uint8)
        self.assertEqual(_deal_frame_array(arr).shape, (4, 5, 3))
        self.assertEqual(env_arr_to_pil_image(arr).mode, 'RGB')

    def test_rgb_channel_last_is_unchanged(self):
        arr = np.zeros((84, 84, 3), dtype=np.uint8)
        self.assertEqual(_deal_frame_array(arr).shape, (84, 84, 3))
        self.assertEqual(env_arr_to_pil_image(arr).mode, 'RGB')

    def test_rgba_channel_last_is_unchanged(self):
        arr = np.zeros((84, 84, 4), dtype=np.uint8)
        self.assertEqual(env_arr_to_pil_image(arr).mode, 'RGBA')

    def test_plain_2d_grayscale_is_unchanged(self):
        arr = np.zeros((4, 5), dtype=np.uint8)
        self.assertEqual(_deal_frame_array(arr).shape, (4, 5))
        self.assertEqual(env_arr_to_pil_image(arr).mode, 'L')


if __name__ == '__main__':
    unittest.main()
