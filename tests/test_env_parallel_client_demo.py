import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.tools.env_parallel_client_demo import _obs_summary, _save_obs_images  # noqa: E402


class EnvParallelClientDemoTest(unittest.TestCase):
    def _data_url(self) -> str:
        img = Image.new('RGB', (8, 8), (0, 128, 255))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

    def test_saves_images_from_message_observation(self):
        payload = {
            'obs': {
                'messages': [
                    {'role': 'system', 'content': 'system prompt'},
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': 'observe'},
                            {'type': 'image_url', 'image_url': {'url': self._data_url()}},
                        ],
                    },
                ],
                'skip_infer': False,
            }
        }

        summary = _obs_summary(payload)
        self.assertEqual(summary['camera_count'], 0)
        self.assertEqual(summary['message_image_count'], 1)

        with tempfile.TemporaryDirectory() as tmp:
            paths = _save_obs_images(
                payload,
                image_dir=Path(tmp),
                rollout_index=0,
                task_name='Snake',
                phase='start',
                max_images_per_observation=1,
            )
            self.assertEqual(len(paths), 1)
            saved = Path(paths[0])
            self.assertTrue(saved.exists())
            with Image.open(saved) as img:
                self.assertEqual(img.size, (8, 8))


if __name__ == '__main__':
    unittest.main()
