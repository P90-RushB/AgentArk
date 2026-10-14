import base64
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.serving.protocol import (  # noqa: E402
    EnvStartPayload,
    EnvStepPayload,
    as_json_dict,
    decode_obs,
    encode_obs,
)


class AttemptImageProtocolTest(unittest.TestCase):
    @staticmethod
    def _image():
        image = Image.new('RGBA', (3, 2))
        image.putdata([
            (3, 11, 27, 0), (43, 59, 71, 127), (89, 101, 113, 255),
            (131, 149, 167, 19), (181, 197, 211, 201), (223, 239, 251, 255),
        ])
        return image

    def test_raw_additional_images_round_trip_start_and_step_json_payloads(self):
        image = self._image()
        digest = hashlib.sha256(b'original image file bytes').hexdigest()
        for include_vis in (False, True):
            for payload_kind in ('start', 'step'):
                with self.subTest(include_vis=include_vis, payload_kind=payload_kind):
                    obs = {
                        'step_msg': 'first actionable observation',
                        'attempt_initial_images': [{'index': 1, 'image': image, 'sha256': digest}],
                    }
                    if include_vis:
                        obs['vis'] = [[Image.new('RGB', (2, 2), color=(17, 31, 47))]]
                    encoded = encode_obs(obs)
                    if payload_kind == 'start':
                        payload = EnvStartPayload(env_id='test-env', unity_id=7, obs=encoded, info={})
                    else:
                        payload = EnvStepPayload(unity_id=7, obs=encoded, reward=0.0, done=False, info={})
                    wire_text = json.dumps(as_json_dict(payload))
                    self.assertNotIn('PIL.Image', wire_text)
                    wire = json.loads(wire_text)
                    entry = wire['obs']['attempt_initial_images'][0]
                    self.assertEqual(set(entry), {'index', 'image', 'sha256'})
                    self.assertEqual(entry['image']['__agentark_type__'], 'agentark.pil_image_png_base64.v1')
                    self.assertEqual(entry['index'], 1)
                    self.assertEqual(entry['sha256'], digest)
                    restored = decode_obs(wire['obs'])
                    decoded_image = restored['attempt_initial_images'][0]['image']
                    self.assertEqual(decoded_image.mode, image.mode)
                    self.assertEqual(decoded_image.size, image.size)
                    self.assertEqual(decoded_image.tobytes(), image.tobytes())
                    self.assertIs(obs['attempt_initial_images'][0]['image'], image)
                    if include_vis:
                        self.assertEqual(restored['vis'][0][0].getpixel((0, 0)), (17, 31, 47))
                    else:
                        self.assertNotIn('vis_format', encoded)
                        self.assertNotIn('vis', restored)

    def test_absent_field_preserves_existing_protocol_shape(self):
        plain = {'step_msg': 'hello', 'skip_infer': False, 'other': {'key': 'value'}}
        self.assertEqual(encode_obs(plain), plain)
        self.assertEqual(decode_obs(plain), plain)
        image = self._image()
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        expected = {
            'step_msg': 'hello', 'vis': [[base64.b64encode(buffer.getvalue()).decode('utf-8')]],
            'vis_format': 'png_base64',
        }
        encoded = encode_obs({'step_msg': 'hello', 'vis': [[image]]})
        self.assertEqual(encoded, expected)
        self.assertNotIn('attempt_initial_images', encoded)
        restored = decode_obs(json.loads(json.dumps(encoded)))
        self.assertEqual(set(restored), {'step_msg', 'vis', 'vis_format'})
        self.assertEqual(restored['vis'][0][0].tobytes(), image.tobytes())

    def test_explicit_empty_field_stays_empty(self):
        obs = {'step_msg': 'hello', 'attempt_initial_images': []}
        self.assertEqual(encode_obs(obs), obs)
        self.assertEqual(decode_obs(json.loads(json.dumps(encode_obs(obs)))), obs)


if __name__ == '__main__':
    unittest.main()
