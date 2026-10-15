import hashlib
import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env import attempt_initial_images as initial_images  # noqa: E402
from agent_ark.ark_env.ark_env import ArkEnv  # noqa: E402
from agent_ark.ark_env.direct_env import EnvInfoManager  # noqa: E402
from agent_ark.ark_env.context_manager import MessageContext  # noqa: E402


def image_cfg(paths):
    return {'env_wrapper_cfg': {'attempt_initial_images': paths}}


class AttemptInitialImageLoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mods = Path(self.tmp.name) / 'Mods'
        self.mods.mkdir()

    def make_image(self, name='reference.png', *, size=(13, 7), color='red', fmt=None):
        path = self.mods / name
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {'lossless': True} if path.suffix.lower() == '.webp' else {}
        Image.new('RGB', size, color).save(path, format=fmt, **save_kwargs)
        return path

    def test_missing_and_empty_are_no_io_defaults(self):
        for cfg in ({}, {'env_wrapper_cfg': {}}, image_cfg([])):
            with self.subTest(cfg=cfg), patch.object(Path, 'open', side_effect=AssertionError('unexpected IO')), patch(
                'agent_ark.ark_env.attempt_initial_images.Path', side_effect=AssertionError('unexpected path lookup')
            ):
                self.assertEqual(initial_images.load_attempt_initial_images(cfg, None), [])

    def test_public_path_resolver_is_pure_and_does_no_path_or_image_operations(self):
        cfg = image_cfg(['references/${task_params.filename}', '${LITERAL_NAME}', 'references/${task_params.filename}'])
        cfg['task_params'] = {'filename': 'not-required-to-exist.png'}
        original = deepcopy(cfg)
        with patch.object(initial_images, 'Path', side_effect=AssertionError('unexpected path lookup')), patch.object(
            Image, 'open', side_effect=AssertionError('unexpected image IO')
        ):
            for disabled in (None, {}, {'env_wrapper_cfg': {}}, image_cfg([])):
                self.assertEqual(initial_images.resolve_attempt_initial_image_paths(disabled), [])
            paths = initial_images.resolve_attempt_initial_image_paths(cfg)
        self.assertEqual(paths, ['references/not-required-to-exist.png', '${LITERAL_NAME}', 'references/not-required-to-exist.png'])
        self.assertEqual(cfg, original)
        paths.append('caller-owned.png')
        self.assertEqual(cfg, original)

    def test_dotted_config_references_use_current_string_values_without_mutation(self):
        red = self.make_image('references/red.png')
        blue = self.make_image('references/blue.png', color='blue')
        cfg = image_cfg(['${task_params.assets.directory}/${task_params.reference_filename}'])
        cfg['task_params'] = {'assets': {'directory': 'references'}, 'reference_filename': 'red.png'}
        original = deepcopy(cfg)
        first = initial_images.load_attempt_initial_images(cfg, self.mods)[0]
        self.assertEqual(first['sha256'], hashlib.sha256(red.read_bytes()).hexdigest())
        self.assertEqual(cfg, original)
        cfg['task_params']['reference_filename'] = 'blue.png'
        second = initial_images.load_attempt_initial_images(cfg, self.mods)[0]
        self.assertEqual(second['sha256'], hashlib.sha256(blue.read_bytes()).hexdigest())
        self.assertEqual(second['image'].getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(set(second), {'index', 'image', 'sha256'})
        self.assertNotIn('blue.png', str({key: value for key, value in second.items() if key != 'image'}))

    def test_environment_style_and_tilde_paths_keep_literal_bytes(self):
        for name in ('${ATTEMPT_IMAGE_NAME}', '$ATTEMPT_IMAGE_NAME', '%ATTEMPT_IMAGE_NAME%', '~/reference.png'):
            with self.subTest(name=name):
                path = self.make_image(name, fmt='PNG')
                with patch.dict(os.environ, {'ATTEMPT_IMAGE_NAME': 'not-the-literal-file.png'}):
                    record = initial_images.load_attempt_initial_images(image_cfg([name]), self.mods)[0]
                    cfg = image_cfg(['${task_params.filename}'])
                    cfg['task_params'] = {'filename': name}
                    referenced = initial_images.load_attempt_initial_images(cfg, self.mods)[0]
                self.assertEqual(record['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(referenced['sha256'], record['sha256'])

    def test_invalid_config_references_fail_before_path_lookup_without_values(self):
        invalid = (
            ('${task_params.reference_filename}', {}, 'unknown configuration reference'),
            ('${task_params.reference_filename}', {'task_params': []}, 'unknown configuration reference'),
            ('${task_params.reference_filename}', {'task_params': SimpleNamespace(reference_filename='SECRET')}, 'unknown configuration reference'),
            ('${task_params.reference_filename}', {'task_params': {'reference_filename': 1}}, 'string values'),
            ('${task_params.reference_filename}', {'task_params': {'reference_filename': None}}, 'string values'),
            ('${task_params.reference_filename}', {'task_params': {'reference_filename': {'secret': 'SECRET'}}}, 'string values'),
            ('${task_params.reference_filename}', {'task_params': {'reference_filename': ''}}, 'nonempty path'),
            ('${task_params..reference_filename}', {}, 'invalid configuration reference'),
            ('${task_params.reference_filename[0]}', {}, 'invalid configuration reference'),
            ('${task_params.reference_filename}', {'task_params': {'reference_filename': '${task_params.reference_filename}'}}, 'recursive or cyclic'),
            ('${task_params.reference_filename}', {'task_params': {'reference_filename': '${other.filename}'}, 'other': {'filename': '${task_params.reference_filename}'}}, 'recursive or cyclic'),
            ('${task_params.reference_filename}', {'task_params': {'reference_filename': '${other.filename}'}, 'other': {'filename': 'SECRET.png'}}, 'recursive or cyclic'),
        )
        for template, extra_cfg, message in invalid:
            cfg = dict(image_cfg([template]), **extra_cfg)
            with self.subTest(template=template, message=message), patch.object(
                initial_images, 'Path', side_effect=AssertionError('unexpected path lookup')
            ):
                with self.assertRaisesRegex(ValueError, message) as raised:
                    initial_images.load_attempt_initial_images(cfg, self.mods)
            self.assertIn('attempt_initial_images[0]', str(raised.exception))
            self.assertNotIn('SECRET', str(raised.exception))

    def test_invalid_reference_in_later_entry_prevents_all_file_reads(self):
        cfg = image_cfg(['valid.png', '${task_params.missing}'])
        with patch.object(Path, 'open', side_effect=AssertionError('unexpected image IO')):
            with self.assertRaisesRegex(ValueError, 'unknown configuration reference'):
                initial_images.load_attempt_initial_images(cfg, self.mods)

    def test_order_hash_copies_and_native_dimensions(self):
        red = self.make_image('red.png')
        blue = self.make_image('blue.webp', size=(5, 19), color='blue')
        records = initial_images.load_attempt_initial_images(image_cfg(['blue.webp', 'red.png']), self.mods)
        self.assertEqual([item['index'] for item in records], [1, 2])
        self.assertEqual([item['image'].size for item in records], [(5, 19), (13, 7)])
        self.assertEqual([item['sha256'] for item in records], [
            hashlib.sha256(blue.read_bytes()).hexdigest(), hashlib.sha256(red.read_bytes()).hexdigest()
        ])
        self.assertEqual(set(records[0]), {'index', 'image', 'sha256'})
        self.assertIsInstance(records[0]['image'], Image.Image)
        self.assertFalse(getattr(records[0]['image'], 'filename', None))
        blue.unlink()
        self.assertEqual(records[0]['image'].getpixel((0, 0)), (0, 0, 255))
        again = initial_images.load_attempt_initial_images(image_cfg(['red.png']), self.mods)
        records[1]['image'].putpixel((0, 0), (0, 255, 0))
        self.assertEqual(again[0]['image'].getpixel((0, 0)), (255, 0, 0))

    def test_relative_paths_use_effective_mod_root_not_cwd_and_allow_absolute(self):
        expected = self.make_image('nested/reference.png', color='blue')
        cwd_file = Path(self.tmp.name) / 'reference.png'
        Image.new('RGB', (2, 2), 'red').save(cwd_file)
        previous_cwd = os.getcwd()
        try:
            os.chdir(self.tmp.name)
            records = initial_images.load_attempt_initial_images(
                image_cfg(['nested/reference.png', str(expected)]), self.mods
            )
        finally:
            os.chdir(previous_cwd)
        self.assertEqual([item['image'].size for item in records], [(13, 7), (13, 7)])
        self.assertEqual(records[0]['sha256'], records[1]['sha256'])
        absolute_only = initial_images.load_attempt_initial_images(image_cfg([str(expected)]), None)
        self.assertEqual(absolute_only[0]['sha256'], records[0]['sha256'])

    def test_relative_path_does_not_fall_back_to_cwd_without_absolute_mod_root(self):
        for root in (None, '', 'relative-mods'):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, 'absolute effective Mods root'):
                initial_images.load_attempt_initial_images(image_cfg(['reference.png']), root)

    def test_exif_orientation_is_applied_without_resizing(self):
        path = self.mods / 'oriented.jpg'
        image = Image.new('RGB', (15, 8), 'red')
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)
        record = initial_images.load_attempt_initial_images(image_cfg(['oriented.jpg']), self.mods)[0]
        self.assertEqual(record['image'].size, (8, 15))
        self.assertEqual(record['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertNotIn(274, record['image'].getexif())

    def decode_message_image(self, records):
        context = MessageContext({'include_images': True})
        parts = context.attempt_initial_images_to_parts({'attempt_initial_images': records})
        image_parts = [part for part in parts if part.get('type') == 'image_url']
        self.assertEqual(len(image_parts), 1)
        url = image_parts[0]['image_url']['url']
        self.assertTrue(url.startswith('data:image/png;base64,'))
        with Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))) as image:
            return image.copy()

    def test_cmyk_jpeg_is_converted_to_rgb_and_reaches_message_image_block(self):
        path = self.mods / 'cmyk.jpg'
        Image.new('CMYK', (9, 6), (0, 200, 40, 30)).save(path)
        records = initial_images.load_attempt_initial_images(image_cfg([path.name]), self.mods)
        self.assertEqual(records[0]['image'].mode, 'RGB')
        transported = self.decode_message_image(records)
        self.assertEqual(transported.mode, 'RGB')
        self.assertEqual(transported.size, (9, 6))
        self.assertEqual(transported.tobytes(), records[0]['image'].tobytes())

    def test_palette_alpha_and_sixteen_bit_png_survive_message_transport(self):
        palette_image = Image.new('P', (2, 1))
        palette_image.putpalette([255, 0, 0, 0, 0, 255] + [0] * (768 - 6))
        palette_image.putdata([0, 1])
        palette_image.info['transparency'] = bytes([0, 255])
        rgba_image = Image.new('RGBA', (2, 3), (30, 60, 90, 123))
        la_image = Image.new('LA', (3, 2), (140, 87))
        sixteen_bit = Image.frombytes('I;16', (3, 1), b''.join(
            value.to_bytes(2, 'little') for value in (0, 20000, 65535)
        ))
        for name, image in (
            ('palette.png', palette_image), ('alpha.png', rgba_image),
            ('gray-alpha.png', la_image), ('sixteen-bit.png', sixteen_bit),
        ):
            with self.subTest(name=name):
                image.save(self.mods / name)
                records = initial_images.load_attempt_initial_images(image_cfg([name]), self.mods)
                transported = self.decode_message_image(records)
                self.assertEqual(transported.size, image.size)
                if name == 'sixteen-bit.png':
                    self.assertEqual([transported.getpixel((x, 0)) for x in range(3)], [0, 20000, 65535])
                else:
                    self.assertEqual(transported.convert('RGBA').tobytes(), image.convert('RGBA').tobytes())

    def test_invalid_config_is_rejected_before_io(self):
        invalid_values = (None, 'reference.png', {'path': 'reference.png'}, [1], [''], ['   '], ['ok.png', False])
        for value in invalid_values:
            with self.subTest(value=value), patch.object(Path, 'open', side_effect=AssertionError('unexpected IO')):
                with self.assertRaisesRegex(ValueError, 'attempt_initial_images'):
                    initial_images.load_attempt_initial_images(image_cfg(value), self.mods)
        with self.assertRaisesRegex(ValueError, 'at most'):
            initial_images.load_attempt_initial_images(image_cfg(['a'] * 17), self.mods)

    def test_missing_corrupt_unsupported_and_animated_files_fail_explicitly(self):
        (self.mods / 'corrupt.png').write_bytes(b'not an image')
        self.make_image('unsupported.gif', fmt='GIF')
        animated = self.mods / 'animated.png'
        Image.new('RGB', (4, 4), 'red').save(
            animated, save_all=True, append_images=[Image.new('RGB', (4, 4), 'blue')], duration=100
        )
        for name, message in (
            ('missing.png', 'could not read'),
            ('corrupt.png', 'not a decodable'),
            ('unsupported.gif', 'static PNG'),
            ('animated.png', 'animated images'),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                initial_images.load_attempt_initial_images(image_cfg([name]), self.mods)

    def test_truncated_pixels_fail_instead_of_returning_partial_image(self):
        path = self.make_image('truncated.png', size=(120, 100))
        data = path.read_bytes()
        path.write_bytes(data[:len(data) // 2])
        with self.assertRaisesRegex(ValueError, 'attempt_initial_images'):
            initial_images.load_attempt_initial_images(image_cfg(['truncated.png']), self.mods)

    def test_file_and_total_byte_limits_are_enforced(self):
        path = self.make_image()
        length = path.stat().st_size
        with patch.object(initial_images, 'MAX_IMAGE_FILE_BYTES', length - 1):
            with self.assertRaisesRegex(ValueError, 'byte file limit'):
                initial_images.load_attempt_initial_images(image_cfg([path.name]), self.mods)
        with patch.object(initial_images, 'MAX_TOTAL_IMAGE_FILE_BYTES', length * 2 - 1):
            with self.assertRaisesRegex(ValueError, 'byte total limit'):
                initial_images.load_attempt_initial_images(image_cfg([path.name, path.name]), self.mods)

    def test_per_image_and_total_pixel_limits_are_enforced(self):
        path = self.make_image(size=(10, 8))
        with patch.object(initial_images, 'MAX_IMAGE_PIXELS', 79):
            with self.assertRaisesRegex(ValueError, 'pixel image limit'):
                initial_images.load_attempt_initial_images(image_cfg([path.name]), self.mods)
        with patch.object(initial_images, 'MAX_TOTAL_IMAGE_PIXELS', 159):
            with self.assertRaisesRegex(ValueError, 'pixel total limit'):
                initial_images.load_attempt_initial_images(image_cfg([path.name, path.name]), self.mods)


class FakeSubEnv:
    """Exercise real ArkEnv reset/step/context paths without a Unity process."""

    def __init__(self, mods, tasks, *, reset_obs=None):
        self.env_info_mgr = SimpleNamespace(mod_path=str(mods), env_config={})
        self.tasks = tasks
        self.reset_calls = []
        self.step_results = []
        self.ml_unity_id_map = {101: 3, 202: 9}
        self.reset_obs = reset_obs or {
            101: {'step_msg': 'ready', 'vis': ['unity-camera-3'], 'skip_infer': False},
            202: {'step_msg': 'ready', 'vis': ['unity-camera-9'], 'skip_infer': False},
        }

    def reset(self, task_name=None, group_seed=None, env_id=None):
        self.reset_calls.append({'task_name': task_name, 'group_seed': group_seed, 'env_id': env_id})
        self.env_info_mgr.env_config = deepcopy(self.tasks[task_name])
        self.env_info_mgr.env_config.update({'task_name': task_name, 'group_seed': group_seed, 'env_id': env_id})
        return deepcopy(self.reset_obs), {'initial_observation': {'warmup_done': True}}

    def step(self, actions, info=None):
        del actions, info
        return deepcopy(self.step_results.pop(0))

    def close(self):
        pass


class ArkEnvAttemptInitialImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mods = Path(self.tmp.name) / 'runtime-sandbox' / 'Mods'
        self.mods.mkdir(parents=True)
        self.path = self.mods / 'reference.png'
        Image.new('RGB', (7, 11), 'red').save(self.path)
        base = {'max_attempts': 3, 'max_steps_per_attempt': 2}
        self.tasks = {
            'enabled': dict(base, **image_cfg(['reference.png'])),
            'omitted': dict(base),
            'empty': dict(base, **image_cfg([])),
        }

    def make_env(self, *, reset_obs=None):
        sub_env = FakeSubEnv(self.mods, self.tasks, reset_obs=reset_obs)
        with patch('agent_ark.ark_env.ark_env.ArkSubEnv', return_value=sub_env):
            env = ArkEnv({'mod_path': 'unused-original-root'})
        self.addCleanup(env.close)
        return env

    @staticmethod
    def append_step(env, *, terminal=False, skip_uid_9=False):
        env.sub_env.step_results.append((
            {
                101: {'step_msg': 'after-action', 'vis': ['unity-next-3'], 'skip_infer': False},
                202: {'step_msg': 'after-action', 'vis': ['unity-next-9'], 'skip_infer': skip_uid_9},
            },
            {101: 1.0 if terminal else 0.0, 202: 1.0 if terminal else 0.0},
            {101: terminal, 202: terminal, '__all__': terminal},
            {},
        ))

    def test_reset_attaches_before_context_snapshot_and_never_changes_vis(self):
        env = self.make_env()
        with patch('agent_ark.ark_env.ark_env.load_attempt_initial_images', wraps=initial_images.load_attempt_initial_images) as load:
            obs, info = env.reset(task_name='enabled', group_seed=123, env_id=0)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(info['sub_env']['reset']['initial_observation'], {'warmup_done': True})
            for uid in (3, 9):
                self.assertEqual(obs[uid]['vis'], [f'unity-camera-{uid}'])
                self.assertEqual(obs[uid]['attempt_initial_images'][0]['image'].size, (7, 11))
                self.assertEqual(set(obs[uid]['attempt_initial_images'][0]), {'index', 'image', 'sha256'})
                self.assertIn('attempt_initial_images', env.rollout_ctx._last_obs_by_uid[uid])
            obs[3]['attempt_initial_images'][0]['image'].putpixel((0, 0), (0, 255, 0))
            self.assertEqual(obs[9]['attempt_initial_images'][0]['image'].getpixel((0, 0)), (255, 0, 0))
            self.append_step(env)
            next_obs, _, _, _ = env.step({3: 'a', 9: 'a'})
            self.assertTrue(all('attempt_initial_images' not in item for item in next_obs.values()))
            self.assertEqual(load.call_count, 1)

    def test_disabled_reset_preserves_shape_and_does_no_image_io(self):
        env = self.make_env()
        with patch.object(Path, 'open', side_effect=AssertionError('unexpected image IO')):
            obs, _ = env.reset(task_name='omitted', group_seed=123, env_id=0)
        self.assertEqual(obs, {3: env.sub_env.reset_obs[101], 9: env.sub_env.reset_obs[202]})

    def test_auto_reset_same_and_new_seed_reloads_for_each_returned_attempt(self):
        for reroll in (False, True):
            with self.subTest(reroll=reroll):
                self.tasks['enabled']['reroll_group_seed_on_same_task'] = reroll
                env = self.make_env()
                with patch('agent_ark.ark_env.ark_env.load_attempt_initial_images', wraps=initial_images.load_attempt_initial_images) as load:
                    first, _ = env.reset(task_name='enabled', group_seed=123, env_id=0)
                    first_hash = first[3]['attempt_initial_images'][0]['sha256']
                    Image.new('RGB', (7, 11), 'blue').save(self.path)
                    self.append_step(env, terminal=True)
                    second, _, done, info = env.step({3: 'a', 9: 'a'})
                    self.assertEqual(load.call_count, 2)
                self.assertFalse(done['__all__'])
                self.assertTrue(info['attempt']['auto_reset'])
                self.assertEqual(second[3]['attempt_initial_images'][0]['image'].getpixel((0, 0)), (0, 0, 255))
                self.assertNotEqual(second[3]['attempt_initial_images'][0]['sha256'], first_hash)
                second_seed = env.sub_env.reset_calls[-1]['group_seed']
                if reroll:
                    self.assertNotEqual(second_seed, 123)
                else:
                    self.assertEqual(second_seed, 123)
                Image.new('RGB', (7, 11), 'red').save(self.path)

    def test_start_attempt_index_ignores_internal_discarded_reset(self):
        env = self.make_env()
        with patch('agent_ark.ark_env.ark_env.load_attempt_initial_images', wraps=initial_images.load_attempt_initial_images) as load:
            obs, info = env.reset(task_name='enabled', group_seed=123, env_id=0, start_attempt_index=3)
        self.assertEqual(len(env.sub_env.reset_calls), 2)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(info['attempt']['index'], 3)
        self.assertEqual(len(obs[3]['attempt_initial_images']), 1)

    def test_config_reference_follows_each_effective_attempt_and_clears_on_task_switch(self):
        blue = self.mods / 'blue.png'
        Image.new('RGB', (7, 11), 'blue').save(blue)
        self.tasks['enabled']['task_params'] = {'reference_filename': 'reference.png'}
        self.tasks['enabled']['env_wrapper_cfg']['attempt_initial_images'] = ['${task_params.reference_filename}']
        env = self.make_env()
        with patch('agent_ark.ark_env.ark_env.load_attempt_initial_images', wraps=initial_images.load_attempt_initial_images) as load:
            first, _ = env.reset(task_name='enabled', group_seed=123, env_id=0)
            self.assertEqual(first[3]['attempt_initial_images'][0]['image'].getpixel((0, 0)), (255, 0, 0))
            self.append_step(env)
            later, _, _, _ = env.step({3: 'a', 9: 'a'})
            self.assertTrue(all('attempt_initial_images' not in item for item in later.values()))
            self.assertEqual(load.call_count, 1)
            # FakeSubEnv materializes these newest task values on auto-reset,
            # just as evaluation overrides enter the real effective config.
            self.tasks['enabled']['task_params']['reference_filename'] = 'blue.png'
            self.append_step(env, terminal=True)
            second, _, _, _ = env.step({3: 'a', 9: 'a'})
            self.assertEqual(load.call_count, 2)
            self.assertEqual(second[3]['attempt_initial_images'][0]['image'].getpixel((0, 0)), (0, 0, 255))
            self.assertNotEqual(first[3]['attempt_initial_images'][0]['sha256'], second[3]['attempt_initial_images'][0]['sha256'])
        with patch.object(initial_images, 'Path', side_effect=AssertionError('unexpected path lookup')):
            omitted, _ = env.reset(task_name='omitted', group_seed=123, env_id=0)
            self.append_step(env)
            after_omitted, _, _, _ = env.step({3: 'a', 9: 'a'})
        self.assertFalse(env._attempt_initial_image_records)
        self.assertTrue(all('attempt_initial_images' not in item for item in omitted.values()))
        self.assertTrue(all('attempt_initial_images' not in item for item in after_omitted.values()))

    def test_switch_from_enabled_to_omitted_or_empty_clears_pending_images(self):
        reset_obs = {
            101: {'step_msg': 'ready', 'vis': [], 'skip_infer': False},
            202: {'step_msg': '', 'vis': [], 'skip_infer': True},
        }
        env = self.make_env(reset_obs=reset_obs)
        for disabled_task in ('omitted', 'empty'):
            env.reset(task_name='enabled', group_seed=123, env_id=0)
            self.assertEqual(env._attempt_initial_image_delivered_uids, {3})
            with patch.object(Path, 'open', side_effect=AssertionError('unexpected image IO')):
                obs, _ = env.reset(task_name=disabled_task, group_seed=123, env_id=0)
                self.append_step(env)
                next_obs, _, _, _ = env.step({3: 'a'})
            self.assertFalse(env._attempt_initial_image_delivered_uids)
            self.assertTrue(all('attempt_initial_images' not in item for item in obs.values()))
            self.assertTrue(all('attempt_initial_images' not in item for item in next_obs.values()))

    def test_real_effective_config_merge_clears_enabled_images_on_same_env_task_switch(self):
        baseline = {
            'override_by_task': True,
            'load_mod_mode': 'task_name',
            'action_mode': 'func',
            'engine_para': {},
            'max_attempts': 3,
            'max_steps_per_attempt': 2,
            'env_wrapper_cfg': {'video_frame_selection': 'decision_only'},
        }
        baseline_text = yaml.safe_dump(baseline)
        for name in ('config.yaml', 'config.yaml.bak'):
            (self.mods / name).write_text(baseline_text, encoding='utf-8')
        for task_name in ('enabled', 'omitted'):
            task_root = self.mods / 'all_tasks' / task_name
            (task_root / 'cfg').mkdir(parents=True)
            (task_root / f'{task_name}.json').write_text('{}', encoding='utf-8')
            (task_root / 'cfg' / 'task_config.yaml').write_text(
                yaml.safe_dump(self.tasks[task_name]), encoding='utf-8'
            )

        env = self.make_env()
        # Keep the production manager and config materialization; replace only
        # the low-level Unity transport already supplied by this test fake.
        env.sub_env.env_info_mgr = EnvInfoManager({'mod_path': str(self.mods)})

        def real_config_reset(task_name=None, group_seed=None, env_id=None):
            env.sub_env.env_info_mgr.reset(task_name=task_name, group_seed=group_seed, env_id=env_id)
            return deepcopy(env.sub_env.reset_obs), {}

        env.sub_env.reset = real_config_reset
        enabled, _ = env.reset(task_name='enabled', group_seed=123, env_id=0)
        self.assertIn('attempt_initial_images', enabled[3])
        self.assertEqual(
            env.sub_env.env_info_mgr.env_config['env_wrapper_cfg']['attempt_initial_images'], ['reference.png']
        )
        # Removing the old source makes stale inherited paths fail visibly.
        self.path.unlink()
        disabled, _ = env.reset(task_name='omitted', group_seed=123, env_id=0)
        self.append_step(env)
        after_disabled, _, _, _ = env.step({3: 'a', 9: 'a'})
        self.assertTrue(all('attempt_initial_images' not in item for item in disabled.values()))
        self.assertTrue(all('attempt_initial_images' not in item for item in after_disabled.values()))
        self.assertNotIn('attempt_initial_images', env.sub_env.env_info_mgr.env_config['env_wrapper_cfg'])
        self.assertEqual((self.mods / 'config.yaml.bak').read_text(encoding='utf-8'), baseline_text)
        effective_yaml = yaml.safe_load((self.mods / 'config.yaml').read_text(encoding='utf-8'))
        effective_json = json.loads((self.mods / 'config.json').read_text(encoding='utf-8'))
        self.assertEqual(effective_yaml, effective_json)

    def test_real_effective_config_overrides_select_reference_on_each_attempt(self):
        blue = self.mods / 'blue.png'
        Image.new('RGB', (7, 11), 'blue').save(blue)
        baseline = {'override_by_task': True, 'load_mod_mode': 'task_name', 'action_mode': 'func',
                    'engine_para': {}, 'max_attempts': 3, 'max_steps_per_attempt': 2}
        baseline_text = yaml.safe_dump(baseline)
        for name in ('config.yaml', 'config.yaml.bak'):
            (self.mods / name).write_text(baseline_text, encoding='utf-8')
        self.tasks['enabled']['task_params'] = {'reference_filename': 'reference.png'}
        self.tasks['enabled']['env_wrapper_cfg']['attempt_initial_images'] = ['${task_params.reference_filename}']
        for task_name in ('enabled', 'omitted'):
            task_root = self.mods / 'all_tasks' / task_name
            (task_root / 'cfg').mkdir(parents=True)
            (task_root / f'{task_name}.json').write_text('{}', encoding='utf-8')
            (task_root / 'cfg' / 'task_config.yaml').write_text(yaml.safe_dump(self.tasks[task_name]), encoding='utf-8')
        env = self.make_env()
        env.sub_env.env_info_mgr = EnvInfoManager({
            'mod_path': str(self.mods), 'env_config_overrides': {'task_params': {'reference_filename': 'blue.png'}}})

        def real_config_reset(task_name=None, group_seed=None, env_id=None):
            env.sub_env.env_info_mgr.reset(task_name=task_name, group_seed=group_seed, env_id=env_id)
            return deepcopy(env.sub_env.reset_obs), {}

        env.sub_env.reset = real_config_reset
        first, _ = env.reset(task_name='enabled', group_seed=123, env_id=0)
        self.assertEqual(first[3]['attempt_initial_images'][0]['image'].getpixel((0, 0)), (0, 0, 255))
        env.sub_env.env_info_mgr.cfg['env_config_overrides']['task_params']['reference_filename'] = 'reference.png'
        self.append_step(env, terminal=True)
        second, _, _, _ = env.step({3: 'a', 9: 'a'})
        self.assertEqual(second[3]['attempt_initial_images'][0]['image'].getpixel((0, 0)), (255, 0, 0))
        omitted, _ = env.reset(task_name='omitted', group_seed=123, env_id=0)
        self.assertTrue(all('attempt_initial_images' not in item for item in omitted.values()))
        self.assertEqual((self.mods / 'config.yaml.bak').read_text(encoding='utf-8'), baseline_text)
        effective_yaml = yaml.safe_load((self.mods / 'config.yaml').read_text(encoding='utf-8'))
        effective_json = json.loads((self.mods / 'config.json').read_text(encoding='utf-8'))
        self.assertEqual(effective_yaml, effective_json)

    def test_skipped_uid_gets_only_its_first_actual_decision_copy_without_redecode(self):
        env = self.make_env(reset_obs={
            101: {'step_msg': 'ready', 'vis': [], 'skip_infer': False},
            202: {'step_msg': '', 'vis': [], 'skip_infer': True},
        })
        with patch('agent_ark.ark_env.ark_env.load_attempt_initial_images', wraps=initial_images.load_attempt_initial_images) as load:
            obs, _ = env.reset(task_name='enabled', group_seed=123, env_id=0)
            self.assertIn('attempt_initial_images', obs[3])
            self.assertNotIn('attempt_initial_images', obs[9])
            self.append_step(env)
            next_obs, _, _, _ = env.step({3: 'a'})
            self.assertNotIn('attempt_initial_images', next_obs[3])
            self.assertIn('attempt_initial_images', next_obs[9])
            self.assertEqual(load.call_count, 1)
            self.append_step(env)
            later, _, _, _ = env.step({3: 'a', 9: 'a'})
            self.assertTrue(all('attempt_initial_images' not in item for item in later.values()))

    def test_uid_absent_at_reset_gets_first_later_decision_without_redecode(self):
        env = self.make_env(reset_obs={
            101: {'step_msg': 'ready', 'vis': [], 'skip_infer': False},
        })
        env.sub_env.ml_unity_id_map = {101: 3}
        with patch('agent_ark.ark_env.ark_env.load_attempt_initial_images', wraps=initial_images.load_attempt_initial_images) as load:
            obs, _ = env.reset(task_name='enabled', group_seed=123, env_id=0)
            self.assertEqual(set(obs), {3})
            self.assertIn('attempt_initial_images', obs[3])
            env.sub_env.ml_unity_id_map[202] = 9
            self.append_step(env)
            first_uid_9, _, _, _ = env.step({3: 'a'})
            self.assertNotIn('attempt_initial_images', first_uid_9[3])
            self.assertIn('attempt_initial_images', first_uid_9[9])
            self.assertEqual(load.call_count, 1)
            self.append_step(env)
            later, _, _, _ = env.step({3: 'a', 9: 'a'})
            self.assertTrue(all('attempt_initial_images' not in item for item in later.values()))

    def test_invalid_enabled_image_fails_before_context_is_started(self):
        env = self.make_env()
        self.path.unlink()
        with patch.object(env.rollout_ctx, 'on_reset', wraps=env.rollout_ctx.on_reset) as context_reset:
            with self.assertRaisesRegex(ValueError, 'could not read image file'):
                env.reset(task_name='enabled', group_seed=123, env_id=0)
        context_reset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
