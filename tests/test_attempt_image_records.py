import base64
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

from PIL import Image, PngImagePlugin


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_ark.ark_env.attempt_initial_images import load_attempt_initial_images  # noqa: E402
from agent_ark.ark_env.ark_env import ArkRolloutContext  # noqa: E402
from agent_ark.agent.api_agent import APIAgent  # noqa: E402
from agent_ark.ark_eval.run_api_agent import _run_case_rollout  # noqa: E402
from agent_ark.ark_eval.trajectory_io import (  # noqa: E402
    TrajectoryJsonlWriter,
    build_eval_trajectory_record,
    encode_history_snapshot,
    encode_trajectory_value,
    history_snapshot_from_record,
    load_trajectory_records,
)
from agent_ark.interaction.hooks import HookManager  # noqa: E402
from agent_ark.interaction.serialization import serialize_obs  # noqa: E402


class AttemptImageRecordsTest(unittest.TestCase):
    @staticmethod
    def _source_image(path, *, mode='RGBA', offset=0):
        image = Image.new(mode, (3, 2))
        if mode == 'RGBA':
            pixels = [
                (3 + offset, 11, 27, 0), (43, 59, 71, 127), (89, 101, 113, 255),
                (131, 149, 167, 19), (181, 197, 211, 201), (223, 239, 251, 255),
            ]
        else:
            pixels = [
                (3 + offset, 11, 27), (43, 59, 71), (89, 101, 113),
                (131, 149, 167), (181, 197, 211), (223, 239, 251),
            ]
        image.putdata(pixels)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text('source-note', 'The source-byte hash includes this metadata.')
        image.save(path, pnginfo=metadata)
        return image, path.read_bytes()

    @staticmethod
    def _record(snapshot, *, include_images=True):
        return build_eval_trajectory_record(
            result={'case_id': 'attempt-images', 'model_name': 'offline-model'},
            case={'task_name': 'ExampleTask', 'group_seed': 3, 'env_id': 0},
            model_runtime={},
            history_snapshot=snapshot,
            prefix_attempts=None,
            include_images=include_images,
        )

    def test_actual_jsonl_round_trip_preserves_order_pixels_and_source_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sources = [root / 'first.png', root / 'second.png']
            expected = [
                self._source_image(sources[0]),
                self._source_image(sources[1], mode='RGB', offset=4),
            ]
            attachments = load_attempt_initial_images(
                {'env_wrapper_cfg': {'attempt_initial_images': [path.name for path in sources]}},
                root,
            )
            first_obs = {'step_msg': 'attempt starts', 'attempt_initial_images': attachments}
            snapshot = {7: [[{
                'obs': first_obs,
                'action': '<tool_call>look()</tool_call>',
                'next_obs': {'step_msg': 'after action'},
                'reward': 0.0,
                'done': True,
            }]]}
            path = root / 'trajectory.jsonl'
            TrajectoryJsonlWriter(str(path)).write(self._record(snapshot))
            for source in sources:
                source.unlink()

            wire_record = json.loads(path.read_text(encoding='utf-8'))
            wire_images = wire_record['history_snapshot']['7'][0][0]['obs']['attempt_initial_images']
            self.assertEqual([entry['index'] for entry in wire_images], [1, 2])
            self.assertTrue(all(set(entry) == {'index', 'image', 'sha256'} for entry in wire_images))
            self.assertTrue(all(entry['image']['__agentark_type__'] == 'agentark.pil_image_png_base64.v1'
                                for entry in wire_images))

            loaded = load_trajectory_records(str(path))[0]
            restored = history_snapshot_from_record(loaded)[7][0][0]
            self.assertNotIn('attempt_initial_images', restored['next_obs'])
            for entry, (image, original_bytes) in zip(restored['obs']['attempt_initial_images'], expected):
                self.assertIsInstance(entry['image'], Image.Image)
                self.assertEqual(entry['image'].mode, image.mode)
                self.assertEqual(entry['image'].size, image.size)
                self.assertEqual(entry['image'].tobytes(), image.tobytes())
                self.assertEqual(entry['sha256'], hashlib.sha256(original_bytes).hexdigest())
                # The hash is provenance of the source file, not of its later PNG transport.
                transport = io.BytesIO()
                entry['image'].save(transport, format='PNG')
                self.assertNotEqual(entry['sha256'], hashlib.sha256(transport.getvalue()).hexdigest())

    def test_include_images_false_omits_pixels_but_keeps_index_and_source_hash(self):
        image = Image.new('RGBA', (3, 2), color=(19, 37, 73, 127))
        digest = hashlib.sha256(b'original source bytes').hexdigest()
        snapshot = {7: [[{
            'obs': {'attempt_initial_images': [{'index': 1, 'image': image, 'sha256': digest}]},
            'next_obs': {},
            'action': 'act',
            'done': True,
        }]]}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'without-images.jsonl'
            TrajectoryJsonlWriter(str(path)).write(self._record(snapshot, include_images=False))
            wire = json.loads(path.read_text(encoding='utf-8'))
            entry = wire['history_snapshot']['7'][0][0]['obs']['attempt_initial_images'][0]
            self.assertEqual(entry, {
                'index': 1,
                'sha256': digest,
                'image': {
                    '__agentark_type__': 'agentark.image_omitted.v1',
                    'source_type': 'pil',
                    'size': [3, 2],
                },
            })
            self.assertNotIn('data', entry['image'])
            restored = history_snapshot_from_record(load_trajectory_records(str(path))[0])
            self.assertEqual(restored[7][0][0]['obs']['attempt_initial_images'][0], {
                'index': 1, 'image': None, 'sha256': digest,
            })

    def test_existing_record_without_attachments_keeps_its_shape(self):
        snapshot = {7: [[{
            'obs': {'step_msg': 'start'},
            'next_obs': {'step_msg': 'end'},
            'action': 'act',
            'reward': 1.0,
            'done': True,
        }]]}
        encoded = encode_history_snapshot(snapshot)
        self.assertEqual(encoded, {'7': snapshot[7]})
        self.assertNotIn('attempt_initial_images', encoded['7'][0][0]['obs'])

    def test_legacy_string_and_bytes_encoding_is_unchanged(self):
        # This feature does not reinterpret arbitrary strings as typed images.
        data_url = 'data:image/png;base64,legacy-value'
        self.assertEqual(encode_trajectory_value(data_url, include_images=False), data_url)
        self.assertEqual(encode_trajectory_value(b'legacy', include_images=False), {
            '__agentark_type__': 'agentark.bytes_base64.v1', 'data': 'bGVnYWN5',
        })

    def test_hook_observation_without_additional_images_is_unchanged(self):
        self.assertEqual(serialize_obs({'step_msg': 'hello', 'skip_infer': False}), {
            'step_msg': 'hello', 'skip_infer': False, 'images': [], 'messages': [],
        })
        self.assertEqual(serialize_obs({'step_msg': 'hello'}, include_messages=False), {
            'step_msg': 'hello', 'skip_infer': False, 'images': [],
        })

    def test_raw_hook_observation_carries_separate_images_without_camera_budget(self):
        image = Image.new('RGBA', (3, 2), color=(19, 37, 73, 127))
        digest = hashlib.sha256(b'source').hexdigest()
        serialized = serialize_obs({
            'vis': [[image]],
            'attempt_initial_images': [{'index': 1, 'image': image, 'sha256': digest}],
        }, max_images=0, include_messages=False)
        self.assertNotIn('messages', serialized)
        self.assertEqual(serialized['images'], [])
        entry = serialized['attempt_initial_images'][0]
        self.assertEqual(set(entry), {'index', 'sha256', 'url'})
        self.assertEqual(entry['index'], 1)
        self.assertEqual(entry['sha256'], digest)
        with Image.open(io.BytesIO(base64.b64decode(entry['url'].split(',', 1)[1]))) as restored:
            self.assertEqual(restored.mode, image.mode)
            self.assertEqual(restored.size, image.size)
            self.assertEqual(restored.tobytes(), image.tobytes())

    def test_first_request_hook_captures_pixels_before_history_is_trimmed(self):
        image = Image.new('RGB', (3, 2), color=(19, 37, 73))
        digest = hashlib.sha256(b'source').hexdigest()

        class FakeProvider:
            provider = 'openai'
            timeout_s = None

            def __init__(self):
                self.requests = []

            def chat_completions_create(self, **kwargs):
                self.requests.append(deepcopy(kwargs['messages']))
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                    content='<tool_call>act()</tool_call>', model_extra={},
                ))], usage=None)

        class FakeEnv:
            max_attempts = 1
            max_steps_per_attempt = 2

            def __init__(self):
                self.context = ArkRolloutContext()
                self._selected_env_cfg = {'env_wrapper_cfg': {'context_manager': {
                    'history': {'max_history_attempts': 1, 'max_history_steps_per_attempt': 1},
                    'messages': {'enabled': True, 'include_images': True, 'image_caption_language': 'en'},
                }}}
                self.sub_env = SimpleNamespace(env_info_mgr=SimpleNamespace(env_config=self._selected_env_cfg))

            def reset(self, **kwargs):
                self.turn = 0
                obs = {7: {
                    'step_msg': '<system_prompt>Play.</system_prompt><task_prompt>Act twice.</task_prompt>',
                    'attempt_initial_images': [{'index': 1, 'image': image.copy(), 'sha256': digest}],
                }}
                return self.context.on_reset(self._selected_env_cfg, obs), {}

            def step(self, actions, info=None):
                self.turn += 1
                terminal = self.turn == 2
                next_obs = {7: {'step_msg': 'after action'}}
                reward = {7: 0.0}
                done = {7: terminal, '__all__': terminal}
                self.context.record_transition(next_obs, actions, reward, done)
                return (
                    self.context.finalize_obs(next_obs, rollout_done=done), reward, done,
                    {'rollout': {'success': terminal, 'truncated': False}},
                )

        provider = FakeProvider()
        agent = object.__new__(APIAgent)
        agent.name = 'offline-agent'
        agent.client = provider
        agent.temperature = 0.0
        agent.thinking_type = None
        agent.reasoning_effort = None
        events = []
        env = FakeEnv()
        rollout = _run_case_rollout(
            env=env,
            model_runtime={'name': 'offline-model', 'agent': agent},
            case={'case_id': 'attempt-images', 'task_name': 'ExampleTask', 'group_seed': 3},
            hook_manager=HookManager([SimpleNamespace(handle_event=events.append)]),
        )
        self.assertEqual(len(provider.requests), 2)
        first_parts = [part for msg in provider.requests[0] if isinstance(msg['content'], list)
                       for part in msg['content']]
        self.assertTrue(any(part.get('type') == 'text' and 'Additional image 1' in part.get('text', '')
                            for part in first_parts))
        self.assertTrue(any(part.get('type') == 'image_url' for part in first_parts))
        first_request_event = next(event for event in events if event['event'] == 'agent_request')
        captured = first_request_event['payload']['obs']['7']['attempt_initial_images'][0]
        self.assertEqual(captured['sha256'], digest)
        with Image.open(io.BytesIO(base64.b64decode(captured['url'].split(',', 1)[1]))) as restored:
            self.assertEqual(restored.tobytes(), image.tobytes())
        finalized = env.context.take_finalized_attempts()
        self.assertEqual(len(finalized[7][0]), 1)
        self.assertNotIn('attempt_initial_images', finalized[7][0][0]['obs'])
        raw_trace = rollout['step_records'][0]['raw_request_by_agent']['7']['request_messages']
        self.assertNotIn('data:image/', json.dumps(raw_trace))


if __name__ == '__main__':
    unittest.main()
