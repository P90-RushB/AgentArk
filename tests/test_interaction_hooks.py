import json
import socket
import threading
import time
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import requests

from agent_ark.agent.api_agent import APIAgent, LLMClient
from agent_ark.agent.codex_agent import CodexAgent
from agent_ark.ark_env.ark_env import ArkRolloutContext
from agent_ark.ark_env.context_manager import MessageContext
from agent_ark.ark_env.serving.session_manager import EnvRuntime
from agent_ark.interaction.hooks import HookManager
from agent_ark.interaction.local_viewer import HumanActionBroker, LocalViewerHook, _VIEWER_HTML
from agent_ark.interaction.serialization import serialize_images, serialize_messages, serialize_obs_map
from agent_ark.ark_eval.run_api_agent import (
    _run_case_rollout,
    build_eval_hook_manager,
    build_model_runtimes,
    collect_raw_request_messages,
    evaluate_case,
    load_eval_config,
    normalize_response_trace,
    validate_player_feedback_eval_contract,
)


class CollectorHook:
    def __init__(self):
        self.events = []

    def start(self):
        return

    def handle_event(self, event):
        self.events.append(event)

    def close(self):
        return


class InteractionHookTest(unittest.TestCase):
    @staticmethod
    def _unused_local_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('127.0.0.1', 0))
            return int(sock.getsockname()[1])

    def test_serializes_obs_images_and_message_images(self):
        img = Image.new('RGB', (2, 2), color=(255, 0, 0))
        obs = {
            7: {
                'step_msg': 'hello',
                'vis': [[img]],
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': 'look'},
                            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}},
                        ],
                    }
                ],
            }
        }

        payload = serialize_obs_map(obs, max_images_per_observation=1)
        item = payload['7']
        self.assertEqual(item['step_msg'], 'hello')
        self.assertEqual(len(item['images']), 1)
        self.assertTrue(item['images'][0]['url'].startswith('data:image/png;base64,'))
        self.assertEqual(item['messages'][0]['content'][1]['url'], 'data:image/png;base64,abc')

    def test_serializes_recent_images_in_chronological_order(self):
        frames = [Image.new('RGB', (2, 2), color=(idx, 0, 0)) for idx in range(4)]

        images = serialize_images([frames], max_images=3)

        self.assertEqual([item['frame_index'] for item in images], [1, 2, 3])

    def test_serializes_raw_messages(self):
        messages = serialize_messages([
            {'role': 'assistant', 'content': 'x' * 20},
        ], text_max_chars=5)
        self.assertEqual(messages[0]['content'], 'xxxxx\n...[truncated]')

    def test_api_agent_preserves_reasoning_in_raw_assistant(self):
        text = APIAgent._combine_reasoning_and_content(
            reasoning='I should choose a smaller force.',
            content='<tool_call>{"name":"PushForward","arguments":{"forceScale":1.2}}</tool_call>',
        )
        self.assertIn('<think>', text)
        self.assertIn('I should choose a smaller force.', text)
        self.assertIn('<tool_call>', text)

    def test_api_agent_provider_controls_extra_body(self):
        class FakeClient:
            def __init__(self, provider, host):
                self.provider = provider
                self.base_url_host = host
                self.calls = []

            def chat_completions_create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='<tool_call>{"name":"PushForward","arguments":{}}</tool_call>',
                                model_extra={},
                            )
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=11,
                        completion_tokens=7,
                        total_tokens=18,
                    ),
                )

        cases = [
            ('openai', '203.0.113.10', {}),
            ('auto', '203.0.113.10', {}),
            ('openrouter', 'example.test', {'reasoning': {'enabled': True}}),
            ('auto', 'openrouter.ai', {'reasoning': {'enabled': True}}),
            ('dashscope', 'example.test', {'enable_thinking': True, 'thinking_budget': 81920}),
            ('auto', 'dashscope.aliyuncs.com', {'enable_thinking': True, 'thinking_budget': 81920}),
        ]
        for provider, host, expected_extra_body in cases:
            with self.subTest(provider=provider, host=host):
                agent = object.__new__(APIAgent)
                agent.client = FakeClient(provider, host)
                agent.temperature = 0.2

                response = agent._call_api([{'role': 'user', 'content': 'hi'}])

                self.assertIn('<tool_call>', response)
                self.assertEqual(agent.client.calls[0]['extra_body'], expected_extra_body)

    def test_llm_client_omits_temperature_when_none(self):
        class FakeCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(choices=[], usage=None)

        completions = FakeCompletions()
        client = object.__new__(LLMClient)
        client.model = 'example-model'
        client.provider = 'openai'
        client.timeout_s = None
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        client.chat_completions_create(
            messages=[{'role': 'user', 'content': 'hi'}],
            temperature=None,
            extra_body={},
        )

        self.assertNotIn('temperature', completions.calls[0])

    def test_api_agent_forward_trace_includes_usage(self):
        class FakeClient:
            provider = 'openai'
            base_url_host = 'example.test'

            def chat_completions_create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='<tool_call>{"name":"PushForward","arguments":{}}</tool_call>',
                                model_extra={},
                            )
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=123,
                        completion_tokens=45,
                        total_tokens=168,
                    ),
                )

        agent = object.__new__(APIAgent)
        agent.client = FakeClient()
        agent.temperature = 0.2

        _, trace = agent.forward_with_trace({0: {'messages': [{'role': 'user', 'content': 'hi'}]}})

        self.assertEqual(trace[0]['usage']['prompt_tokens'], 123)
        self.assertEqual(trace[0]['usage']['completion_tokens'], 45)
        self.assertEqual(trace[0]['usage']['total_tokens'], 168)

    def test_api_agent_hard_timeout_raises(self):
        class SlowClient:
            provider = 'openai'
            base_url_host = 'example.test'
            model = 'slow-model'
            timeout_s = 0.01

            def chat_completions_create(self, **kwargs):
                time.sleep(0.2)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='<tool_call>{"name":"PushForward","arguments":{}}</tool_call>',
                                model_extra={},
                            )
                        )
                    ],
                    usage=None,
                )

        agent = object.__new__(APIAgent)
        agent.client = SlowClient()
        agent.temperature = 0.2
        agent.name = 'slow-agent'

        with self.assertRaises(TimeoutError):
            agent.forward_with_trace({0: {'messages': [{'role': 'user', 'content': 'hi'}]}})

    def test_response_trace_normalization_preserves_usage(self):
        trace = normalize_response_trace({
            0: {
                'assistant_raw': '<tool_call>{}</tool_call>',
                'action_extracted': '<tool_call>{}</tool_call>',
                'usage': {
                    'prompt_tokens': 123,
                    'completion_tokens': 45,
                    'total_tokens': 168,
                },
            },
        })

        self.assertEqual(trace['0']['usage']['prompt_tokens'], 123)
        self.assertEqual(trace['0']['usage']['completion_tokens'], 45)
        self.assertEqual(trace['0']['usage']['total_tokens'], 168)

    def test_build_model_runtimes_passes_openai_provider(self):
        runtimes = build_model_runtimes([
            {
                'name': 'local-openai-test',
                'provider': 'openai',
                'model': 'example-openai-model',
                'base_url': 'http://203.0.113.10:18081/v1',
                'api_key': 'not-needed',
                'temperature': 0.1,
            }
        ])

        self.assertEqual(len(runtimes), 1)
        self.assertEqual(runtimes[0]['provider'], 'openai')
        self.assertEqual(runtimes[0]['base_url'], 'http://203.0.113.10:18081/v1')
        self.assertEqual(runtimes[0]['api_key_env'], None)
        self.assertEqual(runtimes[0]['temperature'], 0.1)
        self.assertEqual(runtimes[0]['agent'].client.provider, 'openai')
        self.assertEqual(runtimes[0]['agent'].client.model, 'example-openai-model')

    def test_build_model_runtimes_allows_omitted_temperature(self):
        runtimes = build_model_runtimes([
            {
                'name': 'default-temperature-provider',
                'provider': 'openai',
                'model': 'example-model',
                'base_url': 'https://example.test/v1',
                'api_key': 'not-needed',
                'temperature': None,
            }
        ])

        self.assertEqual(runtimes[0]['temperature'], None)
        self.assertEqual(runtimes[0]['agent'].temperature, None)

    def test_codex_agent_uses_data_uri_image_input_without_writing_file(self):
        class FakeTextInput:
            def __init__(self, text):
                self.text = text

        class FakeImageInput:
            def __init__(self, url):
                self.url = url

        image_url = 'data:image/png;base64,ZmFrZS1wbmctYnl0ZXM='
        agent = CodexAgent(name='codex-test')
        agent._text_input_cls = FakeTextInput
        agent._image_input_cls = FakeImageInput

        self.assertEqual(agent.config.thread_mode, 'per_agent')
        rendered = agent._messages_to_codex_prompt(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'inspect this observation'},
                        {'type': 'image_url', 'image_url': {'url': image_url}},
                    ],
                }
            ],
            agent_idx=0,
        )
        sdk_input = agent._to_sdk_input(rendered)

        self.assertIn('inspect this observation', rendered.prompt)
        self.assertEqual(rendered.image_urls, [image_url])
        self.assertEqual(len(sdk_input), 2)
        self.assertIsInstance(sdk_input[0], FakeTextInput)
        self.assertIsInstance(sdk_input[1], FakeImageInput)
        self.assertEqual(sdk_input[1].url, image_url)

    def test_codex_agent_without_image_input_keeps_text_only_prompt(self):
        class FakeTextInput:
            def __init__(self, text):
                self.text = text

        agent = CodexAgent(name='codex-test')
        agent._text_input_cls = FakeTextInput
        agent._image_input_cls = None

        rendered = agent._messages_to_codex_prompt(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'inspect this observation'},
                        {
                            'type': 'image_url',
                            'image_url': {'url': 'data:image/png;base64,ZmFrZS1wbmctYnl0ZXM='},
                        },
                    ],
                }
            ],
            agent_idx=0,
        )
        sdk_input = agent._to_sdk_input(rendered)

        self.assertEqual(len(sdk_input), 1)
        self.assertIsInstance(sdk_input[0], FakeTextInput)
        self.assertIn('[image input:', sdk_input[0].text)

    def test_api_and_codex_build_identical_request_messages(self):
        obs = {
            0: {
                'messages': [
                    {'role': 'system', 'content': 'system prompt'},
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': 'task prompt'},
                            {
                                'type': 'image_url',
                                'image_url': {'url': 'data:image/png;base64,ZmFrZQ=='},
                            },
                        ],
                    },
                ],
            }
        }
        api_agent = object.__new__(APIAgent)
        codex_agent = CodexAgent(name='codex-test')

        self.assertEqual(
            APIAgent.build_request_messages(api_agent, obs),
            codex_agent.build_request_messages(obs),
        )

    def test_codex_agent_forward_trace_includes_usage(self):
        class FakeThread:
            def run(self, _input):
                return SimpleNamespace(
                    final_response='<tool_call>{"name":"PushForward","arguments":{"forceScale":1}}</tool_call>',
                    usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
                )

        agent = CodexAgent(name='codex-test', timeout_s=None)
        agent._get_thread = lambda _agent_idx: FakeThread()
        agent._text_input_cls = None

        _, trace = agent.forward_with_trace({0: {'messages': [{'role': 'user', 'content': 'hi'}]}})

        self.assertEqual(trace[0]['usage']['input_tokens'], 10)
        self.assertEqual(trace[0]['usage']['output_tokens'], 5)
        self.assertEqual(trace[0]['usage']['total_tokens'], 15)

    def test_codex_agent_collects_structured_feedback_from_same_player_thread(self):
        class FakeThread:
            def __init__(self):
                self.inputs = []

            def run(self, sdk_input, **kwargs):
                del kwargs
                self.inputs.append(sdk_input)
                if len(self.inputs) == 1:
                    return SimpleNamespace(
                        final_response='<tool_call>{"name":"Tap","arguments":{"x":500,"y":500}}</tool_call>',
                        usage=None,
                    )
                return SimpleNamespace(
                    final_response=(
                        '<player_feedback>{'
                        '"summary":"tap feedback was missing",'
                        '"information_reveal_assessment":{'
                        '"classification":"complete_initially",'
                        '"evidence":"the required controls were visible from the first observation",'
                        '"attempts_considered":[1]},'
                        '"task_defects":[{'
                        '"category":"interaction_feedback","severity":"major",'
                        '"confidence":"high","attempt":1,"first_observed_turn":0,'
                        '"action":"Tap(500,500)","evidence":"no visible response",'
                        '"expected":"tap marker","observed":"unchanged UI"}],'
                        '"non_defect_observations":[],"uncertainties":[]'
                        '}</player_feedback>'
                    ),
                    usage=SimpleNamespace(input_tokens=12, output_tokens=8, total_tokens=20),
                )

        fake_thread = FakeThread()

        class FakeCodex:
            def __init__(self):
                self.thread_start_count = 0

            def thread_start(self, **kwargs):
                del kwargs
                self.thread_start_count += 1
                return fake_thread

        fake_codex = FakeCodex()
        agent = CodexAgent(
            name='codex-player',
            timeout_s=None,
            thread_mode='per_agent',
            black_box_playtest=True,
        )
        agent._codex = fake_codex
        agent._text_input_cls = None

        actions, _ = agent.forward_with_trace(
            {0: {'messages': [{'role': 'user', 'content': 'initial visible state'}]}},
        )

        feedback = agent.collect_player_feedback(
            {'messages': [{'role': 'user', 'content': 'final visible state'}]},
            agent_idx=0,
        )

        self.assertIn('"Tap"', actions[0]['action'])
        self.assertEqual(fake_codex.thread_start_count, 1)
        self.assertEqual(len(fake_thread.inputs), 2)
        self.assertEqual(feedback['status'], 'ok')
        self.assertEqual(feedback['report']['task_defects'][0]['category'], 'interaction_feedback')
        self.assertEqual(feedback['usage']['total_tokens'], 20)
        self.assertIn('Do not output another action', fake_thread.inputs[1])
        self.assertIn('Not completing the task is not itself a defect', fake_thread.inputs[1])

    def test_codex_black_box_player_uses_and_cleans_isolated_cwd(self):
        agent = CodexAgent(name='codex-player', black_box_playtest=True, isolated_cwd=True)

        isolated_path = Path(agent._resolve_cwd())

        self.assertTrue(isolated_path.is_dir())
        self.assertNotEqual(isolated_path.resolve(), Path.cwd().resolve())
        agent.close()
        self.assertFalse(isolated_path.exists())

    def test_codex_player_feedback_schema_rejects_invalid_defect_fields(self):
        base_report = {
            'summary': 'observed issue',
            'information_reveal_assessment': {
                'classification': 'complete_initially',
                'evidence': 'the necessary control was visible at reset',
                'attempts_considered': [1],
            },
            'task_defects': [{
                'category': 'action_execution',
                'severity': 'major',
                'confidence': 'high',
                'attempt': 1,
                'first_observed_turn': 0,
                'action': 'Tap(500,500)',
                'evidence': 'the visible state did not change',
                'expected': 'the button should visibly activate',
                'observed': 'the screen remained unchanged',
            }],
            'non_defect_observations': [],
            'uncertainties': [],
        }
        invalid_mutations = {
            'severity enum': lambda report: report['task_defects'][0].update(severity='banana'),
            'attempt type': lambda report: report['task_defects'][0].update(attempt='first'),
            'turn range': lambda report: report['task_defects'][0].update(first_observed_turn=-1),
            'action type': lambda report: report['task_defects'][0].update(action=None),
            'evidence type': lambda report: report['task_defects'][0].update(evidence=['not text']),
            'non-defect item type': lambda report: report.update(non_defect_observations=[3]),
            'reveal classification': lambda report: report['information_reveal_assessment'].update(classification='surprise'),
            'reveal attempts': lambda report: report['information_reveal_assessment'].update(attempts_considered=['first']),
            'intentional reveal without non-defect': lambda report: report['information_reveal_assessment'].update(classification='intentional_exploration'),
            'suspected missing info without defect': lambda report: (
                report['information_reveal_assessment'].update(classification='suspected_missing_information_defect'),
                report.update(task_defects=[]),
            ),
            'unclear reveal without uncertainty': lambda report: report['information_reveal_assessment'].update(classification='unclear'),
        }

        for label, mutate in invalid_mutations.items():
            with self.subTest(label=label):
                report = deepcopy(base_report)
                mutate(report)
                parsed, error = CodexAgent._parse_player_feedback(json.dumps(report))
                self.assertIsNone(parsed)
                self.assertIsInstance(error, str)

    def test_codex_player_feedback_accepts_intentional_multi_attempt_exploration(self):
        report = {
            'summary': 'later attempts exposed useful history as intended',
            'information_reveal_assessment': {
                'classification': 'intentional_exploration',
                'evidence': 'attempt two included the prior attempt outcome and made the next choice informed',
                'attempts_considered': [1, 2],
            },
            'task_defects': [],
            'non_defect_observations': [
                'The incomplete first attempt was an intentional discovery step rather than missing task information.',
            ],
            'uncertainties': [],
        }

        parsed, error = CodexAgent._parse_player_feedback(json.dumps(report))

        self.assertIsNone(error)
        self.assertEqual(
            parsed['information_reveal_assessment']['classification'],
            'intentional_exploration',
        )

    def test_codex_agent_passes_reasoning_effort_to_run(self):
        class FakeThread:
            def __init__(self):
                self.kwargs = None

            def run(self, _input, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    final_response='<tool_call>{"name":"PushForward","arguments":{"forceScale":1}}</tool_call>',
                    usage=None,
                )

        fake_thread = FakeThread()
        agent = CodexAgent(name='codex-test', timeout_s=None, reasoning_effort='LOW')
        agent._get_thread = lambda _agent_idx: fake_thread
        agent._text_input_cls = None

        agent.forward_with_trace({0: {'messages': [{'role': 'user', 'content': 'hi'}]}})

        self.assertEqual(fake_thread.kwargs, {'effort': 'low'})

    def test_codex_agent_rejects_unknown_reasoning_effort(self):
        with self.assertRaises(ValueError):
            CodexAgent(name='codex-test', reasoning_effort='huge')

    def test_codex_agent_usage_conversion_preserves_nested_breakdowns(self):
        usage = SimpleNamespace(
            last=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5),
            total=SimpleNamespace(input_tokens=30, output_tokens=20, total_tokens=50),
            model_context_window=1234,
        )

        converted = CodexAgent._usage_to_dict(usage)

        self.assertEqual(converted['last']['input_tokens'], 3)
        self.assertEqual(converted['total']['output_tokens'], 20)
        self.assertEqual(converted['model_context_window'], 1234)

    def test_build_model_runtimes_supports_codex_provider(self):
        class FakeCodexAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with patch('agent_ark.ark_eval.run_api_agent.CodexAgent', FakeCodexAgent):
            runtimes = build_model_runtimes([
                {
                    'name': 'codex-local',
                    'provider': 'codex',
                    'model': 'gpt-5.5',
                    'sandbox': 'read_only',
                    'timeout_s': 12,
                    'reasoning_effort': 'low',
                    'thread_mode': 'per_turn',
                }
            ])

        self.assertEqual(len(runtimes), 1)
        self.assertEqual(runtimes[0]['provider'], 'codex')
        self.assertEqual(runtimes[0]['base_url'], None)
        self.assertEqual(runtimes[0]['api_key_env'], None)
        self.assertEqual(runtimes[0]['timeout_s'], 12.0)
        self.assertEqual(runtimes[0]['agent'].kwargs['model'], 'gpt-5.5')
        self.assertEqual(runtimes[0]['agent'].kwargs['sandbox'], 'read_only')
        self.assertEqual(runtimes[0]['reasoning_effort'], 'low')
        self.assertEqual(runtimes[0]['agent'].kwargs['reasoning_effort'], 'low')
        self.assertEqual(runtimes[0]['thread_mode'], 'per_turn')
        self.assertEqual(runtimes[0]['agent'].kwargs['thread_mode'], 'per_turn')

    def test_build_model_runtimes_enables_isolated_black_box_player_feedback(self):
        class FakeCodexAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with patch('agent_ark.ark_eval.run_api_agent.CodexAgent', FakeCodexAgent):
            runtimes = build_model_runtimes([{
                'name': 'codex-player',
                'provider': 'codex',
                'model': 'gpt-5.5',
                'thread_mode': 'per_agent',
                'player_feedback': {'enabled': True},
            }])

        self.assertTrue(runtimes[0]['player_feedback']['enabled'])
        self.assertTrue(runtimes[0]['agent'].kwargs['black_box_playtest'])
        self.assertTrue(runtimes[0]['agent'].kwargs['isolated_cwd'])

    def test_build_model_runtimes_rejects_stateless_player_feedback(self):
        with self.assertRaisesRegex(ValueError, 'thread_mode: per_agent'):
            build_model_runtimes([{
                'name': 'codex-player',
                'provider': 'codex',
                'thread_mode': 'per_turn',
                'player_feedback': {'enabled': True},
            }])

    def test_build_model_runtimes_requires_read_only_player_sandbox(self):
        with self.assertRaisesRegex(ValueError, 'sandbox: read_only'):
            build_model_runtimes([{
                'name': 'codex-player',
                'provider': 'codex',
                'thread_mode': 'per_agent',
                'sandbox': 'workspace_write',
                'player_feedback': {'enabled': True},
            }])

    def test_build_model_runtimes_rejects_non_codex_player_feedback(self):
        with self.assertRaisesRegex(ValueError, 'provider: codex'):
            build_model_runtimes([{
                'name': 'api-player',
                'provider': 'openai',
                'model': 'fake-model',
                'player_feedback': {'enabled': True},
            }])

    def test_player_feedback_eval_contract_requires_all_image_trajectories(self):
        model_cfgs = [{
            'name': 'codex-player',
            'provider': 'codex',
            'player_feedback': {'enabled': True},
        }]
        valid_eval_cfg = {
            'trajectory_save': {
                'enabled': True,
                'output_path': 'tmp/player.jsonl',
                'condition': 'all',
                'include_images': True,
            },
        }

        validate_player_feedback_eval_contract(model_cfgs, valid_eval_cfg, {})

        invalid_configs = {
            'disabled': {'enabled': False, 'output_path': 'tmp/player.jsonl', 'condition': 'all', 'include_images': True},
            'conditional': {'enabled': True, 'output_path': 'tmp/player.jsonl', 'condition': 'ever_success', 'include_images': True},
            'no images': {'enabled': True, 'output_path': 'tmp/player.jsonl', 'condition': 'all', 'include_images': False},
            'no output': {'enabled': True, 'condition': 'all', 'include_images': True},
        }
        for label, trajectory_save in invalid_configs.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, 'trajectory_save'):
                    validate_player_feedback_eval_contract(
                        model_cfgs,
                        {'trajectory_save': trajectory_save},
                        {},
                    )

    def test_codex_provider_defaults_to_gpt55(self):
        class FakeCodexAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with patch('agent_ark.ark_eval.run_api_agent.CodexAgent', FakeCodexAgent):
            runtimes = build_model_runtimes([{'name': 'codex-local', 'provider': 'codex'}])

        self.assertEqual(runtimes[0]['model'], 'gpt-5.5')
        self.assertEqual(runtimes[0]['agent'].kwargs['model'], 'gpt-5.5')
        self.assertEqual(runtimes[0]['thread_mode'], 'per_agent')
        self.assertEqual(runtimes[0]['agent'].kwargs['thread_mode'], 'per_agent')

    def test_auto_reset_message_omits_repeated_task_prompt_but_keeps_new_image(self):
        img = Image.new('RGB', (2, 2), color=(0, 0, 255))
        msg_ctx = MessageContext({'enabled': True, 'max_images_per_section': 1})
        message = msg_ctx.build_auto_reset_messages(
            {'reward': -1.0, 'done': True, 'next_obs': {'step_msg': 'terminal'}},
            {'step_msg': 'new attempt start', 'vis': [[img]]},
            current_attempt_index=1,
            next_attempt_index=2,
            fallback_task_prompt='STATIC TASK PROMPT',
        )[-1]

        content = message['content']
        text = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        self.assertIn('Auto-reset started attempt 2', text)
        self.assertIn('new attempt start', text)
        self.assertNotIn('STATIC TASK PROMPT', text)
        self.assertTrue(any(part.get('type') == 'image_url' for part in content if isinstance(part, dict)))

    def test_auto_reset_message_can_omit_same_seed_new_attempt_image(self):
        img = Image.new('RGB', (2, 2), color=(0, 0, 255))
        msg_ctx = MessageContext({'enabled': True, 'max_images_per_section': 1})
        message = msg_ctx.build_auto_reset_messages(
            {'reward': -1.0, 'done': True, 'next_obs': {'step_msg': 'terminal'}},
            {'step_msg': 'new attempt start', 'vis': [[img]]},
            current_attempt_index=1,
            next_attempt_index=2,
            omit_reset_obs_images=True,
            reset_obs_image_omitted_reason='omitted because attempt 2 uses the same task seed as attempt 1; its initial view is the same as the first attempt initial observation.',
        )[-1]

        content = message['content']
        text = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        self.assertIn('Auto-reset started attempt 2', text)
        self.assertIn('same task seed as attempt 1', text)
        self.assertFalse(any(part.get('type') == 'image_url' for part in content if isinstance(part, dict)))

    def test_initial_messages_separate_task_prompt_and_reset_context(self):
        img = Image.new('RGB', (2, 2), color=(0, 0, 255))
        msg_ctx = MessageContext({'enabled': True, 'max_images_per_section': 1})
        messages = msg_ctx.build_chat_messages(
            {
                'task_prompt': 'STATIC TASK PROMPT',
                'reset_context': 'Goal: red=2 yellow=1 blue=0',
                'step_msg': 'STATIC TASK PROMPT\n\n<reset_context>\nGoal: red=2 yellow=1 blue=0\n</reset_context>',
                'vis': [[img]],
            },
            history_episodes=[],
            current_episode_steps=[],
        )

        content = messages[-1]['content']
        text = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        self.assertEqual(text.count('STATIC TASK PROMPT'), 1)
        self.assertEqual(text.count('Goal: red=2 yellow=1 blue=0'), 1)
        self.assertIn('Reset context:', text)
        self.assertTrue(any(part.get('type') == 'image_url' for part in content if isinstance(part, dict)))

    def test_message_context_orders_recent_frames_chronologically(self):
        frames = [Image.new('RGB', (2, 2), color=(idx, 0, 0)) for idx in range(4)]
        msg_ctx = MessageContext({
            'enabled': True,
            'max_images_per_section': 3,
            'image_caption': True,
        })

        parts = msg_ctx._obs_images_to_parts(
            {'vis': [frames]},
            section='current_attempt',
            step_idx=1,
            which_obs='post',
        )

        captions = [
            part['text']
            for part in parts
            if isinstance(part, dict) and part.get('type') == 'text'
        ]
        self.assertEqual(
            captions,
            [
                'Image: [Current attempt · Step 1 · After action · Camera 1 · Frame 2/4]',
                'Image: [Current attempt · Step 1 · After action · Camera 1 · Frame 3/4]',
                'Image: [Current attempt · Step 1 · After action · Camera 1 · Frame 4/4]',
            ],
        )

    def test_auto_reset_context_follows_same_seed_initial_omission(self):
        img = Image.new('RGB', (2, 2), color=(0, 0, 255))
        msg_ctx = MessageContext({'enabled': True, 'max_images_per_section': 1})
        message = msg_ctx.build_auto_reset_messages(
            {'reward': -1.0, 'done': True, 'next_obs': {'step_msg': 'terminal'}},
            {
                'task_prompt': 'STATIC TASK PROMPT',
                'reset_context': 'Goal: red=2 yellow=1 blue=0',
                'step_msg': 'STATIC TASK PROMPT\n\n<reset_context>\nGoal: red=2 yellow=1 blue=0\n</reset_context>',
                'vis': [[img]],
            },
            current_attempt_index=1,
            next_attempt_index=2,
            omit_reset_obs_images=True,
            reset_obs_image_omitted_reason='omitted because attempt 2 uses the same task seed as attempt 1; its initial view is the same as the first attempt initial observation.',
        )[-1]

        content = message['content']
        text = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        self.assertIn('same task seed as attempt 1', text)
        self.assertNotIn('Goal: red=2', text)
        self.assertNotIn('STATIC TASK PROMPT', text)
        self.assertFalse(any(part.get('type') == 'image_url' for part in content if isinstance(part, dict)))

    def test_step_context_list_messages_are_rendered(self):
        msg_ctx = MessageContext({'enabled': True, 'max_images_per_section': 0})
        messages = msg_ctx.build_step_messages({
            'reward': 0.0,
            'done': False,
            'next_obs': {
                'step_msg': [
                    '<step_context>Door opened</step_context>',
                    'Plain observation note',
                ],
                'vis': [],
            },
            'action': '<tool_call>{"name":"OpenDoor","arguments":{}}</tool_call>',
        })

        content = messages[-1]['content']
        text = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        self.assertIn('Step context:', text)
        self.assertIn('Door opened', text)
        self.assertIn('Plain observation note', text)

    def test_step_message_omits_post_image_when_invalid_action_did_not_reach_unity(self):
        img = Image.new('RGB', (2, 2), color=(0, 0, 255))
        msg_ctx = MessageContext({'enabled': True, 'max_images_per_section': 1})
        messages = msg_ctx.build_step_messages({
            'reward': -1.0,
            'done': False,
            'next_obs': {'step_msg': '[compile] Error: Invalid assistant tool/function-call action', 'vis': [[img]]},
            'action': '<tool_call>bad</tool_call>',
            'omit_next_obs_images': True,
            'next_obs_image_omitted_reason': 'omitted because the previous assistant tool/function-call was invalid and no Unity action was executed; the visual state is unchanged from the previous observation.',
        })

        content = messages[-1]['content']
        text = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        self.assertIn('no Unity action was executed', text)
        self.assertIn('Invalid assistant tool/function-call action', text)
        self.assertFalse(any(part.get('type') == 'image_url' for part in content if isinstance(part, dict)))

    def test_auto_reset_omits_terminal_image_when_invalid_action_did_not_reach_unity(self):
        terminal_img = Image.new('RGB', (2, 2), color=(255, 0, 0))
        reset_img = Image.new('RGB', (2, 2), color=(0, 0, 255))
        msg_ctx = MessageContext({'enabled': True, 'max_images_per_section': 1})
        message = msg_ctx.build_auto_reset_messages(
            {
                'reward': -1.0,
                'done': True,
                'next_obs': {'step_msg': '[compile] Error: Invalid assistant tool/function-call action', 'vis': [[terminal_img]]},
                'omit_next_obs_images': True,
                'next_obs_image_omitted_reason': 'omitted because the previous assistant tool/function-call was invalid and no Unity action was executed; the visual state is unchanged from the previous observation.',
            },
            {'step_msg': 'new attempt start', 'vis': [[reset_img]]},
            current_attempt_index=1,
            next_attempt_index=2,
            omit_terminal_obs_images=True,
            terminal_obs_image_omitted_reason='omitted because the previous assistant tool/function-call was invalid and no Unity action was executed; the visual state is unchanged from the previous observation.',
        )[-1]

        content = message['content']
        text = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        self.assertIn('Terminal observation image(s): omitted because', text)
        self.assertIn('no Unity action was executed', text)
        self.assertEqual(sum(1 for part in content if isinstance(part, dict) and part.get('type') == 'image_url'), 1)

    def test_rollout_auto_reset_applies_image_omission_flags(self):
        img = Image.new('RGB', (2, 2), color=(0, 0, 255))
        env_cfg = {
            'env_wrapper_cfg': {
                'context_manager': {
                    'messages': {
                        'enabled': True,
                        'append_only': True,
                        'return_mode': 'delta',
                        'max_images_per_section': 1,
                    }
                }
            }
        }
        ctx = ArkRolloutContext()
        ctx.on_reset(env_cfg, {0: {'step_msg': 'initial', 'vis': [[img]]}})
        ctx.record_transition(
            {0: {'step_msg': '[compile] Error: Invalid assistant tool/function-call action', 'vis': [[img]], 'skip_infer': True}},
            {0: {'action': '<tool_call>bad</tool_call>', 'assistant': 'bad action'}},
            {0: -1.0},
            {0: True, '__all__': False},
            info={'func_render_errors': {0: 'bad action'}},
        )
        obs = ctx.on_attempt_reset(
            env_cfg,
            {0: {'step_msg': 'new attempt start', 'vis': [[img]]}},
            current_attempt_index=1,
            next_attempt_index=2,
            omit_terminal_obs_images_by_uid={0: True},
            terminal_obs_image_omitted_reason_by_uid={0: 'omitted because the previous assistant tool/function-call was invalid and no Unity action was executed; the visual state is unchanged from the previous observation.'},
            omit_reset_obs_images=True,
            reset_obs_image_omitted_reason='omitted because attempt 2 uses the same task seed as attempt 1; its initial view is the same as the first attempt initial observation.',
        )

        content = obs[0]['messages'][-1]['content']
        text = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        self.assertIn('no Unity action was executed', text)
        self.assertIn('same task seed as attempt 1', text)
        self.assertFalse(any(part.get('type') == 'image_url' for part in content if isinstance(part, dict)))

    def test_auto_reset_message_includes_terminal_render_error_from_skip_infer_done(self):
        img = Image.new('RGB', (2, 2), color=(0, 0, 255))
        env_cfg = {
            'env_wrapper_cfg': {
                'context_manager': {
                    'messages': {
                        'enabled': True,
                        'append_only': True,
                        'return_mode': 'delta',
                        'max_images_per_section': 1,
                    }
                }
            }
        }
        ctx = ArkRolloutContext()
        ctx.on_reset(env_cfg, {0: {'step_msg': 'initial', 'vis': [[img]]}})
        ctx.record_transition(
            {0: {'step_msg': '[compile] Error: Invalid assistant tool/function-call action before Unity execution. The previous assistant output could not be converted into executable Unity code, so no environment action was run. Fix the assistant tool_call format, tool name, or arguments; this is not a Unity environment rendering/image problem. Details: JSONDecodeError', 'skip_infer': True}},
            {0: {'action': '<tool_call>bad</tool_call>', 'assistant': 'bad action'}},
            {0: -1.0},
            {0: True, '__all__': False},
        )
        obs = ctx.on_attempt_reset(
            env_cfg,
            {0: {'step_msg': 'new attempt start', 'vis': [[img]]}},
            current_attempt_index=1,
            next_attempt_index=2,
        )

        message = obs[0]['messages'][-1]
        content = message['content']
        text = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        self.assertIn('Invalid assistant tool/function-call action', text)
        self.assertIn('no environment action was run', text)
        self.assertIn('not a Unity environment rendering/image problem', text)
        self.assertIn('JSONDecodeError', text)
        self.assertIn('Auto-reset started attempt 2', text)
        self.assertTrue(any(part.get('type') == 'image_url' for part in content if isinstance(part, dict)))

    def test_hook_manager_fans_out_events(self):
        hook = CollectorHook()
        manager = HookManager([hook])
        event = manager.emit('agent_response', {'action': '<params>{}</params>'}, source='test')
        self.assertEqual(event['seq'], 1)
        self.assertEqual(hook.events[0]['event'], 'agent_response')
        self.assertEqual(hook.events[0]['payload']['action'], '<params>{}</params>')

    def test_human_action_broker_waits_for_submission(self):
        broker = HumanActionBroker()
        result = []

        def wait():
            result.append(broker.wait_for_action(agent_id=3, timeout=2.0))

        thread = threading.Thread(target=wait)
        thread.start()
        broker.submit('<tool_call>{}</tool_call>', agent_id=3)
        thread.join(timeout=2.0)
        self.assertEqual(result, ['<tool_call>{}</tool_call>'])

    def test_local_viewer_http_submission_reaches_human_broker(self):
        broker = HumanActionBroker()
        viewer = LocalViewerHook(port=self._unused_local_port(), action_broker=broker)
        expected_action = '<tool_call>{"name":"Ping","arguments":{}}</tool_call>'
        result = []

        viewer.start()
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    response = requests.get(f'{viewer.url}/health', timeout=0.2)
                except requests.RequestException:
                    time.sleep(0.05)
                    continue
                if response.status_code == 200:
                    break
                time.sleep(0.05)
            else:
                self.fail('LocalViewerHook did not become healthy in time')

            thread = threading.Thread(
                target=lambda: result.append(broker.wait_for_action(agent_id=0, timeout=2.0))
            )
            thread.start()

            response = requests.post(
                f'{viewer.url}/human/actions',
                json={'agent_id': None, 'action': expected_action},
                timeout=1.0,
            )

            thread.join(timeout=3.0)
            self.assertEqual(response.status_code, 200)
            self.assertFalse(thread.is_alive(), 'Human action waiter should be released by HTTP submission')
            self.assertEqual(result, [expected_action])
        finally:
            viewer.close()

    def test_local_viewer_html_deduplicates_by_event_seq_not_content(self):
        self.assertIn('let seenEventSeqs = new Set();', _VIEWER_HTML)
        self.assertIn('let renderedMessageCounts = new Map();', _VIEWER_HTML)
        self.assertIn('let pendingAssistantByAgent = new Map();', _VIEWER_HTML)
        self.assertIn('if (seenEventSeqs.has(seq)) return;', _VIEWER_HTML)
        self.assertIn('let startIndex = renderedMessageCounts.get(agentId) || 0;', _VIEWER_HTML)
        self.assertIn('for (let idx = startIndex; idx < messages.length; idx += 1)', _VIEWER_HTML)
        self.assertIn('renderedMessageCounts.set(agentId, messages.length);', _VIEWER_HTML)
        self.assertIn("pendingAssistantByAgent.set(String(item.agentId), String(text));", _VIEWER_HTML)
        self.assertIn("pendingAssistantByAgent.delete(agentId);", _VIEWER_HTML)
        self.assertIn('function extractTurnIndex(ev)', _VIEWER_HTML)
        self.assertIn('function buildSeparatorText(text, ev)', _VIEWER_HTML)
        self.assertIn('function buildBubbleMeta(agentId, ev)', _VIEWER_HTML)
        self.assertIn('parts.push(`turn ${turnIndex + 1}`);', _VIEWER_HTML)
        self.assertIn('return `seq ${seq} · ${text}`;', _VIEWER_HTML)
        self.assertNotIn('parts.push(`seq ${seq}`);', _VIEWER_HTML)
        self.assertNotIn("else if (ev.event === 'human_response' && payload.action) appendBubble", _VIEWER_HTML)
        self.assertNotIn('renderedMessageKeys', _VIEWER_HTML)
        self.assertNotIn('stableContentKey', _VIEWER_HTML)

    def test_run_case_rollout_passes_turn_index_to_env_step(self):
        class FakeAgent:
            def reset(self):
                return

            def build_request_messages(self, obs):
                return {1: [{'role': 'user', 'content': 'request'}]}

            def forward_with_trace(self, obs):
                return (
                    {1: {'action': '<tool_call>{}</tool_call>', 'assistant': '<tool_call>{}</tool_call>'}},
                    {1: {'assistant_raw': '<tool_call>{}</tool_call>', 'action_extracted': '<tool_call>{}</tool_call>'}},
                )

        class FakeEnv:
            def __init__(self):
                self.max_attempts = 1
                self.max_steps_per_attempt = 1
                self.last_step_info = None
                self._selected_env_cfg = {'task_name': 'trace-task', 'env_id': 0}

                class _FakeInfoMgr:
                    env_config = {'task_name': 'trace-task', 'env_id': 0}

                self.sub_env = SimpleNamespace(env_info_mgr=_FakeInfoMgr())

            def reset(self, **kwargs):
                return ({1: {'step_msg': 'initial'}}, {})

            def step(self, code_act, info=None):
                self.last_step_info = dict(info or {})
                return (
                    {1: {'step_msg': 'done'}},
                    {1: 1.0},
                    {1: True, '__all__': True},
                    {'rollout': {'success': True, 'truncated': False}},
                )

        env = FakeEnv()
        _run_case_rollout(
            env=env,
            model_runtime={
                'name': 'trace-model',
                'model': 'fake-model',
                'provider': 'fake-provider',
                'base_url': None,
                'api_key_env': None,
                'agent': FakeAgent(),
            },
            case={'case_id': 'case-0000', 'task_name': 'trace-task', 'group_seed': 7, 'env_id': 0},
            hook_manager=HookManager(),
            hooks_cfg={'visualization': {'text_max_chars': 6000, 'max_images_per_observation': 4}},
        )

        self.assertEqual(env.last_step_info, {'turn_index': 0})

    def test_run_case_rollout_emits_raw_agent_trace_fields(self):
        class FakeTracingAgent:
            def reset(self):
                return

            def build_request_messages(self, obs):
                return {
                    1: [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': 'exact request'},
                            ],
                        }
                    ]
                }

            def forward_with_trace(self, obs):
                response_text = '<think>reasoning</think>\n<tool_call>{"name":"Ping","arguments":{}}</tool_call>'
                return (
                    {
                        1: {
                            'action': '<tool_call>{"name":"Ping","arguments":{}}</tool_call>',
                            'assistant': response_text,
                        }
                    },
                    {
                        1: {
                            'assistant_raw': response_text,
                            'action_extracted': '<tool_call>{"name":"Ping","arguments":{}}</tool_call>',
                        }
                    },
                )

        class FakeEnv:
            def __init__(self):
                self.max_attempts = 1
                self.max_steps_per_attempt = 1
                self._selected_env_cfg = {'task_name': 'trace-task', 'env_id': 0}

                class _FakeInfoMgr:
                    env_config = {'task_name': 'trace-task', 'env_id': 0}

                self.sub_env = SimpleNamespace(env_info_mgr=_FakeInfoMgr())

            def reset(self, **kwargs):
                return (
                    {
                        1: {
                            'step_msg': 'initial',
                            'messages': [
                                {
                                    'role': 'user',
                                    'content': [
                                        {'type': 'text', 'text': 'exact request'},
                                    ],
                                }
                            ],
                        }
                    },
                    {},
                )

            def step(self, code_act, info=None):
                return (
                    {1: {'step_msg': 'done'}},
                    {1: 1.0},
                    {1: True, '__all__': True},
                    {'rollout': {'success': True, 'truncated': False}},
                )

        hook = CollectorHook()
        manager = HookManager([hook])
        env = FakeEnv()
        model_runtime = {
            'name': 'trace-model',
            'model': 'fake-model',
            'provider': 'fake-provider',
            'base_url': None,
            'api_key_env': None,
            'agent': FakeTracingAgent(),
        }

        rollout = _run_case_rollout(
            env=env,
            model_runtime=model_runtime,
            case={'case_id': 'case-0000', 'task_name': 'trace-task', 'group_seed': 7, 'env_id': 0},
            hook_manager=manager,
            hooks_cfg={'visualization': {'text_max_chars': 6000, 'max_images_per_observation': 4}},
        )

        request_event = next(event for event in hook.events if event['event'] == 'agent_request')
        response_event = next(event for event in hook.events if event['event'] == 'agent_response')

        self.assertEqual(
            request_event['payload']['raw_trace_by_agent']['1']['request_messages'][0]['content'][0]['text'],
            'exact request',
        )
        self.assertEqual(
            response_event['payload']['raw_trace_by_agent']['1']['assistant_raw'],
            '<think>reasoning</think>\n<tool_call>{"name":"Ping","arguments":{}}</tool_call>',
        )
        self.assertEqual(
            response_event['payload']['raw_trace_by_agent']['1']['action_extracted'],
            '<tool_call>{"name":"Ping","arguments":{}}</tool_call>',
        )
        self.assertEqual(
            rollout['step_records'][0]['raw_request_by_agent']['1']['request_messages'][0]['content'][0]['text'],
            'exact request',
        )
        self.assertNotIn('final_obs', rollout)

    def test_evaluate_case_collects_player_feedback_after_final_observation(self):
        class FakeFeedbackAgent:
            def __init__(self):
                self.feedback_obs = None

            def reset(self):
                return

            def build_request_messages(self, obs):
                return {1: [{'role': 'user', 'content': obs[1].get('step_msg', '')}]}

            def forward_with_trace(self, obs):
                del obs
                action = '<tool_call>{"name":"Tap","arguments":{"x":500,"y":500}}</tool_call>'
                return ({1: {'action': action, 'assistant': action}}, {1: {'assistant_raw': action}})

            def collect_player_feedback(self, obs_dict, *, agent_idx):
                self.feedback_obs = (obs_dict, agent_idx)
                return {
                    'status': 'ok',
                    'report': {
                        'summary': 'visible tap did not change the UI',
                        'information_reveal_assessment': {
                            'classification': 'complete_initially',
                            'evidence': 'the relevant control was visible before the tap',
                            'attempts_considered': [1],
                        },
                        'task_defects': [{'category': 'action_execution'}],
                        'non_defect_observations': [],
                        'uncertainties': [],
                    },
                }

        class FakeFeedbackEnv:
            def __init__(self):
                self.max_attempts = 1
                self.max_steps_per_attempt = 1
                self._selected_env_cfg = {'task_name': 'feedback-task', 'env_id': 0}

                class _FakeInfoMgr:
                    env_config = {'task_name': 'feedback-task', 'env_id': 0}

                self.sub_env = SimpleNamespace(env_info_mgr=_FakeInfoMgr())

            def reset(self, **kwargs):
                del kwargs
                return ({1: {'step_msg': 'initial'}}, {'attempt': {'index': 1}})

            def step(self, code_act, info=None):
                del code_act, info
                return (
                    {1: {'step_msg': 'final visible state'}},
                    {1: 0.0},
                    {1: True, '__all__': True},
                    {
                        'attempt': {'index': 1, 'done': True, 'success': True},
                        'rollout': {'success': True, 'truncated': False},
                    },
                )

        agent = FakeFeedbackAgent()
        result = evaluate_case(
            env=FakeFeedbackEnv(),
            model_runtime={
                'name': 'codex-player',
                'model': 'fake-model',
                'provider': 'codex',
                'base_url': None,
                'api_key_env': None,
                'player_feedback': {'enabled': True},
                'agent': agent,
            },
            case={'case_id': 'feedback-case', 'task_name': 'feedback-task', 'group_seed': 1, 'env_id': 0},
        )

        self.assertEqual(agent.feedback_obs[0]['step_msg'], 'final visible state')
        self.assertEqual(agent.feedback_obs[1], 1)
        self.assertEqual(result['player_feedback']['status'], 'ok')
        self.assertEqual(
            result['player_feedback']['report']['task_defects'][0]['category'],
            'action_execution',
        )

        agent.collect_player_feedback = lambda obs_dict, agent_idx: {
            'status': 'unparsed',
            'assistant_raw': 'not-json',
            'report': None,
        }
        failed_feedback_result = evaluate_case(
            env=FakeFeedbackEnv(),
            model_runtime={
                'name': 'codex-player',
                'model': 'fake-model',
                'provider': 'codex',
                'base_url': None,
                'api_key_env': None,
                'player_feedback': {'enabled': True},
                'agent': agent,
            },
            case={'case_id': 'feedback-case-retry', 'task_name': 'feedback-task', 'group_seed': 1, 'env_id': 0},
        )

        self.assertEqual(failed_feedback_result['status'], 'error')
        self.assertEqual(failed_feedback_result['error_type'], 'PlayerFeedbackGateError')

        plain_agent = FakeFeedbackAgent()
        plain_result = evaluate_case(
            env=FakeFeedbackEnv(),
            model_runtime={
                'name': 'plain-api-agent',
                'model': 'fake-model',
                'provider': 'openai',
                'base_url': None,
                'api_key_env': None,
                'agent': plain_agent,
            },
            case={'case_id': 'plain-case', 'task_name': 'feedback-task', 'group_seed': 1, 'env_id': 0},
        )

        self.assertEqual(plain_result['status'], 'ok')
        self.assertNotIn('player_feedback', plain_result)
        self.assertIsNone(plain_agent.feedback_obs)

    def test_collect_raw_request_messages_omits_image_payloads(self):
        class FakeAgent:
            def build_request_messages(self, obs):
                del obs
                return {
                    3: [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': 'observe carefully'},
                                {
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': 'data:image/png;base64,abc123xyz',
                                    },
                                },
                            ],
                        }
                    ]
                }

        trace = collect_raw_request_messages(FakeAgent(), {3: {'step_msg': 'ignored'}})
        image_url = trace['3']['request_messages'][0]['content'][1]['image_url']['url']

        self.assertIn('omitted image data URL payload', image_url)
        self.assertNotIn('abc123xyz', image_url)

    def test_local_viewer_html_has_raw_trace_panel_anchor(self):
        self.assertIn('Latest Raw Trace', _VIEWER_HTML)
        self.assertIn('rawRequestTrace', _VIEWER_HTML)
        self.assertIn('rawResponseTrace', _VIEWER_HTML)
        self.assertIn('trace-box', _VIEWER_HTML)
        self.assertIn('overflow: auto', _VIEWER_HTML)
        self.assertIn('max-height: 32vh', _VIEWER_HTML)
        self.assertIn('padding-right: 4px', _VIEWER_HTML)
        self.assertIn("ev && ev.event === 'run_start'", _VIEWER_HTML)
        self.assertIn("appendSeparator('new run'", _VIEWER_HTML)

    def test_human_interaction_forces_viewer_browser_open(self):
        manager, action_broker = build_eval_hook_manager({
            'visualization': {
                'enabled': True,
                'open_browser': False,
            },
            'human_interaction': {
                'enabled': True,
                'name': 'human-test',
            },
        })

        self.assertIsNotNone(action_broker)
        viewer = next(hook for hook in manager._hooks if isinstance(hook, LocalViewerHook))
        self.assertTrue(viewer.open_browser)

    def test_human_runtime_does_not_require_models(self):
        hook = CollectorHook()
        manager = HookManager([hook])
        runtimes = build_model_runtimes(
            [],
            hooks_cfg={'human_interaction': {'enabled': True, 'name': 'human-test'}},
            hook_manager=manager,
            action_broker=HumanActionBroker(),
        )
        self.assertEqual(len(runtimes), 1)
        self.assertEqual(runtimes[0]['name'], 'human-test')
        self.assertTrue(runtimes[0]['human_interaction'])

    def test_human_runtime_cannot_replace_configured_codex_player_feedback(self):
        with self.assertRaisesRegex(ValueError, 'human_interaction'):
            build_model_runtimes(
                [{
                    'name': 'codex-player',
                    'provider': 'codex',
                    'player_feedback': {'enabled': True},
                }],
                hooks_cfg={'human_interaction': {'enabled': True}},
                action_broker=HumanActionBroker(),
            )

    def test_eval_config_contains_explicit_hooks(self):
        cfg = load_eval_config('config/ark_env/eval_seed1.example.yaml')
        self.assertIn('hooks', cfg)
        self.assertIn('visualization', cfg['hooks'])
        self.assertIn('human_interaction', cfg['hooks'])

    def test_env_runtime_emits_server_side_events_with_fake_env(self):
        img = Image.new('RGB', (2, 2), color=(0, 255, 0))

        class FakeEnv:
            def __init__(self):
                self.actions = []

            def reset(self, **kwargs):
                return {7: {'step_msg': 'reset', 'vis': [[img]]}}, {'rollout': {'success': False}}

            def step(self, code_act):
                self.actions.append(code_act)
                return (
                    {7: {'step_msg': 'next', 'vis': [[img]]}},
                    {7: 0.25},
                        {7: True, '__all__': True},
                    {'attempt': {'done': True}},
                )

            def close(self):
                return

        hook = CollectorHook()
        runtime = object.__new__(EnvRuntime)
        runtime.env_id = 'fake-env'
        runtime.cfg = {}
        fake_env = FakeEnv()
        runtime.env = fake_env
        runtime.lock = threading.RLock()
        runtime.hooks = HookManager([hook])
        runtime._hook_text_max_chars = 6000
        runtime._hook_max_images_per_observation = 1
        runtime.active_unity_id = None
        runtime.expected_unity_ids = []
        runtime.started = False

        runtime.start_interaction(task_name='task')
        payload = runtime.step('<params>{}</params>', assistant='raw assistant')

        self.assertEqual([event['event'] for event in hook.events], ['env_reset', 'env_step'])
        self.assertEqual(hook.events[0]['payload']['unity_id'], 7)
        self.assertEqual(hook.events[0]['payload']['obs']['7']['step_msg'], 'reset')
        self.assertEqual(fake_env.actions[0][7], {'action': '<params>{}</params>', 'assistant': 'raw assistant'})
        self.assertEqual(payload['unity_id'], 7)
        self.assertTrue(payload['done'])
        self.assertEqual(hook.events[1]['payload']['reward'], 0.25)


if __name__ == '__main__':
    unittest.main()
