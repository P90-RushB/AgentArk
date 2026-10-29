"""Behavioral checks execute the same JavaScript shipped by LocalViewerHook."""
import json
import shutil
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path

from PIL import Image
import numpy as np

from agent_ark.ark_env.ark_env import ArkEnv, ArkRolloutContext
from agent_ark.interaction.hooks import HookManager
from agent_ark.interaction.local_viewer import LocalViewerHook, _VIEWER_HTML
from agent_ark.interaction.serialization import serialize_obs_map


NODE = shutil.which('node')
HARNESS = Path(__file__).with_name('viewer_dom_harness.js')


def message(text, role='user', images=()):
    content = text
    if images:
        content = [{'type': 'text', 'text': text}] + [
            {'type': 'image_url', 'url': f'data:image/png;base64,{image}'} for image in images
        ]
    return {'role': role, 'content': content}


def observation(seq, messages, *, event='env_step', mode=None, observation_id=None, agent='7', **payload):
    payload['next_obs' if event == 'env_step' else 'obs'] = {agent: {'messages': messages}}
    if mode is not None:
        payload['message_mode'] = mode
    if observation_id is not None:
        payload['observation_id'] = observation_id
    return {'seq': seq, 'event': event, 'payload': payload}


@unittest.skipUnless(NODE, 'Node.js is required to execute viewer JavaScript behavior checks')
class LocalViewerBehaviorTest(unittest.TestCase):
    def render(self, events=(), state_events=(), snapshots=False, clear_at=(), state_images=None, expand_model_calls=True):
        result = subprocess.run(
            [NODE, str(HARNESS)],
            input=json.dumps({'html': _VIEWER_HTML, 'events': events, 'state_events': state_events,
                              'snapshots': snapshots, 'clear_at': clear_at, 'state_images': state_images or {},
                              'expand_model_calls': expand_model_calls}),
            text=True, encoding='utf-8', capture_output=True, check=True, timeout=15,
        )
        return json.loads(result.stdout)

    @staticmethod
    def bubbles(result, role=None):
        return [row['html'] for row in result['rows'] if row['className'] != 'bubble-row separator'
                and (role is None or row['className'] == f'bubble-row {role}')]

    def rebuilt_full_events(self, *, history_limit=None, history_enabled=True,
                            current_history=True, assistant=True, same=False, terminal=False, step_count=2):
        """Use the real MessageContext path and its observer boundary, not hand-written messages."""
        context = ArkRolloutContext()
        collector = type('Collector', (), {'handle_event': lambda self, event: self.events.append(event)})()
        collector.events = []
        env = object.__new__(ArkEnv)
        env.rollout_ctx = context
        env.hooks = HookManager([collector])
        cfg = {'env_wrapper_cfg': {'context_manager': {
            'history': {'max_history_attempts': 1 if history_enabled else 0,
                        'max_history_steps_per_attempt': history_limit},
            'messages': {'enabled': True, 'append_only': False, 'return_mode': 'full',
                         'include_current_attempt_history': current_history, 'image_caption': True},
        }}}
        initial_text = 'Current attempt start.\n\nRequired user text stays visible.'
        states = [
            {'step_msg': initial_text if same or index == 0 else f'result{index}',
             'vis': [[Image.new('RGB', (1, 1), 'red' if same else color)]]}
            for index, color in enumerate(('red', 'green', 'blue'))
        ]
        obs = context.on_reset(cfg, {7: deepcopy(states[0])})
        env._emit_env_event('env_reset', {'obs': serialize_obs_map(obs)})
        self.assertNotIn('new_message_start_by_agent', collector.events[-1]['payload'])
        for index, state in enumerate(states[1:1 + step_count], 1):
            done = terminal and index == 2
            actions = {7: {'assistant': 'same action' if assistant else None,
                           'action': 'same action' if assistant else None}}
            raw = {7: deepcopy(state)}
            context.record_transition(raw, actions, {7: 0}, {7: done})
            obs = context.finalize_obs(raw, rollout_done={7: done})
            before = deepcopy(obs)
            env._emit_env_event('env_step', {'next_obs': serialize_obs_map(obs)})
            self.assertEqual(obs, before)
        return collector.events, context, env, cfg

    def test_real_rebuilt_full_initial_rewrite_only_adds_current_step(self):
        events, _, _, _ = self.rebuilt_full_events()
        result = self.render(events, snapshots=True)
        self.assertEqual([len(rows) for rows in result['snapshots']], [2, 4, 6])
        self.assertEqual([sum(row['html'].count('<img ') for row in rows) for rows in result['snapshots']], [1, 2, 3])
        self.assertEqual(events[1]['payload']['new_message_start_by_agent'], {'7': 2})
        self.assertEqual(events[2]['payload']['new_message_start_by_agent'], {'7': 4})

    def test_unmarked_real_rebuilt_history_preserves_legacy_count_contract(self):
        for mode in (None, 'unknown'):
            with self.subTest(mode=mode):
                events, _, _, _ = self.rebuilt_full_events()
                for event in events:
                    event['payload'].pop('message_mode')
                    event['payload'].pop('observation_id')
                    event['payload'].pop('new_message_start_by_agent', None)
                    if mode is not None:
                        event['payload']['message_mode'] = mode
                result = self.render(events, snapshots=True)
                # Reference counts from the original viewer's actual JS renderer.
                self.assertEqual([len(rows) for rows in result['snapshots']], [2, 4, 6])
                self.assertEqual([sum(row['html'].count('<img ') for row in rows) for rows in result['snapshots']], [1, 2, 3])

    def test_first_full_snapshot_ignores_tail_offset_until_history_is_visible(self):
        events, _, _, _ = self.rebuilt_full_events()
        # A late viewer connection or Clear View may start after reset was evicted.
        result = self.render(events[1:2])
        self.assertEqual(len(self.bubbles(result)), 4)
        self.assertEqual(sum(row.count('<img ') for row in self.bubbles(result)), 2)

    def test_raw_fallback_does_not_establish_messages_history_or_count(self):
        raw = {'seq': 1, 'event': 'env_step', 'payload': {'next_obs': {'7': {'step_msg': 'raw'}}}}
        for mode in (None, 'full'):
            with self.subTest(mode=mode):
                incoming = observation(2, [message('system', 'system'), message('initial')], mode=mode,
                                       new_message_start_by_agent={'7': 1})
                result = self.render([raw, incoming])
                self.assertEqual(len(self.bubbles(result)), 3)
                self.assertTrue(any('>system<' in row for row in self.bubbles(result)))

    def test_clear_view_restarts_from_complete_real_full_snapshot(self):
        events, _, _, _ = self.rebuilt_full_events()
        result = self.render(events, snapshots=True, clear_at=[1])
        self.assertEqual([len(rows) for rows in result['snapshots']], [2, 4, 6])
        self.assertEqual([sum(row['html'].count('<img ') for row in rows) for rows in result['snapshots']], [1, 2, 3])
        self.assertEqual(result['raw_count'], '2 raw events stored')

    def test_real_rebuilt_full_bounded_window_preserves_each_new_frame(self):
        events, _, _, _ = self.rebuilt_full_events(history_limit=1, terminal=True)
        result = self.render(events, snapshots=True)
        self.assertEqual([len(rows) for rows in result['snapshots']], [2, 4, 6])
        self.assertEqual([sum(row['html'].count('<img ') for row in rows) for rows in result['snapshots']], [1, 2, 3])
        self.assertEqual(events[2]['payload']['new_message_start_by_agent'], {'7': 2})

    def test_real_rebuilt_full_repeated_images_and_user_header_are_not_stripped(self):
        events, _, _, _ = self.rebuilt_full_events(history_limit=1, same=True)
        result = self.render(events, snapshots=True)
        self.assertEqual([len(rows) for rows in result['snapshots']], [2, 4, 6])
        self.assertEqual(sum(row.count('<img ') for row in self.bubbles(result)), 3)
        # Required text resembles a formatter header; no content normalization is used.
        self.assertEqual(sum('Required user text stays visible.' in row for row in self.bubbles(result)), 3)
        self.assertEqual(sum('Current attempt start.' in row for row in self.bubbles(result)), 3)

    def test_real_rebuilt_full_without_current_history_or_assistant(self):
        for history_enabled, current_history, assistant, expected_rows in [
            (False, True, True, [2, 3, 4]),
            (True, False, True, [2, 3, 4]),
            (True, True, False, [2, 3, 4]),
        ]:
            with self.subTest(history=history_enabled, current=current_history, assistant=assistant):
                events, _, _, _ = self.rebuilt_full_events(
                    history_enabled=history_enabled, current_history=current_history, assistant=assistant,
                )
                result = self.render(events, snapshots=True)
                self.assertEqual([len(rows) for rows in result['snapshots']], expected_rows)
                self.assertEqual(sum(row.count('<img ') for row in self.bubbles(result)), 3)
                for event in events[1:]:
                    count = len(event['payload']['next_obs']['7']['messages'])
                    self.assertEqual(event['payload']['new_message_start_by_agent']['7'], count - 1)

    def test_real_rebuilt_full_skip_and_attempt_reset_clear_tail_metadata(self):
        events, context, env, cfg = self.rebuilt_full_events()
        skipped = {7: {'step_msg': 'skipped', 'skip_infer': True}}
        context.finalize_obs(skipped)
        env._emit_env_event('env_step', {'next_obs': serialize_obs_map(skipped)})
        # Access the real HookManager collector for events emitted after the helper.
        emitted = env.hooks._hooks[0].events
        self.assertNotIn('new_message_start_by_agent', emitted[-1]['payload'])
        reset = context.on_attempt_reset(cfg, {7: {'step_msg': 'new attempt'}})
        env._emit_env_event('env_step', {'next_obs': serialize_obs_map(reset), 'auto_reset': True})
        self.assertEqual(emitted[-1]['payload']['new_message_start_by_agent']['7'],
                         len(emitted[-1]['payload']['next_obs']['7']['messages']) - 1)
        self.assertIn('new attempt', json.dumps(emitted[-1]['payload']['next_obs']))

    def test_real_rebuilt_full_attempt_reset_only_shows_available_new_tail(self):
        for history_enabled, include_terminal, include_history, assistant, expected_rows, expected_images in [
            (True, True, True, True, 5, 3),
            (True, False, True, True, 3, 2),
            (True, True, False, True, 3, 2),
            (True, True, True, False, 4, 3),
            (False, True, True, True, 3, 2),
        ]:
            with self.subTest(history=history_enabled, terminal=include_terminal, include=include_history, assistant=assistant):
                events, context, env, cfg = self.rebuilt_full_events(history_enabled=history_enabled, step_count=0)
                cfg['env_wrapper_cfg']['context_manager']['history']['include_terminal'] = include_terminal
                cfg['env_wrapper_cfg']['context_manager']['messages']['include_history'] = include_history
                context.configure(cfg)
                terminal_obs = {7: {'step_msg': 'terminal', 'vis': [[Image.new('RGB', (1, 1), 'green')]]}}
                actions = {7: {'action': 'end' if assistant else None, 'assistant': 'end' if assistant else None}}
                context.record_transition(terminal_obs, actions, {7: 1}, {7: True})
                context.finalize_attempt([7])
                history = context.take_finalized_attempts()
                new_obs = {7: {'step_msg': 'new attempt', 'vis': [[Image.new('RGB', (1, 1), 'blue')]]}}
                obs = context.on_attempt_reset(cfg, new_obs, history_snapshot=history,
                                               current_attempt_index=1, next_attempt_index=2)
                env._emit_env_event('env_step', {'next_obs': serialize_obs_map(obs), 'auto_reset': True})
                result = self.render(env.hooks._hooks[0].events)
                self.assertEqual(len(self.bubbles(result)), expected_rows)
                self.assertEqual(sum(row.count('<img ') for row in self.bubbles(result)), expected_images)

    def test_attempt_reset_with_array_observation_fields_keeps_terminal_feedback(self):
        events, context, env, cfg = self.rebuilt_full_events(step_count=0)
        # Extra observation fields are not part of the model message surface.
        context._last_obs_by_uid[7]['vector'] = np.array([1, 2])
        terminal = {7: {'step_msg': 'terminal', 'vis': [[Image.new('RGB', (1, 1), 'green')]],
                        'vector': np.array([3, 4])}}
        context.record_transition(terminal, {7: {'action': 'end', 'assistant': 'end'}}, {7: 1}, {7: True})
        context.finalize_attempt([7])
        history = context.take_finalized_attempts()
        obs = context.on_attempt_reset(cfg, {7: {'step_msg': 'new attempt', 'vis': [[Image.new('RGB', (1, 1), 'blue')]]}},
                                       history_snapshot=history, current_attempt_index=1, next_attempt_index=2)
        env._emit_env_event('env_step', {'next_obs': serialize_obs_map(obs), 'auto_reset': True})
        result = self.render(env.hooks._hooks[0].events)
        self.assertEqual(len(self.bubbles(result)), 5)
        self.assertEqual(sum(row.count('<img ') for row in self.bubbles(result)), 3)
        self.assertTrue(any('terminal' in row for row in self.bubbles(result)))

    def test_rebuilt_full_tail_metadata_and_id_are_shared_by_request_mirror(self):
        events, _, _, _ = self.rebuilt_full_events(history_limit=1)
        original = events[1]
        mirrored = deepcopy(original)
        mirrored['seq'] = 4
        mirrored['event'] = 'agent_request'
        mirrored['source'] = 'run_api_agent'
        mirrored['payload']['obs'] = mirrored['payload'].pop('next_obs')
        result = self.render(events[:2] + [mirrored])
        self.assertEqual(len(self.bubbles(result)), 4)

    def test_delta_repeated_text_and_two_camera_frames_are_new_each_step(self):
        same = [message('unchanged', images=('camA', 'camB'))]
        result = self.render([
            observation(1, same, mode='delta', observation_id='reset', event='env_reset'),
            observation(2, same, mode='delta', observation_id='step1'),
            observation(3, [message('updated', images=('camA2', 'camB2'))], mode='delta', observation_id='step2'),
        ])
        self.assertEqual(len(self.bubbles(result)), 3)
        self.assertEqual(sum(row.count('<img ') for row in self.bubbles(result)), 6)
        self.assertIn('camA2', self.bubbles(result)[-1])
        self.assertIn('camB2', self.bubbles(result)[-1])

    def test_delta_env_runtime_and_request_mirrors_share_observation(self):
        events = []
        for index, event in enumerate(('env_reset', 'env_reset', 'agent_request'), 1):
            item = observation(index, [message('frame', images=('same',))], event=event,
                               mode='delta', observation_id='one')
            item['source'] = ('ArkEnv', 'EnvRuntime', 'run_api_agent')[index - 1]
            events.append(item)
        for index, event in enumerate(('env_step', 'env_step', 'agent_request'), 4):
            events.append(observation(index, [message('frame', images=('same',))], event=event,
                                      mode='delta', observation_id='two'))
        result = self.render(events)
        self.assertEqual(len(self.bubbles(result)), 2)
        self.assertEqual(result['raw_count'], '6 raw events stored')

    def test_legacy_full_accumulation_and_duplicate_snapshots(self):
        a, b, c = message('A'), message('B'), message('C')
        result = self.render([
            observation(1, [a], event='env_reset'),
            observation(2, [a], event='agent_request'),
            observation(3, [a, b]), observation(4, [a, b]),
            observation(5, [a, b, c]),
        ])
        self.assertEqual(len(self.bubbles(result)), 3)

    def test_legacy_cross_source_reset_mirror_and_later_true_reset(self):
        first = observation(1, [message('same')], event='env_reset')
        first['source'] = 'ArkEnv'
        mirror = observation(2, [message('same')], event='env_reset')
        mirror['source'] = 'EnvRuntime'
        later = observation(4, [message('same')], event='env_reset')
        later['source'] = 'ArkEnv'
        result = self.render([first, mirror, observation(3, [message('same')], event='agent_request'), later])
        self.assertEqual(len(self.bubbles(result)), 2)

    def test_identical_true_resets_are_not_legacy_mirrors(self):
        cases = [
            ('ArkEnv', 'ArkEnv', None, None),
            ('custom1', 'custom2', None, None),
            ('ArkEnv', 'EnvRuntime', 'reset-one', 'reset-two'),
        ]
        for first_source, second_source, first_id, second_id in cases:
            with self.subTest(sources=(first_source, second_source), ids=(first_id, second_id)):
                first = observation(1, [message('same')], event='env_reset', observation_id=first_id)
                first['source'] = first_source
                second = observation(2, [message('same')], event='env_reset', observation_id=second_id)
                second['source'] = second_source
                self.assertEqual(len(self.bubbles(self.render([first, second]))), 2)

    def test_full_stable_prompt_prefix_sliding_equal_and_shorter_windows(self):
        system, developer = message('rules', 'system'), message('tools', 'developer')
        a, b, c, d, e = [message(text) for text in 'ABCDE']
        result = self.render([
            observation(1, [system, developer, a, b], mode='full'),
            observation(2, [system, developer, b, c], mode='full'),
            observation(3, [system, developer, c], mode='full'),
            observation(4, [system, developer, d], mode='full'),
            observation(5, [system, developer, e], mode='full'),
        ])
        self.assertEqual(len(self.bubbles(result)), 7)
        self.assertEqual(sum('>rules<' in row for row in self.bubbles(result)), 1)
        self.assertIn('>E<', self.bubbles(result)[-1])

    def test_reset_and_case_model_run_boundaries_restart_history(self):
        same = [message('same')]
        result = self.render([
            {'seq': 1, 'event': 'run_start', 'payload': {}},
            observation(2, same, event='env_reset', observation_id='r1'),
            observation(3, same, event='env_reset', observation_id='r2'),
            {'seq': 4, 'event': 'case_start', 'payload': {}},
            observation(5, same),
            {'seq': 6, 'event': 'model_start', 'payload': {}},
            observation(7, same),
        ])
        self.assertEqual(len(self.bubbles(result)), 4)
        restarted = self.render([
            observation(1, same),
            {'seq': 2, 'event': 'run_start', 'payload': {}},
            observation(3, same),
        ])
        self.assertEqual(len(self.bubbles(restarted)), 1)

    def test_attempt_transition_keeps_delta_observation_and_full_overlap(self):
        result = self.render([
            observation(1, [message('attempt1')], mode='delta', observation_id='a1'),
            observation(2, [message('attempt2')], mode='delta', observation_id='a2', auto_reset=True),
            observation(3, [message('attempt2')], mode='delta', observation_id='a2', event='agent_request'),
        ])
        self.assertEqual(len(self.bubbles(result)), 2)
        full = self.render([
            observation(1, [message('attempt1')], mode='full'),
            observation(2, [message('attempt1'), message('attempt2')], mode='full', auto_reset=True),
        ])
        self.assertEqual(len(self.bubbles(full)), 2)

    def test_multi_agent_responses_and_echoes_only_cancel_once_per_agent(self):
        response = {'seq': 2, 'event': 'agent_response', 'payload': {'actions': {
            '7': {'assistant': 'repeat'}, '8': {'assistant': 'repeat'},
        }}}
        first = observation(1, [message('start')], mode='delta', observation_id='start')
        first['payload']['next_obs']['8'] = {'messages': [message('start')]}
        echo = observation(3, [message('repeat', 'assistant'), message('repeat', 'assistant')],
                           mode='delta', observation_id='echo')
        echo['payload']['next_obs']['8'] = {'messages': [message('repeat', 'assistant')]}
        result = self.render([first, response, echo,
                              observation(4, [message('repeat', 'assistant')], mode='delta', observation_id='later')])
        self.assertEqual(len(self.bubbles(result, 'assistant')), 4)
        self.assertTrue(any('8' in row for row in self.bubbles(result, 'assistant')))

    def test_full_echo_does_not_consume_old_matching_assistant(self):
        a, old, new = message('A'), message('repeat', 'assistant'), message('B')
        result = self.render([
            observation(1, [a, old], mode='full'),
            {'seq': 2, 'event': 'agent_response', 'payload': {'actions': {'7': {'assistant': 'repeat'}}}},
            observation(3, [a, old, new, old], mode='full'),
            observation(4, [a, old, new, old, old], mode='full'),
        ])
        self.assertEqual(len(self.bubbles(result, 'assistant')), 3)

    def test_assistant_text_part_array_echo_is_cancelled_once(self):
        echo = {'role': 'assistant', 'content': [{'type': 'text', 'text': 'same'}]}
        result = self.render([
            {'seq': 1, 'event': 'agent_response', 'payload': {'actions': {'7': {'assistant': 'same'}}}},
            observation(2, [echo, echo], mode='delta', observation_id='one'),
        ])
        self.assertEqual(len(self.bubbles(result, 'assistant')), 2)

    def test_unreturned_echo_expires_at_next_observation_and_boundary(self):
        response = {'seq': 1, 'event': 'agent_response', 'payload': {'actions': {'7': {'assistant': 'repeat'}}}}
        result = self.render([
            response, observation(2, [message('no echo')], mode='delta', observation_id='one'),
            observation(3, [message('repeat', 'assistant')], mode='delta', observation_id='two'),
            {'seq': 4, 'event': 'agent_response', 'payload': response['payload']},
            {'seq': 5, 'event': 'case_start', 'payload': {}},
            observation(6, [message('repeat', 'assistant')]),
        ])
        self.assertEqual(len(self.bubbles(result, 'assistant')), 4)

    def test_replayed_seq_including_run_start_preserves_loaded_view(self):
        events = [{'seq': 1, 'event': 'run_start', 'payload': {}}, observation(2, [message('A')])]
        result = self.render(events, state_events=events)
        self.assertEqual(len(self.bubbles(result)), 1)
        self.assertEqual(result['raw_count'], '2 raw events stored')
        self.assertTrue(result['connectedAfterState'])

    def test_changed_stream_sequence_and_new_run_reset_view(self):
        result = self.render([
            observation(8, [message('old')]), observation(3, [message('new')]),
        ])
        self.assertEqual(len(self.bubbles(result)), 1)
        self.assertIn('>new<', self.bubbles(result)[0])
        # A genuinely new run may reuse sequence numbers, with a fresh timestamp.
        result = self.render([
            {'seq': 1, 'time': 10, 'event': 'run_start', 'payload': {}},
            observation(2, [message('old')]),
            {'seq': 1, 'time': 11, 'event': 'run_start', 'payload': {}},
            observation(2, [message('new')]),
        ])
        self.assertEqual(len(self.bubbles(result)), 1)
        self.assertIn('>new<', self.bubbles(result)[0])

    def test_raw_fallback_observation_mirrors_and_repeated_delta_frames(self):
        events = []
        for seq, event, identity in [(1, 'env_reset', 'a'), (2, 'agent_request', 'a'), (3, 'env_step', 'b')]:
            key = 'next_obs' if event == 'env_step' else 'obs'
            events.append({'seq': seq, 'event': event, 'payload': {
                'message_mode': 'delta', 'observation_id': identity,
                key: {'7': {'step_msg': 'raw', 'images': [{'url': 'data:image/png;base64,raw'}]}},
            }})
        self.assertEqual(len(self.bubbles(self.render(events))), 2)

    def test_legacy_raw_fallback_repeats_each_step_but_not_request_mirror(self):
        obs = {'7': {'step_msg': 'unchanged'}}
        events = [
            {'seq': 1, 'event': 'env_reset', 'payload': {'obs': obs}},
            {'seq': 2, 'event': 'agent_request', 'payload': {'obs': obs}},
            {'seq': 3, 'event': 'env_step', 'payload': {'next_obs': obs}},
            {'seq': 4, 'event': 'agent_request', 'payload': {'obs': obs}},
        ]
        self.assertEqual(len(self.bubbles(self.render(events))), 2)

    @staticmethod
    def traced_response(seq, calls=None, *, trace=None, assets=None, turn=0):
        saved = dict(trace or {})
        if calls is not None:
            saved['model_requests'] = calls
        payload = {'turn_index': turn, 'actions': {'7': {'action': 'selected', 'assistant': 'selected'}},
                   'raw_trace_by_agent': {'7': saved}}
        if assets is not None:
            payload['model_trace_images'] = assets
        return {'seq': seq, 'event': 'agent_response', 'payload': payload}

    def test_model_calls_preserve_order_context_and_full_tool_exchange(self):
        calls = [
            {'purpose': 'summary', 'status': 'completed',
             'request_messages': [{'role': 'system', 'content': 'Summarize old visible records'},
                                  {'role': 'user', 'content': 'Prior observations'}],
             'completion': 'Summary memory', 'tools': [], 'usage': {'total_tokens': 18}},
            {'purpose': 'action', 'status': 'completed',
             'request_messages': [{'role': 'system', 'content': 'Harness prompt'},
                                  {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-1', 'function': {'name': 'read_file'}}]},
                                  {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'earlier note'}],
             'tools': [{'type': 'function', 'function': {'name': 'submit_action'}}],
             'parameters': {'tool_choice': 'required'}, 'completion': '',
             'tool_calls': [{'call_id': 'call-2', 'name': 'submit_action', 'arguments_json': '{}'}],
             'response_metadata': {'reasoning': 'public provider summary', 'encrypted_content': 'opaque-ciphertext'},
             'native_assistant': {'message': {'role': 'assistant', 'content': None}},
             'usage': {'total_tokens': 42},
             'request_transport': {'transport': 'codex_sdk_structured_json', 'input_text': 'SDK outer wrapper'}},
        ]
        events = [{'seq': 1, 'event': 'case_start', 'payload': {'case_id': 'case-A'}},
                  {'seq': 2, 'event': 'model_start', 'payload': {'model_name': 'simple-model'}},
                  self.traced_response(3, calls, trace={'agent_events': [{'kind': 'tool_exchange', 'result': 'note result'}],
                                                        'agent_usage': {'summary_calls': 1}}),
                  self.traced_response(4, [calls[1]], trace={'replay': True}, turn=1)]
        result = self.render(events)
        self.assertEqual(len(result['model_calls']), 2)
        first, second = [row['html'] for row in result['model_calls']]
        self.assertIn('case-A · simple-model · 7 · turn 1 · 2 model calls', first)
        self.assertLess(first.index('Call 1 · summary'), first.index('Call 2 · action'))
        for expected in ('Harness prompt', 'Summary memory', 'earlier note', 'read_file', 'tool_call_id',
                         'submit_action', 'required', '(empty assistant content)', 'opaque-ciphertext',
                         'encrypted fields are opaque', 'note result', 'SDK outer wrapper', 'Decision usage'):
            self.assertIn(expected, first)
        self.assertIn('turn 2 · 1 model call · recorded replay', second)
        self.assertEqual(len(self.bubbles(result, 'assistant')), 2)

    def test_legacy_model_trace_is_unavailable_without_invented_request(self):
        result = self.render([self.traced_response(1, trace={'assistant_raw': 'legacy combined response'})])
        self.assertIn('model calls unavailable', result['model_calls'][0]['html'])
        self.assertIn('did not save actual model calls', result['model_calls'][0]['html'])
        self.assertNotIn('Actual request messages', result['model_calls'][0]['html'])
        self.assertEqual(len(self.bubbles(result, 'assistant')), 1)
        self.assertIn('before compaction', _VIEWER_HTML)

    def test_model_call_identity_uses_each_payload_without_boundary_events(self):
        events = []
        for index, (case_id, model_name) in enumerate((('seed-0001', 'model-A'), ('seed-0002', 'model-B')), 1):
            event = self.traced_response(index, [{'purpose': 'action', 'completion': 'selected'}])
            event['payload'].update(case_id=case_id, model_name=model_name)
            events.append(event)
        for live in (True, False):
            with self.subTest(live=live):
                result = self.render(events if live else [], state_events=[] if live else events)
                first, second = [row['html'] for row in result['model_calls']]
                self.assertIn('seed-0001 · model-A · 7 · turn 1', first)
                self.assertIn('seed-0002 · model-B · 7 · turn 1', second)
                self.assertNotIn('seed-0001', second)

    def test_model_call_payload_identity_overrides_stale_boundary_context(self):
        response = self.traced_response(3, [{'purpose': 'action'}])
        response['payload'].update(case_id='current-case', model_name='current-model')
        result = self.render([
            {'seq': 1, 'event': 'case_start', 'payload': {'case_id': 'old-case'}},
            {'seq': 2, 'event': 'model_start', 'payload': {'model_name': 'old-model'}}, response])
        html = result['model_calls'][0]['html']
        self.assertIn('current-case · current-model', html)
        self.assertNotIn('old-case', html)
        self.assertNotIn('old-model', html)

    def test_skipped_inference_is_distinct_from_missing_historical_trace(self):
        result = self.render([self.traced_response(1, trace={'skipped': True})])
        html = result['model_calls'][0]['html']
        self.assertIn('inference skipped', html)
        self.assertIn('no model request was made', html)
        self.assertNotIn('legacy record', html)

    def test_sdk_outer_inputs_and_schema_are_separate_from_portable_messages(self):
        digest = 'd' * 64
        result = self.render([self.traced_response(1, [{
            'purpose': 'action', 'request_messages': [{'role': 'user', 'content': 'portable task input'}],
            'request_transport': {'boundary': 'sdk_turn_input', 'inputs': [
                {'type': 'text', 'text': 'Actual SDK outer wrapper'},
                {'type': 'image', 'url': f'agentark-trace-image:{digest}'}],
                'output_schema': {'type': 'object'}, 'sdk_added_context': 'not_exposed_by_sdk'},
        }], assets={digest: {'data_url': 'data:image/png;base64,YWJj'}})])
        html = result['model_calls'][0]['html']
        self.assertIn('portable task input', html)
        self.assertIn('Actual SDK outer wrapper', html)
        self.assertIn('SDK input 2', html)
        self.assertEqual(html.count('<img '), 1)
        self.assertIn('SDK-added context may be unavailable', html)
        self.assertIn('output_schema', html)

    def test_large_call_bodies_are_lazy_and_old_decisions_have_explicit_view_limit(self):
        call = {'purpose': 'action', 'request_messages': [{'role': 'system', 'content': 'Large hidden-until-open prompt'}]}
        lazy = self.render([self.traced_response(1, [call])], expand_model_calls=False)
        self.assertIn('1 model call', lazy['model_calls'][0]['html'])
        self.assertNotIn('Large hidden-until-open prompt', lazy['model_calls'][0]['html'])
        opened = self.render([self.traced_response(1, [call])])
        self.assertIn('Large hidden-until-open prompt', opened['model_calls'][0]['html'])
        bounded = self.render([self.traced_response(i + 1, [call], turn=i) for i in range(202)],
                              expand_model_calls=False)
        self.assertEqual(len(bounded['model_calls']), 200)
        self.assertIn('turn 3', bounded['model_calls'][0]['html'])
        self.assertIn('Older calls are outside this view', bounded['model_call_limit'])
        self.assertEqual(bounded['raw_count'], '202 raw events stored')

    def test_model_trace_text_and_image_attributes_cannot_inject_html(self):
        attack = '<img src=x onerror="attack()"><script>bad()</script>'
        result = self.render([self.traced_response(1, [{
            'purpose': attack, 'status': 'completed', 'request_messages': [
                {'role': attack, 'content': [{'type': 'text', 'text': attack},
                                           {'type': 'image_url', 'image_url': {'url': 'x" onerror="attack()'}}]}],
            'completion': attack, 'tools': [{'description': attack}], 'parameters': {'key': attack},
        }])])
        html = result['model_calls'][0]['html']
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img ', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('Request image unavailable', html)

    def test_model_trace_images_resolve_only_matching_assets_across_decisions(self):
        digest, missing = 'a' * 64, 'b' * 64
        data_url = 'data:image/png;base64,YWJj'
        call = {'purpose': 'action', 'request_messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'agentark-trace-image:{digest}'}},
            {'type': 'image_url', 'url': f'agentark-trace-image:{missing}'},
        ]}]}
        assets = {digest: {'mime_type': 'image/png', 'data_url': data_url}}
        result = self.render([self.traced_response(1, [call], assets=assets), self.traced_response(2, [call])])
        for card in result['model_calls']:
            self.assertEqual(card['html'].count('<img '), 1)
            self.assertIn(data_url, card['html'])
            self.assertIn(f'Request image unavailable · SHA-256 {missing}', card['html'])
        cleared = self.render([self.traced_response(1, [call], assets=assets), self.traced_response(2, [call])], clear_at=[1])
        self.assertEqual(len(cleared['model_calls']), 1)
        self.assertEqual(cleared['model_calls'][0]['html'].count('<img '), 1)

    def test_late_viewer_state_assets_survive_bootstrap_run_start(self):
        digest = 'c' * 64
        call = {'request_messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'agentark-trace-image:{digest}'}}]}]}
        events = [{'seq': 1, 'event': 'run_start', 'payload': {}}, self.traced_response(8, [call])]
        result = self.render(state_events=events, state_images={digest: {'data_url': 'data:image/png;base64,YWJj'}})
        self.assertEqual(result['model_calls'][0]['html'].count('<img '), 1)
        new_run = self.render(events=[{'seq': 1, 'time': 2, 'event': 'run_start', 'payload': {}}, self.traced_response(2, [call])],
                              state_events=events, state_images={digest: {'data_url': 'data:image/png;base64,YWJj'}})
        self.assertEqual(new_run['model_calls'][0]['html'].count('<img '), 0)
        # Replaying buffered events from a previous run must not resurrect its
        # asset payload after the current run_start during state bootstrap.
        old_events = [self.traced_response(1, [call], assets={digest: {'data_url': 'data:image/png;base64,YWJj'}}),
                      {'seq': 2, 'event': 'run_start', 'payload': {}}, self.traced_response(3, [call])]
        self.assertEqual(self.render(state_events=old_events)['model_calls'][0]['html'].count('<img '), 0)


class LocalViewerTraceAssetTest(unittest.TestCase):
    @staticmethod
    def scoped_response(seq, call, digest=None):
        event = LocalViewerBehaviorTest.traced_response(seq, [call], assets=(
            {digest: {'mime_type': 'image/png', 'data_url': 'data:image/png;base64,YWJj'}} if digest else None))
        event['payload'].update(case_id='case-A', model_name='model-A')
        return event

    @staticmethod
    def request(seq, case_id='case-A', model_name='model-A'):
        return {'seq': seq, 'event': 'agent_request', 'payload': {'case_id': case_id, 'model_name': model_name}}

    def test_buffer_eviction_keeps_assets_for_later_old_image_reuse(self):
        viewer = LocalViewerHook(event_buffer_size=1)
        digest = 'a' * 64
        asset = {'mime_type': 'image/png', 'data_url': 'data:image/png;base64,YWJj'}
        call = {'request_messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'agentark-trace-image:{digest}'}}]}]}
        viewer.handle_event(LocalViewerBehaviorTest.traced_response(1, [call], assets={digest: asset}))
        viewer.handle_event(LocalViewerBehaviorTest.traced_response(2, [call]))
        state = viewer._snapshot_state()
        self.assertEqual([ev['seq'] for ev in state['events']], [2])
        self.assertEqual(state['model_trace_images'], {digest: asset})
        viewer.handle_event(LocalViewerBehaviorTest.traced_response(3, []))
        self.assertEqual(viewer._snapshot_state()['model_trace_images'], {digest: asset})
        # No second asset payload is emitted when a later tool call reopens it.
        viewer.handle_event(LocalViewerBehaviorTest.traced_response(4, [call]))
        self.assertEqual(viewer._snapshot_state()['model_trace_images'], {digest: asset})

    def test_new_run_clears_asset_cache(self):
        viewer = LocalViewerHook()
        viewer.handle_event(LocalViewerBehaviorTest.traced_response(1, [], assets={'a' * 64: {'data_url': 'old'}}))
        viewer.handle_event({'seq': 2, 'event': 'run_start', 'payload': {}})
        self.assertEqual(viewer._snapshot_state()['model_trace_images'], {})

    def test_active_case_can_reread_after_eviction_then_transition_expires_assets(self):
        digest = 'b' * 64
        call = {'request_messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'agentark-trace-image:{digest}'}}]}]}
        for boundary in (self.request(5, case_id='case-B'), self.request(5, model_name='model-B'),
                         {'seq': 5, 'event': 'case_start', 'payload': {'case_id': 'case-B'}}):
            with self.subTest(boundary=boundary):
                viewer = LocalViewerHook(event_buffer_size=1)
                viewer.handle_event(self.request(1))
                viewer.handle_event(self.scoped_response(2, call, digest))
                viewer.handle_event({'seq': 3, 'event': 'env_step', 'payload': {}})
                self.assertIn(digest, viewer._snapshot_state()['model_trace_images'])
                viewer.handle_event(self.scoped_response(4, call))
                self.assertIn(digest, viewer._snapshot_state()['model_trace_images'])
                viewer.handle_event(boundary)
                self.assertEqual(viewer._snapshot_state()['model_trace_images'], {})

    def test_previous_case_buffered_reference_retains_asset_until_evicted(self):
        digest = 'c' * 64
        call = {'request_messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'agentark-trace-image:{digest}'}}]}]}
        viewer = LocalViewerHook(event_buffer_size=2)
        viewer.handle_event(self.request(1))
        viewer.handle_event(self.scoped_response(2, call, digest))
        viewer.handle_event(self.request(3, case_id='case-B'))
        self.assertIn(digest, viewer._snapshot_state()['model_trace_images'])
        viewer.handle_event({'seq': 4, 'event': 'env_reset', 'payload': {}})
        self.assertEqual(viewer._snapshot_state()['model_trace_images'], {})


if __name__ == '__main__':
    unittest.main()
