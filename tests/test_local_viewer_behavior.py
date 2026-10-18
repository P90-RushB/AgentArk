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
from agent_ark.interaction.local_viewer import _VIEWER_HTML
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
    def render(self, events=(), state_events=(), snapshots=False, clear_at=()):
        result = subprocess.run(
            [NODE, str(HARNESS)],
            input=json.dumps({'html': _VIEWER_HTML, 'events': events, 'state_events': state_events,
                              'snapshots': snapshots, 'clear_at': clear_at}),
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


if __name__ == '__main__':
    unittest.main()
