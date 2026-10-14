"""Opt-in packaged-Player compatibility check; uses no model/provider.

All machine paths are CLI arguments. Generated reports and pixels belong outside
the source tree. Requires the named legacy task bundles in the supplied Mods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def action(name, **arguments):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('env-path', 'mod-path', 'pool-root', 'source-root', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--base-port', type=int, default=5207)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import yaml
    from PIL import Image
    from agent_ark.ark_env.ark_env import ArkEnv

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    supplied_mods = args.mod_path.resolve()
    supplied_backup = digest((supplied_mods / 'config.yaml.bak').read_bytes())
    cfg = {
        'env_path': str(args.env_path.resolve()), 'mod_path': str(supplied_mods),
        'task_type': 'RLTask', 'base_port': args.base_port, 'auto_port_scan': True,
        'worker_index': 0,
        'env_config_overrides': {
            'override_by_task': True, 'num_parallel_envs': 1,
            'unity_start_timeout_s': 180, 'no_graphics': False,
        },
        'runtime_sandbox': {
            'enabled': True, 'runtime_platform': 'windows',
            'template_root': str(args.env_path.resolve().parent),
            'shared_task_store_path': str(supplied_mods / 'all_tasks'),
            'pool_root': str(args.pool_root.resolve()), 'pool_size': 1,
            'auto_prepare': True, 'link_mode': 'copy', 'log_resolution': True,
        },
    }
    env = ArkEnv(cfg)
    mods = Path(env.sub_env.env_info_mgr.mod_path).resolve()
    require(mods != supplied_mods, 'A copied runtime is required.')
    backup_hash = digest((mods / 'config.yaml.bak').read_bytes())
    require(backup_hash == supplied_backup, 'Copied baseline differs from supplied baseline.')
    report = {'cases': {}, 'probe': {}, 'baseline_sha256': backup_hash}
    changed_cfg = None
    original_cfg = None

    def config_check(task):
        effective_yaml = yaml.safe_load((mods / 'config.yaml').read_text(encoding='utf-8-sig'))
        effective_json = json.loads((mods / 'config.json').read_text(encoding='utf-8-sig'))
        require(effective_yaml == effective_json == env.sub_env.env_info_mgr.env_config,
                'Effective YAML/JSON/runtime config differ.')
        require(digest((mods / 'config.yaml.bak').read_bytes()) == backup_hash, 'Baseline changed.')
        require(effective_json['task_name'] == task, 'Wrong task selected.')
        require(effective_json['override_by_task'] is True, 'Wrong loading mode.')
        require(effective_json['load_mod_mode'] == 'task_name', 'Wrong mod mode.')
        require(effective_json.get('task_params', {}) == env.sub_env.env_info_mgr.task_config.get('task_params', {}),
                'Task params carried across tasks.')
        return effective_json

    def summarize(obs, label):
        rows = {}
        folder = output / 'frames' / label
        folder.mkdir(parents=True, exist_ok=True)
        for uid, item in obs.items():
            pictures = []
            for camera, frames in enumerate(item.get('vis', [])):
                frames = frames if isinstance(frames, (list, tuple)) else [frames]
                for number, image in enumerate(frames):
                    require(isinstance(image, Image.Image), 'Expected real decoded Player image.')
                    image.save(folder / f'{uid}_camera{camera}_{number}.png')
                    pictures.append({'camera': camera, 'frame': number, 'size': list(image.size),
                                     'mode': image.mode, 'sha256': digest(image.tobytes())})
            extras = item.get('attempt_initial_images', [])
            rows[str(uid)] = {
                'step_msg': item.get('step_msg', ''), 'skip_infer': bool(item.get('skip_infer')),
                'images': pictures,
                'messages_sha256': digest(json.dumps(item.get('messages', []), sort_keys=True,
                                                     ensure_ascii=False).encode()),
                'extra_images': [{'index': x['index'], 'sha256': x['sha256'],
                                  'size': list(x['image'].size)} for x in extras],
            }
        return rows

    try:
        profiles = [
            ('ConnectionCounterplay2D', 17,
             [action('Place', row=r, col=c) for r, c in [(2, 7), (1, 7), (3, 6), (7, 4)]], 1.0, False),
            ('LocalSeatShortageCertificate2D', 2,
             [action('Focus', id=f'A{i:02}') for i in range(1, 7)]
             + [action('MarkAnimal', id=x, selected=True) for x in ['A01', 'A03', 'A04', 'A05']]
             + [action('MarkSeat', id=x, selected=True) for x in ['S03', 'S04', 'S05']]
             + [action('Submit')], 1.0, False),
            ('OneItemRewind2D', 17, [action('GiveUp')], 0.0, True),
        ]
        for task, seed, actions, expected_reward, expected_truncated in profiles:
            obs, _ = env.reset(task_name=task, group_seed=seed, env_id=0, max_attempts=1)
            config_check(task)
            case = {'seed': seed, 'observations': [summarize(obs, task + '/reset')], 'steps': []}
            report['cases'][task] = case
            total = 0.0
            for i, payload in enumerate(actions):
                obs, reward, done, info = env.step({uid: payload for uid in obs})
                require(not info.get('sub_env', {}).get('step', {}).get('func_render_errors'), 'Action rejected.')
                case['observations'].append(summarize(obs, f'{task}/step{i + 1:02}'))
                case['steps'].append({'action': payload, 'reward': {str(k): float(v) for k, v in reward.items()},
                                      'done': {str(k): bool(v) for k, v in done.items()},
                                      'truncated': info.get('rollout', {}).get('truncated')})
                total += sum(float(v) for v in reward.values())
            require(done.get('__all__') is True, f'{task}: did not terminate.')
            require(abs(total - expected_reward) < 1e-6, f'{task}: unexpected reward {total}.')
            require(bool(info.get('rollout', {}).get('truncated')) == expected_truncated,
                    f'{task}: unexpected truncation semantics.')
            require(all(not row['extra_images'] for frame in case['observations'] for row in frame.values()),
                    'Disabled feature changed old task observations.')
            case['total_reward'] = total
            print(f'PASS old task {task}: {len(actions)} actions, reward={total}', flush=True)

        if args.probe:
            changed_cfg = mods / 'all_tasks/EchoGateRelay2D/cfg/task_config.yaml'
            original_cfg = changed_cfg.read_bytes()
            task_cfg = yaml.safe_load(original_cfg)
            fixtures = mods / 'additional_inputs'
            fixtures.mkdir(exist_ok=True)
            expected = []
            for i, size in enumerate([(17, 13), (9, 21)], 1):
                image = Image.new('RGB', size, (i * 71, i * 37, i * 19))
                path = fixtures / f'image{i}.png'
                image.save(path)
                expected.append({'index': i, 'sha256': digest(path.read_bytes()), 'size': list(size)})
            wrapper = task_cfg.setdefault('env_wrapper_cfg', {})
            wrapper['attempt_initial_images'] = ['additional_inputs/image1.png', 'additional_inputs/image2.png']
            wrapper.setdefault('context_manager', {}).setdefault('messages', {}).update(
                append_only=True, return_mode='delta', only_return_messages=False)
            changed_cfg.write_text(yaml.safe_dump(task_cfg, sort_keys=False), encoding='utf-8')
            obs, _ = env.reset(task_name='EchoGateRelay2D', group_seed=17, env_id=0, max_attempts=3)
            config_check('EchoGateRelay2D')
            pid = env.sub_env.env._process.pid
            reset_extra = lambda current: [x for row in summarize(current, 'probe/check').values()
                                          for x in row['extra_images']]
            require(reset_extra(obs) == expected, 'Initial additional images/order/hash mismatch.')
            probe = {'expected_images': expected, 'attempts': []}
            report['probe'] = probe
            for attempt in range(1, 4):
                obs, _, _, _ = env.step({uid: action('Wait') for uid in obs})
                require(reset_extra(obs) == [], 'Ordinary step repeats additional images.')
                obs, _, done, info = env.step({uid: action('GiveUp') for uid in obs})
                probe['attempts'].append({'index': attempt, 'auto_reset': bool(info.get('attempt', {}).get('auto_reset'))})
                if attempt < 3:
                    require(reset_extra(obs) == expected, 'Same-seed automatic reset missed additional images.')
                else:
                    require(done.get('__all__'), 'Final attempt did not terminate.')
                    require(reset_extra(obs) == [], 'Terminal observation repeats additional images.')
            # Remove source files before switching: any stale load would now fail.
            for path in fixtures.glob('image*.png'):
                path.unlink()
            obs, _ = env.reset(task_name='TornPictureAssembly2D', group_seed=17, env_id=0, max_attempts=1)
            effective = config_check('TornPictureAssembly2D')
            require(not effective.get('env_wrapper_cfg', {}).get('attempt_initial_images'), 'Task config leaked.')
            require(reset_extra(obs) == [], 'Task image state leaked.')
            require(env.sub_env.env._process.pid == pid, 'Expected same Player across compatible task switch.')
            obs, _, _, _ = env.step({uid: action('Inspect', pieceId='P01') for uid in obs})
            require(reset_extra(obs) == [], 'Next task action inherited images.')
            probe['switch_same_player'] = True
            probe['switch_images_absent'] = True
            print('PASS enabled 3-attempt images -> omitted next task, same Player', flush=True)
        report['passed'] = True
    finally:
        if changed_cfg is not None and original_cfg is not None:
            changed_cfg.write_bytes(original_cfg)
        env.close()
        require(digest((supplied_mods / 'config.yaml.bak').read_bytes()) == supplied_backup,
                'Original runtime backup changed.')
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Runtime compatibility validation complete.', flush=True)


if __name__ == '__main__':
    main()
