from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


RUNTIME_SANDBOX_LAYOUT_VERSION = 2
DEFAULT_RUNTIME_POOL_MANIFEST = 'runtime_pool_manifest.json'
SHARED_TASK_STORE_DIRNAME = 'all_tasks'

_ENSURE_LOCK = threading.Lock()
_ENSURE_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if lowered in ('0', 'false', 'no', 'n', 'off'):
            return False
    return bool(default)


def _normalize_runtime_platform(raw: Any) -> str:
    value = str(raw or '').strip().lower()
    if not value:
        return 'windows' if os.name == 'nt' else 'linux'
    if value in ('windows', 'win', 'nt'):
        return 'windows'
    if value in ('linux', 'lin'):
        return 'linux'
    raise ValueError(f"Unsupported runtime platform: {raw!r}")


def _worker_name(worker_index: int) -> str:
    return f'worker_{int(worker_index):03d}'


def _manifest_path(pool_root: Path, manifest_name: str) -> Path:
    return pool_root / str(manifest_name or DEFAULT_RUNTIME_POOL_MANIFEST)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return

    def _onerror(func, failed_path, exc_info):
        try:
            os.chmod(failed_path, stat.S_IWRITE)
        except Exception:
            pass
        func(failed_path)

    shutil.rmtree(path, onerror=_onerror)


def _resolve_existing_path(path_value: Any, runtime_platform: str) -> Path:
    candidate = _path_from_value(path_value)
    if candidate.exists():
        return candidate

    suffix_candidates = []
    if runtime_platform == 'windows' and candidate.suffix.lower() != '.exe':
        suffix_candidates.append(Path(str(candidate) + '.exe'))
    elif runtime_platform == 'linux' and not candidate.suffix:
        suffix_candidates.extend([
            Path(str(candidate) + '.x86_64'),
            Path(str(candidate) + '.x86'),
        ])

    for resolved in suffix_candidates:
        if resolved.exists():
            return resolved
    return candidate


def _executable_base_name(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in ('.exe', '.x86_64', '.x86'):
        return path.stem
    return path.name


def _path_from_value(path_value: Any) -> Path:
    expanded = str(path_value)
    for _ in range(3):
        next_value = os.path.expandvars(expanded)
        if next_value == expanded:
            break
        expanded = next_value
    expanded = os.path.expanduser(expanded)
    return Path(expanded)


def _looks_like_unity_executable(path: Path, runtime_platform: str) -> bool:
    if not path.is_file():
        return False
    name = path.name.lower()
    if runtime_platform == 'windows':
        return name.endswith('.exe')
    return name.endswith('.x86_64') or name.endswith('.x86')


def _discover_runtime_entrypoint(runtime_root: Path, runtime_platform: str, preferred_name: Optional[str] = None) -> Path:
    candidates = []
    for item in sorted(runtime_root.iterdir(), key=lambda p: p.name.lower()):
        if not _looks_like_unity_executable(item, runtime_platform):
            continue
        score = 0
        data_dir = runtime_root / f"{_executable_base_name(item)}_Data"
        if data_dir.is_dir():
            score += 10
        if preferred_name:
            preferred = str(preferred_name).strip().lower()
            if item.name.lower() == preferred:
                score += 5
            if item.stem.lower() == preferred:
                score += 5
            if _executable_base_name(item).lower() == preferred:
                score += 5
        candidates.append((score, item.name.lower(), item))

    if not candidates:
        raise FileNotFoundError(
            f"Could not discover Unity executable inside runtime root: {runtime_root}"
        )

    candidates.sort(reverse=True)
    return candidates[0][2]


def _derive_mods_path(runtime_root: Path, env_path: Path) -> Path:
    candidate = runtime_root / f"{_executable_base_name(env_path)}_Data" / 'Resources' / 'Mods'
    if candidate.is_dir():
        return candidate

    matches = []
    for item in sorted(runtime_root.iterdir(), key=lambda p: p.name.lower()):
        if not item.is_dir() or not item.name.endswith('_Data'):
            continue
        mods_path = item / 'Resources' / 'Mods'
        if mods_path.is_dir():
            matches.append(mods_path)

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not resolve Mods directory from runtime root={runtime_root}, env_path={env_path}"
    )


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        common = os.path.commonpath([str(path), str(root)])
    except ValueError:
        return False
    return common == str(root)


def _relative_to_root(path: Path, root: Path) -> str:
    if not _is_within_root(path, root):
        raise ValueError(f"Path {path} is not inside runtime root {root}")
    return os.path.relpath(str(path), str(root))


def _update_hash_for_tree(hasher: 'hashlib._Hash', root: Path, current: Path) -> None:
    rel_root = current.relative_to(root).as_posix() if current != root else '.'
    hasher.update(f"D {rel_root}\n".encode('utf-8'))

    entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
    for entry in entries:
        rel = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            hasher.update(f"L {rel} {os.readlink(entry)}\n".encode('utf-8', errors='ignore'))
            continue
        if entry.is_dir():
            _update_hash_for_tree(hasher, root, entry)
            continue
        st = entry.stat()
        hasher.update(f"F {rel} {st.st_size} {st.st_mtime_ns}\n".encode('utf-8'))


def compute_runtime_template_fingerprint(runtime_root: str | Path) -> str:
    root = Path(runtime_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Runtime root not found for fingerprinting: {runtime_root}")

    hasher = hashlib.sha256()
    _update_hash_for_tree(hasher, root, root)
    return hasher.hexdigest()


def _resolve_template_layout(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sandbox_cfg = dict(cfg.get('runtime_sandbox', {}) or {})
    runtime_platform = _normalize_runtime_platform(sandbox_cfg.get('runtime_platform', None))

    template_root_value = sandbox_cfg.get('template_root', None)
    template_env_value = sandbox_cfg.get('template_env_path', cfg.get('env_path', None))
    template_mod_value = sandbox_cfg.get('template_mod_path', cfg.get('mod_path', None))

    if template_root_value is not None:
        template_path = _resolve_existing_path(template_root_value, runtime_platform)
    elif template_env_value is not None:
        template_path = _resolve_existing_path(template_env_value, runtime_platform)
    else:
        raise ValueError('runtime_sandbox requires template_root or env_path/template_env_path')

    if not template_path.exists():
        raise FileNotFoundError(f"Runtime sandbox template path not found: {template_path}")

    if template_path.is_dir():
        runtime_root = template_path
        preferred_name = None
        if template_env_value is not None:
            preferred_candidate = _path_from_value(template_env_value)
            preferred_name = preferred_candidate.name if preferred_candidate.name else None
        env_path = _discover_runtime_entrypoint(runtime_root, runtime_platform, preferred_name=preferred_name)
    else:
        env_path = template_path
        runtime_root = template_path.parent

    if template_mod_value:
        mod_path = _path_from_value(template_mod_value)
        if not mod_path.exists():
            mod_path = _derive_mods_path(runtime_root, env_path)
    else:
        mod_path = _derive_mods_path(runtime_root, env_path)

    if not mod_path.exists() or not mod_path.is_dir():
        raise FileNotFoundError(f"Runtime sandbox Mods path not found: {mod_path}")

    env_relpath = _relative_to_root(env_path, runtime_root)
    mod_relpath = _relative_to_root(mod_path, runtime_root)

    pool_root_value = sandbox_cfg.get('pool_root', None)
    if pool_root_value in (None, ''):
        raise ValueError('runtime_sandbox.pool_root is required when runtime_sandbox.enabled=true')
    pool_root = _path_from_value(pool_root_value)

    if _is_within_root(pool_root, runtime_root):
        raise ValueError(
            f"runtime_sandbox.pool_root must not be nested inside the runtime template root: {pool_root}"
        )

    shared_task_store_value = sandbox_cfg.get('shared_task_store_path', None)
    if shared_task_store_value in (None, ''):
        raise ValueError(
            'runtime_sandbox.shared_task_store_path is required when runtime_sandbox.enabled=true'
        )
    shared_task_store_path = _path_from_value(shared_task_store_value)
    if not shared_task_store_path.exists() or not shared_task_store_path.is_dir():
        raise FileNotFoundError(
            f"Runtime sandbox shared task store not found or not a directory: {shared_task_store_path}"
        )

    return {
        'runtime_platform': runtime_platform,
        'runtime_root': runtime_root,
        'env_path': env_path,
        'mod_path': mod_path,
        'env_relpath': env_relpath,
        'mod_relpath': mod_relpath,
        'pool_root': pool_root,
        'shared_task_store_path': shared_task_store_path,
        'manifest_name': str(sandbox_cfg.get('manifest_name', DEFAULT_RUNTIME_POOL_MANIFEST) or DEFAULT_RUNTIME_POOL_MANIFEST),
        'link_mode': str(sandbox_cfg.get('link_mode', 'auto') or 'auto'),
    }


def _load_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    if not manifest_path.exists():
        return None
    with manifest_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    raise ValueError(f"Invalid runtime sandbox manifest: {manifest_path}")


def _write_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=False)


def _expected_worker_dirs(pool_root: Path, pool_size: int) -> list[str]:
    return [_worker_name(index) for index in range(int(pool_size))]


def _shared_task_store_link_path(mod_path: Path) -> Path:
    return mod_path / SHARED_TASK_STORE_DIRNAME


def _resolve_effective_link_mode(runtime_platform: str, requested_mode: str) -> str:
    mode = str(requested_mode or 'auto').strip().lower()
    if mode not in ('auto', 'symlink', 'junction', 'copy'):
        raise ValueError(f'Unsupported link_mode: {requested_mode!r}')
    if mode == 'auto':
        return 'junction' if runtime_platform == 'windows' else 'symlink'
    if runtime_platform != 'windows' and mode == 'junction':
        raise ValueError('link_mode=junction is only supported on Windows')
    return mode


def _is_directory_link(path: Path, runtime_platform: str) -> bool:
    if os.path.islink(path):
        return True
    if runtime_platform == 'windows' and path.exists() and path.is_dir():
        try:
            return path.resolve() != path.absolute()
        except Exception:
            return False
    return False


def _remove_existing_path(path: Path, runtime_platform: str) -> None:
    if not path.exists() and not os.path.islink(path):
        return
    if os.path.islink(path) or path.is_file():
        path.unlink()
        return
    if _is_directory_link(path, runtime_platform):
        os.rmdir(path)
        return
    if path.is_dir():
        _remove_tree(path)
        return
    path.unlink()


def _create_directory_junction(link_path: Path, target_path: Path) -> None:
    if os.name != 'nt':
        raise RuntimeError('Directory junctions are only supported on Windows')

    completed = subprocess.run(
        ['cmd', '/c', 'mklink', '/J', str(link_path), str(target_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stdout = (completed.stdout or '').strip()
        stderr = (completed.stderr or '').strip()
        raise RuntimeError(
            f'Failed to create directory junction: link={link_path}, target={target_path}, '
            f'code={completed.returncode}, stdout={stdout!r}, stderr={stderr!r}'
        )


def _shared_task_store_is_compatible(
    link_path: Path,
    shared_task_store_path: Path,
    *,
    runtime_platform: str,
    effective_link_mode: str,
) -> bool:
    if not link_path.exists():
        return False
    if effective_link_mode == 'copy':
        return link_path.is_dir()
    if effective_link_mode == 'symlink' and not os.path.islink(link_path):
        return False
    if effective_link_mode == 'junction':
        if runtime_platform != 'windows' or not link_path.is_dir() or os.path.islink(link_path):
            return False
    try:
        return os.path.samefile(link_path, shared_task_store_path)
    except Exception:
        try:
            return link_path.resolve() == shared_task_store_path.resolve()
        except Exception:
            return False


def _ensure_shared_task_store(
    mod_path: Path,
    shared_task_store_path: Path,
    *,
    runtime_platform: str,
    requested_link_mode: str,
) -> str:
    link_path = _shared_task_store_link_path(mod_path)
    effective_link_mode = _resolve_effective_link_mode(runtime_platform, requested_link_mode)

    if _shared_task_store_is_compatible(
        link_path,
        shared_task_store_path,
        runtime_platform=runtime_platform,
        effective_link_mode=effective_link_mode,
    ):
        return effective_link_mode

    _remove_existing_path(link_path, runtime_platform)

    if effective_link_mode == 'symlink':
        os.symlink(str(shared_task_store_path), str(link_path), target_is_directory=True)
    elif effective_link_mode == 'junction':
        _create_directory_junction(link_path, shared_task_store_path)
    elif effective_link_mode == 'copy':
        shutil.copytree(shared_task_store_path, link_path, symlinks=False)
    else:
        raise ValueError(f'Unsupported effective link mode: {effective_link_mode!r}')

    if not _shared_task_store_is_compatible(
        link_path,
        shared_task_store_path,
        runtime_platform=runtime_platform,
        effective_link_mode=effective_link_mode,
    ):
        raise RuntimeError(
            f'Failed to realize shared task store at {link_path} from {shared_task_store_path} '
            f'using mode={effective_link_mode}'
        )

    return effective_link_mode


def prepare_runtime_pool(
    *,
    runtime_platform: str,
    template_root: str,
    template_env_path: Optional[str],
    template_mod_path: Optional[str],
    shared_task_store_path: Optional[str],
    pool_root: str,
    pool_size: int,
    force_refresh: bool = False,
    clean_extra_workers: bool = False,
    manifest_name: str = DEFAULT_RUNTIME_POOL_MANIFEST,
    link_mode: str = 'auto',
) -> Dict[str, Any]:
    if int(pool_size) <= 0:
        raise ValueError('pool_size must be > 0')

    template_cfg: Dict[str, Any] = {
        'env_path': template_env_path,
        'mod_path': template_mod_path,
        'runtime_sandbox': {
            'enabled': True,
            'runtime_platform': runtime_platform,
            'template_root': template_root,
            'template_env_path': template_env_path,
            'template_mod_path': template_mod_path,
            'shared_task_store_path': shared_task_store_path,
            'pool_root': pool_root,
            'manifest_name': manifest_name,
            'link_mode': link_mode,
        },
    }
    layout = _resolve_template_layout(template_cfg)
    template_fingerprint = compute_runtime_template_fingerprint(layout['runtime_root'])
    effective_link_mode = _resolve_effective_link_mode(layout['runtime_platform'], layout['link_mode'])

    pool_root_path = layout['pool_root']
    pool_root_path.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(pool_root_path, layout['manifest_name'])
    existing = _load_manifest(manifest_path)

    expected_workers = _expected_worker_dirs(pool_root_path, pool_size)
    compatible_existing = bool(
        existing
        and int(existing.get('layout_version', -1)) == int(RUNTIME_SANDBOX_LAYOUT_VERSION)
        and str(existing.get('runtime_platform', '')) == str(layout['runtime_platform'])
        and str(existing.get('template_fingerprint', '')) == str(template_fingerprint)
        and str(existing.get('env_relpath', '')) == str(layout['env_relpath'])
        and str(existing.get('mod_relpath', '')) == str(layout['mod_relpath'])
        and str(existing.get('link_mode', '')) == str(layout['link_mode'])
        and str(existing.get('shared_task_store_path', '')) == str(layout['shared_task_store_path'])
        and str(existing.get('shared_task_store_link_mode', '')) == str(effective_link_mode)
    )

    if not compatible_existing:
        force_refresh = True

    for worker_dir_name in expected_workers:
        worker_root = pool_root_path / worker_dir_name
        if force_refresh and worker_root.exists():
            _remove_tree(worker_root)
        if not worker_root.exists():
            shutil.copytree(layout['runtime_root'], worker_root, symlinks=False)

        worker_mod_path = worker_root / str(layout['mod_relpath'])
        _ensure_shared_task_store(
            worker_mod_path,
            Path(str(layout['shared_task_store_path'])),
            runtime_platform=layout['runtime_platform'],
            requested_link_mode=layout['link_mode'],
        )

    if clean_extra_workers:
        expected_names = set(expected_workers)
        for entry in sorted(pool_root_path.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_dir() or not entry.name.startswith('worker_'):
                continue
            if entry.name not in expected_names:
                _remove_tree(entry)

    manifest = {
        'layout_version': int(RUNTIME_SANDBOX_LAYOUT_VERSION),
        'runtime_platform': layout['runtime_platform'],
        'template_root': str(layout['runtime_root']),
        'template_env_path': str(layout['env_path']),
        'template_mod_path': str(layout['mod_path']),
        'template_fingerprint': template_fingerprint,
        'pool_root': str(pool_root_path),
        'pool_size': int(pool_size),
        'env_relpath': str(layout['env_relpath']),
        'mod_relpath': str(layout['mod_relpath']),
        'shared_task_store_path': str(layout['shared_task_store_path']),
        'shared_task_store_link_mode': str(effective_link_mode),
        'manifest_name': str(layout['manifest_name']),
        'link_mode': str(layout['link_mode']),
        'worker_dirs': expected_workers,
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def ensure_runtime_pool(cfg: Dict[str, Any], *, worker_index: Optional[int] = None) -> Dict[str, Any]:
    sandbox_cfg = dict(cfg.get('runtime_sandbox', {}) or {})
    if not _coerce_bool(sandbox_cfg.get('enabled', False), default=False):
        raise ValueError('runtime_sandbox is not enabled in cfg')

    resolved_worker_index = int(cfg.get('worker_index', 0) if worker_index is None else worker_index)
    if resolved_worker_index < 0:
        raise ValueError(f'worker_index must be >= 0, got {resolved_worker_index}')

    layout = _resolve_template_layout(cfg)
    pool_size = int(sandbox_cfg.get('pool_size', resolved_worker_index + 1) or (resolved_worker_index + 1))
    pool_size = max(pool_size, resolved_worker_index + 1)
    auto_prepare = _coerce_bool(sandbox_cfg.get('auto_prepare', False), default=False)
    force_refresh = _coerce_bool(sandbox_cfg.get('force_refresh', False), default=False)
    clean_extra_workers = _coerce_bool(sandbox_cfg.get('clean_extra_workers', False), default=False)
    manifest_path = _manifest_path(layout['pool_root'], layout['manifest_name'])
    template_fingerprint = compute_runtime_template_fingerprint(layout['runtime_root'])

    cache_key = (
        str(layout['runtime_platform']),
        str(layout['runtime_root']),
        str(layout['pool_root']),
        str(layout['manifest_name']),
        str(template_fingerprint),
        int(pool_size),
        bool(auto_prepare),
        bool(force_refresh),
        bool(clean_extra_workers),
    )

    with _ENSURE_LOCK:
        cached = _ENSURE_CACHE.get(cache_key, None)
        if cached is None:
            manifest = _load_manifest(manifest_path)
            manifest_is_compatible = bool(
                manifest
                and int(manifest.get('layout_version', -1)) == int(RUNTIME_SANDBOX_LAYOUT_VERSION)
                and str(manifest.get('runtime_platform', '')) == str(layout['runtime_platform'])
                and str(manifest.get('template_fingerprint', '')) == str(template_fingerprint)
                and str(manifest.get('env_relpath', '')) == str(layout['env_relpath'])
                and str(manifest.get('mod_relpath', '')) == str(layout['mod_relpath'])
                and str(manifest.get('link_mode', '')) == str(layout['link_mode'])
                and str(manifest.get('shared_task_store_path', '')) == str(layout['shared_task_store_path'])
                and str(manifest.get('shared_task_store_link_mode', ''))
                == str(_resolve_effective_link_mode(layout['runtime_platform'], layout['link_mode']))
                and int(manifest.get('pool_size', 0) or 0) >= int(pool_size)
            )
            if not manifest_is_compatible:
                if not auto_prepare:
                    raise RuntimeError(
                        'Runtime sandbox pool is missing or stale. '
                        f'Expected manifest={manifest_path}. '
                        'Run the prepare_runtime_pool command or enable runtime_sandbox.auto_prepare.'
                    )
                manifest = prepare_runtime_pool(
                    runtime_platform=layout['runtime_platform'],
                    template_root=str(layout['runtime_root']),
                    template_env_path=str(layout['env_path']),
                    template_mod_path=str(layout['mod_path']),
                    shared_task_store_path=str(layout['shared_task_store_path']),
                    pool_root=str(layout['pool_root']),
                    pool_size=pool_size,
                    force_refresh=force_refresh,
                    clean_extra_workers=clean_extra_workers,
                    manifest_name=layout['manifest_name'],
                    link_mode=layout['link_mode'],
                )
            _ENSURE_CACHE[cache_key] = manifest
        else:
            manifest = cached

    worker_root = layout['pool_root'] / _worker_name(resolved_worker_index)
    if not worker_root.exists():
        if not auto_prepare:
            raise RuntimeError(
                f'Runtime sandbox worker directory missing: {worker_root}. '
                'Prepare the runtime pool first or enable runtime_sandbox.auto_prepare.'
            )
        manifest = prepare_runtime_pool(
            runtime_platform=layout['runtime_platform'],
            template_root=str(layout['runtime_root']),
            template_env_path=str(layout['env_path']),
            template_mod_path=str(layout['mod_path']),
            shared_task_store_path=str(layout['shared_task_store_path']),
            pool_root=str(layout['pool_root']),
            pool_size=pool_size,
            force_refresh=force_refresh,
            clean_extra_workers=clean_extra_workers,
            manifest_name=layout['manifest_name'],
            link_mode=layout['link_mode'],
        )
        with _ENSURE_LOCK:
            _ENSURE_CACHE[cache_key] = manifest

    worker_mod_path = worker_root / str(manifest['mod_relpath'])
    worker_task_store_path = _shared_task_store_link_path(worker_mod_path)
    if not _shared_task_store_is_compatible(
        worker_task_store_path,
        Path(str(manifest['shared_task_store_path'])),
        runtime_platform=str(manifest['runtime_platform']),
        effective_link_mode=str(manifest['shared_task_store_link_mode']),
    ):
        if not auto_prepare:
            raise RuntimeError(
                f'Runtime sandbox shared task store is missing or stale: {worker_task_store_path}. '
                'Prepare the runtime pool first or enable runtime_sandbox.auto_prepare.'
            )
        manifest = prepare_runtime_pool(
            runtime_platform=layout['runtime_platform'],
            template_root=str(layout['runtime_root']),
            template_env_path=str(layout['env_path']),
            template_mod_path=str(layout['mod_path']),
            shared_task_store_path=str(layout['shared_task_store_path']),
            pool_root=str(layout['pool_root']),
            pool_size=pool_size,
            force_refresh=force_refresh,
            clean_extra_workers=clean_extra_workers,
            manifest_name=layout['manifest_name'],
            link_mode=layout['link_mode'],
        )
        with _ENSURE_LOCK:
            _ENSURE_CACHE[cache_key] = manifest

    return manifest


def ensure_runtime_pool_range(
    cfg: Dict[str, Any],
    *,
    worker_index_base: int = 0,
    worker_count: int = 1,
) -> Dict[str, Any]:
    if int(worker_count) <= 0:
        raise ValueError('worker_count must be > 0')

    effective_cfg = dict(cfg)
    effective_cfg['worker_index'] = int(worker_index_base) + int(worker_count) - 1
    sandbox_cfg = dict(effective_cfg.get('runtime_sandbox', {}) or {})
    effective_cfg['runtime_sandbox'] = sandbox_cfg
    if not _coerce_bool(sandbox_cfg.get('enabled', False), default=False):
        return {}
    return ensure_runtime_pool(effective_cfg, worker_index=effective_cfg['worker_index'])


def resolve_worker_runtime(cfg: Dict[str, Any], *, worker_index: Optional[int] = None) -> Dict[str, Any]:
    sandbox_cfg = dict(cfg.get('runtime_sandbox', {}) or {})
    if not _coerce_bool(sandbox_cfg.get('enabled', False), default=False):
        raise ValueError('runtime_sandbox is not enabled in cfg')

    resolved_worker_index = int(cfg.get('worker_index', 0) if worker_index is None else worker_index)
    manifest = ensure_runtime_pool(cfg, worker_index=resolved_worker_index)
    pool_root = Path(str(manifest['pool_root']))
    worker_name = _worker_name(resolved_worker_index)
    worker_root = pool_root / worker_name
    env_path = worker_root / str(manifest['env_relpath'])
    mod_path = worker_root / str(manifest['mod_relpath'])
    task_store_path = _shared_task_store_link_path(mod_path)

    if not env_path.exists():
        raise FileNotFoundError(f'Resolved worker env_path does not exist: {env_path}')
    if not mod_path.exists():
        raise FileNotFoundError(f'Resolved worker mod_path does not exist: {mod_path}')
    if not task_store_path.exists():
        raise FileNotFoundError(f'Resolved worker task store path does not exist: {task_store_path}')

    return {
        'worker_index': resolved_worker_index,
        'worker_name': worker_name,
        'runtime_platform': str(manifest['runtime_platform']),
        'runtime_root': str(worker_root),
        'env_path': str(env_path),
        'active_mod_path': str(mod_path),
        'task_store_path': str(task_store_path),
        'pool_root': str(pool_root),
        'manifest_path': str(_manifest_path(pool_root, str(manifest.get('manifest_name', DEFAULT_RUNTIME_POOL_MANIFEST)))),
    }


def resolve_runtime_sandbox_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        return cfg

    sandbox_cfg = dict(cfg.get('runtime_sandbox', {}) or {})
    if not _coerce_bool(sandbox_cfg.get('enabled', False), default=False):
        return dict(cfg)

    out = dict(cfg)
    sandbox_cfg = dict(out.get('runtime_sandbox', {}) or {})

    if sandbox_cfg.get('template_env_path', None) in (None, '') and out.get('env_path', None) not in (None, ''):
        sandbox_cfg['template_env_path'] = out.get('env_path')
    if sandbox_cfg.get('template_mod_path', None) in (None, '') and out.get('mod_path', None) not in (None, ''):
        sandbox_cfg['template_mod_path'] = out.get('mod_path')
    out['runtime_sandbox'] = sandbox_cfg

    resolution = resolve_worker_runtime(out)
    out['env_path'] = resolution['env_path']
    out['mod_path'] = resolution['active_mod_path']
    out['runtime_sandbox']['resolved_runtime_root'] = resolution['runtime_root']
    out['runtime_sandbox']['resolved_env_path'] = resolution['env_path']
    out['runtime_sandbox']['resolved_mod_path'] = resolution['active_mod_path']
    out['runtime_sandbox']['resolved_task_store_path'] = resolution['task_store_path']
    out['runtime_sandbox']['resolved_worker_name'] = resolution['worker_name']

    if _coerce_bool(out['runtime_sandbox'].get('log_resolution', False), default=False):
        print(
            f"[runtime_sandbox] worker_index={resolution['worker_index']} "
            f"runtime_root={resolution['runtime_root']} env_path={resolution['env_path']} "
            f"mod_path={resolution['active_mod_path']} task_store_path={resolution['task_store_path']}"
        )

    return out


def build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare per-worker Unity runtime sandboxes')
    parser.add_argument('--runtime-platform', choices=['linux', 'windows'], default=None)
    parser.add_argument('--template-root', required=True, help='Unity runtime template root or executable path')
    parser.add_argument('--template-env-path', default=None, help='Optional explicit template env_path')
    parser.add_argument('--template-mod-path', default=None, help='Optional explicit template Mods path')
    parser.add_argument('--shared-task-store-path', required=True, help='Host path for the shared all_tasks repository')
    parser.add_argument('--pool-root', required=True, help='Parent directory for prepared worker sandboxes')
    parser.add_argument('--pool-size', type=int, required=True, help='Number of worker sandboxes to prepare')
    parser.add_argument('--force-refresh', action='store_true', help='Rebuild existing worker sandboxes')
    parser.add_argument('--clean-extra-workers', action='store_true', help='Remove worker dirs beyond pool_size')
    parser.add_argument('--manifest-name', default=DEFAULT_RUNTIME_POOL_MANIFEST)
    parser.add_argument('--link-mode', choices=['auto', 'symlink', 'junction', 'copy'], default='auto')
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_prepare_parser().parse_args(argv)
    manifest = prepare_runtime_pool(
        runtime_platform=_normalize_runtime_platform(args.runtime_platform),
        template_root=args.template_root,
        template_env_path=args.template_env_path,
        template_mod_path=args.template_mod_path,
        shared_task_store_path=args.shared_task_store_path,
        pool_root=args.pool_root,
        pool_size=int(args.pool_size),
        force_refresh=bool(args.force_refresh),
        clean_extra_workers=bool(args.clean_extra_workers),
        manifest_name=str(args.manifest_name),
        link_mode=str(args.link_mode),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
