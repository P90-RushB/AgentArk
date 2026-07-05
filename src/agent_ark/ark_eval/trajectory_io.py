from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from agent_ark.utils.image_utils import env_arr_to_pil_image, pil_image_to_base64


TRAJECTORY_FORMAT_VERSION = 1
_IMAGE_MARKER = 'agentark.pil_image_png_base64.v1'
_BYTES_MARKER = 'agentark.bytes_base64.v1'
_IMAGE_OMITTED_MARKER = 'agentark.image_omitted.v1'


def _is_pil_image(value: Any) -> bool:
    return Image is not None and isinstance(value, Image.Image)


def _is_numpy_array(value: Any) -> bool:
    return np is not None and isinstance(value, np.ndarray)


def _encode_image(value: Any, *, source_type: str) -> Dict[str, Any]:
    if _is_pil_image(value):
        img = value
    else:
        img = env_arr_to_pil_image(value)
    return {
        '__agentark_type__': _IMAGE_MARKER,
        'mime_type': 'image/png',
        'source_type': source_type,
        'mode': getattr(img, 'mode', None),
        'size': list(getattr(img, 'size', []) or []),
        'data': pil_image_to_base64(img),
    }


def _encode_omitted_image(value: Any, *, source_type: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        '__agentark_type__': _IMAGE_OMITTED_MARKER,
        'source_type': source_type,
    }
    size = getattr(value, 'size', None)
    if size is not None:
        try:
            out['size'] = list(size)
        except Exception:
            out['size'] = str(size)
    shape = getattr(value, 'shape', None)
    if shape is not None:
        try:
            out['shape'] = list(shape)
        except Exception:
            out['shape'] = str(shape)
    return out


def encode_trajectory_value(value: Any, *, include_images: bool = True) -> Any:
    if _is_pil_image(value):
        return _encode_image(value, source_type='pil') if include_images else _encode_omitted_image(value, source_type='pil')

    if _is_numpy_array(value):
        if include_images:
            try:
                return _encode_image(value, source_type='ndarray')
            except Exception:
                pass
        return _encode_omitted_image(value, source_type='ndarray')

    if np is not None and isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(key): encode_trajectory_value(item, include_images=include_images) for key, item in value.items()}
    if isinstance(value, list):
        return [encode_trajectory_value(item, include_images=include_images) for item in value]
    if isinstance(value, tuple):
        return [encode_trajectory_value(item, include_images=include_images) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            '__agentark_type__': _BYTES_MARKER,
            'data': base64.b64encode(bytes(value)).decode('utf-8'),
        }

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _decode_image(payload: Dict[str, Any]) -> Any:
    data = payload.get('data', '')
    if Image is None or not isinstance(data, str):
        return data
    raw = base64.b64decode(data)
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


def decode_trajectory_value(value: Any) -> Any:
    if isinstance(value, dict):
        marker = value.get('__agentark_type__')
        if marker == _IMAGE_MARKER:
            return _decode_image(value)
        if marker == _BYTES_MARKER:
            data = value.get('data', '')
            return base64.b64decode(data) if isinstance(data, str) else b''
        if marker == _IMAGE_OMITTED_MARKER:
            return None
        return {key: decode_trajectory_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_trajectory_value(item) for item in value]
    return value


def normalize_history_snapshot(history_snapshot: Optional[Dict[Any, Any]]) -> Dict[int, list]:
    if not isinstance(history_snapshot, dict):
        return {}
    out: Dict[int, list] = {}
    for raw_unity_id, attempts in history_snapshot.items():
        try:
            unity_id = int(raw_unity_id)
        except Exception:
            continue
        out[unity_id] = deepcopy(attempts) if isinstance(attempts, list) else []
    return out


def encode_history_snapshot(history_snapshot: Optional[Dict[Any, Any]], *, include_images: bool = True) -> Dict[str, Any]:
    return encode_trajectory_value(normalize_history_snapshot(history_snapshot), include_images=include_images)


def decode_history_snapshot(history_snapshot: Optional[Dict[Any, Any]]) -> Dict[int, list]:
    return normalize_history_snapshot(decode_trajectory_value(history_snapshot or {}))


def slice_history_snapshot(history_snapshot: Optional[Dict[Any, Any]], prefix_attempts: Optional[int]) -> Dict[int, list]:
    snapshot = normalize_history_snapshot(history_snapshot)
    if prefix_attempts is None:
        return snapshot
    limit = max(0, int(prefix_attempts))
    return {unity_id: deepcopy(attempts[:limit]) for unity_id, attempts in snapshot.items()}


def count_history_prefix_attempts(history_snapshot: Optional[Dict[Any, Any]]) -> int:
    snapshot = normalize_history_snapshot(history_snapshot)
    if not snapshot:
        return 0
    return max((len(attempts) for attempts in snapshot.values() if isinstance(attempts, list)), default=0)


def _stable_record_hash(parts: Iterable[Any]) -> str:
    material = json.dumps(encode_trajectory_value(list(parts), include_images=False), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(material.encode('utf-8')).hexdigest()[:12]


def build_eval_trajectory_record(
    *,
    result: Dict[str, Any],
    case: Dict[str, Any],
    model_runtime: Dict[str, Any],
    history_snapshot: Dict[int, list],
    prefix_attempts: Optional[int],
    include_images: bool = True,
) -> Dict[str, Any]:
    prefix_snapshot = slice_history_snapshot(history_snapshot, prefix_attempts)
    prefix_attempt_count = count_history_prefix_attempts(prefix_snapshot)
    source_case_id = str(result.get('case_id', case.get('case_id', 'case')))
    model_name = str(result.get('model_name', model_runtime.get('name', 'model')))
    trajectory_id = 'traj-' + _stable_record_hash([
        source_case_id,
        model_name,
        result.get('requested_task_name', case.get('task_name')),
        result.get('requested_group_seed', case.get('group_seed')),
        result.get('requested_env_id', case.get('env_id')),
        result.get('attempt_group_seed_history'),
        prefix_attempt_count,
        time.time_ns(),
    ])

    record = {
        'format': 'agentark_trajectory',
        'version': TRAJECTORY_FORMAT_VERSION,
        'trajectory_id': trajectory_id,
        'created_at_unix_s': round(time.time(), 6),
        'source': {
            'kind': 'ark_eval',
            'case_id': source_case_id,
            'model_name': model_name,
            'model': result.get('model', model_runtime.get('model')),
            'provider': result.get('provider', model_runtime.get('provider')),
        },
        'task': {
            'task_name': result.get('actual_task_name', result.get('requested_task_name', case.get('task_name'))),
            'requested_task_name': result.get('requested_task_name', case.get('task_name')),
            'group_seed': result.get('actual_rollout_group_seed', result.get('requested_group_seed', case.get('group_seed'))),
            'requested_group_seed': result.get('requested_group_seed', case.get('group_seed')),
            'env_id': result.get('actual_env_id', result.get('requested_env_id', case.get('env_id'))),
            'requested_env_id': result.get('requested_env_id', case.get('env_id')),
        },
        'rollout': {
            'max_attempts': result.get('max_attempts'),
            'max_steps_per_attempt': result.get('max_steps_per_attempt'),
            'rollout_step_budget': result.get('rollout_step_budget'),
            'attempt_group_seed_history': result.get('attempt_group_seed_history', []),
            'score_reward': result.get('score_reward'),
            'last_attempt_reward': result.get('last_attempt_reward'),
            'best_attempt_reward': result.get('best_attempt_reward'),
            'rollout_success': result.get('rollout_success'),
            'ever_attempt_success': result.get('ever_attempt_success'),
            'rollout_truncated': result.get('rollout_truncated'),
            'attempt_rewards': result.get('attempt_rewards', []),
        },
        'prefix': {
            'requested_attempts': prefix_attempts,
            'attempt_count': prefix_attempt_count,
            'target_attempt_index': prefix_attempt_count + 1,
            'include_images': bool(include_images),
        },
        'history_snapshot': encode_history_snapshot(prefix_snapshot, include_images=include_images),
    }
    return record


class TrajectoryJsonlWriter:
    def __init__(self, path: str, *, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not append:
            self.path.unlink()
        self._lock = threading.Lock()

    def write(self, record: Dict[str, Any]) -> None:
        with self._lock:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(encode_trajectory_value(record, include_images=True), ensure_ascii=False) + '\n')


def load_trajectory_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Trajectory file not found: {path}')
    with p.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(decode_trajectory_value(json.loads(text)))
            except Exception as exc:
                raise ValueError(f'Failed to parse trajectory JSONL {path}:{line_no}: {exc}') from exc
    return records


def select_trajectory_record(
    records: List[Dict[str, Any]],
    *,
    trajectory_id: Optional[str] = None,
    case_id: Optional[str] = None,
    model_name: Optional[str] = None,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    if not records:
        raise ValueError('Trajectory file has no records')
    if index is not None:
        idx = int(index)
        try:
            return records[idx]
        except IndexError as exc:
            raise IndexError(f'Trajectory record index out of range: {idx}') from exc

    candidates = list(records)
    if trajectory_id:
        candidates = [record for record in candidates if str(record.get('trajectory_id', '')) == str(trajectory_id)]
    if case_id:
        candidates = [
            record for record in candidates
            if str((record.get('source') or {}).get('case_id', '')) == str(case_id)
        ]
    if model_name:
        candidates = [
            record for record in candidates
            if str((record.get('source') or {}).get('model_name', '')) == str(model_name)
        ]

    if not candidates:
        raise ValueError(
            'No trajectory record matched '
            f'trajectory_id={trajectory_id!r} case_id={case_id!r} model_name={model_name!r}'
        )
    if len(candidates) > 1 and trajectory_id is None and index is None:
        return candidates[-1]
    return candidates[0]


def load_trajectory_record(
    path: str,
    *,
    trajectory_id: Optional[str] = None,
    case_id: Optional[str] = None,
    model_name: Optional[str] = None,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    return select_trajectory_record(
        load_trajectory_records(path),
        trajectory_id=trajectory_id,
        case_id=case_id,
        model_name=model_name,
        index=index,
    )


def history_snapshot_from_record(record: Dict[str, Any], *, prefix_attempts: Optional[int] = None) -> Dict[int, list]:
    snapshot = decode_history_snapshot(record.get('history_snapshot', {}))
    if prefix_attempts is None:
        return snapshot
    return slice_history_snapshot(snapshot, prefix_attempts)


def target_attempt_index_from_record(record: Dict[str, Any], history_snapshot: Dict[int, list]) -> int:
    prefix = record.get('prefix', {}) if isinstance(record.get('prefix', {}), dict) else {}
    target = prefix.get('target_attempt_index', None)
    if target is not None:
        try:
            return max(1, int(target))
        except Exception:
            pass
    return count_history_prefix_attempts(history_snapshot) + 1
