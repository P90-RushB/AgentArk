from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover
    _PILImage = None

try:
    from agent_ark.utils.image_utils import pil_image_to_base64
except Exception:  # pragma: no cover
    pil_image_to_base64 = None

from agent_ark.interaction.hooks import to_jsonable


def truncate_text(value: Any, max_chars: int = 6000) -> str:
    if value is None:
        return ''
    if not isinstance(value, str):
        value = str(value)
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + '\n...[truncated]'


def _is_pil_image(value: Any) -> bool:
    return _PILImage is not None and isinstance(value, _PILImage.Image)


def image_to_data_url(value: Any) -> Optional[str]:
    if isinstance(value, str):
        if value.startswith('data:image/'):
            return value
        return f'data:image/png;base64,{value}'
    if _is_pil_image(value) and pil_image_to_base64 is not None:
        try:
            return f'data:image/png;base64,{pil_image_to_base64(value)}'
        except Exception:
            return None
    return None


def normalize_vis(vis: Any) -> List[List[Any]]:
    if not isinstance(vis, (list, tuple)):
        return []
    if vis and isinstance(vis[0], (list, tuple)):
        return [list(frames) for frames in vis if isinstance(frames, (list, tuple))]
    return [list(vis)]


def serialize_images(vis: Any, *, max_images: int = 4) -> List[Dict[str, Any]]:
    if max_images <= 0:
        return []
    cams = normalize_vis(vis)
    selected = []
    pointers = [len(frames) - 1 for frames in cams]
    while len(selected) < max_images and any(pointer >= 0 for pointer in pointers):
        for cam_idx, frames in enumerate(cams):
            if len(selected) >= max_images:
                break
            frame_idx = pointers[cam_idx]
            if frame_idx < 0:
                continue
            selected.append((cam_idx, frame_idx))
            pointers[cam_idx] = frame_idx - 1
    selected.sort(key=lambda item: (item[1], item[0]))

    out: List[Dict[str, Any]] = []
    for cam_idx, frame_idx in selected:
        frames = cams[cam_idx]
        url = image_to_data_url(frames[frame_idx])
        if not url:
            continue
        out.append({
            'camera_index': cam_idx,
            'frame_index': frame_idx,
            'url': url,
        })
    return out


def serialize_message_part(part: Any, *, text_max_chars: int = 6000) -> Dict[str, Any]:
    if not isinstance(part, dict):
        return {'type': 'text', 'text': truncate_text(part, text_max_chars)}

    part_type = part.get('type')
    if part_type == 'text':
        return {'type': 'text', 'text': truncate_text(part.get('text', ''), text_max_chars)}

    if part_type == 'image_url':
        image_url = part.get('image_url')
        if isinstance(image_url, dict):
            url = image_url.get('url', '')
        else:
            url = image_url
        if isinstance(url, str) and url:
            return {'type': 'image_url', 'url': image_to_data_url(url) or url}
        return {'type': 'image_url', 'url': ''}

    if part_type == 'image':
        if 'image_base64' in part:
            url = image_to_data_url(part.get('image_base64'))
            return {'type': 'image_url', 'url': url or ''}
        if 'image' in part:
            url = image_to_data_url(part.get('image'))
            return {'type': 'image_url', 'url': url or ''}

    return to_jsonable(part)


def serialize_messages(messages: Any, *, text_max_chars: int = 6000) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get('content', '')
        if isinstance(content, list):
            serialized_content = [
                serialize_message_part(part, text_max_chars=text_max_chars)
                for part in content
            ]
        else:
            serialized_content = truncate_text(content, text_max_chars)
        out.append({
            'role': msg.get('role', 'user'),
            'content': serialized_content,
        })
    return out


def serialize_obs(
    obs: Any,
    *,
    text_max_chars: int = 6000,
    max_images: int = 4,
    include_messages: bool = True,
) -> Dict[str, Any]:
    if not isinstance(obs, dict):
        return {'value': to_jsonable(obs)}

    out: Dict[str, Any] = {
        'step_msg': truncate_text(obs.get('step_msg', ''), text_max_chars),
        'skip_infer': bool(obs.get('skip_infer', False)),
        'images': serialize_images(obs.get('vis'), max_images=max_images),
    }

    if include_messages:
        out['messages'] = serialize_messages(obs.get('messages', []), text_max_chars=text_max_chars)

    history = obs.get('history')
    if isinstance(history, list):
        out['history_attempt_count'] = len(history)
        out['history_attempt_step_counts'] = [len(item) if isinstance(item, list) else 0 for item in history[:10]]

    if obs.get('vis_format'):
        out['vis_format'] = str(obs.get('vis_format'))
    return out


def serialize_obs_map(
    obs_map: Any,
    *,
    text_max_chars: int = 6000,
    max_images_per_observation: int = 4,
    include_messages: bool = True,
) -> Dict[str, Any]:
    if not isinstance(obs_map, dict):
        return {'value': to_jsonable(obs_map)}
    return {
        str(agent_id): serialize_obs(
            obs,
            text_max_chars=text_max_chars,
            max_images=max_images_per_observation,
            include_messages=include_messages,
        )
        for agent_id, obs in obs_map.items()
    }


def serialize_action_details(code_act: Any, *, text_max_chars: int = 12000) -> Dict[str, Any]:
    if not isinstance(code_act, dict):
        return {'value': truncate_text(code_act, text_max_chars)}
    out: Dict[str, Any] = {}
    for agent_id, payload in code_act.items():
        if isinstance(payload, dict):
            out[str(agent_id)] = {
                'action': truncate_text(payload.get('action'), text_max_chars),
                'assistant': truncate_text(payload.get('assistant'), text_max_chars),
            }
        else:
            out[str(agent_id)] = {'action': truncate_text(payload, text_max_chars)}
    return out
