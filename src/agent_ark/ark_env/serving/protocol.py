from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class EnvStepPayload:
    unity_id: int
    obs: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class EnvStartPayload:
    env_id: str
    unity_id: int
    obs: Dict[str, Any]
    info: Dict[str, Any]


def _pil_to_base64_png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _base64_png_to_pil(data: str):
    if Image is None:
        return data
    raw = base64.b64decode(data)
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


def encode_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    encoded = dict(obs)
    vis = encoded.get("vis", None)
    if not isinstance(vis, list):
        return encoded

    vis_payload: List[List[str]] = []
    for cam_frames in vis:
        if not isinstance(cam_frames, list):
            vis_payload.append([])
            continue
        frame_payload: List[str] = []
        for frame in cam_frames:
            if Image is not None and hasattr(frame, "save"):
                frame_payload.append(_pil_to_base64_png(frame))
            else:
                frame_payload.append(frame)
        vis_payload.append(frame_payload)

    encoded["vis"] = vis_payload
    encoded["vis_format"] = "png_base64"
    return encoded


def decode_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    decoded = dict(obs)
    if decoded.get("vis_format") != "png_base64":
        return decoded

    vis = decoded.get("vis", None)
    if not isinstance(vis, list):
        return decoded

    out_vis: List[List[Any]] = []
    for cam_frames in vis:
        if not isinstance(cam_frames, list):
            out_vis.append([])
            continue
        frame_list: List[Any] = []
        for frame_b64 in cam_frames:
            if isinstance(frame_b64, str):
                frame_list.append(_base64_png_to_pil(frame_b64))
            else:
                frame_list.append(frame_b64)
        out_vis.append(frame_list)

    decoded["vis"] = out_vis
    return decoded


def as_json_dict(payload_obj: Any) -> Dict[str, Any]:
    return _to_jsonable(asdict(payload_obj))


def _to_jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "__type__": "bytes_base64",
            "data": base64.b64encode(bytes(value)).decode("utf-8"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)
