"""Additive model-call audit images, independent of environment action replay."""
from __future__ import annotations

import base64
import hashlib
import re
from typing import Any


IMAGE_REFERENCE_PREFIX = 'agentark-trace-image:'
_INLINE_IMAGE = re.compile(r'^data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=\r\n]+)$')


def capture_image_reference(value: Any, assets: dict | None) -> str | None:
    """Store exact inline raster bytes once; leave malformed/remote data omitted."""
    if assets is None or not isinstance(value, str):
        return None
    if value.startswith(IMAGE_REFERENCE_PREFIX):
        return value
    match = _INLINE_IMAGE.fullmatch(value)
    if match is None:
        return None
    try:
        raw = base64.b64decode(match[2].replace('\r', '').replace('\n', ''), validate=True)
    except ValueError:
        return None
    if not raw:
        return None
    digest = hashlib.sha256(raw).hexdigest()
    assets.setdefault(digest, {'mime_type': match[1], 'data_url': value})
    return IMAGE_REFERENCE_PREFIX + digest


def referenced_image_assets(value: Any, assets: dict) -> dict:
    """Select only images actually referenced by this saved trace/prefix."""
    found = {}

    def visit(item):
        if isinstance(item, str) and item.startswith(IMAGE_REFERENCE_PREFIX):
            digest = item[len(IMAGE_REFERENCE_PREFIX):]
            if digest in assets:
                found[digest] = assets[digest]
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return found


def validated_image_assets(assets: Any) -> dict:
    """Untrusted saved bytes must match the hash used to attribute an image."""
    if not isinstance(assets, dict):
        return {}
    valid = {}
    for digest, payload in assets.items():
        if not isinstance(payload, dict):
            continue
        candidate = {}
        reference = capture_image_reference(payload.get('data_url'), candidate)
        if reference == IMAGE_REFERENCE_PREFIX + str(digest) and digest in candidate:
            valid[digest] = candidate[digest]
    return valid
