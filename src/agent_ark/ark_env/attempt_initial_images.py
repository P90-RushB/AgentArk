"""Load optional task-authored images for the first decision of each attempt.

The source is ``env_wrapper_cfg.attempt_initial_images: list[str]``. Missing
configuration or an empty list does no filesystem work. Relative paths use the
effective Mods root, never the process working directory. Static PNG, JPEG and
WEBP files retain their native pixels/aspect ratio after EXIF orientation is
applied. PNG-safe grayscale, palette and alpha modes are preserved; other modes
(including CMYK JPEG) convert to RGB/RGBA for lossless PNG message transport.
Each call reads and decodes a fresh copy; callers may cache it only for the
current attempt.

Explicit ``${section.key}`` placeholders read strings or ordered string lists
from the current effective config on each call. Each path template permits one
list reference; configuration references never recurse.

Bounds are deliberately independent of observation frame/image caps: at most
16 files, 16 MiB per file / 64 MiB total, and 32 million decoded pixels per
image / 128 million pixels total. Invalid inputs fail explicitly rather than
silently removing a task-visible reference.
"""

import hashlib
import io
from pathlib import Path
import re
import warnings
from typing import Optional

from PIL import Image, ImageOps


MAX_ATTEMPT_INITIAL_IMAGES = 16
MAX_IMAGE_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_IMAGE_FILE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
MAX_TOTAL_IMAGE_PIXELS = 128_000_000
SUPPORTED_IMAGE_FORMATS = frozenset({'PNG', 'JPEG', 'WEBP'})
_PNG_SAFE_MODES = frozenset({'1', 'L', 'LA', 'P', 'RGB', 'RGBA', 'I', 'I;16', 'I;16B'})
_CONFIG_FIELD = 'env_wrapper_cfg.attempt_initial_images'
_PLACEHOLDER = re.compile(r'\$\{([^}]*)\}')
_CONFIG_KEY = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\Z')


def _resolve_config_references(value: str, env_cfg: dict, field: str) -> list[str]:
    """Expand direct config keys with at most one ordered list per template."""
    fragments = []
    list_index = None
    end = 0
    for match in _PLACEHOLDER.finditer(value):
        fragments.append(value[end:match.start()])
        end = match.end()
        key = match.group(1)
        if '.' not in key:
            # Preserve literal paths, including environment-style placeholders.
            fragments.append(match.group(0))
            continue
        if not _CONFIG_KEY.fullmatch(key):
            raise ValueError(f'{field} contains an invalid configuration reference')
        current = env_cfg
        for component in key.split('.'):
            if not isinstance(current, dict) or component not in current:
                raise ValueError(f'{field} contains an unknown configuration reference')
            current = current[component]
        if isinstance(current, list):
            if list_index is not None:
                raise ValueError(f'{field} permits at most one list configuration reference')
            if len(current) > MAX_ATTEMPT_INITIAL_IMAGES:
                raise ValueError(f'{field} expands to more than {MAX_ATTEMPT_INITIAL_IMAGES} images')
            list_index = len(fragments)
            members = current
        elif isinstance(current, str):
            members = [current]
        else:
            raise ValueError(f'{field} configuration references must name string values or lists of strings')
        for member_index, member in enumerate(members):
            if not isinstance(member, str) or (isinstance(current, list) and not member.strip()):
                raise ValueError(f'{field} list reference member [{member_index}] must be a nonempty string')
            if any('.' in nested.group(1) for nested in _PLACEHOLDER.finditer(member)):
                raise ValueError(f'{field} configuration references must not be recursive or cyclic')
        fragments.append(current)
    fragments.append(value[end:])
    if list_index is None:
        return [''.join(fragments)]
    prefix = ''.join(fragments[:list_index])
    suffix = ''.join(fragments[list_index + 1:])
    return [prefix + member + suffix for member in fragments[list_index]]


def resolve_attempt_initial_image_paths(env_cfg: Optional[dict]) -> list[str]:
    """Validate config and return path strings with direct references resolved.

    A single list reference per template expands in its original order. This
    pure helper performs no filesystem/path operations and never modifies the
    configuration. Relative paths remain relative, and environment-style
    placeholders and tilde characters remain literal.
    """
    wrapper_cfg = env_cfg.get('env_wrapper_cfg', {}) if isinstance(env_cfg, dict) else {}
    if not isinstance(wrapper_cfg, dict) or 'attempt_initial_images' not in wrapper_cfg:
        return []
    paths = wrapper_cfg['attempt_initial_images']
    if not isinstance(paths, list):
        raise ValueError(f'{_CONFIG_FIELD} must be a list of nonempty path strings')
    if not paths:
        return []
    resolved_paths = []
    for offset, value in enumerate(paths):
        field = f'{_CONFIG_FIELD}[{offset}]'
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{field} must be a nonempty path string')
        expanded = _resolve_config_references(value, env_cfg, field)
        if any(not path.strip() for path in expanded):
            raise ValueError(f'{field} must resolve to a nonempty path string')
        resolved_paths.extend(expanded)
        if len(resolved_paths) > MAX_ATTEMPT_INITIAL_IMAGES:
            raise ValueError(f'{_CONFIG_FIELD} permits at most {MAX_ATTEMPT_INITIAL_IMAGES} images after expansion')
    return resolved_paths


def load_attempt_initial_images(env_cfg: Optional[dict], mod_path) -> list[dict]:
    """Return ordered, detached Pillow images and SHA-256 source-byte hashes.

    The returned records contain only ``index`` (one based), ``image`` and
    ``sha256``. Source paths are never included in observations. All configured
    entries are validated before reading any file. A relative entry requires
    an absolute effective Mods root; this prevents an implicit cwd fallback.
    """
    paths = resolve_attempt_initial_image_paths(env_cfg)
    resolved_paths = []
    for offset, value in enumerate(paths):
        field = f'{_CONFIG_FIELD}[{offset}]'
        path = Path(value)
        if not path.is_absolute():
            if mod_path is None or not str(mod_path).strip() or not Path(mod_path).is_absolute():
                raise ValueError(f'{field} requires an absolute effective Mods root for a relative path')
            path = Path(mod_path) / path
        resolved_paths.append(path)

    records = []
    total_bytes = 0
    total_pixels = 0
    for offset, path in enumerate(resolved_paths):
        field = f'{_CONFIG_FIELD}[{offset}]'
        try:
            with path.open('rb') as source:
                data = source.read(MAX_IMAGE_FILE_BYTES + 1)
        except (OSError, ValueError) as exc:
            raise ValueError(f'{field} could not read image file') from exc
        if len(data) > MAX_IMAGE_FILE_BYTES:
            raise ValueError(f'{field} exceeds the {MAX_IMAGE_FILE_BYTES}-byte file limit')
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_IMAGE_FILE_BYTES:
            raise ValueError(f'{_CONFIG_FIELD} exceeds the {MAX_TOTAL_IMAGE_FILE_BYTES}-byte total limit')

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                source_image = Image.open(io.BytesIO(data))
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValueError(f'{field} is not a decodable static PNG, JPEG or WEBP image') from exc

        with source_image:
            if source_image.format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f'{field} must be a static PNG, JPEG or WEBP image')
            if getattr(source_image, 'n_frames', 1) != 1 or getattr(source_image, 'is_animated', False):
                raise ValueError(f'{field} must be static; animated images are unsupported')
            pixels = source_image.width * source_image.height
            if pixels > MAX_IMAGE_PIXELS:
                raise ValueError(f'{field} exceeds the {MAX_IMAGE_PIXELS}-pixel image limit')
            total_pixels += pixels
            if total_pixels > MAX_TOTAL_IMAGE_PIXELS:
                raise ValueError(f'{_CONFIG_FIELD} exceeds the {MAX_TOTAL_IMAGE_PIXELS}-pixel total limit')
            try:
                source_image.load()
                image = ImageOps.exif_transpose(source_image)
                if image.mode not in _PNG_SAFE_MODES:
                    has_alpha = any(band.lower() == 'a' for band in image.getbands())
                    has_alpha = has_alpha or 'transparency' in image.info
                    image = image.convert('RGBA' if has_alpha else 'RGB')
                image = image.copy()
            except (OSError, SyntaxError, ValueError) as exc:
                raise ValueError(f'{field} could not fully decode image pixels or EXIF orientation') from exc
        records.append({
            'index': offset + 1,
            'image': image,
            'sha256': hashlib.sha256(data).hexdigest(),
        })
    return records
