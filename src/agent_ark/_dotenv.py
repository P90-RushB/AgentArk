from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union


_FALSE_VALUES = {"0", "false", "no", "off"}
_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _iter_search_dirs(start: Path) -> Iterable[Path]:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    yield current
    yield from current.parents


def find_agentark_dotenv(start: Optional[Union[os.PathLike, str]] = None) -> Optional[Path]:
    """Find the nearest .env from cwd first, then from the source tree."""
    search_starts = [Path(start) if start is not None else Path.cwd(), Path(__file__)]
    seen = set()
    for search_start in search_starts:
        for directory in _iter_search_dirs(search_start):
            if directory in seen:
                continue
            seen.add(directory)
            candidate = directory / ".env"
            if candidate.is_file():
                return candidate
    return None


def _split_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    escaped = False

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double:
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value


def _unquote_value(value: str) -> str:
    value = _split_inline_comment(value.strip())
    if len(value) < 2 or value[0] != value[-1] or value[0] not in ("'", '"'):
        return value

    quote = value[0]
    inner = value[1:-1]
    if quote == "'":
        return inner

    escapes = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        '"': '"',
        "\\": "\\",
        "$": "$",
    }
    out = []
    index = 0
    while index < len(inner):
        char = inner[index]
        if char == "\\" and index + 1 < len(inner):
            next_char = inner[index + 1]
            if next_char in escapes:
                out.append(escapes[next_char])
                index += 2
                continue
        out.append(char)
        index += 1
    return "".join(out)


def _parse_line(line: str) -> Optional[Tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not _KEY_PATTERN.match(key):
        return None
    return key, _unquote_value(value)


def _expand_value(value: str) -> str:
    expanded = value
    for _ in range(3):
        next_value = os.path.expandvars(expanded)
        if next_value == expanded:
            break
        expanded = next_value
    return os.path.expanduser(expanded)


def load_agentark_dotenv(
    path: Optional[Union[os.PathLike, str]] = None,
    *,
    override: bool = False,
) -> Optional[Path]:
    """Load a local AgentArk .env file into os.environ.

    Existing process environment variables win by default so CI, shells, and
    launch scripts can always override developer-local .env values.
    """
    dotenv_path = Path(path) if path is not None else find_agentark_dotenv()
    if dotenv_path is None or not dotenv_path.is_file():
        return None

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = _expand_value(value)
    return dotenv_path


def auto_load_agentark_dotenv() -> Optional[Path]:
    if os.environ.get("AGENTARK_AUTO_LOAD_DOTENV", "1").strip().lower() in _FALSE_VALUES:
        return None
    return load_agentark_dotenv(override=False)
