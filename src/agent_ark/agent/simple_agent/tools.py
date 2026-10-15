# SPDX-License-Identifier: Apache-2.0
# Adapted from ArkArena application-owned tools, licensed under Apache-2.0.
"""Application-owned tools. No shell, model SDK, or game control channel."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
from types import MappingProxyType
import uuid

from .types import (
    AgentError, AgentInputIdentity, AgentLimitError, AgentRequest,
    AgentReuseError, ImagePart, TextPart, ToolCall, ToolResult, ToolSpec,
)


class ToolError(AgentError):
    """Safe feedback for an invalid tool operation; never includes OS errors."""


@dataclass(frozen=True, slots=True)
class ToolLimits:
    max_calls: int = 32
    max_model_rounds: int = 16
    max_text_bytes: int = 65_536
    max_total_text_bytes: int = 524_288
    max_images: int = 4
    max_image_bytes: int = 5 * 1024 * 1024
    max_total_image_bytes: int = 10 * 1024 * 1024
    max_note_bytes: int = 65_536
    max_workspace_bytes: int = 1024 * 1024
    max_note_files: int = 128
    max_scope_entries: int = 256
    max_record_files: int = 1024
    max_record_file_bytes: int = 32 * 1024 * 1024
    max_record_bytes: int = 64 * 1024 * 1024
    max_records: int = 16
    max_published_bytes: int = 512 * 1024 * 1024
    timeout_seconds: float = 1800.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "timeout_seconds":
                if type(value) not in (int, float) or not 0 < value <= 3600:
                    raise ValueError("tool timeout must be finite and bounded")
            elif type(value) is not int or value <= 0:
                raise ValueError("tool bounds must be positive integers")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _path(value: str, *, empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, str) or len(value) > 512:
        raise ToolError("Use a logical relative path of at most 512 characters.")
    if empty and value in ("", "."):
        return ()
    parts = value.split("/")
    for part in parts:
        if (not part or part in (".", "..") or len(part) > 128 or
                any(ord(c) < 32 or c in '\\:*?"<>|' for c in part) or
                part.endswith((".", " ")) or
                re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?", part)):
            raise ToolError("Path must stay inside records/ or notes/; unsafe path component.")
    return tuple(parts)


def _safe_stat(path: Path, *, directory: bool = False):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400 or
            (getattr(path, "is_junction", lambda: False)()) or
            not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)) or
            (not directory and info.st_nlink != 1)):
        raise ToolError("Links, reparse points and special files are not accessible.")
    return info


def _root(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    for parent in reversed((absolute, *absolute.parents)):
        _safe_stat(parent, directory=True)
    return absolute


def _read_regular(path: Path, bound: int) -> bytes:
    before = _safe_stat(path)
    if before.st_size > bound:
        raise ToolError("File exceeds its byte limit.")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ToolError("File changed during access.")
        data = stream.read(bound + 1)
        after = os.fstat(stream.fileno())
    current = _safe_stat(path)
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink)
    if len(data) > bound or signature(before) != signature(after) or signature(after) != signature(current):
        raise ToolError("File changed during access or exceeded its limit.")
    return data


@dataclass(frozen=True, slots=True)
class RecordSnapshot:
    """Explicit completed record, copied once; never a mutable 'latest' alias."""
    record_id: str
    files: tuple[tuple[str, bytes], ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.record_id):
            raise ValueError("record ID must be a short logical identifier")
        if not isinstance(self.files, tuple) or len(self.files) > 1024:
            raise ValueError("snapshot files must be a bounded immutable tuple")
        names: set[str] = set()
        digest = sha256()
        for entry in self.files:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("snapshot entries must be immutable name/data tuples")
            name, data = entry
            _path(name)
            if name.casefold() in names or not isinstance(data, bytes):
                raise ValueError("snapshot filenames must be unique and data immutable")
            names.add(name.casefold())
            digest.update(_json([name, len(data), sha256(data).hexdigest()]).encode())
        object.__setattr__(self, "digest", digest.hexdigest())

    @classmethod
    def from_directory(cls, record_id: str, directory: str | Path, *, limits: ToolLimits | None = None) -> RecordSnapshot:
        """Caller supplies only filtered own records, never the full evaluator tree."""
        limits = limits or ToolLimits()
        try:
            root = _root(Path(directory))
            files: list[tuple[str, bytes]] = []
            total = entries = 0
            pending = [root]
            while pending:
                current = pending.pop()
                _safe_stat(current, directory=True)
                with os.scandir(current) as children:
                    for child in children:
                        entries += 1
                        if entries > limits.max_record_files * 2:
                            raise ToolError("Record directory entry limit exhausted.")
                        path = Path(child.path)
                        name = path.relative_to(root).as_posix()
                        _path(name)
                        info = path.lstat()
                        if stat.S_ISDIR(info.st_mode):
                            _safe_stat(path, directory=True)
                            pending.append(path)
                        else:
                            data = _read_regular(path, limits.max_record_file_bytes)
                            total += len(data)
                            files.append((name, data))
                            if len(files) > limits.max_record_files or total > limits.max_record_bytes:
                                raise ToolError("Record snapshot capacity exhausted.")
            return cls(record_id, tuple(sorted(files)))
        except OSError:
            raise ToolError("Record directory could not be safely copied.") from None


@dataclass(slots=True)
class ToolSession:
    identity: AgentInputIdentity
    generation: int
    deadline: float
    records: Mapping[str, RecordSnapshot]
    namespace: str
    calls: int = 0
    text_bytes: int = 0
    images: int = 0
    image_bytes: int = 0
    closed: bool = False
    _ids: set[str] = field(default_factory=set)
    _owner: object | None = None

    def close(self) -> None:
        self.closed = True


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _arguments(raw: str) -> dict:
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    if type(value) is not dict:
        raise ValueError("arguments must be an object")
    return value


def _schema(schema: dict, value: object = None, *, preflight: bool = False, depth: int = 0) -> bool:
    """Deliberately bounded tool-schema subset; unsupported keywords fail closed."""
    allowed = {"type", "properties", "required", "additionalProperties", "items", "enum", "minimum", "maximum", "maxLength", "description"}
    if depth > 16 or type(schema) is not dict or set(schema) - allowed:
        raise ValueError("unsupported tool schema")
    types = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool}
    if schema.get("type") not in types:
        raise ValueError("tool schema requires a supported type")
    if preflight:
        if "properties" in schema:
            if type(schema["properties"]) is not dict:
                raise ValueError("invalid properties")
            for child in schema["properties"].values():
                _schema(child, preflight=True, depth=depth + 1)
        if "items" in schema:
            _schema(schema["items"], preflight=True, depth=depth + 1)
        if "required" in schema and (type(schema["required"]) is not list or any(type(v) is not str for v in schema["required"])):
            raise ValueError("invalid required keys")
        if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
            raise ValueError("invalid additionalProperties")
        for name in ("minimum", "maximum", "maxLength"):
            if name in schema and type(schema[name]) is not int:
                raise ValueError("invalid schema bound")
        if "enum" in schema and (type(schema["enum"]) is not list or not schema["enum"]):
            raise ValueError("invalid enum")
        return True
    if type(value) is not types[schema["type"]]:
        return False
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        return False
    if type(value) is dict:
        properties = schema.get("properties", {})
        return (all(k in value for k in schema.get("required", ())) and
                all(_schema(properties[k], v, depth=depth + 1) if k in properties else schema.get("additionalProperties", True) for k, v in value.items()))
    if type(value) is list:
        return "items" not in schema or all(_schema(schema["items"], v, depth=depth + 1) for v in value)
    if type(value) is str and len(value) > schema.get("maxLength", len(value)):
        return False
    return type(value) is not int or schema.get("minimum", value) <= value <= schema.get("maximum", value)


class ToolRegistry:
    """Trusted application extensions; tools never receive an SDK Action client."""
    def __init__(self, *, limits: ToolLimits | None = None) -> None:
        self.limits = limits or ToolLimits()
        self._tools: dict[str, tuple[ToolSpec, Callable]] = {}
        self._owner: object | None = None
        self._binding: tuple[str, str, str] | None = None
        self._current: AgentInputIdentity | None = None
        self._generation = 0
        self._scopes: set[tuple[str, int]] = set()
        self._seen_scopes: set[tuple[str, int]] = set()
        self._lock = threading.RLock()

    def claim(self, owner: object) -> None:
        with self._lock:
            if self._owner is not None and self._owner is not owner:
                raise AgentReuseError("tool registry already has an owner")
            self._owner = owner

    def validate_fresh(self) -> None:
        if self._binding is not None:
            raise AgentReuseError("tool registry already belongs to a lifetime")

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(spec for spec, _ in self._tools.values())

    def register(self, spec: ToolSpec, function: Callable) -> None:
        if self._owner is not None or self._binding is not None:
            raise AgentReuseError("register tools before creating a harness")
        if not isinstance(spec, ToolSpec) or not callable(function) or spec.name in self._tools or len(self._tools) >= 32:
            raise ValueError("invalid or duplicate tool registration")
        _schema(spec.schema, preflight=True)
        self._tools[spec.name] = spec, function

    def accept_input(self, request: AgentRequest) -> None:
        with self._lock:
            identity = request.identity
            binding = (identity.run_id, identity.session_id, identity.agent_id)
            if self._binding is not None and binding != self._binding:
                raise AgentReuseError("tools cannot cross run, Session or agent")
            scope = (identity.episode_id, identity.incarnation)
            if scope not in self._seen_scopes and len(self._seen_scopes) >= self.limits.max_scope_entries:
                raise AgentLimitError("tool scope capacity exhausted")
            if self._current is not None:
                previous = (self._current.episode_id, self._current.incarnation)
                if scope[0] == previous[0] and scope[1] < previous[1]:
                    raise AgentReuseError("tool incarnation regressed")
                if scope != previous and (scope[0] == previous[0] or request.history_policy == "omit"):
                    self._generation += 1
                    self._scopes.clear()
            self._binding, self._current = binding, identity
            self._scopes.add(scope)
            self._seen_scopes.add(scope)

    def begin(self, request: AgentRequest) -> ToolSession:
        with self._lock:
            if self._current != request.identity:
                raise AgentReuseError("tool request is not the accepted input")
            return ToolSession(request.identity, self._generation, time.monotonic() + self.limits.timeout_seconds,
                               MappingProxyType({}), str(self._generation), _owner=self)

    def execute(self, call: ToolCall, session: ToolSession) -> ToolResult:
        with self._lock:
            if (session._owner is not self or session.closed or session.generation != self._generation or
                    self._current is None or (session.identity.episode_id, session.identity.incarnation) !=
                    (self._current.episode_id, self._current.incarnation)):
                raise AgentReuseError("tool scope is no longer active")
            if time.monotonic() >= session.deadline or session.calls >= self.limits.max_calls:
                raise AgentLimitError("tool call/time budget exhausted")
            if call.call_id in session._ids:
                raise AgentReuseError("tool call ID was already executed")
            session.calls += 1
            session._ids.add(call.call_id)
            try:
                if call.name not in self._tools:
                    raise ToolError("Unknown tool. Use one of the declared tools.")
                spec, function = self._tools[call.name]
                try:
                    arguments = _arguments(call.arguments_json)
                    valid = _schema(spec.schema, arguments)
                except (ValueError, RecursionError):
                    valid = False
                if not valid:
                    raise ToolError("Arguments do not match the tool schema; correct them and try a new call ID.")
                parts = function(arguments, session)
                result = ToolResult(call.call_id, parts)
            except ToolError as error:
                result = ToolResult(call.call_id, (TextPart(_json({"error": str(error)[:256]})),), False)
            except Exception:
                result = ToolResult(call.call_id, (TextPart('{"error":"Tool operation failed; no automatic retry."}'),), False)
            text = sum(len(p.text.encode("utf-8")) for p in result.parts if isinstance(p, TextPart))
            images = [p for p in result.parts if isinstance(p, ImagePart)]
            image_bytes = sum(len(p.data) for p in images)
            if (text > self.limits.max_text_bytes or session.text_bytes + text > self.limits.max_total_text_bytes or
                    any(len(p.data) > self.limits.max_image_bytes for p in images) or
                    session.images + len(images) > self.limits.max_images or
                    session.image_bytes + image_bytes > self.limits.max_total_image_bytes):
                raise AgentLimitError("tool result budget exhausted")
            session.text_bytes += text
            session.images += len(images)
            session.image_bytes += image_bytes
            if time.monotonic() >= session.deadline:
                raise AgentLimitError("tool time budget exhausted after operation")
            return result


class FileWorkspace(ToolRegistry):
    """Private notes and immutable, explicitly published own-record snapshots."""
    def __init__(self, work_directory: str | Path, *, limits: ToolLimits | None = None) -> None:
        super().__init__(limits=limits)
        self._work_directory = Path(work_directory)
        self._root: Path | None = None
        self._workspace_id = uuid.uuid4().hex
        self._records: dict[str, tuple[RecordSnapshot, tuple[str, str, str], tuple[str, int]]] = {}
        self._published_bytes = 0
        self._notes: dict[tuple[str, str], tuple[int, str]] = {}
        path = {"type": "string", "maxLength": 512,
                "description": "Exact logical file path discovered with list_files; never an arbitrary OS path."}
        integer = lambda low, high, description: {"type": "integer", "minimum": low, "maximum": high, "description": description}
        self._builtin("list_files",
                      "Discover your virtual workspace: read-only records/<record_id>/ and writable notes/<name>. "
                      "Completed SimpleAgent decisions publish messages.json, assistant.txt, action.txt and images/0000.png "
                      "when present (image number/extension vary); the current decision is published only after completion. "
                      "Compaction removes old text/images from the working request, not published originals: listed images "
                      "can be reopened with view_image while in this agent case/scope and within tool limits. "
                      "Discover actual IDs/paths first, never guess them. Start with "
                      'list_files({"path":"","offset":0,"limit":20}). Returns flat file rows, not directory entries; '
                      "filter by logical prefix and continue with next_offset until null.",
                      {"path": {"type": "string", "maxLength": 512,
                                "description": "Use empty string or '.' for all files; 'records', 'notes' or a listed record root to filter. No trailing slash."},
                       "offset": integer(0, self.limits.max_records * self.limits.max_record_files + self.limits.max_note_files,
                                         "Start at 0; use the returned next_offset for the next page with the same path filter."),
                       "limit": integer(1, 100, "Maximum file rows per page (1-100).")}, self._list)
        self._builtin("read_file",
                      "Read listed UTF-8 text by 1-based lines, including records/.../messages.json, assistant.txt, "
                      "action.txt and your notes. Follow next_line until null for more text. messages.json image paths "
                      "are relative to that record root; concatenate text_chunks without separators. Use view_image "
                      "for images. If listed, example: "
                      'read_file({"path":"records/turn-000001/messages.json","start_line":1,"max_lines":100}).',
                      {"path": path,
                       "start_line": integer(1, 10000000, "First line to read (1-based); use returned next_line to continue."),
                       "max_lines": integer(1, 200, "Maximum lines to return; byte limits may shorten the page.")}, self._read)
        self._builtin("view_image",
                      "Reopen a listed PNG/JPEG/GIF/WebP as actual vision content, including an original image no longer "
                      "in the working request after compaction. Only published, in-scope records within this agent case "
                      "and tool image/call/byte limits are available; SimpleAgent.reset() or SimpleAgent.close() clears this workspace. "
                      'If listed, example: view_image({"path":"records/turn-000001/images/0000.png"}).',
                      {"path": path}, self._image)
        self._builtin("write_note",
                      "Create or replace your UTF-8 model memory at notes/<name>, one filename with no nested directories. "
                      "Notes are your interpretations, not environment facts; records/ is immutable. Example: "
                      'write_note({"path":"notes/plan.md","content":"Recheck the earlier reference image before acting."}).',
                      {"path": {"type": "string", "maxLength": 512,
                                "description": "Writable logical path notes/<name>, e.g. notes/plan.md; no nested directories or OS paths."},
                       "content": {"type": "string", "maxLength": self.limits.max_note_bytes,
                                   "description": "Complete replacement note text, subject to UTF-8 byte and workspace quotas."}}, self._write)

    def _builtin(self, name: str, description: str, properties: dict, function: Callable) -> None:
        self.register(ToolSpec(name, description, _json({"type": "object", "properties": properties,
                                                       "required": list(properties), "additionalProperties": False})), function)

    def publish(self, snapshot: RecordSnapshot, identity: AgentInputIdentity) -> None:
        with self._lock:
            if not isinstance(snapshot, RecordSnapshot) or not isinstance(identity, AgentInputIdentity):
                raise TypeError("publish requires a snapshot and its own agent identity")
            binding = (identity.run_id, identity.session_id, identity.agent_id)
            if self._binding is not None and binding != self._binding:
                raise AgentReuseError("record belongs to another agent lifetime")
            prior = self._records.get(snapshot.record_id)
            item = (snapshot, binding, (identity.episode_id, identity.incarnation))
            if prior is not None:
                if prior != item:
                    raise AgentReuseError("record ID is already bound to different bytes or scope")
                return
            size = sum(len(data) for _, data in snapshot.files)
            if (len(snapshot.files) > self.limits.max_record_files or size > self.limits.max_record_bytes or
                    any(len(data) > self.limits.max_record_file_bytes for _, data in snapshot.files) or
                    len(self._records) >= self.limits.max_records or self._published_bytes + size > self.limits.max_published_bytes):
                raise AgentLimitError("published record capacity exhausted")
            if self._records and any(value[1] != binding for value in self._records.values()):
                raise AgentReuseError("published records cannot mix agent lifetimes")
            self._records[snapshot.record_id] = item
            self._published_bytes += size

    def accept_input(self, request: AgentRequest) -> None:
        with self._lock:
            binding = (request.identity.run_id, request.identity.session_id, request.identity.agent_id)
            if any(value[1] != binding for value in self._records.values()):
                raise AgentReuseError("input is outside the published record owner")
            super().accept_input(request)

    def begin(self, request: AgentRequest) -> ToolSession:
        with self._lock:
            session = super().begin(request)
            session.records = MappingProxyType({key: value[0] for key, value in self._records.items()
                                                if value[1] == self._binding and value[2] in self._scopes})
            return session

    def _note_root(self, session: ToolSession) -> Path:
        if self._root is None:
            parent = _root(self._work_directory)
            self._root = parent / ("agentark-notes-" + self._workspace_id)
            self._root.mkdir()
        _root(self._root)
        directory = self._root / session.namespace
        directory.mkdir(exist_ok=True)
        _safe_stat(directory, directory=True)
        return directory

    def _note(self, parts: tuple[str, ...], session: ToolSession) -> tuple[Path, str]:
        if len(parts) != 2 or parts[0] != "notes":
            raise ToolError("Notes use notes/name.md; nested note directories are not supported.")
        return self._note_root(session) / parts[1], parts[1]

    def _data(self, path: str, session: ToolSession) -> tuple[bytes, str]:
        parts = _path(path)
        if parts[0] == "records" and len(parts) >= 3:
            record = session.records.get(parts[1])
            if record is not None:
                for name, data in record.files:
                    if name == "/".join(parts[2:]):
                        return data, record.digest
            raise ToolError("Record or file is not available in this tool scope.")
        target, name = self._note(parts, session)
        metadata = self._notes.get((session.namespace, name))
        if metadata is None:
            raise ToolError("Note does not exist in this workspace generation.")
        data = _read_regular(target, self.limits.max_note_bytes)
        if (len(data), sha256(data).hexdigest()) != metadata:
            raise ToolError("Note changed outside the controlled tool.")
        return data, metadata[1]

    def _list(self, args: dict, session: ToolSession) -> tuple[TextPart, ...]:
        prefix = "/".join(_path(args["path"], empty=True))
        rows = [{"path": "records/" + key + "/" + name, "bytes": len(data), "record_id": key,
                 "record_digest": record.digest, "sha256": sha256(data).hexdigest(), "read_only": True}
                for key, record in session.records.items() for name, data in record.files]
        rows += [{"path": "notes/" + name, "bytes": size, "sha256": digest, "read_only": False}
                 for (namespace, name), (size, digest) in self._notes.items() if namespace == session.namespace]
        rows = sorted((row for row in rows if not prefix or row["path"] == prefix or row["path"].startswith(prefix + "/")), key=lambda row: row["path"])
        start, limit = args["offset"], args["limit"]
        return (TextPart(_json({"files": rows[start:start + limit], "next_offset": start + limit if start + limit < len(rows) else None})),)

    def _read(self, args: dict, session: ToolSession) -> tuple[TextPart, ...]:
        data, digest = self._data(args["path"], session)
        try:
            text = data.decode("utf-8")
        except UnicodeError:
            raise ToolError("File is not UTF-8 text. Use view_image for images.") from None
        lines = text.splitlines(keepends=True)
        start = args["start_line"] - 1
        selected: list[str] = []
        size = 0
        # JSON escaping may expand text by 6x; reserve metadata space as well.
        bound = min(self.limits.max_text_bytes, self.limits.max_total_text_bytes - session.text_bytes)
        for line in lines[start:start + args["max_lines"]]:
            addition = len(_json(line).encode("utf-8"))
            if size + addition + 1024 > bound:
                if not selected:
                    raise ToolError("Single line exceeds the read limit; publish smaller structured records.")
                break
            selected.append(line)
            size += addition
        end = start + len(selected)
        return (TextPart(_json({"path": args["path"], "digest": digest, "start_line": start + 1,
                               "text": "".join(selected), "next_line": end + 1 if end < len(lines) else None})),)

    def _image(self, args: dict, session: ToolSession) -> tuple[TextPart | ImagePart, ...]:
        data, digest = self._data(args["path"], session)
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif data.startswith((b"GIF87a", b"GIF89a")):
            mime = "image/gif"
        elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            raise ToolError("File is not a supported image.")
        if len(data) > self.limits.max_image_bytes:
            raise ToolError("Image exceeds the per-image byte limit.")
        return (TextPart(_json({"path": args["path"], "digest": digest})), ImagePart(data, mime))

    def _write(self, args: dict, session: ToolSession) -> tuple[TextPart, ...]:
        parts = _path(args["path"])
        data = args["content"].encode("utf-8")
        if len(data) > self.limits.max_note_bytes:
            raise ToolError("Note exceeds its UTF-8 byte limit.")
        target, name = self._note(parts, session)
        key = (session.namespace, name)
        prior = self._notes.get(key)
        if prior is None and len(self._notes) >= self.limits.max_note_files:
            raise ToolError("Note file count exhausted, including earlier generations.")
        if target.exists() or target.is_symlink():
            actual = _read_regular(target, self.limits.max_note_bytes)
            if prior is None or (len(actual), sha256(actual).hexdigest()) != prior:
                raise ToolError("Existing note is not owned by this workspace.")
        elif prior is not None:
            raise ToolError("Owned note was removed outside the controlled tool.")
        total = sum(size for size, _ in self._notes.values()) - (prior[0] if prior else 0) + len(data)
        if total > self.limits.max_workspace_bytes:
            raise ToolError("Workspace quota exhausted, including earlier generations.")
        temporary = target.parent / (".write-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
            _safe_stat(target.parent, directory=True)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        digest = sha256(data).hexdigest()
        self._notes[key] = len(data), digest
        return (TextPart(_json({"path": args["path"], "bytes": len(data), "sha256": digest, "model_memory": True})),)
