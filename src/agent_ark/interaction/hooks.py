from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol


class InteractionHook(Protocol):
    def start(self) -> None:
        ...

    def handle_event(self, event: Dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass
class HookError:
    hook: str
    event: str
    error_type: str
    error: str


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'type': 'bytes', 'size': len(value)}
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


class NoopHook:
    def start(self) -> None:
        return

    def handle_event(self, event: Dict[str, Any]) -> None:
        return

    def close(self) -> None:
        return


class HookManager:
    def __init__(self, hooks: Optional[Iterable[InteractionHook]] = None, *, strict: bool = False):
        self._hooks: List[InteractionHook] = list(hooks or [])
        self._strict = bool(strict)
        self._lock = threading.RLock()
        self._seq = 0
        self.errors: List[HookError] = []

    @property
    def enabled(self) -> bool:
        return bool(self._hooks)

    def add_hook(self, hook: InteractionHook) -> None:
        with self._lock:
            self._hooks.append(hook)

    def start(self) -> None:
        for hook in list(self._hooks):
            self._call_hook(hook, 'start')

    def close(self) -> None:
        for hook in reversed(list(self._hooks)):
            self._call_hook(hook, 'close')

    def emit(
        self,
        event: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        source: Optional[str] = None,
        phase: Optional[str] = None,
        **metadata: Any,
    ) -> Dict[str, Any]:
        if payload is None:
            payload = {}
        with self._lock:
            self._seq += 1
            seq = self._seq

        item: Dict[str, Any] = {
            'seq': seq,
            'time': time.time(),
            'event': str(event),
            'payload': to_jsonable(payload),
        }
        if source is not None:
            item['source'] = str(source)
        if phase is not None:
            item['phase'] = str(phase)
        if metadata:
            item['metadata'] = to_jsonable(metadata)

        for hook in list(self._hooks):
            self._call_hook(hook, 'handle_event', item)
        return item

    def _call_hook(self, hook: InteractionHook, method_name: str, *args: Any) -> None:
        try:
            method = getattr(hook, method_name)
            method(*args)
        except Exception as exc:
            error = HookError(
                hook=type(hook).__name__,
                event=str(args[0].get('event')) if args and isinstance(args[0], dict) else method_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self.errors.append(error)
            if self._strict:
                raise


def ensure_hook_manager(value: Any = None) -> HookManager:
    if isinstance(value, HookManager):
        return value
    if value is None:
        return HookManager()
    if isinstance(value, (list, tuple)):
        return HookManager(value)
    return HookManager([value])
