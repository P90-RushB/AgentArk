"""State objects and errors for the AgentArk env-server lease protocol v2."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


class LeaseProtocolError(RuntimeError):
    """Base error carrying the HTTP semantics exposed by the v2 API."""

    status_code = 409
    code = "lease_protocol_error"
    retryable = False

    def as_detail(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "retryable": bool(self.retryable),
        }


class LeaseNotFound(LeaseProtocolError):
    status_code = 404
    code = "lease_not_found"


class LeaseConflict(LeaseProtocolError):
    status_code = 409
    code = "lease_conflict"


class IdempotencyConflict(LeaseProtocolError):
    status_code = 409
    code = "idempotency_conflict"


class LeaseOperationInProgress(LeaseProtocolError):
    status_code = 409
    code = "operation_in_progress"
    retryable = True


class LeaseGone(LeaseProtocolError):
    status_code = 410
    code = "lease_gone"


class IdempotencyResultGone(LeaseProtocolError):
    status_code = 410
    code = "idempotency_result_gone"


class CachedOperationFailure(LeaseProtocolError):
    status_code = 409
    code = "cached_operation_failure"


def _canonical_json_value(value: Any) -> Any:
    """Normalize JSON-like values before sorting mapping keys.

    History snapshots may originate in Python with integer agent ids while the
    same request, after an HTTP round trip, contains string keys. Treat both as
    the same JSON payload and avoid ``sort_keys`` comparing ``int`` and ``str``.
    """

    if isinstance(value, Mapping):
        return {str(key): _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return value


def request_fingerprint(payload: Dict[str, Any]) -> str:
    """Return a stable digest for an idempotent JSON request payload."""

    canonical = json.dumps(
        _canonical_json_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: f"<{type(value).__module__}.{type(value).__qualname__}:{value!r}>",
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LeaseIdentity:
    env_id: str
    generation: int
    token: str

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.env_id, self.generation, self.token)


@dataclass
class StepReplay:
    fingerprint: str
    state: str = "in_progress"
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    updated_at: float = 0.0


@dataclass
class HeartbeatReplay:
    fingerprint: str
    response: Dict[str, Any]
    updated_at: float


@dataclass
class LeaseRecord:
    identity: LeaseIdentity
    acquire_request_id: str
    acquire_fingerprint: str
    created_at: float
    touched_at: float
    expires_at: float
    active_operations: int = 0
    expire_pending: bool = False
    next_turn_index: int = 1
    acquire_response: Optional[Dict[str, Any]] = None
    step_replays: "OrderedDict[str, StepReplay]" = field(default_factory=OrderedDict)
    heartbeat_replays: "OrderedDict[str, HeartbeatReplay]" = field(default_factory=OrderedDict)


@dataclass
class AcquireReplay:
    fingerprint: str
    state: str
    updated_at: float
    identity: Optional[LeaseIdentity] = None
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class LeaseTombstone:
    identity: LeaseIdentity
    state: str
    updated_at: float
    acquire_request_id: str
    step_replays: "OrderedDict[str, StepReplay]" = field(default_factory=OrderedDict)
    release_response: Optional[Dict[str, Any]] = None
    release_request_id: Optional[str] = None
    error: Optional[str] = None
