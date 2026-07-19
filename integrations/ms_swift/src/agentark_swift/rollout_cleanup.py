"""Version-gated rollout-boundary cleanup for ms-swift 4.4.1.

ms-swift 4.4.1 does not expose a public trajectory-finally hook around the
colocate multi-turn rollout boundary.  AgentArk installs this deliberately
small compatibility patch so Python exceptions release live leases promptly;
the server-side lease TTL remains the final recovery mechanism for process
crashes and other failures that cannot execute ``finally``.
"""

from __future__ import annotations

import logging
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Optional, Type


logger = logging.getLogger(__name__)

SUPPORTED_MS_SWIFT_VERSION = "4.4.1"
PATCH_SENTINEL = "__agentark_rollout_boundary_cleanup_patch__"
ORIGINAL_METHOD_ATTR = "__agentark_original_infer_single_or_multi_turn__"
_VERSION_UNSET = object()


def get_ms_swift_version() -> Optional[str]:
    """Return the installed distribution version without importing trainer code."""

    try:
        return version("ms-swift")
    except PackageNotFoundError:
        return None


def _close_unawaited(coro: Any) -> None:
    close = getattr(coro, "close", None)
    if callable(close):
        try:
            close()
        except BaseException:
            logger.debug("Failed to close AgentArk cleanup coroutine", exc_info=True)


def install_rollout_cleanup_patch(
    *,
    detected_version: object = _VERSION_UNSET,
    trainer_mixin_cls: Optional[Type[Any]] = None,
    invoke_async_hook_fn: Optional[Callable[[Any], Any]] = None,
    agentark_scheduler_cls: Optional[Type[Any]] = None,
) -> bool:
    """Install the 4.4.1 compatibility wrapper once.

    Optional dependency injection keeps the patch testable without constructing
    a Swift trainer.  Production callers use the defaults and only pass the
    already-imported ``AgentArkScheduler`` class to avoid import-order surprises
    when this module is loaded as an external plugin.

    Returns ``True`` only when this call installed a new wrapper.
    """

    resolved_version = get_ms_swift_version() if detected_version is _VERSION_UNSET else detected_version
    if resolved_version != SUPPORTED_MS_SWIFT_VERSION:
        logger.warning(
            "AgentArk rollout-boundary cleanup patch requires ms-swift==%s; "
            "detected %s. AgentArk env/scheduler registration remains enabled, "
            "but lease cleanup now relies on normal scheduler finalization and lease TTL.",
            SUPPORTED_MS_SWIFT_VERSION,
            resolved_version or "not installed",
        )
        return False

    try:
        if trainer_mixin_cls is None:
            from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin

            trainer_mixin_cls = RolloutTrainerMixin
        if invoke_async_hook_fn is None:
            from swift.rollout import invoke_async_hook

            invoke_async_hook_fn = invoke_async_hook
        if agentark_scheduler_cls is None:
            from agentark_swift.scheduler import AgentArkScheduler

            agentark_scheduler_cls = AgentArkScheduler
    except Exception:
        logger.warning(
            "AgentArk could not import the ms-swift 4.4.1 rollout patch targets; "
            "env/scheduler registration remains enabled and lease TTL remains active.",
            exc_info=True,
        )
        return False

    current = trainer_mixin_cls._infer_single_or_multi_turn
    if getattr(current, PATCH_SENTINEL, False):
        return False

    original = current

    @wraps(original)
    def _agentark_infer_with_cleanup(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original(self, *args, **kwargs)
        finally:
            cleanup_coro = None
            try:
                scheduler = getattr(self, "multi_turn_scheduler", None)
                if isinstance(scheduler, agentark_scheduler_cls):
                    cleanup_coro = scheduler.finalize_all(reason="rollout_boundary")
                    invoke_async_hook_fn(cleanup_coro)
            except BaseException:
                # Cleanup is best effort. Never replace the rollout's return
                # value or the exception already propagating from ``original``.
                if cleanup_coro is not None:
                    _close_unawaited(cleanup_coro)
                logger.exception("AgentArk rollout-boundary cleanup failed")

    setattr(_agentark_infer_with_cleanup, PATCH_SENTINEL, True)
    setattr(_agentark_infer_with_cleanup, ORIGINAL_METHOD_ATTR, original)
    trainer_mixin_cls._infer_single_or_multi_turn = _agentark_infer_with_cleanup
    return True


__all__ = [
    "ORIGINAL_METHOD_ATTR",
    "PATCH_SENTINEL",
    "SUPPORTED_MS_SWIFT_VERSION",
    "get_ms_swift_version",
    "install_rollout_cleanup_patch",
]
