"""Version-gated rollout-boundary cleanup for supported ms-swift releases.

The supported ms-swift releases do not expose a public trajectory-finally hook
around the colocate multi-turn rollout boundary.  AgentArk installs this
deliberately small compatibility patch so Python exceptions release live
leases promptly; the server-side lease TTL remains the final recovery
mechanism for process crashes and other failures that cannot execute
``finally``.
"""

from __future__ import annotations

import logging
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Optional, Type


logger = logging.getLogger(__name__)

SUPPORTED_MS_SWIFT_VERSIONS = frozenset({"4.4.1", "4.5.0.dev0"})
# Kept for downstream imports written against the original 4.4.1-only patch.
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
    """Install the compatibility wrapper once on the available trainer paths.

    Optional dependency injection keeps the patch testable without constructing
    a Swift trainer.  Production callers use the defaults and only pass the
    already-imported ``AgentArkScheduler`` class to avoid import-order surprises
    when this module is loaded as an external plugin.

    Returns ``True`` only when this call installed a new wrapper.
    """

    resolved_version = get_ms_swift_version() if detected_version is _VERSION_UNSET else detected_version
    if resolved_version not in SUPPORTED_MS_SWIFT_VERSIONS:
        logger.warning(
            "AgentArk rollout-boundary cleanup patch supports ms-swift versions %s; "
            "detected %s. AgentArk env/scheduler registration remains enabled, "
            "but lease cleanup now relies on normal scheduler finalization and lease TTL.",
            ", ".join(sorted(SUPPORTED_MS_SWIFT_VERSIONS)),
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
            "AgentArk could not import the ms-swift rollout patch targets; "
            "env/scheduler registration remains enabled and lease TTL remains active.",
            exc_info=True,
        )
        return False

    # The HF and Megatron trainers have separate mixins in 4.5.  Keep the
    # injection point singular for tests and callers that only need one path.
    trainer_mixins = [trainer_mixin_cls]
    if trainer_mixin_cls is not None and trainer_mixin_cls.__module__.startswith("swift.rlhf_trainers"):
        try:
            from swift.megatron.trainers.rollout_mixin import RolloutTrainerMixin as MegatronRolloutTrainerMixin
        except Exception:
            MegatronRolloutTrainerMixin = None
        if MegatronRolloutTrainerMixin is not None:
            trainer_mixins.append(MegatronRolloutTrainerMixin)

    installed = False
    for mixin in trainer_mixins:
        current = getattr(mixin, "_infer_single_or_multi_turn", None)
        if current is None or getattr(current, PATCH_SENTINEL, False):
            continue

        original = current

        @wraps(original)
        def _agentark_infer_with_cleanup(self: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            try:
                return _original(self, *args, **kwargs)
            finally:
                cleanup_coro = None
                try:
                    scheduler = getattr(self, "multi_turn_scheduler", None)
                    if isinstance(scheduler, agentark_scheduler_cls):
                        cleanup_coro = scheduler.finalize_all(reason="rollout_boundary")
                        invoke_async_hook_fn(cleanup_coro)
                except BaseException:
                    # Cleanup is best effort. Never replace the rollout's
                    # return value or the exception from the original method.
                    if cleanup_coro is not None:
                        _close_unawaited(cleanup_coro)
                    logger.exception("AgentArk rollout-boundary cleanup failed")

        setattr(_agentark_infer_with_cleanup, PATCH_SENTINEL, True)
        setattr(_agentark_infer_with_cleanup, ORIGINAL_METHOD_ATTR, original)
        mixin._infer_single_or_multi_turn = _agentark_infer_with_cleanup
        installed = True

    return installed


__all__ = [
    "ORIGINAL_METHOD_ATTR",
    "PATCH_SENTINEL",
    "SUPPORTED_MS_SWIFT_VERSION",
    "SUPPORTED_MS_SWIFT_VERSIONS",
    "get_ms_swift_version",
    "install_rollout_cleanup_patch",
]
