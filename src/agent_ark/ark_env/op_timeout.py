"""Hard-timeout helper for env operations that can hang.

The underlying Unity runtime calls (``env.step`` / ``env.reset`` / ``env.close``)
have no internal timeout. If Unity stops responding, those calls block forever,
which would otherwise wedge an ``EnvRuntime`` (it holds a lock) and, through the
FastAPI handler, the HTTP request. ``run_with_timeout`` runs the blocking call on
a worker thread and raises ``OperationTimeout`` if it does not finish in time, so
the caller can treat the env as broken and rebuild it.

Note: the timed-out worker thread is NOT killed (Python cannot safely kill a
thread blocked in a C extension). The contract is that the caller discards the
whole env/process afterwards (close + recreate, ultimately killing the Unity
subprocess), so the orphaned thread dies with that process teardown.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class OperationTimeout(TimeoutError):
    """Raised when a guarded env operation exceeds its time budget."""


def run_with_timeout(fn: Callable[..., T], *args: Any, timeout_s: float, op: str = "op", **kwargs: Any) -> T:
    """Run ``fn(*args, **kwargs)`` on a worker thread, enforcing ``timeout_s``.

    Returns the result if it completes in time, otherwise raises
    ``OperationTimeout``. A non-positive ``timeout_s`` disables the guard and runs
    the call inline.
    """
    if timeout_s is None or timeout_s <= 0:
        return fn(*args, **kwargs)

    # One-shot executor: a dedicated thread per guarded call. We deliberately do
    # not reuse a pool because a hung call leaves its thread occupied for the
    # lifetime of the (about-to-be-destroyed) env.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"envguard-{op}")
    future = executor.submit(fn, *args, **kwargs)
    try:
        result = future.result(timeout=timeout_s)
    except _FuturesTimeout as e:
        raise OperationTimeout(f"env operation '{op}' timed out after {timeout_s}s") from e
    finally:
        # Do not block on shutdown: if the call is still running (timed out), we
        # must not wait for it. The thread will be reclaimed when the env process
        # is torn down.
        executor.shutdown(wait=False)
    return result
