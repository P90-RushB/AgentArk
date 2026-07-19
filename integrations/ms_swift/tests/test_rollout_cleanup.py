from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_ROOT / "src"))

from agentark_swift.rollout_cleanup import (  # noqa: E402
    PATCH_SENTINEL,
    install_rollout_cleanup_patch,
)


class OriginalRolloutError(RuntimeError):
    pass


class FakeAgentArkScheduler:
    def __init__(self, *, cleanup_error: BaseException | None = None) -> None:
        self.cleanup_error = cleanup_error
        self.finalize_reasons: list[str] = []

    async def finalize_all(self, *, reason: str) -> None:
        self.finalize_reasons.append(reason)
        if self.cleanup_error is not None:
            raise self.cleanup_error


class OtherScheduler:
    def __init__(self) -> None:
        self.finalize_reasons: list[str] = []

    async def finalize_all(self, *, reason: str) -> None:
        self.finalize_reasons.append(reason)


def invoke_async_hook(coro: Any) -> Any:
    return asyncio.run(coro)


def install_for(trainer_cls: type[Any]) -> bool:
    return install_rollout_cleanup_patch(
        detected_version="4.4.1",
        trainer_mixin_cls=trainer_cls,
        invoke_async_hook_fn=invoke_async_hook,
        agentark_scheduler_cls=FakeAgentArkScheduler,
    )


class RolloutCleanupPatchTest(unittest.TestCase):
    def test_install_is_idempotent(self) -> None:
        class Trainer:
            def __init__(self) -> None:
                self.multi_turn_scheduler = FakeAgentArkScheduler()
                self.original_calls = 0

            def _infer_single_or_multi_turn(self, value: int) -> int:
                self.original_calls += 1
                return value * 2

        self.assertTrue(install_for(Trainer))
        wrapped = Trainer._infer_single_or_multi_turn
        self.assertTrue(getattr(wrapped, PATCH_SENTINEL))
        self.assertFalse(install_for(Trainer))
        self.assertIs(Trainer._infer_single_or_multi_turn, wrapped)

        trainer = Trainer()
        self.assertEqual(trainer._infer_single_or_multi_turn(4), 8)
        self.assertEqual(trainer.original_calls, 1)
        self.assertEqual(trainer.multi_turn_scheduler.finalize_reasons, ["rollout_boundary"])

    def test_normal_original_result_is_preserved_and_finalized(self) -> None:
        result = object()

        class Trainer:
            def __init__(self) -> None:
                self.multi_turn_scheduler = FakeAgentArkScheduler()

            def _infer_single_or_multi_turn(self) -> object:
                return result

        self.assertTrue(install_for(Trainer))
        trainer = Trainer()

        self.assertIs(trainer._infer_single_or_multi_turn(), result)
        self.assertEqual(trainer.multi_turn_scheduler.finalize_reasons, ["rollout_boundary"])

    def test_original_exception_is_preserved_and_finalized(self) -> None:
        original_error = OriginalRolloutError("rollout failed")

        class Trainer:
            def __init__(self) -> None:
                self.multi_turn_scheduler = FakeAgentArkScheduler()

            def _infer_single_or_multi_turn(self) -> None:
                raise original_error

        self.assertTrue(install_for(Trainer))
        trainer = Trainer()

        with self.assertRaises(OriginalRolloutError) as caught:
            trainer._infer_single_or_multi_turn()
        self.assertIs(caught.exception, original_error)
        self.assertEqual(trainer.multi_turn_scheduler.finalize_reasons, ["rollout_boundary"])

    def test_cleanup_exception_cannot_replace_result_or_original_exception(self) -> None:
        original_error = OriginalRolloutError("original rollout error")

        class Trainer:
            def __init__(self) -> None:
                self.multi_turn_scheduler = FakeAgentArkScheduler(
                    cleanup_error=RuntimeError("cleanup failed")
                )

            def _infer_single_or_multi_turn(self, *, fail: bool) -> str:
                if fail:
                    raise original_error
                return "normal-result"

        self.assertTrue(install_for(Trainer))
        trainer = Trainer()

        with self.assertLogs("agentark_swift.rollout_cleanup", level="ERROR"):
            self.assertEqual(
                trainer._infer_single_or_multi_turn(fail=False),
                "normal-result",
            )
        with self.assertLogs("agentark_swift.rollout_cleanup", level="ERROR"):
            with self.assertRaises(OriginalRolloutError) as caught:
                trainer._infer_single_or_multi_turn(fail=True)

        self.assertIs(caught.exception, original_error)
        self.assertEqual(
            trainer.multi_turn_scheduler.finalize_reasons,
            ["rollout_boundary", "rollout_boundary"],
        )

    def test_non_agentark_scheduler_is_not_finalized(self) -> None:
        class Trainer:
            def __init__(self) -> None:
                self.multi_turn_scheduler = OtherScheduler()

            def _infer_single_or_multi_turn(self) -> str:
                return "result"

        self.assertTrue(install_for(Trainer))
        trainer = Trainer()

        self.assertEqual(trainer._infer_single_or_multi_turn(), "result")
        self.assertEqual(trainer.multi_turn_scheduler.finalize_reasons, [])

    def test_unsupported_version_warns_without_patching(self) -> None:
        class Trainer:
            def _infer_single_or_multi_turn(self) -> str:
                return "unpatched"

        original = Trainer._infer_single_or_multi_turn
        with self.assertLogs("agentark_swift.rollout_cleanup", level="WARNING") as logs:
            installed = install_rollout_cleanup_patch(
                detected_version="4.5.0",
                trainer_mixin_cls=Trainer,
                invoke_async_hook_fn=invoke_async_hook,
                agentark_scheduler_cls=FakeAgentArkScheduler,
            )

        self.assertFalse(installed)
        self.assertIs(Trainer._infer_single_or_multi_turn, original)
        self.assertIn("requires ms-swift==4.4.1", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
