"""Registration side effects loaded by ms-swift's ``--external_plugins``."""

from swift.rollout.gym_env import envs
from swift.rollout.multi_turn import multi_turns

# ms-swift imports an external plugin file as a top-level ``plugin`` module,
# rather than as ``agentark_swift.plugin``. Absolute imports therefore work in
# both external-file and normal package import modes.
from agentark_swift.env import AgentArkEnv
from agentark_swift.rollout_cleanup import install_rollout_cleanup_patch
from agentark_swift.scheduler import AgentArkScheduler

ENV_NAME = "agentark"
SCHEDULER_NAME = "agentark_scheduler"


def register() -> None:
    envs[ENV_NAME] = AgentArkEnv
    multi_turns[SCHEDULER_NAME] = AgentArkScheduler
    install_rollout_cleanup_patch(agentark_scheduler_cls=AgentArkScheduler)


register()

__all__ = ["ENV_NAME", "SCHEDULER_NAME", "register"]
