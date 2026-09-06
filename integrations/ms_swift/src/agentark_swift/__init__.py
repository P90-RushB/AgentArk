"""Legacy external ms-swift integration for AgentArk.

Import :mod:`agentark_swift.plugin` from ms-swift's ``--external_plugins``
option to register the environment and scheduler when Swift does not provide
the built-in AgentArk integration.
"""

from .client import AgentArkHttpClient, AgentArkHttpError, AgentArkStaleLeaseError
from .env import AgentArkEnv
from .scheduler import AgentArkScheduler

__all__ = [
    "AgentArkEnv",
    "AgentArkHttpClient",
    "AgentArkHttpError",
    "AgentArkStaleLeaseError",
    "AgentArkScheduler",
]
