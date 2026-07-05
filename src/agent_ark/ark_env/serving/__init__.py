"""HTTP serving layer for AgentArk environments.

This subpackage exposes an AgentArk env over a framework-agnostic HTTP API
(``acquire_start`` / ``step`` / ``release``) plus the supporting pieces:

- ``protocol``       : request/response payloads + obs (de)serialization
- ``session_manager``: env pool, leasing, self-healing (timeouts/discard/recreate)
- ``env_server``     : FastAPI app
- ``env_client``     : thin HTTP client
- ``task_selector``  : deterministic task selection from an RL group id
- ``warmup_envs``    : pre-create/warm a pool of envs
- ``run_server``     : server entrypoint

It is intentionally independent of any specific RL framework; the env core lives
in the parent ``ark_env`` package.
"""

from agent_ark.ark_env.serving.env_client import EnvHttpClient
from agent_ark.ark_env.serving.protocol import (
    EnvStartPayload,
    EnvStepPayload,
    as_json_dict,
    decode_obs,
    encode_obs,
)
from agent_ark.ark_env.serving.session_manager import EnvRuntime, EnvSessionManager
from agent_ark.ark_env.serving.task_selector import (
    HashTaskSelector,
    TaskSelector,
    get_default_selector,
    resolve_task_for_group,
)

__all__ = [
    "EnvHttpClient",
    "EnvStartPayload",
    "EnvStepPayload",
    "as_json_dict",
    "decode_obs",
    "encode_obs",
    "EnvRuntime",
    "EnvSessionManager",
    "HashTaskSelector",
    "TaskSelector",
    "get_default_selector",
    "resolve_task_for_group",
]
