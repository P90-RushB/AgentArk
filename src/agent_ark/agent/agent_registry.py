"""Explicit agent selection, separate from the model's HTTP provider.

Existing configurations keep their provider-based API/Codex selection. New
agents register their own module and constructor options; the evaluator does
not need to know how they manage context or execute tools.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentType:
    name: str
    module: str
    class_name: str
    backend: str = "api"
    options_key: str | None = None
    options_keyword: str | None = None
    message_mode: str | None = None

    def load_class(self) -> type:
        return getattr(import_module(self.module), self.class_name)


_AGENT_TYPES: dict[str, AgentType] = {}


def register_agent_type(spec: AgentType) -> None:
    """Register trusted Python code, never arbitrary imports from eval YAML."""
    if not isinstance(spec, AgentType):
        raise TypeError("agent registration must be an AgentType")
    if not spec.name or spec.name != spec.name.strip().lower():
        raise ValueError("agent type name must be a nonempty lowercase string")
    if spec.backend not in ("api", "codex"):
        raise ValueError("agent backend must be api or codex")
    if bool(spec.options_key) != bool(spec.options_keyword):
        raise ValueError("agent options key and keyword must be set together")
    if spec.message_mode not in (None, "delta", "full"):
        raise ValueError("agent message mode must be delta or full")
    if spec.name in _AGENT_TYPES:
        raise ValueError(f"agent type is already registered: {spec.name}")
    _AGENT_TYPES[spec.name] = spec


for _spec in (
    AgentType("api", "agent_ark.agent.api_agent", "APIAgent"),
    AgentType("codex", "agent_ark.agent.codex_agent", "CodexAgent", backend="codex"),
    AgentType(
        "simple", "agent_ark.agent.simple_agent", "SimpleAgent",
        options_key="simple_agent", options_keyword="simple_agent_config",
        message_mode="delta",
    ),
):
    register_agent_type(_spec)


def resolve_agent_type(model_cfg: Mapping[str, Any]) -> AgentType:
    provider = str(model_cfg.get("provider", "auto") or "auto").strip().lower()
    raw = model_cfg.get("agent_type")
    name = ("codex" if provider == "codex" else "api") if raw is None else str(raw).strip().lower()
    try:
        spec = _AGENT_TYPES[name]
    except KeyError as exc:
        raise ValueError(f"unknown agent_type: {name!r}; available: {', '.join(_AGENT_TYPES)}") from exc
    if provider == "codex" and spec.backend != "codex" and name != "simple":
        raise ValueError(f"agent_type: {name} requires an HTTP model provider, not provider: codex")
    if spec.backend == "codex" and provider not in ("codex", "auto"):
        raise ValueError("agent_type: codex requires provider: codex (or auto)")
    if "simple_agent" in model_cfg and name != "simple":
        raise ValueError("simple_agent options require agent_type: simple")
    return spec


def agent_constructor_options(spec: AgentType, model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if spec.options_key is None:
        return {}
    options = model_cfg.get(spec.options_key, {})
    if not isinstance(options, dict):
        raise ValueError(f"{spec.options_key} must be an object")
    return {str(spec.options_keyword): deepcopy(options)}


def configure_agent_environment(env_cfg: dict[str, Any], model_runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one agent's message contract to its private environment config.

    API and Codex defaults deliberately retain the existing caller's settings.
    Selecting SimpleAgent supplies the supported incremental message contract
    without changing task files or configurations used by any other agent.
    """
    spec = resolve_agent_type(model_runtime)
    if spec.message_mode is None:
        return env_cfg
    out = deepcopy(env_cfg)
    overrides = out.setdefault("env_config_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("env_config_overrides must be an object")
    wrapper = overrides.setdefault("env_wrapper_cfg", {})
    if not isinstance(wrapper, dict):
        raise ValueError("env_wrapper_cfg must be an object")
    context = wrapper.setdefault("context_manager", {})
    if not isinstance(context, dict):
        raise ValueError("context_manager must be an object")
    messages = context.setdefault("messages", {})
    if not isinstance(messages, dict):
        raise ValueError("context_manager.messages must be an object")
    messages.update(enabled=True, only_return_messages=True, append_only=True, return_mode=spec.message_mode)
    return out
