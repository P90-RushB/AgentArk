# SPDX-License-Identifier: Apache-2.0
# Adapted with permission from the ArkArena stdlib core (ark_arena/harness).
"""Independent, bounded agent state and caller-owned model adapters."""
from .encoding import chat_messages, encode_png, image_from_pixels
from .models import CallableModel, ChatCompletionsModel, ModelResponseError
from .runtime import SimpleRuntime, SimpleRuntimeConfig
from .tools import FileWorkspace, RecordSnapshot, ToolError, ToolLimits, ToolRegistry, ToolSession
from .types import (
    AgentRuntime, ArchiveEntry, AsyncModel, AgentError, AgentEvent,
    AgentInputIdentity, AgentLimitError, AgentRequest, AgentResponse,
    AgentReuseError, AgentSupersededError, AgentUsage, ImagePart, Message, ModelCompletion,
    ModelRequest, ModelUsage, TextPart, ToolCall, ToolResult, ToolSpec,
)

__all__ = [
    "FileWorkspace", "RecordSnapshot", "ToolError", "ToolLimits", "ToolRegistry", "ToolSession",
    "ToolCall", "ToolResult", "ToolSpec", "AgentRuntime", "ArchiveEntry", "AsyncModel",
    "CallableModel", "ChatCompletionsModel", "ModelResponseError", "AgentError", "AgentEvent",
    "AgentInputIdentity", "AgentLimitError", "AgentRequest", "AgentResponse", "AgentReuseError",
    "AgentSupersededError", "AgentUsage", "ImagePart", "Message", "ModelCompletion",
    "ModelRequest", "ModelUsage", "TextPart", "chat_messages", "encode_png", "image_from_pixels",
    "SimpleRuntime", "SimpleRuntimeConfig", "SimpleAgent",
]


def __getattr__(name: str):
    # The stdlib core can be imported without loading the provider/SDK adapters.
    if name == "SimpleAgent":
        from .agent import SimpleAgent
        return SimpleAgent
    raise AttributeError(name)
