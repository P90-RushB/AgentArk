# SPDX-License-Identifier: Apache-2.0
# Adapted with permission from the ArkArena stdlib core (ark_arena/harness).
"""Immutable, environment-neutral inputs for private agent agents."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Literal, Protocol, runtime_checkable


class AgentError(RuntimeError):
    """A bounded, local runtime failure; it grants no action authority."""


class AgentLimitError(AgentError):
    pass


class AgentReuseError(AgentError):
    pass


class AgentSupersededError(AgentError):
    """New accepted input superseded this completion; no action call was made."""



def _text(value: str, name: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"{name} must be a string")


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str

    def __post_init__(self) -> None:
        _text(self.text, "text", empty=True)


@dataclass(frozen=True, slots=True)
class ImagePart:
    data: bytes
    mime_type: str = "image/png"
    origin: Literal["top_left", "bottom_left"] = "top_left"
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("image data must be nonempty immutable bytes")
        if self.mime_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            raise ValueError("unsupported image MIME type")
        if self.origin not in ("top_left", "bottom_left"):
            raise ValueError("unsupported image origin")
        for value in (self.width, self.height):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("image dimensions must be positive integers")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", self.name):
            raise ValueError("invalid tool name")
        _text(self.description, "tool description")
        _text(self.parameters_json, "tool parameter schema")
        if len(self.description) > 4096 or len(self.parameters_json.encode("utf-8")) > 16384:
            raise ValueError("tool declaration exceeds its bound")
        if type(self.schema) is not dict:
            raise ValueError("tool parameters must be a JSON object")

    @property
    def schema(self) -> dict:
        return json.loads(self.parameters_json)


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        for name in ("call_id", "name", "arguments_json"):
            _text(getattr(self, name), name, empty=name == "arguments_json")
        if len(self.call_id) > 256 or len(self.name) > 64 or len(self.arguments_json.encode("utf-8")) > 131072:
            raise ValueError("tool call exceeds its bound")


@dataclass(frozen=True, slots=True)
class NativeAssistantState:
    """Detached API-only assistant replay, never ordinary model-visible text.

    This is the fixed OpenRouter Chat Completions contract, not arbitrary
    provider kwargs. JSON preserves null, whitespace, opaque blocks and order.
    The endpoint/model scope prevents replay through a different backend.
    """
    scope: str
    message_json: str

    def __post_init__(self) -> None:
        _text(self.scope, "native assistant scope")
        _text(self.message_json, "native assistant JSON")
        if len(self.scope) > 256 or len(self.message_json.encode("utf-8")) > 1024 * 1024:
            raise ValueError("native assistant state exceeds its bound")

        def pairs(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise ValueError("duplicate native assistant fields")
                value[key] = item
            return value

        def constant(value):
            raise ValueError("nonfinite native assistant JSON")

        try:
            value = json.loads(self.message_json, object_pairs_hook=pairs, parse_constant=constant)
            # Also reject overflowing numeric literals, e.g. 1e999.
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if (not isinstance(value, dict) or value.get("role") != "assistant"
                or "content" not in value
                or set(value) - {"role", "content", "tool_calls", "reasoning", "reasoning_details"}):
                raise ValueError("invalid native assistant fields")
            if value["content"] is not None and not isinstance(value["content"], (str, list)):
                raise ValueError("invalid native assistant content")
            if isinstance(value["content"], list) and any(
                not isinstance(part, dict) or not isinstance(part.get("text"), str)
                or part.get("type", "text") != "text" or set(part) - {"type", "text"}
                for part in value["content"]
            ):
                raise ValueError("native assistant content must contain only text parts")
            if ("reasoning" in value) == ("reasoning_details" in value):
                raise ValueError("native assistant requires one reasoning representation")
            if "reasoning_details" in value:
                details = value["reasoning_details"]
                if not isinstance(details, list) or not details or any(not isinstance(part, dict) for part in details):
                    raise ValueError("invalid native reasoning details")
            elif not isinstance(value["reasoning"], str) or not value["reasoning"]:
                raise ValueError("invalid native reasoning text")
            self._calls(value)
        except (TypeError, RecursionError, OverflowError) as error:
            raise ValueError("invalid native assistant JSON") from error

    @staticmethod
    def _calls(value: dict) -> tuple[ToolCall, ...]:
        calls = value.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list) or len(calls) > 32:
            raise ValueError("invalid native assistant tool calls")
        parsed = []
        for call in calls:
            if (not isinstance(call, dict) or set(call) != {"id", "type", "function"}
                or call["type"] != "function" or not isinstance(call["function"], dict)
                or set(call["function"]) != {"name", "arguments"}):
                raise ValueError("invalid native assistant tool call")
            parsed.append(ToolCall(call["id"], call["function"]["name"], call["function"]["arguments"]))
        if len({call.call_id for call in parsed}) != len(parsed):
            raise ValueError("duplicate native assistant tool call ID")
        return tuple(parsed)

    @property
    def message(self) -> dict:
        return json.loads(self.message_json)

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self._calls(self.message)

    @property
    def content_text(self) -> str:
        """The portable text projection; the API still replays raw content."""
        content = self.message["content"]
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        return "\n".join(part["text"].strip() for part in content if part["text"].strip())


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    parts: tuple[TextPart | ImagePart, ...]
    success: bool = True

    def __post_init__(self) -> None:
        _text(self.call_id, "tool result call ID")
        if type(self.success) is not bool or not isinstance(self.parts, tuple) or not self.parts or any(
            not isinstance(part, (TextPart, ImagePart)) for part in self.parts
        ):
            raise ValueError("invalid tool result")


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "developer", "user", "assistant", "tool"]
    parts: tuple[TextPart | ImagePart, ...]
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    native_assistant: NativeAssistantState | None = None

    def __post_init__(self) -> None:
        if self.role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError("unsupported message role")
        if not isinstance(self.parts, tuple) or (not self.parts and not self.tool_calls) or any(
            not isinstance(part, (TextPart, ImagePart)) for part in self.parts
        ):
            raise ValueError("message parts must be a nonempty immutable tuple")
        if not isinstance(self.tool_calls, tuple) or any(not isinstance(c, ToolCall) for c in self.tool_calls):
            raise ValueError("tool calls must be an immutable tuple")
        if self.tool_calls and (self.role != "assistant" or len({c.call_id for c in self.tool_calls}) != len(self.tool_calls)):
            raise ValueError("only assistant messages may contain unique tool calls")
        if self.role == "tool":
            _text(self.tool_call_id, "tool call ID")
        elif self.tool_call_id is not None:
            raise ValueError("only tool results may have a call ID")
        if self.native_assistant is not None:
            if (self.role != "assistant" or not isinstance(self.native_assistant, NativeAssistantState)
                or self.native_assistant.tool_calls != self.tool_calls):
                raise ValueError("native assistant state must match the assistant tool calls")
            if (any(not isinstance(part, TextPart) for part in self.parts)
                or "".join(part.text for part in self.parts) != self.native_assistant.content_text):
                raise ValueError("native assistant state must match the assistant content")

    @classmethod
    def text(cls, role: Literal["system", "developer", "user", "assistant"], text: str) -> Message:
        return cls(role, (TextPart(text),))


@dataclass(frozen=True, slots=True)
class AgentInputIdentity:
    run_id: str
    session_id: str
    episode_id: str
    agent_id: str
    incarnation: int
    input_id: str

    def __post_init__(self) -> None:
        for name in ("run_id", "session_id", "episode_id", "agent_id", "input_id"):
            _text(getattr(self, name), name)
        if type(self.incarnation) is not int or self.incarnation < 0:
            raise ValueError("incarnation must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class AgentRequest:
    identity: AgentInputIdentity
    messages: tuple[Message, ...]
    history_policy: Literal["carry", "omit"] = "omit"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, AgentInputIdentity):
            raise ValueError("identity must be a AgentInputIdentity")
        if not isinstance(self.messages, tuple) or not self.messages or any(
            not isinstance(message, Message) for message in self.messages
        ):
            raise ValueError("messages must be a nonempty immutable tuple")
        if self.history_policy not in ("carry", "omit"):
            raise ValueError("unsupported history policy")
        if any(m.role == "tool" or m.tool_calls or m.native_assistant is not None for m in self.messages):
            raise ValueError("environment input cannot supply runtime tool interactions")


@dataclass(frozen=True, slots=True)
class AgentEvent:
    identity: AgentInputIdentity
    event_id: str
    kind: str
    payload_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, AgentInputIdentity):
            raise ValueError("identity must be a AgentInputIdentity")
        _text(self.event_id, "event_id")
        _text(self.kind, "kind")
        _text(self.payload_json, "payload_json", empty=True)


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens, self.total_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("usage counts must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: tuple[Message, ...]
    purpose: Literal["action", "summary"] = "action"
    max_output_tokens: int = 4096
    input_tokens: int | None = None
    token_count_is_estimate: bool = True
    tools: tuple[ToolSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple) or any(not isinstance(m, Message) for m in self.messages):
            raise ValueError("messages must be an immutable tuple")
        if self.purpose not in ("action", "summary"):
            raise ValueError("unsupported model purpose")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("output token bound must be a positive integer")
        if self.input_tokens is not None and (type(self.input_tokens) is not int or self.input_tokens < 0):
            raise ValueError("input token count must be a nonnegative integer")
        if not isinstance(self.tools, tuple) or any(not isinstance(t, ToolSpec) for t in self.tools):
            raise ValueError("tools must be an immutable tuple")
        if len(self.tools) > 32 or len({t.name for t in self.tools}) != len(self.tools):
            raise ValueError("tool declarations must be bounded and unique")
        if self.purpose == "summary" and self.tools:
            raise ValueError("summary calls cannot expose tools")


@dataclass(frozen=True, slots=True)
class ModelCompletion:
    """A model result, optionally with a trusted adapter's closed context pair.

    ``context_messages`` records the native submission and the adapter's
    acknowledgement for future conversation. It grants no execution authority.
    Ordinary provider completions keep the empty default.
    """
    text: str
    usage: ModelUsage = ModelUsage()
    tool_calls: tuple[ToolCall, ...] = ()
    context_messages: tuple[Message, ...] = ()
    native_assistant: NativeAssistantState | None = None

    def __post_init__(self) -> None:
        _text(self.text, "completion text", empty=True)
        if not isinstance(self.usage, ModelUsage):
            raise ValueError("usage must be ModelUsage")
        if not isinstance(self.tool_calls, tuple) or len(self.tool_calls) > 32 or any(not isinstance(c, ToolCall) for c in self.tool_calls):
            raise ValueError("tool calls must be a bounded immutable tuple")
        if len({c.call_id for c in self.tool_calls}) != len(self.tool_calls):
            raise ValueError("duplicate completion tool call ID")
        if self.native_assistant is not None:
            if (not isinstance(self.native_assistant, NativeAssistantState) or self.context_messages
                or self.native_assistant.tool_calls != self.tool_calls):
                raise ValueError("native completion state must match the assistant tool calls")
            if self.text != self.native_assistant.content_text:
                raise ValueError("native completion state must match the assistant content")
        if not isinstance(self.context_messages, tuple) or any(
            not isinstance(message, Message) for message in self.context_messages
        ):
            raise ValueError("completion context must be an immutable message tuple")
        if self.context_messages:
            if self.tool_calls:
                raise ValueError("completion context cannot coexist with tool calls")
            if len(self.context_messages) != 2:
                raise ValueError("completion context must be a complete single call/result pair")
            assistant, result = self.context_messages
            if (assistant.role != "assistant" or len(assistant.tool_calls) != 1
                or result.role != "tool" or result.tool_calls
                or result.tool_call_id != assistant.tool_calls[0].call_id):
                raise ValueError("completion context must be a complete single call/result pair")


@dataclass(frozen=True, slots=True)
class AgentResponse:
    text: str
    usage: ModelUsage = ModelUsage()

    def __post_init__(self) -> None:
        _text(self.text, "response text", empty=True)
        if not isinstance(self.usage, ModelUsage):
            raise ValueError("usage must be ModelUsage")


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    sequence: int
    kind: Literal["input", "event", "completion", "summary_completion", "tool_exchange"]
    identity: AgentInputIdentity
    messages: tuple[Message, ...] = ()
    text: str = ""
    event_id: str | None = None
    usage: ModelUsage = ModelUsage()


@dataclass(frozen=True, slots=True)
class AgentUsage:
    action_calls: int
    summary_calls: int
    action_usage: tuple[ModelUsage, ...]
    summary_usage: tuple[ModelUsage, ...]


@runtime_checkable
class AsyncModel(Protocol):
    async def complete(self, request: ModelRequest) -> ModelCompletion: ...


@runtime_checkable
class AgentRuntime(Protocol):
    def validate_fresh(self) -> None: ...
    def accept_input(self, request: AgentRequest) -> None: ...
    def observe_result(self, event: AgentEvent) -> None: ...
    async def complete(self, request: AgentRequest) -> AgentResponse: ...
    def close(self) -> None: ...
