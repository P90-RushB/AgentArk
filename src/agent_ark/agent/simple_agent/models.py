# SPDX-License-Identifier: Apache-2.0
# Adapted with permission from the ArkArena stdlib core (ark_arena/harness).
"""Stateless adapters; compatible async clients remain caller-owned."""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from .encoding import chat_messages, tool_declarations
from .types import AgentError, ModelCompletion, ModelRequest, ModelUsage, ToolCall


class ModelResponseError(AgentError):
    pass


class CallableModel:
    def __init__(self, function: Callable[[ModelRequest], Awaitable[ModelCompletion]]) -> None:
        if not callable(function):
            raise TypeError("model function must be callable")
        self._function = function

    async def complete(self, request: ModelRequest) -> ModelCompletion:
        pending = self._function(request)
        if not inspect.isawaitable(pending):
            raise ModelResponseError("model callable must return an awaitable")
        response = await pending
        if not isinstance(response, ModelCompletion):
            raise ModelResponseError("model callable returned an invalid completion")
        return response


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


class ChatCompletionsModel:
    """Use ``client.chat.completions.create`` without importing a vendor SDK."""

    def __init__(self, client: object, model: str, *, output_token_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens") -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        self._client = client
        self._model = model
        if output_token_parameter not in ("max_completion_tokens", "max_tokens"):
            raise ValueError("unsupported output token parameter")
        self._output_token_parameter = output_token_parameter
        create = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
        if not callable(create):
            raise TypeError("client must expose chat.completions.create")
        self._create = create

    async def complete(self, request: ModelRequest) -> ModelCompletion:
        kwargs = {self._output_token_parameter: request.max_output_tokens}
        if request.tools:
            kwargs["tools"] = tool_declarations(request.tools)
        pending = self._create(model=self._model, messages=chat_messages(request.messages), **kwargs)
        if not inspect.isawaitable(pending):
            raise ModelResponseError("chat client must be asynchronous")
        response = await pending
        choices = _field(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise ModelResponseError("chat client must return exactly one completion")
        choice = choices[0]
        if _field(choice, "finish_reason") not in (None, "stop", "tool_calls"):
            raise ModelResponseError("chat completion was not complete")
        message = _field(choice, "message")
        text = _field(message, "content")
        calls = _field(message, "tool_calls") or ()
        if (calls and (not request.tools or _field(choice, "finish_reason") not in (None, "tool_calls"))) or _field(message, "refusal"):
            raise ModelResponseError("chat client returned unsupported completion content")
        if calls and text is None:
            text = ""
        if not isinstance(text, str) or not isinstance(calls, (list, tuple)) or len(calls) > 32:
            raise ModelResponseError("chat client returned unsupported completion content")
        if not calls and _field(choice, "finish_reason") == "tool_calls":
            raise ModelResponseError("chat client returned no tool calls")
        usage = _field(response, "usage")
        try:
            counts = ModelUsage(
                _field(usage, "prompt_tokens"), _field(usage, "completion_tokens"), _field(usage, "total_tokens")
            )
        except ValueError:
            raise ModelResponseError("chat client returned invalid usage") from None
        try:
            parsed = []
            for call in calls:
                if _field(call, "type") != "function":
                    raise ValueError("unsupported tool")
                function = _field(call, "function")
                parsed.append(ToolCall(_field(call, "id"), _field(function, "name"), _field(function, "arguments")))
            return ModelCompletion(text, counts, tuple(parsed))
        except (ValueError, TypeError):
            raise ModelResponseError("chat client returned invalid tool calls") from None
