"""Optional LiteLLM SDK transport with task-local, bounded conversation state.

Chat calls use ``acompletion``. The explicitly selected OpenAI Responses route
uses ``aresponses`` so that output item order, phases and opaque reasoning are
replayed unchanged. No provider is inferred from an endpoint or display model.
"""
from __future__ import annotations

import importlib
import json
from hashlib import sha256
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from ..api_agent import APIAgent
from .action_submission import SUBMISSION_NAME
from .encoding import api_chat_messages, chat_messages, tool_declarations
from .models import ModelResponseError, SummaryResponseError
from .types import LiteLLMAssistantState, ModelCompletion, ModelRequest, ModelUsage, ToolCall


def validate_litellm_config(config: Mapping) -> dict:
    unknown = set(config) - {"model", "api"}
    if unknown:
        raise ValueError("unknown simple_agent.litellm settings: " + ", ".join(sorted(unknown)))
    model = config.get("model")
    if (not isinstance(model, str) or model != model.strip() or len(model) > 512
        or "/" not in model or any(not part for part in model.split("/"))
        or any(char.isspace() for char in model)):
        raise ValueError("simple_agent.litellm.model must be an explicit provider-qualified LiteLLM model")
    api = config.get("api", "chat_completions")
    if api not in ("chat_completions", "responses"):
        raise ValueError("simple_agent.litellm.api must be chat_completions or responses")
    if api == "responses":
        if not model.startswith("openai/"):
            raise ValueError("LiteLLM Responses currently requires an explicit openai/ model")
        # Accept the documented bridge spelling but use the native SDK route.
        # The Chat bridge discards output phases and changes output item order.
        if model.startswith("openai/responses/"):
            model = "openai/" + model[len("openai/responses/"):]
    elif model.startswith("openai/responses/"):
        raise ValueError("openai/responses/ models require simple_agent.litellm.api: responses")
    return {"model": model, "api": api}


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _present(value: Any, name: str) -> tuple[bool, Any]:
    if isinstance(value, Mapping):
        return name in value, value.get(name)
    fields = getattr(value, "model_fields_set", None)
    if fields is None:
        fields = getattr(value, "__fields_set__", None)
    if isinstance(fields, (set, frozenset)):
        if name in fields:
            return True, getattr(value, name, None)
    elif hasattr(value, name):
        return True, getattr(value, name)
    extra = _field(value, "model_extra")
    return (True, extra[name]) if isinstance(extra, Mapping) and name in extra else (False, None)


def _json_value(value: Any) -> Any:
    """Detach SDK objects without stringifying unknown implementation objects."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump(mode="json", exclude_unset=True)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("non-string JSON object key")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("unsupported SDK JSON value")


class LiteLLMModel:
    """Caller-independent adapter; the runtime owns all continuation memory."""
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.config = validate_litellm_config(owner.litellm_config)
        self.last_response_metadata: dict = {}

    @property
    def native_scope(self) -> str:
        identity = json.dumps(["litellm", self.config["api"], self.config["model"], self.owner.base_url],
                              ensure_ascii=False, separators=(",", ":"))
        return "simple-litellm:" + sha256(identity.encode("utf-8")).hexdigest()

    def request_messages(self, request: ModelRequest) -> list[dict]:
        messages = (chat_messages(request.messages) if request.purpose == "summary" else
                    api_chat_messages(request.messages, native_scope=self.native_scope))
        if self.config["api"] == "chat_completions":
            if any("responses_output" in message for message in messages):
                raise ModelResponseError("LiteLLM Chat cannot replay Responses state")
            return messages
        result = []
        for message in messages:
            if "responses_output" in message:
                result.extend(message["responses_output"])
                continue
            role, content = message["role"], message.get("content")
            if role == "tool":
                result.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                               "output": content or ""})
                continue
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
            converted = []
            for part in parts:
                if part.get("type") == "text":
                    converted.append({"type": "output_text" if role == "assistant" else "input_text", "text": part["text"]})
                elif part.get("type") == "image_url" and role != "assistant":
                    image = part["image_url"]
                    converted.append({"type": "input_image", "image_url": image["url"] if isinstance(image, dict) else image})
                else:
                    raise ModelResponseError("unsupported LiteLLM Responses message content")
            if converted:
                result.append({"type": "message", "role": role, "content": converted})
            for call in message.get("tool_calls") or ():
                result.append({"type": "function_call", "call_id": call["id"],
                               "name": call["function"]["name"], "arguments": call["function"]["arguments"]})
        return result

    def request_parameters(self, request: ModelRequest) -> dict:
        owner = self.owner
        bound = request.max_output_tokens
        if request.purpose == "action" and owner.max_completion_tokens is not None:
            bound = min(bound, owner.max_completion_tokens)
        responses = self.config["api"] == "responses"
        if responses and bound < 16:
            raise ValueError("LiteLLM OpenAI Responses requires an output token budget of at least 16")
        # In the pinned SDK, Chat's num_retries aliases its provider retry count
        # AND enables an outer retry loop. Native Responses uses direct HTTP and
        # only the outer loop. Select one layer so N means at most N retries.
        kwargs = {"model": self.config["model"], "num_retries": owner.max_retries if responses else None,
                  "max_retries": 0 if responses else owner.max_retries, "drop_params": False}
        kwargs["max_output_tokens" if responses else owner.output_token_parameter] = bound
        if responses:
            kwargs["store"] = False
            kwargs["include"] = ["reasoning.encrypted_content"]
        if owner.temperature is not None:
            kwargs["temperature"] = owner.temperature
        if owner.reasoning_effort is not None:
            kwargs["reasoning" if responses else "reasoning_effort"] = ({"effort": owner.reasoning_effort} if responses else owner.reasoning_effort)
        if owner.timeout_s is not None:
            kwargs["timeout"] = owner.timeout_s
        if request.tools:
            tools = tool_declarations(request.tools)
            if responses:
                tools = [{"type": "function", **tool["function"], "strict": False} for tool in tools]
            kwargs["tools"] = tools
        if request.purpose == "action" and any(tool.name == SUBMISSION_NAME for tool in request.tools):
            kwargs["tool_choice"] = owner.tool_choice
        return kwargs

    def request_transport(self, request: ModelRequest) -> dict:
        # LiteLLM performs provider-specific conversion after this logical input.
        endpoint = None
        if self.owner.base_url:
            try:
                parsed = urlsplit(self.owner.base_url)
                endpoint = urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
            except ValueError:
                endpoint = "<invalid endpoint>"
        return {"backend": "litellm", "api": self.config["api"], "model": self.config["model"],
                "endpoint": endpoint, "kind": "logical_sdk_request",
                "sdk_method": "aresponses" if self.config["api"] == "responses" else "acompletion"}

    def _sdk(self):
        try:
            sdk = importlib.import_module("litellm")
        except ImportError:
            raise RuntimeError("LiteLLM backend requires the optional dependency; install agent-ark[litellm] in the existing Python environment") from None
        # The SDK's process-wide flags override per-call false in several paths.
        # Fail closed instead of silently changing caller configuration or globals.
        if (getattr(sdk, "drop_params", False) or getattr(sdk, "modify_params", False)
            or getattr(sdk, "model_fallbacks", None) or getattr(sdk, "num_retries", None)
            or getattr(sdk, "retry_policy", None)):
            raise ValueError("LiteLLM global parameter modification, retries and fallbacks must be disabled for SimpleAgent")
        return sdk

    def _state(self, native: dict) -> LiteLLMAssistantState:
        try:
            state = LiteLLMAssistantState(self.native_scope, json.dumps(
                _json_value(native), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise ModelResponseError("LiteLLM returned invalid native assistant state") from None
        self.last_response_metadata["native_assistant"] = {"scope": state.scope, "message": state.message}
        return state

    def _chat_completion(self, response: Any, request: ModelRequest) -> ModelCompletion:
        usage = _field(response, "usage")
        self.last_response_metadata["usage"] = APIAgent._usage_to_jsonable(usage)
        choices = _field(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise ModelResponseError("LiteLLM must return exactly one completion")
        choice = choices[0]
        finish = _field(choice, "finish_reason")
        self.last_response_metadata["finish_reason"] = finish
        message = _field(choice, "message")
        content = APIAgent._coerce_response_text(_field(message, "content"))
        refusal = _field(message, "refusal")
        self.last_response_metadata["refusal"] = refusal
        calls = _field(message, "tool_calls")
        calls = () if calls is None else calls
        if not isinstance(calls, (list, tuple)) or len(calls) > 32:
            raise ModelResponseError("LiteLLM returned invalid tool calls")
        incomplete = request.purpose == "summary" and finish == "length"
        if finish not in (None, "stop", "tool_calls") and not incomplete:
            raise ModelResponseError("LiteLLM completion was not complete")
        if refusal:
            raise ModelResponseError("LiteLLM refused completion")
        if calls and (not request.tools or finish not in (None, "tool_calls")):
            raise ModelResponseError("LiteLLM requested unavailable tools")
        if not calls and finish == "tool_calls":
            raise ModelResponseError("LiteLLM returned no tool calls")
        native = {"role": "assistant", "content": _field(message, "content")}
        state_fields = False
        for name in ("reasoning_items", "thinking_blocks", "reasoning_content", "provider_specific_fields"):
            present, item = _present(message, name)
            if present:
                native[name] = item
                state_fields = state_fields or item is not None
        parsed, native_calls = [], []
        try:
            for call in calls:
                if _field(call, "type") != "function":
                    raise ValueError("unsupported tool")
                function = _field(call, "function")
                parsed.append(ToolCall(_field(call, "id"), _field(function, "name"), _field(function, "arguments")))
                item = {"id": _field(call, "id"), "type": "function", "function": {
                    "name": _field(function, "name"), "arguments": _field(function, "arguments")}}
                for source, target in ((call, item), (function, item["function"])):
                    present, fields = _present(source, "provider_specific_fields")
                    if present:
                        target["provider_specific_fields"] = fields
                        state_fields = state_fields or fields is not None
                native_calls.append(item)
            if calls:
                native["tool_calls"] = native_calls
            counts = ModelUsage(_field(usage, "prompt_tokens"), _field(usage, "completion_tokens"), _field(usage, "total_tokens"))
            state = self._state(native) if state_fields and request.purpose == "action" else None
            completion = ModelCompletion(content, counts, tuple(parsed), native_assistant=state)
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise ModelResponseError("LiteLLM returned invalid tools or usage") from None
        return self._validate_completion(completion, request, incomplete)

    def _responses_completion(self, response: Any, request: ModelRequest) -> ModelCompletion:
        usage = _field(response, "usage")
        self.last_response_metadata["usage"] = APIAgent._usage_to_jsonable(usage)
        status = _field(response, "status")
        self.last_response_metadata["status"] = status
        details = _field(response, "incomplete_details")
        incomplete = (request.purpose == "summary" and status == "incomplete"
                      and _field(details, "reason") == "max_output_tokens")
        if _field(response, "error") or (status != "completed" and not incomplete):
            raise ModelResponseError("LiteLLM Responses completion was not complete")
        output = _field(response, "output")
        if not isinstance(output, (list, tuple)) or len(output) > 256:
            raise ModelResponseError("LiteLLM returned invalid Responses output")
        texts, calls = [], []
        try:
            raw_output = _json_value(output)
            for item in raw_output:
                if not isinstance(item, dict):
                    raise ValueError("invalid output item")
                kind = item.get("type")
                if kind == "message":
                    if item.get("role") != "assistant" or (item.get("status", "completed") != "completed" and not incomplete):
                        raise ValueError("incomplete assistant output")
                    content = item.get("content")
                    if not isinstance(content, list):
                        raise ValueError("invalid output content")
                    for part in content:
                        if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                            raise ValueError("unsupported or refused output content")
                        texts.append(part["text"])
                elif kind == "function_call":
                    # OpenAI's function-call status is optional and nullable;
                    # the response's completed status is checked above.
                    if item.get("status") not in (None, "completed"):
                        raise ValueError("incomplete tool output")
                    calls.append(ToolCall(item.get("call_id"), item.get("name"), item.get("arguments")))
                elif kind != "reasoning":
                    raise ValueError("unsupported output item")
            if calls and not request.tools:
                raise ValueError("unavailable tools")
            counts = ModelUsage(_field(usage, "input_tokens"), _field(usage, "output_tokens"), _field(usage, "total_tokens"))
            content = "".join(texts).strip()
            native = {"role": "assistant", "content": content, "responses_output": raw_output}
            if calls:
                native["tool_calls"] = [{"id": call.call_id, "type": "function", "function": {
                    "name": call.name, "arguments": call.arguments_json}} for call in calls]
            state = self._state(native) if raw_output and request.purpose == "action" else None
            completion = ModelCompletion(content, counts, tuple(calls), native_assistant=state)
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise ModelResponseError("LiteLLM returned invalid Responses content, tools or usage") from None
        return self._validate_completion(completion, request, incomplete)

    @staticmethod
    def _validate_completion(completion: ModelCompletion, request: ModelRequest, incomplete: bool) -> ModelCompletion:
        if incomplete:
            raise SummaryResponseError("incomplete", completion.usage)
        if request.purpose == "summary" and not completion.text:
            raise SummaryResponseError("empty_content", completion.usage)
        if request.purpose == "action" and not completion.text and not completion.tool_calls:
            raise ModelResponseError("LiteLLM returned no assistant content or tool calls")
        return completion

    async def complete(self, request: ModelRequest) -> ModelCompletion:
        self.last_response_metadata = {}
        sdk = self._sdk()
        kwargs = self.request_parameters(request)
        if self.owner.base_url is not None:
            kwargs["api_base"] = self.owner.base_url
        if self.owner._api_key is not None:
            kwargs["api_key"] = self.owner._api_key
        messages = self.request_messages(request)
        if self.config["api"] == "responses":
            response = await sdk.aresponses(input=messages, **kwargs)
            return self._responses_completion(response, request)
        response = await sdk.acompletion(messages=messages, **kwargs)
        return self._chat_completion(response, request)

    async def close(self) -> None:
        # LiteLLM manages shared clients; closing globals would affect other agents.
        return None
