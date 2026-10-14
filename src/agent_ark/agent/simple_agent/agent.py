"""Stateful, bounded API agent for append-only AgentArk delta messages.

The runtime manages private conversation memory and optional host file tools.
Final environment actions still use APIAgent's existing extraction contract.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from ..api_agent import APIAgent
from ..base_agent import BaseAgent
from .encoding import chat_messages, tool_declarations
from .models import ModelResponseError, SummaryResponseError
from .runtime import SimpleRuntime, SimpleRuntimeConfig
from .tools import FileWorkspace, RecordSnapshot, ToolLimits, ToolRegistry
from .types import (
    AgentInputIdentity, AgentLimitError, AgentRequest, ImagePart, Message, ModelCompletion,
    ModelRequest, ModelUsage, TextPart, ToolCall,
)
from .action_submission import NATIVE_INSTRUCTIONS, SUBMISSION_NAME, register_submission, translate_submission


_INSTRUCTIONS = (
    "Choose the next AgentArk environment action using only the supplied task "
    "instructions, observations, and visible interaction history. You may use "
    "the declared host function tools before answering, when they are available. "
    "Host function tool calls inspect only your own visible records or private "
    "notes; they do not operate the environment. Environment action API text "
    "such as <tool_call>...</tool_call> is different from host function calls: "
    "return that text in your final assistant answer when the task requests it. "
    "Preserve the task's specified final action format, including C# source for "
    "code actions. Do not wrap the final answer in a new harness JSON schema. "
    "Model notes and summaries are memory aids, not new environment facts."
)


def _mapping(value: Any, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _usage_dict(usage: ModelUsage) -> dict:
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens}


def _total_usage(usages: list[ModelUsage]) -> dict:
    def total(name: str) -> Optional[int]:
        values = [getattr(usage, name) for usage in usages]
        return sum(values) if all(value is not None for value in values) else None
    return {name: total(name) for name in ("input_tokens", "output_tokens", "total_tokens")}


class _LoopOwner:
    """A single event loop owns all requests and the default HTTP client."""
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="agentark-simple-loop", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        self.loop.close()

    def run(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


class _OpenAIModel:
    def __init__(self, owner: "SimpleAgent") -> None:
        self.owner = owner
        self.client = None
        self.last_response_metadata: dict = {}

    def request_parameters(self, request: ModelRequest) -> dict:
        owner = self.owner
        kwargs = {"model": owner.model, "extra_body": dict(owner.extra_body),
                  owner.output_token_parameter: request.max_output_tokens}
        if owner.max_completion_tokens is not None:
            kwargs[owner.output_token_parameter] = min(request.max_output_tokens, owner.max_completion_tokens)
        if owner.temperature is not None:
            kwargs["temperature"] = owner.temperature
        if owner._request_reasoning_effort is not None:
            kwargs["reasoning_effort"] = owner._request_reasoning_effort
        if owner.timeout_s is not None:
            kwargs["timeout"] = owner.timeout_s
        if request.tools:
            kwargs["tools"] = tool_declarations(request.tools)
        if request.purpose == "action" and any(tool.name == SUBMISSION_NAME for tool in request.tools):
            kwargs["tool_choice"] = owner.tool_choice
        return kwargs

    async def complete(self, request: ModelRequest) -> ModelCompletion:
        if self.client is None:
            from openai import AsyncOpenAI
            kwargs = {"api_key": self.owner._api_key, "base_url": self.owner.base_url,
                      "max_retries": self.owner.max_retries}
            if self.owner.timeout_s is not None:
                kwargs["timeout"] = self.owner.timeout_s
            self.client = AsyncOpenAI(**kwargs)
        self.last_response_metadata = {}
        response = await self.client.chat.completions.create(
            messages=chat_messages(request.messages), **self.request_parameters(request))
        usage = _field(response, "usage")
        self.last_response_metadata = {"usage": APIAgent._usage_to_jsonable(usage)}
        choices = _field(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise ModelResponseError("chat client must return exactly one completion")
        choice = choices[0]
        finish_reason = _field(choice, "finish_reason")
        self.last_response_metadata["finish_reason"] = finish_reason
        message = _field(choice, "message")
        content = APIAgent._coerce_response_text(_field(message, "content"))
        reasoning = APIAgent._coerce_response_text(
            _field(message, "reasoning") or _field(message, "reasoning_content")
            or _field(_field(message, "model_extra", {}), "reasoning")
            or _field(_field(message, "model_extra", {}), "reasoning_content"))
        text = APIAgent._combine_reasoning_and_content(reasoning=reasoning, content=content)
        self.last_response_metadata["assistant_raw"] = text
        self.last_response_metadata["refusal"] = _field(message, "refusal")
        summary_incomplete = request.purpose == "summary" and finish_reason == "length"
        if finish_reason not in (None, "stop", "tool_calls") and not summary_incomplete:
            raise ModelResponseError("chat completion was not complete")
        if _field(message, "refusal"):
            raise ModelResponseError("chat client refused completion")
        calls = _field(message, "tool_calls") or ()
        if not isinstance(calls, (list, tuple)) or len(calls) > 32:
            raise ModelResponseError("chat client returned invalid tool calls")
        if calls and (not request.tools or finish_reason not in (None, "tool_calls")):
            raise ModelResponseError("chat client requested unavailable tools")
        if not calls and finish_reason == "tool_calls":
            raise ModelResponseError("chat client returned no tool calls")
        if request.purpose == "summary":
            # Keep provider reasoning in the audit metadata, not model notes.
            text = content
        try:
            parsed = []
            for call in calls:
                if _field(call, "type") != "function":
                    raise ValueError("unsupported tool")
                function = _field(call, "function")
                parsed.append(ToolCall(_field(call, "id"), _field(function, "name"), _field(function, "arguments")))
            counts = ModelUsage(_field(usage, "prompt_tokens"), _field(usage, "completion_tokens"), _field(usage, "total_tokens"))
            completion = ModelCompletion(text, counts, tuple(parsed))
        except (ValueError, TypeError):
            raise ModelResponseError("chat client returned invalid tools or usage") from None
        if summary_incomplete:
            raise SummaryResponseError("incomplete", counts)
        if request.purpose == "summary" and not content:
            raise SummaryResponseError("empty_content", counts)
        return completion

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None


class _TracedModel:
    def __init__(self, backend: Any, *, native_submission: bool = False, config=None, tool_choice: str = "required") -> None:
        self.backend = backend
        self.native_submission = native_submission
        self.config = config
        self.tool_choice = tool_choice
        self.calls: list[dict] = []
        self.usages: list[tuple[str, ModelUsage]] = []
        self.events: list[dict] = []

    def _failure_metadata(self, record: dict, request: ModelRequest, error: Exception | None = None) -> None:
        metadata = getattr(self.backend, "last_response_metadata", None)
        raw_record_usage = record.get("usage") or {}
        usage = ModelUsage(raw_record_usage.get("input_tokens"), raw_record_usage.get("output_tokens"), raw_record_usage.get("total_tokens"))
        if isinstance(error, SummaryResponseError):
            usage = error.usage
        if metadata:
            record["response_metadata"] = dict(metadata)
            raw = metadata.get("usage") or {}
            try:
                if raw:
                    usage = ModelUsage(raw.get("prompt_tokens", raw.get("input_tokens")),
                                       raw.get("completion_tokens", raw.get("output_tokens")), raw.get("total_tokens"))
            except (ValueError, TypeError, AttributeError):
                pass
        record["usage"] = _usage_dict(usage)
        self.usages.append((request.purpose, usage))

    async def complete(self, request: ModelRequest) -> ModelCompletion:
        record = {"purpose": request.purpose, "request_messages": chat_messages(request.messages),
                  "max_output_tokens": request.max_output_tokens, "input_tokens": request.input_tokens,
                  "token_count_is_estimate": request.token_count_is_estimate,
                  "tools": tool_declarations(request.tools), "status": "pending"}
        record["request_images"] = [
            {"message_index": message_index, "part_index": part_index, "role": message.role,
             "sha256": sha256(part.data).hexdigest(), "bytes": len(part.data),
             "mime_type": part.mime_type, "width": part.width, "height": part.height}
            for message_index, message in enumerate(request.messages)
            for part_index, part in enumerate(message.parts) if isinstance(part, ImagePart)]
        parameters = getattr(self.backend, "request_parameters", None)
        if callable(parameters):
            record["parameters"] = parameters(request)
        self.calls.append(record)
        try:
            completion = await self.backend.complete(request)
            if not isinstance(completion, ModelCompletion):
                raise ModelResponseError("model returned an invalid completion")
            record.update({"status": "completed", "completion": completion.text,
                           "tool_calls": [asdict(call) for call in completion.tool_calls],
                           "usage": _usage_dict(completion.usage)})
            metadata = getattr(self.backend, "last_response_metadata", None)
            if metadata:
                record["response_metadata"] = dict(metadata)
            translated = completion
            if (self.native_submission and request.purpose == "action"
                and any(tool.name == SUBMISSION_NAME for tool in request.tools)):
                if self.tool_choice == "required" and not completion.tool_calls:
                    raise ModelResponseError("native tool_choice=required returned no native function call")
                translated, event = translate_submission(completion,
                    max_bytes=self.config.max_action_output_bytes,
                    max_characters=self.config.max_action_output_characters)
                if event is not None:
                    self.events.append(event)
            self.usages.append((request.purpose, completion.usage))
            return translated
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            self._failure_metadata(record, request)
            raise
        except Exception as exc:
            record.update({"status": "failed", "error_type": type(exc).__name__})
            self._failure_metadata(record, request, exc)
            raise


@dataclass
class _AgentState:
    runtime: SimpleRuntime
    model: _TracedModel
    workspace: Optional[FileWorkspace] = None
    directory: Any = None
    pins: tuple[Message, ...] = ()
    env_pins: tuple[Message, ...] = ()
    initialized: bool = False
    turn: int = 0
    attempt: int = 1
    last_final: Optional[str] = None
    published_records: int = 0
    published_bytes: int = 0


class SimpleAgent(BaseAgent):
    """Explicit opt-in alternative to APIAgent, consuming ``messages: delta``.

    ``simple_agent_config`` accepts ``context`` (SimpleRuntimeConfig fields),
    ``tools`` (``enabled``, ``workspace_parent``, ``limits``), and
    ``output_token_parameter`` (``max_completion_tokens`` or ``max_tokens``).
    An injected async ``model_backend`` is caller-owned and useful for tests.
    """
    def __init__(
        self, name: str = "SimpleAgent", api_key: Optional[str] = None,
        base_url: Optional[str] = None, model: str = "qwen3-vl-plus",
        temperature: Optional[float] = 0.2, provider: str = "auto",
        timeout_s: Optional[float] = 1800.0, max_retries: int = 0,
        thinking: Any = None, reasoning_effort: Optional[str] = None,
        max_completion_tokens: Optional[int] = 30000, *,
        simple_agent_config: Optional[Mapping] = None, env_message_mode: str = "delta",
        model_backend: Any = None, tool_setup: Any = None,
    ) -> None:
        super().__init__(name)
        if env_message_mode != "delta":
            raise ValueError("SimpleAgent requires append-only env messages with return_mode: delta")
        self.env_message_mode = env_message_mode
        config = _mapping(simple_agent_config, "simple_agent")
        unknown = set(config) - {"context", "tools", "output_token_parameter", "tool_choice"}
        if unknown:
            raise ValueError("unknown simple_agent settings: " + ", ".join(sorted(unknown)))
        self.context_config = SimpleRuntimeConfig(**_mapping(config.get("context"), "simple_agent.context"))
        self.tools_config = _mapping(config.get("tools"), "simple_agent.tools")
        unknown_tools = set(self.tools_config) - {"enabled", "workspace_parent", "limits"}
        if unknown_tools:
            raise ValueError("unknown simple_agent.tools settings: " + ", ".join(sorted(unknown_tools)))
        self.tools_enabled = self.tools_config.get("enabled", False)
        if type(self.tools_enabled) is not bool:
            raise ValueError("simple_agent.tools.enabled must be boolean")
        if self.tools_enabled and not self.tools_config.get("workspace_parent"):
            raise ValueError("enabled simple_agent tools require workspace_parent")
        self.tool_limits = ToolLimits(**{"max_records": 512,
            **_mapping(self.tools_config.get("limits"), "simple_agent.tools.limits")})
        self.output_token_parameter = config.get("output_token_parameter", "max_completion_tokens")
        if self.output_token_parameter not in ("max_completion_tokens", "max_tokens"):
            raise ValueError("output_token_parameter must be max_completion_tokens or max_tokens")
        self.tool_choice = config.get("tool_choice", "required")
        if self.tool_choice not in ("required", "auto"):
            raise ValueError("simple_agent.tool_choice must be required or auto")
        self.base_url = str(base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model, self.provider = str(model), str(provider or "auto").strip().lower()
        self.temperature = None if temperature is None else float(temperature)
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self.max_retries = int(max_retries)
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive or null")
        if self.max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self.thinking_type = APIAgent._normalize_thinking_type(thinking)
        self.reasoning_effort = APIAgent._normalize_reasoning_effort(reasoning_effort)
        self.max_completion_tokens = APIAgent._normalize_max_completion_tokens(max_completion_tokens)
        if self.thinking_type == "disabled" and self.reasoning_effort is not None:
            raise ValueError("thinking.type=disabled conflicts with reasoning_effort")
        parameter_source = object.__new__(APIAgent)
        parameter_source.client = SimpleNamespace(provider=self.provider, base_url_host=(urlparse(self.base_url).hostname or "").lower())
        parameter_source.thinking_type = self.thinking_type
        parameter_source.reasoning_effort = self.reasoning_effort
        self.extra_body, self._request_reasoning_effort = parameter_source._request_parameters()
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if model_backend is not None and not callable(getattr(model_backend, "complete", None)):
            raise TypeError("model_backend must expose async complete")
        if tool_setup is not None and not callable(tool_setup):
            raise TypeError("tool_setup must be a trusted callable")
        self._tool_setup = tool_setup
        self._backend = model_backend if model_backend is not None else _OpenAIModel(self)
        self._owns_backend = model_backend is None
        self._loop: Optional[_LoopOwner] = None
        self._lock = threading.RLock()
        self._closed = False
        self._run_id = uuid.uuid4().hex
        self._case_id = uuid.uuid4().hex
        self._states: dict[Any, _AgentState] = {}
        self.last_trace_by_agent: dict = {}

    def _state(self, agent_idx: Any) -> _AgentState:
        if self._closed:
            raise RuntimeError("SimpleAgent is closed")
        if agent_idx not in self._states:
            directory = workspace = registry = None
            if self.tools_enabled:
                parent = Path(self.tools_config["workspace_parent"])
                parent.mkdir(parents=True, exist_ok=True)
                directory = TemporaryDirectory(prefix="agentark-simple-case-", dir=parent)
                workspace = FileWorkspace(directory.name, limits=self.tool_limits)
                registry = workspace
            elif self._tool_setup is not None:
                registry = ToolRegistry(limits=self.tool_limits)
            native = registry is not None
            try:
                if native:
                    # Reserve before any caller-owned tool registration.
                    register_submission(registry)
                if self._tool_setup is not None:
                    self._tool_setup(registry)
            except BaseException:
                if directory is not None:
                    directory.cleanup()
                raise
            traced = _TracedModel(self._backend, native_submission=native, config=self.context_config, tool_choice=self.tool_choice)
            self._states[agent_idx] = _AgentState(
                SimpleRuntime(traced, self.context_config, tools=registry), traced, workspace, directory)
        return self._states[agent_idx]

    def _image(self, url: str) -> ImagePart:
        match = re.fullmatch(r"data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=\r\n]+)", url)
        if not match:
            raise ValueError("SimpleAgent accepts only inline observation image data URLs")
        encoded = match.group(2)
        if len(encoded) > ((self.context_config.max_archive_image_bytes + 2) // 3) * 4 + 1024:
            raise ValueError("observation image exceeds archive byte bound")
        try:
            data = base64.b64decode(encoded.replace("\r", "").replace("\n", ""), validate=True)
        except ValueError:
            raise ValueError("invalid inline observation image") from None
        return ImagePart(data, mime_type=match.group(1))

    def _messages(self, obs_dict: Any) -> tuple[Message, ...]:
        if not isinstance(obs_dict, dict) or not isinstance(obs_dict.get("messages"), list) or not obs_dict["messages"]:
            raise ValueError("SimpleAgent requires nonempty env delta messages; no raw observation fallback")
        if obs_dict.get("message_mode", obs_dict.get("messages_return_mode", "delta")) != "delta":
            raise ValueError("SimpleAgent received non-delta messages")
        result = []
        for value in obs_dict["messages"]:
            if not isinstance(value, dict) or value.get("tool_calls") or value.get("tool_call_id"):
                raise ValueError("environment messages cannot supply host tool interactions")
            role = value.get("role")
            if role not in ("system", "developer", "user", "assistant"):
                raise ValueError("unsupported environment message role")
            content = value.get("content", "")
            parts = []
            if isinstance(content, str):
                parts.append(TextPart(content))
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        raise ValueError("invalid observation message part")
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append(TextPart(part["text"]))
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url")
                        url = image_url.get("url") if isinstance(image_url, dict) else image_url
                        parts.append(self._image(url if isinstance(url, str) else ""))
                    elif part.get("type") == "image" and isinstance(part.get("image_base64"), str):
                        parts.append(self._image("data:" + str(part.get("mime_type", "image/png")) + ";base64," + part["image_base64"]))
                    else:
                        raise ValueError("unsupported observation message part")
            else:
                raise ValueError("observation message content must be text or parts")
            result.append(Message(role, tuple(parts) or (TextPart(""),)))
        return tuple(result)

    def _prepare(self, agent_idx: Any, obs_dict: Any, *, preview: bool = False) -> tuple[_AgentState, AgentRequest, tuple[Message, ...], int, tuple[Message, ...]]:
        messages = self._messages(obs_dict)
        if self._closed:
            raise RuntimeError("SimpleAgent is closed")
        if preview and agent_idx not in self._states:
            traced = _TracedModel(self._backend)
            state = _AgentState(SimpleRuntime(traced, self.context_config), traced)
        else:
            state = self._state(agent_idx)
        env_pins = tuple(message for message in messages if message.role in ("system", "developer"))
        dynamic = tuple(message for message in messages if message.role not in ("system", "developer"))
        if not state.initialized:
            native = self.tools_enabled or self._tool_setup is not None
            if native:
                pins = (Message.text("system", NATIVE_INSTRUCTIONS),) + tuple(
                    Message("developer", (TextPart("Original environment task instructions (task rules and payload recipes):\n"),) + message.parts)
                    for message in env_pins)
            else:
                pins = env_pins + (Message.text("developer", _INSTRUCTIONS),)
            # MessageContext merges the task and current observation into one
            # text part. Freeze only its documented task prefix, not reset images.
            if dynamic and dynamic[-1].role == "user":
                last = dynamic[-1]
                parts = list(last.parts)
                for index, part in enumerate(parts):
                    if isinstance(part, TextPart) and "Current observation (step_msg):" in part.text:
                        prompt, current = part.text.split("Current observation (step_msg):", 1)
                        if prompt.strip():
                            pins += (Message.text("developer", "Task instructions from the initial environment message:\n" + prompt.strip()),)
                            parts[index] = TextPart("Current observation (step_msg):" + current)
                            dynamic = dynamic[:-1] + (Message("user", tuple(parts)),)
                        break
                else:
                    # A valid MessageContext image-only reset has task text
                    # but no observation label. Keep its current-user text as
                    # the initial prompt; never pin the preceding history or
                    # the images themselves. Later user messages update state.
                    prompt = "\n".join(part.text for part in last.parts if isinstance(part, TextPart)).strip()
                    if prompt:
                        pins += (Message.text("developer", "Initial task text from the environment (later observations update state):\n" + prompt),)
        else:
            if env_pins and env_pins != state.env_pins:
                raise ValueError("environment instructions changed during a live SimpleAgent case")
            pins = state.pins
        # Remove exactly the immediate previous final-answer echo. Never use
        # whole-message overlap/digest deduplication: repeated feedback is real.
        if state.last_final is not None and dynamic and dynamic[0] == Message.text("assistant", state.last_final):
            dynamic = dynamic[1:]
        if not dynamic:
            raise ValueError("delta must contain a new visible observation or result")
        attempt = state.attempt
        for message in dynamic:
            if message.role == "user":
                for part in message.parts:
                    if isinstance(part, TextPart):
                        matches = re.findall(r"Auto-reset started attempt ([1-9][0-9]*)\.", part.text)
                        if matches:
                            attempt = max(attempt, int(matches[-1]))
        identity = AgentInputIdentity(self._run_id, self._case_id, "attempt-" + str(attempt), str(agent_idx), 0, "turn-" + str(state.turn + 1))
        return state, AgentRequest(identity, pins + dynamic, "carry"), dynamic, attempt, messages

    def build_request_messages(self, obs: dict) -> dict:
        """Preview pre-compaction context without accepting or consuming input.

        ``model_requests`` in forward_with_trace records every actual post-
        compaction request, including summaries and host tool continuations.
        """
        with self._lock:
            out = {}
            for agent_idx, obs_dict in (obs or {}).items():
                if isinstance(obs_dict, dict) and obs_dict.get("skip_infer"):
                    out[agent_idx] = None
                    continue
                state, request, dynamic, _, _ = self._prepare(agent_idx, obs_dict, preview=True)
                pins = tuple(message for message in request.messages if message.role in ("system", "developer"))
                previous = tuple(message for message in state.runtime.working_snapshot if message.role not in ("system", "developer"))
                out[agent_idx] = chat_messages(pins + previous + dynamic)
            return out

    def _publish(self, state: _AgentState, identity: AgentInputIdentity, messages: tuple[Message, ...], final: str, action: Optional[str]) -> dict:
        if state.workspace is None:
            return {"kind": "snapshot", "status": "disabled"}
        files, visible = [], []
        image_count = 0
        extensions = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
        for message in messages:
            parts = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    # Pretty JSON is line-readable. Chunk long strings rather
                    # than putting a whole observation on a quota-sized line;
                    # joining chunks reproduces the exact original text.
                    chunk_size = min(1024, max(1, (self.tool_limits.max_text_bytes - 128) // 6))
                    if len(part.text) > chunk_size:
                        parts.append({"type": "text", "text_chunks": [part.text[offset:offset + chunk_size]
                            for offset in range(0, len(part.text), chunk_size)]})
                    else:
                        parts.append({"type": "text", "text": part.text})
                else:
                    name = f"images/{image_count:04d}" + extensions[part.mime_type]
                    image_count += 1
                    files.append((name, part.data))
                    parts.append({"type": "image", "path": name, "mime_type": part.mime_type})
            visible.append({"role": message.role, "content": parts})
        message_record = {"format": "visible-messages-v1", "text_chunks": "concatenate exactly without a separator",
                          "messages": visible}
        files.extend((("messages.json", json.dumps(message_record, ensure_ascii=False, indent=2).encode("utf-8")),
                      ("assistant.txt", final.encode("utf-8")), ("action.txt", (action or "").encode("utf-8"))))
        size = sum(len(data) for _, data in files)
        limits = self.tool_limits
        if (state.published_records >= limits.max_records or state.published_bytes + size > limits.max_published_bytes
            or len(files) > limits.max_record_files or size > limits.max_record_bytes
            or any(len(data) > limits.max_record_file_bytes for _, data in files)):
            raise AgentLimitError("visible record snapshot capacity exhausted")
        record_id = f"turn-{state.turn:06d}"
        snapshot = RecordSnapshot(record_id, tuple(files))
        state.workspace.publish(snapshot, identity)
        state.published_records += 1
        state.published_bytes += size
        return {"kind": "snapshot", "status": "published", "record_id": record_id,
                "digest": snapshot.digest, "file_count": len(files), "bytes": size}

    async def _complete(self, state: _AgentState, request: AgentRequest):
        if self.timeout_s is None:
            return await state.runtime.complete(request)
        return await asyncio.wait_for(state.runtime.complete(request), timeout=self.timeout_s)

    def configuration_summary(self) -> dict:
        """Reproducible bounds and policy, without credentials or local paths."""
        return {"context": asdict(self.context_config),
                "tools": {"enabled": self.tools_enabled, "limits": asdict(self.tool_limits)},
                "output_token_parameter": self.output_token_parameter,
                "tool_choice": self.tool_choice,
                "env_message_mode": "delta", "sdk_max_retries": self.max_retries,
                "custom_tools": self._tool_setup is not None, "tools_enabled": self.tools_enabled,
                "native_action_submission": self.tools_enabled or self._tool_setup is not None}

    def _trace(self, state: _AgentState, archive_start: int, *, final=None, action=None, error=None, extra_events=()) -> dict:
        events = []
        for entry in state.runtime.history_snapshot[archive_start:]:
            if entry.kind in ("tool_exchange", "summary_completion", "event"):
                events.append({"kind": entry.kind, "sequence": entry.sequence, "text": entry.text,
                               "messages": chat_messages(entry.messages), "usage": _usage_dict(entry.usage)})
        events.extend(state.model.events)
        events.extend(extra_events)
        usages = state.model.usages
        action_usages = [usage for purpose, usage in usages if purpose == "action"]
        summary_usages = [usage for purpose, usage in usages if purpose == "summary"]
        raw_final = state.model.calls[-1].get("completion", final) if state.model.events else final
        trace = {"assistant_raw": raw_final, "action_extracted": action,
            "usage": _total_usage([usage for _, usage in usages]),
            "model_requests": state.model.calls, "agent_events": events,
            "agent_usage": {"action_calls": sum(call["purpose"] == "action" for call in state.model.calls),
                            "summary_calls": sum(call["purpose"] == "summary" for call in state.model.calls),
                            "action": _total_usage(action_usages), "summary": _total_usage(summary_usages)},
            "simple_agent": {**self.configuration_summary(), "turn": state.turn, "attempt": state.attempt}}
        if error is not None:
            trace["error_type"] = type(error).__name__
        return trace

    def forward_with_trace(self, obs: dict) -> tuple[dict, dict]:
        with self._lock:
            responses, traces = {}, {}
            self.last_trace_by_agent = traces
            for agent_idx, obs_dict in (obs or {}).items():
                if isinstance(obs_dict, dict) and obs_dict.get("skip_infer"):
                    responses[agent_idx] = {"action": None, "assistant": None}
                    traces[agent_idx] = {"skipped": True, "assistant_raw": None, "action_extracted": None}
                    continue
                state, request, dynamic, attempt, original_messages = self._prepare(agent_idx, obs_dict)
                if self._loop is None:
                    self._loop = _LoopOwner()
                archive_start = len(state.runtime.history_snapshot)
                # Request traces belong to this decision, not to an additional
                # unbounded copy of every previous full model context.
                state.model.calls = []
                state.model.usages = []
                state.model.events = []
                # Commit the accepted input's identity even if model work fails:
                # the runtime forbids retrying an already attempted action input.
                state.turn += 1
                state.attempt = attempt
                state.pins = tuple(message for message in request.messages if message.role in ("system", "developer"))
                if not state.initialized:
                    state.env_pins = tuple(message for message in original_messages if message.role in ("system", "developer"))
                state.initialized = True
                try:
                    result = self._loop.run(self._complete(state, request))
                except Exception as exc:
                    traces[agent_idx] = self._trace(state, archive_start, error=exc)
                    raise
                action = APIAgent._extract_action(result.text)
                state.last_final = result.text
                try:
                    provider_final = state.model.calls[-1].get("completion", result.text) if state.model.events else result.text
                    snapshot_event = self._publish(state, request.identity, original_messages, provider_final, action)
                except AgentLimitError as exc:
                    traces[agent_idx] = self._trace(state, archive_start, final=result.text, error=exc,
                        extra_events=({"kind": "snapshot", "status": "capacity_exhausted", "turn": state.turn},))
                    raise
                traces[agent_idx] = self._trace(state, archive_start, final=result.text, action=action, extra_events=(snapshot_event,))
                responses[agent_idx] = {"action": action, "assistant": result.text}
            return responses, traces

    def forward_with_details(self, obs: dict) -> dict:
        return self.forward_with_trace(obs)[0]

    def forward(self, obs: dict) -> dict:
        return {agent_idx: value["action"] for agent_idx, value in self.forward_with_details(obs).items()}

    def reset(self, obs: Any = None) -> None:
        """Start a fresh case; no observation is consumed until forward()."""
        with self._lock:
            if self._closed:
                # Serial evaluation closes each case but reuses its agent
                # factory object. Explicit reset starts a new owned client/loop.
                self._closed = False
                if self._owns_backend:
                    self._backend = _OpenAIModel(self)
            for state in self._states.values():
                state.runtime.close()
                if state.directory is not None:
                    state.directory.cleanup()
            self._states.clear()
            self._case_id = uuid.uuid4().hex
            self.last_trace_by_agent = {}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.reset()
            if self._loop is not None:
                try:
                    if self._owns_backend:
                        self._loop.run(self._backend.close())
                finally:
                    self._loop.close()
                    self._loop = None
                    self._closed = True
            self._closed = True
