"""Codex SDK backed AgentArk evaluation agent.

The Codex SDK is not an OpenAI-compatible HTTP endpoint. This agent adapts the
OpenAI-style message payloads produced by AgentArk into Codex text and image
inputs and runs them through local Codex SDK threads.
By default, one Codex thread is kept per AgentArk agent for the duration of an
evaluation case; use thread_mode="per_turn" for stateless, fresh-thread calls.
"""

from __future__ import annotations

import os
import queue
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional

from .api_agent import APIAgent
from .base_agent import BaseAgent


@dataclass
class CodexAgentConfig:
    model: str = "gpt-5.5"
    sandbox: str = "read_only"
    timeout_s: Optional[float] = 600.0
    reasoning_effort: Optional[str] = None
    codex_bin: Optional[str] = None
    cwd: Optional[str] = None
    thread_mode: str = "per_agent"


@dataclass
class CodexRenderedPrompt:
    prompt: str
    image_urls: List[str]


@dataclass
class CodexRunOutput:
    text: str
    usage: Any = None


class CodexAgent(BaseAgent):
    """Run AgentArk actions through the local Codex Python SDK."""

    def __init__(
        self,
        name: str = "CodexAgent",
        model: str = "gpt-5.5",
        sandbox: str = "read_only",
        timeout_s: Optional[float] = 600.0,
        reasoning_effort: Optional[str] = None,
        codex_bin: Optional[str] = None,
        cwd: Optional[str] = None,
        thread_mode: str = "per_agent",
    ) -> None:
        super().__init__(name)
        self.config = CodexAgentConfig(
            model=model,
            sandbox=sandbox,
            timeout_s=timeout_s,
            reasoning_effort=self._normalize_reasoning_effort(reasoning_effort),
            codex_bin=(codex_bin or None),
            cwd=(cwd or None),
            thread_mode=(thread_mode or "per_agent").strip().lower() or "per_agent",
        )
        if self.config.thread_mode not in ("per_turn", "per_agent"):
            raise ValueError("CodexAgent thread_mode must be 'per_turn' or 'per_agent'")

        self._codex_cm: Any = None
        self._codex: Any = None
        self._sandbox_cls: Any = None
        self._text_input_cls: Any = None
        self._image_input_cls: Any = None
        self._threads: Dict[int, Any] = {}
        self._turn_index = 0

    def reset(self, obs: Any = None) -> None:
        self._threads.clear()

    def close(self) -> None:
        self._threads.clear()
        if self._codex_cm is not None:
            exit_fn = getattr(self._codex_cm, "__exit__", None)
            if callable(exit_fn):
                exit_fn(None, None, None)
            else:
                close_fn = getattr(self._codex_cm, "close", None)
                if callable(close_fn):
                    close_fn()
        self._codex_cm = None
        self._codex = None

    def build_request_messages(self, obs: Dict[int, dict]) -> Dict[int, Optional[List[dict]]]:
        requests: Dict[int, Optional[List[dict]]] = {}
        for agent_idx, obs_dict in (obs or {}).items():
            if isinstance(obs_dict, dict) and obs_dict.get("skip_infer"):
                requests[agent_idx] = None
                continue
            requests[agent_idx] = deepcopy(APIAgent._obs_to_messages(self, obs_dict))
        return requests

    def forward_with_trace(
        self,
        obs: Dict[int, dict],
    ) -> tuple[Dict[int, Dict[str, Optional[str]]], Dict[int, Dict[str, Any]]]:
        responses: Dict[int, Dict[str, Optional[str]]] = {}
        trace_by_agent: Dict[int, Dict[str, Any]] = {}

        for agent_idx, obs_dict in (obs or {}).items():
            if isinstance(obs_dict, dict) and obs_dict.get("skip_infer"):
                responses[agent_idx] = {"action": None, "assistant": None}
                trace_by_agent[agent_idx] = {
                    "skipped": True,
                    "assistant_raw": None,
                    "action_extracted": None,
                }
                continue

            messages = APIAgent._obs_to_messages(self, obs_dict)
            rendered = self._messages_to_codex_prompt(messages, agent_idx=int(agent_idx))
            run_output = self._run_codex_with_timeout(rendered, agent_idx=int(agent_idx))
            response_text = run_output.text
            action_text = APIAgent._extract_action(response_text)
            responses[agent_idx] = {
                "action": action_text,
                "assistant": response_text,
            }
            trace_by_agent[agent_idx] = {
                "assistant_raw": response_text,
                "action_extracted": action_text,
            }
            usage = self._usage_to_dict(run_output.usage)
            if usage is not None:
                trace_by_agent[agent_idx]["usage"] = usage

        return responses, trace_by_agent

    def forward_with_details(self, obs: Dict[int, dict]) -> Dict[int, Dict[str, Optional[str]]]:
        responses, _ = self.forward_with_trace(obs)
        return responses

    def forward(self, obs: Dict[int, dict]) -> Dict[int, Optional[str]]:
        return {
            agent_idx: payload.get("action") if isinstance(payload, dict) else None
            for agent_idx, payload in self.forward_with_details(obs).items()
        }

    def _ensure_codex(self) -> Any:
        if self._codex is not None:
            return self._codex

        try:
            from openai_codex import Codex, Sandbox  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "provider: codex requires the optional Codex SDK. "
                "Install it with `pip install openai-codex` or `pip install -e .[codex]`."
            ) from exc

        self._sandbox_cls = Sandbox
        try:
            from openai_codex import TextInput  # type: ignore
            self._text_input_cls = TextInput
        except ImportError:
            self._text_input_cls = None
        try:
            from openai_codex import ImageInput  # type: ignore
            self._image_input_cls = ImageInput
        except ImportError:
            self._image_input_cls = None
        self._codex_cm = self._create_codex_context(Codex)
        enter_fn = getattr(self._codex_cm, "__enter__", None)
        self._codex = enter_fn() if callable(enter_fn) else self._codex_cm
        return self._codex

    def _create_codex_context(self, codex_cls: Any) -> Any:
        codex_bin = self.config.codex_bin
        if not codex_bin:
            return codex_cls()

        try:
            from openai_codex import CodexConfig  # type: ignore
        except ImportError:
            return codex_cls()

        config = CodexConfig(codex_bin=codex_bin)
        for kwargs in ({"config": config}, {"codex_config": config}):
            try:
                return codex_cls(**kwargs)
            except TypeError:
                pass
        return codex_cls(config)

    def _run_codex_with_timeout(self, rendered: CodexRenderedPrompt, *, agent_idx: int) -> CodexRunOutput:
        timeout_s = self.config.timeout_s
        if timeout_s is None or float(timeout_s) <= 0:
            return self._run_codex(rendered, agent_idx=agent_idx)

        result_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=1)

        def _target() -> None:
            try:
                result_queue.put(("ok", self._run_codex(rendered, agent_idx=agent_idx)))
            except BaseException as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=_target, name=f"{self.name}-codex-request", daemon=True)
        thread.start()
        try:
            status, payload = result_queue.get(timeout=float(timeout_s))
        except queue.Empty as exc:
            raise TimeoutError(
                f"Codex SDK request exceeded timeout_s={float(timeout_s):g} "
                f"for model={self.config.model!r}"
            ) from exc

        if status == "error":
            raise payload
        if isinstance(payload, CodexRunOutput):
            return payload
        return CodexRunOutput(text=str(payload or ""))

    def _run_codex(self, rendered: CodexRenderedPrompt, *, agent_idx: int) -> CodexRunOutput:
        thread = self._get_thread(agent_idx)
        kwargs: Dict[str, Any] = {}
        if self.config.reasoning_effort is not None:
            kwargs["effort"] = self.config.reasoning_effort
        result = thread.run(self._to_sdk_input(rendered), **kwargs)
        return CodexRunOutput(
            text=self._result_final_response(result),
            usage=getattr(result, "usage", None),
        )

    def _to_sdk_input(self, rendered: CodexRenderedPrompt) -> Any:
        text_input_cls = self._text_input_cls
        image_input_cls = self._image_input_cls

        if text_input_cls is None:
            return rendered.prompt

        if rendered.image_urls and image_input_cls is not None:
            items = [text_input_cls(text=rendered.prompt)]
            items.extend(image_input_cls(url=url) for url in rendered.image_urls)
            return items

        if not rendered.image_urls:
            return rendered.prompt

        return [text_input_cls(text=rendered.prompt)]

    def _get_thread(self, agent_idx: int) -> Any:
        if self.config.thread_mode == "per_agent" and agent_idx in self._threads:
            return self._threads[agent_idx]

        codex = self._ensure_codex()
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "cwd": self.config.cwd or os.getcwd(),
        }
        sandbox = self._sandbox_value(self.config.sandbox)
        if sandbox is not None:
            kwargs["sandbox"] = sandbox
        thread = codex.thread_start(**kwargs)
        if self.config.thread_mode == "per_agent":
            self._threads[agent_idx] = thread
        return thread

    def _sandbox_value(self, name: str) -> Any:
        sandbox_cls = self._sandbox_cls
        if sandbox_cls is None:
            return None
        normalized = (name or "").strip().lower().replace("-", "_")
        if not normalized:
            return None
        return getattr(sandbox_cls, normalized, None)

    def _messages_to_prompt(self, messages: List[dict], *, agent_idx: int) -> str:
        return self._messages_to_codex_prompt(messages, agent_idx=agent_idx).prompt

    def _messages_to_codex_prompt(self, messages: List[dict], *, agent_idx: int) -> CodexRenderedPrompt:
        self._turn_index += 1
        turn_index = self._turn_index
        lines = [
            "The following is an AgentArk model-evaluation chat transcript.",
            "Answer as the assistant for the next environment action.",
            "Image observations, when present, are attached as Codex SDK image inputs and referenced inline.",
            "",
        ]
        image_urls_out: List[str] = []
        for idx, message in enumerate(messages or []):
            role = str(message.get("role", "message") if isinstance(message, dict) else "message")
            content = message.get("content", "") if isinstance(message, dict) else message
            rendered, image_urls = self._render_message_content(
                content,
                agent_idx=agent_idx,
                turn_index=turn_index,
                message_index=idx,
            )
            lines.append(f"[{role}]")
            lines.append(rendered)
            lines.append("")
            image_urls_out.extend(image_urls)
        return CodexRenderedPrompt(
            prompt="\n".join(lines).strip(),
            image_urls=image_urls_out,
        )

    def _render_message_content(
        self,
        content: Any,
        *,
        agent_idx: int,
        turn_index: int,
        message_index: int,
    ) -> tuple[str, List[str]]:
        if isinstance(content, str):
            return content, []
        if not isinstance(content, list):
            return str(content), []

        parts: List[str] = []
        image_urls: List[str] = []
        image_index = 0
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue

            part_type = str(part.get("type", "") or "").lower()
            if part_type == "text":
                parts.append(str(part.get("text", "") or ""))
                continue
            if part_type == "image_url":
                url = ""
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = str(image_url.get("url", "") or "")
                elif image_url is not None:
                    url = str(image_url)
                image_ref, image_url_out = self._render_image_url(
                    url,
                    agent_idx=agent_idx,
                    turn_index=turn_index,
                    message_index=message_index,
                    image_index=image_index,
                )
                image_index += 1
                parts.append(image_ref)
                if image_url_out:
                    image_urls.append(image_url_out)
                continue

            parts.append(str(part))
        return "\n".join(part for part in parts if part != ""), image_urls

    def _render_image_url(
        self,
        url: str,
        *,
        agent_idx: int,
        turn_index: int,
        message_index: int,
        image_index: int,
    ) -> tuple[str, Optional[str]]:
        if url:
            return (
                f"[image input: agent={agent_idx} turn={turn_index} message={message_index} image={image_index}]",
                url,
            )
        return "[image_url: missing]", None

    @staticmethod
    def _result_final_response(result: Any) -> str:
        value = getattr(result, "final_response", None)
        if value is not None:
            return str(value).strip()
        if isinstance(result, dict):
            for key in ("final_response", "response", "text", "output"):
                if result.get(key) is not None:
                    return str(result[key]).strip()
        return str(result).strip()

    @staticmethod
    def _usage_to_dict(usage: Any) -> Optional[Dict[str, Any]]:
        if usage is None:
            return None
        value = CodexAgent._usage_to_plain(usage)
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        return {"value": value}

    @staticmethod
    def _usage_to_plain(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): CodexAgent._usage_to_plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [CodexAgent._usage_to_plain(item) for item in value]
        if is_dataclass(value):
            return CodexAgent._usage_to_plain(asdict(value))

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return CodexAgent._usage_to_plain(model_dump(mode="json"))
            except TypeError:
                return CodexAgent._usage_to_plain(model_dump())

        dict_fn = getattr(value, "dict", None)
        if callable(dict_fn):
            try:
                return CodexAgent._usage_to_plain(dict_fn())
            except TypeError:
                pass

        out: Dict[str, Any] = {}
        for key in (
            "last",
            "total",
            "model_context_window",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "cached_input_tokens",
            "reasoning_output_tokens",
        ):
            if hasattr(value, key):
                out[key] = CodexAgent._usage_to_plain(getattr(value, key))
        return out or str(value)

    @staticmethod
    def _normalize_reasoning_effort(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        normalized = str(value).strip().lower()
        allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
        if normalized not in allowed:
            raise ValueError(
                "CodexAgent reasoning_effort must be one of: "
                "none, minimal, low, medium, high, xhigh"
            )
        return normalized
