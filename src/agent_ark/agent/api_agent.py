"""Stateless LLM API agent.

This agent is intentionally *stateless*: it does not maintain per-episode chat history.
Environment-side context management (see EnvWrapper/ContextManager) is responsible for
providing any required conversation/messages.

Expected obs schema (per ml_id):
    - step_msg: str
    - vis: List[List[PIL.Image.Image]]  # cameras -> frames
    - messages: Optional[List[dict]]     # OpenAI-compatible messages (if enabled by env)
    - skip_infer: bool

Return value:
    - Dict[ml_id, action_str]
        - In action_mode == 'code': action_str should be a single C# script.
        - In action_mode == 'func': action_str should usually be a
            <tool_call>{"name":"...","arguments":{...}}</tool_call> block.
            Legacy raw JSON / <params>...</params> payloads remain supported for compatibility.
"""

from __future__ import annotations

import os
import json
import queue
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from openai import OpenAI

from .base_agent import BaseAgent
from agent_ark.utils.image_utils import pil_image_to_base64
from agent_ark.utils.parse_utils import extract_tag_content


@dataclass
class LLMClientConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "qwen3-vl-plus"
    provider: str = "auto"
    timeout_s: Optional[float] = 180.0
    max_retries: int = 2


class LLMClient:
    def __init__(self, config: LLMClientConfig) -> None:
        base_url = (config.base_url or "").strip() or None
        provider = (config.provider or "auto").strip().lower() or "auto"

        api_key = (config.api_key or "").strip() or None
        if api_key is None:
            # Common env var fallbacks
            api_key = (
                os.getenv("OPENROUTER_API_KEY")
                or os.getenv("DASHSCOPE_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )

        # Keep behaviour consistent with older agent: if base_url missing, default to DashScope-compatible.
        if base_url is None:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": int(config.max_retries),
        }
        if config.timeout_s is not None:
            client_kwargs["timeout"] = float(config.timeout_s)
        self._client = OpenAI(**client_kwargs)
        self.model = config.model
        self.provider = provider
        self.timeout_s = float(config.timeout_s) if config.timeout_s is not None else None

    @property
    def base_url_host(self) -> str:
        try:
            parsed = urlparse(str(self._client.base_url))
            return (parsed.hostname or "").lower()
        except Exception:
            return ""

    def chat_completions_create(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "extra_body": extra_body or {},
        }
        if self.timeout_s is not None:
            kwargs["timeout"] = float(self.timeout_s)
        return self._client.chat.completions.create(**kwargs)


class APIAgent(BaseAgent):
    """Stateless API agent.

    Notes:
      - Does NOT store message history.
      - If obs contains `messages`, those are sent as-is.
      - Otherwise, a minimal user message is built from step_msg + latest camera frames.
    """

    def __init__(
        self,
        name: str = "APIAgent",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "qwen3-vl-plus",
        temperature: float = 0.2,
        provider: str = "auto",
        timeout_s: Optional[float] = 180.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(name)
        self.client = LLMClient(LLMClientConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
            timeout_s=timeout_s,
            max_retries=max_retries,
        ))
        self.temperature = float(temperature)


    def reset(self) -> None:
        # Intentionally stateless.
        return

    def build_request_messages(self, obs: Dict[int, dict]) -> Dict[int, Optional[List[dict]]]:
        requests: Dict[int, Optional[List[dict]]] = {}
        for agent_idx, obs_dict in (obs or {}).items():
            if isinstance(obs_dict, dict) and obs_dict.get("skip_infer"):
                requests[agent_idx] = None
                continue
            requests[agent_idx] = deepcopy(self._obs_to_messages(obs_dict))
        return requests

    def forward_with_trace(
        self,
        obs: Dict[int, dict],
    ) -> tuple[Dict[int, Dict[str, Optional[str]]], Dict[int, Dict[str, Any]]]:
        responses: Dict[int, Dict[str, Optional[str]]] = {}
        trace_by_agent: Dict[int, Dict[str, Any]] = {}

        for agent_idx, o in (obs or {}).items():
            if isinstance(o, dict) and o.get("skip_infer"):
                responses[agent_idx] = {
                    "action": None,
                    "assistant": None,
                }
                trace_by_agent[agent_idx] = {
                    "skipped": True,
                    "assistant_raw": None,
                    "action_extracted": None,
                }
                continue

            messages = self._obs_to_messages(o)
            response_text, usage = self._call_api_with_usage(messages)
            action_text = self._extract_action(response_text)
            responses[agent_idx] = {
                "action": action_text,
                "assistant": response_text,
            }
            trace_by_agent[agent_idx] = {
                "assistant_raw": response_text,
                "action_extracted": action_text,
                "usage": usage,
            }

        return responses, trace_by_agent

    def forward_with_details(self, obs: Dict[int, dict]) -> Dict[int, Dict[str, Optional[str]]]:
        responses, _ = self.forward_with_trace(obs)
        return responses

    def forward(self, obs: Dict[int, dict]) -> Dict[int, Optional[str]]:
        code_action: Dict[int, Optional[str]] = {}

        for agent_idx, payload in self.forward_with_details(obs).items():
            action_text = payload.get("action") if isinstance(payload, dict) else None
            code_action[agent_idx] = action_text

        return code_action

    def _obs_to_messages(self, obs_dict: dict) -> List[dict]:
        # Preferred path: env already built messages with context/history.
        msgs = obs_dict.get("messages") if isinstance(obs_dict, dict) else None
        if isinstance(msgs, list) and msgs:
            return msgs

        step_msg = ""
        if isinstance(obs_dict, dict):
            step_msg = obs_dict.get("step_msg", "") or ""

        parts: List[dict] = []
        if step_msg:
            parts.append({"type": "text", "text": str(step_msg)})
        else:
            parts.append({"type": "text", "text": ""})

        # Attach latest frame per camera (keep it small/fast).
        vis = obs_dict.get("vis") if isinstance(obs_dict, dict) else None
        if isinstance(vis, list):
            for cam_frames in vis:
                if not isinstance(cam_frames, list) or not cam_frames:
                    continue
                last_frame = cam_frames[-1]
                try:
                    img_b64 = pil_image_to_base64(last_frame)
                except Exception:
                    continue
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })

        return [{"role": "user", "content": parts}]

    def _call_api(self, messages: List[dict]) -> str:
        response_text, _ = self._call_api_with_usage(messages)
        return response_text

    def _call_api_with_usage(self, messages: List[dict]) -> tuple[str, Optional[Dict[str, Any]]]:
        extra_body = self._request_extra_body()
        empty_choices_retries = max(0, int(os.getenv("AGENTARK_EMPTY_CHOICES_RETRIES", "3") or "0"))
        empty_choices_delay_s = max(0.0, float(os.getenv("AGENTARK_EMPTY_CHOICES_RETRY_DELAY_S", "2") or "0"))
        last_completion: Any = None
        for attempt in range(1, empty_choices_retries + 2):
            completion = self._run_api_request_with_timeout(
                messages=messages,
                temperature=self.temperature,
                extra_body=extra_body,
            )
            choices = getattr(completion, "choices", None)
            if choices:
                response_msg = choices[0].message
                content = self._response_text_field(response_msg, "content")
                reasoning = self._response_text_field(response_msg, "reasoning", "reasoning_content")
                response_text = self._combine_reasoning_and_content(reasoning=reasoning, content=content)
                return response_text, self._usage_to_jsonable(getattr(completion, "usage", None))
            last_completion = completion
            self._record_empty_choices_response(completion=completion, messages=messages, attempt=attempt)
            if attempt <= empty_choices_retries and empty_choices_delay_s > 0:
                time.sleep(empty_choices_delay_s)

        raise RuntimeError(
            "LLM chat completion returned no choices after "
            f"{empty_choices_retries + 1} request attempt(s). "
            f"last_response={self._short_json(self._jsonable_value(last_completion), max_chars=4000)}"
        )

    def _run_api_request_with_timeout(
        self,
        *,
        messages: List[dict],
        temperature: float,
        extra_body: Dict[str, Any],
    ) -> Any:
        timeout_s = getattr(self.client, "timeout_s", None)
        if timeout_s is None or float(timeout_s) <= 0:
            return self.client.chat_completions_create(
                messages=messages,
                temperature=temperature,
                extra_body=extra_body,
            )

        result_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=1)

        def _target() -> None:
            try:
                result_queue.put((
                    "ok",
                    self.client.chat_completions_create(
                        messages=messages,
                        temperature=temperature,
                        extra_body=extra_body,
                    ),
                ))
            except BaseException as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=_target, name=f"{self.name}-api-request", daemon=True)
        thread.start()
        try:
            status, payload = result_queue.get(timeout=float(timeout_s))
        except queue.Empty as exc:
            raise TimeoutError(
                f"LLM API request exceeded timeout_s={float(timeout_s):g} "
                f"for model={getattr(self.client, 'model', None)!r}"
            ) from exc

        if status == "error":
            raise payload
        return payload

    def _request_extra_body(self) -> Dict[str, Any]:
        provider = self.client.provider
        if provider == "auto":
            provider = self._infer_provider_from_host(self.client.base_url_host)

        if provider in ("openai", "generic", "none"):
            return {}
        if provider == "openrouter":
            return {"reasoning": {"enabled": True}}
        if provider in ("dashscope", "qwen"):
            return {
                "enable_thinking": True,
                "thinking_budget": 81920,
            }
        raise ValueError(
            f"Unsupported LLM provider={self.client.provider!r}; "
            "expected auto, openai, generic, none, openrouter, dashscope, or qwen"
        )

    def _record_empty_choices_response(self, *, completion: Any, messages: List[dict], attempt: int) -> None:
        path = os.getenv("AGENTARK_API_DEBUG_RESPONSE_PATH", "").strip()
        if not path:
            return
        record = {
            "time": time.time(),
            "agent": self.name,
            "model": getattr(self.client, "model", None),
            "provider": getattr(self.client, "provider", None),
            "base_url_host": self.client.base_url_host,
            "error": "chat completion returned no choices",
            "attempt": int(attempt),
            "completion": self._jsonable_value(completion),
            "messages": self._summarize_messages_for_debug(messages),
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            return

    @staticmethod
    def _summarize_messages_for_debug(messages: Any) -> Any:
        if isinstance(messages, list):
            return [APIAgent._summarize_messages_for_debug(item) for item in messages]
        if isinstance(messages, dict):
            out: Dict[str, Any] = {}
            for key, value in messages.items():
                if key == "image_url":
                    if isinstance(value, dict):
                        image_url = dict(value)
                        url = str(image_url.get("url", ""))
                        image_url["url"] = f"<image_url chars={len(url)} prefix={url[:32]!r}>"
                        out[key] = image_url
                    else:
                        raw = str(value)
                        out[key] = f"<image_url chars={len(raw)} prefix={raw[:32]!r}>"
                else:
                    out[key] = APIAgent._summarize_messages_for_debug(value)
            return out
        if isinstance(messages, str) and len(messages) > 2000:
            return messages[:2000] + f"...<truncated chars={len(messages)}>"
        return messages

    @staticmethod
    def _short_json(value: Any, *, max_chars: int) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = repr(value)
        if len(text) > max_chars:
            return text[:max_chars] + f"...<truncated chars={len(text)}>"
        return text

    @staticmethod
    def _infer_provider_from_host(host: str) -> str:
        if host == "openrouter.ai":
            return "openrouter"
        if host == "dashscope.aliyuncs.com":
            return "dashscope"
        return "openai"

    @staticmethod
    def _response_text_field(response_msg: Any, *field_names: str) -> str:
        model_extra = getattr(response_msg, "model_extra", None)
        if not isinstance(model_extra, dict):
            model_extra = {}

        for field_name in field_names:
            value = getattr(response_msg, field_name, None)
            if value in (None, ""):
                value = model_extra.get(field_name, None)
            text = APIAgent._coerce_response_text(value)
            if text:
                return text
        return ""

    @staticmethod
    def _usage_to_jsonable(usage: Any) -> Optional[Dict[str, Any]]:
        if usage is None:
            return None
        if isinstance(usage, dict):
            return {
                str(key): APIAgent._jsonable_value(value)
                for key, value in usage.items()
            }
        model_dump = getattr(usage, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
                if isinstance(dumped, dict):
                    return {
                        str(key): APIAgent._jsonable_value(value)
                        for key, value in dumped.items()
                    }
            except Exception:
                pass
        out: Dict[str, Any] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if hasattr(usage, key):
                out[key] = APIAgent._jsonable_value(getattr(usage, key))
        return out or {"raw": APIAgent._jsonable_value(usage)}

    @staticmethod
    def _jsonable_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): APIAgent._jsonable_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [APIAgent._jsonable_value(item) for item in value]
        if isinstance(value, tuple):
            return [APIAgent._jsonable_value(item) for item in value]
        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)

    @staticmethod
    def _coerce_response_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text", item.get("content", ""))
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(part.strip() for part in parts if part and part.strip())
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value).strip()

    @staticmethod
    def _combine_reasoning_and_content(*, reasoning: str, content: str) -> str:
        reasoning = (reasoning or "").strip()
        content = (content or "").strip()
        if not reasoning:
            return content
        if not content:
            return reasoning
        if "<think" in content.lower():
            return content
        if reasoning.lower().startswith("<think"):
            return f"{reasoning}\n{content}".strip()
        return f"<think>\n{reasoning}\n</think>\n{content}".strip()

    @staticmethod
    def _extract_action(response_text: str) -> Optional[str]:
        if not response_text:
            return None

        tool_call_blocks = extract_tag_content(response_text, tag="tool_call")
        if tool_call_blocks:
            if len(tool_call_blocks) != 1:
                joined = "\n".join(b.strip() for b in tool_call_blocks if b.strip())
                return f"<tool_call>{joined}</tool_call>" if joined else None
            inner = tool_call_blocks[0].strip()
            return f"<tool_call>{inner}</tool_call>"

        params_blocks = extract_tag_content(response_text, tag="params")
        if params_blocks:
            if len(params_blocks) != 1:
                # If multiple, concatenate to keep something usable.
                joined = "\n".join(b.strip() for b in params_blocks if b.strip())
                return f"<params>{joined}</params>" if joined else None
            inner = params_blocks[0].strip()
            return f"<params>{inner}</params>"

        code_blocks = extract_tag_content(response_text, tag="code")
        if code_blocks:
            if len(code_blocks) != 1:
                return "\n\n".join(b.strip("\n") for b in code_blocks if b.strip()) or None
            return code_blocks[0].strip("\n")

        return response_text
