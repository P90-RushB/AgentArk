"""Codex SDK backed AgentArk evaluation agent.

The Codex SDK is not an OpenAI-compatible HTTP endpoint. This agent adapts the
OpenAI-style message payloads produced by AgentArk into Codex text and image
inputs and runs them through local Codex SDK threads.
By default, one Codex thread is kept per AgentArk agent for the duration of an
evaluation case; use thread_mode="per_turn" for stateless, fresh-thread calls.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from tempfile import TemporaryDirectory
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
    black_box_playtest: bool = False
    isolated_cwd: bool = False


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

    _PLAYER_FEEDBACK_PROMPT = """
The environment rollout is now over. Do not output another action.

Act as a black-box playtest participant reporting only problems that were
directly observable while playing from the task prompt, image observations,
visible/history messages, and action responses. Do not inspect files, source
code, prefabs, configs, hidden state, oracle trajectories, or validation
artifacts, and do not use shell or other tools to obtain task internals.

Not completing the task is not itself a defect. Do not report difficulty,
player mistakes, the need to explore, or the absence of an immediately obvious
solution as task problems. Report a candidate defect only when the play
experience provides concrete evidence such as a prompt/observation
contradiction, missing or misleading visual information, an apparently ignored
or incorrectly applied valid action, broken interaction feedback, impossible
to read state, or an inconsistent visible state transition.

Some tasks intentionally reveal information through interaction, later steps,
or later attempts. Initial uncertainty or incomplete information can therefore
be the intended exploration/learning loop rather than a defect. Judge this from
the player-visible prompt, progression, attempt transitions, and history across
all attempts you actually experienced. Treat it as a defect only when visible
evidence shows that promised/necessary information never becomes discoverable,
or the observed reveal behavior contradicts the player-facing contract.

Return exactly one <player_feedback>...</player_feedback> block containing
valid JSON with this shape:
{
  "summary": "short overall play-experience summary",
  "information_reveal_assessment": {
    "classification": "complete_initially",
    "evidence": "why the information flow appears complete, intentionally exploratory, defective, or still unclear",
    "attempts_considered": [1]
  },
  "task_defects": [
    {
      "category": "action_execution",
      "severity": "major",
      "confidence": "high",
      "attempt": 1,
      "first_observed_turn": 0,
      "action": "exact action payload or concise action issued on that turn",
      "evidence": "what was directly observed before/after which action",
      "expected": "what the prompt or visible affordance led the player to expect",
      "observed": "what visibly happened instead"
    }
  ],
  "non_defect_observations": ["difficulty, exploration, or player-error notes kept separate"],
  "uncertainties": ["anything that needs another black-box run before calling it a defect"]
}
Allowed category values are: description_observation_conflict,
visual_observability, action_execution, interaction_feedback,
visible_state_transition, and other. Allowed severity values are: blocking,
major, and minor. Allowed confidence values are: high, medium, and low.
Allowed information_reveal_assessment.classification values are:
complete_initially, intentional_exploration,
suspected_missing_information_defect, and unclear. When classification is
intentional_exploration, keep it in non_defect_observations. When it is
suspected_missing_information_defect, include a task_defects entry with concrete
player-visible evidence. When it is unclear, use uncertainties rather than
asserting a defect.
Use an empty task_defects list when no concrete task problem was observed.
""".strip()

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
        black_box_playtest: bool = False,
        isolated_cwd: bool = False,
    ) -> None:
        super().__init__(name)
        if cwd and isolated_cwd:
            raise ValueError("CodexAgent cwd and isolated_cwd cannot both be set")
        self.config = CodexAgentConfig(
            model=model,
            sandbox=sandbox,
            timeout_s=timeout_s,
            reasoning_effort=self._normalize_reasoning_effort(reasoning_effort),
            codex_bin=(codex_bin or None),
            cwd=(cwd or None),
            thread_mode=(thread_mode or "per_agent").strip().lower() or "per_agent",
            black_box_playtest=bool(black_box_playtest),
            isolated_cwd=bool(isolated_cwd),
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
        self._isolated_cwd_cm: Optional[TemporaryDirectory[str]] = None

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
        if self._isolated_cwd_cm is not None:
            self._isolated_cwd_cm.cleanup()
            self._isolated_cwd_cm = None

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

    def collect_player_feedback(self, obs_dict: Dict[str, Any], *, agent_idx: int) -> Dict[str, Any]:
        """Collect a post-rollout report from the same black-box player thread."""

        if self.config.thread_mode != "per_agent":
            raise ValueError("Codex player feedback requires thread_mode='per_agent'")
        if not self.config.black_box_playtest:
            raise ValueError("Codex player feedback requires black_box_playtest=True")

        messages = APIAgent._obs_to_messages(self, obs_dict or {})
        rendered = self._messages_to_codex_prompt(
            messages,
            agent_idx=int(agent_idx),
            purpose="feedback",
        )
        rendered.prompt = f"{rendered.prompt}\n\n{self._PLAYER_FEEDBACK_PROMPT}".strip()
        run_output = self._run_codex_with_timeout(rendered, agent_idx=int(agent_idx))
        report, parse_error = self._parse_player_feedback(run_output.text)
        result: Dict[str, Any] = {
            "status": "ok" if report is not None else "unparsed",
            "assistant_raw": run_output.text,
            "report": report,
        }
        if parse_error is not None:
            result["parse_error"] = parse_error
        usage = self._usage_to_dict(run_output.usage)
        if usage is not None:
            result["usage"] = usage
        return result

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
            "cwd": self._resolve_cwd(),
        }
        sandbox = self._sandbox_value(self.config.sandbox)
        if sandbox is not None:
            kwargs["sandbox"] = sandbox
        thread = codex.thread_start(**kwargs)
        if self.config.thread_mode == "per_agent":
            self._threads[agent_idx] = thread
        return thread

    def _resolve_cwd(self) -> str:
        if not self.config.isolated_cwd:
            return self.config.cwd or os.getcwd()
        if self._isolated_cwd_cm is None:
            self._isolated_cwd_cm = TemporaryDirectory(prefix="agentark-black-box-player-")
        return self._isolated_cwd_cm.name

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

    def _messages_to_codex_prompt(
        self,
        messages: List[dict],
        *,
        agent_idx: int,
        purpose: str = "action",
    ) -> CodexRenderedPrompt:
        self._turn_index += 1
        turn_index = self._turn_index
        if purpose == "feedback":
            lines = [
                "The following is the final visible state of an AgentArk black-box playtest.",
                "The action rollout has ended; respond with player feedback, not another environment action.",
                "Image observations, when present, are attached as Codex SDK image inputs and referenced inline.",
                "",
            ]
        else:
            lines = [
                "The following is an AgentArk model-evaluation chat transcript.",
                "Answer as the assistant for the next environment action.",
                "Image observations, when present, are attached as Codex SDK image inputs and referenced inline.",
                "",
            ]
            if self.config.black_box_playtest:
                lines.extend([
                    "This is a black-box player run. Use only the transcript and attached observations.",
                    "Do not inspect files, repositories, source code, configs, hidden state, or oracle artifacts,",
                    "and do not use shell or other tools to seek task-internal information.",
                    "",
                ])
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

    @staticmethod
    def _parse_player_feedback(text: Any) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        raw = str(text or "").strip()
        match = re.search(r"<player_feedback>\s*(.*?)\s*</player_feedback>", raw, flags=re.DOTALL)
        payload = match.group(1).strip() if match else raw
        if payload.startswith("```") and payload.endswith("```"):
            payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
            payload = re.sub(r"\s*```$", "", payload)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            return None, f"invalid player feedback JSON: {exc}"
        if not isinstance(parsed, dict):
            return None, "player feedback JSON must be an object"
        expected_types = {
            "summary": str,
            "information_reveal_assessment": dict,
            "task_defects": list,
            "non_defect_observations": list,
            "uncertainties": list,
        }
        for key, expected_type in expected_types.items():
            if not isinstance(parsed.get(key), expected_type):
                return None, f"player feedback field {key!r} must be {expected_type.__name__}"
        if not parsed["summary"].strip():
            return None, "player feedback field 'summary' must not be empty"
        reveal = parsed["information_reveal_assessment"]
        reveal_classification = reveal.get("classification")
        allowed_reveal_classifications = {
            "complete_initially",
            "intentional_exploration",
            "suspected_missing_information_defect",
            "unclear",
        }
        if reveal_classification not in allowed_reveal_classifications:
            return None, (
                "player feedback information_reveal_assessment.classification must be one of: "
                f"{', '.join(sorted(allowed_reveal_classifications))}"
            )
        reveal_evidence = reveal.get("evidence")
        if not isinstance(reveal_evidence, str) or not reveal_evidence.strip():
            return None, "player feedback information_reveal_assessment.evidence must be a non-empty string"
        attempts_considered = reveal.get("attempts_considered")
        if not isinstance(attempts_considered, list) or not attempts_considered:
            return None, "player feedback information_reveal_assessment.attempts_considered must be a non-empty list"
        for index, attempt in enumerate(attempts_considered):
            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
                return None, (
                    "player feedback information_reveal_assessment."
                    f"attempts_considered[{index}] must be a positive integer"
                )
        for list_key in ("non_defect_observations", "uncertainties"):
            for index, item in enumerate(parsed[list_key]):
                if not isinstance(item, str):
                    return None, f"player feedback {list_key}[{index}] must be str"
        required_defect_fields = {
            "category",
            "severity",
            "confidence",
            "attempt",
            "first_observed_turn",
            "action",
            "evidence",
            "expected",
            "observed",
        }
        for index, defect in enumerate(parsed["task_defects"]):
            if not isinstance(defect, dict):
                return None, f"player feedback task_defects[{index}] must be an object"
            missing = sorted(required_defect_fields.difference(defect.keys()))
            if missing:
                return None, f"player feedback task_defects[{index}] is missing: {', '.join(missing)}"
            allowed_values = {
                "category": {
                    "description_observation_conflict",
                    "visual_observability",
                    "action_execution",
                    "interaction_feedback",
                    "visible_state_transition",
                    "other",
                },
                "severity": {"blocking", "major", "minor"},
                "confidence": {"high", "medium", "low"},
            }
            for field, allowed in allowed_values.items():
                if not isinstance(defect.get(field), str) or defect[field] not in allowed:
                    return None, (
                        f"player feedback task_defects[{index}].{field} must be one of: "
                        f"{', '.join(sorted(allowed))}"
                    )
            attempt = defect.get("attempt")
            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
                return None, f"player feedback task_defects[{index}].attempt must be a positive integer"
            turn = defect.get("first_observed_turn")
            if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
                return None, (
                    f"player feedback task_defects[{index}].first_observed_turn "
                    "must be a non-negative integer"
                )
            for field in ("action", "evidence", "expected", "observed"):
                value = defect.get(field)
                if not isinstance(value, str) or not value.strip():
                    return None, (
                        f"player feedback task_defects[{index}].{field} "
                        "must be a non-empty string"
                    )
        if reveal_classification == "intentional_exploration" and not parsed["non_defect_observations"]:
            return None, "intentional exploration must be recorded in non_defect_observations"
        if reveal_classification == "suspected_missing_information_defect" and not parsed["task_defects"]:
            return None, "suspected missing information must include a concrete task_defects entry"
        if reveal_classification == "unclear" and not parsed["uncertainties"]:
            return None, "unclear information reveal must be recorded in uncertainties"
        return parsed, None

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
