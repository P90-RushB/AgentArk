"""Closed native final-action transport; it never executes environment actions."""
from __future__ import annotations

import json
import math
import re
from importlib.resources import files

from .models import ModelResponseError
from .types import Message, ModelCompletion, TextPart, ToolSpec


SUBMISSION_NAME = "submit_action"
SUBMISSION_SPEC = ToolSpec(
    SUBMISSION_NAME,
    "Finish this decision by selecting one environment action. This selects an action only; "
    "the environment executes it later and reports the outcome in the next observation. "
    "Use tool_call for inner JSON {name,arguments}, params for legacy parameter content, "
    "code for the complete plain C# source, or raw for a delimiter-free legacy action.",
    json.dumps({"type": "object", "properties": {
        "format": {"type": "string", "enum": ["tool_call", "params", "code", "raw"]},
        "content": {"type": "string"}}, "required": ["format", "content"],
        "additionalProperties": False}, separators=(",", ":")),
)

NATIVE_INSTRUCTIONS = files(__package__).joinpath("info", "native_system_prompt.txt").read_text(encoding="utf-8")

_ENVELOPE = re.compile(r"<\s*/?\s*(?:tool_call|params|code)\b[^>]*>", re.IGNORECASE)


def _error(detail: str) -> ModelResponseError:
    return ModelResponseError("invalid submit_action: " + detail)


def _strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise _error("duplicate JSON fields")
            result[key] = value
        return result

    def constant(value):
        raise _error("nonfinite JSON numbers")

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise _error("nonfinite JSON numbers")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        finite(value)
        return value
    except ModelResponseError:
        raise
    except (ValueError, TypeError, RecursionError):
        raise _error("malformed JSON") from None


def _unreachable_submission(args, session):
    raise _error("submission cannot execute as a host tool")


def register_submission(registry) -> None:
    """Reserve the declaration before caller-owned host tool registration."""
    registry.register(SUBMISSION_SPEC, _unreachable_submission)


def translate_submission(completion: ModelCompletion, *, max_bytes: int, max_characters: int):
    """Return (completion, event) while preserving the provider's native pair.

    A final submission consumes one finite control slot per decision. It never
    enters ToolRegistry.execute, and ordinary host calls remain runtime-owned.
    """
    calls = [call for call in completion.tool_calls if call.name == SUBMISSION_NAME]
    if not calls:
        return completion, None
    if len(calls) != 1 or len(completion.tool_calls) != 1 or completion.context_messages:
        raise _error("require exactly one standalone submission")
    call = calls[0]
    if len(call.arguments_json.encode("utf-8")) > 131072:
        raise _error("native arguments exceed their byte bound")
    arguments = _strict_json(call.arguments_json)
    if not isinstance(arguments, dict) or set(arguments) != {"format", "content"}:
        raise _error("arguments must contain only format and content")
    kind, content = arguments["format"], arguments["content"]
    if kind not in ("tool_call", "params", "code", "raw") or not isinstance(content, str) or not content.strip():
        raise _error("unsupported format or empty content")
    try:
        size = len(content.encode("utf-8"))
    except UnicodeEncodeError:
        raise _error("content must be valid Unicode") from None
    if size > max_bytes or len(content) > max_characters:
        raise _error("decoded action exceeds its bound")
    if kind == "tool_call":
        payload = _strict_json(content)
        if (not isinstance(payload, dict) or set(payload) != {"name", "arguments"}
            or not isinstance(payload["name"], str) or not payload["name"].strip()
            or not isinstance(payload["arguments"], dict)):
            raise _error("tool_call content must be an inner name/arguments object")
        # Escape literal delimiters inside JSON strings. JSON decoding by the
        # existing environment restores the exact values without nested XML.
        try:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            body = body.replace("<", "\\u003c").replace(">", "\\u003e")
            body.encode("utf-8")
        except (ValueError, UnicodeEncodeError):
            raise _error("tool_call content must be valid finite Unicode JSON") from None
        translated = "<tool_call>" + body + "</tool_call>"
    else:
        if _ENVELOPE.search(content):
            raise _error("content cannot contain nested action envelopes")
        if kind == "code" and content.lstrip().startswith("```"):
            raise _error("code content must be complete plain source without fences")
        translated = "<params>" + content + "</params>" if kind == "params" else content
    if len(translated.encode("utf-8")) > max_bytes or len(translated) > max_characters:
        raise _error("translated action exceeds its bound")
    assistant = Message("assistant", (TextPart(completion.text),) if completion.text else (), (call,),
                        native_assistant=completion.native_assistant)
    ack = Message("tool", (TextPart(
        "Synthetic adapter acknowledgement: Adapter selected action; not executed yet; next observation gives outcome."),), tool_call_id=call.call_id)
    event = {"kind": "action_translation", "call_id": call.call_id,
             "format": kind, "translated_action": translated, "source": "simple_agent_adapter"}
    return ModelCompletion(translated, completion.usage, context_messages=(assistant, ack)), event
