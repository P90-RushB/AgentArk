# SPDX-License-Identifier: Apache-2.0
# Adapted with permission from the ArkArena stdlib core (ark_arena/harness).
"""A bounded private conversation with transactional, source-labelled compaction."""
from __future__ import annotations

import hashlib
import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from .encoding import encoded_messages, tool_declarations
from .compaction import SourceReference, SummaryAtom, image_atom, metadata as compaction_metadata, pack_atoms, text_atom
from .models import SubmissionResponseError, SummaryResponseError
from .tools import FileWorkspace, ToolRegistry
from .types import (
    ArchiveEntry, AsyncModel, AgentError, AgentEvent, AgentInputIdentity,
    AgentLimitError, AgentRequest, AgentResponse, AgentReuseError, AgentSupersededError,
    AgentUsage, ImagePart, Message, ModelCompletion, ModelRequest, ModelUsage, TextPart, ToolCall, ToolResult,
)


class _DecisionDeadline:
    """Cancel this task at a deadline, including on Python 3.10.

    A provider can catch cancellation; ``check`` still prevents any later tool
    execution or successful completion. External cancellation stays cancellation.
    No detached provider task is created, and no caller-owned client is closed.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.expired = False
        self._marker = object()
        self._task = None
        self._handle = None
        self._initial_cancels = 0

    async def __aenter__(self):
        self._task = asyncio.current_task()
        self._loop = asyncio.get_running_loop()
        self._end_at = self._loop.time() + self.seconds
        cancelling = getattr(self._task, "cancelling", None)
        self._initial_cancels = cancelling() if cancelling is not None else 0
        self._handle = self._loop.call_later(self.seconds, self._expire)
        return self

    def _expire(self) -> None:
        self.expired = True
        self._task.cancel(self._marker)

    def check(self) -> None:
        if self.expired or self._loop.time() >= self._end_at:
            raise AgentLimitError("decision time budget exhausted")

    async def __aexit__(self, exception_type, exception, traceback):
        self._handle.cancel()
        if not self.expired:
            return False
        uncancel = getattr(self._task, "uncancel", None)
        remaining = uncancel() if uncancel is not None else self._initial_cancels
        if exception_type is asyncio.CancelledError:
            # On 3.11+ also distinguish an external cancel queued after ours.
            if remaining > self._initial_cancels or exception.args != (self._marker,):
                return False
            raise AgentLimitError("decision time budget exhausted") from None
        if exception_type is None:
            raise AgentLimitError("decision time budget exhausted")
        return False


@dataclass(frozen=True, slots=True)
class SimpleRuntimeConfig:
    max_context_characters: int = 524_288
    # Encoded messages include base64 images; leave room for API request envelopes.
    max_context_bytes: int = 16_000_000
    max_context_tokens: int = 256 * 1024
    output_reserve_tokens: int = 20_480
    recent_turns: int = 2
    max_summary_calls: int = 4
    max_images: int = 256
    max_image_bytes: int = 16 * 1024 * 1024
    image_token_estimate: int = 1_024
    max_archive_entries: int = 4_096
    max_archive_characters: int = 32 * 1024 * 1024
    max_archive_bytes: int = 512 * 1024 * 1024
    max_archive_image_bytes: int = 512 * 1024 * 1024
    summary_input_characters: int = 196_608
    summary_input_bytes: int = 16_000_000
    summary_input_tokens: int = 48_000
    summary_output_characters: int = 32_000
    summary_output_bytes: int = 128_000
    summary_output_tokens: int = 32 * 1024
    max_action_output_characters: int = 120_000
    max_action_output_bytes: int = 512_000
    max_model_calls: int = 4_096
    timeout_seconds: float = 1800.0
    compaction_policy: str = "message_chunks_v1"

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "compaction_policy":
                if value not in ("whole_turn_v1", "message_chunks_v1"):
                    raise ValueError("unsupported compaction_policy")
                continue
            if name == "timeout_seconds":
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("timeout_seconds has an invalid bound")
                continue
            minimum = 0 if name in ("recent_turns", "max_summary_calls", "max_images", "max_image_bytes") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} has an invalid bound")
        if self.output_reserve_tokens >= self.max_context_tokens:
            raise ValueError("output reserve must leave room for input")


@dataclass(frozen=True, slots=True)
class _WorkingEntry:
    sequence: int
    kind: str
    identity: AgentInputIdentity
    messages: tuple[Message, ...]
    source_sequences: tuple[int, ...]
    # Generated notes refer to their actual summary_completion archive text.
    origin_sequence: int | None = None


def _scope(identity: AgentInputIdentity) -> tuple[str, int]:
    return identity.episode_id, identity.incarnation


def _metrics(messages: tuple[Message, ...]) -> tuple[int, int, int, int]:
    characters = images = image_bytes = 0
    for message in messages:
        characters += sum(len(c.arguments_json) + len(c.name) + len(c.call_id) for c in message.tool_calls)
        if message.native_assistant is not None:
            characters += len(message.native_assistant.message_json)
        for part in message.parts:
            if isinstance(part, TextPart):
                characters += len(part.text)
            else:
                images += 1
                image_bytes += len(part.data)
    # Conservatively include API-only replay in addition to the portable text
    # projection. Opaque reasoning must not bypass working/archive byte limits.
    native_bytes = sum(len(message.native_assistant.message_json.encode("utf-8"))
                       + len(message.native_assistant.scope.encode("utf-8"))
                       for message in messages if message.native_assistant is not None)
    return characters, len(encoded_messages(messages)) + native_bytes, images, image_bytes


class SimpleRuntime:
    """One runtime per stable agent. No SDK imports or action interpretation.

    ``history_snapshot`` contains immutable originals. ``working_snapshot`` is
    the model context, with model notes clearly separated from original facts.
    Token estimates use text UTF-8 bytes and a configured image estimate,
    separately from encoded message byte bounds. Supply a model counter for
    exact/model-specific accounting.
    """

    def __init__(
        self, model: AsyncModel, config: SimpleRuntimeConfig | None = None, *,
        summarizer: AsyncModel | None = None,
        token_counter: Callable[[tuple[Message, ...]], int] | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        if not callable(getattr(model, "complete", None)):
            raise TypeError("model must expose async complete")
        if summarizer is not None and not callable(getattr(summarizer, "complete", None)):
            raise TypeError("summarizer must expose async complete")
        self._model = model
        self._summarizer = summarizer or model
        self.config = config or SimpleRuntimeConfig()
        if not isinstance(self.config, SimpleRuntimeConfig):
            raise TypeError("config must be SimpleRuntimeConfig")
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token counter must be callable")
        self._token_counter = token_counter
        if tools is not None and not isinstance(tools, ToolRegistry):
            raise TypeError("tools must be a ToolRegistry")
        self.tools = tools
        if tools is not None:
            tools.validate_fresh()
            tools.claim(self)
        self._initialize()

    def _initialize(self) -> None:
        self._binding: tuple[str, str, str] | None = None
        self._current: AgentInputIdentity | None = None
        self._pinned: tuple[Message, ...] | None = None
        self._archive: list[ArchiveEntry] = []
        self._working: list[_WorkingEntry] = []
        self._inputs: dict[AgentInputIdentity, str] = {}
        self._events: dict[str, str] = {}
        self._completed: set[AgentInputIdentity] = set()
        self._action_attempted: set[AgentInputIdentity] = set()
        self._summary_attempts: dict[AgentInputIdentity, int] = {}
        self._required_events: set[int] = set()
        self._archive_characters = self._archive_bytes = self._archive_image_bytes = 0
        self._action_calls = self._summary_calls = 0
        self._action_usage: list[ModelUsage] = []
        self._summary_usage: list[ModelUsage] = []
        self._busy = self._closed = False
        self._deadline: _DecisionDeadline | None = None

    @property
    def history_snapshot(self) -> tuple[ArchiveEntry, ...]:
        return tuple(self._archive)

    @property
    def working_snapshot(self) -> tuple[Message, ...]:
        return (self._pinned or ()) + tuple(message for entry in self._working for message in entry.messages)

    @property
    def usage(self) -> AgentUsage:
        return AgentUsage(self._action_calls, self._summary_calls, tuple(self._action_usage), tuple(self._summary_usage))

    def close(self) -> None:
        if self._busy:
            raise AgentReuseError("cannot close an active runtime")
        self._closed = True

    def validate_fresh(self) -> None:
        """Fail preflight if this instance already belongs to a lifetime."""
        if self._binding is not None or self._archive or self._busy or self._closed:
            raise AgentReuseError("runtime must start in a fresh, open lifetime")
        if self.tools is not None:
            self.tools.validate_fresh()

    def reset(self) -> None:
        """Explicitly clear a closed lifetime; the model/client remains owned by its caller."""
        if self._busy or (self._binding is not None and not self._closed):
            raise AgentReuseError("reset requires a closed runtime lifetime")
        if self.tools is not None and self._binding is not None:
            raise AgentReuseError("create fresh tools for a new runtime lifetime")
        self._initialize()

    def _check_binding(self, identity: AgentInputIdentity) -> None:
        binding = (identity.run_id, identity.session_id, identity.agent_id)
        if self._binding is not None and binding != self._binding:
            raise AgentReuseError("runtime is bound to another run, Session, or stable agent")

    def _archive_cost(self, entry: ArchiveEntry) -> tuple[int, int, int]:
        characters, message_bytes, _, image_bytes = _metrics(entry.messages)
        characters += len(entry.text)
        metadata = json.dumps({
            "sequence": entry.sequence, "kind": entry.kind,
            "identity": [entry.identity.run_id, entry.identity.session_id, entry.identity.episode_id,
                         entry.identity.agent_id, entry.identity.incarnation, entry.identity.input_id],
            "text": entry.text, "event_id": entry.event_id,
            "usage": [entry.usage.input_tokens, entry.usage.output_tokens, entry.usage.total_tokens],
        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        byte_count = message_bytes + len(metadata)
        if entry.compaction_json is not None:
            characters += len(entry.compaction_json)
            byte_count += len(entry.compaction_json.encode("utf-8"))
        return characters, byte_count, image_bytes

    def _check_archive_capacity(self, entry: ArchiveEntry) -> tuple[int, int, int]:
        characters, byte_count, image_bytes = self._archive_cost(entry)
        if (len(self._archive) + 1 > self.config.max_archive_entries
            or self._archive_characters + characters > self.config.max_archive_characters
            or self._archive_bytes + byte_count > self.config.max_archive_bytes
            or self._archive_image_bytes + image_bytes > self.config.max_archive_image_bytes):
            raise AgentLimitError("immutable archive bound exhausted")
        return characters, byte_count, image_bytes

    def _append_archive(self, entry: ArchiveEntry) -> None:
        characters, byte_count, image_bytes = self._check_archive_capacity(entry)
        self._archive.append(entry)
        self._archive_characters += characters
        self._archive_bytes += byte_count
        self._archive_image_bytes += image_bytes

    def _tool_audit_entry(self, identity: AgentInputIdentity, call: ToolCall,
                          result: ToolResult | None = None, error: Exception | None = None) -> ArchiveEntry:
        sequence = len(self._archive) + 1
        metadata = {"kind": "tool_result", "call_id": call.call_id, "name": call.name,
                    "success": result.success if result is not None else False,
                    "result_returned": result is not None}
        if error is not None:
            # Bounded local diagnostics; an exception does not prove that a
            # trusted operation with side effects was rolled back.
            metadata["error_type"] = type(error).__name__[:64]
        parts = result.parts if result is not None else (TextPart(json.dumps({
            "error_type": metadata.get("error_type"), "result_returned": False}, separators=(",", ":"))),)
        pair = (Message("assistant", (), (call,)), Message("tool", parts, tool_call_id=call.call_id))
        return ArchiveEntry(sequence, "event", identity, pair,
                            text=json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                            event_id=f"tool-result-{sequence}")

    def _reserve_tool_audit(self, identity: AgentInputIdentity, call: ToolCall, session) -> None:
        """Fail before executing if its bounded result cannot be archived.

        FileWorkspace's four built-in handlers return at most one text and
        one image part. Trusted custom handlers use the registry-wide limits;
        accounting reserves per-part overhead as well as raw payload bytes.
        This reservation is conservative and never changes working context.
        """
        sample = self._tool_audit_entry(identity, call, ToolResult(call.call_id, (TextPart(""),)))
        characters, byte_count, image_bytes = self._archive_cost(sample)
        limits = self.tools.limits
        text_bound = max(1024, min(limits.max_text_bytes, max(0, limits.max_total_text_bytes - session.text_bytes)))
        image_count = max(0, limits.max_images - session.images)
        if type(self.tools) is FileWorkspace and call.name in ("list_files", "read_file", "write_note", "view_image"):
            image_count = min(1, image_count) if call.name == "view_image" else 0
            part_count = 1 + image_count
        else:
            part_count = text_bound + 1 + image_count
        image_bound = min(max(0, limits.max_total_image_bytes - session.image_bytes),
                          limits.max_image_bytes * image_count)
        characters += text_bound + 128
        byte_count += text_bound * 6 + ((image_bound + 2) // 3) * 4 + part_count * 128 + 512
        image_bytes += image_bound
        cfg = self.config
        if (len(self._archive) + 1 > cfg.max_archive_entries
            or self._archive_characters + characters > cfg.max_archive_characters
            or self._archive_bytes + byte_count > cfg.max_archive_bytes
            or self._archive_image_bytes + image_bytes > cfg.max_archive_image_bytes):
            raise AgentLimitError("tool audit archive capacity exhausted before operation")

    def accept_input(self, request: AgentRequest) -> None:
        if not isinstance(request, AgentRequest):
            raise TypeError("input must be AgentRequest")
        identity = request.identity
        self._check_binding(identity)
        digest = hashlib.sha256(encoded_messages(request.messages) + request.history_policy.encode("ascii")).hexdigest()
        prior = self._inputs.get(identity)
        if prior is not None:
            if digest != prior:
                raise AgentReuseError("input identity has conflicting bytes")
            return
        if self._closed:
            raise AgentReuseError("runtime lifetime is closed")
        pinned = tuple(message for message in request.messages if message.role in ("system", "developer"))
        dynamic = tuple(message for message in request.messages if message.role not in ("system", "developer"))
        if self._pinned is not None and pinned != self._pinned:
            raise AgentReuseError("live runtime instructions or schema changed")
        if self._current is not None and identity.episode_id == self._current.episode_id and identity.incarnation < self._current.incarnation:
            raise AgentReuseError("input incarnation regressed")
        sequence = len(self._archive) + 1
        archive_entry = ArchiveEntry(sequence, "input", identity, request.messages)
        # A rejected input must not clear/supersede the tool workspace either.
        self._check_archive_capacity(archive_entry)
        if self.tools is not None:
            self.tools.accept_input(request)
        self._append_archive(archive_entry)
        self._binding = (identity.run_id, identity.session_id, identity.agent_id)
        if self._current is None:
            self._working = [entry for entry in self._working if entry.identity.episode_id == identity.episode_id
                             and entry.identity.incarnation in (0, identity.incarnation)]
        elif (identity.episode_id != self._current.episode_id and request.history_policy == "omit") or (
            identity.episode_id == self._current.episode_id and identity.incarnation > self._current.incarnation
        ):
            self._working.clear()
        self._required_events.intersection_update(entry.sequence for entry in self._working)
        self._pinned = pinned
        self._current = identity
        self._inputs[identity] = digest
        self._working.append(_WorkingEntry(sequence, "input", identity, dynamic, (sequence,)))

    accept = accept_input

    def observe_result(self, event: AgentEvent) -> None:
        if not isinstance(event, AgentEvent):
            raise TypeError("result must be AgentEvent")
        self._check_binding(event.identity)
        digest = hashlib.sha256(repr(event.identity).encode("utf-8") + event.kind.encode("utf-8") + event.payload_json.encode("utf-8")).hexdigest()
        prior = self._events.get(event.event_id)
        if prior is not None:
            if prior != digest:
                raise AgentReuseError("event identity has conflicting bytes")
            return
        message = Message.text("user", f"Environment result ({event.kind}):\n{event.payload_json}")
        sequence = len(self._archive) + 1
        self._append_archive(ArchiveEntry(sequence, "event", event.identity, (message,), event_id=event.event_id))
        self._binding = (event.identity.run_id, event.identity.session_id, event.identity.agent_id)
        self._events[event.event_id] = digest
        if not self._closed and (self._current is None or _scope(event.identity) == _scope(self._current)):
            self._working.append(_WorkingEntry(sequence, "event", event.identity, (message,), (sequence,)))
            self._required_events.add(sequence)

    observe = observe_result

    def _tokens(self, messages: tuple[Message, ...], byte_count: int) -> int:
        native_bytes = sum(len(message.native_assistant.message_json.encode("utf-8"))
                           for message in messages if message.native_assistant is not None)
        if self._token_counter is None:
            text_bytes = sum(len(part.text.encode("utf-8")) for message in messages
                             for part in message.parts if isinstance(part, TextPart))
            text_bytes += sum(len(c.arguments_json.encode("utf-8")) + len(c.name) + len(c.call_id)
                              for message in messages for c in message.tool_calls)
            text_bytes += native_bytes
            images = sum(isinstance(part, ImagePart) for message in messages for part in message.parts)
            return math.ceil(text_bytes / 4) + len(messages) * 8 + images * self.config.image_token_estimate
        value = self._token_counter(messages)
        if type(value) is not int or value < 0:
            raise AgentError("model token counter returned an invalid count")
        # A portable counter need not know this API-only state. Include a
        # conservative allowance even with a caller-supplied text counter.
        return value + math.ceil(native_bytes / 4)

    def _fits(self, messages: tuple[Message, ...], *, summary: bool = False,
              native_headroom: bool = False) -> bool:
        if self.tools is not None and not summary:
            messages += (self._tool_budget_message(),)
        characters, byte_count, image_count, image_bytes = _metrics(messages)
        cfg = self.config
        if summary:
            return (characters <= cfg.summary_input_characters and byte_count <= cfg.summary_input_bytes
                    and image_count <= cfg.max_images and image_bytes <= cfg.max_image_bytes
                    and self._tokens(messages, byte_count) <= cfg.summary_input_tokens)
        character_limit = max(1, cfg.max_context_characters * 3 // 4) if native_headroom else cfg.max_context_characters
        byte_limit = max(1, cfg.max_context_bytes * 3 // 4) if native_headroom else cfg.max_context_bytes
        return (characters <= character_limit and byte_count <= byte_limit
                and image_count <= cfg.max_images and image_bytes <= cfg.max_image_bytes
                and self._tokens(messages, byte_count) + cfg.output_reserve_tokens <= cfg.max_context_tokens)

    def _fresh_decision_fits(self, messages: tuple[Message, ...]) -> bool:
        """Reserve 25% of working char/byte caps before a new native tool turn.

        This is a best-effort watermark, not a guarantee for all tool rounds or
        unbounded batches. Native continuations still use the unchanged hard
        limits and must never compact their signature-bound prefix. Tools-free
        runtimes retain their existing hard-fit policy.

        Offline legacy-prefix rehydration may override this method to delegate
        to ``_fits``; restore the default before the next fresh decision.
        """
        return self._fits(messages, native_headroom=self.tools is not None)

    def _model_request(self, messages: tuple[Message, ...], purpose: str) -> ModelRequest:
        budget_messages = messages + ((self._tool_budget_message(),) if self.tools is not None and purpose == "action" else ())
        token_count = self._tokens(budget_messages, len(encoded_messages(budget_messages)))
        return ModelRequest(messages, purpose, self.config.summary_output_tokens if purpose == "summary" else self.config.output_reserve_tokens,
                            token_count, self._token_counter is None or any(m.native_assistant is not None for m in messages),
                            self.tools.specs if self.tools is not None and purpose == "action" else ())

    def _tool_budget_message(self) -> Message:
        return Message.text("developer", json.dumps(tool_declarations(self.tools.specs), ensure_ascii=False))

    async def _call_model(self, model: AsyncModel, request: ModelRequest) -> ModelCompletion:
        if self._deadline is not None:
            self._deadline.check()
        if self._action_calls + self._summary_calls >= self.config.max_model_calls:
            raise AgentLimitError("lifetime model call budget exhausted")
        if request.purpose == "summary":
            self._summary_calls += 1
        else:
            self._action_calls += 1
        result = await model.complete(request)
        if self._deadline is not None:
            self._deadline.check()
        return result

    def _summary_span(self, working: list[_WorkingEntry], required_events: frozenset[int],
                      retry_instruction: str | None = None) -> tuple[_WorkingEntry, ...]:
        inputs = [entry for entry in working if entry.kind == "input"]
        protected = {entry.sequence for entry in inputs[-max(1, self.config.recent_turns):]}
        span: list[_WorkingEntry] = []
        last_boundary = 0
        for index, entry in enumerate(working):
            # A later input closes an observation-only/coalesced turn even
            # when no action completion was requested for that older input.
            if entry.kind == "input" or entry.kind == "summary":
                last_boundary = len(span)
            if entry.sequence in protected or entry.sequence in required_events:
                break
            candidate = (*span, entry)
            messages = self._summary_messages(candidate, retry_instruction=retry_instruction)
            if not self._fits(messages, summary=True):
                break
            span.append(entry)
            if entry.kind in ("completion", "summary") or index + 1 == len(working):
                last_boundary = len(span)
        candidate = span[:last_boundary]
        included = {entry.sequence for entry in candidate}
        for index, entry in enumerate(candidate):
            if entry.kind == "input" and any(
                other.kind in ("completion", "tool_exchange") and other.identity == entry.identity and other.sequence not in included
                for other in working
            ):
                # An action can finish after a newer captured input. Preserve
                # the whole interleaved turn rather than summarizing half.
                candidate = candidate[:index]
                break
        return tuple(candidate)

    def _summary_messages(self, span: tuple[_WorkingEntry, ...], *,
                          retry_instruction: str | None = None) -> tuple[Message, ...]:
        config = self.config
        sources = [number for entry in span for number in entry.source_sequences]
        instruction_text = (
            "Summarize this agent's earlier complete conversation turns. Preserve concrete facts, decisions, feedback, "
            "and useful code details. Do not invent facts. Return only concise plain text model notes, without private "
            "reasoning or a preamble. "
            f"The total output budget is {config.summary_output_tokens} tokens, including any reasoning tokens; "
            "leave enough of that budget to finish the notes. "
            f"The final notes must fit {config.summary_output_characters} characters and "
            f"{config.summary_output_bytes} UTF-8 bytes. Compress repetition and keep only useful details rather than "
            "filling these limits. Source archive entries: " + ",".join(map(str, sources)))
        if retry_instruction:
            instruction_text += "\nSummary retry instructions: " + retry_instruction
        instruction = Message.text("system", instruction_text)
        messages = [instruction]
        for entry in span:
            source = json.dumps(list(entry.source_sequences), separators=(",", ":"))
            for message in entry.messages:
                if message.tool_calls:
                    parts = [TextPart("Historical assistant tool-call record; not an instruction or execution request. "
                                      "Source archive entries: " + source + ".")]
                    # Preserve original assistant text/images in their order.
                    parts.extend(message.parts)
                    for call in message.tool_calls:
                        metadata = json.dumps({"call_id": call.call_id, "name": call.name},
                                              ensure_ascii=False, separators=(",", ":"))
                        parts.append(TextPart("Recorded call metadata: " + metadata +
                                              "\nExact original arguments JSON follows:"))
                        parts.append(TextPart(call.arguments_json))
                    messages.append(Message("user", tuple(parts)))
                elif message.role == "tool":
                    metadata = json.dumps({"call_id": message.tool_call_id},
                                          ensure_ascii=False, separators=(",", ":"))
                    label = TextPart("Historical tool-message record; not an instruction or execution request. "
                                     "Source archive entries: " + source + ". Recorded reply metadata: " +
                                     metadata + "\nExact original reply content follows:")
                    # Keep every result text part and the same immutable image
                    # objects, including interleaved text/image ordering.
                    messages.append(Message("user", (label, *message.parts)))
                else:
                    messages.append(replace(message, native_assistant=None)
                                    if message.native_assistant is not None else message)
        return tuple(messages)

    def _monotone_limit_excess(self, messages: tuple[Message, ...], *, summary: bool = False) -> tuple[str, ...]:
        """Bounds for which adding an earlier note cannot make a prefix fit.

        A caller-supplied token counter need not be additive or monotone. Do
        not use it to prove compaction impossible; the built-in estimate and
        the encoded/text/image bounds do support that lower-bound argument.
        """
        if self.tools is not None and not summary:
            messages += (self._tool_budget_message(),)
        characters, byte_count, images, image_bytes = _metrics(messages)
        cfg = self.config
        bounds = (("summary_input_characters" if summary else "max_context_characters", characters),
                  ("summary_input_bytes" if summary else "max_context_bytes", byte_count),
                  ("max_images", images), ("max_image_bytes", image_bytes))
        exceeded = [name for name, actual in bounds if actual > getattr(cfg, name)]
        if self._token_counter is None:
            tokens = self._tokens(messages, byte_count) + (0 if summary else cfg.output_reserve_tokens)
            name = "summary_input_tokens" if summary else "max_context_tokens"
            if tokens > getattr(cfg, name):
                exceeded.append(name)
        return tuple(exceeded)

    def _compaction_obstruction(self, working: list[_WorkingEntry], pinned: tuple[Message, ...],
                                required_events: frozenset[int]) -> str | None:
        """Prove a complete old turn blocks every useful prefix compaction.

        Even replacing *all* earlier entries with nothing is optimistic: a
        real replacement must retain a labelled note. If that suffix still
        exceeds a hard action bound and its first complete turn cannot enter
        the summarizer by itself, rewriting older notes cannot unblock it.
        """
        inputs = [entry for entry in working if entry.kind == "input"]
        protected = {entry.sequence for entry in inputs[-max(1, self.config.recent_turns):]}
        boundary = next((index for index, entry in enumerate(working)
                         if entry.sequence in protected or entry.sequence in required_events), len(working))
        for index, entry in enumerate(working[:boundary]):
            if entry.kind != "input":
                continue
            end = next((position for position in range(index + 1, boundary)
                        if working[position].kind in ("input", "summary")), boundary)
            # Preserve the same complete-turn contract as _summary_span.
            # Interleaved late completions need the larger original group;
            # this conservative check simply leaves them to the usual path.
            if any(other.kind in ("completion", "tool_exchange") and other.identity == entry.identity
                   for other in (*working[:index], *working[end:])):
                continue
            related = [position for position in range(index + 1, end)
                       if working[position].kind in ("completion", "tool_exchange")
                       and working[position].identity == entry.identity]
            minimum_end = related[-1] if related else index + 1
            complete_at = next((position for position in range(minimum_end, end)
                                if working[position].kind == "completion"), None)
            if complete_at is not None:
                # A completion is already a legal _summary_span boundary.
                # Later consumed observer events may remain in the suffix;
                # they are not an inseparable part of that earlier turn.
                end = complete_at + 1
            turn = tuple(working[index:end])
            summary_bounds = self._monotone_limit_excess(self._summary_messages(turn), summary=True)
            if not summary_bounds:
                continue
            # Successful compaction clears signature-bound replay from the
            # working projection. Exclude it here too, so removable opaque
            # bytes cannot cause a false infeasibility diagnosis.
            suffix = pinned + tuple(replace(message, native_assistant=None)
                                    for item in working[index:] for message in item.messages)
            action_bounds = self._monotone_limit_excess(suffix)
            if action_bounds:
                return ("compaction cannot fit context bounds: complete earlier turn exceeds summary input bounds ("
                        + ", ".join(summary_bounds) + "); unchanged suffix exceeds hard action bounds ("
                        + ", ".join(action_bounds) + ")")
        return None

    async def _compact_whole_turn(self, identity: AgentInputIdentity, *, original=None, pinned=None, required_events=None,
                       fresh_decision: bool = False):
        original = tuple(self._working) if original is None else original
        staged = list(original)
        pinned = (self._pinned or ()) if pinned is None else pinned
        required_events = frozenset(self._required_events) if required_events is None else required_events

        def snapshot() -> tuple[Message, ...]:
            return pinned + tuple(message for entry in staged for message in entry.messages)

        def commit():
            # Model work keeps its original input snapshot. Newly accepted
            # inputs/results remain in the live context without being rebound
            # to this older input. Episode/churn resets may remove the source
            # span entirely, in which case only the private snapshot changes.
            if tuple(self._working[:len(original)]) == original:
                self._working = [*staged, *self._working[len(original):]]
            return snapshot(), tuple(staged)

        target_fits = self._fresh_decision_fits if fresh_decision else self._fits
        remaining = self.config.max_summary_calls - self._summary_attempts.get(identity, 0)
        retry_instruction = None
        last_failure = None

        def retry(reason: str) -> None:
            nonlocal retry_instruction, last_failure
            failure = SummaryResponseError(reason)
            last_failure = str(failure)
            retry_instruction = (failure.retry_instruction +
                f" Keep the complete response within the existing {self.config.summary_output_tokens}-token output budget; "
                f"summary content must fit {self.config.summary_output_characters} characters and {self.config.summary_output_bytes} UTF-8 bytes.")

        for _ in range(remaining):
            if target_fits(snapshot()):
                return commit()
            obstruction = self._compaction_obstruction(staged, pinned, required_events)
            if obstruction:
                raise AgentLimitError(obstruction)
            span = self._summary_span(staged, required_events, retry_instruction)
            if not span:
                # Protected inputs/pins and required feedback can exceed the
                # soft watermark. A fresh decision may still use a hard-fitting
                # prefix; do not misreport soft pressure as context overflow.
                if fresh_decision and self._fits(snapshot()):
                    return commit()
                if retry_instruction:
                    raise AgentLimitError("summary retry instructions and a complete source turn cannot fit summary input bounds")
                raise AgentLimitError("current input, required feedback, or retained turns cannot fit context bounds")
            messages = self._summary_messages(span, retry_instruction=retry_instruction)
            self._summary_attempts[identity] = self._summary_attempts.get(identity, 0) + 1
            try:
                completion = await self._call_model(self._summarizer, self._model_request(messages, "summary"))
            except SummaryResponseError as failure:
                self._summary_usage.append(failure.usage)
                if self._deadline is not None:
                    self._deadline.check()
                retry(failure.reason)
                continue
            if (not isinstance(completion, ModelCompletion) or completion.tool_calls
                or completion.context_messages or completion.native_assistant is not None):
                raise AgentError("summary model returned an invalid completion")
            self._append_archive(ArchiveEntry(len(self._archive) + 1, "summary_completion", identity, text=completion.text, usage=completion.usage))
            self._summary_usage.append(completion.usage)
            output_message = (Message.text("assistant", completion.text),)
            output_tokens = self._tokens(output_message, len(encoded_messages(output_message)))
            if not completion.text.strip():
                retry("empty_content")
                continue
            if (len(completion.text) > self.config.summary_output_characters
                or len(completion.text.encode("utf-8")) > self.config.summary_output_bytes
                or output_tokens > self.config.summary_output_tokens):
                retry("output_bounds")
                continue
            sources = tuple(number for entry in span for number in entry.source_sequences)
            note = Message.text("user", f"Model notes; not environment facts. Source archive entries {','.join(map(str, sources))}:\n{completion.text}")
            replacement = _WorkingEntry(span[0].sequence, "summary", span[0].identity, (note,), sources)
            # Stage changes until every summary succeeds. Original working state
            # and append-only observer results survive cancellation or failure.
            candidate = [replacement, *staged[len(span):]]
            old_messages = tuple(message for entry in span for message in entry.messages)
            if _metrics((note,))[1] >= _metrics(old_messages)[1]:
                retry("not_reduced")
                continue
            # A changed prefix invalidates signature-bound replay, including
            # blocks attached to retained recent turns. Originals remain in
            # the immutable archive; only the new working projection changes.
            staged = [replace(entry, messages=tuple(replace(message, native_assistant=None)
                      for message in entry.messages)) for entry in candidate]
            retry_instruction = last_failure = None
        if not target_fits(snapshot()):
            # The watermark is best effort within existing summary budgets.
            # Actual hard-fit failures retain the original failure semantics.
            if fresh_decision and self._fits(snapshot()):
                return commit()
            if last_failure:
                raise AgentLimitError("summary call bound exhausted: " + last_failure) from None
            raise AgentLimitError("summary call bound exhausted")
        return commit()

    def _safe_summary_prefixes(self, working: list[_WorkingEntry], identity: AgentInputIdentity,
                               required_events: frozenset[int]) -> tuple[tuple[_WorkingEntry, ...], ...]:
        """Complete old prefixes, independent of the summarizer's input size.

        Protect identities rather than only input rows: A/input, B/current,
        A/late-completion must never make the first A row independently old.
        """
        inputs = [entry for entry in working if entry.kind == "input"]
        protected = {entry.identity for entry in inputs[-max(1, self.config.recent_turns):]}
        protected.add(identity)
        if self._current is not None:
            protected.add(self._current)
        boundary = next((index for index, entry in enumerate(working)
                         if entry.identity in protected or entry.sequence in required_events), len(working))
        result = []
        for end in range(boundary, 0, -1):
            if (working[end - 1].kind not in ("completion", "summary")
                and end < len(working) and working[end].kind not in ("input", "summary")):
                continue
            included_inputs = {entry.identity for entry in working[:end] if entry.kind == "input"}
            if any(entry.identity in included_inputs and entry.kind in ("completion", "tool_exchange")
                   for entry in working[end:]):
                continue
            result.append(tuple(working[:end]))
        return tuple(result)

    def _chunk_atoms(self, span: tuple[_WorkingEntry, ...]) -> tuple[SummaryAtom, ...]:
        """Flatten ordinary parts, keeping each closed tool batch indivisible."""
        records = []
        for entry in span:
            sequence = entry.origin_sequence or entry.sequence
            archived = self._archive[sequence - 1] if 0 < sequence <= len(self._archive) else None
            if entry.kind == "summary" and archived is not None and archived.kind == "summary_completion":
                reference = SourceReference(sequence, -1, 0, "model_notes", "", 0)
                records.append((Message.text("user", archived.text), (reference,), entry))
                continue
            original_indices = ([index for index, message in enumerate(archived.messages)
                                 if message.role not in ("system", "developer")]
                                if archived is not None and entry.kind == "input" else [])
            for index, message in enumerate(entry.messages):
                message_index = original_indices[index] if original_indices else index
                kind = "text"
                if archived is not None and not archived.messages and entry.kind == "completion":
                    message_index, kind = -1, "completion_text"
                references = []
                for part_index, part in enumerate(message.parts):
                    if isinstance(part, TextPart):
                        references.append(SourceReference(sequence, message_index, part_index, kind,
                            hashlib.sha256(part.text.encode("utf-8")).hexdigest(), 0, len(part.text)))
                    else:
                        references.append(SourceReference(sequence, message_index, part_index, "image",
                            hashlib.sha256(part.data).hexdigest(), image_whole=True))
                for call_index, call in enumerate(message.tool_calls):
                    value = json.dumps({"call_id": call.call_id, "name": call.name,
                                        "arguments_json": call.arguments_json},
                                       ensure_ascii=False, separators=(",", ":"))
                    references.append(SourceReference(sequence, message_index, -1, "tool_call",
                        hashlib.sha256(value.encode("utf-8")).hexdigest(), call_index=call_index))
                records.append((replace(message, native_assistant=None), tuple(references), entry))
        atoms = []
        position = 0
        while position < len(records):
            message, references, entry = records[position]
            if message.tool_calls:
                ids = {call.call_id for call in message.tool_calls}
                stop = position + 1 + len(ids)
                replies = records[position + 1:stop]
                if (len(replies) != len(ids) or any(item[0].role != "tool" for item in replies)
                    or {item[0].tool_call_id for item in replies} != ids):
                    raise AgentLimitError("historical tool group is not a closed assistant/result batch")
                group = records[position:stop]
                # Reuse the exact portable tool-record projection, with native
                # reasoning excluded and original opaque IDs/arguments intact.
                projected = tuple(projected for original, _, owner in group
                                  for projected in self._summary_messages((replace(owner, messages=(original,)),))[1:])
                atoms.append(SummaryAtom(projected, tuple(item[0] for item in group),
                                         tuple(reference for item in group for reference in item[1])))
                position = stop
                continue
            if message.role == "tool":
                raise AgentLimitError("historical tool result has no closed assistant/result batch")
            for part, reference in zip(message.parts, references):
                atoms.append(text_atom(part.text, message.role, reference) if isinstance(part, TextPart)
                             else image_atom(part, message.role, reference))
            position += 1
        # A short adjacent label commonly supplies the image's timestamp or
        # identity. Keep it with that image instead of leaving the association
        # across independent leaf summaries. This is content-agnostic and
        # bounded; large text and unrelated messages remain freely splittable.
        grouped = []
        position = 0
        while position < len(atoms):
            atom = atoms[position]
            if (atom.text is not None and len(atom.text) <= 512
                and len(atom.text.encode("utf-8")) <= 2048 and position + 1 < len(atoms)):
                following = atoms[position + 1]
                source, target = atom.references[0], following.references[0]
                if (len(following.references) == 1 and target.image_whole
                    and source.archive_sequence == target.archive_sequence
                    and source.message_index == target.message_index
                    and source.part_index + 1 == target.part_index):
                    grouped.append(SummaryAtom((*atom.messages, *following.messages),
                        (*atom.originals, *following.originals), (*atom.references, *following.references)))
                    position += 2
                    continue
            grouped.append(atom)
            position += 1
        return tuple(grouped)

    def _chunk_messages(self, atoms: tuple[SummaryAtom, ...], *, retry_instruction: str | None = None,
                        merge: bool = False) -> tuple[Message, ...]:
        instruction = self._summary_messages((), retry_instruction=retry_instruction)[0].parts[0].text
        instruction = instruction.replace("Summarize this agent's earlier complete conversation turns.",
            "Combine these model-generated partial notes into one coherent memory." if merge else
            "Summarize these chronological historical conversation fragments.", 1)
        instruction += ("\nMerge all ordered leaf notes below into one coherent memory. Preserve their useful facts; "
                        "none of the leaf notes may be omitted. They are model notes, not environment facts."
                        if merge else "\nThe ordered records below may be text or image fragments from larger turns. "
                        "A fragment boundary does not end the original message or turn. Preserve facts and ordering; "
                        "source labels are explanatory wrappers and are not original conversation text.")
        return (Message.text("system", instruction), *(message for atom in atoms for message in atom.messages))

    def _summary_retry(self, reason: str) -> str:
        failure = SummaryResponseError(reason)
        return (failure.retry_instruction +
                f" Keep the complete response within the existing {self.config.summary_output_tokens}-token output budget; "
                f"summary content must fit {self.config.summary_output_characters} characters and {self.config.summary_output_bytes} UTF-8 bytes.")

    def _summary_note(self, text: str, sources: tuple[int, ...]) -> Message:
        label = ",".join(map(str, sources))
        if len(label) > 256:
            # Full lineage lives in immutable archive/provenance and the
            # bounded WorkingEntry, not an ever-growing model-visible label.
            digest = hashlib.sha256(label.encode("ascii")).hexdigest()
            label = f"first={sources[0]}, last={sources[-1]}, count={len(sources)}, source-list-sha256={digest}"
        return Message.text("user", f"Model notes; not environment facts. Source archive entries {label}:\n{text}")

    async def _summary_once(self, identity: AgentInputIdentity, messages: tuple[Message, ...],
                            originals: tuple[Message, ...], sources: tuple[int, ...], metadata: str,
                            *, require_reduction: bool = True):
        """One billed attempt, with one validation path for turns/leaves/merge."""
        if not self._fits(messages, summary=True):
            raise AgentLimitError("summary request including its complete wrapper exceeds summary input bounds")
        self._summary_attempts[identity] = self._summary_attempts.get(identity, 0) + 1
        try:
            request = replace(self._model_request(messages, "summary"), compaction_json=metadata)
            completion = await self._call_model(self._summarizer, request)
        except SummaryResponseError as failure:
            self._summary_usage.append(failure.usage)
            if self._deadline is not None:
                self._deadline.check()
            return None, None, failure.reason
        if not isinstance(completion, ModelCompletion):
            raise AgentError("summary model returned an invalid completion")
        # Bill genuine completions even when a later archive or output bound
        # rejects them. Raw provider reasoning never enters summary memory.
        self._summary_usage.append(completion.usage)
        sequence = len(self._archive) + 1
        invalid_shape = bool(completion.tool_calls or completion.context_messages or completion.native_assistant is not None)
        archive_metadata = metadata
        if invalid_shape:
            archive_metadata = json.dumps(dict(json.loads(metadata), status="rejected", reason="invalid_completion_shape"),
                                          ensure_ascii=False, separators=(",", ":"))
        self._append_archive(ArchiveEntry(sequence, "summary_completion", identity,
                             text=completion.text, usage=completion.usage, compaction_json=archive_metadata))
        if invalid_shape:
            raise AgentError("summary model returned an invalid completion")
        output = (Message.text("assistant", completion.text),)
        if not completion.text.strip():
            return None, sequence, "empty_content"
        if (len(completion.text) > self.config.summary_output_characters
            or len(completion.text.encode("utf-8")) > self.config.summary_output_bytes
            or self._tokens(output, len(encoded_messages(output))) > self.config.summary_output_tokens):
            return None, sequence, "output_bounds"
        note = self._summary_note(completion.text, sources)
        if require_reduction and _metrics((note,))[1] >= _metrics(originals)[1]:
            return None, sequence, "not_reduced"
        return note, sequence, None

    async def _summarize_chunk_plan(self, identity: AgentInputIdentity, span: tuple[_WorkingEntry, ...],
                                    leaves: tuple[tuple[SummaryAtom, ...], ...], plan_id: str):
        sources = tuple(number for entry in span for number in entry.source_sequences)
        completed = []
        count = len(leaves)
        for index, leaf in enumerate(leaves):
            retry_instruction = None
            while True:
                # Reserve one call for every pending leaf and the final merge.
                reserved = count - index - 1 + (1 if count > 1 else 0)
                if self.config.max_summary_calls - self._summary_attempts.get(identity, 0) <= reserved:
                    raise AgentLimitError("summary call bound exhausted before all planned fragments were summarized")
                messages = self._chunk_messages(leaf, retry_instruction=retry_instruction)
                audit = compaction_metadata(self.config.compaction_policy, "leaf", plan_id,
                    tuple(reference for atom in leaf for reference in atom.references), leaf_index=index, leaf_count=count)
                originals = tuple(message for atom in leaf for message in atom.originals)
                leaf_sources = tuple(dict.fromkeys(reference.archive_sequence for atom in leaf for reference in atom.references))
                # Intermediate notes may need explanatory context for a tiny
                # tail fragment. The final labelled result must still reduce
                # the complete original prefix before commit.
                note, sequence, reason = await self._summary_once(identity, messages, originals, leaf_sources, audit,
                                                               require_reduction=count == 1)
                if reason is None:
                    completed.append((note, sequence))
                    break
                retry_instruction = self._summary_retry(reason)
        if count == 1:
            note, sequence = completed[0]
            # The final working note references the complete original prefix.
            body = self._archive[sequence - 1].text
            return self._summary_note(body, sources), sequence
        merge_atoms = tuple(text_atom(self._archive[sequence - 1].text, "user",
                            SourceReference(sequence, -1, 0, "leaf_summary", "", 0))
                            for _, sequence in completed)
        retry_instruction = None
        while self._summary_attempts.get(identity, 0) < self.config.max_summary_calls:
            messages = self._chunk_messages(merge_atoms, retry_instruction=retry_instruction, merge=True)
            if not self._fits(messages, summary=True):
                raise AgentLimitError("all leaf summaries and the complete merge wrapper cannot fit summary input bounds")
            audit = compaction_metadata(self.config.compaction_policy, "merge", plan_id,
                    tuple(reference for atom in merge_atoms for reference in atom.references), leaf_count=count)
            # The final note carries the original prefix's lineage label.
            # Compare like-for-like with that complete source prefix, rather
            # than charging its label against only the unlabelled leaf bodies.
            originals = tuple(message for entry in span for message in entry.messages)
            note, sequence, reason = await self._summary_once(identity, messages, originals, sources, audit)
            if reason is None:
                return note, sequence
            retry_instruction = self._summary_retry(reason)
        raise AgentLimitError("summary call bound exhausted before the complete merge succeeded")

    async def _compact(self, identity: AgentInputIdentity, *, original=None, pinned=None, required_events=None,
                       fresh_decision: bool = False):
        if self.config.compaction_policy == "whole_turn_v1":
            return await self._compact_whole_turn(identity, original=original, pinned=pinned,
                required_events=required_events, fresh_decision=fresh_decision)
        original = tuple(self._working) if original is None else original
        staged = list(original)
        pinned = (self._pinned or ()) if pinned is None else pinned
        required_events = frozenset(self._required_events) if required_events is None else required_events
        target_fits = self._fresh_decision_fits if fresh_decision else self._fits
        changed = False
        retry_instruction = last_failure = None

        def snapshot():
            return pinned + tuple(message for entry in staged for message in entry.messages)

        def commit():
            if tuple(self._working[:len(original)]) == original:
                candidate = [*staged, *self._working[len(original):]]
                if changed:
                    candidate = [replace(entry, messages=tuple(replace(message, native_assistant=None)
                                 for message in entry.messages)) for entry in candidate]
                self._working = candidate
            return snapshot(), tuple(staged)

        while not target_fits(snapshot()):
            remaining = self.config.max_summary_calls - self._summary_attempts.get(identity, 0)
            if remaining <= 0:
                if fresh_decision and self._fits(snapshot()):
                    return commit()
                raise AgentLimitError("summary call bound exhausted" + (": " + last_failure if last_failure else ""))
            prefixes = self._safe_summary_prefixes(staged, identity, required_events)
            if not prefixes:
                if fresh_decision and self._fits(snapshot()):
                    return commit()
                raise AgentLimitError("current input, required feedback, or retained turns cannot fit context bounds")
            plan_id = hashlib.sha256((repr(identity) + repr(tuple(entry.sequence for entry in prefixes[0])) +
                       str(self._summary_attempts.get(identity, 0))).encode("utf-8")).hexdigest()[:24]
            normal_span = self._summary_span(staged, required_events, retry_instruction)
            safe_lengths = {len(prefix) for prefix in prefixes}
            normal = None
            # Keep the established one-call path when useful. The legacy
            # obstruction is only a signal to plan chunks immediately, never
            # a proof that the new policy cannot summarize an oversized turn.
            needs_chunks = self._compaction_obstruction(staged, pinned, required_events) is not None
            if (not needs_chunks or retry_instruction) and len(normal_span) in safe_lengths:
                for candidate in prefixes:
                    if len(candidate) > len(normal_span):
                        continue
                    atoms = self._chunk_atoms(candidate)
                    try:
                        audit = compaction_metadata(self.config.compaction_policy, "whole_turn", plan_id,
                                tuple(reference for atom in atoms for reference in atom.references))
                    except AgentLimitError:
                        # Bound provenance by selecting a smaller complete
                        # prefix; never truncate or silently omit references.
                        continue
                    messages = self._summary_messages(candidate, retry_instruction=retry_instruction)
                    if self._fits(messages, summary=True):
                        normal = candidate, messages, audit
                        break
            if normal is not None:
                span, messages, audit = normal
                sources = tuple(number for entry in span for number in entry.source_sequences)
                originals = tuple(message for entry in span for message in entry.messages)
                note, sequence, reason = await self._summary_once(identity, messages, originals, sources, audit)
                if reason is not None:
                    last_failure = str(SummaryResponseError(reason))
                    retry_instruction = self._summary_retry(reason)
                    continue
            else:
                failure = None
                for span in prefixes:
                    try:
                        atoms = self._chunk_atoms(span)
                        # A single leaf needs no merge; multiple leaves reserve
                        # one call and have at most three source leaves.
                        one = self._chunk_messages(atoms)
                        if self._fits(one, summary=True):
                            leaves = (atoms,)
                        else:
                            leaves = pack_atoms(atoms,
                                lambda chunk: self._fits(self._chunk_messages(chunk), summary=True),
                                lambda chunk: not self._monotone_limit_excess(self._chunk_messages(chunk), summary=True),
                                min(3, remaining - 1))
                        # Validate bounded audit metadata before any model work.
                        for index, leaf in enumerate(leaves):
                            compaction_metadata(self.config.compaction_policy, "leaf", plan_id,
                                tuple(reference for atom in leaf for reference in atom.references),
                                leaf_index=index, leaf_count=len(leaves))
                        break
                    except AgentLimitError as error:
                        failure = error
                else:
                    if fresh_decision and self._fits(snapshot()):
                        return commit()
                    raise failure
                note, sequence = await self._summarize_chunk_plan(identity, span, leaves, plan_id)
                sources = tuple(number for entry in span for number in entry.source_sequences)
                originals = tuple(message for entry in span for message in entry.messages)
                if _metrics((note,))[1] >= _metrics(originals)[1]:
                    raise AgentLimitError("merged summary did not reduce the complete source prefix")
            replacement = _WorkingEntry(span[0].sequence, "summary", span[0].identity, (note,), sources, sequence)
            staged = [replace(entry, messages=tuple(replace(message, native_assistant=None)
                      for message in entry.messages)) for entry in (replacement, *staged[len(span):])]
            changed = True
            retry_instruction = last_failure = None
        return commit()

    async def complete(self, request: AgentRequest) -> AgentResponse:
        if self._busy:
            raise AgentReuseError("runtime already has active model work")
        self.accept_input(request)
        if request.identity != self._current:
            raise AgentSupersededError("input was superseded before completion")
        if self._closed or request.identity in self._action_attempted:
            raise AgentReuseError("input is closed, superseded, or already completed")
        self._busy = True
        session = None
        try:
            if self.tools is not None:
                session = self.tools.begin(request)
            timeout = self.config.timeout_seconds
            if self.tools is not None:
                timeout = min(timeout, self.tools.limits.timeout_seconds)
            async with _DecisionDeadline(timeout) as deadline:
                self._deadline = deadline
                return await self._complete_decision(request, session)
        finally:
            if session is not None:
                session.close()
            self._busy = False
            self._deadline = None

    async def _complete_decision(self, request: AgentRequest, session) -> AgentResponse:
        included_events = self._required_events.copy()
        pinned = self._pinned or ()
        messages, local_working = await self._compact(request.identity, fresh_decision=True)
        self._action_attempted.add(request.identity)
        usages: list[ModelUsage] = []
        native_tool_turn = False
        rounds = self.tools.limits.max_model_rounds if self.tools is not None else 1
        for _ in range(rounds):
            if not self._fits(messages):
                if native_tool_turn:
                    # The tool result continues the same assistant turn. Do
                    # not rewrite a prefix used to generate its native blocks.
                    raise AgentLimitError("native tool continuation cannot fit context bounds without changing its prefix")
                messages, local_working = await self._compact(request.identity, original=local_working,
                    pinned=pinned, required_events=frozenset(included_events))
            if session is not None and session.generation != self.tools._generation:
                raise AgentSupersededError("tool scope was superseded")
            submission_failure = None
            try:
                completion = await self._call_model(self._model, self._model_request(messages, "action"))
            except SubmissionResponseError as error:
                submission_failure = error
                completion = error.completion
            if not isinstance(completion, ModelCompletion):
                raise AgentError("action model returned an invalid completion")
            usages.append(completion.usage)
            self._action_usage.append(completion.usage)
            native = completion.native_assistant
            context = completion.context_messages or (Message("assistant", (TextPart(completion.text),),
                       completion.tool_calls, native_assistant=native),)
            native_json = "".join(message.native_assistant.message_json
                                  for message in context if message.native_assistant is not None)
            if (len(completion.text) + len(native_json) > self.config.max_action_output_characters
                or len(completion.text.encode("utf-8")) + len(native_json.encode("utf-8")) > self.config.max_action_output_bytes):
                self._append_archive(ArchiveEntry(len(self._archive) + 1, "completion", request.identity,
                                                messages=context,
                                                text=completion.text, usage=completion.usage))
                raise AgentLimitError("action output exceeds its hard bounds")
            if not completion.tool_calls:
                break
            if session is None:
                raise AgentError("model requested unavailable tools")
            assistant = Message("assistant", (TextPart(completion.text),) if completion.text else (), completion.tool_calls,
                                native_assistant=native)
            native_tool_turn = native_tool_turn or native is not None or submission_failure is not None
            group = [assistant]
            if submission_failure is not None:
                group.extend(self._submission_feedback(submission_failure, session))
            else:
                for call in completion.tool_calls:
                    self._deadline.check()
                    self._reserve_tool_audit(request.identity, call, session)
                    try:
                        result = self.tools.execute(call, session)
                    except Exception as error:
                        self._append_archive(self._tool_audit_entry(request.identity, call, error=error))
                        raise
                    # Keep the exact returned result before checking another limit
                    # or executing the next call. These complete single-call audit
                    # pairs stay out of the model's working conversation.
                    self._append_archive(self._tool_audit_entry(request.identity, call, result))
                    self._deadline.check()
                    group.append(Message("tool", result.parts, tool_call_id=result.call_id))
            sequence = len(self._archive) + 1
            exchange = tuple(group)
            self._append_archive(ArchiveEntry(sequence, "tool_exchange", request.identity, exchange, usage=completion.usage))
            messages += exchange
            entry = _WorkingEntry(sequence, "tool_exchange", request.identity, exchange, (sequence,))
            local_working += (entry,)
            if self._current is not None and _scope(request.identity) == _scope(self._current):
                self._working.append(entry)
        else:
            raise AgentLimitError("model tool round budget exhausted")
        sequence = len(self._archive) + 1
        self._append_archive(ArchiveEntry(sequence, "completion", request.identity,
                                        messages=completion.context_messages or (context if native is not None else ()),
                                        text=completion.text, usage=completion.usage))
        self._completed.add(request.identity)
        if self._current is not None and _scope(request.identity) == _scope(self._current):
            # A native submission pair is trusted adapter evidence. The action
            # stays in archive.text; neither this call nor its acknowledgement
            # is executed by the runtime or dispatched through ToolRegistry.
            self._working.append(_WorkingEntry(sequence, "completion", request.identity, context, (sequence,)))
            self._required_events.difference_update(included_events)
        def total(field: str):
            values = [getattr(usage, field) for usage in usages]
            return sum(values) if all(v is not None for v in values) else None
        return AgentResponse(completion.text, ModelUsage(total("input_tokens"), total("output_tokens"), total("total_tokens")))

    def _submission_feedback(self, failure: SubmissionResponseError, session) -> tuple[Message, ...]:
        """Reject every call without dispatch, charging the existing tool budget.

        This runs only after ordinary completion/native output checks. Feedback
        has fixed guidance and a bounded validator diagnostic; no submitted
        arguments are interpolated into it.
        """
        self._deadline.check()
        calls = failure.completion.tool_calls
        text = json.dumps({"error": "invalid_action_submission", "detail": failure.reason,
            "batch_rejected": True, "executed": False,
            "retry": "No call in this batch executed. Correct the format and use new call IDs. "
                     "Call host tools separately; finish with exactly one standalone submit_action "
                     "whose arguments contain only format and content."}, separators=(",", ":"))
        size = len(text.encode("utf-8"))
        limits = self.tools.limits
        # Match ToolRegistry's scope and call-ID rules without invoking any
        # handler, even for a host call preceding the invalid submission.
        with self.tools._lock:
            if (session._owner is not self.tools or session.closed or session.generation != self.tools._generation
                or self.tools._current is None or _scope(session.identity) != _scope(self.tools._current)):
                raise AgentReuseError("tool scope is no longer active")
            if time.monotonic() >= session.deadline or session.calls + len(calls) > limits.max_calls:
                raise AgentLimitError("tool call/time budget exhausted")
            if any(call.call_id in session._ids for call in calls):
                raise AgentReuseError("tool call ID was already executed")
            if size > limits.max_text_bytes or session.text_bytes + size * len(calls) > limits.max_total_text_bytes:
                raise AgentLimitError("tool result budget exhausted")
            session.calls += len(calls)
            session._ids.update(call.call_id for call in calls)
            session.text_bytes += size * len(calls)
        return tuple(Message("tool", (TextPart(text),), tool_call_id=call.call_id) for call in calls)
