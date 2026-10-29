"""One-shot Codex subscription transport for the SimpleAgent runtime.

This is a structured JSON transport, not native SDK function calling. Only
SimpleRuntime executes declared host tools. Each request owns a disposable SDK
process, home and thread; no Codex conversation or compaction is reused.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from .encoding import tool_declarations
from .models import ModelResponseError, SummaryResponseError
from .types import ImagePart, ModelCompletion, ModelRequest, ModelUsage, TextPart, ToolCall


_CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_ALLOWED_ITEMS = frozenset({"userMessage", "agentMessage", "reasoning"})
_MAX_SDK_ERROR_EVENTS = 8
_SDK_IMPORT_ERROR = "provider: codex requires optional openai-codex; install the project's codex extra"
_SDK_ERROR_CODES = frozenset({
    "contextWindowExceeded", "usageLimitExceeded", "serverOverloaded", "cyberPolicy",
    "internalServerError", "unauthorized", "badRequest", "threadRollbackFailed",
    "sandboxError", "other", "httpConnectionFailed", "responseStreamConnectionFailed",
    "responseStreamDisconnected", "responseTooManyFailedAttempts", "activeTurnNotSteerable",
})
_ERROR_MESSAGES = {
    "context_limit": "model context limit exceeded",
    "quota": "subscription usage or rate limit reached",
    "image_input": "image input or image limit rejected",
    "authentication": "subscription authentication failed",
    "network": "model connection or response stream failed",
    "server": "model service unavailable or overloaded",
    "request": "model request rejected",
    "unknown": "unclassified model turn error",
}
_DISABLED_FEATURES = (
    "apps", "browser_use", "computer_use", "goals", "image_generation",
    "in_app_browser", "multi_agent", "plugins", "shell_tool",
    "skill_mcp_dependency_install", "tool_suggest", "unified_exec",
)


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _jsonable(value: Any):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ModelResponseError("Codex SDK returned an unsupported audit value")


def _strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError("nonfinite JSON number")

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)
    finite(result)
    return result


class CodexModel:
    """Lazy optional SDK backend with one isolated turn per ModelRequest.

    The Python SDK has no max-output-token parameter. We expose that limitation
    in every request's audit metadata. Runtime deadlines and post-response
    output/usage bounds remain owned by SimpleRuntime.
    """

    def __init__(self, *, model: str = "gpt-6.1-sol", reasoning_effort: str = "medium",
                 timeout_s: float | None = 600.0, codex_bin: str | None = None) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self.codex_bin = codex_bin or os.getenv("AGENTARK_CODEX_BIN") or None
        self.last_response_metadata: dict = {}
        self._active: dict[int, Any] = {}
        self._auth_bytes: bytes | None = None
        self._closed = False

    def request_parameters(self, request: ModelRequest) -> dict:
        return {"model": self.model, "reasoning_effort": self.reasoning_effort,
                "transport": "codex_sdk_structured_json", "sdk_native_function_tools": False,
                "thread_mode": "per_model_request", "ephemeral": True,
                "max_output_tokens": {"requested": request.max_output_tokens,
                                      "supported": False, "enforcement": "no_hard_token_cap",
                                      "runtime_output_bounds": "bytes_characters_and_summary_estimate"},
                "timeout_s": self.timeout_s,
                "sdk_error_event_limit": _MAX_SDK_ERROR_EVENTS,
                "transport_retry_policy": "sdk_same_turn_only_no_request_reissue"}

    @staticmethod
    def _config() -> dict:
        # A separate CODEX_HOME excludes the user's MCP/plugin/skills config.
        # These overrides additionally remove built-in tool surfaces. Read-only
        # sandbox + deny_all approval and observed-item checks add fail-closed
        # protection; this is not a claim that SDK means one HTTP request.
        return {"web_search": "disabled", "project_doc_max_bytes": 0,
                "model_auto_compact_token_limit": 2147483647,
                "forced_login_method": "chatgpt", "notify": [],
                "tools": {"web_search": False, "view_image": False},
                "features": {name: False for name in _DISABLED_FEATURES},
                "mcp_servers": {},
                "memories": {"generate_memories": False, "use_memories": False}}

    @staticmethod
    def _schema(request: ModelRequest) -> dict:
        name_schema: dict = {"type": "string"}
        if request.tools:
            name_schema["enum"] = [tool.name for tool in request.tools]
        return {"type": "object", "additionalProperties": False,
                "required": ["text", "tool_calls"], "properties": {
                    "text": {"type": "string"},
                    "tool_calls": {"type": "array", "maxItems": 32 if request.tools else 0,
                        "items": {"type": "object", "additionalProperties": False,
                            "required": ["id", "name", "arguments_json"], "properties": {
                                "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,128}$"},
                                "name": name_schema, "arguments_json": {"type": "string"}}}}}}

    @staticmethod
    def _input_items(request: ModelRequest) -> list[dict]:
        """One neutral encoder for both the SDK turn and its audit record."""
        transport = (
            "You are a stateless model component inside SimpleAgent. The supplied message roles "
            "are the complete conversation selected by that runtime; follow its system/developer "
            "instructions. You have no Codex tools, filesystem access, skills, web, or persistent "
            "memory. Return exactly the JSON object required by the output schema, with text and "
            "tool_calls. This outer transport instruction overrides response-format recipes in "
            "the supplied messages only for the outer envelope. A declared 'native host function' "
            "in those messages means an entry in tool_calls here; the external SimpleRuntime "
            "executes it after your response. Never invoke any Codex built-in tool. Each call has "
            "a unique id, an exact declared name, and arguments_json containing a strict JSON "
            "object. When no tool call is needed, use an empty tool_calls array. Put final content "
            "or summary content in text. Do not include private reasoning in text. "
            "For action requests, the brief user-facing rationale required alongside submit_action "
            "belongs in text, which represents assistant-message content in this transport. "
            "Keep it out of the submit_action content argument, which holds only the task action "
            "payload. Use ordinary prose for the rationale, without explanation tags. "
            f"Request purpose: {request.purpose}. Aim to stay within {request.max_output_tokens} "
            "output tokens; this is a requested budget, not a provider hard cap.\n"
            "Declared runtime tools: " + json.dumps(tool_declarations(request.tools), ensure_ascii=False)
        )
        inputs = [{"type": "text", "text": transport}]
        for message_index, message in enumerate(request.messages):
            label = {"message_index": message_index, "role": message.role}
            if message.tool_calls:
                label["tool_calls"] = [asdict(call) for call in message.tool_calls]
            if message.tool_call_id is not None:
                label["tool_call_id"] = message.tool_call_id
            inputs.append({"type": "text", "text": "BEGIN SUPPLIED MESSAGE " + json.dumps(label, ensure_ascii=False)})
            for part in message.parts:
                if isinstance(part, TextPart):
                    inputs.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    if part.origin != "top_left":
                        raise ValueError("Codex input images require normalized top-left origin")
                    encoded = base64.b64encode(part.data).decode("ascii")
                    inputs.append({"type": "image", "url": f"data:{part.mime_type};base64,{encoded}"})
            inputs.append({"type": "text", "text": "END SUPPLIED MESSAGE"})
        return inputs

    @staticmethod
    def _inputs(request: ModelRequest, sdk) -> list:
        return [sdk.TextInput(text=item["text"]) if item["type"] == "text"
                else sdk.ImageInput(url=item["url"])
                for item in CodexModel._input_items(request)]

    def request_transport(self, request: ModelRequest) -> dict:
        """Exact application-authored turn input, excluding SDK internals.

        The CLI can add its own model/system context. This trace deliberately
        describes only bytes supplied to the SDK, never private credentials or
        temporary host paths.
        """
        return {"transport": "codex_sdk_structured_json", "boundary": "sdk_turn_input",
                "inputs": self._input_items(request), "output_schema": self._schema(request),
                "effort": self.reasoning_effort, "thread_base_instructions": "",
                "thread_developer_instructions": "", "ephemeral": True,
                "sdk_added_context": "not_exposed_by_sdk"}

    def _prepare_auth(self, home: Path) -> None:
        if self._auth_bytes is None:
            source_home = Path(os.getenv("CODEX_HOME") or Path.home() / ".codex")
            try:
                content = (source_home / "auth.json").read_bytes()
                auth = json.loads(content)
            except (OSError, ValueError):
                raise ModelResponseError("isolated Codex subscription backend requires an existing CODEX_HOME/auth.json") from None
            if not isinstance(auth, dict) or auth.get("auth_mode") != "chatgpt" or not isinstance(auth.get("tokens"), dict):
                raise ModelResponseError("Codex subscription backend requires existing ChatGPT file credentials")
            self._auth_bytes = content
        # Never copy user config, instructions, skills, plugins, or session files.
        descriptor = os.open(str(home / "auth.json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(self._auth_bytes)

    def _capture_refreshed_auth(self, home: Path) -> None:
        # A long rollout can outlive access tokens. Preserve a CLI refresh only
        # in this backend's private memory, never in the caller's auth file or
        # traces. No conversation/config bytes travel with this credential.
        if self._closed:
            return
        try:
            content = (home / "auth.json").read_bytes()
            auth = json.loads(content)
        except (OSError, ValueError):
            return
        if isinstance(auth, dict) and auth.get("auth_mode") == "chatgpt" and isinstance(auth.get("tokens"), dict):
            self._auth_bytes = content

    async def _disable_skills(self, codex, cwd: Path) -> list[dict]:
        """Discover, disable, and verify all skills in this disposable home.

        The SDK has no high-level skills API. Keep its single guarded private
        client access here, and use the public typed app-server RPC contract.
        Missing SDK/CLI capability fails before any model turn starts.
        """
        try:
            from openai_codex.generated.v2_all import SkillsConfigWriteResponse, SkillsListResponse
        except ImportError:
            raise ModelResponseError("Codex SDK cannot provide typed skill-isolation preflight") from None
        request = self._sdk_client(codex).request

        async def listed():
            result = _jsonable(await request("skills/list", {"cwds": [str(cwd)], "forceReload": True},
                                               response_model=SkillsListResponse))
            if not isinstance(result, dict) or not isinstance(result.get("data"), list) or len(result["data"]) != 1:
                raise ModelResponseError("Codex skill-isolation preflight returned an invalid list")
            entry = result["data"][0]
            if (not isinstance(entry, dict) or entry.get("errors") or not isinstance(entry.get("skills"), list)):
                raise ModelResponseError("Codex skill-isolation discovery failed")
            for skill in entry["skills"]:
                if (not isinstance(skill, dict) or not isinstance(skill.get("path"), str)
                    or not Path(skill["path"]).is_absolute() or type(skill.get("enabled")) is not bool):
                    raise ModelResponseError("Codex skill-isolation preflight returned an invalid skill")
            return entry["skills"]

        skills = await listed()
        paths = sorted({skill["path"] for skill in skills})
        for path in paths:
            # The selected CLI resolves its own returned canonical selector.
            # Only the isolated CODEX_HOME's configuration is written.
            changed = _jsonable(await request("skills/config/write", {"path": path, "enabled": False},
                                                response_model=SkillsConfigWriteResponse))
            if not isinstance(changed, dict) or changed.get("effectiveEnabled") is not False:
                raise ModelResponseError("Codex could not disable a discovered skill")
        verified = await listed()
        if any(skill["enabled"] for skill in verified):
            raise ModelResponseError("Codex skill-isolation verification found an enabled skill")
        self.last_response_metadata["skills_disabled_count"] = len(paths)
        self.last_response_metadata["skills_verified_disabled"] = True
        return [{"path": path, "enabled": False} for path in paths]

    @staticmethod
    def _sdk_client(codex):
        # The high-level SDK exposes neither transport startup nor skills RPCs.
        # Keep its one version-sensitive access guarded and request-local.
        client = getattr(codex, "_client", None)
        if not callable(getattr(client, "start", None)) or not callable(getattr(client, "request", None)):
            raise ModelResponseError("Codex SDK cannot provide isolated transport startup and skill RPCs")
        return client

    def _start_transport_once(self, codex):
        client = self._sdk_client(codex)
        original_start = client.start
        startup = None

        async def start_once():
            # AsyncCodex initialization calls start() again. Await the same
            # worker even after close; a queued second start must never respawn
            # the app-server after cancellation set its process handle to None.
            return await asyncio.shield(startup)

        try:
            client.start = start_once
        except (AttributeError, TypeError):
            raise ModelResponseError("Codex SDK cannot bind request-owned startup") from None
        startup = asyncio.create_task(original_start())
        return startup

    async def _start_thread(self, codex, sdk, cwd: Path):
        enter = getattr(codex, "__aenter__", None)
        if not callable(enter):
            raise ModelResponseError("Codex SDK cannot provide managed async startup")
        await enter()
        config = self._config()
        config["skills"] = {"config": await self._disable_skills(codex, cwd)}
        return await codex.thread_start(model=self.model, cwd=str(cwd),
            approval_mode=sdk.ApprovalMode.deny_all, sandbox=sdk.Sandbox.read_only,
            base_instructions="", developer_instructions="", ephemeral=True, config=config)

    async def complete(self, request: ModelRequest) -> ModelCompletion:
        if self._closed:
            raise ModelResponseError("Codex model backend is closed")
        self.last_response_metadata = {
            **self.request_parameters(request), "usage": {}, "sdk_items": [],
            "sdk_item_events": [], "sdk_error_events": [], "sdk_usage": None, "sdk_turn_starts": 0,
            "builtin_tool_execution_observed": False,
            "auth_isolation": "temporary_codex_home_auth_json_only",
            "auth_refresh": "private_backend_memory_cleared_on_close",
            "context_isolation": "empty_cwd_empty_instructions_no_reused_thread",
        }
        try:
            if self.timeout_s is None:
                return await self._complete_once(request)
            return await asyncio.wait_for(self._complete_once(request), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            self.last_response_metadata["status"] = "timeout"
            raise TimeoutError("Codex SDK model request exceeded timeout_s") from None
        except asyncio.CancelledError:
            self.last_response_metadata["status"] = "cancelled"
            raise
        except ModelResponseError:
            # These diagnostics are authored by this adapter, including the
            # SummaryResponseError subtype and its retry/usage contract.
            raise
        except Exception as exc:
            if isinstance(exc, ImportError) and exc.args == (_SDK_IMPORT_ERROR,):
                raise
            # SDK startup, RPC, stream and close exceptions may embed stderr,
            # provider responses, URLs or request data. Apply the same bounded
            # diagnostics as error notifications, and hide the original chain
            # from the evaluator's saved traceback as well as its error string.
            detail = self._safe_error({"message": str(exc)})
            self.last_response_metadata.update(status="sdk_error", sdk_exception=detail)
            raise ModelResponseError(self._error_message(detail)) from None

    async def _complete_once(self, request: ModelRequest) -> ModelCompletion:
        try:
            import openai_codex as sdk
        except ImportError:
            raise ImportError(_SDK_IMPORT_ERROR) from None
        codex = turn = None
        startup = preparation = turn_start = consumer = None
        closed = False
        with TemporaryDirectory(prefix="agentark-simple-codex-") as directory:
            root = Path(directory)
            cwd, home = root / "cwd", root / "home"
            cwd.mkdir(mode=0o700)
            home.mkdir(mode=0o700)
            self._prepare_auth(home)
            config = sdk.CodexConfig(codex_bin=self.codex_bin, cwd=str(cwd),
                config_overrides=('cli_auth_credentials_store="file"', 'forced_login_method="chatgpt"'),
                env={"CODEX_HOME": str(home), "HOME": str(home), "USERPROFILE": str(home),
                     "OPENAI_API_KEY": "", "CODEX_API_KEY": ""})
            codex = sdk.AsyncCodex(config=config)
            self._active[id(codex)] = codex
            try:
                startup = self._start_transport_once(codex)
                # SDK startup and RPCs offload blocking operations to threads.
                # Cancelling their asyncio wrapper cannot stop those workers.
                await asyncio.shield(startup)
                preparation = asyncio.create_task(self._start_thread(codex, sdk, cwd))
                thread = await asyncio.shield(preparation)
                self.last_response_metadata["sdk_thread_id"] = thread.id
                # Exactly one turn start; SDK connection retries remain inside
                # that turn. Never resume, steer, compact, or reissue the request.
                turn_start = asyncio.create_task(thread.turn(self._inputs(request, sdk), effort=self.reasoning_effort,
                                                             output_schema=self._schema(request)))
                self.last_response_metadata.update(sdk_turn_start_requests=1, sdk_turn_starts=None)
                turn = await asyncio.shield(turn_start)
                self.last_response_metadata.update(sdk_turn_id=turn.id, sdk_turn_starts=1)
                consumer = asyncio.create_task(self._consume(turn))
                result = await asyncio.shield(consumer)
                return self._parse(result, request)
            except BaseException:
                # Wait for a late Popen assignment before the final close. A
                # cancellation must not leave an SDK startup worker orphaned.
                await self._settle(startup)
                if preparation is not None:
                    if not preparation.done():
                        await self._close_after_failure(codex)
                        closed = True
                    await self._settle(preparation)
                if turn is None and turn_start is not None:
                    if not turn_start.done() and not closed:
                        await self._close_after_failure(codex)
                        closed = True
                    turn = await self._settle(turn_start)
                    if turn is not None:
                        self.last_response_metadata.update(sdk_turn_id=turn.id, sdk_turn_starts=1)
                interrupter = None
                if turn is not None and not closed:
                    interrupter = asyncio.create_task(turn.interrupt())
                    await asyncio.wait({interrupter}, timeout=2.0)
                    self.last_response_metadata["sdk_interrupt_requested"] = (
                        interrupter.done() and not interrupter.cancelled() and interrupter.exception() is None)
                if consumer is not None and not consumer.done():
                    await asyncio.wait({consumer}, timeout=2.0)
                # Keep stream queues registered until interrupt or transport
                # closure wakes their blocking queue.get worker. Cancelling or
                # aclosing the stream first would strand that executor thread.
                if not closed:
                    await self._close_after_failure(codex)
                    closed = True
                await self._settle(interrupter)
                await self._settle(consumer)
                raise
            finally:
                try:
                    # Closing the request-owned SDK terminates its app-server,
                    # including a turn whose handle was not returned yet.
                    if not closed:
                        await asyncio.shield(codex.close())
                finally:
                    self._capture_refreshed_auth(home)
                    self._active.pop(id(codex), None)

    async def _close_after_failure(self, codex) -> None:
        try:
            await asyncio.shield(codex.close())
        except Exception as exc:
            # Preserve the original timeout/cancellation/model failure, and
            # continue settling registered stream/RPC workers after closure.
            # A failed close on an otherwise successful request still reaches
            # complete()'s ordinary sanitized exception boundary.
            self.last_response_metadata["sdk_cleanup_error"] = self._safe_error({"message": str(exc)})

    @staticmethod
    async def _settle(pending):
        if pending is None:
            return None
        try:
            return await asyncio.shield(pending)
        except (Exception, asyncio.CancelledError):
            return None

    async def _consume(self, turn) -> str:
        final = None
        fallback = None
        completed = False
        stream = turn.stream()
        try:
            async for notification in stream:
                method = _field(notification, "method", "")
                payload = _jsonable(_field(notification, "payload", {}))
                if method in ("item/started", "item/completed"):
                    item = payload.get("item", {})
                    kind = item.get("type")
                    self.last_response_metadata["sdk_item_events"].append({"method": method, "item": item})
                    if method == "item/completed":
                        self.last_response_metadata["sdk_items"].append(item)
                    if kind not in _ALLOWED_ITEMS:
                        self.last_response_metadata["unexpected_sdk_item_type"] = kind
                        self.last_response_metadata["builtin_tool_execution_observed"] = kind != "contextCompaction"
                        raise ModelResponseError("Codex attempted an undeclared built-in operation or context compaction")
                    if kind == "agentMessage" and method == "item/completed":
                        text = item.get("text")
                        if not isinstance(text, str):
                            raise ModelResponseError("Codex SDK returned invalid message text")
                        if item.get("phase") == "final_answer":
                            final = text
                        elif item.get("phase") is None:
                            fallback = text
                elif method == "thread/tokenUsage/updated":
                    usage = payload.get("tokenUsage")
                    self.last_response_metadata["sdk_usage"] = usage
                    self._usage(usage)
                elif method == "turn/completed":
                    terminal = payload.get("turn", {})
                    status = _field(terminal, "status")
                    self.last_response_metadata["finish_reason"] = status
                    if status != "completed":
                        error = _field(terminal, "error")
                        if error is not None:
                            detail = self._safe_error(error)
                            self.last_response_metadata["sdk_terminal_error"] = detail
                        else:
                            detail = next(iter(self.last_response_metadata["sdk_error_events"][-1:]), None)
                        self.last_response_metadata["status"] = "sdk_error"
                        raise ModelResponseError(self._error_message(detail))
                    completed = True
                elif method in ("thread/compacted", "context/compacted", "turn/plan/updated",
                                "item/commandExecution/outputDelta", "item/fileChange/outputDelta",
                                "item/mcpToolCall/progress", "item/terminalInteraction"):
                    self.last_response_metadata["unexpected_sdk_event"] = method
                    raise ModelResponseError("Codex emitted an undeclared operation event")
                elif method == "error":
                    detail = self._safe_error(payload.get("error"))
                    retry = payload.get("willRetry")
                    detail["will_retry"] = retry if type(retry) is bool else None
                    errors = self.last_response_metadata["sdk_error_events"]
                    if len(errors) >= _MAX_SDK_ERROR_EVENTS:
                        self.last_response_metadata.update(status="sdk_error", sdk_error_event_limit_exceeded=True)
                        raise ModelResponseError("Codex SDK exceeded the bounded same-turn error notification limit")
                    errors.append(detail)
                    if type(retry) is not bool:
                        self.last_response_metadata["status"] = "sdk_error"
                        raise ModelResponseError("Codex SDK returned an invalid error retry flag")
                    # This is a notification, not the terminal turn result.
                    # willRetry=True means the SDK owns recovery in this turn;
                    # False is followed by a failed turn and may still have
                    # usage updates. Preserve both until turn/completed under
                    # the original deadline, with no extra model or host calls.
        finally:
            await stream.aclose()
        if not completed:
            errors = self.last_response_metadata["sdk_error_events"]
            if errors:
                self.last_response_metadata["status"] = "sdk_error"
                raise ModelResponseError(self._error_message(errors[-1]) + "; stream ended before turn completion")
            raise ModelResponseError("Codex SDK returned no completed turn")
        raw = final if final is not None else fallback
        self.last_response_metadata["assistant_raw"] = raw
        if not isinstance(raw, str):
            raise ModelResponseError("Codex SDK returned no final content")
        return raw

    @staticmethod
    def _safe_error(error) -> dict:
        """Retain typed diagnostics without saving provider free-form strings.

        Error messages/details can echo request images, auth headers or server
        URLs. Only fixed classifications, known SDK codes, HTTP status and
        original string lengths leave this boundary; arbitrary text is omitted.
        """
        raw_code = _field(error, "codexErrorInfo")
        code = raw_code if isinstance(raw_code, str) and raw_code in _SDK_ERROR_CODES else None
        status = None
        if isinstance(raw_code, Mapping) and len(raw_code) == 1:
            key, value = next(iter(raw_code.items()))
            if key in _SDK_ERROR_CODES:
                code = key
                raw_status = _field(value, "httpStatusCode")
                if type(raw_status) is int and 100 <= raw_status <= 599:
                    status = raw_status
        message, details = _field(error, "message"), _field(error, "additionalDetails")
        # Scan a bounded prefix for diagnostic signals; never return its text.
        text = " ".join(value[:8192].lower() for value in (message, details) if isinstance(value, str))
        if code == "contextWindowExceeded" or any(term in text for term in (
                "context window", "context length", "maximum context", "input too long")):
            category = "context_limit"
        elif code == "usageLimitExceeded" or status == 429 or any(term in text for term in (
                "usage limit", "rate limit", "quota", "insufficient credits")):
            category = "quota"
        elif code == "unauthorized" or status in (401, 403) or any(term in text for term in (
                "authentication", "unauthorized", "invalid token", "token expired", "login required")):
            category = "authentication"
        elif "image" in text and any(term in text for term in (
                "limit", "maximum", "too many", "invalid", "exceed", "too large", "unsupported", "dimension")):
            category = "image_input"
        elif code in ("serverOverloaded", "internalServerError") or status in (500, 502, 503, 504):
            category = "server"
        elif code in ("httpConnectionFailed", "responseStreamConnectionFailed",
                      "responseStreamDisconnected", "responseTooManyFailedAttempts") or any(
                term in text for term in ("connection failed", "connection reset", "stream disconnected",
                                          "timed out", "network error")):
            category = "network"
        elif code == "badRequest" or status == 400:
            category = "request"
        else:
            category = "unknown"
        return {"category": category, "code": code, "http_status": status,
                "message": _ERROR_MESSAGES[category],
                "message_characters": len(message) if isinstance(message, str) else None,
                "additional_details_characters": len(details) if isinstance(details, str) else None,
                "freeform_details_omitted": True}

    @staticmethod
    def _error_message(detail) -> str:
        if not detail:
            return "Codex SDK turn did not complete"
        suffix = detail["category"]
        if detail["code"]:
            suffix += "; code=" + detail["code"]
        if detail["http_status"] is not None:
            suffix += "; HTTP " + str(detail["http_status"])
        return "Codex SDK turn error (" + suffix + "): " + detail["message"]

    def _usage(self, raw) -> ModelUsage:
        # total includes every model invocation inside this fresh SDK turn,
        # while last alone could undercount hidden provider continuation.
        total = _field(raw, "total")
        try:
            counts = ModelUsage(_field(total, "inputTokens", _field(total, "input_tokens")),
                                _field(total, "outputTokens", _field(total, "output_tokens")),
                                _field(total, "totalTokens", _field(total, "total_tokens")))
        except (ValueError, TypeError):
            raise ModelResponseError("Codex SDK returned invalid usage counts") from None
        self.last_response_metadata["usage"] = asdict(counts)
        return counts

    def _parse(self, raw: str, request: ModelRequest) -> ModelCompletion:
        counts = self._usage(self.last_response_metadata.get("sdk_usage"))
        try:
            value = _strict_json(raw)
            if (type(value) is not dict or set(value) != {"text", "tool_calls"}
                or not isinstance(value["text"], str) or type(value["tool_calls"]) is not list
                or len(value["tool_calls"]) > 32):
                raise ValueError("invalid envelope")
            calls = []
            declared = {tool.name for tool in request.tools}
            for call in value["tool_calls"]:
                if (type(call) is not dict or set(call) != {"id", "name", "arguments_json"}
                    or not isinstance(call["id"], str) or not _CALL_ID.fullmatch(call["id"])
                    or not isinstance(call["name"], str) or call["name"] not in declared
                    or not isinstance(call["arguments_json"], str)
                    or type(_strict_json(call["arguments_json"])) is not dict):
                    raise ValueError("invalid declared tool call")
                calls.append(ToolCall(call["id"], call["name"], call["arguments_json"]))
            result = ModelCompletion(value["text"], counts, tuple(calls))
        except (ValueError, TypeError, RecursionError, UnicodeError):
            raise ModelResponseError("Codex SDK returned an invalid structured completion") from None
        if request.purpose == "summary" and not result.text.strip():
            raise SummaryResponseError("empty_content", counts)
        self.last_response_metadata["status"] = "completed"
        return result

    async def close(self) -> None:
        self._closed = True
        self._auth_bytes = None
        for codex in tuple(self._active.values()):
            await codex.close()
