import asyncio
import base64
import json
import tempfile
import time
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_ark.agent.simple_agent.agent import SimpleAgent
from agent_ark.agent.simple_agent.action_submission import SUBMISSION_SPEC
from agent_ark.agent.simple_agent.encoding import encode_png
from agent_ark.agent.simple_agent.types import (
    AgentLimitError, ImagePart, ModelCompletion, ModelUsage, TextPart, ToolCall, ToolSpec,
)


ACTION = '<tool_call>{"name":"tap","arguments":{"x":3}}</tool_call>'


def obs(text="look", *, image=None, system=None):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    parts = [{"type": "text", "text": text}]
    if image is not None:
        parts.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")}})
    messages.append({"role": "user", "content": parts})
    return {0: {"messages": messages}}


def text_of(request):
    return "\n".join(part.text for message in request.messages for part in message.parts if isinstance(part, TextPart))


def submission(kind, content, *, call_id="submit", text="", usage=None):
    return ModelCompletion(text, usage or ModelUsage(2, 3, 5), (ToolCall(call_id, "submit_action",
        json.dumps({"format": kind, "content": content}, ensure_ascii=False)),))


class FakeModel:
    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.requests = []
        self.loops = []

    async def complete(self, request):
        self.requests.append(request)
        self.loops.append(asyncio.get_running_loop())
        if request.purpose == "summary":
            return ModelCompletion("Earlier observations and actions retained as notes.", ModelUsage(2, 3, 5))
        if self.replies:
            return self.replies.pop(0)
        if any(tool.name == "submit_action" for tool in request.tools):
            return submission("tool_call", ACTION[len("<tool_call>"):-len("</tool_call>")], usage=ModelUsage(7, 11, 18))
        return ModelCompletion(ACTION, ModelUsage(7, 11, 18))


class SimpleAgentTests(unittest.TestCase):
    def agent(self, backend=None, **kwargs):
        agent = SimpleAgent(provider="openai", model_backend=backend or FakeModel(), **kwargs)
        self.addCleanup(agent.close)
        return agent

    def test_preserves_raw_answer_and_deduplicates_only_immediate_echo(self):
        raw = "<think>I can tap.</think>\n" + ACTION
        backend = FakeModel([ModelCompletion(raw), ModelCompletion(ACTION), ModelCompletion(ACTION)])
        agent = self.agent(backend)
        first, trace = agent.forward_with_trace(obs("same observation", system="task"))
        self.assertEqual(first[0], {"action": ACTION, "assistant": raw})
        self.assertEqual(trace[0]["model_requests"][0]["completion"], raw)
        second = {0: {"messages": [{"role": "assistant", "content": raw},
                                  {"role": "user", "content": "same observation"}]}}
        preview = agent.build_request_messages(second)
        self.assertEqual(len(agent._states[0].runtime.history_snapshot), 2)
        self.assertEqual(sum("same observation" in str(message["content"]) for message in preview[0] if message["role"] == "user"), 2)
        agent.forward(second)
        sent = text_of(backend.requests[1])
        self.assertEqual(sent.count(raw), 1)
        self.assertEqual(sum(part.text.count("same observation") for message in backend.requests[1].messages
                             if message.role == "user" for part in message.parts if isinstance(part, TextPart)), 2)
        agent.forward({0: {"messages": [{"role": "assistant", "content": "different previous answer"},
                                       {"role": "user", "content": "same observation"}]}})
        self.assertIn("different previous answer", text_of(backend.requests[2]))
        self.assertEqual(sum(part.text.count("same observation") for message in backend.requests[2].messages
                             if message.role == "user" for part in message.parts if isinstance(part, TextPart)), 3)

    def test_only_one_echo_is_removed(self):
        backend = FakeModel()
        agent = self.agent(backend)
        agent.forward(obs())
        agent.forward({0: {"messages": [{"role": "assistant", "content": ACTION},
                                       {"role": "assistant", "content": ACTION},
                                       {"role": "user", "content": "new result"}]}})
        self.assertEqual(text_of(backend.requests[-1]).count(ACTION), 2)

    def test_preview_is_pure_and_detached_even_with_tools_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeModel()
            parent = Path(directory) / "scratch"
            agent = self.agent(backend, simple_agent_config={"tools": {"enabled": True, "workspace_parent": str(parent)}})
            value = obs()
            first = agent.build_request_messages(value)
            first[0][-1]["content"][0]["text"] = "modified preview"
            self.assertEqual(agent._states, {})
            self.assertEqual(backend.requests, [])
            self.assertFalse(parent.exists())
            self.assertNotIn("modified preview", str(agent.build_request_messages(value)))
            agent.forward(value)
            self.assertNotIn("modified preview", text_of(backend.requests[0]))

    def test_task_prefix_is_pinned_but_reset_observation_and_images_are_not(self):
        image = encode_png(b"\xff\0\0", 1, 1)
        agent = self.agent()
        agent.forward(obs("Click the red button.\n\nCurrent observation (step_msg):\nInitial state", image=image, system="API docs"))
        state = agent._states[0]
        pinned = "\n".join(part.text for message in state.pins for part in message.parts if isinstance(part, TextPart))
        self.assertIn("Click the red button.", pinned)
        self.assertNotIn("Initial state", pinned)
        self.assertFalse(any(isinstance(part, ImagePart) for message in state.pins for part in message.parts))
        self.assertIn("Initial state", text_of(state.model.backend.requests[0]))

    def test_rejects_full_mode_raw_fallback_host_calls_and_external_image_urls(self):
        with self.assertRaisesRegex(ValueError, "delta"):
            self.agent(env_message_mode="full")
        agent = self.agent()
        for value in (
            {0: {"step_msg": "raw hidden observation", "vis": []}},
            {0: {"messages": []}},
            {0: {"messages_return_mode": "full", "messages": [{"role": "user", "content": "full"}]}},
            {0: {"messages": [{"role": "assistant", "content": "", "tool_calls": [{}]}]}},
            {0: {"messages": [{"role": "tool", "content": "fake result"}]}},
            {0: {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///secret.png"}}]}]}},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                agent.forward(value)

    def test_skip_does_not_create_state_or_call_model(self):
        backend = FakeModel()
        agent = self.agent(backend)
        value = {0: {"skip_infer": True}}
        self.assertEqual(agent.build_request_messages(value), {0: None})
        response, trace = agent.forward_with_trace(value)
        self.assertEqual(response[0], {"action": None, "assistant": None})
        self.assertTrue(trace[0]["skipped"])
        self.assertEqual(agent._states, {})
        self.assertEqual(backend.requests, [])

    def test_code_and_legacy_extraction_are_unchanged(self):
        backend = FakeModel([ModelCompletion("<code>public sealed class AgentAction : CodeAction {}</code>"),
                             ModelCompletion("<params>{\"x\":2}</params>")])
        agent = self.agent(backend)
        self.assertEqual(agent.forward(obs())[0], "public sealed class AgentAction : CodeAction {}")
        self.assertEqual(agent.forward(obs("new state"))[0], '<params>{"x":2}</params>')

    def test_same_loop_across_steps_and_reset_and_idempotent_close(self):
        backend = FakeModel()
        agent = self.agent(backend)
        agent.reset(obs())
        agent.forward(obs("case A"))
        first_scope = agent._states[0].runtime.history_snapshot[0].identity.session_id
        agent.forward(obs("case A next"))
        agent.reset()
        agent.forward(obs("case B"))
        self.assertEqual(len(set(backend.loops)), 1)
        self.assertNotEqual(first_scope, agent._states[0].runtime.history_snapshot[0].identity.session_id)
        self.assertNotIn("case A", text_of(backend.requests[-1]))
        thread = agent._loop.thread
        agent.close()
        agent.close()
        self.assertFalse(thread.is_alive())
        self.assertTrue(backend.loops[0].is_closed())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            agent.forward(obs())

    def test_visible_attempt_auto_reset_carries_history_without_hidden_state(self):
        backend = FakeModel()
        agent = self.agent(backend)
        agent.forward(obs("first attempt facts"))
        value = obs("Attempt 1 ended. Auto-reset started attempt 2.\nnew observations")
        value[0]["attempt_seed"] = "PRIVATE-SEED"
        value[0]["task_runtime"] = "PRIVATE-RUNTIME"
        _, trace = agent.forward_with_trace(value)
        self.assertEqual(trace[0]["simple_agent"]["attempt"], 2)
        self.assertIn("first attempt facts", text_of(backend.requests[-1]))
        self.assertNotIn("PRIVATE", text_of(backend.requests[-1]))
        self.assertEqual(agent._states[0].runtime.history_snapshot[-2].identity.episode_id, "attempt-2")

    def test_compaction_trace_contains_actual_summary_and_action_requests_and_usage(self):
        backend = FakeModel()
        agent = self.agent(backend, simple_agent_config={"context": {
            "max_context_characters": 2100, "recent_turns": 1,
            "output_reserve_tokens": 64, "max_context_tokens": 10000}})
        found = None
        for turn in range(7):
            _, trace = agent.forward_with_trace(obs("observation " + str(turn) + " " + "x" * 480))
            if trace[0]["agent_usage"]["summary_calls"]:
                found = trace[0]
                break
        self.assertIsNotNone(found)
        self.assertEqual([item["purpose"] for item in found["model_requests"]], ["summary", "action"])
        self.assertEqual(found["agent_usage"]["action"]["total_tokens"], 18)
        self.assertEqual(found["agent_usage"]["summary"]["total_tokens"], 5)
        self.assertEqual(found["usage"]["total_tokens"], 23)
        self.assertTrue(any(event["kind"] == "summary_completion" for event in found["agent_events"]))
        self.assertIn("Model notes; not environment facts", str(found["model_requests"][-1]["request_messages"]))

    def test_tools_use_filtered_immutable_own_snapshots_and_keep_host_dialogue(self):
        image = encode_png(b"\xff\0\0", 1, 1)
        backend = FakeModel([
            ModelCompletion(ACTION),
            ModelCompletion("", ModelUsage(1, 2, 3), (ToolCall("r", "read_file", json.dumps({
                "path": "records/turn-000001/messages.json", "start_line": 1, "max_lines": 20})),)),
            ModelCompletion("", ModelUsage(2, 3, 5), (ToolCall("v", "view_image", json.dumps({
                "path": "records/turn-000001/images/0000.png"})),)),
            ModelCompletion(ACTION, ModelUsage(3, 4, 7)),
            ModelCompletion(ACTION),
        ])
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(backend, simple_agent_config={"tool_choice": "auto", "tools": {"enabled": True, "workspace_parent": directory}})
            initial = obs("Visible initial state", image=image)
            initial[0].update({"seed": "HIDDEN-SEED", "runtime": {"solution": "HIDDEN-SOLUTION"}, "raw_logs": "HIDDEN-LOG"})
            _, first_trace = agent.forward_with_trace(initial)
            fingerprint = first_trace[0]["model_requests"][0]["request_images"][0]
            self.assertEqual(fingerprint["sha256"], sha256(image).hexdigest())
            self.assertEqual(fingerprint["bytes"], len(image))
            snapshot_event = first_trace[0]["agent_events"][-1]
            self.assertEqual(snapshot_event["status"], "published")
            _, trace = agent.forward_with_trace(obs("result"))
            self.assertEqual(len(trace[0]["model_requests"]), 3)
            self.assertEqual(trace[0]["agent_usage"]["action_calls"], 3)
            self.assertEqual(trace[0]["usage"]["total_tokens"], 15)
            self.assertTrue(any(isinstance(part, ImagePart) for message in backend.requests[-1].messages for part in message.parts))
            dialogue = str(trace[0]["model_requests"])
            self.assertIn("Visible initial state", dialogue)
            self.assertNotIn("HIDDEN-", dialogue)
            agent.forward(obs("later"))
            self.assertTrue(any(message.role == "tool" for message in backend.requests[-1].messages))
            # Physical notes namespace and record lifetime disappear on reset.
            private_directory = Path(agent._states[0].directory.name)
            agent.reset()
            self.assertFalse(private_directory.exists())

    def test_snapshot_capacity_fails_closed_before_returning_an_environment_action(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(simple_agent_config={"tools": {"enabled": True, "workspace_parent": directory,
                "limits": {"max_records": 1}}})
            agent.forward(obs("first"))
            with self.assertRaisesRegex(AgentLimitError, "snapshot capacity"):
                agent.forward_with_trace(obs("second"))
            trace = agent.last_trace_by_agent
            self.assertIsNone(trace[0]["action_extracted"])
            self.assertEqual(trace[0]["agent_events"][-1]["status"], "capacity_exhausted")
            self.assertEqual(trace[0]["usage"]["total_tokens"], 18)
            self.assertEqual(len(agent._states[0].workspace._records), 1)

    def test_default_snapshot_budget_supports_more_than_twenty_visible_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(simple_agent_config={"tools": {"enabled": True, "workspace_parent": directory}})
            self.assertEqual(agent.tool_limits.max_records, 512)
            for turn in range(21):
                response, trace = agent.forward_with_trace(obs("visible result " + str(turn)))
                self.assertEqual(response[0]["action"], ACTION)
                self.assertEqual(trace[0]["agent_events"][-1]["status"], "published")
            self.assertEqual(len(agent._states[0].workspace._records), 21)

    def test_parallel_agent_ids_and_fresh_cases_cannot_read_each_others_records(self):
        backend = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(backend, simple_agent_config={"tool_choice": "auto", "tools": {"enabled": True, "workspace_parent": directory}})
            agent.forward({0: obs("agent ZERO")[0], 1: obs("agent ONE")[0]})
            self.assertNotEqual(agent._states[0].directory.name, agent._states[1].directory.name)
            backend.replies = [ModelCompletion("", tool_calls=(ToolCall("r", "read_file", json.dumps({
                "path": "records/turn-000001/messages.json", "start_line": 1, "max_lines": 20})),)), ModelCompletion(ACTION)]
            agent.forward(obs("ZERO next"))
            self.assertIn("agent ZERO", text_of(backend.requests[-1]))
            self.assertNotIn("agent ONE", text_of(backend.requests[-1]))
            agent.reset()
            backend.replies = [ModelCompletion("", tool_calls=(ToolCall("r2", "read_file", json.dumps({
                "path": "records/turn-000001/messages.json", "start_line": 1, "max_lines": 20})),)), ModelCompletion(ACTION)]
            agent.forward(obs("fresh case"))
            self.assertNotIn("agent ZERO", text_of(backend.requests[-1]))
            self.assertNotIn("agent ONE", text_of(backend.requests[-1]))

    def test_timeout_settles_model_cancellation_and_retains_actual_failed_trace(self):
        class BlockingModel(FakeModel):
            cancelled = False

            async def complete(self, request):
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        backend = BlockingModel()
        agent = self.agent(backend, timeout_s=1, simple_agent_config={"context": {"timeout_seconds": 0.02}})
        started = time.monotonic()
        with self.assertRaises(AgentLimitError):
            agent.forward(obs())
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(backend.cancelled)
        trace = agent.last_trace_by_agent[0]
        self.assertEqual(trace["model_requests"][0]["status"], "cancelled")
        thread = agent._loop.thread
        agent.close()
        self.assertFalse(thread.is_alive())

    def test_default_async_openai_backend_parameters_tokens_reasoning_and_cleanup(self):
        clients = []

        class Client:
            def __init__(self, **kwargs):
                self.kwargs, self.requests, self.closed, self.loops = kwargs, [], False, []
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
                clients.append(self)

            async def create(self, **kwargs):
                self.requests.append(kwargs)
                self.loops.append(asyncio.get_running_loop())
                return {"choices": [{"finish_reason": "stop", "message": {"content": ACTION, "reasoning": "safe reasoning"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}

            async def close(self):
                self.closed = True
                self.loops.append(asyncio.get_running_loop())

        with patch("openai.AsyncOpenAI", Client):
            agent = SimpleAgent(api_key="do-not-log-this", base_url="https://openrouter.ai/api/v1", provider="openrouter",
                model="fake-model", thinking=True, reasoning_effort="high", max_completion_tokens=20000,
                simple_agent_config={"output_token_parameter": "max_tokens", "context": {"output_reserve_tokens": 123}})
            self.addCleanup(agent.close)
            _, trace = agent.forward_with_trace(obs())
            agent.forward(obs("next"))
            agent.close()
        self.assertEqual(len(clients), 1)
        client = clients[0]
        self.assertEqual(client.requests[0]["max_tokens"], 123)
        self.assertNotIn("max_completion_tokens", client.requests[0])
        self.assertEqual(client.requests[0]["extra_body"], {"reasoning": {"enabled": True, "effort": "high"}})
        self.assertNotIn("reasoning_effort", client.requests[0])
        self.assertEqual(client.kwargs["max_retries"], 0)
        self.assertEqual(len(set(client.loops)), 1)
        self.assertTrue(client.closed)
        self.assertNotIn("do-not-log-this", str(trace))
        self.assertEqual(trace[0]["model_requests"][0]["response_metadata"]["usage"]["total_tokens"], 7)

    def test_image_only_task_text_and_original_snapshots_survive_prompt_pinning(self):
        image = encode_png(b"\0\xff\0", 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(simple_agent_config={"tools": {"enabled": True, "workspace_parent": directory}})
            initial_text = "TASK: operate only the GREEN button."
            agent.forward(obs(initial_text, image=image, system="env API"))
            state = agent._states[0]
            self.assertIn(initial_text, "\n".join(part.text for message in state.pins for part in message.parts if isinstance(part, TextPart)))
            self.assertFalse(any(isinstance(part, ImagePart) for message in state.pins for part in message.parts))
            snapshot = state.workspace._records["turn-000001"][0]
            data = dict(snapshot.files)["messages.json"].decode("utf-8")
            self.assertIn(initial_text, data)
            self.assertIn("env API", data)
            self.assertNotIn("Host function tool calls", data)

    def test_original_long_message_snapshot_is_exact_and_line_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(simple_agent_config={"tools": {"enabled": True, "workspace_parent": directory}})
            original = "original task instructions\nCurrent observation (step_msg):\n" + "α\"" * 6000
            agent.forward(obs(original))
            data = dict(agent._states[0].workspace._records["turn-000001"][0].files)["messages.json"]
            record = json.loads(data)
            part = record["messages"][0]["content"][0]
            self.assertEqual("".join(part["text_chunks"]), original)
            self.assertLess(max(len(line) for line in data.splitlines()), agent.tool_limits.max_text_bytes)

    def test_close_then_explicit_reset_reopens_new_case_without_old_client(self):
        backend = FakeModel()
        agent = self.agent(backend)
        agent.forward(obs("first case"))
        old_loop = backend.loops[0]
        agent.close()
        agent.reset()
        agent.forward(obs("second case"))
        self.assertIsNot(old_loop, backend.loops[-1])
        self.assertNotIn("first case", text_of(backend.requests[-1]))
        self.assertTrue(old_loop.is_closed())

    def test_custom_trusted_tool_setup_is_fresh_for_each_case(self):
        registries = []

        def setup(registry):
            registries.append(registry)
            registry.register(ToolSpec("memory_hint", "Get a caller-provided visible memory hint.",
                                      '{"type":"object","properties":{},"additionalProperties":false}'),
                              lambda args, session: (TextPart("Caller-provided public hint"),))

        backend = FakeModel([ModelCompletion("", tool_calls=(ToolCall("hint", "memory_hint", "{}"),)), ModelCompletion(ACTION)])
        agent = self.agent(backend, tool_setup=setup, simple_agent_config={"tool_choice": "auto"})
        _, trace = agent.forward_with_trace(obs())
        self.assertIn("Caller-provided public hint", text_of(backend.requests[-1]))
        self.assertTrue(trace[0]["simple_agent"]["custom_tools"])
        agent.reset()
        backend.replies = [ModelCompletion("", tool_calls=(ToolCall("hint2", "memory_hint", "{}"),)), ModelCompletion(ACTION)]
        agent.forward(obs())
        self.assertEqual(len(registries), 2)
        self.assertIsNot(registries[0], registries[1])

    def test_failed_provider_completion_retains_billed_usage_and_raw_answer(self):
        class Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            async def create(self, **kwargs):
                return {"choices": [{"finish_reason": "length", "message": {"content": "partial answer"}}],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 60, "total_tokens": 110}}

            async def close(self):
                pass

        with patch("openai.AsyncOpenAI", Client):
            agent = SimpleAgent(api_key="fake", provider="openai")
            self.addCleanup(agent.close)
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                agent.forward(obs())
            trace = agent.last_trace_by_agent[0]
            self.assertEqual(trace["usage"]["total_tokens"], 110)
            self.assertEqual(trace["agent_usage"]["action_calls"], 1)
            record = trace["model_requests"][0]
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["response_metadata"]["assistant_raw"], "partial answer")
            self.assertIsNone(trace["action_extracted"])

    def test_provider_tool_finish_without_tool_calls_retains_usage_and_executes_nothing(self):
        class Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            async def create(self, **kwargs):
                # Anonymized shape observed from an OpenRouter-compatible
                # provider: reasoning announces a tool, but no call exists.
                return {"choices": [{"finish_reason": "tool_calls", "message": {
                    "content": None, "tool_calls": None, "reasoning": "start list_files"}}],
                    "usage": {"prompt_tokens": 2926, "completion_tokens": 92, "total_tokens": 3018}}

            async def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, patch("openai.AsyncOpenAI", Client), \
                patch("agent_ark.agent.simple_agent.agent.FileWorkspace.execute") as execute:
            agent = SimpleAgent(api_key="fake", provider="openrouter", simple_agent_config={
                "tools": {"enabled": True, "workspace_parent": directory}})
            self.addCleanup(agent.close)
            with self.assertRaisesRegex(RuntimeError, "no tool calls"):
                agent.forward_with_trace(obs("visible observation"))
            execute.assert_not_called()
            trace = agent.last_trace_by_agent[0]
            self.assertIsNone(trace["assistant_raw"])
            self.assertIsNone(trace["action_extracted"])
            self.assertEqual(trace["usage"], {"input_tokens": 2926, "output_tokens": 92, "total_tokens": 3018})
            self.assertEqual(len(trace["model_requests"]), 1)
            request = trace["model_requests"][0]
            self.assertEqual(request["status"], "failed")
            self.assertEqual(request["purpose"], "action")
            self.assertTrue(any(tool["function"]["name"] == "list_files" for tool in request["tools"]))
            self.assertEqual(request["response_metadata"]["finish_reason"], "tool_calls")
            self.assertEqual(request["response_metadata"]["assistant_raw"], "start list_files")
            self.assertEqual(trace["agent_events"], [])
            self.assertEqual([entry.kind for entry in agent._states[0].runtime.history_snapshot], ["input"])

    def test_summary_events_and_usage_remain_when_later_action_fails(self):
        class Model(FakeModel):
            fail_action = False

            async def complete(self, request):
                if self.fail_action and request.purpose == "action":
                    self.last_response_metadata = {"usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}}
                    raise RuntimeError("provider parser failure")
                return await super().complete(request)

        backend = Model()
        agent = self.agent(backend, simple_agent_config={"context": {"max_context_characters": 2250,
            "recent_turns": 1, "output_reserve_tokens": 64, "max_context_tokens": 10000}})
        for turn in range(2):
            agent.forward(obs("Current observation (step_msg):" + str(turn) + "x" * 480))
        backend.fail_action = True
        with self.assertRaises(RuntimeError):
            agent.forward(obs("Current observation (step_msg):final" + "x" * 480))
        trace = agent.last_trace_by_agent[0]
        self.assertEqual([call["purpose"] for call in trace["model_requests"]], ["summary", "action"])
        self.assertEqual(trace["usage"]["total_tokens"], 55)
        self.assertTrue(any(event["kind"] == "summary_completion" for event in trace["agent_events"]))

    def test_native_submission_runs_host_tools_then_two_decisions_with_native_history(self):
        payload = json.dumps({"name": "tap", "arguments": {"x": 3}})
        first_rationale = "Tap the visible button to advance the task."
        second_rationale = "Tap again after checking the earlier observation."
        backend = FakeModel([
            ModelCompletion("Need my own files.", tool_calls=(ToolCall("list", "list_files", json.dumps({
                "path": "records", "offset": 0, "limit": 10})),)),
            submission("tool_call", payload, call_id="submit1", text=first_rationale),
            ModelCompletion("Check the earlier visible observation.", tool_calls=(ToolCall("read", "read_file", json.dumps({
                "path": "records/turn-000001/messages.json", "start_line": 1, "max_lines": 20})),)),
            submission("tool_call", payload, call_id="submit2", text=second_rationale),
        ])
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(backend, simple_agent_config={"tools": {"enabled": True, "workspace_parent": directory}})
            with patch("agent_ark.agent.simple_agent.agent.FileWorkspace.execute", wraps=agent._state(0).workspace.execute) as execute:
                legacy_format = "First output <think>THINK_CONTENT</think>, then <tool_call> JSON </tool_call>."
                response, trace = agent.forward_with_trace(obs("initial visible state", system=legacy_format))
                self.assertEqual(response[0], {"action": ACTION, "assistant": ACTION})
                self.assertEqual(execute.call_count, 1)
                self.assertEqual(trace[0]["assistant_raw"], first_rationale)
                self.assertEqual(trace[0]["model_requests"][-1]["tool_calls"][0]["name"], "submit_action")
                event = next(event for event in trace[0]["agent_events"] if event["kind"] == "action_translation")
                self.assertEqual(event, {"kind": "action_translation", "call_id": "submit1", "format": "tool_call",
                    "translated_action": ACTION, "source": "simple_agent_adapter"})
                messages = [{"role": "assistant", "content": ACTION}, {"role": "user", "content": "Actual execution result: button moved."}]
                agent.forward({0: {"messages": messages}})
                self.assertEqual(execute.call_count, 2)
            request = backend.requests[2]
            self.assertEqual(request.messages[0].role, "system")
            self.assertIn("native action transport protocol", text_of(request).lower())
            transport = "".join(part.text for part in request.messages[0].parts)
            self.assertIn("their examples do not change this native submission protocol", transport)
            for tag in ("<think>", "</think>", "<rationale>", "</rationale>"):
                self.assertNotIn(tag, transport)
            self.assertIn("native model/backend reasoning when available", transport)
            self.assertIn("hidden or absent", transport)
            self.assertIn("Literal text required inside a task payload remains allowed", transport)
            self.assertIn("exactly one standalone native submit_action", transport)
            self.assertIn("user-facing action rationale in the assistant message's content", transport)
            self.assertIn("argument holds only the task action payload, never the rationale", transport)
            self.assertTrue(any(message.role == "developer" and legacy_format in "".join(
                part.text for part in message.parts) for message in request.messages))
            self.assertEqual(sum(message.role == "system" for message in request.messages), 1)
            self.assertTrue(any(message.role == "assistant" and message.tool_calls and message.tool_calls[0].name == "submit_action"
                                for message in request.messages))
            previous_submission = next(message for message in request.messages if message.tool_calls
                                       and message.tool_calls[0].call_id == "submit1")
            self.assertEqual(previous_submission.parts, (TextPart(first_rationale),))
            assistant_prose = "\n".join(part.text for message in request.messages if message.role == "assistant"
                                         for part in message.parts if isinstance(part, TextPart))
            self.assertNotIn(ACTION, assistant_prose)
            self.assertIn("Synthetic adapter acknowledgement", text_of(request))
            self.assertIn("not executed yet", text_of(request))
            self.assertIn("Actual execution result", text_of(request))
            final_entry = agent._states[0].runtime.history_snapshot[-1]
            self.assertEqual(final_entry.text, ACTION)
            self.assertEqual(len(final_entry.messages), 2)
            self.assertEqual(final_entry.messages[0].tool_calls[0].name, "submit_action")
            self.assertEqual(final_entry.messages[0].parts, (TextPart(second_rationale),))

    def test_native_full_csharp_source_unicode_and_all_newlines_are_exact(self):
        source = "\nusing System;\npublic sealed class AgentAction : CodeAction\n{\n" + "\n".join(
            f'    // full source line {index}: 中文 αβ' for index in range(350)) + '\n    string label = "\\u003ccode\\u003e";\n}\n\n'
        backend = FakeModel([submission("code", source)])
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(backend, simple_agent_config={"tools": {"enabled": True, "workspace_parent": directory}})
            response, trace = agent.forward_with_trace(obs("run a full C# action"))
            self.assertEqual(response[0]["action"], source)
            self.assertEqual(response[0]["assistant"], source)
            event = next(event for event in trace[0]["agent_events"] if event["kind"] == "action_translation")
            self.assertEqual(event["translated_action"], source)
            self.assertEqual(trace[0]["assistant_raw"], "")
            native_argument = trace[0]["model_requests"][0]["tool_calls"][0]["arguments_json"]
            self.assertEqual(json.loads(native_argument)["content"], source)

    def test_native_unicode_function_values_restore_exactly_with_safe_json_delimiters(self):
        payload = {"name": "submit_label", "arguments": {"label": "中文\n<tool_call>literal</tool_call>", "amount": 1.25}}
        backend = FakeModel([submission("tool_call", json.dumps(payload, ensure_ascii=False))])
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(backend, simple_agent_config={"tools": {"enabled": True, "workspace_parent": directory}})
            action = agent.forward(obs())[0]
            self.assertEqual(action.count("<tool_call>"), 1)
            self.assertEqual(action.count("</tool_call>"), 1)
            self.assertEqual(json.loads(action[len("<tool_call>"):-len("</tool_call>")]), payload)

    def test_native_params_preserves_legacy_extraction_and_raw_has_no_envelope_bypass(self):
        backend = FakeModel([submission("params", "  legacy parameter text  "), submission("raw", '{"plan":"R7"}')])
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(backend, simple_agent_config={"tools": {"enabled": True, "workspace_parent": directory}})
            self.assertEqual(agent.forward(obs())[0], "<params>legacy parameter text</params>")
            self.assertEqual(agent.forward(obs("next"))[0], '{"plan":"R7"}')

    def test_submit_action_name_is_reserved_before_trusted_tool_setup(self):
        setup_names = []

        def setup(registry):
            setup_names.extend(spec.name for spec in registry.specs)
            registry.register(SUBMISSION_SPEC, lambda args, session: (TextPart("should never execute"),))

        backend = FakeModel()
        agent = self.agent(backend, tool_setup=setup)
        with self.assertRaises((ValueError, RuntimeError)):
            agent.forward(obs())
        self.assertIn("submit_action", setup_names)
        self.assertEqual(backend.requests, [])

    def test_native_submission_rejects_mixed_duplicate_and_malformed_calls_without_side_effects(self):
        invalid_arguments = [
            '{"format":"raw","content":"a","extra":1}',
            '{"format":"raw","format":"code","content":"a"}',
            '{"format":"tool_call","content":"{bad JSON}"}',
            json.dumps({"format": "tool_call", "content": '{"name":"f","arguments":{"x":NaN}}'}),
            json.dumps({"format": "tool_call", "content": '{"name":"f","arguments":{"x":1e400}}'}),
            json.dumps({"format": "tool_call", "content": '{"name":"f","arguments":{"x":1,"x":2}}'}),
            json.dumps({"format": "tool_call", "content": '{"name":"f","arguments":{},"extra":1}'}),
            json.dumps({"format": "tool_call", "content": '{"name":" ","arguments":{}}'}),
            json.dumps({"format": "tool_call", "content": '{"name":"f","arguments":[]}'}),
            json.dumps({"format": "raw", "content": "<code>nested</code>"}),
            json.dumps({"format": "params", "content": "<tool_call>nested</tool_call>"}),
            json.dumps({"format": "code", "content": 'string x = "</code>";'}),
            json.dumps({"format": "code", "content": "```csharp\nsource\n```"}),
            json.dumps({"format": "unknown", "content": "a"}),
            json.dumps({"format": "raw", "content": []}),
            json.dumps({"format": "raw", "content": ""}),
        ]
        malformed = [ModelCompletion("", ModelUsage(10, 20, 30), (ToolCall("s", "submit_action", arguments),))
                     for arguments in invalid_arguments]
        valid = ToolCall("s", "submit_action", '{"format":"raw","content":"a"}')
        malformed.extend([
            ModelCompletion("", ModelUsage(10, 20, 30), (valid, ToolCall("host", "write_note",
                '{"path":"notes/forbidden.txt","content":"SIDE EFFECT"}'))),
            ModelCompletion("", ModelUsage(10, 20, 30), (valid, ToolCall("s2", "submit_action", valid.arguments_json))),
        ])
        for completion in malformed:
            with self.subTest(arguments=completion.tool_calls), tempfile.TemporaryDirectory() as directory:
                backend = FakeModel([completion])
                agent = self.agent(backend, simple_agent_config={"tools": {"enabled": True, "workspace_parent": directory}})
                with patch("agent_ark.agent.simple_agent.agent.FileWorkspace.execute") as execute:
                    with self.assertRaisesRegex(RuntimeError, "submit_action"):
                        agent.forward(obs())
                    execute.assert_not_called()
                self.assertEqual(agent.last_trace_by_agent[0]["usage"]["total_tokens"], 30)
                self.assertIsNone(agent.last_trace_by_agent[0]["action_extracted"])
                self.assertEqual(agent.last_trace_by_agent[0]["model_requests"][0]["status"], "failed")
                agent.close()

    def test_native_decoded_action_limit_rejects_instead_of_truncating(self):
        backend = FakeModel([submission("code", "x" * 100)])
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(backend, simple_agent_config={"context": {"max_action_output_bytes": 64},
                "tools": {"enabled": True, "workspace_parent": directory}})
            with self.assertRaisesRegex(RuntimeError, "decoded action exceeds"):
                agent.forward(obs())
            self.assertEqual(json.loads(agent.last_trace_by_agent[0]["model_requests"][0]["tool_calls"][0]["arguments_json"])["content"], "x" * 100)
            self.assertIsNone(agent.last_trace_by_agent[0]["action_extracted"])

    def test_submission_is_declared_and_budgeted_only_on_action_requests(self):
        from agent_ark.agent.simple_agent.encoding import tool_declarations
        from agent_ark.agent.simple_agent.tools import FileWorkspace

        backend = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            tool_chars = len(json.dumps(tool_declarations((*FileWorkspace(directory).specs, SUBMISSION_SPEC)), ensure_ascii=False))
            # Reserve a fixed message allowance in addition to tool help. The
            # declarations are still budgeted and repeated inputs must compact.
            agent = self.agent(backend, simple_agent_config={"context": {"max_context_characters": tool_chars + 4000,
                "recent_turns": 1, "output_reserve_tokens": 64, "max_context_tokens": 20000},
                "tools": {"enabled": True, "workspace_parent": directory}})
            found = False
            for turn in range(10):
                backend.replies.append(submission("raw", "plain action " + str(turn)))
                _, trace = agent.forward_with_trace(obs("Current observation (step_msg):" + "x" * 800))
                if trace[0]["agent_usage"]["summary_calls"]:
                    found = True
                    break
            self.assertTrue(found)
            for request in backend.requests:
                if request.purpose == "summary":
                    self.assertEqual(request.tools, ())
                    self.assertFalse(any(message.tool_calls or message.role == "tool" for message in request.messages))
                else:
                    self.assertIn("submit_action", [tool.name for tool in request.tools])
                    self.assertIsNotNone(request.input_tokens)
                    self.assertGreater(request.input_tokens, 0)

    def test_default_required_policy_is_in_every_actual_native_action_request(self):
        clients = []

        class Client:
            def __init__(self, **kwargs):
                self.requests = []
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
                clients.append(self)

            async def create(self, **kwargs):
                self.requests.append(kwargs)
                if len(self.requests) <= 2:
                    name, arguments = "list_files", '{"path":"records","offset":0,"limit":10}'
                else:
                    name = "submit_action"
                    arguments = json.dumps({"format": "tool_call", "content": '{"name":"tap","arguments":{"x":3}}'})
                return {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{
                    "id": "call-" + str(len(self.requests)), "type": "function", "function": {"name": name, "arguments": arguments}}]}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}

            async def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, patch("openai.AsyncOpenAI", Client):
            agent = SimpleAgent(api_key="fake", provider="openai", simple_agent_config={
                "tools": {"enabled": True, "workspace_parent": directory}})
            self.addCleanup(agent.close)
            response, trace = agent.forward_with_trace(obs())
            self.assertEqual(response[0]["action"], ACTION)
            self.assertEqual(agent.tool_choice, "required")
            self.assertEqual(trace[0]["simple_agent"]["tool_choice"], "required")
            self.assertEqual(len(clients[0].requests), 3)
            for kwargs in clients[0].requests:
                self.assertEqual(kwargs["tool_choice"], "required")
                self.assertIn("submit_action", [tool["function"]["name"] for tool in kwargs["tools"]])
            self.assertEqual([request["parameters"]["tool_choice"] for request in trace[0]["model_requests"]], ["required"] * 3)
            self.assertEqual(trace[0]["usage"]["total_tokens"], 15)

    def test_required_rejects_plaintext_without_guessing_or_retrying_and_auto_preserves_it(self):
        class Client:
            requests = []

            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            async def create(self, **kwargs):
                self.requests.append(kwargs)
                return {"choices": [{"finish_reason": "stop", "message": {"content": ACTION, "tool_calls": None}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}

            async def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, patch("openai.AsyncOpenAI", Client), \
                patch("agent_ark.agent.simple_agent.agent.FileWorkspace.execute") as execute:
            required = SimpleAgent(api_key="fake", provider="openai", simple_agent_config={
                "tools": {"enabled": True, "workspace_parent": directory}})
            self.addCleanup(required.close)
            with self.assertRaisesRegex(RuntimeError, "tool_choice=required returned no native"):
                required.forward(obs())
            self.assertEqual(len(Client.requests), 1)
            self.assertEqual(required.last_trace_by_agent[0]["usage"]["total_tokens"], 30)
            self.assertIsNone(required.last_trace_by_agent[0]["action_extracted"])
            self.assertEqual(required.last_trace_by_agent[0]["model_requests"][0]["completion"], ACTION)
            self.assertEqual(required.last_trace_by_agent[0]["model_requests"][0]["status"], "failed")
            auto = SimpleAgent(api_key="fake", provider="openai", simple_agent_config={"tool_choice": "auto",
                "tools": {"enabled": True, "workspace_parent": directory}})
            self.addCleanup(auto.close)
            response, trace = auto.forward_with_trace(obs())
            self.assertEqual(response[0]["action"], ACTION)
            self.assertEqual(Client.requests[-1]["tool_choice"], "auto")
            self.assertEqual(trace[0]["model_requests"][0]["parameters"]["tool_choice"], "auto")
            execute.assert_not_called()

    def test_tool_choice_validation_happens_before_any_request(self):
        for choice in ("none", "submit_action", "REQUIRED", None, 1, {}, []):
            with self.subTest(choice=choice), patch("openai.AsyncOpenAI") as client:
                with self.assertRaisesRegex(ValueError, "tool_choice"):
                    SimpleAgent(simple_agent_config={"tool_choice": choice})
                client.assert_not_called()

    def test_required_policy_also_applies_to_caller_owned_model_backend(self):
        backend = FakeModel([ModelCompletion(ACTION, ModelUsage(4, 6, 10))])
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(backend, simple_agent_config={
                "tools": {"enabled": True, "workspace_parent": directory}})
            with patch("agent_ark.agent.simple_agent.agent.FileWorkspace.execute") as execute:
                with self.assertRaisesRegex(RuntimeError, "tool_choice=required returned no native"):
                    agent.forward(obs())
                execute.assert_not_called()
            self.assertEqual(len(backend.requests), 1)
            self.assertEqual(agent.last_trace_by_agent[0]["usage"]["total_tokens"], 10)
            self.assertIsNone(agent.last_trace_by_agent[0]["action_extracted"])

    def test_tool_choice_is_omitted_on_summary_and_tools_disabled_requests(self):
        backend = FakeModel()
        agent = self.agent(backend)
        from agent_ark.agent.simple_agent.types import Message, ModelRequest
        action = ModelRequest((Message.text("user", "visible"),), purpose="action")
        summary = ModelRequest((Message.text("user", "earlier visible"),), purpose="summary")
        from agent_ark.agent.simple_agent.agent import _OpenAIModel
        adapter = _OpenAIModel(agent)
        self.assertNotIn("tool_choice", adapter.request_parameters(action))
        self.assertNotIn("tools", adapter.request_parameters(action))
        self.assertNotIn("tool_choice", adapter.request_parameters(summary))
        self.assertNotIn("tools", adapter.request_parameters(summary))
        agent.forward(obs())
        self.assertEqual(len(backend.requests), 1)


if __name__ == "__main__":
    unittest.main()
