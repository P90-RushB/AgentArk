"""Exercise installed LiteLLM against loopback HTTP only, without provider calls."""
from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from agent_ark.agent.simple_agent.agent import SimpleAgent


ACTION = '<tool_call>{"name":"tap","arguments":{"x":3}}</tool_call>'
PRIVATE = "SYNTHETIC_ENCRYPTED_ONLY+/="


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize("max_retries", [0, 1])
def test_real_sdk_http_retries_are_explicit_and_bounded(monkeypatch, api, max_retries):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    pytest.importorskip("litellm")
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            if len(received) == 1:
                status = 500
                payload = {"error": {"message": "Synthetic transient failure", "type": "server_error"}}
            elif api == "chat_completions":
                status = 200
                payload = {"id": "synthetic-chat", "object": "chat.completion", "created": 0, "model": "synthetic-model",
                           "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ACTION}}],
                           "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
            else:
                status = 200
                payload = {"id": "synthetic-response", "object": "response", "created_at": 0, "model": "synthetic-model",
                           "status": "completed", "output": [{"type": "message", "id": "synthetic-message", "role": "assistant",
                               "status": "completed", "content": [{"type": "output_text", "text": ACTION, "annotations": []}]}],
                           "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent = None
    try:
        agent = SimpleAgent(api_key="synthetic-local-key", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                            temperature=None, max_retries=max_retries, timeout_s=30,
                            simple_agent_config={"backend": "litellm", "litellm": {"model": "openai/synthetic-model", "api": api}})
        observation = {0: {"messages": [{"role": "user", "content": "Visible state"}]}}
        if max_retries:
            assert agent.forward(observation)[0] == ACTION
        else:
            with pytest.raises(Exception, match="(?i)(synthetic|server)"):
                agent.forward(observation)
        assert len(received) == max_retries + 1
        expected_path = "/v1/responses" if api == "responses" else "/v1/chat/completions"
        assert all(path == expected_path for path, body in received)
    finally:
        if agent is not None:
            agent.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_real_sdk_parallel_agent_close_preserves_other_inflight_client_and_history(monkeypatch, caplog, api):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    pytest.importorskip("litellm")
    received = []
    received_lock = threading.Lock()
    both_initial_requests = threading.Event()
    b_continuation_inflight = threading.Event()
    release_b_continuation = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            wire = json.dumps(body)
            owner = "A" if "AGENT_A_TURN_1" in wire else "B"
            continuation = "AGENT_B_TURN_2" in wire
            with received_lock:
                received.append((owner, self.path, body))
                if len(received) >= 2:
                    both_initial_requests.set()
            if continuation:
                b_continuation_inflight.set()
                ready = release_b_continuation.wait(timeout=3)
            else:
                # Both real SDK transports must reach the same server before
                # either first request completes, proving overlapping requests.
                ready = both_initial_requests.wait(timeout=3)
            if not ready:
                self.send_error(504, "Synthetic synchronization timeout")
                return
            turn = 2 if continuation else 1
            state = "SYNTHETIC_STATE_" + owner + "_" + str(turn)
            if api == "chat_completions":
                payload = {"id": "chat-" + state, "object": "chat.completion", "created": 0,
                           "model": "synthetic-model", "choices": [{"index": 0, "finish_reason": "stop",
                               "message": {"role": "assistant", "content": ACTION, "reasoning_content": state}}],
                           "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
            else:
                payload = {"id": "response-" + state, "object": "response", "created_at": 0,
                           "model": "synthetic-model", "status": "completed", "output": [
                               {"type": "reasoning", "id": "rs-" + state, "summary": [], "encrypted_content": state},
                               {"type": "message", "id": "msg-" + state, "role": "assistant", "status": "completed",
                                "phase": "final_answer", "content": [{"type": "output_text", "text": ACTION, "annotations": []}]}],
                           "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agents = []
    try:
        options = {"api_key": "synthetic-local-key", "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                   "model": "same-visible-model", "temperature": None, "max_retries": 0, "timeout_s": 5,
                   "simple_agent_config": {"backend": "litellm", "litellm": {
                       "model": "openai/synthetic-model", "api": api}}}
        a, b = SimpleAgent(**options), SimpleAgent(**options)
        agents.extend((a, b))

        def observation(text):
            return {0: {"messages": [{"role": "user", "content": text}]}}

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_a = executor.submit(a.forward, observation("AGENT_A_TURN_1"))
            first_b = executor.submit(b.forward, observation("AGENT_B_TURN_1"))
            assert first_a.result(timeout=10) == {0: ACTION}
            assert first_b.result(timeout=10) == {0: ACTION}
            a_loop, b_loop = a._loop, b._loop
            assert a_loop is not b_loop and a_loop.loop is not b_loop.loop
            continued_b = executor.submit(b.forward, observation("AGENT_B_TURN_2"))
            assert b_continuation_inflight.wait(timeout=3)
            # B is waiting on actual HTTP while A shuts down its private loop.
            # A must not close B's SDK client or cancel B's background tasks.
            a.close()
            assert not a_loop.thread.is_alive() and b_loop.thread.is_alive()
            release_b_continuation.set()
            assert continued_b.result(timeout=10) == {0: ACTION}

        assert len(received) == 3
        expected_path = "/v1/responses" if api == "responses" else "/v1/chat/completions"
        assert all(path == expected_path and body["model"] == "synthetic-model" for _, path, body in received)
        b_body = next(body for owner, _, body in received if owner == "B" and "AGENT_B_TURN_2" in json.dumps(body))
        b_wire = json.dumps(b_body)
        assert "AGENT_B_TURN_1" in b_wire and "SYNTHETIC_STATE_B_1" in b_wire
        assert "AGENT_A_TURN_1" not in b_wire and "SYNTHETIC_STATE_A_1" not in b_wire
    finally:
        release_b_continuation.set()
        both_initial_requests.set()
        for agent in agents:
            agent.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not any("Task was destroyed" in record.getMessage() for record in caplog.records)


def test_real_sdk_chat_host_continuation_keeps_thinking_and_provider_call_signature(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    pytest.importorskip("litellm")
    received, sent = [], []
    opaque_id = "synthetic_call__mpthought__" + "Signed+/=" * 30

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            first = len(received) == 1
            message = {"role": "assistant", "content": "Inspect visible records." if first else "Tap the visible point.",
                       "thinking_blocks": [{"type": "thinking", "thinking": PRIVATE, "signature": "synthetic-signed+/="}],
                       "tool_calls": [{"id": opaque_id if first else "synthetic_submit", "type": "function",
                           "function": {"name": "list_files" if first else "submit_action", "arguments": json.dumps(
                               {"path": "records", "offset": 0, "limit": 10} if first else
                               {"format": "tool_call", "content": ACTION[11:-12]})},
                           "provider_specific_fields": {"thought_signature": "synthetic-tool-signed+/="}}]}
            sent.append(deepcopy(message))
            payload = {"id": "synthetic-chat", "object": "chat.completion", "created": 0, "model": "synthetic-model",
                       "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent = None
    try:
        agent = SimpleAgent(api_key="synthetic-local-key", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                            temperature=None, max_retries=0, timeout_s=30,
                            simple_agent_config={"backend": "litellm", "litellm": {"model": "openai/synthetic-model"},
                                                 "tools": {"enabled": True, "workspace_parent": str(tmp_path)}})
        result, traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "Visible state"}]}})
        assert result[0]["action"] == ACTION
        assert len(received) == 2 and all(path == "/v1/chat/completions" for path, body in received)
        replayed = next(message for message in received[1][1]["messages"] if message.get("tool_calls"))
        # LiteLLM may add its own normalized provider_specific_fields. The
        # original state and provider signature must remain unchanged.
        assert all(replayed[name] == value for name, value in sent[0].items())
        assert next(message for message in received[1][1]["messages"] if message["role"] == "tool")["tool_call_id"] == opaque_id
        assert traces[0]["usage"]["total_tokens"] == 24
        assert PRIVATE not in result[0]["assistant"]
    finally:
        if agent is not None:
            agent.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("token_parameter", ["max_tokens", "max_completion_tokens"])
def test_real_sdk_responses_wire_preserves_output_order_phase_encryption_and_tool_call_ids(monkeypatch, tmp_path, token_parameter):
    # Loading the bundled cost map prevents an otherwise unnecessary hosted-map
    # lookup during SDK import. All model requests go to the loopback server.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    pytest.importorskip("litellm")
    received, outputs = [], []
    call_ids = ["synthetic_host_call+/=" + "signed_" * 50, "synthetic_submit_1", "synthetic_submit_2"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body))
            index = len(received) - 1
            name = "list_files" if index == 0 else "submit_action"
            arguments = {"path": "records", "offset": 0, "limit": 10} if index == 0 else {
                "format": "tool_call", "content": ACTION[11:-12]}
            # The commentary message deliberately precedes a second reasoning
            # item. Reordering by item type loses the provider's original turn.
            output = [
                {"type": "reasoning", "id": "rs-first-" + str(index), "summary": [],
                 "encrypted_content": PRIVATE + str(index)},
                {"type": "message", "id": "msg-" + str(index), "role": "assistant", "status": "completed",
                 "phase": "commentary", "content": [{"type": "output_text", "text": "Inspect visible records." if index == 0 else "Tap the visible point.", "annotations": []}]},
                {"type": "reasoning", "id": "rs-second-" + str(index), "summary": [],
                 "encrypted_content": PRIVATE + "second" + str(index)},
                {"type": "function_call", "id": "fc-" + str(index), "call_id": call_ids[index],
                 "name": name, "arguments": json.dumps(arguments), "status": "completed"},
            ]
            outputs.append(deepcopy(output))
            payload = {"id": "resp-" + str(index), "object": "response", "created_at": 0,
                       "model": "openai/gpt6", "status": "completed", "output": output,
                       "error": None, "incomplete_details": None,
                       "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12,
                                 "output_tokens_details": {"reasoning_tokens": 4}}}
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent = None
    try:
        agent = SimpleAgent(api_key="synthetic-local-key", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                            temperature=None, max_retries=0, timeout_s=30, max_completion_tokens=123,
                            simple_agent_config={"backend": "litellm", "litellm": {
                                "model": "openai/openai/gpt6", "api": "responses"},
                                "output_token_parameter": token_parameter,
                                "tools": {"enabled": True, "workspace_parent": str(tmp_path)}})
        result, traces = agent.forward_with_trace({0: {"messages": [{"role": "user", "content": "Visible state"}]}})
        assert result[0]["action"] == ACTION
        agent.forward({0: {"messages": [{"role": "user", "content": "Environment feedback: selected."}]}})
        assert len(received) == 3
        assert all(path == "/v1/responses" for path, body in received)
        first, continuation, next_step = [body for path, body in received]
        assert first["model"] == "openai/gpt6"
        assert first["store"] is False
        assert first["max_output_tokens"] == 123
        assert "max_tokens" not in first and "max_completion_tokens" not in first
        assert all(tool["type"] == "function" and tool["strict"] is False for tool in first["tools"])
        assert all("function" not in tool for tool in first["tools"])
        assert any(tool["name"] == "submit_action" for tool in first["tools"])
        offset = len(first["input"])
        assert continuation["input"][:offset] == first["input"]
        assert continuation["input"][offset:offset + len(outputs[0])] == outputs[0]
        result_item = continuation["input"][offset + len(outputs[0])]
        assert result_item["type"] == "function_call_output"
        assert result_item["call_id"] == call_ids[0]
        assert "tool_call_id" not in result_item
        # The full first and final submission outputs survive to the next
        # environment step, followed by the new, actually visible feedback.
        first_start = next(index for index, item in enumerate(next_step["input"]) if item.get("id") == "rs-first-0")
        second_start = next(index for index, item in enumerate(next_step["input"]) if item.get("id") == "rs-first-1")
        assert next_step["input"][first_start:first_start + len(outputs[0])] == outputs[0]
        assert next_step["input"][second_start:second_start + len(outputs[1])] == outputs[1]
        assert any(item.get("type") == "function_call_output" and item.get("call_id") == call_ids[1] for item in next_step["input"])
        assert "Environment feedback: selected." in json.dumps(next_step["input"][-1])
        assert traces[0]["usage"] == {"input_tokens": 10, "output_tokens": 14, "total_tokens": 24}
        assert PRIVATE not in result[0]["assistant"]
        assert "synthetic-local-key" not in json.dumps(traces)
    finally:
        if agent is not None:
            agent.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
