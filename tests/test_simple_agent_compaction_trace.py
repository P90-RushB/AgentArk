"""Compaction provenance is audit data, never provider request parameters."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_ark.agent.simple_agent.agent import SimpleAgent, _TracedModel
from agent_ark.agent.simple_agent.models import ChatCompletionsModel
from agent_ark.agent.simple_agent.types import (
    AgentInputIdentity, ArchiveEntry, Message, ModelRequest,
)


def test_fragment_provenance_reaches_request_and_event_traces_without_entering_provider_wire():
    received = []

    async def create(**kwargs):
        received.append(kwargs)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Earlier facts retained."}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}

    metadata = {"version": 1, "policy": "message_chunks_v1", "stage": "leaf",
                "plan_id": "AUDIT_ONLY_PLAN", "fragments": [{"archive_sequence": 1,
                "message_index": 0, "part_index": 0, "kind": "text", "text_start": 0, "text_end": 4}]}
    encoded = json.dumps(metadata)
    backend = ChatCompletionsModel(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), "test-model")
    traced = _TracedModel(backend)
    completion = asyncio.run(traced.complete(ModelRequest(
        (Message.text("user", "Earlier visible history."),), purpose="summary", compaction_json=encoded)))
    entry = ArchiveEntry(2, "summary_completion", AgentInputIdentity("r", "s", "e", "a", 0, "i"),
                         text=completion.text, usage=completion.usage, compaction_json=encoded)
    state = SimpleNamespace(runtime=SimpleNamespace(history_snapshot=(entry,)), model=traced, turn=1, attempt=1)
    agent = SimpleAgent(provider="openai", model_backend=backend)
    try:
        trace = agent._trace(state, 0)
        assert trace["model_requests"][0]["compaction"] == metadata
        assert trace["agent_events"][0]["compaction"] == metadata
        assert trace["agent_usage"]["summary"] == {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12}
        assert "AUDIT_ONLY_PLAN" not in json.dumps(received)
        assert set(received[0]) == {"model", "messages", "max_completion_tokens"}
    finally:
        agent.close()


@pytest.mark.parametrize("metadata", ["[]", '{"bad":NaN}', '{"bad":1e999}', "{", '{"text":"' + "界" * 22000 + '"}'],
                         ids=["array", "nonfinite", "overflow", "malformed", "utf8_byte_limit"])
def test_compaction_audit_metadata_is_finite_bounded_json_object(metadata):
    with pytest.raises(ValueError, match="compaction metadata"):
        ModelRequest((Message.text("user", "history"),), purpose="summary", compaction_json=metadata)
