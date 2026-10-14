"""Compatibility with the separately developed SimpleAgent implementation."""
from types import SimpleNamespace

import pytest
from PIL import Image

from agent_ark.agent.simple_agent.agent import SimpleAgent
from agent_ark.agent.simple_agent.types import ImagePart
from agent_ark.ark_env.context_manager import MessageContext


@pytest.mark.parametrize("format_name", ["openai", "raw_base64"])
def test_simple_agent_receives_attempt_images_in_both_message_formats(format_name):
    observation = {"attempt_initial_images": [
        {"index": 1, "image": Image.new("RGB", (3, 2), "yellow"), "sha256": "first"},
        {"index": 2, "image": Image.new("RGB", (3, 2), "magenta"), "sha256": "second"},
    ]}
    context = MessageContext({"enabled": True, "max_images_per_section": 0,
                              "image_part_format": format_name})
    messages = context.build_chat_messages(observation, [], [])
    simple = object.__new__(SimpleAgent)
    simple.context_config = SimpleNamespace(max_archive_image_bytes=4096)
    parsed = simple._messages({"messages": messages})
    assert sum(isinstance(part, ImagePart) for message in parsed for part in message.parts) == 2
