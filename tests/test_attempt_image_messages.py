"""Offline message and client contracts for optional attempt-initial images."""

import json
from copy import deepcopy
from hashlib import sha256
from unittest.mock import patch

import pytest
from PIL import Image

from agent_ark.agent.api_agent import APIAgent
from agent_ark.agent.codex_agent import CodexAgent, CodexRunOutput
from agent_ark.ark_env.ark_env import ArkRolloutContext
from agent_ark.ark_env.context_manager import ContextManager, MessageContext
from agent_ark.utils.image_utils import pil_image_to_base64


def _image(color):
    return Image.new("RGB", (3, 2), color)


def _entry(index, color):
    image = _image(color)
    return {"index": index, "image": image, "sha256": sha256(image.tobytes()).hexdigest()}


def _obs(*, extras=False, text="look"):
    obs = {"step_msg": text, "vis": [[_image("red"), _image("green")], [_image("blue")]]}
    if extras:
        obs["attempt_initial_images"] = [_entry(1, "yellow"), _entry(2, "magenta")]
    return obs


def _parts(messages):
    return [part for message in messages for part in message.get("content", [])
            if isinstance(message.get("content"), list) and isinstance(part, dict)]


def _urls(messages):
    return [part["image_url"]["url"] if part["type"] == "image_url"
            else "data:" + part["mime_type"] + ";base64," + part["image_base64"]
            for part in _parts(messages) if part.get("type") in ("image_url", "image")]


def _labels(messages):
    return "\n".join(part.get("text", "") for part in _parts(messages) if part.get("type") == "text")


def _data_url(image):
    return "data:image/png;base64," + pil_image_to_base64(image)


def _cfg(**message_cfg):
    return {"env_wrapper_cfg": {"context_manager": {
        "messages": {"enabled": True, "max_images_per_section": 1, **message_cfg},
        "history": {"max_history_attempts": 2, "max_history_steps_per_attempt": 8},
    }}}


def test_absent_feature_preserves_existing_initial_message_bytes_and_raw_fallback():
    obs = {"step_msg": "look", "vis": [[_image("blue")]]}
    expected_parts = [{"type": "text", "text": "Current observation (step_msg):\nlook"},
                      {"type": "image_url", "image_url": {"url": _data_url(obs["vis"][0][0])}}]
    context = MessageContext({"enabled": True, "max_images_per_section": 1})
    expected = [{"role": "system", "content": context.default_system_prompt},
                {"role": "user", "content": expected_parts}]
    for build in (context.build_chat_messages, context.build_messages):
        assert json.dumps(build(obs, [], [])) == json.dumps(expected)
    agent = object.__new__(APIAgent)
    assert agent._obs_to_messages(obs) == [{"role": "user", "content": [
        {"type": "text", "text": "look"}, expected_parts[1]]}]


@pytest.mark.parametrize("format_name", ["openai", "raw_base64"])
@pytest.mark.parametrize("camera_cap", [0, 1, 3])
def test_extras_follow_workspace_images_outside_camera_cap(format_name, camera_cap):
    obs = _obs(extras=True)
    original_vis = deepcopy(obs["vis"])
    context = MessageContext({"enabled": True, "max_images_per_section": camera_cap,
                              "image_part_format": format_name})
    messages = context.build_chat_messages(obs, [], [])
    urls = _urls(messages)
    assert len(urls) == camera_cap + 2
    assert urls[-2:] == [_data_url(item["image"]) for item in obs["attempt_initial_images"]]
    assert _labels(messages).count("Additional image 1") == 1
    assert _labels(messages).count("Additional image 2") == 1
    assert "Reference" not in _labels(messages)
    assert "painting" not in _labels(messages)
    assert [[image.tobytes() for image in camera] for camera in obs["vis"]] == [
        [image.tobytes() for image in camera] for camera in original_vis]
    assert len(obs["vis"]) == 2


@pytest.mark.parametrize("language,label", [("en", "Additional image 1"), ("zh", "附加图像1")])
def test_label_is_neutral_and_independent_of_optional_camera_captions(language, label):
    context = MessageContext({"image_caption": False, "image_caption_language": language})
    parts = context.attempt_initial_images_to_parts(_obs(extras=True))
    assert parts[0] == {"type": "text", "text": label}
    assert parts[1]["type"] == "image_url"


def test_image_opt_out_disables_additional_images_and_their_labels_in_every_renderer():
    context = MessageContext({"enabled": True, "include_images": False})
    obs = _obs(extras=True)
    step = {"obs": obs, "next_obs": obs, "reward": 0, "done": False, "action": "move"}
    outputs = [context.build_chat_messages(obs, [], []), context.build_messages(obs, [], []),
               context.build_step_messages(step), context.build_auto_reset_messages(step, obs),
               [{"role": "user", "content": context.build_step_delta_parts(step, 1)}]]
    for messages in outputs:
        assert not _urls(messages)
        assert "Additional image" not in _labels(messages)


def test_same_seed_reset_omits_workspace_but_keeps_fresh_extras_without_repeating_task():
    context = MessageContext({"enabled": True, "max_images_per_section": 1})
    reset_obs = _obs(extras=True, text="new attempt")
    messages = context.build_auto_reset_messages(
        {"next_obs": {"step_msg": "terminal"}, "reward": -1, "done": True}, reset_obs,
        current_attempt_index=1, next_attempt_index=2, fallback_task_prompt="STATIC TASK",
        omit_reset_obs_images=True, reset_obs_image_omitted_reason="same seed")
    assert _urls(messages) == [_data_url(item["image"]) for item in reset_obs["attempt_initial_images"]]
    assert "same seed" in _labels(messages)
    assert "STATIC TASK" not in _labels(messages)
    assert "new attempt" not in _labels(messages)


@pytest.mark.parametrize("return_mode", ["full", "delta"])
def test_rollout_append_history_keeps_one_event_per_attempt(return_mode):
    context = ArkRolloutContext()
    cfg = _cfg(append_only=True, return_mode=return_mode)
    initial = context.on_reset(cfg, {7: _obs(extras=True)})
    assert _labels(initial[7]["messages"]).count("Additional image 1") == 1
    next_obs = {7: _obs(text="after move")}
    context.record_transition(next_obs, {7: "move"}, {7: 0}, {7: False})
    stepped = context.finalize_obs(next_obs)
    assert _labels(stepped[7]["messages"]).count("Additional image 1") == (1 if return_mode == "full" else 0)
    reset = context.on_attempt_reset(cfg, {7: _obs(extras=True)}, current_attempt_index=1,
                                     next_attempt_index=2, omit_reset_obs_images=True)
    assert _labels(reset[7]["messages"]).count("Additional image 1") == (2 if return_mode == "full" else 1)


@pytest.mark.parametrize("chat_transcript", [True, False])
@pytest.mark.parametrize("delayed_initial", [True, False])
def test_nonappend_history_renders_an_attachment_event_once(chat_transcript, delayed_initial):
    context = MessageContext({"enabled": True, "append_only": False, "max_images_per_section": 2})
    initial = _obs(extras=not delayed_initial)
    first_result = _obs(extras=delayed_initial, text="first result")
    final_result = _obs(text="second result")
    steps = [{"obs": initial, "next_obs": first_result, "action": "one", "reward": 0, "done": False},
             {"obs": first_result, "next_obs": final_result, "action": "two", "reward": 0, "done": False}]
    build = context.build_chat_messages if chat_transcript else context.build_messages
    messages = build(final_result, [], steps)
    assert _labels(messages).count("Additional image 1") == 1
    assert _labels(messages).count("Additional image 2") == 1


@pytest.mark.parametrize("chat_transcript", [True, False])
def test_nonappend_without_current_history_keeps_delayed_initial_images(chat_transcript):
    context = MessageContext({"enabled": True, "append_only": False,
                              "include_current_attempt_history": False,
                              "max_images_per_section": 0})
    current = _obs(extras=True, text="first visible decision")
    steps = [{"obs": _obs(), "next_obs": current, "action": "wait", "reward": 0, "done": False}]
    build = context.build_chat_messages if chat_transcript else context.build_messages
    messages = build(current, [], steps)
    assert _urls(messages) == [_data_url(entry["image"]) for entry in current["attempt_initial_images"]]
    assert _labels(messages).count("Additional image 1") == 1
    assert _labels(messages).count("Additional image 2") == 1


@pytest.mark.parametrize("legacy_context", [False, True])
def test_delayed_initial_event_is_rendered_in_step_even_with_no_camera_budget(legacy_context):
    cfg = _cfg(append_only=True, return_mode="delta", max_images_per_section=0)
    context = ContextManager() if legacy_context else ArkRolloutContext()
    if legacy_context:
        context.on_reset(cfg, {7: 7}, {7: _obs()})
    else:
        context.on_reset(cfg, {7: _obs()})
    next_obs = {7: _obs(extras=True)}
    if legacy_context:
        context.record_step(next_obs, {7: "move"}, {7: 0}, {7: False})
    else:
        context.record_transition(next_obs, {7: "move"}, {7: 0}, {7: False})
    messages = context.finalize_obs(next_obs)[7]["messages"]
    assert len(_urls(messages)) == 2
    assert _labels(messages).count("Additional image 1") == 1


@pytest.mark.parametrize("agent_kind", ["api", "codex"])
def test_raw_fallback_trace_and_actual_request_include_extras_without_state(agent_kind):
    obs = {7: _obs(extras=True)}
    captured = []
    agent = object.__new__(APIAgent) if agent_kind == "api" else CodexAgent(timeout_s=None)
    first = agent.build_request_messages(obs)
    second = agent.build_request_messages(obs)
    assert first == second
    assert len(_urls(first[7])) == 4  # latest frame per camera, then two extras
    assert _urls(first[7])[-2:] == [_data_url(entry["image"]) for entry in obs[7]["attempt_initial_images"]]
    if agent_kind == "api":
        def fake_call(messages):
            captured.append(deepcopy(messages))
            return "move", None
        with patch.object(agent, "_call_api_with_usage", side_effect=fake_call):
            agent.forward_with_trace(obs)
        assert captured == [first[7]]
    else:
        def fake_run(rendered, **kwargs):
            captured.append(rendered)
            return CodexRunOutput("move")
        with patch.object(agent, "_run_codex_with_timeout", side_effect=fake_run):
            agent.forward_with_trace(obs)
        assert captured[0].image_urls == _urls(first[7])
        assert "Additional image 1" in captured[0].prompt
    assert agent.build_request_messages(obs) == first


@pytest.mark.parametrize("legacy_context", [False, True])
def test_first_decision_after_skipped_reset_gets_initial_images(legacy_context):
    cfg = _cfg(append_only=True, return_mode="delta", max_images_per_section=0)
    context = ContextManager() if legacy_context else ArkRolloutContext()
    skipped = {7: {"skip_infer": True, "step_msg": "waiting", "vis": []}}
    if legacy_context:
        context.on_reset(cfg, {7: 7}, skipped)
    else:
        context.on_reset(cfg, skipped)
    messages = context.finalize_obs({7: _obs(extras=True)})[7]["messages"]
    assert len(_urls(messages)) == 2
    assert _labels(messages).count("Additional image 1") == 1


@pytest.mark.parametrize("format_name", ["openai", "raw_base64"])
def test_api_and_codex_clients_receive_real_images_for_both_message_formats(format_name):
    context = MessageContext({"enabled": True, "max_images_per_section": 0,
                              "image_part_format": format_name})
    messages = context.build_chat_messages(_obs(extras=True), [], [])
    original = deepcopy(messages)
    urls = _urls(messages)
    api = object.__new__(APIAgent)
    requests = api.build_request_messages({7: {"messages": messages}})[7]
    assert _urls(requests) == urls
    assert all(part["type"] != "image" for part in _parts(requests))
    assert messages == original

    codex = CodexAgent(timeout_s=None)
    rendered = codex._messages_to_codex_prompt(messages, agent_idx=7)
    assert rendered.image_urls == urls
    assert "image_base64" not in rendered.prompt
    assert "Additional image 1" in rendered.prompt

def test_invalid_optional_entries_do_not_break_rendering_or_emit_unpaired_labels():
    obs = {"attempt_initial_images": [None, {"index": True, "image": _image("blue")},
                                     {"index": 0, "image": _image("blue")},
                                     {"index": 1, "image": "a path is not an image"}, _entry(2, "red")]}
    parts = MessageContext().attempt_initial_images_to_parts(obs)
    assert len(parts) == 2
    assert parts[0]["text"] == "Additional image 2"
