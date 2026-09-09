"""Build a human-readable AgentArk replay review workbench.

The builder combines local task source/configuration with image-inclusive
trajectory records published in the public AgentArk Hugging Face dataset.  It
keeps only a highest- and lowest-scoring record per task, decodes the exact
model-visible frames, and emits a portable static site.

This module intentionally does not mutate task source or the packaged runtime.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests
import yaml


HF_REPO_ID = "P90-RushB/AgentArk"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"
HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"
DEFAULT_IDS = (41, 43, 44, 46, *range(69, 101))
IMAGE_MARKER = "agentark.pil_image_png_base64.v1"
REVIEW_NOTES_SCHEMA = "agentark.human_review.notes.v1"
REVIEW_NOTES_FILENAME = "review_notes.json"

_PRINT_LOCK = threading.Lock()
_REVIEW_NOTES_LOCK = threading.Lock()


def _log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _read_review_notes(output_root: Path) -> Dict[str, Any]:
    path = output_root / REVIEW_NOTES_FILENAME
    if not path.exists():
        return {"schema": REVIEW_NOTES_SCHEMA, "notes": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != REVIEW_NOTES_SCHEMA:
        raise ValueError(f"Invalid review notes schema in {path}")
    notes = payload.get("notes")
    if not isinstance(notes, dict) or any(
        not str(task_id).isdigit() or not isinstance(note, str)
        for task_id, note in notes.items()
    ):
        raise ValueError(f"Invalid review notes payload in {path}")
    return {
        "schema": REVIEW_NOTES_SCHEMA,
        "notes": {str(task_id): note for task_id, note in notes.items()},
    }


def _write_review_notes(output_root: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    notes = payload.get("notes")
    if not isinstance(notes, dict):
        raise ValueError("Review notes request must contain a notes object")
    normalized: Dict[str, str] = {}
    for task_id, note in notes.items():
        key = str(task_id)
        if not key.isdigit() or not isinstance(note, str):
            raise ValueError("Review note keys must be task ids and values must be strings")
        if len(note) > 100_000:
            raise ValueError(f"Review note for Task{key} exceeds 100000 characters")
        normalized[key] = note
    document = {"schema": REVIEW_NOTES_SCHEMA, "notes": normalized}
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / REVIEW_NOTES_FILENAME
    temporary = output_root / f"{REVIEW_NOTES_FILENAME}.tmp"
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return document


def _request_bytes(url: str, *, timeout_s: float = 600.0) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            response = requests.get(url, timeout=(20.0, timeout_s))
            response.raise_for_status()
            return response.content
        except Exception as exc:  # pragma: no cover - network timing is external
            last_error = exc
            if attempt == 3:
                break
    raise RuntimeError(f"Unable to download {url}: {last_error}")


def _download_verified(url: str, target: Path, expected_sha256: Optional[str]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        existing_size = target.stat().st_size if target.exists() else 0
        if existing_size > 0 and expected_sha256:
            existing_digest = hashlib.sha256()
            with target.open("rb") as existing_handle:
                for chunk in iter(lambda: existing_handle.read(1024 * 1024), b""):
                    existing_digest.update(chunk)
            if existing_digest.hexdigest().lower() == expected_sha256.lower():
                return
        headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20.0, 45.0)) as response:
                response.raise_for_status()
                resumed = existing_size > 0 and response.status_code == 206
                mode = "ab" if resumed else "wb"
                if existing_size > 0 and not resumed:
                    _log(f"  server did not honor Range; restarting {target.name}")
                with target.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if expected_sha256 and actual.lower() != expected_sha256.lower():
                target.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SHA-256 mismatch for {url}: expected={expected_sha256} actual={actual}"
                )
            return
        except Exception as exc:  # pragma: no cover - network timing is external
            last_error = exc
            partial = target.stat().st_size if target.exists() else 0
            _log(
                f"  retry {attempt}/3 from {partial / (1024 * 1024):.1f} MiB "
                f"after {type(exc).__name__}: {exc}"
            )
    raise RuntimeError(f"Unable to download and verify {url}: {last_error}")


def _jsonl_objects(payload: bytes) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_number, raw in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if isinstance(record, dict):
            record["_registry_line"] = line_number
            records.append(record)
    return records


def fetch_record_registry() -> List[Dict[str, Any]]:
    return _jsonl_objects(_request_bytes(f"{HF_RESOLVE_BASE}/registry/records.jsonl"))


def _configured_task_id(directory: Path, config: Mapping[str, Any]) -> Optional[int]:
    match = re.match(r"^Task(\d+)(?:_|$)", directory.name)
    if match:
        return int(match.group(1))
    info = config.get("task_info") if isinstance(config.get("task_info"), dict) else {}
    try:
        return int(info.get("id"))
    except (TypeError, ValueError):
        pass
    identities = [
        str(config.get("task_name") or ""),
        str(info.get("name") or ""),
        *(str(value) for value in info.get("legacy_names", []) or []),
    ]
    for identity in identities:
        match = re.search(r"(?:^|_)Task(\d+)(?:_|$)", identity)
        if match:
            return int(match.group(1))
    return None


def discover_tasks(
    task_roots: Path | Iterable[Path],
    ids: Iterable[int],
    *,
    include_gui: bool = False,
) -> Dict[int, Dict[str, Any]]:
    wanted = set(int(item) for item in ids)
    found: Dict[int, Dict[str, Any]] = {}
    roots = [task_roots] if isinstance(task_roots, Path) else list(task_roots)
    for task_root in roots:
        if not task_root.exists():
            continue
        for directory in task_root.iterdir():
            if not directory.is_dir():
                continue
            config_path = directory / "task_config.yaml"
            if not config_path.exists():
                config_path = directory / "cfg" / "task_config.yaml"
            if not config_path.exists():
                continue
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            task_id = _configured_task_id(directory, config)
            if task_id not in wanted or task_id in found:
                continue
            info = config.get("task_info") if isinstance(config.get("task_info"), dict) else {}
            tags = [str(value) for value in info.get("tags", []) or []]
            display_name = str(info.get("name") or config.get("task_name") or directory.name)
            is_gui = display_name.lower().startswith("gui") or any(tag.lower() == "gui" for tag in tags)
            if is_gui and not include_gui:
                continue
            found[task_id] = {
                "id": task_id,
                "directory": directory,
                "folder": directory.name,
                "config_path": config_path,
                "config": config,
                "name": display_name,
                "tags": tags,
            }
    return found


def select_trajectory_entries(
    registry: Sequence[Mapping[str, Any]], task_ids: Iterable[int]
) -> Dict[int, Dict[str, Any]]:
    wanted = set(int(item) for item in task_ids)
    selected: Dict[int, Dict[str, Any]] = {}
    for item in registry:
        try:
            task_id = int(item.get("task_id"))
        except (TypeError, ValueError):
            continue
        record_kind = str(item.get("record_kind") or "").strip().lower()
        if not record_kind:
            # Older record-registry rows only declared ``kind=record``. Their
            # immutable path still carries the result/trajectory distinction.
            filename = Path(str(item.get("path") or "")).name.lower()
            if filename.endswith("_trajectories.jsonl"):
                record_kind = "trajectories"
            elif filename.endswith("_results.jsonl"):
                record_kind = "results"
        if task_id not in wanted or record_kind != "trajectories":
            continue
        candidate = dict(item)
        current = selected.get(task_id)
        if current is None or int(candidate.get("records", 0)) > int(current.get("records", 0)):
            selected[task_id] = candidate
    return selected


def _score(record: Mapping[str, Any]) -> float:
    rollout = record.get("rollout") if isinstance(record.get("rollout"), dict) else {}
    try:
        return float(rollout.get("score_reward", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _seed(record: Mapping[str, Any]) -> int:
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    try:
        return int(task.get("group_seed", task.get("requested_group_seed", 0)))
    except (TypeError, ValueError):
        return 0


def select_reward_extremes(records: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not records:
        raise ValueError("Trajectory file contains no records")
    high = sorted(records, key=lambda item: (-_score(item), _seed(item)))[0]
    low_candidates = sorted(records, key=lambda item: (_score(item), _seed(item)))
    low = next((item for item in low_candidates if item is not high), high)
    if _score(low) == _score(high) and len(records) > 1:
        low = sorted(records, key=_seed)[-1]
    return high, low


def _first_agent_attempts(record: Mapping[str, Any]) -> List[List[Dict[str, Any]]]:
    snapshot = record.get("history_snapshot")
    if not isinstance(snapshot, dict) or not snapshot:
        return []
    agent_key = sorted(snapshot.keys(), key=lambda value: int(value) if str(value).isdigit() else str(value))[0]
    attempts = snapshot.get(agent_key)
    if not isinstance(attempts, list):
        return []
    return [attempt for attempt in attempts if isinstance(attempt, list)]


def _all_steps(record: Mapping[str, Any]) -> List[Tuple[int, int, Dict[str, Any]]]:
    flattened: List[Tuple[int, int, Dict[str, Any]]] = []
    for attempt_index, attempt in enumerate(_first_agent_attempts(record), start=1):
        for step_index, step in enumerate(attempt, start=1):
            if isinstance(step, dict):
                flattened.append((attempt_index, step_index, step))
    return flattened


def _runtime_capture_turns(record: Mapping[str, Any]) -> List[List[Dict[str, Any]]]:
    """Return exact model-request images when full history was not published."""

    capture = record.get("runtime_request_capture")
    if not isinstance(capture, dict):
        return []
    turns = capture.get("turns")
    if not isinstance(turns, list):
        return []
    captured: List[List[Dict[str, Any]]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        agents = turn.get("agents")
        if not isinstance(agents, dict) or not agents:
            continue
        agent_key = sorted(
            agents.keys(),
            key=lambda value: int(value) if str(value).isdigit() else str(value),
        )[0]
        agent = agents.get(agent_key)
        images = agent.get("images") if isinstance(agent, dict) else None
        if not isinstance(images, list):
            continue
        captured.append(
            [
                image
                for image in images
                if isinstance(image, dict) and isinstance(image.get("data"), str)
            ]
        )
    return captured


def _extract_task_prompt(record: Mapping[str, Any]) -> str:
    for _, _, step in _all_steps(record):
        obs = step.get("obs") if isinstance(step.get("obs"), dict) else {}
        prompt = obs.get("task_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if "[task prompt]" in prompt:
            prompt = prompt[prompt.index("[task prompt]") :]
        marker = prompt.find("<tool_docs>")
        if marker >= 0:
            prompt = prompt[:marker]
        return prompt.strip()
    return ""


def _normalize_authored_prompt(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    value = value.replace("''", "'")
    paragraphs: List[str] = []
    current: List[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


def _extract_authored_task_prompt(task_directory: Path) -> str:
    """Best-effort prompt recovery from editable prefabs or task DLL strings."""

    property_pattern = re.compile(
        r"^  taskDescription:[ \t]*(\S.*?)(?=^  [A-Za-z_][A-Za-z0-9_]*:[ \t]*)",
        re.MULTILINE | re.DOTALL,
    )
    for prefab in sorted(task_directory.rglob("*.prefab")):
        try:
            text = prefab.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        match = property_pattern.search(text)
        if match:
            prompt = _normalize_authored_prompt(match.group(1))
            if prompt:
                return prompt

    declaration_pattern = re.compile(
        r"\b(?:const|static\s+readonly)\s+string\s+Default(?:Task)?Description\s*=\s*"
    )
    string_literal_pattern = re.compile(r'@?"(?:""|\\.|[^"\\])*"', re.DOTALL)
    for source in sorted(task_directory.rglob("*.cs")):
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        declaration = declaration_pattern.search(text)
        if not declaration:
            continue
        start = declaration.end()
        in_string = False
        verbatim = False
        escaped = False
        end = None
        index = start
        while index < len(text):
            character = text[index]
            if in_string:
                if verbatim:
                    if character == '"' and index + 1 < len(text) and text[index + 1] == '"':
                        index += 2
                        continue
                    if character == '"':
                        in_string = False
                elif escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '@' and index + 1 < len(text) and text[index + 1] == '"':
                in_string = True
                verbatim = True
                index += 2
                continue
            elif character == '"':
                in_string = True
                verbatim = False
            elif character == ';':
                end = index
                break
            index += 1
        if end is None:
            continue
        try:
            parts: List[str] = []
            for literal in string_literal_pattern.findall(text[start:end]):
                if literal.startswith('@"'):
                    parts.append(literal[2:-1].replace('""', '"'))
                else:
                    parts.append(json.loads(literal))
            prompt = "".join(parts).strip()
        except (TypeError, json.JSONDecodeError):
            continue
        if prompt:
            return prompt

    marker = "[task prompt]".encode("utf-16le")
    for assembly in sorted(task_directory.rglob("*.dll")):
        try:
            raw = assembly.read_bytes()
        except OSError:
            continue
        start = raw.find(marker)
        if start < 0:
            continue
        decoded = raw[start : start + 64 * 1024].decode("utf-16le", errors="replace")
        end = next(
            (
                index
                for index, character in enumerate(decoded)
                if ord(character) not in {9, 10, 13} and not 32 <= ord(character) <= 126
            ),
            len(decoded),
        )
        prompt = decoded[:end].strip()
        if prompt:
            return prompt
    return ""


def _tool_call(action: Any) -> Dict[str, Any]:
    text = "" if action is None else str(action)
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not match:
        return {"name": "未解析动作", "arguments": {}, "raw": text}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"name": "格式错误动作", "arguments": {}, "raw": text}
    return {
        "name": str(payload.get("name") or "未命名动作"),
        "arguments": payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {},
        "raw": text,
    }


def _visible_text(observation: Any) -> str:
    if not isinstance(observation, dict):
        return ""
    parts: List[str] = []
    for key in ("reset_context", "step_msg", "step_context", "observation_context"):
        value = observation.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def _encoded_frames(observation: Any) -> List[Tuple[int, int, Dict[str, Any]]]:
    if not isinstance(observation, dict):
        return []
    vis = observation.get("vis")
    if not isinstance(vis, list):
        return []
    result: List[Tuple[int, int, Dict[str, Any]]] = []
    for camera_index, camera_frames in enumerate(vis):
        if not isinstance(camera_frames, list):
            continue
        for frame_index, frame in enumerate(camera_frames):
            if isinstance(frame, dict) and isinstance(frame.get("data"), str):
                result.append((camera_index, frame_index, frame))
    return result


def _write_frames(
    observation: Any,
    replay_dir: Path,
    relative_dir: str,
    *,
    phase: str,
    attempt_index: int,
    step_index: int,
    semantic_images: bool,
) -> List[Dict[str, Any]]:
    if not semantic_images:
        return []
    output: List[Dict[str, Any]] = []
    for camera_index, frame_index, encoded in _encoded_frames(observation):
        try:
            raw = base64.b64decode(encoded["data"], validate=True)
        except Exception:
            continue
        digest = hashlib.sha256(raw).hexdigest()[:16]
        filename = f"{digest}.png"
        target = replay_dir / filename
        if not target.exists():
            target.write_bytes(raw)
        size = encoded.get("size") if isinstance(encoded.get("size"), list) else []
        output.append(
            {
                "path": f"{relative_dir}/{filename}",
                "phase": phase,
                "attempt": attempt_index,
                "step": step_index,
                "camera": camera_index,
                "camera_frame": frame_index,
                "width": size[0] if len(size) > 0 else None,
                "height": size[1] if len(size) > 1 else None,
                "sha256_16": digest,
            }
        )
    return output


def _step_sentence(step_number: int, tool: Mapping[str, Any], reward: Any, done: bool, after: str) -> str:
    arguments = tool.get("arguments") or {}
    arg_text = "、".join(f"{key}={value}" for key, value in arguments.items()) or "无参数"
    outcome = "本轮结束" if done else "环境继续"
    feedback = re.sub(r"\s+", " ", after).strip()
    if len(feedback) > 260:
        feedback = feedback[:257] + "…"
    sentence = f"第 {step_number} 步调用 {tool.get('name')}（{arg_text}），reward={reward}，{outcome}。"
    if feedback:
        sentence += f" 环境反馈：{feedback}"
    return sentence


def build_replay_summary(
    record: Dict[str, Any],
    *,
    kind: str,
    task_dir: Path,
    task_relative_dir: str,
    semantic_images: bool,
) -> Dict[str, Any]:
    replay_dir = task_dir / kind
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_relative_dir = f"{task_relative_dir}/{kind}"
    steps: List[Dict[str, Any]] = []
    playback_frames: List[Dict[str, Any]] = []
    previous_digest: Optional[str] = None
    flat_number = 0
    max_frames_in_one_observation = 0
    for attempt_index, attempt_step_index, step in _all_steps(record):
        flat_number += 1
        before = step.get("obs") if isinstance(step.get("obs"), dict) else {}
        after_obs = step.get("next_obs") if isinstance(step.get("next_obs"), dict) else {}
        before_frames = _write_frames(
            before,
            replay_dir,
            replay_relative_dir,
            phase="before",
            attempt_index=attempt_index,
            step_index=attempt_step_index,
            semantic_images=semantic_images,
        )
        after_frames = _write_frames(
            after_obs,
            replay_dir,
            replay_relative_dir,
            phase="after",
            attempt_index=attempt_index,
            step_index=attempt_step_index,
            semantic_images=semantic_images,
        )
        max_frames_in_one_observation = max(
            max_frames_in_one_observation, len(before_frames), len(after_frames)
        )
        for frame in [*before_frames, *after_frames]:
            digest = frame["sha256_16"]
            if digest == previous_digest:
                continue
            playback_frames.append(frame)
            previous_digest = digest
        tool = _tool_call(step.get("action"))
        reward = step.get("reward", 0)
        done = bool(step.get("done", False))
        after_text = _visible_text(after_obs)
        steps.append(
            {
                "number": flat_number,
                "attempt": attempt_index,
                "attempt_step": attempt_step_index,
                "tool": tool,
                "reward": reward,
                "done": done,
                "before_text": _visible_text(before),
                "after_text": after_text,
                "before_frames": before_frames,
                "after_frames": after_frames,
                "explanation_zh": _step_sentence(flat_number, tool, reward, done, after_text),
            }
        )
    trajectory_detail = "full"
    limitation_zh = ""
    if not steps:
        capture_turns = _runtime_capture_turns(record)
        if capture_turns:
            trajectory_detail = "observation_only"
            limitation_zh = (
                "该已发布 record 没有 history_snapshot，因而无法还原 agent 动作、逐步 reward "
                "和精确环境转移；下方按轮次展示的是模型请求中实际包含的图片。"
            )
            for turn_index, images in enumerate(capture_turns, start=1):
                observation = {"vis": [images]}
                before_frames = _write_frames(
                    observation,
                    replay_dir,
                    replay_relative_dir,
                    phase="model_request",
                    attempt_index=1,
                    step_index=turn_index,
                    semantic_images=semantic_images,
                )
                max_frames_in_one_observation = max(
                    max_frames_in_one_observation, len(before_frames)
                )
                for frame in before_frames:
                    digest = frame["sha256_16"]
                    if digest == previous_digest:
                        continue
                    playback_frames.append(frame)
                    previous_digest = digest
                steps.append(
                    {
                        "number": turn_index,
                        "attempt": 1,
                        "attempt_step": turn_index,
                        "tool": {
                            "name": "动作未随 record 发布",
                            "arguments": {},
                            "raw": "",
                        },
                        "reward": None,
                        "done": False,
                        "before_text": "",
                        "after_text": "",
                        "before_frames": before_frames,
                        "after_frames": [],
                        "record_observation_only": True,
                        "explanation_zh": (
                            f"第 {turn_index} 轮模型请求：展示该轮实际发送给 agent 的 "
                            f"{len(before_frames)} 张图片；发布记录未包含本轮 action 与逐步 reward。"
                        ),
                    }
                )
        else:
            trajectory_detail = "summary_only"
            limitation_zh = (
                "该已发布 record 只有 rollout 汇总，没有可还原的动作、逐步反馈或模型可见帧。"
            )
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    rollout = record.get("rollout") if isinstance(record.get("rollout"), dict) else {}
    return {
        "kind": kind,
        "seed": _seed(record),
        "score_reward": _score(record),
        "last_attempt_reward": rollout.get("last_attempt_reward"),
        "best_attempt_reward": rollout.get("best_attempt_reward"),
        "rollout_success": bool(rollout.get("rollout_success", False)),
        "rollout_truncated": bool(rollout.get("rollout_truncated", False)),
        "max_attempts": rollout.get("max_attempts"),
        "max_steps_per_attempt": rollout.get("max_steps_per_attempt"),
        "attempt_group_seed_history": rollout.get("attempt_group_seed_history") or [],
        "attempt_rewards": rollout.get("attempt_rewards") or [],
        "case_id": source.get("case_id"),
        "model_name": source.get("model_name"),
        "task_name": task.get("task_name") or task.get("requested_task_name"),
        "trajectory_detail": trajectory_detail,
        "record_limitation_zh": limitation_zh,
        "steps": steps,
        "frames": playback_frames,
        "frame_count": len(playback_frames),
        "max_frames_in_one_observation": max_frames_in_one_observation,
        "has_agent_transition_video": max_frames_in_one_observation > 1,
    }


def _parse_standard_trajectory(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "markdown": "", "actions": []}
    markdown = path.read_text(encoding="utf-8")
    actions: List[Dict[str, Any]] = []
    current_heading = "标准轨迹"
    action_number = 0
    for line in markdown.splitlines():
        if line.startswith("## "):
            current_heading = line[3:].strip()
        for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", line):
            action_number += 1
            tool = _tool_call(f"<tool_call>{match.group(1)}</tool_call>")
            actions.append(
                {
                    "number": action_number,
                    "section": current_heading,
                    "tool": tool,
                    "explanation_zh": _step_sentence(action_number, tool, "见验证记录", False, ""),
                }
            )
    return {"exists": True, "markdown": markdown, "actions": actions}


def _critical_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    info = config.get("task_info") if isinstance(config.get("task_info"), dict) else {}
    engine = config.get("engine_para") if isinstance(config.get("engine_para"), dict) else {}
    wrapper = config.get("env_wrapper_cfg") if isinstance(config.get("env_wrapper_cfg"), dict) else {}
    messages = {}
    context = wrapper.get("context_manager") if isinstance(wrapper.get("context_manager"), dict) else {}
    if isinstance(context.get("messages"), dict):
        messages = context["messages"]
    return {
        "task_name": config.get("task_name"),
        "version": info.get("version"),
        "tags": info.get("tags") or [],
        "action_mode": config.get("action_mode"),
        "obs_mode": config.get("obs_mode"),
        "capture_interval": config.get("capture_interval"),
        "video_frame_selection": wrapper.get("video_frame_selection"),
        "max_images_per_section": messages.get("max_images_per_section"),
        "width": config.get("width"),
        "height": config.get("height"),
        "max_attempts": config.get("max_attempts"),
        "max_steps_per_attempt": config.get("max_steps_per_attempt"),
        "time_between_decisions": config.get("time_between_decisions"),
        "time_scale": engine.get("time_scale"),
        "reroll_group_seed_on_same_task": config.get("reroll_group_seed_on_same_task"),
    }


def _semantic_images(config: Mapping[str, Any]) -> bool:
    tags = [str(value).lower() for value in config.get("task_info", {}).get("tags", []) or []]
    width = int(config.get("width", 0) or 0)
    height = int(config.get("height", 0) or 0)
    return "text-obs" not in tags and not (0 < width <= 32 and 0 < height <= 32)


def build_one_task(
    task: Mapping[str, Any],
    entry: Mapping[str, Any],
    curated: Mapping[str, Any],
    output_root: Path,
    work_root: Path,
    *,
    force: bool,
) -> Dict[str, Any]:
    task_id = int(task["id"])
    task_output = output_root / "tasks" / f"task_{task_id:03d}"
    manifest_path = task_output / "task.json"
    if manifest_path.exists() and not force:
        _log(f"[{task_id:03d}] reuse completed task")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    if force and task_output.exists():
        shutil.rmtree(task_output)
    task_output.mkdir(parents=True, exist_ok=True)
    relative_task_dir = f"tasks/task_{task_id:03d}"
    remote_path = str(entry["path"])
    temp_path = work_root / f"task_{task_id:03d}.jsonl"
    _log(f"[{task_id:03d}] downloading {int(entry.get('size_bytes', 0)) / (1024 * 1024):.1f} MiB")
    _download_verified(
        f"{HF_RESOLVE_BASE}/{remote_path}",
        temp_path,
        str(entry.get("sha256") or "") or None,
    )
    records: List[Dict[str, Any]] = []
    completed = False
    try:
        with temp_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        high, low = select_reward_extremes(records)
        config = task["config"]
        semantic_images = _semantic_images(config)
        replays = [
            build_replay_summary(
                high,
                kind="high",
                task_dir=task_output,
                task_relative_dir=relative_task_dir,
                semantic_images=semantic_images,
            ),
            build_replay_summary(
                low,
                kind="low",
                task_dir=task_output,
                task_relative_dir=relative_task_dir,
                semantic_images=semantic_images,
            ),
        ]
        standard = _parse_standard_trajectory(task["directory"] / "action_trajectories.md")
        record_prompt = _extract_task_prompt(high) or _extract_task_prompt(low)
        authored_prompt = "" if record_prompt else _extract_authored_task_prompt(task["directory"])
        prompt = record_prompt or authored_prompt
        limitations = [
            replay["record_limitation_zh"]
            for replay in replays
            if replay.get("record_limitation_zh")
        ]
        if not prompt:
            limitations.append(
                "已发布 replay 未携带任务 prompt，当前可用的打包任务文件中也无法可靠恢复原文；"
                "中文玩法说明来自任务配置、命名和可观察 replay，不能替代原始英文 prompt。"
            )
        task_manifest = {
            "id": task_id,
            "name": task["name"],
            "folder": task["folder"],
            "title_zh": curated.get("title") or task["name"],
            "family": curated.get("family") or "未分类",
            "summary_zh": curated.get("summary") or "",
            "play_zh": curated.get("play") or [],
            "audit_zh": curated.get("audit") or [],
            "config": _critical_config(config),
            "semantic_modality": "image" if semantic_images else "text",
            "prompt_en": prompt,
            "prompt_provenance": (
                "record" if record_prompt else "authored_source" if authored_prompt else "unavailable"
            ),
            "record_limitations_zh": list(dict.fromkeys(limitations)),
            "standard_trajectory": standard,
            "replays": replays,
            "hf": {
                "repo_id": HF_REPO_ID,
                "dataset_url": HF_DATASET_URL,
                "record_path": remote_path,
                "record_url": f"{HF_RESOLVE_BASE}/{remote_path}",
                "sha256": entry.get("sha256"),
                "record_set": entry.get("record_set"),
                "model": entry.get("model"),
                "reasoning_effort": entry.get("reasoning_effort"),
            },
        }
        manifest_path.write_text(
            json.dumps(task_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        completed = True
        _log(
            f"[{task_id:03d}] ready high={replays[0]['score_reward']:.4f} "
            f"low={replays[1]['score_reward']:.4f} frames={replays[0]['frame_count']}+{replays[1]['frame_count']}"
        )
        return task_manifest
    finally:
        if completed:
            temp_path.unlink(missing_ok=True)


def _copy_site_assets(output_root: Path) -> None:
    source = Path(__file__).with_name("human_review_assets")
    for name in ("index.html", "app.js", "styles.css"):
        shutil.copy2(source / name, output_root / name)


def _write_root_manifest(output_root: Path, *, title: Optional[str] = None) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    tasks_root = output_root / "tasks"
    for path in sorted(tasks_root.glob("task_*/task.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    results.sort(key=lambda item: int(item["id"]))
    data = {
        "schema": "agentark.human_review.v1",
        "title": title or "AgentArk 人工审核工作台",
        "task_count": len(results),
        "task_ids": [item["id"] for item in results],
        "hf_dataset_url": HF_DATASET_URL,
        "tasks": results,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "data.js").write_text(
        "window.AGENTARK_REVIEW_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    return data


def build_workbench(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).resolve()
    task_root = repo_root / "llm_rl/Assets/llm_gym/rl_train/RLTaskDev/AgentTask"
    task_roots = [task_root, *(Path(value).resolve() for value in args.fallback_task_root or [])]
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    curated_path = Path(args.curated).resolve() if args.curated else (
        repo_root / "AgentArk/config/human_review/task41_100_non_gui_zh.json"
    )
    if not curated_path.exists():
        raise FileNotFoundError(f"Chinese curation file not found: {curated_path}")
    curated = json.loads(curated_path.read_text(encoding="utf-8"))

    requested_ids = tuple(int(value) for value in (args.task_ids or DEFAULT_IDS))
    tasks = discover_tasks(task_roots, requested_ids, include_gui=bool(args.include_gui))
    missing_local = sorted(set(requested_ids) - set(tasks))
    if missing_local:
        scope = "tasks" if args.include_gui else "non-GUI tasks"
        raise RuntimeError(f"Missing local {scope}: {missing_local}")
    registry = fetch_record_registry()
    entries = select_trajectory_entries(registry, tasks)
    missing_remote = sorted(set(tasks) - set(entries))
    if missing_remote:
        raise RuntimeError(f"Missing HF image-inclusive trajectories: {missing_remote}")

    _copy_site_assets(output_root)
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                build_one_task,
                tasks[task_id],
                entries[task_id],
                curated.get(str(task_id), {}),
                output_root,
                work_root,
                force=bool(args.force),
            ): task_id
            for task_id in sorted(tasks)
        }
        for future in as_completed(futures):
            results.append(future.result())
    _write_root_manifest(output_root, title=args.title)
    try:
        work_root.rmdir()
    except OSError:
        pass
    _log(f"Workbench ready: {output_root / 'index.html'}")
    return output_root


def _save_live_observation_frames(
    observation: Mapping[str, Any],
    clip_dir: Path,
    clip_relative_dir: str,
    *,
    phase: str,
    step_index: int,
    previous_digest: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    output: List[Dict[str, Any]] = []
    vis = observation.get("vis")
    if not isinstance(vis, list):
        return output, previous_digest
    for camera_index, camera_frames in enumerate(vis):
        if not isinstance(camera_frames, list):
            continue
        for camera_frame, image in enumerate(camera_frames):
            if not hasattr(image, "save"):
                continue
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            raw = buffer.getvalue()
            digest = hashlib.sha256(raw).hexdigest()[:16]
            filename = f"{digest}.png"
            target = clip_dir / filename
            if not target.exists():
                target.write_bytes(raw)
            width, height = getattr(image, "size", (None, None))
            output.append(
                {
                    "path": f"{clip_relative_dir}/{filename}",
                    "phase": phase,
                    "attempt": 1,
                    "step": step_index,
                    "camera": camera_index,
                    "camera_frame": camera_frame,
                    "width": width,
                    "height": height,
                    "sha256_16": digest,
                }
            )
            previous_digest = digest
    return output, previous_digest


def capture_unity_clips(args: argparse.Namespace) -> None:
    from agent_ark.ark_env.ark_env import ArkEnv

    output_root = Path(args.output).resolve()
    _copy_site_assets(output_root)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    requested = set(int(value) for value in args.task_ids)
    tasks = [task for task in manifest["tasks"] if int(task["id"]) in requested]
    missing = requested - {int(task["id"]) for task in tasks}
    if missing:
        raise RuntimeError(f"Tasks not found in workbench: {sorted(missing)}")

    for task in tasks:
        task_id = int(task["id"])
        clips: List[Dict[str, Any]] = []
        for replay in task["replays"]:
            actions = [step["tool"]["raw"] for step in replay["steps"] if step["attempt"] == 1]
            if not actions:
                continue
            clip_dir = output_root / "tasks" / f"task_{task_id:03d}" / "unity" / replay["kind"]
            if clip_dir.exists() and args.force:
                shutil.rmtree(clip_dir)
            clip_dir.mkdir(parents=True, exist_ok=True)
            clip_relative_dir = f"tasks/task_{task_id:03d}/unity/{replay['kind']}"
            cfg = {
                "env_path": str(Path(args.env_path).resolve()),
                "mod_path": str(Path(args.mod_path).resolve()),
                "task_type": "RLTask",
                "base_port": int(args.base_port),
                "env_config_overrides": {
                    "num_parallel_envs": 1,
                    "obs_mode": "video",
                    "capture_interval": int(args.capture_interval),
                    "engine_para": {"time_scale": float(args.time_scale)},
                    "env_wrapper_cfg": {
                        "video_frame_selection": "transition_and_decision",
                        "context_manager": {"messages": {"max_images_per_section": 512}},
                    },
                },
            }
            _log(
                f"[{task_id:03d}/{replay['kind']}] packaged Unity capture "
                f"seed={replay['seed']} actions={len(actions)}"
            )
            env = ArkEnv(cfg)
            frames: List[Dict[str, Any]] = []
            previous_digest: Optional[str] = None
            rewards: List[float] = []
            try:
                obs, _ = env.reset(
                    task_name=replay["task_name"],
                    group_seed=int(replay["seed"]),
                    env_id=0,
                    max_attempts=1,
                )
                agent_id = sorted(obs.keys())[0]
                initial, previous_digest = _save_live_observation_frames(
                    obs[agent_id],
                    clip_dir,
                    clip_relative_dir,
                    phase="reset",
                    step_index=0,
                    previous_digest=previous_digest,
                )
                frames.extend(initial)
                for step_index, action in enumerate(actions, start=1):
                    next_obs, reward, done, info = env.step({agent_id: action})
                    render_errors = info.get("func_render_errors", {}) if isinstance(info, dict) else {}
                    if render_errors:
                        raise RuntimeError(
                            f"Task{task_id}/{replay['kind']} action was not executed: {render_errors}"
                        )
                    rewards.append(float(reward.get(agent_id, 0.0)))
                    next_item = next_obs.get(agent_id, {}) if isinstance(next_obs, dict) else {}
                    captured, previous_digest = _save_live_observation_frames(
                        next_item,
                        clip_dir,
                        clip_relative_dir,
                        phase="transition_and_decision",
                        step_index=step_index,
                        previous_digest=previous_digest,
                    )
                    frames.extend(captured)
                    if done.get("__all__", False) or info.get("truncated", {}).get("__all__", False):
                        break
                    obs = next_obs
                clips.append(
                    {
                        "kind": replay["kind"],
                        "seed": replay["seed"],
                        "capture_interval": int(args.capture_interval),
                        "time_scale": float(args.time_scale),
                        "nominal_fps": 50.0 / int(args.capture_interval),
                        "frame_count": len(frames),
                        "frames": frames,
                        "rewards": rewards,
                        "source": "packaged Unity runtime diagnostic override",
                        "agent_observation_unchanged": False,
                        "note_zh": "额外诊断重放：强制 video + transition_and_decision，用于人工查看决策间运动；不代表原 agent 收到的图像集合。",
                    }
                )
            finally:
                env.close()
            _log(f"[{task_id:03d}/{replay['kind']}] captured {len(frames)} frames")
        task["diagnostic_clips"] = clips
        task_json = output_root / "tasks" / f"task_{task_id:03d}" / "task.json"
        task_json.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_root_manifest(output_root, title=manifest.get("title"))


def validate_workbench(args: argparse.Namespace) -> None:
    from PIL import Image

    output_root = Path(args.output).resolve()
    data = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    errors: List[str] = []
    actual = [int(task["id"]) for task in data.get("tasks", [])]
    expected = (
        set(int(value) for value in args.task_ids)
        if args.task_ids
        else set(int(value) for value in data.get("task_ids", actual))
    )
    if set(actual) != expected or len(actual) != len(set(actual)):
        errors.append(f"task ids mismatch: expected={sorted(expected)} actual={actual}")
    image_count = 0
    for task in data.get("tasks", []):
        task_id = int(task["id"])
        replays = task.get("replays") or []
        if [item.get("kind") for item in replays] != ["high", "low"]:
            errors.append(f"Task{task_id}: missing high/low replay pair")
            continue
        if float(replays[0]["score_reward"]) < float(replays[1]["score_reward"]):
            errors.append(f"Task{task_id}: high score is below low score")
        if not task.get("summary_zh") or not task.get("play_zh") or not task.get("audit_zh"):
            errors.append(f"Task{task_id}: incomplete Chinese review copy")
        if not task.get("prompt_en") and not task.get("record_limitations_zh"):
            errors.append(f"Task{task_id}: missing player prompt")
        for replay in replays:
            if not replay.get("steps"):
                errors.append(f"Task{task_id}/{replay.get('kind')}: no steps")
            if replay.get("trajectory_detail", "full") != "full" and not replay.get(
                "record_limitation_zh"
            ):
                errors.append(
                    f"Task{task_id}/{replay.get('kind')}: incomplete replay is not disclosed"
                )
            for frame in replay.get("frames", []):
                path = output_root / frame["path"]
                if not path.exists():
                    errors.append(f"missing frame: {frame['path']}")
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:
                    errors.append(f"invalid frame {frame['path']}: {exc}")
                image_count += 1
        for clip in task.get("diagnostic_clips", []) or []:
            for frame in clip.get("frames", []):
                if not (output_root / frame["path"]).exists():
                    errors.append(f"missing diagnostic frame: {frame['path']}")
                image_count += 1
    if errors:
        raise RuntimeError("Workbench validation failed:\n- " + "\n- ".join(errors))
    print(
        f"validation_ok=True tasks={len(actual)} replay_pairs={len(actual) * 2} "
        f"referenced_images={image_count}",
        flush=True,
    )


def serve_workbench(args: argparse.Namespace) -> None:
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit

    root = Path(args.output).resolve()
    if not (root / "index.html").exists():
        raise FileNotFoundError(f"No built workbench at {root}")

    class WorkbenchHandler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args: Any, **handler_kwargs: Any) -> None:
            super().__init__(*handler_args, directory=str(root), **handler_kwargs)

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/api/review-notes":
                super().do_GET()
                return
            try:
                with _REVIEW_NOTES_LOCK:
                    payload = _read_review_notes(root)
                self._send_json(200, payload)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def do_PUT(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/api/review-notes":
                self._send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2_000_000:
                    raise ValueError("Review notes request must be between 1 byte and 2 MB")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Review notes request must be a JSON object")
                with _REVIEW_NOTES_LOCK:
                    document = _write_review_notes(root, payload)
                self._send_json(200, document)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

    server = ThreadingHTTPServer((args.host, int(args.port)), WorkbenchHandler)
    url = f"http://{args.host}:{int(args.port)}/"
    print(f"AgentArk human review workbench: {url}", flush=True)
    print(f"Review notes: {root / REVIEW_NOTES_FILENAME}", flush=True)
    if args.open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    default_repo = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="Build the AgentArk human replay review workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Download selected HF records and build the static workbench")
    build.add_argument("--repo-root", default=str(default_repo))
    build.add_argument("--output", default=str(default_repo / "AgentArk/tmp/human_review_task41_100"))
    build.add_argument("--workers", type=int, default=6)
    build.add_argument("--force", action="store_true")
    build.add_argument("--task-ids", nargs="*", type=int)
    build.add_argument(
        "--fallback-task-root",
        action="append",
        default=[],
        help="Optional additional task root, such as packaged Mods/all_tasks; earlier roots take precedence.",
    )
    build.add_argument(
        "--include-gui",
        action="store_true",
        help="Include GUI-tagged tasks. They are excluded by default for backward compatibility.",
    )
    build.add_argument(
        "--curated",
        help="UTF-8 JSON containing Chinese title/family/summary/play/audit copy keyed by task id",
    )
    build.add_argument("--title", help="Workbench title shown in the browser")
    build.set_defaults(func=build_workbench)

    serve = subparsers.add_parser("serve", help="Serve an already-built workbench")
    serve.add_argument("--output", default=str(default_repo / "AgentArk/tmp/human_review_task41_100"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=18182)
    serve.add_argument("--open-browser", action="store_true")
    serve.set_defaults(func=serve_workbench)

    capture = subparsers.add_parser(
        "capture-unity",
        help="Replay selected high/low actions in the packaged runtime and capture extra transition frames",
    )
    capture.add_argument("--output", default=str(default_repo / "AgentArk/tmp/human_review_task41_100"))
    capture.add_argument("--env-path", required=True)
    capture.add_argument("--mod-path", required=True)
    capture.add_argument("--task-ids", nargs="+", type=int, required=True)
    capture.add_argument("--base-port", type=int, default=5005)
    capture.add_argument("--capture-interval", type=int, default=2)
    capture.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Diagnostic Unity time scale; 1.0 preserves enough rendered frames for human playback.",
    )
    capture.add_argument("--force", action="store_true")
    capture.set_defaults(func=capture_unity_clips)

    validate = subparsers.add_parser("validate", help="Validate a built workbench and every referenced frame")
    validate.add_argument("--output", default=str(default_repo / "AgentArk/tmp/human_review_task41_100"))
    validate.add_argument("--task-ids", nargs="*", type=int)
    validate.set_defaults(func=validate_workbench)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Any:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
