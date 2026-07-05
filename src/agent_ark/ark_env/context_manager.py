import random
import re
import hashlib
from collections import deque
from copy import deepcopy
from typing import Dict, List, Any, Optional

try:
    from agent_ark.utils.parse_utils import (
        OBSERVATION_CONTEXT_LABELS,
        OBSERVATION_CONTEXT_TAGS,
        parse_system_task_prompt,
        strip_observation_context_blocks,
        unwrap_observation_context_blocks,
    )
except Exception:  # pragma: no cover
    OBSERVATION_CONTEXT_LABELS = {}
    OBSERVATION_CONTEXT_TAGS = ('reset_context', 'step_context', 'observation_context')
    parse_system_task_prompt = None
    strip_observation_context_blocks = None
    unwrap_observation_context_blocks = None

try:
    from agent_ark.utils.image_utils import env_arr_to_pil_image, pil_image_to_base64
except Exception:  # pragma: no cover
    env_arr_to_pil_image = None
    pil_image_to_base64 = None

try:
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover
    _PILImage = None


def _cfg_value(cfg: Optional[dict], primary_key: str, legacy_key: str, default=None):
    if not isinstance(cfg, dict):
        return default
    if primary_key in cfg:
        return cfg.get(primary_key)
    if legacy_key in cfg:
        return cfg.get(legacy_key)
    return default


def _cfg_int(cfg: Optional[dict], primary_key: str, legacy_key: str, default: int = 0) -> int:
    raw = _cfg_value(cfg, primary_key, legacy_key, default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _join_step_msg_parts(step_msg: Any) -> str:
    if step_msg is None:
        return ''
    if isinstance(step_msg, (list, tuple)):
        parts = [_join_step_msg_parts(item).strip() for item in step_msg]
        return '\n\n'.join(part for part in parts if part)
    return str(step_msg)


def _format_observation_context_field(field_name: str, value: Any) -> str:
    text = _join_step_msg_parts(value).strip()
    if not text:
        return ''
    label = OBSERVATION_CONTEXT_LABELS.get(field_name, 'Observation context')
    return f'{label}:\n{text}'


class HistoryContext:
    """Keep limited attempt/step history for obs augmentation."""

    @staticmethod
    def _strip_prompt_from_step_msg(step_msg: Any) -> Any:
        """Remove attempt-invariant prompt blocks from a step message.

        We treat <system_prompt>...</system_prompt> and <task_prompt>...</task_prompt>
        as static prompt content that should not be repeated in historical attempts.
        """
        step_msg = _join_step_msg_parts(step_msg)
        if not step_msg:
            return ''

        low = step_msg.lower()
        static_tags = ('system_prompt', 'task_prompt', 'tool_manifest', 'code_wrapper')
        if not any(f'<{tag}' in low for tag in static_tags):
            if unwrap_observation_context_blocks is not None:
                return unwrap_observation_context_blocks(step_msg).strip()
            return step_msg.strip()

        # Legacy payloads are often: optional prefix + <system_prompt>...</system_prompt> + task text.
        # The task text is static prompt content and should not be repeated as observation text.
        if '<system_prompt' in low and '<task_prompt' not in low:
            start = low.find('<system_prompt')
            return step_msg[:start].strip() if start >= 0 else ''

        # If explicit prompt/internal tags exist, strip tagged blocks and keep any remaining text.
        if any(f'<{tag}' in low for tag in static_tags):
            cleaned = step_msg
            for tag in static_tags:
                cleaned = re.sub(fr'<{tag}\b[^>]*>.*?</{tag}>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
            if unwrap_observation_context_blocks is not None:
                cleaned = unwrap_observation_context_blocks(cleaned)
            cleaned = cleaned.strip()
            return cleaned

        # Legacy format: everything after </system_prompt> is considered task prompt.
        # In that case, keep only any prefix before the system prompt start tag.
        start = low.find('<system_prompt')
        if start >= 0:
            return step_msg[:start].strip()

        return step_msg

    def __init__(self, cfg=None):
        self._episodes: Dict[int, deque] = {}
        self._current: Dict[int, List[dict]] = {}
        self._injected_samples: Dict[int, List[List[dict]]] = {}
        self._finalized_episodes: Dict[int, List[List[dict]]] = {}
        self.configure(cfg or {})

    def configure(self, cfg: dict):
        cfg = cfg or {}
        self.max_history_attempts = max(0, _cfg_int(cfg, 'max_history_attempts', 'max_episodes', 0))
        self.max_history_steps_per_attempt = max(
            0, _cfg_int(cfg, 'max_history_steps_per_attempt', 'max_steps_per_episode', 0)
        )
        self.max_episodes = self.max_history_attempts
        self.max_steps_per_episode = self.max_history_steps_per_attempt
        self.sample_mode = str(cfg.get('sample_mode', 'random')).lower()
        self.sample_size = max(1, int(cfg.get('sample_size', 1)))
        self.include_terminal = bool(cfg.get('include_terminal', True))
        self.share_across_agents = bool(cfg.get('share_across_agents', False))
        self.enabled = self.max_history_attempts > 0

        if not self.enabled:
            self._episodes.clear()
            self._current.clear()
            self._injected_samples.clear()
            self._finalized_episodes.clear()
            return

        for aid, pool in list(self._episodes.items()):
            while len(pool) > self.max_episodes:
                pool.popleft()
            if self.max_steps_per_episode > 0:
                trimmed = deque()
                for ep in pool:
                    if len(ep) > self.max_steps_per_episode:
                        ep = ep[-self.max_steps_per_episode:]
                    trimmed.append(ep)
                self._episodes[aid] = trimmed

    def start_episode(self, agent_ids, history_snapshot: Optional[Dict[int, List[List[dict]]]] = None):
        if not self.enabled:
            return
        snapshot = history_snapshot if isinstance(history_snapshot, dict) else {}
        self._injected_samples = {
            int(aid): deepcopy(snapshot.get(aid, []))
            for aid in agent_ids
        }
        for aid in agent_ids:
            self._episodes.setdefault(aid, deque())
            self._current[aid] = []

    def _finalize_episode(self, agent_id: int):
        cur = self._current.get(agent_id, [])
        if not cur:
            return
        finalized = deepcopy(cur)
        self._episodes.setdefault(agent_id, deque())
        self._episodes[agent_id].append(finalized)
        while len(self._episodes[agent_id]) > self.max_episodes:
            self._episodes[agent_id].popleft()
        self._finalized_episodes.setdefault(agent_id, []).append(finalized)
        self._current[agent_id] = []

    def record(self, agent_id: int, obs: dict, action, reward, done: bool, *, finalize: bool = True, assistant=None):
        if not self.enabled:
            return
        self._episodes.setdefault(agent_id, deque())
        self._current.setdefault(agent_id, [])

        stored_obs = deepcopy(obs)
        stored_obs.pop('history', None)
        if isinstance(stored_obs, dict) and 'step_msg' in stored_obs:
            cleaned = self._strip_prompt_from_step_msg(stored_obs.get('step_msg'))
            if isinstance(cleaned, str) and not cleaned.strip():
                stored_obs.pop('step_msg', None)
            else:
                stored_obs['step_msg'] = cleaned

        step = {
            'obs': stored_obs,
            'next_obs': None,
            'action': action,
            'assistant': assistant,
            'reward': reward,
            'done': bool(done),
        }

        cur = self._current[agent_id]
        cur.append(step)
        if self.max_steps_per_episode > 0 and len(cur) > self.max_steps_per_episode:
            cur.pop(0)

        if done and finalize:
            if not self.include_terminal:
                cur.pop()
            self._finalize_episode(agent_id)

    def record_with_next_obs(
        self,
        agent_id: int,
        obs: dict,
        next_obs: Optional[dict],
        action,
        reward,
        done: bool,
        *,
        finalize: bool = True,
        assistant=None,
        omit_next_obs_images: bool = False,
        next_obs_image_omitted_reason: Optional[str] = None,
    ):
        """Record a transition with both obs and next_obs snapshots.

        This is useful for single-attempt environments where the LLM
        benefits from seeing both pre-action and post-action visuals/messages.
        """
        if not self.enabled:
            return
        self._episodes.setdefault(agent_id, deque())
        self._current.setdefault(agent_id, [])

        stored_obs = deepcopy(obs)
        stored_obs.pop('history', None)
        if isinstance(stored_obs, dict) and 'step_msg' in stored_obs:
            cleaned = self._strip_prompt_from_step_msg(stored_obs.get('step_msg'))
            if isinstance(cleaned, str) and not cleaned.strip():
                stored_obs.pop('step_msg', None)
            else:
                stored_obs['step_msg'] = cleaned
        stored_next_obs = deepcopy(next_obs) if isinstance(next_obs, dict) else None
        if stored_next_obs is not None:
            stored_next_obs.pop('history', None)
            if isinstance(stored_next_obs, dict) and 'step_msg' in stored_next_obs:
                cleaned = self._strip_prompt_from_step_msg(stored_next_obs.get('step_msg'))
                if isinstance(cleaned, str) and not cleaned.strip():
                    stored_next_obs.pop('step_msg', None)
                else:
                    stored_next_obs['step_msg'] = cleaned

        step = {
            'obs': stored_obs,
            'next_obs': stored_next_obs,
            'action': action,
            'assistant': assistant,
            'reward': reward,
            'done': bool(done),
        }
        if omit_next_obs_images:
            step['omit_next_obs_images'] = True
            if next_obs_image_omitted_reason:
                step['next_obs_image_omitted_reason'] = str(next_obs_image_omitted_reason)

        cur = self._current[agent_id]
        cur.append(step)
        if self.max_steps_per_episode > 0 and len(cur) > self.max_steps_per_episode:
            cur.pop(0)

        if done and finalize:
            if not self.include_terminal:
                cur.pop()
            self._finalize_episode(agent_id)

    def finalize_episode(self, agent_id: int):
        """Finalize the current attempt for agent_id.

        This is used when we want terminal-step messages to still see the last transition
        in the current attempt step history before moving it into the history pool.
        """
        if not self.enabled:
            return
        cur = self._current.get(agent_id, [])
        if not cur:
            return
        if not self.include_terminal:
            try:
                cur.pop()
            except Exception:
                pass
        self._finalize_episode(agent_id)

    def current_episode(self, agent_id: int) -> List[dict]:
        """Return a snapshot of the current attempt's step history (not finalized)."""
        if not self.enabled:
            return []
        return deepcopy(self._current.get(agent_id, []))

    def _sample_episodes(self, pool: deque):
        if not pool:
            return []
        count = min(self.sample_size, len(pool))
        if self.sample_mode == 'latest':
            selected = list(pool)[-count:]
        else:
            selected = random.sample(list(pool), count)
        return deepcopy(selected)

    def sample_batch(self, agent_ids):
        if not self.enabled:
            return {aid: [] for aid in agent_ids}

        if self._injected_samples:
            return {
                aid: deepcopy(self._injected_samples.get(aid, []))
                for aid in agent_ids
            }

        if self.share_across_agents:
            merged = []
            for pool in self._episodes.values():
                merged.extend(list(pool))
            shared = self._sample_episodes(deque(merged)) if merged else []
            return {aid: deepcopy(shared) for aid in agent_ids}

        return {
            aid: self._sample_episodes(self._episodes.get(aid, deque()))
            for aid in agent_ids
        }

    def take_finalized_episodes(self) -> Dict[int, List[List[dict]]]:
        if not self.enabled:
            return {}
        out = deepcopy(self._finalized_episodes)
        self._finalized_episodes = {}
        return out


class MessageContext:
    """Build LLM-ready messages from obs + history.

    The output schema is intentionally simple and model-agnostic:
    - messages: List[{"role": "user"|"system"|"assistant", "content": str|list}]
    - Multi-modal content uses a list of parts: {"type": "text", "text": ...} and
            OpenAI-compatible: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

        image parts format can be configured via `image_part_format`:
            - 'openai' (default): uses image_url data URI parts.
            - 'raw_base64': uses {type:'image', image_base64:'...', mime_type:'image/png'} for custom clients.
    """

    def __init__(self, cfg: Optional[dict] = None, history_cfg: Optional[dict] = None):
        self.configure(cfg or {}, history_cfg=history_cfg)

    def configure(self, cfg: dict, *, history_cfg: Optional[dict] = None):
        cfg = cfg or {}
        history_limit_cfg = history_cfg if isinstance(history_cfg, dict) else cfg
        self.enabled = bool(cfg.get('enabled', False))
        self.only_return_messages = bool(cfg.get('only_return_messages', False))
        self.include_history = bool(cfg.get('include_history', True))
        self.include_current_attempt_history = bool(
            _cfg_value(cfg, 'include_current_attempt_history', 'include_current_episode_history', True)
        )
        self.include_current_episode_history = self.include_current_attempt_history
        # Normal env paths reuse HistoryContext's attempt/step limits for message rendering.
        # Standalone callers without a history_cfg can still fall back to local legacy keys.
        self.max_history_attempts = _cfg_value(
            history_limit_cfg,
            'max_history_attempts',
            'max_history_episodes',
            None,
        )
        self.max_history_steps_per_attempt = _cfg_value(
            history_limit_cfg,
            'max_history_steps_per_attempt',
            'max_history_steps_per_episode',
            None,
        )
        try:
            self.max_history_attempts = int(self.max_history_attempts) if self.max_history_attempts is not None else None
        except Exception:
            self.max_history_attempts = None
        try:
            self.max_history_steps_per_attempt = (
                int(self.max_history_steps_per_attempt) if self.max_history_steps_per_attempt is not None else None
            )
        except Exception:
            self.max_history_steps_per_attempt = None
        self.max_history_episodes = self.max_history_attempts
        self.max_history_steps_per_episode = self.max_history_steps_per_attempt
        self.include_images = bool(cfg.get('include_images', True))
        self.max_images_per_section = int(cfg.get('max_images_per_section', 2))
        self.text_max_chars = int(cfg.get('text_max_chars', 6000))

        # RL-friendly message modes:
        # - append_only: do not rebuild history each step; keep messages stable and append deltas.
        # - return_mode:
        #     - 'full': return full messages (base + appended deltas)
        #     - 'delta': return only the current step's delta message (framework maintains history)
        # - history_only_on_attempt_start: when rebuilding, include history attempts only at attempt start.
        self.append_only = bool(cfg.get('append_only', True))
        self.return_mode = str(cfg.get('return_mode', 'delta')).strip().lower()
        if self.return_mode not in ('full', 'delta'):
            self.return_mode = 'full'
        self.history_only_on_attempt_start = bool(
            _cfg_value(cfg, 'history_only_on_attempt_start', 'history_only_on_reset', False)
        )
        self.history_only_on_reset = self.history_only_on_attempt_start
        self.dedupe_initial_history_images = bool(cfg.get('dedupe_initial_history_images', False))
        self.repeat_task_prompt_on_auto_reset = bool(cfg.get('repeat_task_prompt_on_auto_reset', False))

        self.image_part_format = str(cfg.get('image_part_format', 'openai')).strip().lower()
        if self.image_part_format not in ('openai', 'raw_base64'):
            self.image_part_format = 'openai'

        # Optional: add a short text caption before each image.
        self.image_caption = bool(cfg.get('image_caption', False))
        self.image_caption_language = str(cfg.get('image_caption_language', 'en')).strip().lower()

        # If step_msg contains <system_prompt>...</system_prompt>, we can either
        # merge system text into the first user text part or emit a real
        # system-role message.
        # Default keeps backward-compat: merge into user so callers that only send user messages still work.
        self.merge_system_task_prompt = bool(cfg.get('merge_system_task_prompt', False))
        self.prefer_english_lang = bool(cfg.get('prefer_english_lang', True))

        # Fallback system prompt when we want to emit a system role but cannot parse one.
        self.default_system_prompt = str(
            cfg.get('default_system_prompt', 'You are a helpful assistant in a 3D environment.')
        )

    @staticmethod
    def merge_adjacent_text_parts(parts: Any) -> List[dict]:
        """Merge adjacent text parts while preserving non-text part order.

        This keeps multimodal ordering stable (especially around image parts)
        while reducing fragmentation when multiple text chunks are emitted
        back-to-back within one step.
        """
        if not isinstance(parts, list):
            return []

        merged: List[dict] = []
        for part in parts:
            if not isinstance(part, dict):
                continue

            if part.get('type') == 'text':
                text = part.get('text')
                if text is None:
                    text = ''
                if not isinstance(text, str):
                    text = str(text)

                if merged and isinstance(merged[-1], dict) and merged[-1].get('type') == 'text':
                    prev_text = merged[-1].get('text')
                    if not isinstance(prev_text, str):
                        prev_text = '' if prev_text is None else str(prev_text)

                    if prev_text and text:
                        if prev_text.endswith('\n') or text.startswith('\n'):
                            merged[-1]['text'] = prev_text + text
                        else:
                            merged[-1]['text'] = prev_text + '\n\n' + text
                    else:
                        merged[-1]['text'] = prev_text + text
                else:
                    merged.append({'type': 'text', 'text': text})
            else:
                merged.append(part)

        return merged

    def build_step_delta_parts(self, step: dict, step_idx: int) -> List[dict]:
        """Build append-only delta parts for a single transition.

        Delta is meant to be appended to an existing prompt+context message, or returned
        standalone in return_mode='delta'. It should not re-emit system/task prompts.
        """
        if not isinstance(step, dict):
            return []

        parts: List[dict] = []

        post_obs = step.get('next_obs') if isinstance(step.get('next_obs'), dict) else None
        show_post = (
            bool(post_obs)
            and not bool(step.get('omit_next_obs_images', False))
            and self.include_images
            and int(self.max_images_per_section) > 0
        )

        step_text = self._fmt_step(
            step,
            step_idx,
            include_next_obs=not show_post,
            next_obs_is_next_step=False,
            post_obs_shown=show_post,
            pre_obs_image_skipped=True,
        )
        if isinstance(step_text, str) and step_text:
            parts.append({'type': 'text', 'text': step_text})

        if show_post and isinstance(post_obs, dict):
            # For delta updates, prefer showing only the post-action images (new state).
            cap = max(0, int(self.max_images_per_section))
            parts.extend(self._obs_images_to_parts(
                post_obs,
                section='current_attempt',
                step_idx=step_idx,
                which_obs='post',
                limit=cap,
            ))

        return self.merge_adjacent_text_parts(parts)

    def _try_parse_system_task(self, step_msg: Any) -> tuple[Optional[str], Optional[str]]:
        if parse_system_task_prompt is None:
            return None, None
        if not isinstance(step_msg, str):
            return None, None
        if '<system_prompt>' not in step_msg.lower():
            return None, None
        try:
            system_prompt, task_prompt = parse_system_task_prompt(step_msg, prefer_english_lang=self.prefer_english_lang)
            system_prompt = system_prompt.strip() if isinstance(system_prompt, str) else None
            task_prompt = task_prompt.strip() if isinstance(task_prompt, str) else None
            return system_prompt or None, task_prompt or None
        except Exception:
            return None, None

    @staticmethod
    def _is_pil_image(obj: Any) -> bool:
        return _PILImage is not None and isinstance(obj, _PILImage.Image)

    def _make_image_part(self, img_b64: str) -> dict:
        if self.image_part_format == 'raw_base64':
            return {
                'type': 'image',
                'mime_type': 'image/png',
                'image_base64': img_b64,
            }
        # default: OpenAI-compatible
        return {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/png;base64,{img_b64}',
            },
        }

    def _make_caption_part(
        self,
        section: str,
        cam_idx: int,
        frame_idx: int,
        frame_count: int,
        episode_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
        which_obs: Optional[str] = None,
    ) -> dict:
        # Keep captions short; do not add if language is unknown.
        lang = self.image_caption_language
        if lang in ('zh', 'cn', 'chinese'):
            bits = []
            if episode_idx is not None:
                bits.append(f'历史尝试{episode_idx}')
            if step_idx is not None:
                bits.append(f'step{step_idx}')
            if which_obs:
                bits.append('动作前' if which_obs == 'pre' else ('动作后' if which_obs == 'post' else which_obs))
            bits.append(f'相机{cam_idx}')
            bits.append(f'帧{frame_idx}/{frame_count}')
            prefix = '，'.join(bits)
            text = f'[{section}] {prefix}'
        else:
            bits = []
            if section == 'history':
                bits.append(f'History attempt {episode_idx}' if episode_idx is not None else 'History')
            elif section == 'current':
                bits.append('Initial view')
            else:
                bits.append('Current attempt')

            if step_idx is not None:
                bits.append(f'Step {step_idx}')
            if which_obs:
                bits.append('Before action' if which_obs == 'pre' else ('After action' if which_obs == 'post' else which_obs))
            bits.append(f'Camera {cam_idx + 1}')
            bits.append(f'Frame {frame_idx}/{frame_count}')
            text = 'Image: [' + ' \u00b7 '.join(bits) + ']'
        return {'type': 'text', 'text': text}

    def _normalize_vis(self, vis: Any) -> List[List[Any]]:
        """Normalize obs['vis'] into List[List[frame]] where frame is PIL.Image or ndarray.

        Current EnvWrapper.post_process_obs produces:
          vis = List[camera], each camera is List[PIL.Image] (frames)
        Legacy/remote env may produce:
          vis = List[ndarray] (single-camera frames)
        """
        if not isinstance(vis, (list, tuple)):
            return []

        # If already list-of-list, keep.
        if vis and isinstance(vis[0], (list, tuple)):
            cams: List[List[Any]] = []
            for cam_frames in vis:
                if isinstance(cam_frames, (list, tuple)):
                    cams.append(list(cam_frames))
            return cams

        # Otherwise treat as a single camera with a list of frames.
        return [list(vis)]

    def _select_recent_frames(self, cams: List[List[Any]], limit: int) -> List[tuple[int, int, Any, int]]:
        """Round-robin pick recent frames across cameras, returned oldest to newest.

        Returns tuples: (cam_idx, frame_idx_1based, frame_obj, frame_count)
        """
        if limit <= 0:
            return []

        pointers = [len(frames) - 1 for frames in cams]
        selected: List[tuple[int, int, Any, int]] = []
        while len(selected) < limit and any(p >= 0 for p in pointers):
            for cam_idx, frames in enumerate(cams):
                if len(selected) >= limit:
                    break
                p = pointers[cam_idx]
                if p < 0:
                    continue
                frame_obj = frames[p]
                selected.append((cam_idx, p + 1, frame_obj, len(frames)))
                pointers[cam_idx] = p - 1
            selected.sort(key=lambda item: (item[1], item[0]))
        return selected

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if not isinstance(text, str):
            return ''
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    def _obs_images_to_parts(
        self,
        obs: dict,
        *,
        section: str,
        episode_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
        which_obs: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        if not self.include_images:
            return []
        if pil_image_to_base64 is None:
            return []

        vis = obs.get('vis', None)
        cams = self._normalize_vis(vis)
        if not cams:
            return []

        parts: List[dict] = []
        eff_limit = self.max_images_per_section if limit is None else int(limit)
        eff_limit = max(0, eff_limit)
        selected = self._select_recent_frames(cams, eff_limit)
        for cam_idx, frame_idx_1based, frame_obj, frame_count in selected:
            try:
                if self._is_pil_image(frame_obj):
                    img = frame_obj
                elif env_arr_to_pil_image is not None:
                    img = env_arr_to_pil_image(frame_obj)
                else:
                    continue

                img_b64 = pil_image_to_base64(img)
                if self.image_caption:
                    parts.append(self._make_caption_part(
                        section=section,
                        cam_idx=cam_idx,
                        frame_idx=frame_idx_1based,
                        frame_count=frame_count,
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        which_obs=which_obs,
                    ))
                parts.append(self._make_image_part(img_b64))
            except Exception:
                continue

        return parts

    def _step_image_parts(
        self,
        *,
        pre_obs: dict,
        post_obs: Optional[dict],
        section: str,
        episode_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
        skip_pre: bool = False,
    ) -> List[dict]:
        """Return image parts for a single transition, capped by max_images_per_section total."""
        if not self.include_images:
            return []
        cap = max(0, int(self.max_images_per_section))
        if cap <= 0:
            return []
        pre_cap = (cap + 1) // 2
        post_cap = cap // 2

        out: List[dict] = []
        if not skip_pre and pre_cap > 0:
            out.extend(self._obs_images_to_parts(
                pre_obs or {},
                section=section,
                episode_idx=episode_idx,
                step_idx=step_idx,
                which_obs='pre',
                limit=pre_cap,
            ))
        if isinstance(post_obs, dict) and post_cap > 0:
            out.extend(self._obs_images_to_parts(
                post_obs,
                section=section,
                episode_idx=episode_idx,
                step_idx=step_idx,
                which_obs='post',
                limit=post_cap,
            ))
        return out

    @staticmethod
    def _strip_prompt_from_step_msg(step_msg: Any) -> str:
        """Remove attempt-invariant prompt blocks from a step message."""
        step_msg = _join_step_msg_parts(step_msg)
        if not step_msg:
            return ''
        low = step_msg.lower()
        static_tags = ('system_prompt', 'task_prompt', 'tool_manifest', 'code_wrapper')
        if not any(f'<{tag}' in low for tag in static_tags):
            if unwrap_observation_context_blocks is not None:
                return unwrap_observation_context_blocks(step_msg).strip()
            return step_msg.strip()

        # Legacy payloads are often: optional prefix + <system_prompt>...</system_prompt> + task text.
        # The task text is static prompt content and should not be repeated as observation text.
        if '<system_prompt' in low and '<task_prompt' not in low:
            start = low.find('<system_prompt')
            return step_msg[:start].strip() if start >= 0 else ''

        # Strip explicit tagged blocks when present.
        cleaned = step_msg
        for tag in static_tags:
            cleaned = re.sub(fr'<{tag}\b[^>]*>.*?</{tag}>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        if unwrap_observation_context_blocks is not None:
            cleaned = unwrap_observation_context_blocks(cleaned)
        cleaned = cleaned.strip()
        # Legacy: if only system_prompt exists, keep prefix before it.
        if not cleaned and '<task_prompt' not in low:
            start = low.find('<system_prompt')
            if start >= 0:
                cleaned = step_msg[:start].strip()
        return cleaned

    def _obs_image_fingerprint(self, obs: dict) -> Optional[str]:
        """Best-effort fingerprint of the most recent frame (for dedupe)."""
        if not self.include_images or pil_image_to_base64 is None:
            return None
        if not isinstance(obs, dict):
            return None
        vis = obs.get('vis', None)
        cams = self._normalize_vis(vis)
        if not cams:
            return None
        selected = self._select_recent_frames(cams, limit=1)
        if not selected:
            return None
        _, _, frame_obj, _ = selected[0]
        try:
            if self._is_pil_image(frame_obj):
                img = frame_obj
            elif env_arr_to_pil_image is not None:
                img = env_arr_to_pil_image(frame_obj)
            else:
                return None
            img_b64 = pil_image_to_base64(img)
            if not isinstance(img_b64, str) or not img_b64:
                return None
            return hashlib.md5(img_b64.encode('utf-8')).hexdigest()
        except Exception:
            return None

    def _obs_equivalent_for_initial_dedupe(self, a: dict, b: dict) -> bool:
        """Heuristic: treat two initial observations as equivalent if text+image match."""
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        a_msg = self._observation_text_from_obs(a)
        b_msg = self._observation_text_from_obs(b)
        if a_msg and b_msg and a_msg == b_msg:
            return True
        a_fp = self._obs_image_fingerprint(a)
        b_fp = self._obs_image_fingerprint(b)
        return bool(a_fp and b_fp and a_fp == b_fp)

    def _observation_text_from_obs(self, obs: Any) -> str:
        if not isinstance(obs, dict):
            return ''

        field_parts: List[str] = []
        for field_name in OBSERVATION_CONTEXT_TAGS:
            formatted = _format_observation_context_field(field_name, obs.get(field_name, None))
            if formatted:
                field_parts.append(formatted)

        raw_step_msg = _join_step_msg_parts(obs.get('step_msg', ''))
        task_prompt = obs.get('task_prompt', None)
        if isinstance(task_prompt, str) and task_prompt and raw_step_msg.startswith(task_prompt):
            raw_step_msg = raw_step_msg[len(task_prompt):].lstrip()

        if field_parts and strip_observation_context_blocks is not None:
            raw_step_msg = strip_observation_context_blocks(raw_step_msg)

        msg = self._strip_prompt_from_step_msg(raw_step_msg)
        parts = list(field_parts)
        if msg and msg not in parts:
            parts.append(msg)
        return '\n\n'.join(parts).strip()

    def _fmt_step(
        self,
        step: dict,
        step_idx: int,
        *,
        include_next_obs: bool,
        next_obs_is_next_step: bool,
        post_obs_shown: bool,
        pre_obs_image_skipped: bool = False,
    ) -> str:
        rew = step.get('reward')
        done = step.get('done')
        action = step.get('action')

        obs_msg = ''
        obs = step.get('obs') or {}
        if isinstance(obs, dict):
            obs_msg = self._observation_text_from_obs(obs)

        next_obs_msg = ''
        next_obs = step.get('next_obs') or {}
        if isinstance(next_obs, dict):
            next_obs_msg = self._observation_text_from_obs(next_obs)

        txt = [f"Step {step_idx}: reward={rew}, done={done}"]
        if obs_msg:
            txt.append("Obs(step_msg):\n" + obs_msg)
        if action:
            txt.append("Action(code/params):\n" + str(action))

        if pre_obs_image_skipped:
            txt.append("Obs image: pre-action image omitted (same as current observation)")
        # next_obs relationship notes
        if next_obs_is_next_step:
            txt.append("Transition: post-action state is shown as the next step's Obs.")
        elif include_next_obs:
            if next_obs_msg:
                txt.append("NextObs(step_msg):\n" + next_obs_msg)
            else:
                txt.append("NextObs: (available but no step_msg)")
        elif isinstance(step.get('next_obs'), dict):
            if post_obs_shown:
                txt.append("NextObs: (shown below via post image)")
            else:
                txt.append("NextObs: (available but omitted here)")
        image_note = step.get('next_obs_image_omitted_reason', None)
        if image_note:
            txt.append("NextObs image(s): " + str(image_note))
        return "\n".join(txt)

    def _message_text_budget(self) -> List[int]:
        try:
            return [max(0, int(self.text_max_chars))]
        except Exception:
            return [0]

    def _consume_text_budget(self, text: Any, remaining_text: Optional[List[int]]) -> str:
        if text is None:
            return ''
        if not isinstance(text, str):
            text = str(text)
        if remaining_text is None:
            return text
        if not remaining_text or remaining_text[0] <= 0:
            return ''
        limit = max(0, int(remaining_text[0]))
        if len(text) > limit:
            remaining_text[0] = 0
            return self._truncate(text, limit)
        remaining_text[0] -= len(text)
        return text

    def _append_budgeted_text_part(self, parts: List[dict], text: Any, remaining_text: Optional[List[int]]) -> None:
        chunk = self._consume_text_budget(text, remaining_text)
        if chunk:
            parts.append({'type': 'text', 'text': chunk})

    def _build_assistant_message(self, step: dict) -> Optional[dict]:
        if not isinstance(step, dict):
            return None
        assistant = step.get('assistant', None)
        if assistant is None or assistant == '':
            assistant = step.get('action', None)
        if assistant is None:
            return None
        if not isinstance(assistant, str):
            assistant = str(assistant)
        return {'role': 'assistant', 'content': assistant}

    def _build_obs_user_message(
        self,
        obs: Optional[dict],
        *,
        system_prompt: Optional[str] = None,
        task_prompt: Optional[str] = None,
        prefix_lines: Optional[List[str]] = None,
        obs_label: Optional[str] = 'Observation (step_msg):',
        section: str = 'current',
        episode_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
        which_obs: Optional[str] = None,
        include_images: bool = True,
        image_limit: Optional[int] = None,
    ) -> dict:
        remaining_text = self._message_text_budget()
        parts: List[dict] = []

        if self.merge_system_task_prompt and system_prompt:
            self._append_budgeted_text_part(parts, system_prompt, remaining_text)

        for line in prefix_lines or []:
            self._append_budgeted_text_part(parts, line, remaining_text)

        if task_prompt:
            self._append_budgeted_text_part(parts, task_prompt, remaining_text)

        obs_msg = ''
        if isinstance(obs, dict):
            obs_msg = self._observation_text_from_obs(obs)
        if obs_msg:
            if obs_label:
                self._append_budgeted_text_part(parts, f"{obs_label}\n{obs_msg}", remaining_text)
            else:
                self._append_budgeted_text_part(parts, obs_msg, remaining_text)

        if include_images and isinstance(obs, dict):
            parts.extend(self._obs_images_to_parts(
                obs,
                section=section,
                episode_idx=episode_idx,
                step_idx=step_idx,
                which_obs=which_obs,
                limit=image_limit,
            ))

        parts = self.merge_adjacent_text_parts(parts)
        if not parts:
            parts = [{'type': 'text', 'text': ''}]
        return {'role': 'user', 'content': parts}

    def _build_transition_user_message(
        self,
        step: dict,
        *,
        step_idx: Optional[int] = None,
        section: str = 'current_attempt',
        episode_idx: Optional[int] = None,
    ) -> dict:
        reward = step.get('reward', None) if isinstance(step, dict) else None
        done = step.get('done', None) if isinstance(step, dict) else None
        next_obs = step.get('next_obs', None) if isinstance(step, dict) else None

        if step_idx is None:
            status_line = f"Step result: reward={reward}, done={done}"
        else:
            status_line = f"Step {step_idx} result: reward={reward}, done={done}"
        prefix_lines = [status_line]
        omit_next_obs_images = bool(step.get('omit_next_obs_images', False)) if isinstance(step, dict) else False
        image_note = step.get('next_obs_image_omitted_reason', None) if isinstance(step, dict) else None
        if omit_next_obs_images and image_note:
            prefix_lines.append("Environment response image(s): " + str(image_note))

        return self._build_obs_user_message(
            next_obs if isinstance(next_obs, dict) else {},
            prefix_lines=prefix_lines,
            obs_label='Environment response (step_msg):',
            section=section,
            episode_idx=episode_idx,
            step_idx=step_idx,
            which_obs='post',
            include_images=not omit_next_obs_images,
        )

    def _build_auto_reset_user_message(
        self,
        step: dict,
        reset_obs: dict,
        *,
        current_attempt_index: Optional[int] = None,
        next_attempt_index: Optional[int] = None,
        fallback_system_prompt: Optional[str] = None,
        fallback_task_prompt: Optional[str] = None,
        omit_terminal_obs_images: bool = False,
        terminal_obs_image_omitted_reason: Optional[str] = None,
        omit_reset_obs_images: bool = False,
        reset_obs_image_omitted_reason: Optional[str] = None,
    ) -> dict:
        reward = step.get('reward', None) if isinstance(step, dict) else None
        done = step.get('done', None) if isinstance(step, dict) else None
        terminal_obs = step.get('next_obs', None) if isinstance(step, dict) else None

        terminal_msg = ''
        if isinstance(terminal_obs, dict):
            terminal_msg = self._observation_text_from_obs(terminal_obs)

        system_prompt, task_prompt = self._try_parse_system_task(
            reset_obs.get('step_msg', '') if isinstance(reset_obs, dict) else ''
        )
        if not system_prompt and isinstance(fallback_system_prompt, str) and fallback_system_prompt.strip():
            system_prompt = fallback_system_prompt.strip()
        if not task_prompt and isinstance(fallback_task_prompt, str) and fallback_task_prompt.strip():
            task_prompt = fallback_task_prompt.strip()
        if not task_prompt and isinstance(reset_obs, dict):
            obs_task_prompt = reset_obs.get('task_prompt', None)
            if isinstance(obs_task_prompt, str) and obs_task_prompt.strip():
                task_prompt = obs_task_prompt.strip()

        prefix_lines = [f"Step result: reward={reward}, done={done}"]
        if terminal_msg:
            prefix_lines.append(f"Terminal observation (step_msg):\n{terminal_msg}")

        if current_attempt_index is not None and next_attempt_index is not None:
            reset_intro = f"Attempt {current_attempt_index} ended. Auto-reset started attempt {next_attempt_index}."
        else:
            reset_intro = 'Attempt ended. Auto-reset started a new attempt.'

        remaining_text = self._message_text_budget()
        parts: List[dict] = []

        if self.merge_system_task_prompt and system_prompt:
            self._append_budgeted_text_part(parts, system_prompt, remaining_text)

        for line in prefix_lines:
            self._append_budgeted_text_part(parts, line, remaining_text)

        if isinstance(terminal_obs, dict):
            if omit_terminal_obs_images:
                if terminal_obs_image_omitted_reason:
                    self._append_budgeted_text_part(
                        parts,
                        'Terminal observation image(s): ' + str(terminal_obs_image_omitted_reason),
                        remaining_text,
                    )
            else:
                terminal_images = self._obs_images_to_parts(
                    terminal_obs,
                    section='current_attempt',
                    step_idx=current_attempt_index,
                    which_obs='post',
                )
                if terminal_images:
                    self._append_budgeted_text_part(parts, 'Terminal observation image(s):', remaining_text)
                    parts.extend(terminal_images)

        reset_msg = ''
        if isinstance(reset_obs, dict):
            reset_msg = self._observation_text_from_obs(reset_obs)
        reset_text_blocks: List[str] = [reset_intro]
        if self.repeat_task_prompt_on_auto_reset and task_prompt:
            reset_text_blocks.append(task_prompt)
        if reset_msg and not omit_reset_obs_images:
            reset_text_blocks.append(f'New attempt observation (step_msg):\n{reset_msg}')
        if omit_reset_obs_images and reset_obs_image_omitted_reason:
            reset_text_blocks.append('New attempt initial observation text/image omitted: ' + str(reset_obs_image_omitted_reason))
        self._append_budgeted_text_part(parts, '\n\n'.join(block for block in reset_text_blocks if block), remaining_text)

        if isinstance(reset_obs, dict) and not omit_reset_obs_images:
            reset_images = self._obs_images_to_parts(
                reset_obs,
                section='current',
                step_idx=next_attempt_index,
                which_obs='pre',
            )
            if reset_images:
                self._append_budgeted_text_part(parts, 'New attempt observation image(s):', remaining_text)
                parts.extend(reset_images)

        parts = self.merge_adjacent_text_parts(parts)
        if not parts:
            parts = [{'type': 'text', 'text': ''}]
        return {'role': 'user', 'content': parts}

    def _build_attempt_transcript(
        self,
        steps: List[dict],
        *,
        section: str,
        episode_idx: Optional[int] = None,
        include_attempt_header: bool = False,
        include_task_prompt: bool = False,
        task_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        current_obs_for_dedupe: Optional[dict] = None,
    ) -> List[dict]:
        if not isinstance(steps, list) or not steps:
            return []

        messages: List[dict] = []
        first_step = steps[0] if isinstance(steps[0], dict) else {}
        first_obs = first_step.get('obs', {}) if isinstance(first_step, dict) else {}

        prefix_lines: List[str] = []
        if include_attempt_header:
            if episode_idx is not None:
                prefix_lines.append(f"Past attempt {episode_idx} start.")
            else:
                prefix_lines.append('Current attempt start.')

        skip_initial_image = False
        if (
            section == 'history'
            and self.dedupe_initial_history_images
            and isinstance(current_obs_for_dedupe, dict)
            and isinstance(first_obs, dict)
        ):
            try:
                skip_initial_image = self._obs_equivalent_for_initial_dedupe(first_obs, current_obs_for_dedupe)
            except Exception:
                skip_initial_image = False
        if skip_initial_image:
            prefix_lines.append('Initial observation image omitted (same as current attempt start).')

        messages.append(self._build_obs_user_message(
            first_obs if isinstance(first_obs, dict) else {},
            system_prompt=system_prompt,
            task_prompt=task_prompt if include_task_prompt else None,
            prefix_lines=prefix_lines,
            obs_label='Current observation (step_msg):' if include_task_prompt else 'Observation (step_msg):',
            section=section,
            episode_idx=episode_idx,
            step_idx=1,
            which_obs='pre',
            include_images=not skip_initial_image,
        ))

        for step_idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            assistant_msg = self._build_assistant_message(step)
            if assistant_msg is not None:
                messages.append(assistant_msg)
            messages.append(self._build_transition_user_message(
                step,
                step_idx=step_idx,
                section=section,
                episode_idx=episode_idx,
            ))

        return messages

    def build_chat_messages(
        self,
        current_obs: dict,
        history_episodes: List[List[dict]],
        current_episode_steps: List[dict],
        *,
        fallback_system_prompt: Optional[str] = None,
        fallback_task_prompt: Optional[str] = None,
    ) -> List[dict]:
        cur_step_msg = current_obs.get('step_msg', '') if isinstance(current_obs, dict) else ''
        system_prompt, task_prompt = self._try_parse_system_task(cur_step_msg)
        if not system_prompt and isinstance(fallback_system_prompt, str) and fallback_system_prompt.strip():
            system_prompt = fallback_system_prompt.strip()
        if not task_prompt and isinstance(fallback_task_prompt, str) and fallback_task_prompt.strip():
            task_prompt = fallback_task_prompt.strip()
        if not task_prompt and isinstance(current_obs, dict):
            obs_task_prompt = current_obs.get('task_prompt', None)
            if isinstance(obs_task_prompt, str) and obs_task_prompt.strip():
                task_prompt = obs_task_prompt.strip()

        messages: List[dict] = []
        if not self.merge_system_task_prompt:
            messages.append({'role': 'system', 'content': system_prompt or self.default_system_prompt})

        at_attempt_start = not bool(current_episode_steps)
        if self.include_history and history_episodes and (not self.history_only_on_attempt_start or at_attempt_start):
            eps_src = history_episodes
            if isinstance(self.max_history_attempts, int) and self.max_history_attempts > 0:
                eps_src = history_episodes[-self.max_history_attempts:]

            for ep_i, ep in enumerate(eps_src, start=1):
                steps = ep if isinstance(ep, list) else []
                if isinstance(self.max_history_steps_per_attempt, int) and self.max_history_steps_per_attempt > 0:
                    steps = steps[-self.max_history_steps_per_attempt:]
                messages.extend(self._build_attempt_transcript(
                    steps,
                    section='history',
                    episode_idx=ep_i,
                    include_attempt_header=True,
                    current_obs_for_dedupe=current_obs if at_attempt_start else None,
                ))

        if self.include_current_attempt_history and current_episode_steps:
            steps_src = current_episode_steps
            if isinstance(self.max_history_steps_per_attempt, int) and self.max_history_steps_per_attempt > 0:
                steps_src = current_episode_steps[-self.max_history_steps_per_attempt:]
            messages.extend(self._build_attempt_transcript(
                steps_src,
                section='current_attempt',
                include_attempt_header=True,
                include_task_prompt=True,
                task_prompt=task_prompt,
                system_prompt=system_prompt,
            ))
        else:
            messages.append(self._build_obs_user_message(
                current_obs if isinstance(current_obs, dict) else {},
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                obs_label='Current observation (step_msg):',
                section='current',
            ))

        return messages

    def build_step_messages(self, step: dict, *, step_idx: Optional[int] = None) -> List[dict]:
        messages: List[dict] = []
        assistant_msg = self._build_assistant_message(step)
        if assistant_msg is not None:
            messages.append(assistant_msg)
        messages.append(self._build_transition_user_message(step, step_idx=step_idx, section='current_attempt'))
        return messages

    def build_auto_reset_messages(
        self,
        step: dict,
        reset_obs: dict,
        *,
        current_attempt_index: Optional[int] = None,
        next_attempt_index: Optional[int] = None,
        fallback_system_prompt: Optional[str] = None,
        fallback_task_prompt: Optional[str] = None,
        omit_terminal_obs_images: bool = False,
        terminal_obs_image_omitted_reason: Optional[str] = None,
        omit_reset_obs_images: bool = False,
        reset_obs_image_omitted_reason: Optional[str] = None,
    ) -> List[dict]:
        messages: List[dict] = []
        assistant_msg = self._build_assistant_message(step)
        if assistant_msg is not None:
            messages.append(assistant_msg)
        messages.append(self._build_auto_reset_user_message(
            step,
            reset_obs,
            current_attempt_index=current_attempt_index,
            next_attempt_index=next_attempt_index,
            fallback_system_prompt=fallback_system_prompt,
            fallback_task_prompt=fallback_task_prompt,
            omit_terminal_obs_images=omit_terminal_obs_images,
            terminal_obs_image_omitted_reason=terminal_obs_image_omitted_reason,
            omit_reset_obs_images=omit_reset_obs_images,
            reset_obs_image_omitted_reason=reset_obs_image_omitted_reason,
        ))
        return messages

    def build_messages(
        self,
        current_obs: dict,
        history_episodes: List[List[dict]],
        current_episode_steps: List[dict],
        *,
        fallback_system_prompt: Optional[str] = None,
        fallback_task_prompt: Optional[str] = None,
    ) -> List[dict]:
        # Compose messages with text + images.
        cur_step_msg = current_obs.get('step_msg', '')
        system_prompt, task_prompt = self._try_parse_system_task(cur_step_msg)
        if not system_prompt and isinstance(fallback_system_prompt, str) and fallback_system_prompt.strip():
            system_prompt = fallback_system_prompt.strip()
        if not task_prompt and isinstance(fallback_task_prompt, str) and fallback_task_prompt.strip():
            task_prompt = fallback_task_prompt.strip()
        if not task_prompt and isinstance(current_obs, dict):
            obs_task_prompt = current_obs.get('task_prompt', None)
            if isinstance(obs_task_prompt, str) and obs_task_prompt.strip():
                task_prompt = obs_task_prompt.strip()

        # Build parts incrementally so history text can be placed near history images.
        parts: List[dict] = []

        remaining_text = int(self.text_max_chars)
        def _add_text(label: str, text: str):
            nonlocal remaining_text
            if remaining_text <= 0:
                return
            if not isinstance(text, str) or not text:
                return
            chunk = text
            if len(chunk) > remaining_text:
                chunk = self._truncate(chunk, remaining_text)
            parts.append({'type': 'text', 'text': chunk})
            remaining_text -= len(chunk)

        # Desired order:
        #   system prompt, task prompt, past attempts..., current attempt...
        # Prompts first (attempt-invariant).
        prompt_blocks: List[str] = []
        if self.merge_system_task_prompt and system_prompt:
            prompt_blocks.append(system_prompt)
        if task_prompt:
            prompt_blocks.append(task_prompt)
        remainder = self._observation_text_from_obs(current_obs)
        if remainder:
            prompt_blocks.append("Current observation (step_msg):\n" + str(remainder))
        _add_text('prompts', "\n\n".join(prompt_blocks))

        at_attempt_start = not bool(current_episode_steps)

        # Past history attempts (before current attempt).
        if self.include_history and history_episodes and (not self.history_only_on_attempt_start or at_attempt_start):
            eps_src = history_episodes
            if isinstance(self.max_history_attempts, int) and self.max_history_attempts > 0:
                eps_src = history_episodes[-self.max_history_attempts:]

            for ep_i, ep in enumerate(eps_src, start=1):
                steps = ep if isinstance(ep, list) else []
                if isinstance(self.max_history_steps_per_attempt, int) and self.max_history_steps_per_attempt > 0:
                    steps = steps[-self.max_history_steps_per_attempt:]

                _add_text('history_attempt_header', f"Past attempt {ep_i}: {len(steps)} step(s)")

                for si, st in enumerate(steps, start=1):
                    if not isinstance(st, dict):
                        continue
                    is_last = (si == len(steps))
                    pre_obs = st.get('obs') or {}
                    post_obs = st.get('next_obs') if isinstance(st.get('next_obs'), dict) else None
                    omit_post = bool(st.get('omit_next_obs_images', False))
                    show_post = bool(post_obs) and not omit_post and self.include_images and int(self.max_images_per_section) > 0

                    skip_pre = False
                    if self.dedupe_initial_history_images and at_attempt_start and si == 1:
                        try:
                            if self._obs_equivalent_for_initial_dedupe(pre_obs, current_obs):
                                skip_pre = True
                        except Exception:
                            skip_pre = False

                    # When we dedupe (omit) the initial image, emit an explicit short note so the
                    # user still sees *why* the image is missing even if the step text gets truncated.
                    if skip_pre:
                        _add_text(
                            'history_initial_image_dedup_note',
                            f"Note: Past attempt {ep_i} step {si} initial observation image omitted (same as current attempt start).",
                        )

                    step_text = self._fmt_step(
                        st,
                        si,
                        include_next_obs=is_last and not show_post,
                        next_obs_is_next_step=not is_last,
                        post_obs_shown=show_post,
                        pre_obs_image_skipped=skip_pre,
                    )
                    _add_text('history_step', step_text)

                    parts.extend(self._step_image_parts(
                        pre_obs=pre_obs,
                        post_obs=None if omit_post else post_obs,
                        section='history',
                        episode_idx=ep_i,
                        step_idx=si,
                        skip_pre=skip_pre,
                    ))

        if at_attempt_start:
            # Only show the baseline image(s) at attempt start.
            parts.extend(self._obs_images_to_parts(current_obs, section='current'))

        # Current attempt history text + per-step images.
        if self.include_current_attempt_history and current_episode_steps:
            steps_src = current_episode_steps
            if isinstance(self.max_history_steps_per_attempt, int) and self.max_history_steps_per_attempt > 0:
                steps_src = current_episode_steps[-self.max_history_steps_per_attempt:]

            _add_text('current_attempt_history_header', "Current attempt history:")
            for i, st in enumerate(steps_src, start=1):
                if not isinstance(st, dict):
                    continue
                is_last = (i == len(steps_src))
                post_obs = st.get('next_obs') if isinstance(st.get('next_obs'), dict) else None
                omit_post = bool(st.get('omit_next_obs_images', False))
                show_post = bool(post_obs) and not omit_post and self.include_images and int(self.max_images_per_section) > 0

                step_text = self._fmt_step(
                    st,
                    i,
                    include_next_obs=is_last and not show_post,
                    next_obs_is_next_step=not is_last,
                    post_obs_shown=show_post,
                    pre_obs_image_skipped=False,
                )
                _add_text('current_attempt_step', step_text)

                pre_obs = st.get('obs') or {}
                parts.extend(self._step_image_parts(
                    pre_obs=pre_obs,
                    post_obs=None if omit_post else post_obs,
                    section='current_attempt',
                    step_idx=i,
                    skip_pre=False,
                ))

        # Emit system role if requested and we have (or want) one.
        parts = self.merge_adjacent_text_parts(parts)

        if not self.merge_system_task_prompt:
            sys = system_prompt or self.default_system_prompt
            return [
                {'role': 'system', 'content': sys},
                {'role': 'user', 'content': parts},
            ]

        return [{'role': 'user', 'content': parts}]


class ContextManager:
    """Manage observation history and related context for EnvWrapper."""

    def __init__(self, history_cfg=None):
        self.history_ctx = HistoryContext(history_cfg or {})
        self.msg_ctx = MessageContext({})
        self._last_obs_by_lid: Dict[int, dict] = {}
        self._ml_unity_id_map: Dict[int, int] = {}
        self._episode_prompts_by_lid: Dict[int, dict] = {}
        self._pending_done_by_lid: Dict[int, bool] = {}
        # Append-only message state per logical agent id.
        # When msg_ctx.append_only is enabled, we will keep message content stable and only append step deltas.
        self._msg_state_by_lid: Dict[int, dict] = {}
        # Pending transitions captured in record_step (used when HistoryContext is disabled).
        self._pending_transitions_by_lid: Dict[int, List[dict]] = {}

    def configure(self, env_cfg: dict | None):
        wrapper_cfg = (env_cfg or {}).get('env_wrapper_cfg', {}) or {}
        # Canonical config location:
        #   env_wrapper_cfg.context_manager.{history,messages}
        cfg = wrapper_cfg.get('context_manager', {}) or {}

        history_cfg = (cfg.get('history', {}) if isinstance(cfg, dict) else {}) or {}
        self.history_ctx.configure(history_cfg)

        msg_cfg = (cfg.get('messages', {}) if isinstance(cfg, dict) else {}) or {}
        self.msg_ctx.configure(msg_cfg, history_cfg=history_cfg)

    def update_mapping(self, ml_unity_id_map: Dict[int, int]):
        self._ml_unity_id_map = ml_unity_id_map or {}

    def _logical_id(self, ml_id):
        return self._ml_unity_id_map.get(ml_id, ml_id)

    @staticmethod
    def _snapshot(obs: dict):
        snap = deepcopy(obs)
        snap.pop('history', None)
        snap.pop('messages', None)
        return snap

    def on_reset(
        self,
        env_cfg: dict,
        ml_unity_id_map: Dict[int, int],
        obs: Dict[int, dict],
        history_snapshot: Optional[Dict[int, List[List[dict]]]] = None,
    ):
        self.configure(env_cfg)
        self.update_mapping(ml_unity_id_map)
        self._last_obs_by_lid = {}
        self._episode_prompts_by_lid = {}
        self._pending_done_by_lid = {}
        self._msg_state_by_lid = {}
        self._pending_transitions_by_lid = {}

        # Cache system/task prompt from the initial reset observation (per-agent).
        if self.msg_ctx and self.msg_ctx.enabled:
            for ml_id, obs_dict in (obs or {}).items():
                if not isinstance(obs_dict, dict) or obs_dict.get('skip_infer'):
                    continue
                lid = self._logical_id(ml_id)
                try:
                    sys_p, task_p = self.msg_ctx._try_parse_system_task(obs_dict.get('step_msg', ''))
                except Exception:
                    sys_p, task_p = None, None
                if sys_p or task_p:
                    self._episode_prompts_by_lid[lid] = {
                        'system_prompt': sys_p,
                        'task_prompt': task_p,
                    }

        # Start new attempt in history (if enabled) and attach history samples.
        if self.history_ctx and self.history_ctx.enabled:
            logical_ids = list({self._logical_id(ml_id) for ml_id in obs.keys()})
            normalized_snapshot = {}
            if isinstance(history_snapshot, dict):
                normalized_snapshot = {
                    int(lid): deepcopy(history_snapshot.get(lid, []))
                    for lid in logical_ids
                }
            self.history_ctx.start_episode(logical_ids, history_snapshot=normalized_snapshot)
            obs = self._attach_history(obs)

        # Snapshot last obs for delta computation *before* messages potentially overwrite obs.
        raw_snaps = {
            self._logical_id(ml_id): self._snapshot(obs_dict)
            for ml_id, obs_dict in (obs or {}).items()
            if isinstance(obs_dict, dict)
        }

        # Attach messages (may use history samples).
        obs = self._attach_messages(obs, is_reset=True)

        # Keep raw snapshots even if only_return_messages stripped the obs.
        self._last_obs_by_lid = raw_snaps
        return obs

    def _attach_history(self, obs: Dict[int, dict]):
        if not (self.history_ctx and self.history_ctx.enabled):
            return obs
        logical_ids = {ml_id: self._logical_id(ml_id) for ml_id in obs.keys()}
        samples = self.history_ctx.sample_batch(list(logical_ids.values()))
        for ml_id, lid in logical_ids.items():
            obs[ml_id]['history'] = samples.get(lid, [])
        return obs

    def record_step(
        self,
        next_obs: Dict[int, dict],
        code_act: Any,
        reward: Dict[int, float],
        done: Dict[int, bool],
        info: Optional[Dict[str, Any]] = None,
    ):
        # We may need transitions for message deltas even when HistoryContext is disabled.
        need_transition = bool(self.msg_ctx and self.msg_ctx.enabled and self.msg_ctx.append_only)
        if not need_transition and not (self.history_ctx and self.history_ctx.enabled):
            return
        for ml_id, obs_dict in next_obs.items():
            if not isinstance(obs_dict, dict):
                continue
            is_done = bool(done.get(ml_id, False)) if isinstance(done, dict) else False
            if obs_dict.get('skip_infer') and not is_done:
                continue
            lid = self._logical_id(ml_id)
            prev_obs = self._last_obs_by_lid.get(lid)
            if prev_obs is None:
                continue
            action_text = code_act.get(ml_id) if isinstance(code_act, dict) else None
            self._pending_done_by_lid[lid] = is_done
            func_render_errors = info.get('func_render_errors', {}) if isinstance(info, dict) else {}
            has_func_render_error = False
            if isinstance(func_render_errors, dict):
                has_func_render_error = ml_id in func_render_errors or str(ml_id) in func_render_errors

            transition = {
                'obs': prev_obs,
                'next_obs': self._snapshot(obs_dict),
                'action': action_text,
                'reward': reward.get(ml_id),
                'done': is_done,
            }
            if has_func_render_error:
                transition['omit_next_obs_images'] = True
                transition['next_obs_image_omitted_reason'] = (
                    'omitted because the previous assistant tool/function-call was invalid and no Unity action was executed; '
                    'the visual state is unchanged from the previous observation.'
                )
            self._pending_transitions_by_lid.setdefault(lid, []).append(transition)

            if self.history_ctx and self.history_ctx.enabled:
                self.history_ctx.record_with_next_obs(
                    agent_id=lid,
                    obs=prev_obs,
                    next_obs=self._snapshot(obs_dict),
                    action=action_text,
                    reward=reward.get(ml_id),
                    done=is_done,
                    omit_next_obs_images=bool(transition.get('omit_next_obs_images', False)),
                    next_obs_image_omitted_reason=transition.get('next_obs_image_omitted_reason', None),
                    finalize=False,
                )

    def finalize_obs(self, obs: Dict[int, dict], done: Optional[Dict[Any, Any]] = None):
        # Snapshot raw obs before messages potentially overwrite obs.
        raw_snaps = {
            self._logical_id(ml_id): self._snapshot(obs_dict)
            for ml_id, obs_dict in (obs or {}).items()
            if isinstance(obs_dict, dict)
        }

        # Attach history samples for obs dict (if enabled) for other consumers.
        if self.history_ctx and self.history_ctx.enabled:
            obs = self._attach_history(obs)

        # Attach messages (may use append-only deltas).
        obs = self._attach_messages(obs, is_reset=False)
        # Unity ML-Agents only returns a subset of agents on each step (DecisionSteps/TerminalSteps).
        # We must *merge* updates; overwriting here would drop cached prev_obs for agents that
        # simply didn't appear this step, breaking per-agent history transitions.
        finalized_lids: List[int] = []
        for ml_id in obs.keys():
            lid = self._logical_id(ml_id)
            if done is not None:
                is_done = bool((done or {}).get(ml_id, False))
            else:
                is_done = bool(self._pending_done_by_lid.get(lid, False))
            if is_done:
                # Terminal for this agent: do not carry terminal obs into the next episode.
                self._last_obs_by_lid.pop(lid, None)
                finalized_lids.append(lid)
                continue
            snap = raw_snaps.get(lid)
            if isinstance(snap, dict):
                self._last_obs_by_lid[lid] = snap

        # Finalize done agents *after* messages are built, so terminal-step messages still
        # see the last transition in the current attempt steps (not in history attempts).
        if self.history_ctx and self.history_ctx.enabled:
            for lid in finalized_lids:
                try:
                    self.history_ctx.finalize_episode(lid)
                except Exception:
                    pass
        self._pending_done_by_lid = {}
        return obs

    def _attach_messages(self, obs: Dict[int, dict], *, is_reset: bool):
        if not (self.msg_ctx and self.msg_ctx.enabled):
            return obs

        # Append-only mode: keep a stable base message and append per-step deltas.
        if bool(getattr(self.msg_ctx, 'append_only', False)):
            for ml_id, obs_dict in obs.items():
                if not isinstance(obs_dict, dict) or obs_dict.get('skip_infer'):
                    continue
                lid = self._logical_id(ml_id)

                cached = self._episode_prompts_by_lid.get(lid, {}) if isinstance(self._episode_prompts_by_lid, dict) else {}
                fb_sys = cached.get('system_prompt') if isinstance(cached, dict) else None
                fb_task = cached.get('task_prompt') if isinstance(cached, dict) else None

                # Initialize base message on reset.
                if is_reset or lid not in self._msg_state_by_lid:
                    history_episodes = obs_dict.get('history', []) if isinstance(obs_dict, dict) else []
                    base_messages = self.msg_ctx.build_messages(
                        current_obs=obs_dict,
                        history_episodes=history_episodes if isinstance(history_episodes, list) else [],
                        current_episode_steps=[],
                        fallback_system_prompt=fb_sys,
                        fallback_task_prompt=fb_task,
                    )

                    state: dict = {
                        'steps_emitted': 0,
                        'system': None,
                        'user_parts': [],
                    }

                    if not self.msg_ctx.merge_system_task_prompt:
                        # Expected: [system, user]
                        if isinstance(base_messages, list) and len(base_messages) >= 2:
                            state['system'] = base_messages[0].get('content')
                            state['user_parts'] = deepcopy(base_messages[1].get('content') or [])
                    else:
                        if isinstance(base_messages, list) and len(base_messages) >= 1:
                            state['user_parts'] = deepcopy(base_messages[0].get('content') or [])

                    self._msg_state_by_lid[lid] = state

                    # Reset should return the base message regardless of return_mode.
                    messages_out = base_messages
                else:
                    state = self._msg_state_by_lid.get(lid, {})
                    steps_emitted = int(state.get('steps_emitted', 0) or 0)

                    # Build delta(s) since last emit.
                    new_steps: List[dict] = []
                    if self.history_ctx and self.history_ctx.enabled:
                        ep_steps = self.history_ctx.current_episode(lid)
                        if isinstance(ep_steps, list) and len(ep_steps) > steps_emitted:
                            new_steps = ep_steps[steps_emitted:]
                    else:
                        pending = self._pending_transitions_by_lid.get(lid, [])
                        if isinstance(pending, list) and pending:
                            new_steps = pending

                    delta_parts: List[dict] = []
                    for j, st in enumerate(new_steps, start=1):
                        step_idx = steps_emitted + j
                        delta_parts.extend(self.msg_ctx.build_step_delta_parts(st, step_idx))
                    delta_parts = self.msg_ctx.merge_adjacent_text_parts(delta_parts)

                    # Advance counters + clear pending.
                    if new_steps:
                        state['steps_emitted'] = steps_emitted + len(new_steps)
                    self._pending_transitions_by_lid[lid] = []

                    if self.msg_ctx.return_mode == 'delta':
                        messages_out = [{'role': 'user', 'content': delta_parts}]
                    else:
                        if delta_parts:
                            state_user_parts = state.get('user_parts')
                            if not isinstance(state_user_parts, list):
                                state_user_parts = []
                            state_user_parts.extend(delta_parts)
                            state['user_parts'] = state_user_parts
                            self._msg_state_by_lid[lid] = state

                        if not self.msg_ctx.merge_system_task_prompt:
                            sys = state.get('system') or fb_sys or self.msg_ctx.default_system_prompt
                            messages_out = [
                                {'role': 'system', 'content': sys},
                                {'role': 'user', 'content': deepcopy(state.get('user_parts') or [])},
                            ]
                        else:
                            messages_out = [{'role': 'user', 'content': deepcopy(state.get('user_parts') or [])}]

                if self.msg_ctx.only_return_messages:
                    obs[ml_id] = {'messages': messages_out, 'skip_infer': bool(obs_dict.get('skip_infer', False))}
                else:
                    obs_dict['messages'] = messages_out
            return obs

        for ml_id, obs_dict in obs.items():
            if obs_dict.get('skip_infer'):
                continue
            lid = self._logical_id(ml_id)
            history_episodes = obs_dict.get('history', []) if isinstance(obs_dict, dict) else []
            current_ep = self.history_ctx.current_episode(lid) if (self.history_ctx and self.history_ctx.enabled) else []

            cached = self._episode_prompts_by_lid.get(lid, {}) if isinstance(self._episode_prompts_by_lid, dict) else {}
            fb_sys = cached.get('system_prompt') if isinstance(cached, dict) else None
            fb_task = cached.get('task_prompt') if isinstance(cached, dict) else None
            messages = self.msg_ctx.build_messages(
                current_obs=obs_dict,
                history_episodes=history_episodes if isinstance(history_episodes, list) else [],
                current_episode_steps=current_ep,
                fallback_system_prompt=fb_sys,
                fallback_task_prompt=fb_task,
            )
            if self.msg_ctx.only_return_messages:
                # Keep wrapper simple: store messages under a single key and strip noisy fields.
                obs[ml_id] = {'messages': messages, 'skip_infer': obs_dict.get('skip_infer', False)}
            else:
                obs_dict['messages'] = messages
        return obs

    def take_finalized_episodes(self) -> Dict[int, List[List[dict]]]:
        if not self.history_ctx:
            return {}
        return self.history_ctx.take_finalized_episodes()
