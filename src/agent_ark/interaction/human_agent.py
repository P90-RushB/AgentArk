from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from agent_ark.agent.base_agent import BaseAgent
from agent_ark.interaction.hooks import HookManager, ensure_hook_manager
from agent_ark.interaction.local_viewer import HumanActionBroker
from agent_ark.interaction.serialization import serialize_obs_map


class HumanInteractiveAgent(BaseAgent):
    def __init__(
        self,
        *,
        name: str = 'human-local',
        action_broker: Optional[HumanActionBroker] = None,
        hooks: Optional[HookManager] = None,
        timeout_s: Optional[float] = None,
        text_max_chars: int = 6000,
        max_images_per_observation: int = 4,
    ):
        super().__init__(name)
        self.action_broker = action_broker or HumanActionBroker()
        self.hooks = ensure_hook_manager(hooks)
        self.timeout_s = timeout_s
        self.text_max_chars = int(text_max_chars)
        self.max_images_per_observation = int(max_images_per_observation)

    def reset(self, *args: Any, **kwargs: Any) -> None:
        return

    def build_request_messages(self, obs: Dict[int, dict]) -> Dict[int, Optional[list[dict]]]:
        requests: Dict[int, Optional[list[dict]]] = {}
        for agent_idx, obs_dict in (obs or {}).items():
            if isinstance(obs_dict, dict) and obs_dict.get('skip_infer'):
                requests[agent_idx] = None
                continue
            messages = obs_dict.get('messages') if isinstance(obs_dict, dict) else None
            requests[agent_idx] = deepcopy(messages) if isinstance(messages, list) else None
        return requests

    def forward_with_trace(
        self,
        obs: Dict[int, dict],
    ) -> tuple[Dict[int, Dict[str, Optional[str]]], Dict[int, Dict[str, Optional[str] | bool]]]:
        responses: Dict[int, Dict[str, Optional[str]]] = {}
        trace_by_agent: Dict[int, Dict[str, Optional[str] | bool]] = {}
        for agent_idx, obs_dict in (obs or {}).items():
            if isinstance(obs_dict, dict) and obs_dict.get('skip_infer'):
                responses[agent_idx] = {'action': None, 'assistant': None}
                trace_by_agent[agent_idx] = {
                    'skipped': True,
                    'assistant_raw': None,
                    'action_extracted': None,
                }
                continue

            self.hooks.emit(
                'human_request',
                {
                    'agent_id': int(agent_idx),
                    'obs': serialize_obs_map(
                        {agent_idx: obs_dict},
                        text_max_chars=self.text_max_chars,
                        max_images_per_observation=self.max_images_per_observation,
                    ),
                },
                source='human_agent',
            )
            action_text = self.action_broker.wait_for_action(agent_id=agent_idx, timeout=self.timeout_s)
            responses[agent_idx] = {'action': action_text, 'assistant': action_text}
            trace_by_agent[agent_idx] = {
                'assistant_raw': action_text,
                'action_extracted': action_text,
            }
            self.hooks.emit(
                'human_response',
                {'agent_id': int(agent_idx), 'action': action_text},
                source='human_agent',
            )
        return responses, trace_by_agent

    def forward_with_details(self, obs: Dict[int, dict]) -> Dict[int, Dict[str, Optional[str]]]:
        responses, _ = self.forward_with_trace(obs)
        return responses

    def forward(self, obs: Dict[int, dict]) -> Dict[int, Optional[str]]:
        return {
            agent_idx: payload.get('action') if isinstance(payload, dict) else None
            for agent_idx, payload in self.forward_with_details(obs).items()
        }
