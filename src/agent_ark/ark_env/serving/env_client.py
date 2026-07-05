from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import requests

from agent_ark.ark_env.serving.protocol import decode_obs
from agent_ark.ark_eval.trajectory_io import encode_history_snapshot


class EnvHttpClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    @staticmethod
    def _raise_for_status_with_detail(resp: requests.Response):
        try:
            resp.raise_for_status()
            return
        except requests.HTTPError as e:
            detail = ""
            try:
                payload = resp.json()
                detail = json.dumps(payload, ensure_ascii=False)
            except Exception:
                detail = resp.text
            raise requests.HTTPError(
                f"{e}. response_body={detail}",
                response=resp,
                request=resp.request,
            )

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._session.post(self._url(path), json=payload, timeout=self.timeout)
        self._raise_for_status_with_detail(resp)
        return resp.json()

    def _get(self, path: str) -> Dict[str, Any]:
        resp = self._session.get(self._url(path), timeout=self.timeout)
        self._raise_for_status_with_detail(resp)
        return resp.json()

    def _delete(self, path: str) -> Dict[str, Any]:
        resp = self._session.delete(self._url(path), timeout=self.timeout)
        self._raise_for_status_with_detail(resp)
        return resp.json()

    async def acreate_env(self, cfg: Dict[str, Any], env_id: Optional[str] = None) -> Dict[str, Any]:
        return await asyncio.to_thread(self._post, "/v1/envs", {"cfg": cfg, "env_id": env_id})

    async def avalidate_env_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self._post, "/v1/envs/validate", {"cfg": cfg})

    async def alist_envs(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._get, "/v1/envs")

    async def aclose_env(self, env_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._delete, f"/v1/envs/{env_id}")

    async def arelease_env(self, env_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._post, f"/v1/envs/{env_id}/release", {})

    async def astart_env(
        self,
        env_id: str,
        *,
        task_name: Optional[str] = None,
        group_seed: Optional[int] = None,
        unity_env_id: Optional[int] = None,
        history_snapshot: Optional[Dict[int, list]] = None,
        start_attempt_index: Optional[int] = None,
        uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = await asyncio.to_thread(
            self._post,
            f"/v1/envs/{env_id}/start",
            {
                "task_name": task_name,
                "group_seed": group_seed,
                "unity_env_id": unity_env_id,
                "history_snapshot": encode_history_snapshot(history_snapshot),
                "start_attempt_index": start_attempt_index,
                "uid": uid,
            },
        )
        if isinstance(payload.get("obs"), dict):
            payload["obs"] = decode_obs(payload["obs"])
        return payload

    async def astep_env(
        self,
        env_id: str,
        action: Optional[str],
        assistant: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = await asyncio.to_thread(
            self._post,
            f"/v1/envs/{env_id}/step",
            {"action": action, "assistant": assistant},
        )
        if isinstance(payload.get("obs"), dict):
            payload["obs"] = decode_obs(payload["obs"])
        return payload

    async def aacquire_start_env(
        self,
        cfg: Dict[str, Any],
        env_id: Optional[str] = None,
        *,
        task_name: Optional[str] = None,
        group_seed: Optional[int] = None,
        unity_env_id: Optional[int] = None,
        history_snapshot: Optional[Dict[int, list]] = None,
        start_attempt_index: Optional[int] = None,
        uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = await asyncio.to_thread(
            self._post,
            "/v1/envs/acquire_start",
            {
                "cfg": cfg,
                "env_id": env_id,
                "task_name": task_name,
                "group_seed": group_seed,
                "unity_env_id": unity_env_id,
                "history_snapshot": encode_history_snapshot(history_snapshot),
                "start_attempt_index": start_attempt_index,
                "uid": uid,
            },
        )
        if isinstance(payload.get("obs"), dict):
            payload["obs"] = decode_obs(payload["obs"])
        return payload

    def close(self):
        self._session.close()
