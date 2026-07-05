from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_ark.ark_env.op_timeout import OperationTimeout
from agent_ark.ark_env.serving.session_manager import EnvSessionManager



app = FastAPI(title="AgentArk Env Server", version="0.1.0")
manager = EnvSessionManager()
logger = logging.getLogger(__name__)


class CreateEnvRequest(BaseModel):
    cfg: Dict[str, Any]
    env_id: Optional[str] = None


class ValidateEnvRequest(BaseModel):
    cfg: Dict[str, Any]


class StepEnvRequest(BaseModel):
    action: Optional[str] = None
    assistant: Optional[str] = None


class AcquireStartEnvRequest(BaseModel):
    cfg: Dict[str, Any]
    env_id: Optional[str] = None
    task_name: Optional[str] = None
    group_seed: Optional[int] = None
    unity_env_id: Optional[int] = None
    history_snapshot: Optional[Dict[str, Any]] = None
    start_attempt_index: Optional[int] = None
    # RL group id (e.g. verl GRPO uid). When task_name is not pinned, the server
    # deterministically maps uid -> (task_name, group_seed) so all samples in one
    # group share the same task. uid does not propagate into the env runtime.
    uid: Optional[str] = None


class StartEnvRequest(BaseModel):
    task_name: Optional[str] = None
    group_seed: Optional[int] = None
    unity_env_id: Optional[int] = None
    history_snapshot: Optional[Dict[str, Any]] = None
    start_attempt_index: Optional[int] = None
    uid: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "env_count": len(manager.list_envs())}


@app.post("/v1/envs")
def create_env(req: CreateEnvRequest) -> Dict[str, Any]:
    try:
        return manager.create_env(cfg=req.cfg, env_id=req.env_id)
    except OperationTimeout as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/v1/envs/validate")
def validate_env(req: ValidateEnvRequest) -> Dict[str, Any]:
    try:
        return manager.validate_env_cfg(req.cfg)
    except OperationTimeout as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.get("/v1/envs")
def list_envs() -> Dict[str, Any]:
    return {"items": manager.list_envs()}


@app.delete("/v1/envs/{env_id}")
def close_env(env_id: str) -> Dict[str, Any]:
    ok = manager.close_env(env_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Unknown env_id: {env_id}")
    return {"ok": True}


@app.post("/v1/envs/{env_id}/start")
def start_env(env_id: str, req: StartEnvRequest) -> Dict[str, Any]:
    try:
        return manager.start_env(
            env_id,
            task_name=req.task_name,
            group_seed=req.group_seed,
            unity_env_id=req.unity_env_id,
            history_snapshot=req.history_snapshot,
            start_attempt_index=req.start_attempt_index,
            uid=req.uid,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except OperationTimeout as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/v1/envs/{env_id}/step")
def step_env(env_id: str, req: StepEnvRequest) -> Dict[str, Any]:
    try:
        return manager.step_env(env_id=env_id, action=req.action, assistant=req.assistant)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except OperationTimeout as e:
        logger.exception("step_env timed out: env_id=%s", env_id)
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
    except Exception as e:
        logger.exception("step_env failed: env_id=%s", env_id)
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


@app.post("/v1/envs/acquire_start")
def acquire_start_env(req: AcquireStartEnvRequest) -> Dict[str, Any]:
    try:
        return manager.acquire_start_env(
            cfg=req.cfg,
            env_id=req.env_id,
            task_name=req.task_name,
            group_seed=req.group_seed,
            unity_env_id=req.unity_env_id,
            history_snapshot=req.history_snapshot,
            start_attempt_index=req.start_attempt_index,
            uid=req.uid,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except OperationTimeout as e:
        logger.exception("acquire_start_env timed out: env_id=%s", req.env_id)
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
    except Exception as e:
        logger.exception("acquire_start_env failed: env_id=%s", req.env_id)
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


@app.post("/v1/envs/{env_id}/release")
def release_env(env_id: str) -> Dict[str, Any]:
    ok = manager.release_env(env_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Unknown env_id: {env_id}")
    return {"ok": True}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentArk env server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
