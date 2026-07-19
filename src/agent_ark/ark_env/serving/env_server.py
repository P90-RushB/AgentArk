from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_ark.ark_env.op_timeout import OperationTimeout
from agent_ark.ark_env.serving.lease_protocol import LeaseProtocolError
from agent_ark.ark_env.serving.session_manager import EnvSessionManager


manager = EnvSessionManager()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    manager.start_reaper()
    try:
        yield
    finally:
        manager.shutdown()


app = FastAPI(title="AgentArk Env Server", version="0.2.0", lifespan=_lifespan)


@app.exception_handler(LeaseProtocolError)
async def _lease_protocol_error_handler(
    _request: Request,
    exc: LeaseProtocolError,
) -> JSONResponse:
    headers = {"Retry-After": "1"} if exc.retryable else None
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.as_detail()},
        headers=headers,
    )


def _v2_error_detail(code: str, message: str, *, retryable: bool) -> Dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "retryable": retryable,
    }


def _invoke_v2(operation: str, call: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return call()
    except LeaseProtocolError:
        # Handled centrally so v1 fencing errors and v2 errors have one stable
        # wire representation.
        raise
    except OperationTimeout as exc:
        logger.exception("%s timed out", operation)
        raise HTTPException(
            status_code=503,
            detail=_v2_error_detail(
                "operation_timeout",
                f"{type(exc).__name__}: {exc}",
                retryable=True,
            ),
            headers={"Retry-After": "1"},
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=_v2_error_detail(
                "invalid_request",
                str(exc),
                retryable=False,
            ),
        ) from exc
    except Exception as exc:
        logger.exception("%s failed", operation)
        raise HTTPException(
            status_code=500,
            detail=_v2_error_detail(
                "env_operation_failed",
                f"{type(exc).__name__}: {exc}",
                retryable=False,
            ),
        ) from exc


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


class AcquireStartEnvV2Request(AcquireStartEnvRequest):
    acquire_request_id: str
    client_id: Optional[str] = None


class LeaseRequestV2(BaseModel):
    server_epoch: str
    lease_id: str
    lease_generation: int


class StepEnvV2Request(LeaseRequestV2):
    action_id: str
    turn_index: int
    action: Optional[str] = None
    assistant: Optional[str] = None


class ReleaseEnvV2Request(LeaseRequestV2):
    release_request_id: str


class HeartbeatEnvV2Request(LeaseRequestV2):
    heartbeat_id: str


class HeartbeatManyItemV2(HeartbeatEnvV2Request):
    env_id: str


class HeartbeatManyV2Request(BaseModel):
    leases: List[HeartbeatManyItemV2]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "env_count": len(manager.list_envs()),
        **manager.protocol_status(),
    }


@app.get("/v2/capabilities")
def capabilities_v2() -> Dict[str, Any]:
    return manager.protocol_status()


@app.post("/v1/envs")
def create_env(req: CreateEnvRequest) -> Dict[str, Any]:
    try:
        return manager.create_env(cfg=req.cfg, env_id=req.env_id)
    except LeaseProtocolError:
        raise
    except OperationTimeout as e:
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/v1/envs/validate")
def validate_env(req: ValidateEnvRequest) -> Dict[str, Any]:
    try:
        return manager.validate_env_cfg(req.cfg)
    except LeaseProtocolError:
        raise
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
    except LeaseProtocolError:
        raise
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
    except LeaseProtocolError:
        raise
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
    except LeaseProtocolError:
        raise
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


@app.post("/v2/envs/acquire_start")
def acquire_start_env_v2(req: AcquireStartEnvV2Request) -> Dict[str, Any]:
    return _invoke_v2(
        "acquire_start_env_v2",
        lambda: manager.acquire_start_env_v2(
            cfg=req.cfg,
            acquire_request_id=req.acquire_request_id,
            client_id=req.client_id,
            env_id=req.env_id,
            task_name=req.task_name,
            group_seed=req.group_seed,
            unity_env_id=req.unity_env_id,
            history_snapshot=req.history_snapshot,
            start_attempt_index=req.start_attempt_index,
            uid=req.uid,
        ),
    )


@app.post("/v2/envs/{env_id}/step")
def step_env_v2(env_id: str, req: StepEnvV2Request) -> Dict[str, Any]:
    return _invoke_v2(
        "step_env_v2",
        lambda: manager.step_env_v2(
            env_id,
            server_epoch=req.server_epoch,
            lease_id=req.lease_id,
            lease_generation=req.lease_generation,
            action_id=req.action_id,
            turn_index=req.turn_index,
            action=req.action,
            assistant=req.assistant,
        ),
    )


@app.post("/v2/envs/{env_id}/heartbeat")
def heartbeat_env_v2(env_id: str, req: HeartbeatEnvV2Request) -> Dict[str, Any]:
    return _invoke_v2(
        "heartbeat_env_v2",
        lambda: manager.heartbeat_env_v2(
            env_id,
            server_epoch=req.server_epoch,
            lease_id=req.lease_id,
            lease_generation=req.lease_generation,
            heartbeat_id=req.heartbeat_id,
        ),
    )


@app.post("/v2/leases/heartbeat")
def heartbeat_many_v2(req: HeartbeatManyV2Request) -> Dict[str, Any]:
    leases = [
        item.model_dump() if hasattr(item, "model_dump") else item.dict()
        for item in req.leases
    ]
    return _invoke_v2(
        "heartbeat_many_v2",
        lambda: manager.heartbeat_many_v2(leases),
    )


@app.post("/v2/envs/{env_id}/release")
def release_env_v2(env_id: str, req: ReleaseEnvV2Request) -> Dict[str, Any]:
    return _invoke_v2(
        "release_env_v2",
        lambda: manager.release_env_v2(
            env_id,
            server_epoch=req.server_epoch,
            lease_id=req.lease_id,
            lease_generation=req.lease_generation,
            release_request_id=req.release_request_id,
        ),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentArk env server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
