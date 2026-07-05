"""Soak / stress harness for the AgentArk env server (diagnostic, read-only on code).

Purpose: before long RL training, probe whether the env can hang across many
resets / task switches / Unity restarts, and whether a stuck env is ever
recovered or just leaks out of the pool.

It does NOT modify server code. It only drives the public HTTP API
(acquire_start / step / release) under concurrency, with a *short* client
timeout so a server-side hang surfaces as a timeout we can record (instead of
blocking forever). It also polls /v1/envs to watch the in_use / env-count
curve, which reveals zombie envs (leased forever) and pool growth.

Usage (py3.10, env server already running):

    PYTHONPATH=src "$MLAGENTS_PYTHON_BIN" -m agent_ark.tools.env_soak_test \\
        --base-url http://127.0.0.1:18080 \\
        --workers 6 --rounds 200 --max-steps 6 \\
        --step-timeout 60 --acquire-timeout 120 \\
        --output tmp/env_soak_report.json

Gentle fault injection only: normal/empty/oversized actions, frequent uid churn
(=> task switches => possible Unity recreate). No inputs intentionally designed
to crash Unity.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


# ----------------------------- HTTP helpers --------------------------------- #

class _Client:
    """Minimal HTTP client with PER-CALL timeout, so a server hang -> timeout."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def post(self, path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        resp = self._session.post(self.base_url + path, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def get(self, path: str, timeout: float) -> Dict[str, Any]:
        resp = self._session.get(self.base_url + path, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


# ----------------------------- metrics -------------------------------------- #

@dataclass
class Stats:
    lock: threading.Lock = field(default_factory=threading.Lock)
    acquire_ok: int = 0
    acquire_timeout: int = 0
    acquire_error: int = 0
    step_ok: int = 0
    step_timeout: int = 0
    step_error: int = 0
    release_ok: int = 0
    release_error: int = 0
    rounds_done: int = 0
    rounds_failed: int = 0
    acquire_latencies: List[float] = field(default_factory=list)
    step_latencies: List[float] = field(default_factory=list)
    # env_ids that hit a timeout (suspected stuck); used to check later reuse.
    suspected_stuck_env_ids: List[str] = field(default_factory=list)
    errors_sample: List[str] = field(default_factory=list)

    def add_error(self, msg: str):
        with self.lock:
            if len(self.errors_sample) < 50:
                self.errors_sample.append(msg)


def _build_env_cfg() -> Dict[str, Any]:
    return {
        "env_path": os.environ["AGENTARK_ENV_PATH"],
        "mod_path": os.environ["AGENTARK_MOD_PATH"],
        "task_type": "RLTask",
        "runtime_sandbox": {
            "enabled": True,
            "runtime_platform": "linux",
            "template_root": os.environ["AGENTARK_RUNTIME_TEMPLATE_ROOT"],
            "template_env_path": os.environ["AGENTARK_ENV_PATH"],
            "template_mod_path": os.environ["AGENTARK_MOD_PATH"],
            "shared_task_store_path": os.environ["AGENTARK_TASK_STORE_PATH"],
            "pool_root": os.environ["AGENTARK_RUNTIME_POOL_ROOT"],
            "pool_size": 10,
            "auto_prepare": True,
            "link_mode": "symlink",
        },
        "env_config_overrides": {
            "num_parallel_envs": 1,
            "virtual_display": True,
        },
    }


def _random_action(rng: random.Random) -> Optional[str]:
    """Gentle action variety: normal text, empty, oversized; no crash attempts."""
    roll = rng.random()
    if roll < 0.15:
        return ""  # empty action
    if roll < 0.25:
        return "x" * rng.randint(5000, 20000)  # oversized but harmless text
    # normal-ish text action
    return rng.choice([
        "move forward",
        "<tool_call>{\"name\": \"noop\"}</tool_call>",
        "look around and act",
        "press button A",
    ])


# ----------------------------- worker loop ---------------------------------- #

def worker_loop(
    worker_id: int,
    client: _Client,
    env_cfg: Dict[str, Any],
    stats: Stats,
    *,
    rounds: int,
    max_steps: int,
    acquire_timeout: float,
    step_timeout: float,
    release_timeout: float,
    stop_flag: threading.Event,
    rng_seed: int,
):
    rng = random.Random(rng_seed)
    for r in range(rounds):
        if stop_flag.is_set():
            return
        uid = str(uuid.uuid4())  # new uid each round -> task may switch -> recreate
        env_id: Optional[str] = None
        round_ok = False
        try:
            # acquire_start
            t0 = time.time()
            try:
                started = client.post(
                    "/v1/envs/acquire_start",
                    {"cfg": env_cfg, "env_id": None, "task_name": None, "uid": uid},
                    timeout=acquire_timeout,
                )
                dt = time.time() - t0
                with stats.lock:
                    stats.acquire_ok += 1
                    stats.acquire_latencies.append(dt)
                env_id = str(started.get("env_id", "") or "")
            except requests.Timeout:
                with stats.lock:
                    stats.acquire_timeout += 1
                stats.add_error(f"[w{worker_id} r{r}] acquire TIMEOUT after {acquire_timeout}s uid={uid}")
                continue
            except Exception as e:
                with stats.lock:
                    stats.acquire_error += 1
                stats.add_error(f"[w{worker_id} r{r}] acquire ERR {type(e).__name__}: {str(e)[:200]}")
                continue

            if not env_id:
                stats.add_error(f"[w{worker_id} r{r}] empty env_id")
                continue

            # steps
            n_steps = rng.randint(1, max_steps)
            for s in range(n_steps):
                if stop_flag.is_set():
                    break
                act = _random_action(rng)
                t1 = time.time()
                try:
                    payload = client.post(
                        f"/v1/envs/{env_id}/step",
                        {"action": act, "assistant": act},
                        timeout=step_timeout,
                    )
                    with stats.lock:
                        stats.step_ok += 1
                        stats.step_latencies.append(time.time() - t1)
                    if bool(payload.get("done", False)):
                        break
                except requests.Timeout:
                    with stats.lock:
                        stats.step_timeout += 1
                        stats.suspected_stuck_env_ids.append(env_id)
                    stats.add_error(
                        f"[w{worker_id} r{r}] step TIMEOUT after {step_timeout}s env={env_id} step={s}"
                    )
                    # do NOT release: mimic a real hang; we want to see if the
                    # server recovers this env or leaks it.
                    env_id = None
                    break
                except Exception as e:
                    with stats.lock:
                        stats.step_error += 1
                    stats.add_error(
                        f"[w{worker_id} r{r}] step ERR env={env_id} {type(e).__name__}: {str(e)[:200]}"
                    )
                    break
            round_ok = True
        finally:
            if env_id:
                try:
                    client.post(f"/v1/envs/{env_id}/release", {}, timeout=release_timeout)
                    with stats.lock:
                        stats.release_ok += 1
                except Exception as e:
                    with stats.lock:
                        stats.release_error += 1
                    stats.add_error(f"[w{worker_id} r{r}] release ERR env={env_id} {type(e).__name__}: {str(e)[:150]}")
            with stats.lock:
                if round_ok:
                    stats.rounds_done += 1
                else:
                    stats.rounds_failed += 1


# ----------------------------- fault injection ------------------------------ #

def _list_unity_pids() -> List[int]:
    """Best-effort: find Unity env subprocess pids (AgentArk runtime)."""
    import subprocess
    pids: List[int] = []
    for pat in ("AgentArk", ".x86_64"):
        try:
            out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True, timeout=10)
            for line in out.stdout.split():
                try:
                    pids.append(int(line.strip()))
                except ValueError:
                    pass
        except Exception:
            pass
    return sorted(set(pids))


def fault_injector(
    stop_flag: threading.Event,
    *,
    mode: str,
    interval: float,
    log: List[Dict[str, Any]],
    freeze_seconds: float = 200.0,
):
    """Periodically disrupt a random Unity subprocess to exercise self-healing.

    - kill-unity: SIGKILL the process. mlagents detects the dead pipe and raises
      quickly, exercising the discard+recreate path on a *crashed* env.
    - stop-unity: SIGSTOP the process (freeze, not kill) so it stays alive but
      stops responding. env.step() then gets NO reply, which must be caught by
      the server-side step_timeout_s watchdog. After ``freeze_seconds`` the frozen
      pid is SIGKILLed so it does not linger as a stopped zombie.

    Only env subprocesses are targeted; the FastAPI server parent is never hit.
    """
    import os
    import signal
    if mode not in ("kill-unity", "stop-unity"):
        return
    rng = random.Random(4242)
    frozen: List[tuple] = []  # (pid, unfreeze_at_kill_time)
    stop_flag.wait(interval)
    while not stop_flag.is_set():
        # Clean up previously frozen pids: SIGKILL them so they don't linger.
        now = time.time()
        still = []
        for pid, kill_at in frozen:
            if now >= kill_at:
                try:
                    os.kill(pid, signal.SIGKILL)
                    log.append({"t": round(now, 2), "frozen_then_killed_pid": pid})
                except Exception:
                    pass
            else:
                still.append((pid, kill_at))
        frozen = still

        pids = _list_unity_pids()
        if pids:
            victim = rng.choice(pids)
            try:
                if mode == "kill-unity":
                    os.kill(victim, signal.SIGKILL)
                    log.append({"t": round(time.time(), 2), "killed_pid": victim, "alive_unity": len(pids)})
                else:  # stop-unity
                    os.kill(victim, signal.SIGSTOP)
                    frozen.append((victim, time.time() + freeze_seconds))
                    log.append({"t": round(time.time(), 2), "stopped_pid": victim, "alive_unity": len(pids)})
            except Exception as e:
                log.append({"t": round(time.time(), 2), "fault_error": f"{type(e).__name__}: {str(e)[:120]}"})
        else:
            log.append({"t": round(time.time(), 2), "note": "no unity pids found"})
        stop_flag.wait(interval)

    # Final cleanup: kill any still-frozen pids on exit.
    import os
    import signal
    for pid, _ in frozen:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


# ----------------------------- pool monitor --------------------------------- #

def monitor_pool(client: _Client, stop_flag: threading.Event, samples: List[Dict[str, Any]], interval: float):
    while not stop_flag.is_set():
        try:
            health = client.get("/health", timeout=10)
            envs = client.get("/v1/envs", timeout=10)
            items = envs.get("items", [])
            in_use = sum(1 for it in items if it.get("in_use"))
            samples.append({
                "t": round(time.time(), 2),
                "env_count": health.get("env_count", len(items)),
                "total_items": len(items),
                "in_use": in_use,
            })
        except Exception as e:
            samples.append({"t": round(time.time(), 2), "error": f"{type(e).__name__}: {str(e)[:120]}"})
        stop_flag.wait(interval)


# ----------------------------- main ----------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="AgentArk env server soak/stress test")
    p.add_argument("--base-url", default="http://127.0.0.1:18080")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--rounds", type=int, default=200, help="rounds per worker")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--acquire-timeout", type=float, default=120.0)
    p.add_argument("--step-timeout", type=float, default=60.0)
    p.add_argument("--release-timeout", type=float, default=30.0)
    p.add_argument("--monitor-interval", type=float, default=5.0)
    p.add_argument(
        "--fault-mode",
        choices=["gentle", "kill-unity", "stop-unity"],
        default="gentle",
        help="gentle: only odd actions. kill-unity: SIGKILL a random Unity "
        "subprocess (tests crashed-env recovery). stop-unity: SIGSTOP-freeze a "
        "Unity subprocess (tests the step_timeout_s watchdog for an unresponsive env).",
    )
    p.add_argument("--fault-interval", type=float, default=45.0, help="seconds between fault injections")
    p.add_argument("--output", default="tmp/env_soak_report.json")
    args = p.parse_args()

    client = _Client(args.base_url)
    env_cfg = _build_env_cfg()
    stats = Stats()
    stop_flag = threading.Event()
    pool_samples: List[Dict[str, Any]] = []
    fault_log: List[Dict[str, Any]] = []

    mon = threading.Thread(target=monitor_pool, args=(client, stop_flag, pool_samples, args.monitor_interval), daemon=True)
    mon.start()

    fault_thread = None
    if args.fault_mode != "gentle":
        fault_thread = threading.Thread(
            target=fault_injector,
            args=(stop_flag,),
            kwargs={"mode": args.fault_mode, "interval": args.fault_interval, "log": fault_log},
            daemon=True,
        )
        fault_thread.start()

    t_start = time.time()
    print(f"[soak] start workers={args.workers} rounds/worker={args.rounds} "
          f"step_timeout={args.step_timeout}s acquire_timeout={args.acquire_timeout}s "
          f"fault_mode={args.fault_mode}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                worker_loop, wid, client, env_cfg, stats,
                rounds=args.rounds, max_steps=args.max_steps,
                acquire_timeout=args.acquire_timeout, step_timeout=args.step_timeout,
                release_timeout=args.release_timeout, stop_flag=stop_flag, rng_seed=1000 + wid,
            )
            for wid in range(args.workers)
        ]
        for f in futs:
            f.result()

    stop_flag.set()
    mon.join(timeout=2)
    elapsed = time.time() - t_start

    # Post-run pool check: are suspected-stuck envs still leased / leaked?
    final_pool = None
    try:
        final_pool = client.get("/v1/envs", timeout=10)
    except Exception as e:
        final_pool = {"error": f"{type(e).__name__}: {str(e)[:120]}"}

    def _pct(xs, q):
        return round(statistics.quantiles(xs, n=100)[q - 1], 3) if len(xs) >= 100 else (round(max(xs), 3) if xs else None)

    report = {
        "config": vars(args),
        "elapsed_s": round(elapsed, 1),
        "counters": {
            "acquire_ok": stats.acquire_ok,
            "acquire_timeout": stats.acquire_timeout,
            "acquire_error": stats.acquire_error,
            "step_ok": stats.step_ok,
            "step_timeout": stats.step_timeout,
            "step_error": stats.step_error,
            "release_ok": stats.release_ok,
            "release_error": stats.release_error,
            "rounds_done": stats.rounds_done,
            "rounds_failed": stats.rounds_failed,
        },
        "latency_s": {
            "acquire_mean": round(statistics.mean(stats.acquire_latencies), 3) if stats.acquire_latencies else None,
            "acquire_p99": _pct(stats.acquire_latencies, 99),
            "acquire_max": round(max(stats.acquire_latencies), 3) if stats.acquire_latencies else None,
            "step_mean": round(statistics.mean(stats.step_latencies), 3) if stats.step_latencies else None,
            "step_p99": _pct(stats.step_latencies, 99),
            "step_max": round(max(stats.step_latencies), 3) if stats.step_latencies else None,
        },
        "suspected_stuck_env_ids": sorted(set(stats.suspected_stuck_env_ids)),
        "pool_curve": pool_samples,
        "fault_log": fault_log,
        "final_pool": final_pool,
        "errors_sample": stats.errors_sample,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[soak] DONE", flush=True)
    print(json.dumps(report["counters"], ensure_ascii=False), flush=True)
    print(json.dumps(report["latency_s"], ensure_ascii=False), flush=True)
    print(f"[soak] suspected_stuck_env_ids={report['suspected_stuck_env_ids']}", flush=True)
    if isinstance(final_pool, dict) and "items" in final_pool:
        leaked = [it for it in final_pool["items"] if it.get("in_use")]
        print(f"[soak] final pool: total={len(final_pool['items'])} still_in_use={len(leaked)}", flush=True)
    print(f"[soak] report saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
