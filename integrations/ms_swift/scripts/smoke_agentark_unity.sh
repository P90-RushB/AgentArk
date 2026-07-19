#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENTARK_REPO_ROOT="${AGENTARK_REPO_ROOT:-$(cd "$INTEGRATION_ROOT/../.." && pwd)}"
if [[ -f "$AGENTARK_REPO_ROOT/.env" ]]; then
  AGENTARK_EXPORTED_ENV_SNAPSHOT="$(export -p)"
  set -a
  # shellcheck disable=SC1091
  source "$AGENTARK_REPO_ROOT/.env"
  set +a
  eval "$AGENTARK_EXPORTED_ENV_SNAPSHOT"
  unset AGENTARK_EXPORTED_ENV_SNAPSHOT
fi
AGENTARK_PYTHON_BIN="${AGENTARK_PYTHON_BIN:-${MLAGENTS_PYTHON_BIN:-python}}"
AGENTARK_SERVER_URL="${AGENTARK_SERVER_URL:-http://127.0.0.1:18080}"
AGENTARK_PROTOCOL_VERSION="${AGENTARK_PROTOCOL_VERSION:-v2}"
AGENTARK_RUNTIME_CONFIG="${AGENTARK_RUNTIME_CONFIG:-$AGENTARK_REPO_ROOT/config/ark_env/agentark_runtime_config.example.yaml}"
SMOKE_COPIES="${AGENTARK_SMOKE_COPIES:-2}"
SMOKE_TIMEOUT="${AGENTARK_HTTP_TIMEOUT:-1200}"

if [[ ! -x "$AGENTARK_PYTHON_BIN" ]] && ! command -v "$AGENTARK_PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERR] AgentArk Python is not executable: $AGENTARK_PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -x "$AGENTARK_PYTHON_BIN" ]]; then
  AGENTARK_PYTHON_BIN="$(command -v "$AGENTARK_PYTHON_BIN")"
fi
if [[ ! -f "$AGENTARK_RUNTIME_CONFIG" ]]; then
  echo "[ERR] Runtime config not found: $AGENTARK_RUNTIME_CONFIG" >&2
  exit 2
fi

export PYTHONPATH="$AGENTARK_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export AGENTARK_PROTOCOL_VERSION

set +e
POOL_JSON="$($AGENTARK_PYTHON_BIN "$SCRIPT_DIR/check_agentark_server.py" \
  --server-url "$AGENTARK_SERVER_URL" 2>/dev/null)"
CHECK_STATUS=$?
set -e
if (( CHECK_STATUS != 0 )); then
  echo "[ERR] AgentArk server is not healthy. Start it in another terminal:" >&2
  echo "      AGENTARK_PYTHON_BIN=<agentark-python> $SCRIPT_DIR/run_agentark_server.sh" >&2
  exit 2
fi

ENV_COUNT="$($AGENTARK_PYTHON_BIN -c 'import json,sys; print(json.load(sys.stdin)["env_count"])' <<<"$POOL_JSON")"
if (( ENV_COUNT == 0 )); then
  echo "[INFO] Pool is empty; warming $SMOKE_COPIES Unity envs..."
  "$AGENTARK_PYTHON_BIN" -m agent_ark.ark_env.serving.warmup_envs \
    --config "$AGENTARK_RUNTIME_CONFIG" \
    --num-envs "$SMOKE_COPIES" \
    --protocol-version "$AGENTARK_PROTOCOL_VERSION"
elif (( ENV_COUNT < SMOKE_COPIES )); then
  echo "[ERR] Pool has $ENV_COUNT envs, fewer than smoke copies=$SMOKE_COPIES." >&2
  echo "      Restart with an empty pool and warm the requested capacity." >&2
  exit 2
fi

"$AGENTARK_PYTHON_BIN" "$SCRIPT_DIR/check_agentark_server.py" \
  --server-url "$AGENTARK_SERVER_URL" \
  --required-idle "$SMOKE_COPIES"

SMOKE_ARGS=(
  --runtime-config "$AGENTARK_RUNTIME_CONFIG"
  --server-url "$AGENTARK_SERVER_URL"
  --protocol-version "$AGENTARK_PROTOCOL_VERSION"
  --copies "$SMOKE_COPIES"
  --timeout "$SMOKE_TIMEOUT"
)
if [[ -n "${AGENTARK_SMOKE_TASK_NAME:-}" ]]; then
  SMOKE_ARGS+=(--task-name "$AGENTARK_SMOKE_TASK_NAME")
fi
if [[ -n "${AGENTARK_SMOKE_GROUP_SEED:-}" ]]; then
  SMOKE_ARGS+=(--group-seed "$AGENTARK_SMOKE_GROUP_SEED")
fi
if [[ "${AGENTARK_SMOKE_ALLOW_NO_IMAGE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--allow-no-image)
fi
if [[ -n "${AGENTARK_SMOKE_STEP_ACTION:-}" ]]; then
  SMOKE_ARGS+=(--step-action "$AGENTARK_SMOKE_STEP_ACTION")
fi

"$AGENTARK_PYTHON_BIN" "$SCRIPT_DIR/smoke_unity_group.py" "${SMOKE_ARGS[@]}"

"$AGENTARK_PYTHON_BIN" "$SCRIPT_DIR/check_agentark_server.py" \
  --server-url "$AGENTARK_SERVER_URL" \
  --required-idle "$SMOKE_COPIES"
