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

if [[ ! -x "$AGENTARK_PYTHON_BIN" ]] && ! command -v "$AGENTARK_PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERR] AgentArk Python is not executable: $AGENTARK_PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -x "$AGENTARK_PYTHON_BIN" ]]; then
  AGENTARK_PYTHON_BIN="$(command -v "$AGENTARK_PYTHON_BIN")"
fi
if [[ ! -f "$AGENTARK_REPO_ROOT/src/agent_ark/ark_env/serving/run_server.py" ]]; then
  echo "[ERR] AgentArk checkout not found: $AGENTARK_REPO_ROOT" >&2
  exit 2
fi

if ! AGENTARK_SERVER_BIND="$($AGENTARK_PYTHON_BIN - "$AGENTARK_SERVER_URL" 2>/dev/null <<'PY'
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
if parsed.scheme != "http" or not parsed.hostname:
    raise SystemExit(2)
print(f"{parsed.hostname}\t{parsed.port or 80}")
PY
)"; then
  echo "[ERR] AGENTARK_SERVER_URL must be an HTTP URL with a valid host and port: $AGENTARK_SERVER_URL" >&2
  exit 2
fi
IFS=$'\t' read -r AGENTARK_SERVER_HOST AGENTARK_SERVER_PORT <<<"$AGENTARK_SERVER_BIND"
AGENTARK_SERVER_HOST="${HOST:-$AGENTARK_SERVER_HOST}"
AGENTARK_SERVER_PORT="${PORT:-$AGENTARK_SERVER_PORT}"
unset AGENTARK_SERVER_BIND

export MLAGENTS_PYTHON_BIN="$AGENTARK_PYTHON_BIN"
export PYTHONPATH="$AGENTARK_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[INFO] AgentArk Env Server bind=$AGENTARK_SERVER_HOST:$AGENTARK_SERVER_PORT client_url=$AGENTARK_SERVER_URL"
exec "$AGENTARK_PYTHON_BIN" -m agent_ark.ark_env.serving.run_server \
  --host "$AGENTARK_SERVER_HOST" \
  --port "$AGENTARK_SERVER_PORT"
