#!/usr/bin/env bash
set -euo pipefail

AGENT_ARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env so MLAGENTS_PYTHON_BIN and friends are available to this shell
# script. Explicitly exported variables still take precedence.
if [[ -f "$AGENT_ARK_ROOT/.env" ]]; then
  AGENTARK_EXPORTED_ENV_SNAPSHOT="$(export -p)"
  set -a
  # shellcheck disable=SC1091
  source "$AGENT_ARK_ROOT/.env"
  set +a
  eval "$AGENTARK_EXPORTED_ENV_SNAPSHOT"
  unset AGENTARK_EXPORTED_ENV_SNAPSHOT
fi

MLAGENTS_PYTHON_BIN="${MLAGENTS_PYTHON_BIN:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"

if [[ ! -x "$MLAGENTS_PYTHON_BIN" ]] && ! command -v "$MLAGENTS_PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERR] MLAGENTS_PYTHON_BIN is not executable: $MLAGENTS_PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -x "$MLAGENTS_PYTHON_BIN" ]]; then
  MLAGENTS_PYTHON_BIN="$(command -v "$MLAGENTS_PYTHON_BIN")"
fi

export PYTHONPATH="$AGENT_ARK_ROOT/src:${PYTHONPATH:-}"

exec "$MLAGENTS_PYTHON_BIN" -m agent_ark.ark_env.serving.run_server --host "$HOST" --port "$PORT"
