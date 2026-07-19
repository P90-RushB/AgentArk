#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTARK_REPO_ROOT="${AGENTARK_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

if [[ ! -x "$AGENTARK_REPO_ROOT/scripts/run_env_server_mlagents.sh" ]]; then
  echo "[ERR] AgentArk checkout not found: $AGENTARK_REPO_ROOT" >&2
  exit 2
fi

# Keep AGENTARK_PYTHON_BIN consistent with the ms-swift integration while the
# shared server entry point continues to use MLAGENTS_PYTHON_BIN.
if [[ -n "${AGENTARK_PYTHON_BIN:-}" ]]; then
  export MLAGENTS_PYTHON_BIN="$AGENTARK_PYTHON_BIN"
fi

echo "[INFO] Starting the AgentArk Env Server for the VERL HTTP v1 recipe"
echo "[INFO] Keep this terminal running; execute warmup from another terminal"
exec "$AGENTARK_REPO_ROOT/scripts/run_env_server_mlagents.sh"
