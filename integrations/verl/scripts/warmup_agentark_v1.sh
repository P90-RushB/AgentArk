#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENTARK_REPO_ROOT="${AGENTARK_REPO_ROOT:-$(cd "$INTEGRATION_ROOT/../.." && pwd)}"

usage() {
  cat <<'EOF'
Usage: warmup_agentark_v1.sh [options]

Options:
  --verl-root PATH       VERL checkout (or set VERL_ROOT)
  --num-envs N           v1 envs to warm (default: VERL env_cfg pool_size)
  --rendered-config PATH Resolved warmup config output
  --snapshot PATH        Warmup snapshot output
  -h, --help             Show this help
EOF
}

VERL_CHECKOUT="${VERL_ROOT:-}"
NUM_ENVS="${AGENTARK_VERL_WARMUP_NUM_ENVS:-}"
RENDERED_CONFIG="$AGENTARK_REPO_ROOT/tmp/verl_v1_runtime_config.json"
SNAPSHOT="$AGENTARK_REPO_ROOT/tmp/verl_v1_warmup_snapshot.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verl-root)
      [[ $# -ge 2 ]] || { echo "[ERR] --verl-root needs a value" >&2; exit 2; }
      VERL_CHECKOUT="$2"
      shift 2
      ;;
    --num-envs)
      [[ $# -ge 2 ]] || { echo "[ERR] --num-envs needs a value" >&2; exit 2; }
      NUM_ENVS="$2"
      shift 2
      ;;
    --rendered-config)
      [[ $# -ge 2 ]] || { echo "[ERR] --rendered-config needs a value" >&2; exit 2; }
      RENDERED_CONFIG="$2"
      shift 2
      ;;
    --snapshot)
      [[ $# -ge 2 ]] || { echo "[ERR] --snapshot needs a value" >&2; exit 2; }
      SNAPSHOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$VERL_CHECKOUT" ]]; then
  echo "[ERR] set VERL_ROOT or pass --verl-root" >&2
  exit 2
fi
if [[ ! -f "$VERL_CHECKOUT/agentark_recipe/agentark_env_agent/config/env_cfg.yaml" ]]; then
  echo "[ERR] AgentArk VERL env config not found under: $VERL_CHECKOUT" >&2
  exit 2
fi

# Load AgentArk runtime paths while preserving explicitly exported overrides.
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
if [[ ! -x "$AGENTARK_PYTHON_BIN" ]] && ! command -v "$AGENTARK_PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERR] AgentArk Python is not executable: $AGENTARK_PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -x "$AGENTARK_PYTHON_BIN" ]]; then
  AGENTARK_PYTHON_BIN="$(command -v "$AGENTARK_PYTHON_BIN")"
fi

RENDER_ARGS=(
  "$AGENTARK_PYTHON_BIN" "$INTEGRATION_ROOT/render_runtime_config.py"
  --verl-root "$VERL_CHECKOUT"
  --output "$RENDERED_CONFIG"
)
if [[ -n "$NUM_ENVS" ]]; then
  if [[ ! "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] --num-envs must be a positive integer: $NUM_ENVS" >&2
    exit 2
  fi
  RENDER_ARGS+=(--num-envs "$NUM_ENVS")
fi

echo "[INFO] Resolving the exact VERL env_cfg used by training"
"${RENDER_ARGS[@]}"

export PYTHONPATH="$AGENTARK_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[INFO] Warming the AgentArk HTTP v1 pool; the Env Server must be running in another terminal"
"$AGENTARK_PYTHON_BIN" -m agent_ark.ark_env.serving.warmup_envs \
  --config "$RENDERED_CONFIG" \
  --protocol-version v1 \
  --output "$SNAPSHOT"

echo "[OK] VERL v1 warmup complete"
echo "[INFO] Snapshot: $SNAPSHOT"
