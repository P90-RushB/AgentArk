#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENTARK_REPO_ROOT="${AGENTARK_REPO_ROOT:-$(cd "$INTEGRATION_ROOT/../.." && pwd)}"
VERL_CHECKOUT="${VERL_ROOT:-}"

usage() {
  cat <<'EOF'
Usage: run_verl_training.sh [VERL launcher Hydra overrides...]

Required environment variables:
  VERL_ROOT   Path to the external VERL checkout
  MODEL_PATH Model checkpoint path
  DATA_DIR   Directory containing train.parquet and test.parquet

This wrapper preserves the external recipe as the trainer-side source of truth.
It performs compatibility and launch-safety checks, then invokes its launcher.
Pass --preflight-only as the first argument to stop before launching VERL.
EOF
}

PREFLIGHT_ONLY=false
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--preflight-only" ]]; then
  PREFLIGHT_ONLY=true
  shift
fi
if [[ -z "$VERL_CHECKOUT" ]]; then
  echo "[ERR] set VERL_ROOT to the external VERL checkout" >&2
  exit 2
fi
if [[ ! -d "$VERL_CHECKOUT" ]]; then
  echo "[ERR] VERL checkout directory not found: $VERL_CHECKOUT" >&2
  exit 2
fi
VERL_CHECKOUT="$(cd "$VERL_CHECKOUT" && pwd)"
export VERL_ROOT="$VERL_CHECKOUT"
LAUNCHER="$VERL_CHECKOUT/agentark_recipe/agentark_env_agent/run_qwen3_5_9b_agentark_env_grpo.sh"
if [[ ! -f "$LAUNCHER" ]]; then
  echo "[ERR] AgentArk VERL launcher not found: $LAUNCHER" >&2
  exit 2
fi

CHECK_PYTHON="${PYTHON:-python}"
if [[ ! -x "$CHECK_PYTHON" ]] && ! command -v "$CHECK_PYTHON" >/dev/null 2>&1; then
  echo "[ERR] VERL Python is not executable: $CHECK_PYTHON" >&2
  exit 2
fi
"$CHECK_PYTHON" "$INTEGRATION_ROOT/check_compatibility.py" --checkout "$VERL_CHECKOUT"

NNODES="${NNODES:-1}"
if [[ "$NNODES" != "1" ]]; then
  echo "[ERR] the reviewed AgentArk VERL recipe is single-node only (NNODES must be 1)" >&2
  exit 2
fi
export NNODES

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-16}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-4}"
for NAME in TRAIN_BATCH_SIZE ROLLOUT_N AGENT_NUM_WORKERS; do
  VALUE="${!NAME}"
  if [[ ! "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] $NAME must be a positive integer: $VALUE" >&2
    exit 2
  fi
done
TOTAL_TRAJECTORIES=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
if (( TOTAL_TRAJECTORIES % AGENT_NUM_WORKERS != 0 )); then
  echo "[ERR] TRAIN_BATCH_SIZE * ROLLOUT_N must be divisible by AGENT_NUM_WORKERS: " \
       "$TOTAL_TRAJECTORIES % $AGENT_NUM_WORKERS != 0" >&2
  exit 2
fi
export TRAIN_BATCH_SIZE ROLLOUT_N AGENT_NUM_WORKERS

# Current external launchers pass -1 as a concrete trainer limit, which makes
# current VERL stop after the first step. Hydra null restores VERL's
# dataloader/epoch-derived limit. Positive explicit limits remain unchanged.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:--1}"
if [[ "$TOTAL_TRAINING_STEPS" == "-1" ]]; then
  TOTAL_TRAINING_STEPS="null"
elif [[ "$TOTAL_TRAINING_STEPS" != "null" && ! "$TOTAL_TRAINING_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERR] TOTAL_TRAINING_STEPS must be a positive integer, -1, or null" >&2
  exit 2
fi
export TOTAL_TRAINING_STEPS

# Import only the five runtime paths consumed by the external recipe. This
# prevents unrelated API keys in AgentArk's .env from entering VERL/Ray or the
# launcher's xtrace output. Explicitly exported values take precedence.
AGENTARK_RUNTIME_KEYS=(
  AGENTARK_ENV_PATH
  AGENTARK_MOD_PATH
  AGENTARK_TASK_STORE_PATH
  AGENTARK_RUNTIME_TEMPLATE_ROOT
  AGENTARK_RUNTIME_POOL_ROOT
)
declare -A AGENTARK_EXPLICIT_VALUES=()
for KEY in "${AGENTARK_RUNTIME_KEYS[@]}"; do
  if [[ -v "$KEY" ]]; then
    AGENTARK_EXPLICIT_VALUES["$KEY"]="${!KEY}"
  fi
done
if [[ -f "$AGENTARK_REPO_ROOT/.env" ]]; then
  while IFS= read -r -d '' ENTRY; do
    export "$ENTRY"
  done < <(
    set -a
    # shellcheck disable=SC1091
    source "$AGENTARK_REPO_ROOT/.env"
    set +a
    for KEY in "${AGENTARK_RUNTIME_KEYS[@]}"; do
      if [[ -v "$KEY" ]]; then
        printf '%s\0' "$KEY=${!KEY}"
      fi
    done
  )
fi
for KEY in "${!AGENTARK_EXPLICIT_VALUES[@]}"; do
  export "$KEY=${AGENTARK_EXPLICIT_VALUES[$KEY]}"
done
for KEY in "${AGENTARK_RUNTIME_KEYS[@]}"; do
  if [[ ! -v "$KEY" || -z "${!KEY}" ]]; then
    echo "[ERR] missing $KEY; define it in $AGENTARK_REPO_ROOT/.env or export it" >&2
    exit 2
  fi
done

if [[ -z "${MODEL_PATH:-}" || -z "${DATA_DIR:-}" ]]; then
  echo "[ERR] MODEL_PATH and DATA_DIR are required" >&2
  usage >&2
  exit 2
fi
if [[ ! -s "$DATA_DIR/train.parquet" || ! -s "$DATA_DIR/test.parquet" ]]; then
  echo "[ERR] DATA_DIR must contain non-empty train.parquet and test.parquet: $DATA_DIR" >&2
  exit 2
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
export DATA_DIR

if [[ "$PREFLIGHT_ONLY" == true ]]; then
  echo "[OK] VERL training preflight passed"
  echo "[INFO] trajectories_per_batch=$TOTAL_TRAJECTORIES workers=$AGENT_NUM_WORKERS total_steps=$TOTAL_TRAINING_STEPS"
  exit 0
fi

# The reviewed external launcher enables shell xtrace before sourcing
# AGENT_ARK_ROOT/.env. Give it an empty, temporary root because the required
# runtime variables are already exported safely above.
SAFE_AGENTARK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/agentark-verl-root.XXXXXX")"
cleanup() {
  rmdir "$SAFE_AGENTARK_ROOT" 2>/dev/null || true
}
trap cleanup EXIT
export AGENT_ARK_ROOT="$SAFE_AGENTARK_ROOT"

echo "[INFO] Launching external AgentArk VERL recipe"
echo "[INFO] trajectories_per_batch=$TOTAL_TRAJECTORIES workers=$AGENT_NUM_WORKERS total_steps=$TOTAL_TRAINING_STEPS"
cd "$VERL_CHECKOUT"
bash "$LAUNCHER" "$@"
