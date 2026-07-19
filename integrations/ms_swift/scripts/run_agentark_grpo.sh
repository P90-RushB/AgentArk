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
PLUGIN_PATH="${AGENTARK_SWIFT_PLUGIN:-$INTEGRATION_ROOT/src/agentark_swift/plugin.py}"
SWIFT_PYTHON_BIN="${AGENTARK_SWIFT_PYTHON:-python}"
SWIFT_BIN="${AGENTARK_SWIFT_BIN:-}"
MODEL_DIR="${AGENTARK_MODEL:-}"
TUNER_TYPE="${AGENTARK_TUNER_TYPE:-lora}"
TORCH_DTYPE="${AGENTARK_TORCH_DTYPE:-bfloat16}"
LEARNING_RATE="${AGENTARK_LEARNING_RATE:-}"
OPTIM="${AGENTARK_OPTIM:-}"
GRADIENT_CHECKPOINTING="${AGENTARK_GRADIENT_CHECKPOINTING:-}"
SAVE_ONLY_MODEL="${AGENTARK_SAVE_ONLY_MODEL:-}"
ENABLE_THINKING="${AGENTARK_ENABLE_THINKING:-}"
FREEZE_VIT="${AGENTARK_FREEZE_VIT:-}"
FREEZE_ALIGNER="${AGENTARK_FREEZE_ALIGNER:-}"
FREEZE_LLM="${AGENTARK_FREEZE_LLM:-}"

MAX_STEPS="${AGENTARK_MAX_STEPS:-1}"
PER_DEVICE_BATCH="${AGENTARK_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
WORLD_SIZE="${AGENTARK_WORLD_SIZE:-${NPROC_PER_NODE:-1}}"
GRAD_ACCUM="${AGENTARK_GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_GENERATIONS="${AGENTARK_NUM_GENERATIONS:-2}"
NUM_ITERATIONS="${AGENTARK_NUM_ITERATIONS:-1}"
GENERATION_BATCH_SIZE="${AGENTARK_GENERATION_BATCH_SIZE:-}"
TICKET_RESERVE_PERCENT="${AGENTARK_TICKET_RESERVE_PERCENT:-10}"

MAX_TURNS="${AGENTARK_MAX_TURNS:-2}"
MAX_LENGTH="${AGENTARK_MAX_LENGTH:-6144}"
MAX_COMPLETION_LENGTH="${AGENTARK_MAX_COMPLETION_LENGTH:-512}"
VLLM_MAX_MODEL_LEN="${AGENTARK_VLLM_MAX_MODEL_LEN:-$((MAX_LENGTH + MAX_COMPLETION_LENGTH))}"
VLLM_GPU_MEMORY_UTILIZATION="${AGENTARK_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
VLLM_TENSOR_PARALLEL_SIZE="${AGENTARK_VLLM_TENSOR_PARALLEL_SIZE:-1}"
AGENTARK_ASSISTANT_LOSS_SCOPE="${AGENTARK_ASSISTANT_LOSS_SCOPE:-all_turns}"

AGENTARK_SERVER_URL="${AGENTARK_SERVER_URL:-http://127.0.0.1:18080}"
AGENTARK_PROTOCOL_VERSION="${AGENTARK_PROTOCOL_VERSION:-v2}"
AGENTARK_HTTP_TIMEOUT="${AGENTARK_HTTP_TIMEOUT:-1200}"
AGENTARK_RELEASE_TIMEOUT="${AGENTARK_RELEASE_TIMEOUT:-30}"
AGENTARK_HEARTBEAT_TIMEOUT="${AGENTARK_HEARTBEAT_TIMEOUT:-5}"
AGENTARK_RUNTIME_CONFIG="${AGENTARK_RUNTIME_CONFIG:-$AGENTARK_REPO_ROOT/config/ark_env/agentark_runtime_config.example.yaml}"

# These values participate in ticket-capacity checks or define the adapter's
# validated Swift contract. Allowing a duplicate trailing CLI option would let
# argparse replace the checked value after preflight. Keep a single source of
# truth through the corresponding AGENTARK_* variables instead.
for AGENTARK_EXTRA_ARG in "$@"; do
  if [[ "$AGENTARK_EXTRA_ARG" != --* ]]; then
    continue
  fi
  AGENTARK_EXTRA_KEY="${AGENTARK_EXTRA_ARG%%=*}"
  AGENTARK_EXTRA_KEY="${AGENTARK_EXTRA_KEY#--}"
  AGENTARK_EXTRA_KEY="${AGENTARK_EXTRA_KEY//-/_}"
  case "$AGENTARK_EXTRA_KEY" in
    dataset|max_steps|per_device_train_batch_size|gradient_accumulation_steps|\
      num_generations|num_iterations|generation_batch_size|steps_per_generation|\
      sequence_parallel_size|dynamic_sample|max_resample_times|truncation_strategy|\
      model|external_plugins|multi_turn_scheduler|gym_env|use_gym_env|use_vllm|\
      vllm_mode|max_turns|loss_scale|output_dir|tuner_type|torch_dtype|\
      learning_rate|optim|gradient_checkpointing|save_only_model|enable_thinking|\
      freeze_vit|freeze_aligner|freeze_llm)
      echo "[ERR] Do not override --$AGENTARK_EXTRA_KEY through trailing CLI arguments." >&2
      echo "      Use the documented AGENTARK_* variable; this keeps preflight and Swift arguments identical." >&2
      exit 2
      ;;
  esac
done
unset AGENTARK_EXTRA_ARG AGENTARK_EXTRA_KEY

if [[ ! -x "$SWIFT_PYTHON_BIN" ]] && ! command -v "$SWIFT_PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERR] Swift Python is not executable: $SWIFT_PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -x "$SWIFT_PYTHON_BIN" ]]; then
  SWIFT_PYTHON_BIN="$(command -v "$SWIFT_PYTHON_BIN")"
fi
if [[ -z "$SWIFT_BIN" ]]; then
  SWIFT_BIN="$(dirname "$SWIFT_PYTHON_BIN")/swift"
fi
if [[ ! -x "$SWIFT_BIN" ]] && ! command -v "$SWIFT_BIN" >/dev/null 2>&1; then
  echo "[ERR] Swift CLI is not executable: $SWIFT_BIN" >&2
  exit 2
fi
if [[ -z "$MODEL_DIR" ]]; then
  echo "[ERR] Set AGENTARK_MODEL to a local model directory or a Swift-supported model ID." >&2
  exit 2
fi
if [[ ! -f "$PLUGIN_PATH" ]]; then
  echo "[ERR] AgentArk Swift plugin not found: $PLUGIN_PATH" >&2
  exit 2
fi
if [[ ! -f "$AGENTARK_RUNTIME_CONFIG" ]]; then
  echo "[ERR] AgentArk runtime config not found: $AGENTARK_RUNTIME_CONFIG" >&2
  exit 2
fi
if [[ "$AGENTARK_ASSISTANT_LOSS_SCOPE" != "all_turns" && "$AGENTARK_ASSISTANT_LOSS_SCOPE" != "last_round" ]]; then
  echo "[ERR] AGENTARK_ASSISTANT_LOSS_SCOPE must be all_turns or last_round." >&2
  exit 2
fi
if [[ "$AGENTARK_PROTOCOL_VERSION" != "v1" && "$AGENTARK_PROTOCOL_VERSION" != "v2" ]]; then
  echo "[ERR] AGENTARK_PROTOCOL_VERSION must be v1 or v2." >&2
  exit 2
fi
if [[ "$TUNER_TYPE" != "lora" && "$TUNER_TYPE" != "full" ]]; then
  echo "[ERR] AGENTARK_TUNER_TYPE must be lora or full." >&2
  exit 2
fi
for AGENTARK_BOOL_NAME in GRADIENT_CHECKPOINTING SAVE_ONLY_MODEL ENABLE_THINKING FREEZE_VIT FREEZE_ALIGNER FREEZE_LLM; do
  AGENTARK_BOOL_VALUE="${!AGENTARK_BOOL_NAME}"
  if [[ -n "$AGENTARK_BOOL_VALUE" && "$AGENTARK_BOOL_VALUE" != "true" && "$AGENTARK_BOOL_VALUE" != "false" ]]; then
    echo "[ERR] AGENTARK_${AGENTARK_BOOL_NAME} must be true, false, or unset." >&2
    exit 2
  fi
done
unset AGENTARK_BOOL_NAME AGENTARK_BOOL_VALUE
if ! [[ "$WORLD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERR] AGENTARK_WORLD_SIZE must be a positive integer." >&2
  exit 2
fi
if ! [[ "$VLLM_TENSOR_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERR] AGENTARK_VLLM_TENSOR_PARALLEL_SIZE must be a positive integer." >&2
  exit 2
fi
if [[ "${NNODES:-1}" != "1" ]]; then
  echo "[ERR] The bundled launcher supports one training node; found NNODES=${NNODES}." >&2
  exit 2
fi
if (( WORLD_SIZE % VLLM_TENSOR_PARALLEL_SIZE != 0 )); then
  echo "[ERR] AGENTARK_WORLD_SIZE must be divisible by AGENTARK_VLLM_TENSOR_PARALLEL_SIZE." >&2
  exit 2
fi

SWIFT_TUNER_ARGS=(--tuner_type "$TUNER_TYPE")
if [[ "$TUNER_TYPE" == "lora" ]]; then
  LEARNING_RATE="${LEARNING_RATE:-1e-5}"
  SWIFT_TUNER_ARGS+=(
    --lora_rank "${AGENTARK_LORA_RANK:-8}"
    --lora_alpha "${AGENTARK_LORA_ALPHA:-16}"
  )
else
  LEARNING_RATE="${LEARNING_RATE:-1e-6}"
  OPTIM="${OPTIM:-adafactor}"
  GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
fi

SWIFT_MODEL_ARGS=(--torch_dtype "$TORCH_DTYPE")
if [[ -n "$ENABLE_THINKING" ]]; then
  SWIFT_MODEL_ARGS+=(--enable_thinking "$ENABLE_THINKING")
fi
if [[ -n "$FREEZE_VIT" ]]; then
  SWIFT_MODEL_ARGS+=(--freeze_vit "$FREEZE_VIT")
fi
if [[ -n "$FREEZE_ALIGNER" ]]; then
  SWIFT_MODEL_ARGS+=(--freeze_aligner "$FREEZE_ALIGNER")
fi
if [[ -n "$FREEZE_LLM" ]]; then
  SWIFT_MODEL_ARGS+=(--freeze_llm "$FREEZE_LLM")
fi

SWIFT_OPTIM_ARGS=()
if [[ -n "$OPTIM" ]]; then
  SWIFT_OPTIM_ARGS+=(--optim "$OPTIM")
fi
if [[ -n "$GRADIENT_CHECKPOINTING" ]]; then
  SWIFT_OPTIM_ARGS+=(--gradient_checkpointing "$GRADIENT_CHECKPOINTING")
fi
if [[ -n "$SAVE_ONLY_MODEL" ]]; then
  SWIFT_OPTIM_ARGS+=(--save_only_model "$SAVE_ONLY_MODEL")
fi

SWIFT_VERSION="$($SWIFT_PYTHON_BIN -c "import importlib.metadata as m; print(m.version('ms-swift'))")"
if [[ "$SWIFT_VERSION" != "4.4.1" ]]; then
  echo "[ERR] This integration is version-guarded for ms-swift 4.4.1; found $SWIFT_VERSION." >&2
  exit 2
fi

CAPACITY_ARGS=(
  --max-steps "$MAX_STEPS"
  --per-device-train-batch-size "$PER_DEVICE_BATCH"
  --world-size "$WORLD_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --num-generations "$NUM_GENERATIONS"
  --num-iterations "$NUM_ITERATIONS"
  --reserve-percent "$TICKET_RESERVE_PERCENT"
)
SWIFT_GENERATION_ARGS=()
if [[ -n "$GENERATION_BATCH_SIZE" ]]; then
  CAPACITY_ARGS+=(--generation-batch-size "$GENERATION_BATCH_SIZE")
  SWIFT_GENERATION_ARGS+=(--generation_batch_size "$GENERATION_BATCH_SIZE")
fi

REQUIRED_TICKETS="$($SWIFT_PYTHON_BIN "$SCRIPT_DIR/check_ticket_capacity.py" \
  "${CAPACITY_ARGS[@]}" --print-required)"

RUN_ID="${AGENTARK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
GENERATED_DATA_DIR="${AGENTARK_GENERATED_DATA_DIR:-$INTEGRATION_ROOT/data/generated}"
if [[ -n "${AGENTARK_TICKET_DATASET:-}" ]]; then
  TICKET_DATASET="$AGENTARK_TICKET_DATASET"
else
  mkdir -p "$GENERATED_DATA_DIR"
  TICKET_DATASET="$GENERATED_DATA_DIR/$RUN_ID.jsonl"
  GENERATE_ARGS=(
    --output "$TICKET_DATASET"
    --run-id "$RUN_ID"
    --count "$REQUIRED_TICKETS"
  )
  if [[ -n "${AGENTARK_TASK_NAME:-}" ]]; then
    GENERATE_ARGS+=(--task-name "$AGENTARK_TASK_NAME")
  fi
  if [[ -n "${AGENTARK_GROUP_SEED:-}" ]]; then
    GENERATE_ARGS+=(--group-seed "$AGENTARK_GROUP_SEED")
  elif [[ -n "${AGENTARK_GROUP_SEED_BASE:-}" ]]; then
    GENERATE_ARGS+=(--group-seed-base "$AGENTARK_GROUP_SEED_BASE")
  fi
  "$SWIFT_PYTHON_BIN" "$SCRIPT_DIR/generate_tickets.py" "${GENERATE_ARGS[@]}"
fi

"$SWIFT_PYTHON_BIN" "$SCRIPT_DIR/check_ticket_capacity.py" \
  --dataset "$TICKET_DATASET" \
  "${CAPACITY_ARGS[@]}"

if [[ -n "$GENERATION_BATCH_SIZE" ]]; then
  REQUIRED_IDLE="$GENERATION_BATCH_SIZE"
else
  REQUIRED_IDLE=$((PER_DEVICE_BATCH * WORLD_SIZE * GRAD_ACCUM))
fi
"$SWIFT_PYTHON_BIN" "$SCRIPT_DIR/check_agentark_server.py" \
  --server-url "$AGENTARK_SERVER_URL" \
  --protocol-version "$AGENTARK_PROTOCOL_VERSION" \
  --required-idle "$REQUIRED_IDLE"

OUTPUT_DIR="${AGENTARK_OUTPUT_DIR:-$GENERATED_DATA_DIR/runs/$RUN_ID}"
mkdir -p "$OUTPUT_DIR"

export AGENTARK_SERVER_URL AGENTARK_PROTOCOL_VERSION AGENTARK_RUNTIME_CONFIG
export AGENTARK_HTTP_TIMEOUT AGENTARK_RELEASE_TIMEOUT AGENTARK_HEARTBEAT_TIMEOUT
export AGENTARK_ASSISTANT_LOSS_SCOPE
export PATH="$(dirname "$SWIFT_BIN"):$PATH"
export PYTHONPATH="$INTEGRATION_ROOT/src:${AGENTARK_SWIFT_COMPAT_DIR:-$SCRIPT_DIR/compat}${PYTHONPATH:+:$PYTHONPATH}"
export NPROC_PER_NODE="$WORLD_SIZE"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[INFO] ms-swift=$SWIFT_VERSION model=$MODEL_DIR tuner=$TUNER_TYPE dtype=$TORCH_DTYPE"
echo "[INFO] tickets=$TICKET_DATASET required_unique_groups=$REQUIRED_TICKETS"
echo "[INFO] rollout_trajectories=$REQUIRED_IDLE server=$AGENTARK_SERVER_URL"
echo "[INFO] output=$OUTPUT_DIR"

"$SWIFT_BIN" rlhf \
  --rlhf_type grpo \
  --model "$MODEL_DIR" \
  --check_model false \
  --dataset "$TICKET_DATASET" \
  --split_dataset_ratio 0 \
  --load_from_cache_file false \
  --external_plugins "$PLUGIN_PATH" \
  --multi_turn_scheduler agentark_scheduler \
  --gym_env agentark \
  --use_gym_env true \
  --max_turns "$MAX_TURNS" \
  --loss_scale default \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --sleep_level "${AGENTARK_SLEEP_LEVEL:-1}" \
  "${SWIFT_TUNER_ARGS[@]}" \
  "${SWIFT_MODEL_ARGS[@]}" \
  "${SWIFT_OPTIM_ARGS[@]}" \
  --max_length "$MAX_LENGTH" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  --save_strategy steps \
  --save_steps "${AGENTARK_SAVE_STEPS:-1}" \
  --save_total_limit "${AGENTARK_SAVE_TOTAL_LIMIT:-1}" \
  --logging_steps "${AGENTARK_LOGGING_STEPS:-1}" \
  --warmup_ratio "${AGENTARK_WARMUP_RATIO:-0}" \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --num_generations "$NUM_GENERATIONS" \
  "${SWIFT_GENERATION_ARGS[@]}" \
  --num_iterations "$NUM_ITERATIONS" \
  --temperature "${AGENTARK_TEMPERATURE:-1.0}" \
  --beta "${AGENTARK_BETA:-0}" \
  --log_completions true \
  --report_to "${AGENTARK_REPORT_TO:-none}" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
