#!/usr/bin/env bash
set -euo pipefail

AGENTARK_ROOT="${AGENTARK_ROOT:-}"
PYTHON_BIN="${AGENTARK_PYTHON_BIN:-python3}"
DATASET="${AGENTARK_TICKET_DATASET:-snake-tickets.jsonl}"
RUN_ID="${AGENTARK_TICKET_RUN_ID:-snake-run}"
COUNT="${AGENTARK_TICKET_COUNT:-600}"
TASK_NAME="${AGENTARK_TASK_NAME:-Snake}"
GROUP_SEED_BASE="${AGENTARK_GROUP_SEED_BASE:-1234}"

usage() {
  cat >&2 <<'EOF'
usage: generate_snake_tickets.sh [options]

options:
  --repo-root PATH          AgentArk repository root
  --python PATH             Python executable
  --output PATH             JSONL output path
  --run-id ID               Ticket run identifier
  --count N                 Number of tickets/groups
  --task-name NAME          AgentArk task name
  --group-seed-base N       First group seed
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      AGENTARK_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --output)
      DATASET="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --count)
      COUNT="$2"
      shift 2
      ;;
    --task-name)
      TASK_NAME="$2"
      shift 2
      ;;
    --group-seed-base)
      GROUP_SEED_BASE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$AGENTARK_ROOT" ]]; then
  echo "[ERR] set AGENTARK_ROOT or pass --repo-root" >&2
  exit 2
fi

generator="$AGENTARK_ROOT/integrations/ms_swift/scripts/generate_tickets.py"
[[ -f "$generator" ]] || {
  echo "[ERR] ticket generator not found: $generator" >&2
  exit 2
}
[[ -x "$PYTHON_BIN" ]] || {
  echo "[ERR] Python executable is not usable: $PYTHON_BIN" >&2
  exit 2
}

mkdir -p "$(dirname "$DATASET")"
PYTHONPATH="$AGENTARK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$generator" \
  --output "$DATASET" \
  --run-id "$RUN_ID" \
  --count "$COUNT" \
  --task-name "$TASK_NAME" \
  --group-seed-base "$GROUP_SEED_BASE" \
  --force

echo "[OK] dataset: $DATASET"
