#!/bin/bash
# HPO sweep launcher: runs run_sequential_hpo.py for each (model, dataset[, combo]).
#
# Usage:
#   ./run_hpo.sh --model <deepmf|itemknn|lightgcn|featureitemknn|lightgcnm|sibrar> \
#                [--pca9] [--combos <c> [<c> ...]] \
#                --gpus <g> [<g> ...] --datasets <ds> [<ds> ...] \
#                [--pci-order] [--retry-oom] [--dry-run]
#
# Example:
#   ./run_hpo.sh --model sibrar --pca9 --combos e --gpus 0 1 --datasets onion emma
set -uo pipefail

VALID_MODELS=(deepmf itemknn lightgcn featureitemknn lightgcnm sibrar)
COMBO_MODELS=(featureitemknn lightgcnm sibrar)

usage() {
  echo "Usage: $0 --model <${VALID_MODELS[*]// /|}> [--pca9] [--combos <c> [<c> ...]] --gpus <g> [<g> ...] --datasets <ds> [<ds> ...] [--pci-order] [--retry-oom] [--dry-run]" >&2
}

MODEL=""
PCA9=0
COMBOS=()
GPUS=()
PCI_ORDER=0
RETRY_OOM=0
DRY_RUN=0
DATASETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --pca9)
      PCA9=1
      shift
      ;;
    --combos)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        COMBOS+=("$1")
        shift
      done
      ;;
    --gpus)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        GPUS+=("$1")
        shift
      done
      ;;
    --datasets)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --pci-order)
      PCI_ORDER=1
      shift
      ;;
    --retry-oom)
      RETRY_OOM=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --*)
      echo "Error: unknown flag '$1'." >&2
      usage
      exit 1
      ;;
    *)
      echo "Error: unexpected argument '$1' — datasets must be passed via --datasets." >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required." >&2
  usage
  exit 1
fi

model_valid=0
for m in "${VALID_MODELS[@]}"; do
  [[ "$MODEL" == "$m" ]] && model_valid=1
done
if [[ $model_valid -eq 0 ]]; then
  echo "Error: unknown --model '$MODEL'. Valid options: ${VALID_MODELS[*]}" >&2
  exit 1
fi

is_combo_model=0
for m in "${COMBO_MODELS[@]}"; do
  [[ "$MODEL" == "$m" ]] && is_combo_model=1
done

if [[ $is_combo_model -eq 1 ]]; then
  if [[ ${#COMBOS[@]} -eq 0 ]]; then
    echo "Error: --model $MODEL requires --combos (e.g. --combos atv atve e)." >&2
    exit 1
  fi
else
  if [[ ${#COMBOS[@]} -gt 0 ]]; then
    echo "Error: --model $MODEL does not take --combos (no modality-combo dimension)." >&2
    exit 1
  fi
  if [[ $PCA9 -eq 1 ]]; then
    echo "Error: --pca9 has no effect for --model $MODEL (no pca9 config variant exists for this model)." >&2
    exit 1
  fi
fi

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "Error: --gpus is required (e.g. --gpus 0 1 2)." >&2
  exit 1
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "Error: --datasets is required (e.g. --datasets onion emma)." >&2
  usage
  exit 1
fi

CONFIG_ROOT="${HPO_CONFIG_ROOT:-config_files}"
LOG_ROOT="${HPO_LOG_ROOT:-logs}"
PYTHON_BIN="${HPO_PYTHON_BIN:-python}"
RUNNER_SCRIPT="${HPO_RUNNER_SCRIPT:-run_sequential_hpo.py}"

if [[ $PCA9 -eq 1 ]]; then
  MODE_TAG="pca9"
  CONFIG_DIR="${CONFIG_ROOT}/pca9"
  TUNED_DIR="${CONFIG_ROOT}/tuned/pca9"
else
  MODE_TAG="full"
  CONFIG_DIR="${CONFIG_ROOT}/models"
  TUNED_DIR="${CONFIG_ROOT}/tuned/full"
fi

CONFIGS=()
for ds in "${DATASETS[@]}"; do
  if [[ $is_combo_model -eq 1 ]]; then
    for combo in "${COMBOS[@]}"; do
      CONFIGS+=("${MODEL}_${ds}_${combo}")
    done
  else
    CONFIGS+=("${MODEL}_${ds}")
  fi
done

DS_TAG=$(IFS=_; echo "${DATASETS[*]}")
if [[ $is_combo_model -eq 1 ]]; then
  COMBO_TAG=$(IFS=_; echo "${COMBOS[*]}")
  LOG_DIR_NAME="hpo_${MODE_TAG}_${MODEL}_${COMBO_TAG}_${DS_TAG}"
else
  LOG_DIR_NAME="hpo_${MODE_TAG}_${MODEL}_${DS_TAG}"
fi
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_ROOT}/${LOG_DIR_NAME}_${TS}"

RETRY_ARGS=()
if [[ $RETRY_OOM -eq 1 ]]; then
  RETRY_ARGS=(--retry_oom)
fi

log() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "$@"
  else
    echo "$@" | tee -a "$SUMMARY"
  fi
}

if [[ $DRY_RUN -eq 0 ]]; then
  mkdir -p "$LOG_DIR"
  SUMMARY="$LOG_DIR/_summary.log"
  : > "$SUMMARY"
  if [[ $PCI_ORDER -eq 1 ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
  fi
fi

log "Model: $MODEL"
log "Datasets: ${DATASETS[*]}"
if [[ $is_combo_model -eq 1 ]]; then
  log "Combos: ${COMBOS[*]}"
fi
log "GPUs: ${GPUS[*]}"
if [[ $PCI_ORDER -eq 1 ]]; then
  log "CUDA_DEVICE_ORDER: PCI_BUS_ID"
fi
if [[ $RETRY_OOM -eq 1 ]]; then
  log "Retry OOM: enabled"
fi
log "Config dir: $CONFIG_DIR"
log "Tuned dir: $TUNED_DIR"
log "Logging to $LOG_DIR"

for cfg in "${CONFIGS[@]}"; do
  tuned_out="${TUNED_DIR}/${cfg}.yml"

  if [[ -f "$tuned_out" ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      log "SKIP    $cfg  (already tuned -> $tuned_out)"
    else
      log "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP   $cfg  (already tuned -> $tuned_out)"
    fi
    continue
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    log "PENDING $cfg"
    continue
  fi

  log_file="$LOG_DIR/${cfg}.log"
  log "=================================================================="
  log "[$(date '+%Y-%m-%d %H:%M:%S')] START  $cfg  (log: $log_file)"
  log "=================================================================="

  "$PYTHON_BIN" -u "$RUNNER_SCRIPT" \
    --config "${CONFIG_DIR}/${cfg}" --gpus "${GPUS[@]}" "${RETRY_ARGS[@]}" \
    2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}

  if [[ "$status" -eq 0 ]]; then
    log "[$(date '+%Y-%m-%d %H:%M:%S')] DONE   $cfg  (exit 0)"
  else
    log "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $cfg  (exit $status)"
  fi
done

if [[ $DRY_RUN -eq 0 ]]; then
  log "=================================================================="
  log "All $MODEL HPO runs (${DATASETS[*]}) finished at $(date '+%Y-%m-%d %H:%M:%S')"
  log "=================================================================="
fi
