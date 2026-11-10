#!/bin/bash
# Sequential HPO for CalibratedReranker over the given backbone(s) and dataset(s).
# Automatically infers the input_path from $RESULTS_BASE/<dataset>/recs/.
#
# Usage:
#   ./run_hpo_calibrated_reranker.sh [--retry-oom] [--calibration-metric mahalanobis|energy] <dataset> [<dataset> ...] --backbone <backbone> [<backbone> ...]
#   ./run_hpo_calibrated_reranker.sh onion --backbone LightGCN
#   ./run_hpo_calibrated_reranker.sh onion emma --backbone LightGCN DeepMF
#   ./run_hpo_calibrated_reranker.sh --retry-oom onion --backbone LightGCN   # re-scan only crashed/OOM candidates
#   ./run_hpo_calibrated_reranker.sh --calibration-metric energy emma --backbone LightGCN DeepMF ItemKNN
#       # tuned config: config_files/tuned/full/calibrated_reranker_<dataset>_<backbone>_energy.yml
set -uo pipefail

VALID_DATASETS=(onion emma)
VALID_CALIBRATION_METRICS=(mahalanobis energy)
PYTHON=${PYTHON:-python}
RESULTS_BASE="/opt/scratch/marta/results"

# ── argument parsing ──────────────────────────────────────────────────────────
DATASETS=()
BACKBONES=()
RETRY_FLAG=()
CALIBRATION_METRIC="mahalanobis"
mode="dataset"  # dataset | backbone | calibration_metric

for arg in "$@"; do
  case "$mode" in
    calibration_metric)
      CALIBRATION_METRIC="$arg"
      mode="dataset"
      continue
      ;;
  esac
  if [[ "$arg" == "--backbone" ]]; then
    mode="backbone"
  elif [[ "$arg" == "--retry-oom" ]]; then
    RETRY_FLAG=(--retry_oom)
  elif [[ "$arg" == "--calibration-metric" ]]; then
    mode="calibration_metric"
  elif [[ "$arg" == --* ]]; then
    echo "Unknown flag '$arg'." >&2
    exit 1
  elif [[ "$mode" == "backbone" ]]; then
    BACKBONES+=("$arg")
  else
    DATASETS+=("$arg")
  fi
done

if [[ ${#DATASETS[@]} -eq 0 || ${#BACKBONES[@]} -eq 0 ]]; then
  echo "Usage: $0 [--calibration-metric mahalanobis|energy] <dataset> [<dataset> ...] --backbone <backbone> [<backbone> ...]" >&2
  exit 1
fi

for ds in "${DATASETS[@]}"; do
  if [[ ! " ${VALID_DATASETS[*]} " =~ " ${ds} " ]]; then
    echo "Unknown dataset '$ds'. Valid options: ${VALID_DATASETS[*]}" >&2
    exit 1
  fi
done

if [[ ! " ${VALID_CALIBRATION_METRICS[*]} " =~ " ${CALIBRATION_METRIC} " ]]; then
  echo "Unknown --calibration-metric '$CALIBRATION_METRIC'. Valid options: ${VALID_CALIBRATION_METRICS[*]}" >&2
  exit 1
fi

# ── logging setup ─────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
DS_TAG=$(IFS=_; echo "${DATASETS[*]}")
BB_TAG=$(IFS=_; echo "${BACKBONES[*]}")
LOG_DIR="logs/hpo_calibrated_reranker_${DS_TAG}_${BB_TAG}_${TS}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/_summary.log"

echo "Datasets:  ${DATASETS[*]}" | tee "$SUMMARY"
echo "Backbones: ${BACKBONES[*]}" | tee -a "$SUMMARY"
echo "Logging to $LOG_DIR" | tee -a "$SUMMARY"

# ── helpers ───────────────────────────────────────────────────────────────────

infer_input_path() {
  local dataset="$1" backbone="$2" total_folds="${3:-10}"
  local recs_dir

  for recs_dir in "${RESULTS_BASE}/${dataset}/recs" "${RESULTS_BASE}/${dataset}/full/recs"; do
    [[ -d "$recs_dir" ]] || continue

    # Prefer fold=0 file so the reranker can substitute fold indices at runtime.
    local fold0
    fold0=$(find "$recs_dir" -maxdepth 1 -name "${backbone}_*fold=0.tsv" | sort | head -1)
    if [[ -n "$fold0" ]]; then
      # Verify all other folds exist — the reranker will replace "fold=0" in the
      # path string, so every fold file must be present before we proceed.
      local missing=()
      for (( i=1; i<total_folds; i++ )); do
        local sibling="${fold0/fold=0/fold=$i}"
        [[ -f "$sibling" ]] || missing+=("fold=$i")
      done
      if [[ ${#missing[@]} -gt 0 ]]; then
        echo "[WARN] ${backbone} recs for ${dataset} in ${recs_dir}: fold=0 found but missing: ${missing[*]}" >&2
        continue
      fi
      # $fold0 is already absolute since $recs_dir is absolute.
      echo "${fold0}"
      return
    fi

    # Fall back to any single matching file (backbone without per-fold splits).
    local single
    single=$(find "$recs_dir" -maxdepth 1 -name "${backbone}_*.tsv" | sort | head -1)
    if [[ -n "$single" ]]; then
      echo "${single}"
      return
    fi
  done

  echo ""
}

write_temp_config() {
  local dataset="$1" input_path="$2" calibration_metric="$3" cfg_name="$4"
  local base_cfg="config_files/models/calibrated_reranker_${dataset}.yml"
  # onion186 has no template of its own: it shares onion's (GEMS) setup.
  [[ -f "$base_cfg" ]] || base_cfg="config_files/models/calibrated_reranker_onion.yml"
  local tmp_cfg="config_files/models/${cfg_name}.yml"

  $PYTHON - "$base_cfg" "$tmp_cfg" "$input_path" "$calibration_metric" "$dataset" <<'PYEOF'
import sys
from utils import load_yaml, save_yaml

src, dst, new_path, calibration_metric, dataset = sys.argv[1:6]
config = load_yaml(src)
config["experiment"]["dataset"] = dataset
config["experiment"]["models"]["external.CalibratedReranker"]["input_path"] = new_path
config["experiment"]["models"]["external.CalibratedReranker"]["calibration_metric"] = calibration_metric
save_yaml(config, dst)
PYEOF
  echo "$tmp_cfg"
}

# ── main loop ─────────────────────────────────────────────────────────────────
TEMP_CONFIGS=()

for ds in "${DATASETS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    if [[ "$CALIBRATION_METRIC" == "mahalanobis" ]]; then
      cfg_name="calibrated_reranker_${ds}_${backbone,,}"
    else
      cfg_name="calibrated_reranker_${ds}_${backbone,,}_${CALIBRATION_METRIC}"
    fi
    tuned_out="config_files/tuned/full/${cfg_name}.yml"

    if [[ -f "$tuned_out" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP   ${ds}/${backbone}  (already tuned -> $tuned_out)" | tee -a "$SUMMARY"
      continue
    fi

    input_path=$(infer_input_path "$ds" "$backbone")
    if [[ -z "$input_path" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP   ${ds}/${backbone}  (no rec file found in ${RESULTS_BASE}/${ds}/recs/ or ${RESULTS_BASE}/${ds}/full/recs/ matching '${backbone}_*.tsv')" | tee -a "$SUMMARY"
      continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FOUND  ${ds}/${backbone}  -> $input_path" | tee -a "$SUMMARY"

    tmp_cfg=$(write_temp_config "$ds" "$input_path" "$CALIBRATION_METRIC" "$cfg_name")
    TEMP_CONFIGS+=("$tmp_cfg")

    log_file="$LOG_DIR/${cfg_name}.log"
    echo "==================================================================" | tee -a "$SUMMARY"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START  ${ds}/${backbone}  (log: $log_file)" | tee -a "$SUMMARY"
    echo "==================================================================" | tee -a "$SUMMARY"

    $PYTHON -u run_sequential_hpo.py \
      --config "config_files/models/${cfg_name}" --gpus 0 "${RETRY_FLAG[@]}" \
      2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE   ${ds}/${backbone}  (exit 0)" | tee -a "$SUMMARY"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED ${ds}/${backbone}  (exit $status)" | tee -a "$SUMMARY"
    fi
  done
done

# ── cleanup temp configs ──────────────────────────────────────────────────────
for tmp in "${TEMP_CONFIGS[@]}"; do
  rm -f "$tmp"
done

echo "==================================================================" | tee -a "$SUMMARY"
echo "All CalibratedReranker HPO runs finished at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SUMMARY"
echo "==================================================================" | tee -a "$SUMMARY"
