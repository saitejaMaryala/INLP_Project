#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_deadline_pipeline.sh
# Optional env vars:
#   PYTHON_EXE=/path/to/python
#   BIAS_DATA_DIR=data/bias_in_bios
#   JIGSAW_DATA_DIR=data/jigsaw_uni
#   JIGSAW_CKPT=outputs/fp32_classifier_uni.pt

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

resolve_python() {
  if [[ -n "${PYTHON_EXE:-}" ]]; then
    echo "$PYTHON_EXE"
    return 0
  fi

  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
    return 0
  fi

  # Git Bash / MSYS on Windows can execute python.exe directly.
  if [[ -f ".venv/Scripts/python.exe" ]]; then
    echo ".venv/Scripts/python.exe"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  die "Could not locate Python. Set PYTHON_EXE explicitly."
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    rm -rf "$dst"
    cp -r "$src" "$dst"
  fi
}

PYTHON_EXE="$(resolve_python)"
BIAS_DATA_DIR="${BIAS_DATA_DIR:-data/bias_in_bios}"
JIGSAW_DATA_DIR="${JIGSAW_DATA_DIR:-data/jigsaw_uni}"
JIGSAW_CKPT="${JIGSAW_CKPT:-outputs/fp32_classifier_uni.pt}"
RUN_STAMP="$(timestamp)"
RUN_ROOT="results/pipeline_runs/$RUN_STAMP"
BIAS_RUN_ROOT="$RUN_ROOT/bias"
JIGSAW_RUN_ROOT="$RUN_ROOT/jigsaw"

mkdir -p "$BIAS_RUN_ROOT" "$JIGSAW_RUN_ROOT"

[[ -d "$BIAS_DATA_DIR" ]] || die "Bias data directory not found: $BIAS_DATA_DIR"
[[ -f "$JIGSAW_DATA_DIR/train.csv" ]] || die "Jigsaw train.csv not found at: $JIGSAW_DATA_DIR/train.csv"

if [[ ! -f "$JIGSAW_CKPT" ]]; then
  echo "[WARN] Jigsaw FP32 checkpoint missing at: $JIGSAW_CKPT"
  echo "[INFO] Bootstrapping Jigsaw FP32 baseline once..."
  "$PYTHON_EXE" -m pretrained.unbert_ju.model
fi
[[ -f "$JIGSAW_CKPT" ]] || die "Jigsaw checkpoint missing after bootstrap: $JIGSAW_CKPT"

echo "[INFO] Project root : $ROOT_DIR"
echo "[INFO] Python       : $PYTHON_EXE"
echo "[INFO] Bias data    : $BIAS_DATA_DIR"
echo "[INFO] Jigsaw data  : $JIGSAW_DATA_DIR"
echo "[INFO] Jigsaw ckpt  : $JIGSAW_CKPT"
echo "[INFO] Run root     : $RUN_ROOT"

run_bias() {
  local mode="$1"
  local out_dir="$BIAS_RUN_ROOT/$mode"
  mkdir -p "$out_dir"

  echo
  echo "=== Bias-in-Bios | mode=$mode ==="
  "$PYTHON_EXE" -m training.compare_quant_bios \
    --data-dir "$BIAS_DATA_DIR" \
    --results-dir "$out_dir" \
    --int8-mode dynamic \
    --score-calibration isotonic \
    --froc-eps 0.02 \
    --froc-mode "$mode"
}

JIGSAW_FILE="pretrained/unbert_ju/compare_quantized.py"
JIGSAW_BACKUP="$(mktemp)"
cp "$JIGSAW_FILE" "$JIGSAW_BACKUP"

restore_jigsaw_file() {
  cp "$JIGSAW_BACKUP" "$JIGSAW_FILE"
  rm -f "$JIGSAW_BACKUP"
}
trap restore_jigsaw_file EXIT

set_jigsaw_mode() {
  local mode="$1"
  MODE="$mode" "$PYTHON_EXE" - <<'PY'
import os
import re
from pathlib import Path

mode = os.environ["MODE"]
path = Path("pretrained/unbert_ju/compare_quantized.py")
text = path.read_text(encoding="utf-8")
updated, count = re.subn(
    r'^FROC_MODE\s*=\s*"(?:strict|pragmatic)"\s*$',
    f'FROC_MODE = "{mode}"',
    text,
    count=1,
    flags=re.MULTILINE,
)
if count != 1:
    raise SystemExit("Failed to update FROC_MODE in compare_quantized.py")
path.write_text(updated, encoding="utf-8")
print(f"[INFO] Updated Jigsaw FROC_MODE -> {mode}")
PY
}

run_jigsaw() {
  local mode="$1"
  local dst="$JIGSAW_RUN_ROOT/$mode"

  echo
  echo "=== Jigsaw | mode=$mode ==="
  set_jigsaw_mode "$mode"
  DATA_DIR="$JIGSAW_DATA_DIR" "$PYTHON_EXE" -m pretrained.unbert_ju.compare_quantized

  mkdir -p "$dst"
  copy_if_exists "outputs/phase23_${mode}" "$dst/phase23"
  copy_if_exists "outputs/quantization_results.json" "$dst/quantization_results.json"
  copy_if_exists "outputs/fairness_comparison.png" "$dst/fairness_comparison.png"
  copy_if_exists "outputs/representation_drift.png" "$dst/representation_drift.png"
}

verify_artifacts() {
  local required=(
    "$BIAS_RUN_ROOT/strict/phase23_strict/metrics_before_after.csv"
    "$BIAS_RUN_ROOT/pragmatic/phase23_pragmatic/metrics_before_after.csv"
    "$JIGSAW_RUN_ROOT/strict/phase23/metrics_before_after.csv"
    "$JIGSAW_RUN_ROOT/pragmatic/phase23/metrics_before_after.csv"
  )

  for p in "${required[@]}"; do
    [[ -f "$p" ]] || die "Missing required artifact: $p"
  done

  echo "[INFO] Artifact verification passed."
}

summarize_outputs() {
  echo
  echo "=== Output summary ==="
  echo "Bias strict      : $BIAS_RUN_ROOT/strict"
  echo "Bias pragmatic   : $BIAS_RUN_ROOT/pragmatic"
  echo "Jigsaw strict    : $JIGSAW_RUN_ROOT/strict/phase23"
  echo "Jigsaw pragmatic : $JIGSAW_RUN_ROOT/pragmatic/phase23"
  echo
  echo "Legacy folders (also updated by underlying scripts):"
  echo "- Bias   : results/bios/phase23_strict, results/bios/phase23_pragmatic"
  echo "- Jigsaw : outputs/phase23_strict, outputs/phase23_pragmatic"
}

# Run all four required jobs.
run_bias strict
run_bias pragmatic
run_jigsaw strict
run_jigsaw pragmatic

verify_artifacts
summarize_outputs

echo
echo "=== Pipeline complete ==="
echo "Timestamped run: $RUN_ROOT"
