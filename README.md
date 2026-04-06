# INLP Project: Quantization and Fairness

This repository compares FP32, FP16, and INT8 (dynamic/static-attempt) inference for fairness-sensitive NLP tasks.

## What is implemented

- Bias-in-Bios quantization comparison with calibration controls:
  - `--int8-mode {dynamic,static}`
  - `--calib-samples`
  - `--calib-batch-size`
  - `--calib-batches`
- Jigsaw quantization comparison with static calibration attempt and automatic dynamic fallback when unsupported.
- Results include explicit quantization metadata (requested mode, applied mode, fallback reason).

## Environment setup

From repo root:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset layout

### Bias-in-Bios

Place prepared Bias-in-Bios files under:

- `data/bias_in_bios/`

Use the same files expected by `utils/data_loader.py`.

### Jigsaw

Place files under:

- `data/jigsaw_uni/train.csv`
- `data/jigsaw_uni/test.csv`
- `data/jigsaw_uni/all_data.csv`
- `data/jigsaw_uni/test_public_expanded.csv`
- `data/jigsaw_uni/test_private_expanded.csv`
- `data/jigsaw_uni/identity_individual_annotations.csv`
- `data/jigsaw_uni/toxicity_individual_annotations.csv`

If your data is elsewhere, set `DATA_DIR`.

Windows PowerShell:

```powershell
$env:DATA_DIR = "D:\path\to\jigsaw_uni"
```

Linux/macOS:

```bash
export DATA_DIR=/path/to/jigsaw_uni
```

## Reproduce Bias-in-Bios run

### 1) Train FP32 baseline

```bash
python -m training.train_bios_clean \
  --data-dir data/bias_in_bios \
  --save-dir models/bios \
  --cache-dir cache/bios
```

Checkpoint output:

- `models/bios/best_bios_fp32.pt`

### 2) Run quantization/fairness comparison with calibration flags

```bash
python -m training.compare_quant_bios \
  --data-dir data/bias_in_bios \
  --model-path models/bios/best_bios_fp32.pt \
  --results-dir results/bios \
  --cache-dir cache/bios \
  --int8-mode static \
  --calib-samples 2048 \
  --calib-batch-size 32 \
  --calib-batches 32
```

Primary outputs:

- `results/bios/bios_results.json`
- `results/bios/*.png`

## Reproduce Jigsaw run

### 1) Train FP32 baseline

```bash
python -m pretrained.unbert_ju.model
```

Checkpoint output:

- `outputs/fp32_classifier_uni.pt`

### 2) Run quantization/fairness comparison

```bash
python -m pretrained.unbert_ju.compare_quantized
```

Primary outputs:

- `outputs/quantization_results.json`
- `outputs/fairness_comparison.png`
- `outputs/representation_drift.png`
- `outputs/dpd_heatmap.png`
- `outputs/eod_heatmap.png`

## Calibration behavior notes

- Bias-in-Bios script supports static mode selection via CLI and records applied/fallback metadata.
- Jigsaw script attempts static PTQ first (default in code), but may fall back to dynamic INT8 depending on PyTorch/ONNX/toolchain support on your machine.
- Always check the JSON metadata:
  - `quantization_meta.requested_mode`
  - `quantization_meta.applied_mode`
  - `quantization_meta.fallback_reason`

## Team handoff artifacts

Consolidated run summary is stored in:

- `results/final_results_summary.md`
- `results/final_results_summary.json`

These capture the latest local run numbers and static-fallback status for both tasks.
