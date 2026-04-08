# INLP Project: Quantization and Fairness

This repository contains the frozen-BERT quantization experiments for Jigsaw and Bias-in-Bios, plus the Phase 2/3 fairness post-processing pipeline. The main goals are to compare FP32, FP16, and INT8 inference, measure fairness drift, and document how ROC-aware thresholding changes the final decisions.

## Quick Start (Recommended)

From repository root on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# one-shot deadline workflow (archive + rerun + diff)
.\run_deadline_pipeline.ps1 -SkipKill
```

If you only need one dataset workflow, use the detailed commands in the Workflow section below.

## Project Structure

Top-level layout used in the final pipeline:

```text
INLP_Project/
|-- training/
|   |-- train_bios_clean.py
|   `-- compare_quant_bios.py
|-- pretrained/unbert_ju/
|   |-- model.py
|   |-- compare_quantized.py
|   `-- phase23_pipeline.py
|-- data/
|   |-- bias_in_bios/
|   `-- jigsaw_uni/
|-- outputs/
|   |-- phase23_strict/
|   `-- phase23_pragmatic/
|-- results/
|   |-- bios/phase23_strict/
|   |-- bios/phase23_pragmatic/
|   `-- archive/
`-- run_deadline_pipeline.ps1
```

## File Responsibilities

- `training/train_bios_clean.py`: trains the Bias-in-Bios FP32 classifier.
- `training/compare_quant_bios.py`: runs Bias-in-Bios quantization and fairness analysis.
- `pretrained/unbert_ju/model.py`: trains the Jigsaw FP32 classifier.
- `pretrained/unbert_ju/compare_quantized.py`: runs Jigsaw quantization and Phase 2/3 post-processing.
- `pretrained/unbert_ju/phase23_pipeline.py`: shared calibration, transport, policy, and verification logic.
- `run_deadline_pipeline.ps1`: one-shot PowerShell pipeline that archives outputs, reruns Jigsaw, and writes a diff report.
- `results/` and `outputs/`: generated artifacts from the latest runs.

## Setup

From the repository root on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are using a different interpreter, keep the same dependency set in that environment.

## Data Layout

### Bias-in-Bios

Expected under:

- `data/bias_in_bios/`

This should match the structure expected by `utils/data_loader.py`.

### Jigsaw

Expected under:

- `data/jigsaw_uni/train.csv`
- `data/jigsaw_uni/test.csv`
- `data/jigsaw_uni/all_data.csv`
- `data/jigsaw_uni/test_public_expanded.csv`
- `data/jigsaw_uni/test_private_expanded.csv`
- `data/jigsaw_uni/identity_individual_annotations.csv`
- `data/jigsaw_uni/toxicity_individual_annotations.csv`

If the dataset is in another location, set `DATA_DIR` before running the Jigsaw scripts:

```powershell
$env:DATA_DIR = "D:\path\to\jigsaw_uni"
```

## Workflows

Choose one of these run paths based on your goal.

## What Phase-1, Phase-2, and Phase-3 Mean

The fairness pipeline is split into three logical stages:

- Phase-1 (base model and quantization evaluation): run FP32, FP16, and INT8 inference and compute baseline utility/fairness metrics from raw model scores and default thresholding.
- Phase-2 (ROC-aware target and policy construction): calibrate scores (optional), compute per-group ROC geometry, and derive group-specific threshold targets or randomized policies under the configured transport budget (`froc_eps`).
- Phase-3 (post-processing application and verification): apply the Phase-2 policy to predictions, then export before/after metrics, ROC-gap deltas, transport diagnostics, and threshold-invariance checks.

In short: Phase-1 measures the model as-is, Phase-2 computes fairness-aware correction, and Phase-3 validates the corrected operating point.

### Workflow A: Full handoff pipeline (recommended)

```powershell
.\run_deadline_pipeline.ps1 -SkipKill
```

Use this when you want reproducible, timestamped outputs with archival and diff generation.

### Workflow B: Bias-in-Bios only

#### 1) Train FP32 model

Train the frozen-BERT classifier head first:

```powershell
python -m training.train_bios_clean `
  --data-dir data/bias_in_bios `
  --save-dir models/bios `
  --cache-dir cache/bios
```

This produces the FP32 checkpoint used by the comparison script.

#### 2) Run quantization + fairness + Phase 2/3

Run the comparison and fairness evaluation after training:

```powershell
python -m training.compare_quant_bios `
  --data-dir data/bias_in_bios `
  --model-path models/bios/best_bios_fp32.pt `
  --results-dir results/bios `
  --cache-dir cache/bios `
  --int8-mode dynamic `
  --score-calibration none `
  --froc-eps 0.02
```

Useful flags:

- `--int8-mode {dynamic,static}` controls the INT8 path.
- `--score-calibration {none,platt,isotonic}` selects score calibration.
- `--froc-eps` sets the L1 transport budget for Phase 2/3.
- `--skip-per-class-variance` skips the expensive per-class fairness variance pass.
- `--sanity-check` runs a smaller, faster version of the pipeline.

### Workflow C: Jigsaw only

#### 1) Train FP32 model

Train the Jigsaw FP32 classifier with the frozen encoder:

```powershell
python -m pretrained.unbert_ju.model
```

This writes the Jigsaw checkpoint used by the quantization run.

#### 2) Run quantization + fairness + Phase 2/3

Run the full Jigsaw comparison:

```powershell
python -m pretrained.unbert_ju.compare_quantized
```

This performs FP32, FP16, and INT8 evaluation, then calls the Phase 2/3 pipeline for calibration, ROC-gap analysis, threshold transport, and randomized policy checks.

#### 3) Reproduce strict vs pragmatic Jigsaw modes

The Jigsaw runner uses a mode constant in `pretrained/unbert_ju/compare_quantized.py`:

- `FROC_MODE = "strict"` writes to `outputs/phase23_strict/`
- `FROC_MODE = "pragmatic"` writes to `outputs/phase23_pragmatic/`

To regenerate both modes from the same codebase:

1. Set `FROC_MODE = "strict"`, run `python -m pretrained.unbert_ju.compare_quantized`.
2. Set `FROC_MODE = "pragmatic"`, run `python -m pretrained.unbert_ju.compare_quantized`.

This gives fully comparable strict and pragmatic Jigsaw artifacts.

### Workflow E: Reproduce strict vs pragmatic Bias-in-Bios

Run both modes explicitly with the same settings:

```powershell
python -m training.compare_quant_bios --sanity-check --int8-mode dynamic --score-calibration isotonic --froc-eps 0.02 --froc-mode strict
python -m training.compare_quant_bios --sanity-check --int8-mode dynamic --score-calibration isotonic --froc-eps 0.02 --froc-mode pragmatic
```

For full runs, remove `--sanity-check`.

### Workflow D: What the deadline script does

`run_deadline_pipeline.ps1` performs the following sequence:

- Archives the latest Bias-in-Bios artifacts.
- Archives the previous Jigsaw Phase 2/3 outputs as the original FROC snapshot.
- Reruns Jigsaw with the current implementation.
- Saves the new Jigsaw Phase 2/3 outputs as the current FROC snapshot.
- Writes a diff report under `results/archive/<timestamp>/jigsaw_froc_diff.txt`.

## What Gets Written

### Bias-in-Bios outputs

- `results/bios/bios_results.json`
- `results/bios/phase23_strict/phase23_verification_report.md`
- `results/bios/phase23_strict/transport_diagnostics.json`
- `results/bios/phase23_strict/threshold_invariance.csv`
- `results/bios/phase23_strict/metrics_before_after.csv`
- `results/bios/phase23_pragmatic/phase23_verification_report.md`
- `results/bios/phase23_pragmatic/transport_diagnostics.json`
- `results/bios/phase23_pragmatic/threshold_invariance.csv`
- `results/bios/phase23_pragmatic/metrics_before_after.csv`
- `results/bios/*.png`

### Jigsaw outputs

- `outputs/quantization_results.json`
- `outputs/fairness_comparison.png`
- `outputs/representation_drift.png`
- `outputs/dpd_heatmap.png`
- `outputs/eod_heatmap.png`
- `outputs/phase23_strict/metrics_before_after.csv`
- `outputs/phase23_strict/roc_gap.csv`
- `outputs/phase23_strict/thresholds.json`
- `outputs/phase23_strict/transport_diagnostics.json`
- `outputs/phase23_strict/threshold_invariance.csv`
- `outputs/phase23_strict/phase23_verification_report.md`
- `outputs/phase23_pragmatic/metrics_before_after.csv`
- `outputs/phase23_pragmatic/roc_gap.csv`
- `outputs/phase23_pragmatic/thresholds.json`
- `outputs/phase23_pragmatic/transport_diagnostics.json`
- `outputs/phase23_pragmatic/threshold_invariance.csv`
- `outputs/phase23_pragmatic/phase23_verification_report.md`

### Archive outputs

- `results/archive/<timestamp>/bias_current/`
- `results/archive/<timestamp>/jigsaw_original_froc/`
- `results/archive/<timestamp>/jigsaw_current_froc/`
- `results/archive/<timestamp>/jigsaw_froc_diff.txt`

## Current Results Summary

The latest run completed successfully and produced the following high-level outcomes:

- Bias-in-Bios: the current Phase 2/3 pipeline satisfied the L1 budget and produced the full diagnostic bundle.
- Jigsaw: the strict Algorithm-1 run reduced ROC gap, but it also lowered utility and raised post-processing fairness metrics compared with the earlier pragmatic FROC variant.
- The archived diff in `results/archive/20260408_1616/jigsaw_froc_diff.txt` captures the difference between the earlier Jigsaw output and the current run.

The most important interpretation is that the strict algorithm is mechanically correct and reproducible, but it is not always the best practical operating point. On Jigsaw, the older pragmatic FROC snapshot appears more utility-friendly; on Bias-in-Bios, the current pipeline behaves more stably and stays within the configured epsilon budget.

## How To Read The Reports

- `score_auc` is the original score-ranking quality before thresholding.
- `policy_auc` is the decision quality after the randomized threshold policy is applied.
- `ROC gap` measures how well the group ROC curves align.
- `DPD` and `EOD` are the post-threshold fairness metrics.
- `max_l1_after` should stay at or below `froc_eps`.
- `threshold_invariance.csv` shows the threshold sweep used to verify whether the post-processing remains stable across decision thresholds.

## Troubleshooting

- If static INT8 fails, the scripts may fall back to dynamic INT8 and record that in the quantization metadata.
- If a run looks slow, check whether per-class Bias-in-Bios variance is enabled; that pass is intentionally expensive.
- If a dataset path is wrong, verify the directory structure above before rerunning.
- Always inspect the generated JSON metadata to confirm `requested_mode`, `applied_mode`, and `fallback_reason`.

## Related Documentation

- `VM_SETUP.md` for VM setup and deployment notes.
- `docs/Project_plan.md` for the original project framing.
- `docs/training_flow.md` for the training and quantization flow.
- `docs/model_strategy.md` for the frozen-encoder model selection rationale.
- `docs/work_done_and_results.md` for the detailed implementation and results summary.
