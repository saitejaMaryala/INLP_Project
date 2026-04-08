# Work Done and Results

This document records the implementation work completed in the repository and summarizes the latest validated results from the Phase 2/3 runs.

## Handoff Navigation

If someone is reading this repository for the first time, use this order:

1. Read `README.md` for environment setup and workflow commands.
2. Run `run_deadline_pipeline.ps1` for the reproducible archive-and-rerun path.
3. Inspect `outputs/phase23/phase23_verification_report.md` for Jigsaw results.
4. Inspect `results/bios/phase23_strict/phase23_verification_report.md` and `results/bios/phase23_pragmatic/phase23_verification_report.md` for Bias-in-Bios results.
5. Inspect `results/archive/<timestamp>/jigsaw_froc_diff.txt` for pragmatic-vs-strict comparison.

### Minimal execution order

For a clean rerun from scratch:

1. Install dependencies from `requirements.txt`.
2. Ensure datasets exist under `data/jigsaw_uni/` and `data/bias_in_bios/`.
3. Execute `run_deadline_pipeline.ps1 -SkipKill`.
4. Validate that the three key Phase 2/3 artifacts exist for both tasks: verification report, transport diagnostics, and threshold invariance table.

## Replication Checklist (Strict vs Pragmatic)

Use this section when you need to rerun and replicate all reported strict/pragmatic results.

## Phase Definitions

For this repository, the three phases are:

- Phase-1: evaluate the trained model (FP32/FP16/INT8) before fairness post-processing; this is the baseline utility and fairness snapshot.
- Phase-2: construct the fairness-aware correction by calibrating scores (if enabled), aligning group ROC operating targets, and computing deterministic thresholds or randomized threshold policies.
- Phase-3: apply the Phase-2 correction to predictions and verify outcomes via reports (before/after metrics, ROC gap change, transport diagnostics, threshold invariance).

This phase split is used consistently for both Jigsaw and Bias-in-Bios runs.

### Bias-in-Bios replication

Strict mode:

```powershell
python -m training.compare_quant_bios --sanity-check --int8-mode dynamic --score-calibration isotonic --froc-eps 0.02 --froc-mode strict
```

Pragmatic mode:

```powershell
python -m training.compare_quant_bios --sanity-check --int8-mode dynamic --score-calibration isotonic --froc-eps 0.02 --froc-mode pragmatic
```

Expected outputs:

- `results/bios/phase23_strict/phase23_verification_report.md`
- `results/bios/phase23_pragmatic/phase23_verification_report.md`

For full experiments, remove `--sanity-check`.

### Jigsaw replication

The Jigsaw script does not yet expose a CLI switch, so mode is selected by the constant `FROC_MODE` in `pretrained/unbert_ju/compare_quantized.py`.

1. Set `FROC_MODE = "strict"`, then run:

```powershell
python -m pretrained.unbert_ju.compare_quantized
```

2. Set `FROC_MODE = "pragmatic"`, then run again:

```powershell
python -m pretrained.unbert_ju.compare_quantized
```

Expected outputs:

- `outputs/phase23_strict/phase23_verification_report.md`
- `outputs/phase23_pragmatic/phase23_verification_report.md`

If only the archived pragmatic baseline is needed, use `results/archive/20260408_1616/jigsaw_froc_diff.txt` together with the strict report.

## What Was Built

### Core pipeline

The project now supports a frozen-BERT workflow for both datasets:

- BERT-base-uncased is used as the encoder.
- The encoder remains frozen during classifier training.
- The classifier head stays in FP32.
- FP16 and INT8 variants are evaluated after training.
- Phase 2/3 adds ROC-aware group thresholding, score calibration, randomized policies, and threshold-sweep verification.

### Pragmatic FROC vs strict Algorithm-1

There are two distinct versions of the post-processing logic in this project:

- Pragmatic FROC: the earlier implementation that was optimized for usable results and smoother behavior on the datasets.
- Strict Algorithm-1: the current implementation that follows the FAIRROC-style geometry more closely.

The pragmatic version is more direct. It aims to find a useful threshold adjustment quickly, with fewer geometric restrictions, so it tends to preserve utility better. In practice, this means it is less aggressive about forcing points through the exact boundary logic and it can keep accuracy and macro-F1 closer to the original model.

The strict Algorithm-1 version is more theoretical. It explicitly applies the boundary and area-based selection logic, enforces the transport budget, and uses the hypograph-style checks and randomized convex policy construction described in the pipeline. That makes it more faithful to the algorithmic specification, but also more conservative in how it moves the operating point.

In short:

- Pragmatic FROC = better for practical utility and faster convergence to a useful thresholding policy.
- Strict Algorithm-1 = better for fidelity to the formal method and reproducibility of the geometric rules.

Both implementations are now directly available in code:

- Shared pipeline mode switch: `pretrained/unbert_ju/phase23_pipeline.py` via `froc_mode` (`strict` or `pragmatic`).
- Bias-in-Bios CLI switch: `training/compare_quant_bios.py` via `--froc-mode {strict,pragmatic}`.
- Jigsaw runner mode constant: `pretrained/unbert_ju/compare_quantized.py` via `FROC_MODE`.

To run both modes for Bias-in-Bios:

```powershell
python -m training.compare_quant_bios --sanity-check --int8-mode dynamic --score-calibration isotonic --froc-eps 0.02 --froc-mode strict
python -m training.compare_quant_bios --sanity-check --int8-mode dynamic --score-calibration isotonic --froc-eps 0.02 --froc-mode pragmatic
```

Artifacts are written to mode-specific folders:

- `results/bios/phase23_strict/`
- `results/bios/phase23_pragmatic/`

#### Code-level difference

The pragmatic path is the simpler deterministic branch in the pipeline. It is centered on the helper pair `find_group_thresholds(...)` and `apply_group_thresholds(...)`:

- compute per-group ROC curves with `compute_group_roc(...)` and `_safe_roc_curve(...)`;
- interpolate curves onto a shared grid with `interpolate_roc(...)`;
- choose a single threshold per group by minimizing squared distance to a target ROC operating point;
- apply those thresholds directly at inference time.

This branch is effectively a direct operating-point matching routine. It does not build a transport policy, does not perform randomized threshold mixing, and does not enforce the stricter geometric selection logic. As a result, it is easier to interpret and usually less destructive to utility.

The strict Algorithm-1 path is the calibrated transport branch built around `froc_pipeline(...)` and `derive_group_transport_targets(...)`:

- calibrate scores first using `calibrate_scores(...)` with `none`, `platt`, or `isotonic` mapping;
- identify disadvantaged and privileged groups by per-group AUC;
- interpolate both group ROC curves to a shared `common_fpr` grid;
- run `_fairroc_algorithm(...)` to move the privileged ROC toward the disadvantaged ROC under an $L_1$ transport budget `eps`;
- use `_in_hypograph(...)` to detect already-feasible points;
- use `_cutshift(...)` and `_project_l1_boundary(...)` when a point violates the boundary constraint;
- compare left-shift and up-shift candidates via `_area_loss_with_candidate(...)` and `_triangle_area(...)`;
- build per-group randomized two-threshold policies with `build_randomized_group_policies(...)` and `_build_group_random_policy(...)`;
- apply the randomized policy with `apply_randomized_group_policy(...)` and verify stability with `threshold_invariance_check(...)`.

So the technical gap is not just "different thresholds." The strict path introduces calibration, shared-grid transport, epsilon-bounded geometric projection, hypograph feasibility checks, area-loss-based candidate selection, and randomized convex policy execution. The pragmatic path skips that machinery and instead uses deterministic point matching on the ROC surface.

### Jigsaw implementation

The Jigsaw path now runs the full quantization and fairness comparison pipeline and writes a separate Phase 2/3 bundle. The implementation includes:

- calibration support for none, Platt, and isotonic methods,
- epsilon-constrained L1 transport,
- group-specific threshold policies,
- ROC-gap reporting,
- threshold invariance checks,
- archived original-vs-current FROC comparisons.

### Bias-in-Bios implementation

The Bias-in-Bios path now includes:

- quantization mode selection,
- calibration controls,
- a configurable epsilon budget,
- optional per-class fairness variance computation,
- full Phase 2/3 artifact export.

## Validation Runs

The latest end-to-end execution completed successfully through the deadline pipeline script:

```powershell
.\run_deadline_pipeline.ps1 -SkipKill
```

That script archived the current artifacts, reran Jigsaw, and generated a diff report for the original vs. current FROC variants.

The archived diff compares the pragmatic Jigsaw snapshot against the strict current implementation, which is why the report is useful for explaining the tradeoff in the final write-up.

## Key Results

### Jigsaw

The Jigsaw archive now contains two result profiles: the earlier pragmatic FROC snapshot and the strict Algorithm-1 snapshot. Both reduce ROC gap, but they differ materially in fairness/utility tradeoff.

#### Pragmatic FROC (archived baseline)

Source: `results/archive/20260408_1616/jigsaw_froc_diff.txt` (original FROC block).

- FP32: ROC gap 0.00705001 -> 0.00206793, DPD 0.083866 -> 0.063252, EOD 0.057462 -> 0.039767, acc 0.894500 -> 0.894400, f1_macro 0.668468 -> 0.667463.
- INT8: ROC gap 0.00624513 -> 0.00000066, DPD 0.011128 -> 0.007199, EOD 0.010257 -> 0.001091, acc 0.926200 -> 0.926450, f1_macro 0.557059 -> 0.557833.
- FP16: ROC gap 0.00702745 -> 0.00207472, DPD 0.083811 -> 0.061079, EOD 0.057403 -> 0.037823, acc 0.894450 -> 0.894450, f1_macro 0.668397 -> 0.667361.

Technical reading:

- ROC alignment improved strongly for all precisions.
- Fairness metrics (DPD/EOD) improved for all precisions.
- Utility was nearly unchanged (accuracy and macro-F1 deltas close to zero).
- This profile is operationally stable and utility-preserving.

#### Strict Algorithm-1 FROC (current implementation)

Source: `outputs/phase23/phase23_verification_report.md` (current run).

Main observations from `outputs/phase23/phase23_verification_report.md`:

- FP32 ROC gap improved from 0.00705001 to 0.00299793.
- FP32 accuracy dropped from 0.926100 to 0.788150 after post-processing.
- FP32 DPD rose from 0.024907 to 0.054290.
- FP32 EOD rose from 0.037805 to 0.105004.
- INT8 ROC gap improved from 0.00624513 to 0.00059715.
- INT8 accuracy dropped from 0.926550 to 0.716150.
- INT8 EOD rose from 0.009644 to 0.112466.
- The transport budget was respected with `max_l1_after = 0.02`.

Interpretation:

- The strict Algorithm-1 implementation is reproducible and obeys the transport constraint.
- The same strictness makes it overly conservative on Jigsaw.
- The earlier pragmatic FROC variant remains more utility-friendly in the archived diff.

#### Side-by-side conclusion

- Both variants improve ROC gap.
- Pragmatic FROC improves DPD/EOD and preserves utility.
- Strict Algorithm-1 FROC enforces stronger geometric constraints (transport budget, boundary logic, randomized policy), but in this run it increases disparity metrics and reduces utility.
- For publication/reporting: present strict Algorithm-1 as the formal method and pragmatic FROC as the better empirical operating point on Jigsaw.

### Bias-in-Bios

Bias-in-Bios now has explicit strict/pragmatic runs from the same sanity-check setup (dynamic INT8, isotonic calibration, eps=0.02).

#### Strict Algorithm-1 (Bias)

Source: `results/bios/phase23_strict/phase23_verification_report.md`.

- FP32: ROC gap 0.00045675 -> 0.00020103, DPD 0.000928 -> 0.020495, EOD 0.056006 -> 0.020863, acc 0.982857 -> 0.929286, f1_macro 0.871849 -> 0.721188.
- FP16: ROC gap 0.00048233 -> 0.00020103, DPD 0.000928 -> 0.008813, EOD 0.056006 -> 0.009740, acc 0.982857 -> 0.937857, f1_macro 0.871849 -> 0.739709.
- INT8: ROC gap 0.00086053 -> 0.00048964, DPD 0.000348 -> 0.003595, EOD 0.025974 -> 0.042208, acc 0.976429 -> 0.911071, f1_macro 0.779655 -> 0.677418.

#### Pragmatic FROC (Bias)

Source: `results/bios/phase23_pragmatic/phase23_verification_report.md`.

- FP32: ROC gap 0.00045675 -> 0.00059243, DPD 0.000928 -> 0.000290, EOD 0.056006 -> 0.047078, acc 0.982857 -> 0.982857, f1_macro 0.871849 -> 0.872483.
- FP16: ROC gap 0.00048233 -> 0.00059243, DPD 0.000928 -> 0.000290, EOD 0.056006 -> 0.047078, acc 0.982857 -> 0.982857, f1_macro 0.871849 -> 0.872483.
- INT8: ROC gap 0.00086053 -> 0.00257356, DPD 0.000348 -> 0.000348, EOD 0.025974 -> 0.025974, acc 0.976429 -> 0.976429, f1_macro 0.779655 -> 0.779655.

#### Side-by-side conclusion (Bias)

- Strict mode improved ROC-gap alignment for all three precisions, but with substantial utility loss.
- Pragmatic mode preserved utility almost exactly and usually improved or maintained fairness metrics, but did not consistently reduce ROC gap.
- The same pattern seen on Jigsaw is present on Bias: strict gives stronger geometric correction, pragmatic gives better operational utility.

## Artifacts Produced

The run generated the following useful outputs:

- `outputs/phase23/phase23_verification_report.md`
- `outputs/phase23/transport_diagnostics.json`
- `outputs/phase23/threshold_invariance.csv`
- `outputs/phase23/metrics_before_after.csv`
- `outputs/phase23/roc_gap.csv`
- `results/bios/phase23_strict/phase23_verification_report.md`
- `results/bios/phase23_strict/transport_diagnostics.json`
- `results/bios/phase23_strict/threshold_invariance.csv`
- `results/bios/phase23_pragmatic/phase23_verification_report.md`
- `results/bios/phase23_pragmatic/transport_diagnostics.json`
- `results/bios/phase23_pragmatic/threshold_invariance.csv`
- `results/archive/20260408_1616/jigsaw_froc_diff.txt`

## Practical Takeaway

The work now supports two clearly distinct stories:

1. The strict Algorithm-1 FAIRROC implementation is the correct theoretical version and is fully instrumented.
2. The earlier pragmatic FROC variant may be the better practical choice on Jigsaw when utility matters more than geometric strictness.

That distinction is the main result to carry into the final write-up.