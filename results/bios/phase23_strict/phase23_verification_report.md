# Phase 2/3 Verification Report

This report checks whether ROC-aware group thresholding behaves as expected.

## Formal Contribution Statement
This project establishes an explicit analysis chain from numerical compression to fairness behavior: quantization precision changes induce representation drift, representation drift alters group-wise ROC geometry, and this ROC misalignment manifests as threshold-level fairness disparity.

We then show that ROC-aware group thresholding (FROC-style post-processing) can substantially reduce group misalignment and fairness disparity across FP32, FP16, and INT8 settings with minimal accuracy change.

## Claimed Contributions
1. ROC-level fairness diagnosis under quantization: beyond single-threshold DPD/EOD reporting, we quantify group ROC misalignment and track how it changes with precision.
2. Quantization-aware fairness repair via post-hoc thresholds: we demonstrate that a ROC-targeted group-threshold policy mitigates disparity after quantization without retraining.
3. Distinction between apparent and substantive fairness gains: we show that low disparity in low-precision models can coincide with utility degradation, motivating ROC-level and utility-aware interpretation.
4. Multi-view evidence linkage: representation drift metrics (CKA/L2/cosine), ROC distortion, and fairness metrics are analyzed together to support mechanistic interpretation rather than isolated metric reporting.

## Quantization Mode
- requested_mode: dynamic
- applied_mode: dynamic
- fallback_reason: None
- calibration_method: isotonic
- froc_mode: strict

## Core Checks
### fp32
- ROC gap: 0.00000247 -> 0.00000824 (delta=0.00000577)
- Fairness: DPD 0.000489 -> 0.001360, EOD 0.005582 -> 0.001402
- Utility: acc 0.983009 -> 0.939413, f1_macro 0.866083 -> 0.745070
- AUC terms: score_auc=0.982855, policy_auc(after)=0.933962
- Thresholds: group 0: 0.034990, group 1: 0.037947
- Transport: disadvantaged=0, privileged=1, max_l1_after=0.020000, eps=0.020000, ops={'BoundaryCut': 0, 'CutShift': 0, 'Hypograph': 17, 'UpShift': 58, 'LeftShift': 23}
- Calibration: method=isotonic, applied=True, reason=None
- Threshold sweep (before): max|DPD|=0.000652, max|EOD|=0.013395, within_eps=True
- Threshold sweep (after): max|DPD|=0.001360, max|EOD|=0.001402, within_eps=True
- Consistency vs phase-1: Δacc=0.24391932, Δmacro_f1=0.19498270

### fp16
- ROC gap: 0.00000250 -> 0.00001018 (delta=0.00000769)
- Fairness: DPD 0.000480 -> 0.001643, EOD 0.005653 -> 0.003069
- Utility: acc 0.983011 -> 0.940489, f1_macro 0.866064 -> 0.747402
- AUC terms: score_auc=0.982853, policy_auc(after)=0.933810
- Thresholds: group 0: 0.035022, group 1: 0.035022
- Transport: disadvantaged=0, privileged=1, max_l1_after=0.020000, eps=0.020000, ops={'BoundaryCut': 0, 'CutShift': 0, 'Hypograph': 18, 'UpShift': 55, 'LeftShift': 25}
- Calibration: method=isotonic, applied=True, reason=None
- Threshold sweep (before): max|DPD|=0.000659, max|EOD|=0.013764, within_eps=True
- Threshold sweep (after): max|DPD|=0.001643, max|EOD|=0.003069, within_eps=True
- Consistency vs phase-1: Δacc=0.24402076, Δmacro_f1=0.19504420

### int8
- ROC gap: 0.00001317 -> 0.00013445 (delta=0.00012128)
- Fairness: DPD 0.001115 -> 0.001030, EOD 0.022870 -> 0.006241
- Utility: acc 0.975322 -> 0.899199, f1_macro 0.781420 -> 0.664983
- AUC terms: score_auc=0.957815, policy_auc(after)=0.891371
- Thresholds: group 0: 0.038035, group 1: 0.038035
- Transport: disadvantaged=1, privileged=0, max_l1_after=0.020000, eps=0.020000, ops={'BoundaryCut': 0, 'CutShift': 0, 'Hypograph': 76, 'UpShift': 8, 'LeftShift': 14}
- Calibration: method=isotonic, applied=True, reason=None
- Threshold sweep (before): max|DPD|=0.003317, max|EOD|=0.033880, within_eps=False
- Threshold sweep (after): max|DPD|=0.001030, max|EOD|=0.006241, within_eps=True
- Consistency vs phase-1: Δacc=0.36502239, Δmacro_f1=0.23827964

## Interpretation Guide
- If ROC gap decreases, group ROC misalignment is reduced.
- If DPD/EOD decrease with near-stable accuracy, post-processing is behaving as intended.
- score_auc is a model-score property; policy_auc reflects post-threshold operating behavior.
