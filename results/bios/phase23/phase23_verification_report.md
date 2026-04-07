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
- requested_mode: None
- applied_mode: None
- fallback_reason: None

## Core Checks
### fp32
- ROC gap: 0.00000504 -> 0.00000000 (delta=-0.00000504)
- Fairness: DPD 0.000726 -> 0.000098, EOD 0.009077 -> 0.000103
- Utility: acc 0.982840 -> 0.982845, f1_macro 0.861879 -> 0.861910
- AUC terms: score_auc=0.981374, policy_auc(after)=0.826563
- Thresholds: group 0: 0.508245, group 1: 0.490562
- Consistency vs phase-1: Δacc=0.24635988, Δmacro_f1=0.19287885

### fp16
- ROC gap: 0.00000509 -> 0.00000000 (delta=-0.00000509)
- Fairness: DPD 0.000724 -> 0.000104, EOD 0.009130 -> 0.000109
- Utility: acc 0.982837 -> 0.982845, f1_macro 0.861850 -> 0.861901
- AUC terms: score_auc=0.981368, policy_auc(after)=0.826543
- Thresholds: group 0: 0.508203, group 1: 0.490607
- Consistency vs phase-1: Δacc=0.24635700, Δmacro_f1=0.19295963

### int8
- ROC gap: 0.00005083 -> 0.00000000 (delta=-0.00005083)
- Fairness: DPD 0.000244 -> 0.000029, EOD 0.006811 -> 0.000089
- Utility: acc 0.968371 -> 0.968367, f1_macro 0.601885 -> 0.601813
- AUC terms: score_auc=0.953723, policy_auc(after)=0.562177
- Thresholds: group 0: 0.495108, group 1: 0.504769
- Consistency vs phase-1: Δacc=0.36882053, Δmacro_f1=0.05764526

## Interpretation Guide
- If ROC gap decreases, group ROC misalignment is reduced.
- If DPD/EOD decrease with near-stable accuracy, post-processing is behaving as intended.
- score_auc is a model-score property; policy_auc reflects post-threshold operating behavior.
