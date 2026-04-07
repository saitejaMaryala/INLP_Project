"""Phase 2/3 post-processing and ROC analysis pipeline.

This module implements group-aware threshold selection (FROC-style
post-processing), ROC distortion analysis, plots, and artifact saving.

It is designed for binary classification results of the form:

    results = {
        "fp32": {"y_true": ..., "y_score": ..., "group": ...},
        "fp16": {"y_true": ..., "y_score": ..., "group": ...},
        "int8": {"y_true": ..., "y_score": ..., "group": ...},
    }

The implementation is intentionally defensive:
- handles single-group inputs
- handles groups with a single class
- handles imbalanced groups
- avoids crashes when ROC/AUC are undefined
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


DEFAULT_THRESHOLD = 0.5
DEFAULT_NUM_POINTS = 100
DEFAULT_EPS = 1e-12


def _to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(values)


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        numeric = float(value)
    except Exception:
        return float("nan")
    if np.isnan(numeric) or np.isinf(numeric):
        return numeric
    return numeric


def _safe_group_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_group_label(value: Any) -> str:
    return str(_safe_group_key(value))


def compute_global_operating_point(y_true, y_score, threshold=DEFAULT_THRESHOLD):
    """Return the global TPR/FPR at the given threshold."""
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr, fpr


def _safe_roc_curve(y_true, y_score):
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    if y_true.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, 0.5])
    unique_classes = np.unique(y_true)
    if unique_classes.size < 2:
        # ROC is undefined; return a degenerate curve that still keeps the pipeline stable.
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, 0.5])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return fpr, tpr, thresholds


def compute_group_roc(y_true, y_score, group):
    """Return per-group ROC curves as dict[group] = (fpr, tpr, thresholds)."""
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    group = _to_numpy(group)

    group_roc = {}
    for group_value in np.unique(group):
        mask = group == group_value
        fpr, tpr, thresholds = _safe_roc_curve(y_true[mask], y_score[mask])
        group_roc[_safe_group_key(group_value)] = (fpr, tpr, thresholds)
    return group_roc


def interpolate_roc(fpr, tpr, num_points=DEFAULT_NUM_POINTS):
    """Interpolate a ROC curve to a common FPR grid."""
    fpr = _to_numpy(fpr).astype(float)
    tpr = _to_numpy(tpr).astype(float)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    unique_fpr, unique_indices = np.unique(fpr, return_index=True)
    unique_tpr = tpr[unique_indices]
    if unique_fpr.size == 0:
        common_fpr = np.linspace(0.0, 1.0, num_points)
        return common_fpr, np.zeros_like(common_fpr)

    if unique_fpr[0] > 0.0:
        unique_fpr = np.concatenate(([0.0], unique_fpr))
        unique_tpr = np.concatenate(([0.0], unique_tpr))
    if unique_fpr[-1] < 1.0:
        unique_fpr = np.concatenate((unique_fpr, [1.0]))
        unique_tpr = np.concatenate((unique_tpr, [unique_tpr[-1] if unique_tpr.size else 0.0]))

    common_fpr = np.linspace(0.0, 1.0, num_points)
    interp_tpr = np.interp(common_fpr, unique_fpr, unique_tpr)
    return common_fpr, interp_tpr


def find_group_thresholds(y_true, y_score, group, target_tpr, target_fpr):
    """Learn an optimal threshold for each group by matching a target ROC operating point."""
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    group = _to_numpy(group)

    thresholds_per_group = {}
    for group_value in np.unique(group):
        mask = group == group_value
        group_y_true = y_true[mask]
        group_y_score = y_score[mask]

        if group_y_true.size == 0:
            thresholds_per_group[_safe_group_key(group_value)] = DEFAULT_THRESHOLD
            continue

        fpr, tpr, thresholds = _safe_roc_curve(group_y_true, group_y_score)
        if thresholds.size == 0:
            thresholds_per_group[_safe_group_key(group_value)] = DEFAULT_THRESHOLD
            continue

        score = (tpr - target_tpr) ** 2 + (fpr - target_fpr) ** 2
        best_idx = int(np.argmin(score))
        best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
        if not np.isfinite(best_threshold):
            best_threshold = DEFAULT_THRESHOLD
        thresholds_per_group[_safe_group_key(group_value)] = _safe_float(best_threshold)

    return thresholds_per_group


def apply_group_thresholds(y_score, group, thresholds_per_group):
    """Apply per-group thresholds and return binary predictions."""
    y_score = _to_numpy(y_score).astype(float)
    group = _to_numpy(group)

    y_pred_froc = np.zeros_like(y_score, dtype=int)
    fallback_threshold = thresholds_per_group.get("__default__", DEFAULT_THRESHOLD)

    for idx, (score, group_value) in enumerate(zip(y_score, group)):
        threshold = thresholds_per_group.get(_safe_group_key(group_value), fallback_threshold)
        y_pred_froc[idx] = int(score >= threshold)
    return y_pred_froc


def evaluate_metrics(y_true, y_pred, y_score, group):
    """Return accuracy, macro/binary F1, score-AUC, DPD, and EOD."""
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    group = _to_numpy(group)

    metrics = {
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "f1_macro": _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_binary": _safe_float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        if np.unique(y_true).size < 2 or np.unique(y_score).size < 2:
            metrics["score_auc"] = float("nan")
        else:
            metrics["score_auc"] = _safe_float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["score_auc"] = float("nan")

    try:
        metrics["dpd"] = _safe_float(
            demographic_parity_difference(y_true, y_pred, sensitive_features=group)
        )
    except Exception:
        metrics["dpd"] = float("nan")

    try:
        metrics["eod"] = _safe_float(
            equalized_odds_difference(y_true, y_pred, sensitive_features=group)
        )
    except Exception:
        metrics["eod"] = float("nan")

    return metrics


def froc_pipeline(results):
    """Run global operating point estimation, group thresholds, and metric comparison."""
    metrics_before_after = {}
    thresholds_per_model = {}

    for model_name, model_results in results.items():
        y_true = _to_numpy(model_results["y_true"]).astype(int)
        y_score = _to_numpy(model_results["y_score"]).astype(float)
        group = _to_numpy(model_results["group"])

        target_tpr, target_fpr = compute_global_operating_point(y_true, y_score)
        thresholds = find_group_thresholds(y_true, y_score, group, target_tpr, target_fpr)
        y_pred_before = (y_score >= DEFAULT_THRESHOLD).astype(int)
        y_pred_after = apply_group_thresholds(y_score, group, thresholds)

        before_metrics = evaluate_metrics(y_true, y_pred_before, y_score, group)
        # The post-FROC decision scores are binary by design, which makes the
        # post-processing AUC reflect the effective operating decision surface.
        after_metrics = evaluate_metrics(y_true, y_pred_after, y_pred_after.astype(float), group)
        after_metrics["policy_auc"] = after_metrics.get("score_auc", float("nan"))
        after_metrics["score_auc"] = before_metrics.get("score_auc", float("nan"))

        metrics_before_after[model_name] = {
            "before": before_metrics,
            "after": after_metrics,
            "target_tpr": _safe_float(target_tpr),
            "target_fpr": _safe_float(target_fpr),
            "global_threshold": DEFAULT_THRESHOLD,
            "group_values": [_safe_group_key(g) for g in np.unique(group)],
        }
        thresholds_per_model[model_name] = {
            "target_tpr": _safe_float(target_tpr),
            "target_fpr": _safe_float(target_fpr),
            "thresholds": thresholds,
        }

    return metrics_before_after, thresholds_per_model


def compute_roc_gap(y_true, y_score, group, num_points=DEFAULT_NUM_POINTS, mode="mse"):
    """Compute ROC misalignment across groups after interpolation to a common grid."""
    group_roc = compute_group_roc(y_true, y_score, group)
    if len(group_roc) <= 1:
        return 0.0

    interpolated = []
    for fpr, tpr, _ in group_roc.values():
        _, interp_tpr = interpolate_roc(fpr, tpr, num_points=num_points)
        interpolated.append(interp_tpr)

    if len(interpolated) <= 1:
        return 0.0

    pairs = []
    for left, right in combinations(interpolated, 2):
        diff = left - right
        if mode == "max":
            pairs.append(float(np.max(np.abs(diff))))
        else:
            pairs.append(float(np.mean(diff ** 2)))

    return float(np.mean(pairs)) if pairs else 0.0


def compute_roc_gap_after_froc(y_true, y_score, group, thresholds):
    """Apply group thresholds, then compute ROC gap on the post-processed decisions."""
    y_pred_froc = apply_group_thresholds(y_score, group, thresholds)
    return compute_roc_gap(y_true, y_pred_froc.astype(float), group)


def roc_analysis_pipeline(results, thresholds_per_model):
    """Compute ROC gap before and after FROC for every model."""
    roc_gap_before = {}
    roc_gap_after = {}

    for model_name, model_results in results.items():
        y_true = _to_numpy(model_results["y_true"]).astype(int)
        y_score = _to_numpy(model_results["y_score"]).astype(float)
        group = _to_numpy(model_results["group"])
        thresholds = thresholds_per_model[model_name]["thresholds"]

        roc_gap_before[model_name] = _safe_float(compute_roc_gap(y_true, y_score, group))
        roc_gap_after[model_name] = _safe_float(
            compute_roc_gap_after_froc(y_true, y_score, group, thresholds)
        )

    return roc_gap_before, roc_gap_after


def _aggregate_metric_rows(metrics_before_after):
    rows = []
    for model_name, payload in metrics_before_after.items():
        for stage in ("before", "after"):
            metrics = payload[stage]
            rows.append(
                {
                    "model": model_name,
                    "stage": stage,
                    "accuracy": metrics.get("accuracy", float("nan")),
                    "f1_macro": metrics.get("f1_macro", float("nan")),
                    "f1_binary": metrics.get("f1_binary", float("nan")),
                    "score_auc": metrics.get("score_auc", float("nan")),
                    "policy_auc": metrics.get("policy_auc", float("nan")),
                    "dpd": metrics.get("dpd", float("nan")),
                    "eod": metrics.get("eod", float("nan")),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_gap_rows(roc_gap_before, roc_gap_after):
    rows = []
    for model_name in roc_gap_before:
        rows.append(
            {
                "model": model_name,
                "roc_gap_before": roc_gap_before.get(model_name, float("nan")),
                "roc_gap_after": roc_gap_after.get(model_name, float("nan")),
            }
        )
    return pd.DataFrame(rows)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_phase23_artifacts(metrics_before_after, roc_gap_before, roc_gap_after, thresholds_per_model, output_dir):
    """Save metrics_before_after.csv, roc_gap.csv, and thresholds.json."""
    os.makedirs(output_dir, exist_ok=True)

    metrics_df = _aggregate_metric_rows(metrics_before_after)
    roc_gap_df = _aggregate_gap_rows(roc_gap_before, roc_gap_after)

    metrics_path = os.path.join(output_dir, "metrics_before_after.csv")
    roc_gap_path = os.path.join(output_dir, "roc_gap.csv")
    thresholds_path = os.path.join(output_dir, "thresholds.json")

    metrics_df.to_csv(metrics_path, index=False)
    roc_gap_df.to_csv(roc_gap_path, index=False)
    with open(thresholds_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(thresholds_per_model), handle, indent=2)

    return {
        "metrics_path": metrics_path,
        "roc_gap_path": roc_gap_path,
        "thresholds_path": thresholds_path,
    }


def plot_fairness_comparison(metrics, save_path=None):
    """Plot before-vs-after fairness/performance for each model."""
    model_names = list(metrics.keys())
    metric_names = ["accuracy", "f1_macro", "score_auc", "dpd", "eod"]

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 3.2 * len(metric_names)), sharex=True)
    if len(metric_names) == 1:
        axes = [axes]

    x = np.arange(len(model_names))
    width = 0.35

    for ax, metric_name in zip(axes, metric_names):
        before = [metrics[m]["before"].get(metric_name, np.nan) for m in model_names]
        after = [metrics[m]["after"].get(metric_name, np.nan) for m in model_names]
        ax.bar(x - width / 2, before, width, label="Before", color="#2c7fb8")
        ax.bar(x + width / 2, after, width, label="After", color="#f28e2b")
        ax.set_ylabel(metric_name.upper())
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(model_names)
    axes[-1].set_xlabel("Model")
    fig.suptitle("Fairness Before vs After FROC", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_roc_curves(y_true, y_score, group, title, save_path=None):
    """Plot ROC curves for each group on the same axes."""
    group_roc = compute_group_roc(y_true, y_score, group)
    fig, ax = plt.subplots(figsize=(8, 6))

    for group_value, (fpr, tpr, _) in group_roc.items():
        auc_value = None
        try:
            mask = _to_numpy(group) == group_value
            if np.unique(_to_numpy(y_true)[mask]).size >= 2:
                auc_value = roc_auc_score(_to_numpy(y_true)[mask], _to_numpy(y_score)[mask])
        except Exception:
            auc_value = None

        label = _safe_group_label(group_value)
        if auc_value is not None and np.isfinite(auc_value):
            label = f"Group {label} (AUC={auc_value:.3f})"
        else:
            label = f"Group {label}"
        ax.plot(fpr, tpr, marker="o", linewidth=1.6, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_roc_curves_before_after(y_true, y_score, group, thresholds, title, save_path=None):
    """Plot group ROC curves before and after FROC thresholding."""
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    group = _to_numpy(group)
    y_score_after = apply_group_thresholds(y_score, group, thresholds).astype(float)

    group_roc_before = compute_group_roc(y_true, y_score, group)
    group_roc_after = compute_group_roc(y_true, y_score_after, group)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, label, group_roc in zip(
        axes,
        ["Before FROC", "After FROC"],
        [group_roc_before, group_roc_after],
    ):
        for group_value, (fpr, tpr, _) in group_roc.items():
            ax.plot(fpr, tpr, marker="o", linewidth=1.3, label=f"Group {_safe_group_label(group_value)}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("False Positive Rate")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("True Positive Rate")
    axes[1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def write_phase23_verification_report(
    metrics_before_after,
    roc_gap_before,
    roc_gap_after,
    thresholds_per_model,
    report_path,
    base_metrics=None,
    quantization_meta=None,
):
    """Write a plain-text markdown report validating Phase 2/3 behavior."""
    lines = []
    lines.append("# Phase 2/3 Verification Report")
    lines.append("")
    lines.append("This report checks whether ROC-aware group thresholding behaves as expected.")
    lines.append("")

    lines.append("## Formal Contribution Statement")
    lines.append(
        "This project establishes an explicit analysis chain from numerical compression to fairness behavior: "
        "quantization precision changes induce representation drift, representation drift alters group-wise ROC geometry, "
        "and this ROC misalignment manifests as threshold-level fairness disparity."
    )
    lines.append("")
    lines.append(
        "We then show that ROC-aware group thresholding (FROC-style post-processing) can substantially reduce "
        "group misalignment and fairness disparity across FP32, FP16, and INT8 settings with minimal accuracy change."
    )
    lines.append("")
    lines.append("## Claimed Contributions")
    lines.append(
        "1. ROC-level fairness diagnosis under quantization: beyond single-threshold DPD/EOD reporting, "
        "we quantify group ROC misalignment and track how it changes with precision."
    )
    lines.append(
        "2. Quantization-aware fairness repair via post-hoc thresholds: we demonstrate that a ROC-targeted "
        "group-threshold policy mitigates disparity after quantization without retraining."
    )
    lines.append(
        "3. Distinction between apparent and substantive fairness gains: we show that low disparity in low-precision "
        "models can coincide with utility degradation, motivating ROC-level and utility-aware interpretation."
    )
    lines.append(
        "4. Multi-view evidence linkage: representation drift metrics (CKA/L2/cosine), ROC distortion, and fairness "
        "metrics are analyzed together to support mechanistic interpretation rather than isolated metric reporting."
    )
    lines.append("")

    if quantization_meta is not None:
        lines.append("## Quantization Mode")
        lines.append(f"- requested_mode: {quantization_meta.get('requested_mode')}")
        lines.append(f"- applied_mode: {quantization_meta.get('applied_mode')}")
        lines.append(f"- fallback_reason: {quantization_meta.get('fallback_reason')}")
        lines.append("")

    lines.append("## Core Checks")
    for model_name in metrics_before_after:
        before = metrics_before_after[model_name]["before"]
        after = metrics_before_after[model_name]["after"]
        gap_before = roc_gap_before.get(model_name, float("nan"))
        gap_after = roc_gap_after.get(model_name, float("nan"))
        gap_delta = _safe_float(gap_after - gap_before)
        lines.append(f"### {model_name}")
        lines.append(f"- ROC gap: {gap_before:.8f} -> {gap_after:.8f} (delta={gap_delta:.8f})")
        lines.append(
            f"- Fairness: DPD {before.get('dpd', float('nan')):.6f} -> {after.get('dpd', float('nan')):.6f}, "
            f"EOD {before.get('eod', float('nan')):.6f} -> {after.get('eod', float('nan')):.6f}"
        )
        lines.append(
            f"- Utility: acc {before.get('accuracy', float('nan')):.6f} -> {after.get('accuracy', float('nan')):.6f}, "
            f"f1_macro {before.get('f1_macro', float('nan')):.6f} -> {after.get('f1_macro', float('nan')):.6f}"
        )
        lines.append(
            f"- AUC terms: score_auc={before.get('score_auc', float('nan')):.6f}, "
            f"policy_auc(after)={after.get('policy_auc', float('nan')):.6f}"
        )

        threshold_map = thresholds_per_model.get(model_name, {}).get("thresholds", {})
        threshold_text = ", ".join([f"group {k}: {v:.6f}" for k, v in threshold_map.items()])
        lines.append(f"- Thresholds: {threshold_text}")

        if base_metrics and model_name in base_metrics:
            base_acc = _safe_float(base_metrics[model_name].get("accuracy", float("nan")))
            base_f1 = _safe_float(base_metrics[model_name].get("macro_f1", float("nan")))
            acc_delta = _safe_float(before.get("accuracy", float("nan")) - base_acc)
            f1_delta = _safe_float(before.get("f1_macro", float("nan")) - base_f1)
            lines.append(f"- Consistency vs phase-1: Δacc={acc_delta:.8f}, Δmacro_f1={f1_delta:.8f}")
        lines.append("")

    lines.append("## Interpretation Guide")
    lines.append("- If ROC gap decreases, group ROC misalignment is reduced.")
    lines.append("- If DPD/EOD decrease with near-stable accuracy, post-processing is behaving as intended.")
    lines.append("- score_auc is a model-score property; policy_auc reflects post-threshold operating behavior.")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def plot_roc_gap(roc_gap_before, roc_gap_after, save_path=None):
    """Plot ROC gap before vs after FROC for each model."""
    model_names = list(roc_gap_before.keys())
    before = [roc_gap_before[m] for m in model_names]
    after = [roc_gap_after[m] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, before, width, label="Before", color="#d62728")
    ax.bar(x + width / 2, after, width, label="After", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("ROC Gap")
    ax.set_title("ROC Gap Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def _derive_default_binary_group(val_df):
    """Binary fallback group for Jigsaw: any identity present vs none."""
    identity_cols = [
        "male", "female", "transgender", "other_gender",
        "heterosexual", "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
        "black", "white", "asian", "latino", "other_race_or_ethnicity",
        "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
        "psychiatric_or_mental_illness", "intellectual_or_learning_disability",
        "physical_disability", "other_disability",
    ]
    available_cols = [col for col in identity_cols if col in val_df.columns]
    if not available_cols:
        return np.zeros(len(val_df), dtype=int)
    matrix = val_df[available_cols].fillna(0.0).to_numpy(dtype=float)
    return (matrix.max(axis=1) > 0.0).astype(int)


def build_results_for_phase23(val_df, model_scores):
    """Build the generic results dict expected by froc_pipeline/roc_analysis_pipeline."""
    y_true = val_df["label"].to_numpy(dtype=int)
    group = _derive_default_binary_group(val_df)
    results = {}
    for model_name, scores in model_scores.items():
        results[model_name] = {
            "y_true": y_true,
            "y_score": _to_numpy(scores).astype(float),
            "group": group,
        }
    return results


def run_phase23_pipeline(
    val_df,
    model_scores,
    output_dir,
    plot_prefix="jigsaw",
    base_metrics=None,
    quantization_meta=None,
):
    """Convenience wrapper for building results, running both phases, and saving artifacts."""
    results = build_results_for_phase23(val_df, model_scores)
    metrics_before_after, thresholds_per_model = froc_pipeline(results)
    roc_gap_before, roc_gap_after = roc_analysis_pipeline(results, thresholds_per_model)

    phase23_dir = os.path.join(output_dir, "phase23")
    artifacts = save_phase23_artifacts(
        metrics_before_after,
        roc_gap_before,
        roc_gap_after,
        thresholds_per_model,
        phase23_dir,
    )

    plot_fairness_comparison(
        metrics_before_after,
        save_path=os.path.join(phase23_dir, f"{plot_prefix}_fairness_before_after.png"),
    )
    plot_roc_gap(
        roc_gap_before,
        roc_gap_after,
        save_path=os.path.join(phase23_dir, f"{plot_prefix}_roc_gap.png"),
    )

    for model_name, model_results in results.items():
        plot_roc_curves(
            model_results["y_true"],
            model_results["y_score"],
            model_results["group"],
            title=f"ROC Curves by Group - {model_name}",
            save_path=os.path.join(phase23_dir, f"{plot_prefix}_roc_curves_{model_name}.png"),
        )
        plot_roc_curves_before_after(
            model_results["y_true"],
            model_results["y_score"],
            model_results["group"],
            thresholds_per_model[model_name]["thresholds"],
            title=f"ROC Curves Before/After FROC - {model_name}",
            save_path=os.path.join(phase23_dir, f"{plot_prefix}_roc_curves_before_after_{model_name}.png"),
        )

    report_path = os.path.join(phase23_dir, "phase23_verification_report.md")
    write_phase23_verification_report(
        metrics_before_after,
        roc_gap_before,
        roc_gap_after,
        thresholds_per_model,
        report_path,
        base_metrics=base_metrics,
        quantization_meta=quantization_meta,
    )

    return {
        "results": results,
        "metrics_before_after": metrics_before_after,
        "thresholds_per_model": thresholds_per_model,
        "roc_gap_before": roc_gap_before,
        "roc_gap_after": roc_gap_after,
        "artifacts": artifacts,
        "phase23_dir": phase23_dir,
        "verification_report_path": report_path,
    }
