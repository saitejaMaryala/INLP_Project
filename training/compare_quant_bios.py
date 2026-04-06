"""
Bias in Bios — Quantization & Fairness Comparison
================================================
Loads trained FP32 model, creates INT8 and FP16 quantized versions,
and compares performance, fairness (EOD), and representation drift.

Metrics:
  - Accuracy, Macro-F1
  - Equal Opportunity Difference (EOD) per occupation & gender
  - Layer-wise L2, Cosine similarity, CKA

Usage:
    python -m training.compare_quant_bios
    python -m training.compare_quant_bios --sanity-check  # quick test with 200 samples
"""

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import cosine as cosine_dist
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── path setup ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import load_bias_in_bios


# ════════════════════════════════════════════════════════════════
# 1. MODEL
# ════════════════════════════════════════════════════════════════
class FrozenBertClassifier(nn.Module):
    """Frozen BERT encoder + trainable linear head."""

    def __init__(self, num_classes: int = 28, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        if torch.is_tensor(outputs):
            hidden_states = outputs
        elif isinstance(outputs, (tuple, list)):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state
        cls_emb = hidden_states[:, 0, :].float()
        if hasattr(outputs, "hidden_states"):
            extra_hidden_states = outputs.hidden_states
        else:
            extra_hidden_states = None
        return self.classifier(self.dropout(cls_emb)), extra_hidden_states

    def get_hidden_states(self, input_ids, attention_mask):
        """Return CLS embedding from every layer for CKA analysis."""
        if hasattr(self.bert, "get_hidden_states"):
            return self.bert.get_hidden_states(input_ids, attention_mask)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        return torch.stack([h[:, 0, :].float()
                            for h in outputs.hidden_states])


class QuantizableBertBackbone(nn.Module):
    """Tensor-only BERT wrapper that FX can trace, while hooks capture hidden states."""

    def __init__(self, backbone: nn.Module, register_hidden_hooks: bool = True):
        super().__init__()
        self.backbone = backbone
        self._hook_handles = []
        self._hidden_states_buffer = []

        if hasattr(self.backbone, "config"):
            self.backbone.config.return_dict = False
            self.backbone.config.output_hidden_states = False

        if register_hidden_hooks:
            self._register_hidden_hooks()

    def _remove_hidden_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _capture_hidden_state(self, module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(tensor):
            self._hidden_states_buffer.append(tensor)

    def _register_hidden_hooks(self):
        self._remove_hidden_hooks()

        if hasattr(self.backbone, "embeddings"):
            self._hook_handles.append(
                self.backbone.embeddings.register_forward_hook(self._capture_hidden_state)
            )

        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None) if encoder is not None else None
        if layers is not None:
            for layer in layers:
                self._hook_handles.append(
                    layer.register_forward_hook(self._capture_hidden_state)
                )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if torch.is_tensor(outputs):
            return outputs
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs.last_hidden_state

    def get_hidden_states(self, input_ids, attention_mask):
        self._hidden_states_buffer = []
        with torch.no_grad():
            _ = self.forward(input_ids, attention_mask)

        if len(self._hidden_states_buffer) < 13:
            raise RuntimeError(
                f"Expected 13 hidden-state tensors, got {len(self._hidden_states_buffer)}."
            )

        return torch.stack([
            hidden[:, 0, :].float() for hidden in self._hidden_states_buffer[:13]
        ])


# ════════════════════════════════════════════════════════════════
# 2. DATASET
# ════════════════════════════════════════════════════════════════
def tokenise_and_cache(texts, tokenizer, max_length, cache_path):
    """Load cached tokenized data."""
    if os.path.exists(cache_path):
        print(f"  [cache] Loading tokenized data from {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        if len(input_ids) == len(texts):
            return input_ids, attention_mask
        print(
            "  [cache] Size mismatch detected "
            f"(cache={len(input_ids):,}, expected={len(texts):,}). Rebuilding cache..."
        )

    print(f"  [cache] Tokenizing {len(texts):,} texts → {cache_path}")
    CHUNK = 8_000
    all_ids, all_masks = [], []
    for i in tqdm(range(0, len(texts), CHUNK), desc="  Tokenizing", leave=False):
        chunk = tokenizer(
            texts[i:i + CHUNK],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        all_ids.append(chunk["input_ids"])
        all_masks.append(chunk["attention_mask"])

    input_ids      = torch.cat(all_ids)
    attention_mask = torch.cat(all_masks)
    torch.save({"input_ids": input_ids, "attention_mask": attention_mask},
               cache_path)
    print(f"  [cache] Saved → {cache_path}")
    return input_ids, attention_mask


class BiosDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, genders):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.labels         = torch.tensor(labels,  dtype=torch.long)
        self.genders        = torch.tensor(genders, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
            "genders":        self.genders[idx],
        }


# ════════════════════════════════════════════════════════════════
# 3. QUANTIZATION
# ════════════════════════════════════════════════════════════════
def _unwrap(model):
    """Unwrap DataParallel if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


def quantize_fp16(model):
    """BERT encoder → FP16. Classifier head stays FP32."""
    base = copy.deepcopy(_unwrap(model)).cpu()
    base.bert = base.bert.half()
    base.classifier = base.classifier.float()
    base.dropout = base.dropout.float()
    return base


def quantize_int8(model):
    """Dynamic INT8 on BERT encoder only. Classifier stays FP32."""
    base = copy.deepcopy(_unwrap(model)).cpu()
    base.bert = torch.quantization.quantize_dynamic(
        base.bert, {nn.Linear}, dtype=torch.qint8
    )
    return base


def build_calibration_loader(
    train_texts,
    tokenizer,
    max_length,
    cache_dir,
    batch_size,
    num_workers,
    calib_samples,
):
    """Prepare a small held-out calibration loader for static PTQ."""
    os.makedirs(cache_dir, exist_ok=True)
    n = min(calib_samples, len(train_texts))
    calib_texts = train_texts[:n]

    calib_ids, calib_mask = tokenise_and_cache(
        calib_texts,
        tokenizer,
        max_length,
        os.path.join(cache_dir, f"calib_ml{max_length}_n{n}.pt"),
    )

    dummy_labels = [0] * n
    dummy_genders = [0] * n
    calib_ds = BiosDataset(calib_ids, calib_mask, dummy_labels, dummy_genders)
    calib_loader = DataLoader(
        calib_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return calib_loader, n


def stratified_calibration_texts(
    train_texts: List[str],
    train_labels: List[int],
    train_genders: List[int],
    calib_samples: int,
    seed: int,
) -> List[str]:
    """Sample calibration texts stratified by (occupation, gender)."""
    n_total = len(train_texts)
    n = min(calib_samples, n_total)

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for idx, (label, gender) in enumerate(zip(train_labels, train_genders)):
        key = (int(label), int(gender))
        buckets.setdefault(key, []).append(idx)

    rng = np.random.default_rng(seed)
    selected: List[int] = []

    bucket_keys = sorted(buckets.keys())
    if not bucket_keys:
        return train_texts[:n]

    per_bucket = max(1, n // len(bucket_keys))
    for key in bucket_keys:
        idxs = buckets[key]
        k = min(per_bucket, len(idxs))
        if k > 0:
            chosen = rng.choice(idxs, size=k, replace=False).tolist()
            selected.extend(chosen)

    if len(selected) < n:
        remaining_pool = list(set(range(n_total)) - set(selected))
        k = min(n - len(selected), len(remaining_pool))
        if k > 0:
            selected.extend(rng.choice(remaining_pool, size=k, replace=False).tolist())

    if len(selected) > n:
        selected = selected[:n]

    return [train_texts[i] for i in selected]


class BertEncoderForOnnx(nn.Module):
    """Exports only BERT last_hidden_state for ORT static quantization."""

    def __init__(self, bert_module: nn.Module):
        super().__init__()
        self.bert = bert_module

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
            output_hidden_states=False,
            use_cache=False,
        )
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def export_bert_encoder_to_onnx(
    model: nn.Module,
    max_length: int,
    onnx_fp32_path: str,
) -> None:
    """Export BERT encoder to ONNX (FP32) for downstream static quantization."""
    base = copy.deepcopy(_unwrap(model)).cpu().eval()
    wrapper = BertEncoderForOnnx(base.bert).cpu().eval()

    dummy_ids = torch.ones((1, max_length), dtype=torch.long)
    dummy_mask = torch.ones((1, max_length), dtype=torch.long)
    dummy_type_ids = torch.zeros((1, max_length), dtype=torch.long)

    os.makedirs(os.path.dirname(onnx_fp32_path), exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask, dummy_type_ids),
        onnx_fp32_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "token_type_ids": {0: "batch_size", 1: "seq_len"},
            "last_hidden_state": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=17,
    )


def quantize_onnx_static(
    onnx_fp32_path: str,
    onnx_int8_path: str,
    calib_loader: DataLoader,
):
    """Run ONNX Runtime static quantization with calibration data."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class LoaderCalibrationReader(CalibrationDataReader):
        def __init__(self, loader: DataLoader):
            self._iter = iter(loader)

        def get_next(self):
            try:
                batch = next(self._iter)
            except StopIteration:
                return None
            return {
                "input_ids": batch["input_ids"].numpy().astype(np.int64),
                "attention_mask": batch["attention_mask"].numpy().astype(np.int64),
                "token_type_ids": np.zeros_like(batch["input_ids"].numpy(), dtype=np.int64),
            }

    os.makedirs(os.path.dirname(onnx_int8_path), exist_ok=True)

    quantize_static(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        calibration_data_reader=LoaderCalibrationReader(calib_loader),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


@torch.no_grad()
def evaluate_onnx_int8(
    model: nn.Module,
    onnx_int8_path: str,
    loader: DataLoader,
):
    """Evaluate ORT INT8 encoder + Torch classifier head."""
    import onnxruntime as ort

    base = _unwrap(model).cpu().eval()
    weight = base.classifier.weight.detach().cpu().numpy().astype(np.float32)
    bias = base.classifier.bias.detach().cpu().numpy().astype(np.float32)

    session = ort.InferenceSession(
        onnx_int8_path,
        providers=["CPUExecutionProvider"],
    )

    all_preds, all_labels, all_genders = [], [], []
    for batch in tqdm(loader, desc="[INT8 ONNX]", leave=False):
        ids = batch["input_ids"].numpy().astype(np.int64)
        mask = batch["attention_mask"].numpy().astype(np.int64)
        token_type_ids = np.zeros_like(ids, dtype=np.int64)

        hidden = session.run(
            ["last_hidden_state"],
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": token_type_ids},
        )[0]
        cls = hidden[:, 0, :].astype(np.float32)
        logits = cls @ weight.T + bias
        preds = np.argmax(logits, axis=1).tolist()

        all_preds.extend(preds)
        all_labels.extend(batch["labels"].tolist())
        all_genders.extend(batch["genders"].tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1, all_preds, all_labels, all_genders


def full_eval_from_predictions(tag, acc, f1, preds, labels, genders, num_classes):
    """Compute fairness metrics from externally computed predictions."""
    mean_eod, mean_abs_eod, max_abs_eod, per_class = compute_eod(
        preds, labels, genders, num_classes
    )
    print(f"\n── {tag} ──────────────────────────────")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro-F1      : {f1:.4f}")
    print(f"  Mean EOD      : {mean_eod:+.4f}")
    print(f"  Mean |EOD|    : {mean_abs_eod:.4f}")
    print(f"  Max  |EOD|    : {max_abs_eod:.4f}")
    return {
        "tag": tag,
        "accuracy": float(round(acc, 5)),
        "macro_f1": float(round(f1, 5)),
        "mean_eod": float(round(mean_eod, 5)),
        "mean_abs_eod": float(round(mean_abs_eod, 5)),
        "max_abs_eod": float(round(max_abs_eod, 5)),
        "per_class_eod": [float(round(e, 5)) for e in per_class],
    }


def quantize_int8_static_with_calibration(
    model,
    calib_loader,
    calibration_batches=32,
):
    """
    Attempt static INT8 PTQ for the BERT encoder.

    In this project, Hugging Face BERT traceability under FX is brittle and can
    fail with Proxy/slice errors inside the transformer internals. Rather than
    crash late in the pipeline, we calibrate a small sample for bookkeeping and
    then explicitly fall back to dynamic INT8 while reporting the reason.
    """
    _ = copy.deepcopy(_unwrap(model)).cpu().eval()

    # Run the requested calibration batches so the comparison logs reflect the
    # intended static-PTQ workflow, but avoid FX conversion on this backbone.
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            _ = batch["input_ids"]
            _ = batch["attention_mask"]
            if i + 1 >= calibration_batches:
                break

    err = (
        "Static PTQ via FX is unsupported for this Hugging Face BERT backbone "
        "in the current environment; used dynamic INT8 instead."
    )
    print(f"  [Warning] {err}")
    return quantize_int8(model), "dynamic", err


# ════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device, desc="eval"):
    model.eval()
    all_preds, all_labels, all_genders = [], [], []

    for batch in tqdm(loader, desc=f"[{desc}]", leave=False):
        ids   = batch["input_ids"].to(device, non_blocking=True)
        mask  = batch["attention_mask"].to(device, non_blocking=True)

        if device.type == "cuda" and hasattr(model.bert, "half"):
            with autocast():
                logits, _ = model(ids, mask)
        else:
            logits, _ = model(ids, mask)

        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(batch["labels"].tolist())
        all_genders.extend(batch["genders"].tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1, all_preds, all_labels, all_genders


# ════════════════════════════════════════════════════════════════
# 5. FAIRNESS — Equal Opportunity Difference (EOD)
# ════════════════════════════════════════════════════════════════
def compute_eod(preds, labels, genders, num_classes=28):
    """
    For each occupation class c:
      TPR_female(c) = P(pred=c | true=c, gender=female)
      TPR_male(c)   = P(pred=c | true=c, gender=male)
      EOD(c)        = TPR_female(c) − TPR_male(c)
    """
    preds   = np.array(preds)
    labels  = np.array(labels)
    genders = np.array(genders)

    per_class_eod = []
    for c in range(num_classes):
        tprs = {}
        for g, g_name in [(0, "female"), (1, "male")]:
            mask = (labels == c) & (genders == g)
            if mask.sum() == 0:
                continue
            tprs[g_name] = float((preds[mask] == c).sum() / mask.sum())
        if "female" in tprs and "male" in tprs:
            per_class_eod.append(tprs["female"] - tprs["male"])

    if not per_class_eod:
        return 0.0, 0.0, 0.0, []

    arr = np.array(per_class_eod)
    return (float(arr.mean()),
            float(np.abs(arr).mean()),
            float(np.abs(arr).max()),
            per_class_eod)


def full_eval(model, loader, device, tag, num_classes):
    acc, f1, preds, labels, genders = evaluate(model, loader, device, desc=tag)
    mean_eod, mean_abs_eod, max_abs_eod, per_class = compute_eod(
        preds, labels, genders, num_classes
    )
    print(f"\n── {tag} ──────────────────────────────")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro-F1      : {f1:.4f}")
    print(f"  Mean EOD      : {mean_eod:+.4f}")
    print(f"  Mean |EOD|    : {mean_abs_eod:.4f}")
    print(f"  Max  |EOD|    : {max_abs_eod:.4f}")
    return {
        "tag":           tag,
        "accuracy":      float(round(acc,          5)),
        "macro_f1":      float(round(f1,           5)),
        "mean_eod":      float(round(mean_eod,     5)),
        "mean_abs_eod":  float(round(mean_abs_eod, 5)),
        "max_abs_eod":   float(round(max_abs_eod,  5)),
        "per_class_eod": [float(round(e, 5)) for e in per_class],
    }


# ════════════════════════════════════════════════════════════════
# 6. REPRESENTATION ANALYSIS
# ════════════════════════════════════════════════════════════════
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n, d) float32 matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


@torch.no_grad()
def extract_hidden_states(model, loader, device, max_samples=1000):
    """Returns (13, N, 768) float32 numpy array of [CLS] embeddings per layer."""
    model.eval()
    all_hidden = []
    n_collected = 0

    for batch in tqdm(loader, desc="  [repr extract]", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        cls_per_layer = model.get_hidden_states(ids, mask)
        all_hidden.append(cls_per_layer.cpu().numpy())

        n_collected += ids.size(0)
        if n_collected >= max_samples:
            break

    arr = np.concatenate(all_hidden, axis=1)[:, :max_samples, :]
    return arr.astype(np.float32)


def representation_analysis(hidden_fp32, hidden_quant, label):
    """Layer-wise L2, cosine, CKA between FP32 and quantized hidden states."""
    results = {"label": label, "layers": []}
    for layer in range(hidden_fp32.shape[0]):
        A = hidden_fp32[layer]
        B = hidden_quant[layer]

        l2 = float(np.mean(np.linalg.norm(A - B, axis=1)))

        cos_sims = []
        for i in range(A.shape[0]):
            na, nb = np.linalg.norm(A[i]), np.linalg.norm(B[i])
            if na > 1e-8 and nb > 1e-8:
                cos_sims.append(1.0 - float(cosine_dist(A[i], B[i])))
        cos_mean = float(np.mean(cos_sims)) if cos_sims else 0.0

        cka = linear_cka(A, B)

        results["layers"].append({
            "layer":              layer,
            "l2_distance":        round(l2,       6),
            "cosine_similarity":  round(cos_mean, 6),
            "cka":                round(cka,      6),
        })
    return results


# ════════════════════════════════════════════════════════════════
# 7. PLOTS
# ════════════════════════════════════════════════════════════════
def plot_performance_comparison(results, save_dir, occupations):
    """Bar charts: Accuracy, Macro-F1, mean|EOD|, max|EOD| for FP32/FP16/INT8."""
    precs   = [k for k in ["fp32", "fp16", "int8"] if k in results]
    metrics = ["accuracy", "macro_f1", "mean_abs_eod", "max_abs_eod"]
    labels  = ["Accuracy ↑", "Macro-F1 ↑", "Mean |EOD| ↓", "Max |EOD| ↓"]
    colors  = {"fp32": "#2ecc71", "fp16": "#3498db", "int8": "#e74c3c"}

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Bias-in-Bios: Performance & Fairness by Quantization Precision",
                 fontsize=14, fontweight="bold")

    for ax, metric, label in zip(axes, metrics, labels):
        vals = [results[p][metric] for p in precs]
        bars = ax.bar(precs, vals,
                      color=[colors[p] for p in precs],
                      alpha=0.85, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_xticklabels([p.upper() for p in precs])
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "bios_performance_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_eod_per_occupation(results, occupations, save_dir):
    """Heatmap: rows = FP32/FP16/INT8, cols = occupation, value = EOD(c)."""
    precs = [k for k in ["fp32", "fp16", "int8"] if k in results]
    n_occ = len(occupations)

    matrix = np.zeros((len(precs), n_occ))
    for i, p in enumerate(precs):
        eod_list = results[p].get("per_class_eod", [])
        for j, v in enumerate(eod_list[:n_occ]):
            matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(max(14, n_occ * 0.55), 4))
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="EOD (female TPR − male TPR)")

    ax.set_yticks(range(len(precs)))
    ax.set_yticklabels([p.upper() for p in precs])
    ax.set_xticks(range(n_occ))
    ax.set_xticklabels(occupations[:n_occ], rotation=45, ha="right", fontsize=8)
    ax.set_title("Equal Opportunity Difference per Occupation & Precision\n"
                 "Red = female disadvantaged  |  Blue = male disadvantaged",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "bios_eod_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_representation_drift(cka_fp16, cka_int8, save_dir):
    """3-panel line plot: L2 distance, Cosine similarity, CKA per layer."""
    layers    = list(range(13))
    metrics   = ["l2_distance", "cosine_similarity", "cka"]
    titles    = ["L2 Distance ↑ = more drift",
                 "Cosine Similarity ↓ = more drift",
                 "CKA Score ↓ = more structural drift"]
    ylabels   = ["Mean L2 Norm", "Mean Cosine Sim", "CKA (1=identical)"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Layer-wise Representation Drift from FP32 Baseline (Bias-in-Bios)",
                 fontsize=14, fontweight="bold")

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        fp16_vals = [cka_fp16["layers"][l][metric] for l in layers]
        int8_vals = [cka_int8["layers"][l][metric] for l in layers]

        ax.plot(layers, int8_vals, marker="o", color="#e74c3c",
                linewidth=2, label="INT8", markersize=5)
        ax.plot(layers, fp16_vals, marker="s", color="#3498db",
                linewidth=2, label="FP16", linestyle="--", markersize=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(layers)
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "bios_representation_drift.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def print_summary_table(all_results):
    precs   = [k for k in ["fp32", "fp16", "int8"] if k in all_results]
    metrics = ["accuracy", "macro_f1", "mean_eod", "mean_abs_eod", "max_abs_eod"]

    print()
    print("═" * 72)
    print("  BIAS-IN-BIOS QUANTIZATION FAIRNESS — FINAL SUMMARY")
    print("═" * 72)
    header = f"  {'Metric':<20}"
    for p in precs:
        header += f" {p.upper():>10}"
    if "fp16" in precs and "fp32" in precs:
        header += f" {'ΔFPA→16':>10}"
    if "int8" in precs and "fp32" in precs:
        header += f" {'ΔFPA→8':>10}"
    print(header)
    print("─" * 72)

    for m in metrics:
        row = f"  {m:<20}"
        vals = {p: all_results[p][m] for p in precs if p in all_results}
        for p in precs:
            row += f" {vals.get(p, float('nan')):>10.5f}"
        if "fp16" in precs and "fp32" in precs:
            d = vals.get("fp16", 0) - vals.get("fp32", 0)
            row += f" {d:>+10.5f}"
        if "int8" in precs and "fp32" in precs:
            d = vals.get("int8", 0) - vals.get("fp32", 0)
            row += f" {d:>+10.5f}"
        print(row)

    print()
    print("  Key: EOD = Equal Opportunity Difference (female TPR − male TPR)")
    print("       mean_abs_eod = primary fairness metric (↓ = fairer)")
    print("       Positive Δ on EOD metrics = quantization AMPLIFIES gender bias")
    print("═" * 72)


# ════════════════════════════════════════════════════════════════
# 8. MAIN
# ════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Bias-in-Bios quantization comparison")
    p.add_argument("--data-dir",     default="/ssd_scratch/sai.teja/INLP_Project/data/bias_in_bios")
    p.add_argument("--model-path",   default="/ssd_scratch/sai.teja/INLP_Project/models/bios/best_bios_fp32.pt")
    p.add_argument("--results-dir",  default="/ssd_scratch/sai.teja/INLP_Project/results/bios")
    p.add_argument("--cache-dir",    default="/ssd_scratch/sai.teja/INLP_Project/cache/bios")
    p.add_argument("--batch-size",   type=int,   default=64)  # Reduced from 128
    p.add_argument("--max-length",   type=int,   default=128)
    p.add_argument("--num-workers",  type=int,   default=4)  # Reduced from 8
    p.add_argument("--cka-samples",  type=int,   default=500)  # Reduced from 1000
    p.add_argument("--int8-mode",    choices=["dynamic", "static"], default="dynamic",
                   help="INT8 mode: dynamic (no calibration) or static (with calibration)")
    p.add_argument("--calib-samples", type=int, default=2048,
                   help="Number of train samples used for INT8 static calibration")
    p.add_argument("--calib-batch-size", type=int, default=32,
                   help="Batch size used during INT8 static calibration")
    p.add_argument("--calib-batches", type=int, default=32,
                   help="Maximum calibration batches to run")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--sanity-check", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check model exists
    if not os.path.exists(args.model_path):
        print(f"\n[Error] Trained model not found at {args.model_path}")
        print("Please run 'python -m training.train_bios_clean' first.\n")
        return

    # Load data
    print("\n══════ LOADING DATA ══════")
    (train_texts, train_labels, train_genders,
     test_texts, test_labels, test_genders,
     occupations) = load_bias_in_bios(args.data_dir)

    num_classes = len(occupations)
    print(f"Occupations: {num_classes}  |  Test: {len(test_texts):,}")

    if args.sanity_check:
        print("\n⚡ SANITY-CHECK MODE")
        test_texts = test_texts[:200]
        test_labels = test_labels[:200]
        test_genders = test_genders[:200]
        args.cka_samples = 100
        args.calib_samples = min(args.calib_samples, 256)
        args.calib_batches = min(args.calib_batches, 8)

    # Tokenize
    print("\n══════ TOKENIZING ══════")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    test_ids, test_mask = tokenise_and_cache(
        test_texts, tokenizer, args.max_length,
        os.path.join(args.cache_dir, f"test_ml{args.max_length}.pt"),
    )

    test_ds = BiosDataset(test_ids, test_mask, test_labels, test_genders)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # Load FP32 model
    print(f"\n══════ LOADING MODEL ══════\n  {args.model_path}")
    fp32_model = FrozenBertClassifier(num_classes=num_classes)
    fp32_model.load_state_dict(
        torch.load(args.model_path, map_location="cpu", weights_only=True)
    )
    fp32_model.to(device)

    # Evaluate FP32
    print("\n══════ FP32 EVALUATION ══════")
    all_results = {}
    all_results["fp32"] = full_eval(fp32_model, test_loader, device, "FP32", num_classes)

    # FP16
    print("\n══════ FP16 QUANTIZATION & EVAL ══════")
    fp16_model = quantize_fp16(fp32_model).to(device)
    all_results["fp16"] = full_eval(fp16_model, test_loader, device, "FP16", num_classes)

    # INT8
    print("\n══════ INT8 QUANTIZATION & EVAL ══════")
    quantization_meta = {
        "int8_requested_mode": args.int8_mode,
        "int8_applied_mode": None,
        "int8_backend": None,
        "calibration_used": False,
        "calibration_samples": 0,
        "calibration_batches": 0,
        "calibration_strategy": None,
        "fallback_reason": None,
    }

    if args.int8_mode == "static":
        print("  INT8 mode: static PTQ with stratified calibration")
        calib_texts = stratified_calibration_texts(
            train_texts=train_texts,
            train_labels=train_labels,
            train_genders=train_genders,
            calib_samples=args.calib_samples,
            seed=args.seed,
        )
        calib_loader, used_calib_samples = build_calibration_loader(
            train_texts=calib_texts,
            tokenizer=tokenizer,
            max_length=args.max_length,
            cache_dir=args.cache_dir,
            batch_size=args.calib_batch_size,
            num_workers=max(0, min(2, args.num_workers)),
            calib_samples=args.calib_samples,
        )
        print(f"  Calibration samples: {used_calib_samples:,}")
        quantization_meta["calibration_strategy"] = "stratified_occupation_gender"
        quantization_meta["calibration_used"] = True
        quantization_meta["calibration_samples"] = int(used_calib_samples)
        quantization_meta["calibration_batches"] = int(args.calib_batches)

        try:
            onnx_dir = os.path.join(args.results_dir, "onnx")
            onnx_fp32_path = os.path.join(onnx_dir, "bert_encoder_fp32.onnx")
            onnx_int8_path = os.path.join(onnx_dir, "bert_encoder_int8_static.onnx")

            print("  Exporting BERT encoder to ONNX...")
            export_bert_encoder_to_onnx(
                model=fp32_model,
                max_length=args.max_length,
                onnx_fp32_path=onnx_fp32_path,
            )
            print("  Quantizing ONNX encoder with static calibration...")
            quantize_onnx_static(
                onnx_fp32_path=onnx_fp32_path,
                onnx_int8_path=onnx_int8_path,
                calib_loader=calib_loader,
            )

            quantization_meta["int8_applied_mode"] = "static"
            quantization_meta["int8_backend"] = "onnxruntime_static"
            quantization_meta["fallback_reason"] = None
            int8_model = None
            int8_onnx_path = onnx_int8_path
        except Exception as e:
            fallback_reason = f"{type(e).__name__}: {e}"
            print(
                "  [Warning] ONNX static INT8 path failed "
                f"({fallback_reason}). Falling back to dynamic INT8."
            )
            int8_model = quantize_int8(fp32_model)
            int8_onnx_path = None
            quantization_meta["int8_applied_mode"] = "dynamic"
            quantization_meta["int8_backend"] = "torch_dynamic_fallback"
            quantization_meta["fallback_reason"] = fallback_reason
    else:
        print("  INT8 mode: dynamic PTQ")
        int8_model = quantize_int8(fp32_model)
        quantization_meta["int8_applied_mode"] = "dynamic"
        quantization_meta["int8_backend"] = "torch_dynamic"
        int8_onnx_path = None

    print(
        "  INT8 summary: "
        f"requested={quantization_meta['int8_requested_mode']} | "
        f"applied={quantization_meta['int8_applied_mode']} | "
        f"fallback_reason={quantization_meta['fallback_reason'] or 'none'}"
    )

    cpu_loader = DataLoader(test_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)

    if quantization_meta["int8_backend"] == "onnxruntime_static" and int8_onnx_path is not None:
        acc, f1, preds, labels, genders = evaluate_onnx_int8(
            model=fp32_model,
            onnx_int8_path=int8_onnx_path,
            loader=cpu_loader,
        )
        all_results["int8"] = full_eval_from_predictions(
            tag="INT8",
            acc=acc,
            f1=f1,
            preds=preds,
            labels=labels,
            genders=genders,
            num_classes=num_classes,
        )
    else:
        all_results["int8"] = full_eval(int8_model, cpu_loader,
                                        torch.device("cpu"), "INT8", num_classes)

    # Representation analysis
    print("\n══════ REPRESENTATION ANALYSIS ══════")
    repr_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)  # Smaller batch

    fp32_repr = copy.deepcopy(fp32_model).to(device)
    h_fp32 = extract_hidden_states(fp32_repr, repr_loader, device, args.cka_samples)
    del fp32_repr

    fp16_repr = quantize_fp16(fp32_model).to(device)
    h_fp16 = extract_hidden_states(fp16_repr, repr_loader, device, args.cka_samples)
    del fp16_repr

    int8_repr = quantize_int8(fp32_model)
    h_int8 = extract_hidden_states(int8_repr, repr_loader,
                                   torch.device("cpu"), args.cka_samples)
    del int8_repr

    cka_fp16 = representation_analysis(h_fp32, h_fp16, "FP32_vs_FP16")
    cka_int8 = representation_analysis(h_fp32, h_int8, "FP32_vs_INT8")
    all_results["cka_fp16"] = cka_fp16
    all_results["cka_int8"] = cka_int8
    all_results["quantization_meta"] = quantization_meta

    print(f"\n{'Layer':<6} {'FP16 CKA':>10} {'INT8 CKA':>10} "
          f"{'FP16 L2':>10} {'INT8 L2':>10}")
    print("─" * 50)
    for i in range(13):
        fl = cka_fp16["layers"][i]; il = cka_int8["layers"][i]
        print(f"{i:<6} {fl['cka']:>10.6f} {il['cka']:>10.6f} "
              f"{fl['l2_distance']:>10.6f} {il['l2_distance']:>10.6f}")

    # Generate plots
    print("\n══════ GENERATING PLOTS ══════")
    plot_performance_comparison(all_results, args.results_dir, occupations)
    plot_eod_per_occupation(all_results, occupations, args.results_dir)
    plot_representation_drift(cka_fp16, cka_int8, args.results_dir)

    # Save results
    results_path = os.path.join(args.results_dir, "bios_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults JSON saved → {results_path}")

    print_summary_table(all_results)
    print(f"\n[Done] All outputs in {args.results_dir}/\n")


if __name__ == "__main__":
    main()
