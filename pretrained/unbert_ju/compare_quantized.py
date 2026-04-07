"""
BERT Quantization & Fairness Comparison Script
==============================================
Loads trained FP32 BERT model, creates INT8 and FP16 quantized versions,
and compares their performance on toxicity classification fairness metrics.

Evaluates:
  - Demographic Parity Difference (DPD)
  - Equal Opportunity Difference (EOD)
  - Layer-wise L2, Cosine Similarity, CKA representation drift

Install dependencies:
  pip install torch transformers fairlearn pandas scikit-learn tqdm matplotlib
"""

# ─────────────────────────────────────────────
# 0. IMPORTS & CONFIG
# ─────────────────────────────────────────────
import copy
import json
import os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.metrics import demographic_parity_difference, equal_opportunity_difference
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pretrained.unbert_ju.phase23_pipeline import run_phase23_pipeline

warnings.filterwarnings("ignore")

# ── Reproducibility ──
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Paths ──
DATA_DIR        = os.environ.get("DATA_DIR", "data/jigsaw_uni")
TRAIN_CSV       = TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
MODEL_PATH      = "outputs/fp32_classifier_uni.pt"
OUTPUT_DIR      = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters ──
MAX_LEN         = 128
BATCH_SIZE      = 32
TOXICITY_THRESH = 0.5
TRAIN_SAMPLE    = 80_000
VAL_SAMPLE      = 20_000
REPR_SAMPLE     = 1_000        # fixed sample for CKA / L2 analysis

# INT8 quantization mode:
#   "dynamic" = weight-only PTQ (no calibration)
#   "static"  = activation-aware PTQ with calibration pass
INT8_MODE        = "static"
CALIB_SAMPLE     = 8192
CALIB_BATCH_SIZE = 16
CALIB_BATCHES    = 128

# ── Sensitive identity columns in Jigsaw ──
IDENTITY_COLS = [
    "male", "female", "transgender", "other_gender",
    "heterosexual", "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "psychiatric_or_mental_illness", "intellectual_or_learning_disability", 
    "physical_disability", "other_disability",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_jigsaw(path, n_train=TRAIN_SAMPLE, n_val=VAL_SAMPLE, seed=SEED):
    """Load, binarise, and split the Jigsaw train CSV."""
    print(f"\n[Data] Loading {path} …")
    df = pd.read_csv(path)

    # Binarize target column (continuous toxicity score → binary)
    df["label"] = (df["target"] >= TOXICITY_THRESH).astype(int)

    # Binarise identity columns
    for col in IDENTITY_COLS:
        if col in df.columns:
            df[col] = (df[col] >= 0.5).astype(float)
        else:
            df[col] = np.nan

    df = df[["comment_text", "label"] + [c for c in IDENTITY_COLS if c in df.columns]]
    df = df.dropna(subset=["comment_text"]).reset_index(drop=True)

    # Shuffle and split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_total = (n_train or len(df)) + (n_val or 0)
    df = df.iloc[:n_total] if n_total < len(df) else df

    split = n_train if n_train else int(0.8 * len(df))
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df   = df.iloc[split:].reset_index(drop=True)

    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}")
    print(f"  Toxic rate — train: {train_df['label'].mean():.3f}  val: {val_df['label'].mean():.3f}")
    return train_df, val_df


class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts  = df["comment_text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class FrozenBertClassifier(nn.Module):
    """
    BERT encoder (frozen) → [CLS] embedding → Dropout → Linear head (FP32).
    output_hidden_states=True so we can extract all 12 layer representations.
    """
    def __init__(self, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            output_hidden_states=True,
        )
        # Freeze entire BERT encoder
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        logits  = self.classifier(self.dropout(cls_emb))
        return logits, outputs.hidden_states


# ─────────────────────────────────────────────
# 3. QUANTIZATION
# ─────────────────────────────────────────────

def build_calibration_loader(train_df, tokenizer, max_len=MAX_LEN, batch_size=CALIB_BATCH_SIZE,
                             calib_samples=CALIB_SAMPLE):
    """Create a held-out calibration loader for static PTQ."""
    n = min(calib_samples, len(train_df))
    calib_df = train_df.iloc[:n].reset_index(drop=True)
    calib_ds = JigsawDataset(calib_df, tokenizer, max_len=max_len)
    calib_dl = DataLoader(calib_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return calib_dl, n


def apply_int8_quantization(fp32_model, mode="dynamic", calib_loader=None,
                            calibration_batches=CALIB_BATCHES):
    """
    INT8 PTQ on the frozen BERT encoder.
    mode="dynamic" uses weight-only dynamic quantization.
    mode="static" runs FX prepare/convert with a calibration pass.
    """
    base_model = copy.deepcopy(fp32_model).cpu().eval()

    if mode == "dynamic":
        print("\n[Quantize] Applying INT8 dynamic quantization …")
        base_model.bert = torch.quantization.quantize_dynamic(
            base_model.bert,
            {nn.Linear},
            dtype=torch.qint8,
        )
        print("  INT8 model ready.")
        return base_model, {
            "requested_mode": mode,
            "applied_mode": "dynamic",
            "fallback_reason": None,
        }

    if calib_loader is None:
        raise ValueError("Static INT8 mode requires calib_loader.")

    print("\n[Quantize] Applying INT8 static PTQ with calibration …")
    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

        first_batch = next(iter(calib_loader))
        example_inputs = (first_batch["input_ids"], first_batch["attention_mask"])
        qconfig_mapping = get_default_qconfig_mapping("fbgemm")

        prepared_bert = prepare_fx(
            base_model.bert,
            qconfig_mapping,
            example_inputs=example_inputs,
        )

        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                prepared_bert(batch["input_ids"], batch["attention_mask"])
                if i + 1 >= calibration_batches:
                    break

        base_model.bert = convert_fx(prepared_bert)
        print("  INT8 static model ready.")
        return base_model, {
            "requested_mode": mode,
            "applied_mode": "static",
            "fallback_reason": None,
        }
    except Exception as e:
        fallback_reason = f"{type(e).__name__}: {e}"
        print(
            "  [Warning] Static PTQ calibration failed "
            f"({fallback_reason}). Falling back to dynamic INT8."
        )
        fallback_model, fallback_meta = apply_int8_quantization(base_model, mode="dynamic")
        fallback_meta["requested_mode"] = "static"
        fallback_meta["fallback_reason"] = fallback_reason
        return fallback_model, fallback_meta


def apply_fp16_model(fp32_model):
    """Return a half-precision copy of the model (GPU required for inference)."""
    print("\n[Quantize] Casting model to FP16 …")
    fp16_model = type(fp32_model)()
    fp16_model.load_state_dict(fp32_model.state_dict())
    fp16_model = fp16_model.half()
    return fp16_model


# ─────────────────────────────────────────────
# 4. INFERENCE
# ─────────────────────────────────────────────

def run_inference(model, df, tokenizer, precision="fp32", collect_hidden=False):
    """
    Run model over df; return predictions and (optionally) stacked hidden states.
    precision: "fp32" | "fp16" | "int8"
    """
    model.eval()
    use_device = "cpu" if precision == "int8" else DEVICE

    # FP16 inference on GPU; INT8 must stay on CPU
    if precision == "fp16":
        if not torch.cuda.is_available():
            print("  [Warning] No GPU — falling back to CPU FP32 for FP16 evaluation.")
            precision = "fp32"
        else:
            model = model.to(use_device)
    elif precision == "fp32":
        model = model.to(use_device)

    ds = JigsawDataset(df, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_preds, all_probs = [], []
    hidden_states_accum  = [[] for _ in range(13)]   # 13 layers

    with torch.no_grad():
        for batch in tqdm(dl, desc=f"  Inference [{precision}]", leave=False):
            ids  = batch["input_ids"]
            mask = batch["attention_mask"]

            if precision == "fp16":
                ids  = ids.to(use_device)
                mask = mask.to(use_device)
                with torch.cuda.amp.autocast():
                    logits, hs = model(ids, mask)
            elif precision == "int8":
                logits, hs = model(ids, mask)
            else:
                ids  = ids.to(use_device)
                mask = mask.to(use_device)
                logits, hs = model(ids, mask)

            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)

            if collect_hidden:
                for layer_idx, layer_hs in enumerate(hs):
                    hidden_states_accum[layer_idx].append(
                        layer_hs[:, 0, :].float().cpu()
                    )

    hidden_out = None
    if collect_hidden:
        hidden_out = [torch.cat(h, dim=0) for h in hidden_states_accum]

    return np.array(all_preds), np.array(all_probs), hidden_out


# ─────────────────────────────────────────────
# 5. FAIRNESS METRICS
# ─────────────────────────────────────────────

def compute_fairness_metrics(val_df, preds, precision_label):
    """
    Compute DPD and EOD for every identity column that has sufficient coverage.
    Returns a dict of {metric_name: value}.
    """
    results = {}
    labels  = val_df["label"].values
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")
    results["accuracy"]  = acc
    results["macro_f1"]  = f1

    dpd_per_attr, eod_per_attr = {}, {}

    for col in IDENTITY_COLS:
        if col not in val_df.columns:
            continue
        mask = val_df[col].notna()
        if mask.sum() < 200:
            continue
        sub_preds  = preds[mask]
        sub_labels = labels[mask]
        sub_attr   = val_df.loc[mask, col].values.astype(int)

        if len(np.unique(sub_attr)) < 2:
            continue

        try:
            dpd = demographic_parity_difference(
                sub_labels, sub_preds, sensitive_features=sub_attr
            )
            eod = equal_opportunity_difference(
                sub_labels, sub_preds, sensitive_features=sub_attr
            )
            dpd_per_attr[col] = abs(dpd)
            eod_per_attr[col] = abs(eod)
        except Exception:
            pass

    results["dpd_per_attr"] = dpd_per_attr
    results["eod_per_attr"] = eod_per_attr

    if dpd_per_attr:
        results["mean_DPD"] = np.mean(list(dpd_per_attr.values()))
        results["mean_EOD"] = np.mean(list(eod_per_attr.values()))
    else:
        results["mean_DPD"] = results["mean_EOD"] = float("nan")

    print(f"\n  [{precision_label}] acc={acc:.4f}  macro-F1={f1:.4f}  "
          f"mean-DPD={results['mean_DPD']:.4f}  mean-EOD={results['mean_EOD']:.4f}")
    return results


# ─────────────────────────────────────────────
# 6. REPRESENTATION ANALYSIS
# ─────────────────────────────────────────────

def centered_kernel_alignment(X, Y):
    """
    Linear CKA via HSIC.
    X, Y: tensors of shape (N, D)
    Returns scalar CKA score in [0, 1].
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    K = X @ X.T
    L = Y @ Y.T
    hsic_kl = (K * L).sum()
    norm     = torch.sqrt((K * K).sum() * (L * L).sum())
    return (hsic_kl / norm).item() if norm > 0 else 0.0


def compute_representation_metrics(fp32_hidden, quant_hidden):
    """
    Layer-wise L2 distance, cosine similarity, and CKA between FP32 and quantized.
    hidden: list of 13 tensors, each shape (N, 768).
    Returns dict of lists, one value per layer.
    """
    l2_scores, cos_scores, cka_scores = [], [], []

    for layer_idx in range(len(fp32_hidden)):
        h_fp32  = fp32_hidden[layer_idx].float()
        h_quant = quant_hidden[layer_idx].float()

        # L2 distance (mean over samples)
        l2  = (h_fp32 - h_quant).norm(dim=1).mean().item()

        # Cosine similarity (mean over samples)
        cos = nn.functional.cosine_similarity(h_fp32, h_quant, dim=1).mean().item()

        # CKA
        n   = min(512, h_fp32.shape[0])
        cka = centered_kernel_alignment(h_fp32[:n], h_quant[:n])

        l2_scores.append(l2)
        cos_scores.append(cos)
        cka_scores.append(cka)

    return {"l2": l2_scores, "cosine": cos_scores, "cka": cka_scores}


# ─────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────

def plot_representation_drift(metrics_int8, metrics_fp16, save_path):
    layers = list(range(13))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Layer-wise Representation Drift from FP32 Baseline", fontsize=14)

    for ax, key, title, ylabel in zip(
        axes,
        ["l2",     "cosine",               "cka"],
        ["L2 Distance",  "Cosine Similarity",    "CKA Score"],
        ["Mean L2 Norm", "Mean Cosine Similarity","CKA (1=identical)"],
    ):
        ax.plot(layers, metrics_int8[key], marker="o", label="INT8", color="crimson")
        ax.plot(layers, metrics_fp16[key], marker="s", label="FP16", color="steelblue",
                linestyle="--")
        ax.set_title(title); ax.set_xlabel("Layer"); ax.set_ylabel(ylabel)
        ax.set_xticks(layers)
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved representation drift plot → {save_path}")


def plot_fairness_comparison(results_all, save_path):
    precisions = list(results_all.keys())
    dpd_vals   = [results_all[p].get("mean_DPD", 0) for p in precisions]
    eod_vals   = [results_all[p].get("mean_EOD", 0) for p in precisions]
    f1_vals    = [results_all[p].get("macro_f1", 0) for p in precisions]

    x    = np.arange(len(precisions))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, dpd_vals, w, label="Mean DPD ↓",    color="#e74c3c")
    ax.bar(x,     eod_vals, w, label="Mean EOD ↓",    color="#e67e22")
    ax.bar(x + w, f1_vals,  w, label="Macro F1  ↑",   color="#2ecc71")

    ax.set_xticks(x); ax.set_xticklabels(precisions)
    ax.set_title("Fairness & Performance by Quantization Precision")
    ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved fairness comparison plot → {save_path}")


def plot_per_attribute_heatmap(results_all, metric_key, title, save_path):
    precisions = list(results_all.keys())
    all_attrs  = sorted(set(
        attr for p in precisions
        for attr in results_all[p].get(metric_key, {}).keys()
    ))
    if not all_attrs:
        return

    matrix = np.full((len(precisions), len(all_attrs)), np.nan)
    for i, p in enumerate(precisions):
        attr_dict = results_all[p].get(metric_key, {})
        for j, attr in enumerate(all_attrs):
            if attr in attr_dict:
                matrix[i, j] = attr_dict[attr]

    fig, ax = plt.subplots(figsize=(max(10, len(all_attrs) * 0.9), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(all_attrs))); ax.set_xticklabels(all_attrs, rotation=45, ha="right")
    ax.set_yticks(range(len(precisions))); ax.set_yticklabels(precisions)
    plt.colorbar(im, ax=ax, label="Score")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved heatmap → {save_path}")


# ─────────────────────────────────────────────
# 8. SUMMARY REPORT
# ─────────────────────────────────────────────

def print_summary_report(results_all, repr_metrics):
    print("\n" + "═" * 60)
    print("  QUANTIZATION FAIRNESS ANALYSIS — SUMMARY REPORT")
    print("═" * 60)


def to_builtin(value):
    """Convert numpy/scalar containers to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_results_json(results_all, repr_metrics, quantization_meta, save_path):
    payload = {
        "results": to_builtin(results_all),
        "representation_metrics": to_builtin(repr_metrics),
        "quantization_meta": to_builtin(quantization_meta),
        "config": {
            "int8_mode": INT8_MODE,
            "calib_sample": CALIB_SAMPLE,
            "calib_batch_size": CALIB_BATCH_SIZE,
            "calib_batches": CALIB_BATCHES,
            "repr_sample": REPR_SAMPLE,
            "seed": SEED,
        },
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved results JSON -> {save_path}")

    header = f"{'Metric':<30} {'FP32':>10} {'FP16':>10} {'INT8':>10}"
    print(header); print("─" * 60)

    for metric in ["accuracy", "macro_f1", "mean_DPD", "mean_EOD"]:
        row = f"{metric:<30}"
        for p in ["fp32", "fp16", "int8"]:
            val = results_all.get(p, {}).get(metric, float("nan"))
            row += f" {val:>10.4f}"
        print(row)

    print("\n── Representation Drift (INT8 vs FP32) ──")
    if repr_metrics.get("int8"):
        m = repr_metrics["int8"]
        print(f"  Layer 1  CKA={m['cka'][1]:.4f}  L2={m['l2'][1]:.4f}  cos={m['cosine'][1]:.4f}")
        print(f"  Layer 6  CKA={m['cka'][6]:.4f}  L2={m['l2'][6]:.4f}  cos={m['cosine'][6]:.4f}")
        print(f"  Layer 12 CKA={m['cka'][12]:.4f}  L2={m['l2'][12]:.4f}  cos={m['cosine'][12]:.4f}")

    print("\n── Interpretation ──")
    fp32_dpd = results_all.get("fp32", {}).get("mean_DPD", 0)
    int8_dpd = results_all.get("int8", {}).get("mean_DPD", 0)
    if int8_dpd > fp32_dpd:
        delta = int8_dpd - fp32_dpd
        print(f"  ⚠  INT8 amplifies Demographic Parity Difference by {delta:.4f} ({delta/max(fp32_dpd,1e-6)*100:.1f}%)")
    else:
        print("  ✓  INT8 does not worsen Demographic Parity Difference.")
    print("═" * 60)


# ─────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BERT QUANTIZATION & FAIRNESS COMPARISON")
    print("=" * 60)
    
    # Check if trained model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n[Error] Trained model not found at {MODEL_PATH}")
        print("Please run 'model.py' first to train the FP32 baseline.\n")
        return

    # Load data
    train_df, val_df = load_jigsaw(TRAIN_CSV)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Fixed subset for representation analysis
    repr_df = val_df.sample(n=min(REPR_SAMPLE, len(val_df)), random_state=SEED).reset_index(drop=True)

    # Load trained FP32 model
    print(f"\n[Model] Loading trained model from {MODEL_PATH} …")
    fp32_model = FrozenBertClassifier(num_labels=2)
    fp32_model.load_state_dict(torch.load(MODEL_PATH))
    print("  Loaded FP32 model successfully.")

    # Evaluate FP32 baseline
    print("\n[Eval] Collecting FP32 predictions and hidden states …")
    fp32_preds, fp32_probs, _ = run_inference(fp32_model, val_df, tokenizer, precision="fp32")
    _, _, fp32_hidden = run_inference(
        fp32_model, repr_df, tokenizer, precision="fp32", collect_hidden=True
    )

    results_all  = {}
    repr_metrics = {}

    print("\n[Fairness] FP32 …")
    results_all["fp32"] = compute_fairness_metrics(val_df, fp32_preds, "FP32")

    # INT8 quantization
    if INT8_MODE == "static":
        calib_loader, used_calib_samples = build_calibration_loader(
            train_df,
            tokenizer,
            max_len=MAX_LEN,
            batch_size=CALIB_BATCH_SIZE,
            calib_samples=CALIB_SAMPLE,
        )
        print(f"\n[Quantize] Static INT8 calibration samples: {used_calib_samples:,}")
        int8_model, quantization_meta = apply_int8_quantization(
            fp32_model,
            mode="static",
            calib_loader=calib_loader,
            calibration_batches=CALIB_BATCHES,
        )
    else:
        int8_model, quantization_meta = apply_int8_quantization(fp32_model, mode="dynamic")

    print("\n[Eval] INT8 predictions and hidden states …")
    int8_preds, int8_probs, _ = run_inference(
        int8_model, val_df, tokenizer, precision="int8"
    )
    _, _, int8_hidden_repr = run_inference(
        int8_model, repr_df, tokenizer, precision="int8", collect_hidden=True
    )

    print("\n[Fairness] INT8 …")
    results_all["int8"] = compute_fairness_metrics(val_df, int8_preds, "INT8")

    print("\n[Representation] INT8 vs FP32 layer drift …")
    repr_metrics["int8"] = compute_representation_metrics(fp32_hidden, int8_hidden_repr)

    # FP16 quantization (GPU only)
    if torch.cuda.is_available():
        fp16_model = apply_fp16_model(fp32_model)
        print("\n[Eval] FP16 predictions and hidden states …")
        fp16_preds, fp16_probs, _ = run_inference(
            fp16_model, val_df, tokenizer, precision="fp16"
        )
        _, _, fp16_hidden_repr = run_inference(
            fp16_model, repr_df, tokenizer, precision="fp16", collect_hidden=True
        )
        print("\n[Fairness] FP16 …")
        results_all["fp16"] = compute_fairness_metrics(val_df, fp16_preds, "FP16")
        repr_metrics["fp16"] = compute_representation_metrics(fp32_hidden, fp16_hidden_repr)
    else:
        print("\n[FP16] Skipped — no GPU detected. FP16 results will be absent from plots.")
        results_all["fp16"] = results_all["fp32"].copy()
        repr_metrics["fp16"] = repr_metrics["int8"]

    # Generate plots
    print("\n[Plot] Generating visualizations …")
    plot_representation_drift(
        repr_metrics["int8"], repr_metrics["fp16"],
        save_path=os.path.join(OUTPUT_DIR, "representation_drift.png"),
    )
    plot_fairness_comparison(
        results_all,
        save_path=os.path.join(OUTPUT_DIR, "fairness_comparison.png"),
    )
    plot_per_attribute_heatmap(
        results_all, "dpd_per_attr",
        title="Demographic Parity Difference per Identity Attribute",
        save_path=os.path.join(OUTPUT_DIR, "dpd_heatmap.png"),
    )
    plot_per_attribute_heatmap(
        results_all, "eod_per_attr",
        title="Equal Opportunity Difference per Identity Attribute",
        save_path=os.path.join(OUTPUT_DIR, "eod_heatmap.png"),
    )
    save_results_json(
        results_all,
        repr_metrics,
        quantization_meta,
        save_path=os.path.join(OUTPUT_DIR, "quantization_results.json"),
    )

    # Phase 2/3 post-processing: group-aware thresholds and ROC distortion analysis.
    model_scores = {
        "fp32": fp32_probs,
        "int8": int8_probs,
    }
    if torch.cuda.is_available():
        model_scores["fp16"] = fp16_probs
    else:
        model_scores["fp16"] = fp32_probs

    phase23_report = run_phase23_pipeline(
        val_df=val_df,
        model_scores=model_scores,
        output_dir=OUTPUT_DIR,
        plot_prefix="jigsaw",
        base_metrics=results_all,
        quantization_meta=quantization_meta,
    )
    print("\n[Phase 2/3] Saved outputs:")
    print(f"  metrics_before_after → {phase23_report['artifacts']['metrics_path']}")
    print(f"  roc_gap             → {phase23_report['artifacts']['roc_gap_path']}")
    print(f"  thresholds           → {phase23_report['artifacts']['thresholds_path']}")
    print(f"  verification_report  → {phase23_report['verification_report_path']}")
    print(f"  phase23_dir          → {phase23_report['phase23_dir']}")

    # Print summary
    print_summary_report(results_all, repr_metrics)
    print(f"\n[Done] All outputs written to ./{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
