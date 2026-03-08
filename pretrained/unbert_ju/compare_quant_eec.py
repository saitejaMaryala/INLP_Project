"""
Quantization + EEC Counterfactual Fairness Evaluation
======================================================
Assumes a trained FP32 model (fp32_classifier_uni.pt) already exists.

PIPELINE:
  1. Load trained FP32 FrozenBertClassifier
  2. Quantize → INT8 (dynamic, CPU) and FP16 (GPU if available)
  3. Load EEC CSV as probe input ONLY (no training, no labels)
  4. Run all 3 models → P(toxic) score per sentence
  5. Build counterfactual pairs by (Template, Emotion word)
  6. Compute per-pair:
       CFR  — binary prediction flip rate
       MCPS — mean |P(toxic|A) − P(toxic|B)|  [PRIMARY metric]
  7. Separate results for Gender axis and Race axis
  8. Save plots + print summary

FILE STRUCTURE EXPECTED:
  fp32_classifier_uni.pt              ← your trained checkpoint
  data/Equity-Evaluation-Corpus.csv   ← EEC probe CSV
  outputs/                            ← created automatically

INSTALL:
  pip install torch transformers pandas numpy scikit-learn tqdm matplotlib
"""

# ════════════════════════════════════════════════════════════════
# 0.  IMPORTS & CONFIG
# ════════════════════════════════════════════════════════════════
import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
MODEL_PATH  = "models/fp32_classifier_uni.pt"   # ← your checkpoint
EEC_CSV     = "/ssd_scratch/sai.teja/INLP_Project/data/Equity-Evaluation-Corpus.csv"
OUTPUT_DIR  = "outputs_eec"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Inference config ─────────────────────────────────────────────
MAX_LEN    = 128
BATCH_SIZE = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# EEC race label strings exactly as they appear in the CSV
RACE_A = "African-American"
RACE_B = "European"  # Note: dataset has 'European' not 'European-American'


# ════════════════════════════════════════════════════════════════
# 1.  MODEL DEFINITION
#     Must match exactly what was used during training
# ════════════════════════════════════════════════════════════════
class FrozenBertClassifier(nn.Module):
    """
    Frozen BERT encoder + trainable linear head.
    output_hidden_states=True exposes all 12 layer tensors.
    """
    def __init__(self, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            output_hidden_states=True,
        )
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]          # [CLS] token
        logits  = self.classifier(self.dropout(cls_emb))
        return logits, outputs.hidden_states


# ════════════════════════════════════════════════════════════════
# 2.  LOAD TRAINED FP32 MODEL
# ════════════════════════════════════════════════════════════════
def load_fp32_model(path):
    print(f"\n[Model] Loading FP32 checkpoint from {path} …")
    model = FrozenBertClassifier(num_labels=2)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print("  FP32 model loaded successfully.")
    return model


# ════════════════════════════════════════════════════════════════
# 3.  QUANTIZATION
# ════════════════════════════════════════════════════════════════
def build_int8_model(fp32_model):
    """
    Dynamic INT8 quantization — targets every nn.Linear in the frozen
    BERT encoder.  PyTorch dynamic quant runs on CPU only.
    The classifier head stays FP32 (it is not inside bert.*).
    """
    print("\n[Quantize] Building INT8 model …")
    # quantize_dynamic works on the model in-place on a copy;
    # we pass fp32_model already on CPU (loaded with map_location='cpu')
    int8_model = torch.quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    int8_model.eval()
    print("  INT8 model ready.")
    return int8_model


def build_fp16_model(fp32_model):
    """
    FP16 — casts all parameters to half precision.
    Inference uses torch.cuda.amp.autocast() on GPU.
    Falls back to FP32 on CPU (AMP not supported there).
    """
    print("\n[Quantize] Building FP16 model …")
    if not torch.cuda.is_available():
        print("  [Warning] No GPU found — FP16 will run as FP32 on CPU.")
        fp16_model = FrozenBertClassifier(num_labels=2)
        fp16_model.load_state_dict(fp32_model.state_dict())
        fp16_model.eval()
        return fp16_model, False          # flag: is_true_fp16

    fp16_model = FrozenBertClassifier(num_labels=2)
    fp16_model.load_state_dict(fp32_model.state_dict())
    fp16_model = fp16_model.half().to(DEVICE)
    fp16_model.eval()
    print("  FP16 model ready (on GPU).")
    return fp16_model, True


# ════════════════════════════════════════════════════════════════
# 4.  EEC DATASET  (sentences as input only — no labels)
# ════════════════════════════════════════════════════════════════
class SentenceDataset(Dataset):
    """Tokenises a plain list of strings for BERT inference."""
    def __init__(self, sentences, tokenizer, max_len=MAX_LEN):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.sentences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def load_eec(path):
    print(f"\n[EEC] Loading {path} …")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}   Columns: {df.columns.tolist()}")

    required = {"Sentence", "Template", "Person", "Gender", "Race", "Emotion word"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"EEC CSV is missing columns: {missing}")

    print(f"  Unique Gender values : {df['Gender'].unique().tolist()}")
    print(f"  Unique Race values   : {df['Race'].unique().tolist()}")
    return df


# ════════════════════════════════════════════════════════════════
# 5.  INFERENCE  →  P(toxic) per sentence
# ════════════════════════════════════════════════════════════════
def get_toxic_probs(model, sentences, tokenizer, precision, is_true_fp16=False):
    """
    Run model on `sentences` and return numpy array of P(toxic) in [0,1].

    precision      : "fp32" | "fp16" | "int8"
    is_true_fp16   : True if the model is genuinely on GPU in half precision
    """
    use_device = "cpu" if precision == "int8" else DEVICE

    ds = SentenceDataset(sentences, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dl, desc=f"    [{precision}] inference", leave=False):
            ids  = batch["input_ids"]
            mask = batch["attention_mask"]

            if precision == "int8":
                # INT8 model stays on CPU
                logits, _ = model(ids, mask)

            elif precision == "fp16" and is_true_fp16:
                ids  = ids.to(use_device)
                mask = mask.to(use_device)
                with torch.cuda.amp.autocast():
                    logits, _ = model(ids, mask)

            else:  # fp32 or fp16-fallback
                ids  = ids.to(use_device)
                mask = mask.to(use_device)
                logits, _ = model(ids, mask)

            # Softmax → take P(class=1) = P(toxic)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


def score_all_models(eec_df, fp32_model, int8_model, fp16_model,
                     fp16_is_true, tokenizer):
    """
    Run all three models over every sentence in eec_df.
    Adds columns  prob_fp32 / prob_fp16 / prob_int8  to eec_df (in-place copy).
    """
    sentences = eec_df["Sentence"].tolist()
    print("\n[EEC] Scoring sentences with FP32 …")
    fp32_model.to(DEVICE)
    probs_fp32 = get_toxic_probs(fp32_model, sentences, tokenizer,
                                 "fp32", False)

    print("[EEC] Scoring sentences with INT8 …")
    fp32_model.cpu()   # free GPU memory before INT8 (cpu-only)
    probs_int8 = get_toxic_probs(int8_model, sentences, tokenizer,
                                 "int8", False)

    print("[EEC] Scoring sentences with FP16 …")
    probs_fp16 = get_toxic_probs(fp16_model, sentences, tokenizer,
                                 "fp16", fp16_is_true)

    scored = eec_df.copy()
    scored["prob_fp32"] = probs_fp32
    scored["prob_fp16"] = probs_fp16
    scored["prob_int8"] = probs_int8
    return scored


# ════════════════════════════════════════════════════════════════
# 6.  PAIR BUILDING
#     Pairing key = (Template, Emotion word)  — holds context constant
#     Within each group, swap Gender OR Race
# ════════════════════════════════════════════════════════════════
def build_gender_pairs(scored_df):
    """
    For every (Template, Emotion word) group, pair each male-name row
    with a female-name row positionally.
    Returns DataFrame with columns:
      sentence_A/B, person_A/B, prob_fp32/fp16/int8 _A/B
    """
    pairs = []
    for (tmpl, emo), grp in scored_df.groupby(["Template", "Emotion word"]):
        males   = grp[grp["Gender"] == "male"].reset_index(drop=True)
        females = grp[grp["Gender"] == "female"].reset_index(drop=True)
        if males.empty or females.empty:
            continue
        n = min(len(males), len(females))
        for i in range(n):
            pairs.append({
                "template":      tmpl,
                "emotion_word":  emo,
                "sentence_A":    males.iloc[i]["Sentence"],
                "sentence_B":    females.iloc[i]["Sentence"],
                "person_A":      males.iloc[i]["Person"],
                "person_B":      females.iloc[i]["Person"],
                "prob_fp32_A":   males.iloc[i]["prob_fp32"],
                "prob_fp32_B":   females.iloc[i]["prob_fp32"],
                "prob_fp16_A":   males.iloc[i]["prob_fp16"],
                "prob_fp16_B":   females.iloc[i]["prob_fp16"],
                "prob_int8_A":   males.iloc[i]["prob_int8"],
                "prob_int8_B":   females.iloc[i]["prob_int8"],
            })
    df = pd.DataFrame(pairs)
    print(f"  Gender pairs: {len(df):,}")
    return df


def build_race_pairs(scored_df):
    """
    Pair African-American-name rows with European-American-name rows
    within each (Template, Emotion word) group.
    """
    pairs = []
    for (tmpl, emo), grp in scored_df.groupby(["Template", "Emotion word"]):
        aa = grp[grp["Race"] == RACE_A].reset_index(drop=True)
        ea = grp[grp["Race"] == RACE_B].reset_index(drop=True)
        if aa.empty or ea.empty:
            continue
        n = min(len(aa), len(ea))
        for i in range(n):
            pairs.append({
                "template":      tmpl,
                "emotion_word":  emo,
                "sentence_A":    aa.iloc[i]["Sentence"],
                "sentence_B":    ea.iloc[i]["Sentence"],
                "person_A":      aa.iloc[i]["Person"],
                "person_B":      ea.iloc[i]["Person"],
                "prob_fp32_A":   aa.iloc[i]["prob_fp32"],
                "prob_fp32_B":   ea.iloc[i]["prob_fp32"],
                "prob_fp16_A":   aa.iloc[i]["prob_fp16"],
                "prob_fp16_B":   ea.iloc[i]["prob_fp16"],
                "prob_int8_A":   aa.iloc[i]["prob_int8"],
                "prob_int8_B":   ea.iloc[i]["prob_int8"],
            })
    df = pd.DataFrame(pairs)
    print(f"  Race pairs:   {len(df):,}")
    return df


# ════════════════════════════════════════════════════════════════
# 7.  CFR + MCPS COMPUTATION
# ════════════════════════════════════════════════════════════════
def compute_cfr_mcps(probs_A, probs_B, threshold=0.5):
    """
    CFR  = fraction of pairs where binary prediction FLIPS.
           Will be near 0% for a toxicity model on neutral EEC sentences —
           expected and correct.  Report it but do not treat it as primary.

    MCPS = mean |P(toxic|A) − P(toxic|B)|
           PRIMARY metric.  Captures demographic sensitivity in the
           continuous probability space even when no flip occurs.
           A higher MCPS under INT8 vs FP32 = quantization noise is
           amplifying sensitivity to demographic tokens.

    Returns dict with both scalars + raw delta array for plotting.
    """
    preds_A = (probs_A >= threshold).astype(int)
    preds_B = (probs_B >= threshold).astype(int)

    flips      = (preds_A != preds_B)
    cfr        = flips.mean()
    delta_prob = np.abs(probs_A - probs_B)
    mcps       = delta_prob.mean()

    return {
        "CFR":         cfr,
        "MCPS":        mcps,
        "n_pairs":     len(probs_A),
        "n_flips":     int(flips.sum()),
        "delta_probs": delta_prob,
        "probs_A":     probs_A,
        "probs_B":     probs_B,
    }


def evaluate_pairs(pairs_df, swap_axis_label):
    """
    Given a pairs DataFrame (output of build_gender_pairs or build_race_pairs),
    compute CFR and MCPS for each precision level.
    Returns dict: { "fp32": {...}, "fp16": {...}, "int8": {...} }
    Returns None if no pairs available.
    """
    print(f"\n[Metrics] Computing CFR & MCPS — {swap_axis_label} axis …")
    
    if len(pairs_df) == 0:
        print(f"  [Warning] No {swap_axis_label} pairs available — skipping.")
        return None
    
    results = {}
    for prec in ["fp32", "fp16", "int8"]:
        pA = pairs_df[f"prob_{prec}_A"].values
        pB = pairs_df[f"prob_{prec}_B"].values
        m  = compute_cfr_mcps(pA, pB)
        results[prec] = m
        print(f"  {prec.upper():5s} | pairs={m['n_pairs']:,}  "
              f"flips={m['n_flips']:,}  "
              f"CFR={m['CFR']*100:6.2f}%  "
              f"MCPS={m['MCPS']:.5f}")
    return results


# ════════════════════════════════════════════════════════════════
# 8.  PLOTS
# ════════════════════════════════════════════════════════════════
COLORS = {"fp32": "#2ecc71", "fp16": "#3498db", "int8": "#e74c3c"}


def plot_mcps_bar(gender_results, race_results, save_path):
    """MCPS bar chart — Gender and Race side by side per precision."""
    precisions  = ["fp32", "fp16", "int8"]
    gender_mcps = [gender_results[p]["MCPS"] for p in precisions]
    race_mcps   = [race_results[p]["MCPS"]   for p in precisions]

    x = np.arange(len(precisions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, gender_mcps, w, label="Gender MCPS",
                color=["#3498db"] * 3, alpha=0.85)
    b2 = ax.bar(x + w/2, race_mcps,   w, label="Race MCPS",
                color=["#e74c3c"] * 3, alpha=0.85)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0003,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels([p.upper() for p in precisions])
    ax.set_ylabel("Mean Counterfactual Probability Shift  (↓ = fairer)")
    ax.set_title("MCPS by Precision Level — EEC Counterfactual Fairness\n"
                 "(Primary metric: higher = more demographic sensitivity)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_cfr_bar(gender_results, race_results, save_path):
    """CFR bar chart (secondary — expected to be near zero)."""
    precisions = ["fp32", "fp16", "int8"]
    g_cfr = [gender_results[p]["CFR"] * 100 for p in precisions]
    r_cfr = [race_results[p]["CFR"]   * 100 for p in precisions]

    x = np.arange(len(precisions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, g_cfr, w, label="Gender CFR", color="#2ecc71", alpha=0.85)
    ax.bar(x + w/2, r_cfr, w, label="Race CFR",   color="#f39c12", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([p.upper() for p in precisions])
    ax.set_ylabel("Counterfactual Flip Rate %  (↓ = fairer)")
    ax.set_title("CFR by Precision Level — EEC Counterfactual Fairness\n"
                 "(Secondary metric — low absolute values expected on toxicity model;\n"
                 "relative INT8 vs FP32 increase is the key signal)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_delta_distributions(gender_results, race_results, save_path):
    """
    2-row × 3-col grid:
      Row 0 — Gender |ΔP(toxic)| histograms per precision
      Row 1 — Race   |ΔP(toxic)| histograms per precision
    Shows whether INT8 widens the distribution (= more bias amplification).
    """
    precisions = ["fp32", "fp16", "int8"]
    fig, axes  = plt.subplots(2, 3, figsize=(16, 8), sharey="row")
    fig.suptitle("Distribution of |ΔP(toxic)| across Counterfactual Pairs\n"
                 "Wider / right-shifted = more demographic sensitivity",
                 fontsize=13)

    for col_idx, prec in enumerate(precisions):
        # Gender row
        ax_g = axes[0][col_idx]
        d_g  = gender_results[prec]["delta_probs"]
        ax_g.hist(d_g, bins=40, color=COLORS[prec], edgecolor="white", alpha=0.85)
        ax_g.axvline(gender_results[prec]["MCPS"], color="black",
                     linestyle="--", linewidth=1.5,
                     label=f"MCPS={gender_results[prec]['MCPS']:.4f}")
        ax_g.set_title(f"Gender — {prec.upper()}\n"
                       f"CFR={gender_results[prec]['CFR']*100:.2f}%")
        ax_g.set_xlabel("|ΔP(toxic)|")
        ax_g.legend(fontsize=8); ax_g.grid(alpha=0.25)

        # Race row
        ax_r = axes[1][col_idx]
        d_r  = race_results[prec]["delta_probs"]
        ax_r.hist(d_r, bins=40, color=COLORS[prec], edgecolor="white", alpha=0.85)
        ax_r.axvline(race_results[prec]["MCPS"], color="black",
                     linestyle="--", linewidth=1.5,
                     label=f"MCPS={race_results[prec]['MCPS']:.4f}")
        ax_r.set_title(f"Race — {prec.upper()}\n"
                       f"CFR={race_results[prec]['CFR']*100:.2f}%")
        ax_r.set_xlabel("|ΔP(toxic)|")
        ax_r.legend(fontsize=8); ax_r.grid(alpha=0.25)

    axes[0][0].set_ylabel("Count (Gender pairs)")
    axes[1][0].set_ylabel("Count (Race pairs)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_prob_scatter(gender_results, race_results, save_path):
    """
    2-row × 3-col scatter: P(toxic|A) vs P(toxic|B) per precision.
    Points on the diagonal = perfectly counterfactually fair.
    Spread off diagonal = demographic sensitivity.
    """
    precisions = ["fp32", "fp16", "int8"]
    fig, axes  = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("P(toxic|sentence_A) vs P(toxic|sentence_B)\n"
                 "Diagonal = perfect counterfactual fairness; "
                 "spread = demographic sensitivity",
                 fontsize=13)

    for col_idx, prec in enumerate(precisions):
        for row_idx, (results, label) in enumerate(
            [(gender_results, "Gender"), (race_results, "Race")]
        ):
            ax = axes[row_idx][col_idx]
            ax.scatter(results[prec]["probs_A"],
                       results[prec]["probs_B"],
                       alpha=0.15, s=5, color=COLORS[prec])
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect fairness")
            ax.set_title(f"{label} — {prec.upper()}\n"
                         f"MCPS={results[prec]['MCPS']:.4f}  "
                         f"CFR={results[prec]['CFR']*100:.2f}%")
            ax.set_xlabel("P(toxic | group A name)")
            ax.set_ylabel("P(toxic | group B name)")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend(fontsize=7); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ════════════════════════════════════════════════════════════════
# 9.  SUMMARY REPORT
# ════════════════════════════════════════════════════════════════
def print_summary(gender_results, race_results):
    precisions = ["fp32", "fp16", "int8"]

    print()
    print("═" * 70)
    print("  EEC COUNTERFACTUAL FAIRNESS — FINAL SUMMARY")
    print("═" * 70)
    print(f"  {'Metric':<35} {'FP32':>10} {'FP16':>10} {'INT8':>10}")
    print("─" * 70)

    for label, res in [("Gender", gender_results), ("Race", race_results)]:
        for metric, fmt in [("CFR",  lambda v: f"{v*100:9.2f}%"),
                            ("MCPS", lambda v: f"{v:10.5f}")]:
            row = f"  {label} {metric:<32}"
            for p in precisions:
                row += f" {fmt(res[p][metric])}"
            print(row)
        print("─" * 70)

    print()
    print("  Quantization amplification (INT8 vs FP32 baseline):")
    print()
    for label, res in [("Gender", gender_results), ("Race", race_results)]:
        fp32_mcps = res["fp32"]["MCPS"]
        int8_mcps = res["int8"]["MCPS"]
        fp32_cfr  = res["fp32"]["CFR"]
        int8_cfr  = res["int8"]["CFR"]
        delta_m   = int8_mcps - fp32_mcps
        delta_c   = (int8_cfr - fp32_cfr) * 100
        sym_m = "⚠ " if delta_m > 0 else "✓ "
        sym_c = "⚠ " if delta_c > 0 else "✓ "
        print(f"  {sym_m}{label} MCPS change : {delta_m:+.5f}")
        print(f"  {sym_c}{label} CFR  change : {delta_c:+.2f}%")
        print()

    print("  ── Interpretation note ──────────────────────────────────────")
    print("  MCPS is the primary metric here. EEC sentences are neutral")
    print("  emotional statements — a toxicity classifier will rarely cross")
    print("  the 0.5 decision boundary, so CFR ≈ 0% is expected and correct.")
    print("  A MCPS that is HIGHER under INT8 than FP32 is evidence that")
    print("  quantization noise amplifies demographic token sensitivity in")
    print("  the continuous probability space even without a prediction flip.")
    print("═" * 70)


# ════════════════════════════════════════════════════════════════
# 10.  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  QUANTIZATION + EEC COUNTERFACTUAL FAIRNESS PIPELINE")
    print("=" * 70)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # ── Step 1: Load trained FP32 model ──────────────────────────
    fp32_model = load_fp32_model(MODEL_PATH)

    # ── Step 2: Quantize ─────────────────────────────────────────
    int8_model             = build_int8_model(fp32_model)
    fp16_model, fp16_true  = build_fp16_model(fp32_model)

    # ── Step 3: Load EEC (sentences only — no labels used) ───────
    eec_df = load_eec(EEC_CSV)

    # ── Step 4: Run all 3 models → P(toxic) per sentence ─────────
    scored_df = score_all_models(
        eec_df, fp32_model, int8_model, fp16_model, fp16_true, tokenizer
    )

    # ── Step 5: Build counterfactual pairs ────────────────────────
    print("\n[Pairs] Building counterfactual pairs …")
    gender_pairs = build_gender_pairs(scored_df)
    race_pairs   = build_race_pairs(scored_df)

    # ── Step 6: Compute CFR and MCPS per precision ────────────────
    gender_results = evaluate_pairs(gender_pairs, "Gender")
    race_results   = evaluate_pairs(race_pairs,   "Race")

    # ── Step 7: Save pair DataFrames for inspection ───────────────
    if len(gender_pairs) > 0:
        gender_pairs.to_csv(os.path.join(OUTPUT_DIR, "gender_pairs_scored.csv"), index=False)
    if len(race_pairs) > 0:
        race_pairs.to_csv(  os.path.join(OUTPUT_DIR, "race_pairs_scored.csv"),   index=False)
    print(f"\n  Scored pair CSVs saved to {OUTPUT_DIR}/")

    # ── Step 8: Plots (only if we have results) ───────────────────
    if gender_results and race_results:
        print("\n[Plots] Generating visualisations …")
        plot_mcps_bar(gender_results, race_results,
                      os.path.join(OUTPUT_DIR, "eec_mcps_bar.png"))
        plot_cfr_bar(gender_results, race_results,
                     os.path.join(OUTPUT_DIR, "eec_cfr_bar.png"))
        plot_delta_distributions(gender_results, race_results,
                                 os.path.join(OUTPUT_DIR, "eec_delta_distributions.png"))
        plot_prob_scatter(gender_results, race_results,
                          os.path.join(OUTPUT_DIR, "eec_prob_scatter.png"))
    elif gender_results:
        print("\n[Plots] Only Gender results available — generating Gender-only plots …")
        # You could add gender-only plots here if needed
        print("  (Skipping plots that require both Gender and Race data)")
    else:
        print("\n[Plots] Insufficient data for plotting.")

    # ── Step 9: Print summary ─────────────────────────────────────
    if gender_results and race_results:
        print_summary(gender_results, race_results)
    elif gender_results:
        print("\n[Summary] Only Gender results available.")
        print(f"  Gender MCPS (FP32/FP16/INT8): {gender_results['fp32']['MCPS']:.5f} / {gender_results['fp16']['MCPS']:.5f} / {gender_results['int8']['MCPS']:.5f}")
        print(f"  Gender CFR  (FP32/FP16/INT8): {gender_results['fp32']['CFR']*100:.2f}% / {gender_results['fp16']['CFR']*100:.2f}% / {gender_results['int8']['CFR']*100:.2f}%")
    else:
        print("\n[Summary] No pairs available for evaluation.")
    print(f"\n[Done] All outputs written to ./{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()