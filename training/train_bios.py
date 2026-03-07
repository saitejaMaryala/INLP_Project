"""
Bias in Bios — Frozen BERT Quantization Fairness Pipeline
==========================================================
Trains a frozen BERT encoder + classifier on 28 occupation classes,
then evaluates fairness (EOD) and representation drift (CKA) across
FP32 → FP16 → INT8 quantization levels.

Usage:
    Full run:     python -m training.train_bios --epochs 5
    Sanity check: python -m training.train_bios --sanity-check
"""

import argparse
import copy
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.spatial.distance import cosine as cosine_dist
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on the path so `utils.*` imports work when running
# as `python training/train_bios.py` from the repo root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loaders import load_bias_in_bios


# ============================================================================
# 1. MODEL
# ============================================================================

class FrozenBertClassifier(nn.Module):
    """Frozen BERT encoder (output_hidden_states=True) + trainable classifier."""

    def __init__(self, num_classes: int = 28, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :].float()  # always FP32 for classifier
        return self.classifier(self.dropout(cls_emb))

    def extract_hidden_states(self, input_ids, attention_mask):
        """Return CLS embeddings from every layer. Shape: (num_layers, batch, 768)."""
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.hidden_states: tuple of (batch, seq_len, 768) × 13 layers
        return torch.stack([h[:, 0, :] for h in outputs.hidden_states])


# ============================================================================
# 2. DATASET
# ============================================================================

class BiosDataset(Dataset):
    def __init__(self, texts, labels, genders, tokenizer, max_length=256,
                 batch_tok_size=10000):
        # Tokenize in chunks so we get a progress bar
        all_ids, all_masks = [], []
        for i in tqdm(range(0, len(texts), batch_tok_size),
                      desc="Tokenizing", leave=False):
            chunk = tokenizer(
                texts[i:i + batch_tok_size],
                padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            all_ids.append(chunk["input_ids"])
            all_masks.append(chunk["attention_mask"])
        self.input_ids = torch.cat(all_ids)
        self.attention_mask = torch.cat(all_masks)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.genders = torch.tensor(genders, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "genders": self.genders[idx],
        }


# ============================================================================
# 3. TRAINING
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="eval"):
    model.eval()
    total_loss, all_preds, all_labels, all_genders = 0.0, [], [], []

    for batch in tqdm(loader, desc=f"[{desc}]", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_genders.extend(batch["genders"].tolist())

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, acc, f1, all_preds, all_labels, all_genders


# ============================================================================
# 4. FAIRNESS — Equal Opportunity Difference (EOD)
# ============================================================================

def compute_eod(preds, labels, genders, num_classes=28):
    """
    EOD = TPR(female) − TPR(male) per class, then averaged.
    Gender mapping: 0=female, 1=male.
    """
    preds = np.array(preds)
    labels = np.array(labels)
    genders = np.array(genders)

    per_class_eod = []
    for c in range(num_classes):
        tprs = {}
        for g, g_name in [(0, "female"), (1, "male")]:
            mask = (labels == c) & (genders == g)
            if mask.sum() == 0:
                continue
            tprs[g_name] = (preds[mask] == c).sum() / mask.sum()
        if "female" in tprs and "male" in tprs:
            per_class_eod.append(tprs["female"] - tprs["male"])

    mean_eod = float(np.mean(per_class_eod)) if per_class_eod else 0.0
    max_eod = float(np.max(np.abs(per_class_eod))) if per_class_eod else 0.0
    return mean_eod, max_eod, per_class_eod


# ============================================================================
# 5. QUANTIZATION
# ============================================================================

def quantize_fp16(model):
    """Quantize BERT encoder to FP16, keep classifier FP32."""
    m = copy.deepcopy(model).cpu()
    m.bert = m.bert.half()
    # Keep classifier in FP32
    m.classifier = m.classifier.float()
    m.dropout = m.dropout.float()
    return m


def quantize_int8(model):
    """Dynamic INT8 quantization on BERT encoder only."""
    m = copy.deepcopy(model).cpu()
    m.bert = torch.quantization.quantize_dynamic(
        m.bert, {nn.Linear}, dtype=torch.qint8
    )
    return m


# ============================================================================
# 6. REPRESENTATION ANALYSIS — CKA, L2, Cosine
# ============================================================================

def linear_cka(X, Y):
    """Linear CKA between two (n, d) matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


@torch.no_grad()
def extract_hidden_states(model, loader, device, max_samples=1000):
    """
    Returns numpy array of shape (13, N, 768) — CLS embeddings per layer.
    Runs on CPU for INT8 compatibility; on `device` otherwise.
    """
    model.eval()
    all_hidden = []  # will become list of (13, batch, 768) tensors
    n = 0

    for batch in tqdm(loader, desc="[extract repr]", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        # (13, batch, 768)
        cls_per_layer = torch.stack([h[:, 0, :] for h in outputs.hidden_states])
        all_hidden.append(cls_per_layer.cpu().float().numpy())

        n += input_ids.size(0)
        if n >= max_samples:
            break

    arr = np.concatenate(all_hidden, axis=1)[:, :max_samples, :]
    return arr  # (13, N, 768)


def representation_analysis(hidden_fp32, hidden_quant, label):
    """Compare across 13 layers. Return dict of per-layer metrics."""
    num_layers = hidden_fp32.shape[0]
    results = {"label": label, "layers": []}

    for layer in range(num_layers):
        fp32_mat = hidden_fp32[layer]
        quant_mat = hidden_quant[layer]

        l2 = float(np.mean(np.linalg.norm(fp32_mat - quant_mat, axis=1)))
        cos_sims = [
            1 - cosine_dist(fp32_mat[i], quant_mat[i])
            for i in range(fp32_mat.shape[0])
            if np.linalg.norm(fp32_mat[i]) > 0 and np.linalg.norm(quant_mat[i]) > 0
        ]
        cos_mean = float(np.mean(cos_sims)) if cos_sims else 0.0
        cka = linear_cka(fp32_mat, quant_mat)

        results["layers"].append({
            "layer": layer,
            "l2_distance": round(l2, 6),
            "cosine_similarity": round(cos_mean, 6),
            "cka": round(cka, 6),
        })

    return results


# ============================================================================
# 7. FULL EVALUATION HELPER (accuracy + EOD for a given model variant)
# ============================================================================

def full_eval(model, loader, criterion, device, tag, num_classes):
    """Run evaluation + EOD. Returns summary dict."""
    loss, acc, f1, preds, labels, genders = evaluate(
        model, loader, criterion, device, desc=tag
    )
    mean_eod, max_eod, per_class = compute_eod(preds, labels, genders, num_classes)

    print(f"\n--- {tag} ---")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {f1:.4f}")
    print(f"  Mean EOD : {mean_eod:+.4f}")
    print(f"  Max |EOD|: {max_eod:.4f}")

    return {
        "tag": tag,
        "accuracy": round(acc, 5),
        "macro_f1": round(f1, 5),
        "mean_eod": round(mean_eod, 5),
        "max_abs_eod": round(max_eod, 5),
        "per_class_eod": [round(e, 5) for e in per_class],
    }


# ============================================================================
# 8. MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Bias-in-Bios frozen BERT pipeline")
    p.add_argument("--data-dir", default="data/bias_in_bios")
    p.add_argument("--save-dir", default="models/bios")
    p.add_argument("--results-dir", default="results/bios")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--cka-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanity-check", action="store_true",
                   help="Quick run: 500 train, 200 test, 2 epochs")
    p.add_argument("--skip-quantization", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Seed ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Directories ---
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs available: {n_gpu}")

    # ------------------------------------------------------------------
    # A. LOAD DATA
    # ------------------------------------------------------------------
    print("\n========== LOADING DATA ==========")
    (train_texts, train_labels, train_genders,
     test_texts, test_labels, test_genders,
     occupations) = load_bias_in_bios(args.data_dir)

    num_classes = len(occupations)
    print(f"Number of occupation classes: {num_classes}")

    if args.sanity_check:
        print("\n⚡ SANITY-CHECK MODE — trimming data")
        train_texts = train_texts[:500]
        train_labels = train_labels[:500]
        train_genders = train_genders[:500]
        test_texts = test_texts[:200]
        test_labels = test_labels[:200]
        test_genders = test_genders[:200]
        args.epochs = 2
        args.cka_samples = 100

    # --- Tokenizer & datasets ---
    print("\nTokenizing...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = BiosDataset(train_texts, train_labels, train_genders,
                                tokenizer, args.max_length)
    test_dataset = BiosDataset(test_texts, test_labels, test_genders,
                               tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------
    # B. BUILD MODEL
    # ------------------------------------------------------------------
    print("\n========== BUILDING MODEL ==========")
    model = FrozenBertClassifier(num_classes=num_classes)
    if n_gpu > 1:
        print(f"Wrapping model in DataParallel ({n_gpu} GPUs)")
        model = nn.DataParallel(model)
    model.to(device)

    # Class weights for imbalanced occupations
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Only optimize classifier params (unfrozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    # ------------------------------------------------------------------
    # C. TRAINING LOOP
    # ------------------------------------------------------------------
    print("\n========== TRAINING (FP32) ==========")
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, test_loader, criterion, device, desc=f"Epoch {epoch} val"
        )
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
              f"time={elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, "best_bios_fp32.pt")
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, save_path)
            print(f"  ✓ Best model saved ({val_acc:.4f}) → {save_path}")

    # ------------------------------------------------------------------
    # D. RELOAD BEST MODEL FOR EVALUATION
    # ------------------------------------------------------------------
    print("\n========== LOADING BEST CHECKPOINT ==========")
    base_model = FrozenBertClassifier(num_classes=num_classes)
    base_model.load_state_dict(torch.load(
        os.path.join(args.save_dir, "best_bios_fp32.pt"),
        map_location="cpu", weights_only=True
    ))

    # Keep a CPU copy for quantization later
    fp32_model = copy.deepcopy(base_model)

    # Wrap for multi-GPU eval
    if n_gpu > 1:
        base_model = nn.DataParallel(base_model)
    base_model.to(device)

    # ------------------------------------------------------------------
    # E. FP32 EVALUATION + EOD
    # ------------------------------------------------------------------
    print("\n========== FP32 EVALUATION ==========")
    fp32_results = full_eval(base_model, test_loader, criterion, device,
                             "FP32", num_classes)

    all_results = {"fp32": fp32_results}

    if args.skip_quantization:
        print("\n⏩ Skipping quantization (--skip-quantization flag)")
    else:
        # --------------------------------------------------------------
        # F. FP16 QUANTIZATION + EVAL
        # --------------------------------------------------------------
        print("\n========== FP16 QUANTIZATION ==========")
        fp16_model = quantize_fp16(fp32_model)

        # FP16 inference needs GPU (half not supported on CPU for BERT)
        if n_gpu > 1:
            fp16_model = nn.DataParallel(fp16_model)
        fp16_model.to(device)

        fp16_results = full_eval(fp16_model, test_loader, criterion, device,
                                 "FP16", num_classes)
        all_results["fp16"] = fp16_results

        # --------------------------------------------------------------
        # G. INT8 QUANTIZATION + EVAL
        # --------------------------------------------------------------
        print("\n========== INT8 QUANTIZATION ==========")
        int8_model = quantize_int8(fp32_model)

        # INT8 runs on CPU only
        cpu_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)
        cpu_criterion = nn.CrossEntropyLoss(weight=weight_tensor.cpu())
        int8_results = full_eval(int8_model, cpu_loader, cpu_criterion,
                                 torch.device("cpu"), "INT8", num_classes)
        all_results["int8"] = int8_results

        # --------------------------------------------------------------
        # H. REPRESENTATION ANALYSIS (CKA / L2 / Cosine)
        # --------------------------------------------------------------
        print("\n========== REPRESENTATION ANALYSIS ==========")
        cka_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)

        # FP32 hidden states (on GPU then move to CPU)
        eval_model_fp32 = copy.deepcopy(fp32_model).to(device)
        hidden_fp32 = extract_hidden_states(eval_model_fp32, cka_loader,
                                            device, args.cka_samples)
        np.save(os.path.join(args.results_dir, "bios_hidden_fp32.npy"),
                hidden_fp32)
        print(f"FP32 hidden states: {hidden_fp32.shape}")
        del eval_model_fp32

        # FP16 hidden states
        eval_model_fp16 = quantize_fp16(fp32_model).to(device)
        hidden_fp16 = extract_hidden_states(eval_model_fp16, cka_loader,
                                            device, args.cka_samples)
        np.save(os.path.join(args.results_dir, "bios_hidden_fp16.npy"),
                hidden_fp16)
        print(f"FP16 hidden states: {hidden_fp16.shape}")
        del eval_model_fp16

        # INT8 hidden states (CPU)
        eval_model_int8 = quantize_int8(fp32_model)
        hidden_int8 = extract_hidden_states(eval_model_int8, cpu_loader,
                                            torch.device("cpu"),
                                            args.cka_samples)
        np.save(os.path.join(args.results_dir, "bios_hidden_int8.npy"),
                hidden_int8)
        print(f"INT8 hidden states: {hidden_int8.shape}")
        del eval_model_int8

        # CKA analysis
        cka_fp16 = representation_analysis(hidden_fp32, hidden_fp16, "FP32_vs_FP16")
        cka_int8 = representation_analysis(hidden_fp32, hidden_int8, "FP32_vs_INT8")
        all_results["cka_fp16"] = cka_fp16
        all_results["cka_int8"] = cka_int8

        print("\nLayer-wise CKA summary:")
        print(f"{'Layer':<8} {'FP16 CKA':<12} {'INT8 CKA':<12} {'FP16 L2':<12} {'INT8 L2':<12}")
        print("-" * 56)
        for i in range(len(cka_fp16["layers"])):
            fp16_l = cka_fp16["layers"][i]
            int8_l = cka_int8["layers"][i]
            print(f"{i:<8} {fp16_l['cka']:<12.6f} {int8_l['cka']:<12.6f} "
                  f"{fp16_l['l2_distance']:<12.6f} {int8_l['l2_distance']:<12.6f}")

        # Fairness delta summary
        print("\n========== FAIRNESS DELTA SUMMARY ==========")
        print(f"{'Metric':<15} {'FP32':<10} {'FP16':<10} {'INT8':<10} {'Δ FP16':<10} {'Δ INT8':<10}")
        print("-" * 65)
        for metric in ["accuracy", "macro_f1", "mean_eod", "max_abs_eod"]:
            v32 = fp32_results[metric]
            v16 = fp16_results[metric]
            v8 = int8_results[metric]
            print(f"{metric:<15} {v32:<10.5f} {v16:<10.5f} {v8:<10.5f} "
                  f"{v16 - v32:<+10.5f} {v8 - v32:<+10.5f}")

    # ------------------------------------------------------------------
    # I. SAVE ALL RESULTS
    # ------------------------------------------------------------------
    results_path = os.path.join(args.results_dir, "bios_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")
    print("Done.")


if __name__ == "__main__":
    main()
