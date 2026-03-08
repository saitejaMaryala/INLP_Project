"""
Bias in Bios — Training Script (Frozen BERT)
===========================================
Trains FrozenBertClassifier on Bias-in-Bios dataset and saves checkpoint.

Features:
  - Pre-tokenization caching (saves 8-15 min on subsequent runs)
  - AMP training with GradScaler
  - Class-weighted loss for occupation imbalance
  - DataParallel support for multi-GPU
  - Saves best model checkpoint by validation accuracy

Usage:
    python -m training.train_bios                   # full training
    python -m training.train_bios --sanity-check    # quick test (500 samples, 2 epochs)
    
After training, run comparison script:
    python -m training.compare_quant_bios           # quantization comparison
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── path setup ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loaders import load_bias_in_bios


# ════════════════════════════════════════════════════════════════
# 1.  MODEL
# ════════════════════════════════════════════════════════════════
class FrozenBertClassifier(nn.Module):
    """
    Frozen BERT encoder + trainable linear head.

    FIX 5: BERT forward runs inside torch.no_grad() — even though
    requires_grad=False on all BERT params, PyTorch still allocates
    intermediate activation tensors unless no_grad is explicit.
    This saves ~30% GPU memory and ~15% forward-pass time.
    """

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
        # FIX 5: no_grad on the frozen encoder
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        # Cast to float32 so AMP doesn't push the linear head into fp16
        cls_emb = outputs.last_hidden_state[:, 0, :].float()
        return self.classifier(self.dropout(cls_emb))


# ════════════════════════════════════════════════════════════════
# 2.  DATASET  — pre-tokenised & cached
# ════════════════════════════════════════════════════════════════
def tokenise_and_cache(texts, tokenizer, max_length, cache_path):
    """
    FIX 1: Tokenise once, save to disk.  Subsequent runs load in <1 s.
    Cache key is (cache_path).  If the file exists it is loaded directly.
    """
    if os.path.exists(cache_path):
        print(f"  [cache] Loading tokenised data from {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        return data["input_ids"], data["attention_mask"]

    print(f"  [cache] Tokenising {len(texts):,} texts → {cache_path}")
    # Process in chunks to show progress
    CHUNK = 8_000
    all_ids, all_masks = [], []
    for i in tqdm(range(0, len(texts), CHUNK), desc="  Tokenising", leave=False):
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
    """
    Receives pre-tokenised tensors — no tokenisation overhead at training time.
    """
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
# 3.  TRAINING
# ════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, epoch):
    """
    FIX 4: AMP training with GradScaler.
    Only the classifier head (768 × num_classes) accumulates gradients —
    AMP gives a free ~2× speedup on A6000 tensor cores for even this
    small matmul.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        ids   = batch["input_ids"].to(device, non_blocking=True)
        mask  = batch["attention_mask"].to(device, non_blocking=True)
        lbls  = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        with autocast():                         # FP16 forward pass
            logits = model(ids, mask)
            loss = criterion(logits, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * lbls.size(0)
        correct    += (logits.argmax(dim=1) == lbls).sum().item()
        total      += lbls.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="eval"):
    model.eval()
    total_loss  = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc=f"[{desc}]", leave=False):
        ids   = batch["input_ids"].to(device, non_blocking=True)
        mask  = batch["attention_mask"].to(device, non_blocking=True)
        lbls  = batch["labels"].to(device, non_blocking=True)

        with autocast():
            logits = model(ids, mask)
            loss = criterion(logits, lbls)

        total_loss += loss.item() * lbls.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(lbls.cpu().tolist())

    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, acc, f1


# ════════════════════════════════════════════════════════════════
# 4.  UNWRAP HELPER
# ════════════════════════════════════════════════════════════════
def _unwrap(model):
    """Unwrap DataParallel if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ════════════════════════════════════════════════════════════════
# 5.  ARGS
# ════════════════════════════════════════════════════════════════
def parse_args():
    """
    p.add_argument("--data-dir",     default="/ssd_scratch/sai.teja/INLP_Project/data/bias_in_bios")
    p.add_argument("--save-dir",     default="/ssd_scratch/sai.teja/INLP_Project/models/bios",
                   help="Where to save trained checkpoint")
    p.add_argument("--cache-dir",    default="/ssd_scratch/sai.teja/INLP_Project/cache/bios",
                   help="Where to store pre-tokenised .pt files")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch-size",   type=int,   default=64)  # Reduced from 128 to prevent OOM
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--max-length",   type=int,   default=128)
    p.add_argument("--num-workers",  type=int,   default=4)  # Reduced from 8
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--sanity-check", action="store_true",
                   help="500 train / 200 test / 2 epochs — quick smoke test")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
# 6.  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir,    exist_ok=True)
    os.makedirs(args.cache_dir,   exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs: {n_gpu}")

    # ── A. LOAD DATA ──────────────────────────────────────────────
    print("\n══════ LOADING DATA ══════")
    (train_texts, train_labels, train_genders,
     test_texts,  test_labels,  test_genders,
     occupations) = load_bias_in_bios(args.data_dir)

    num_classes = len(occupations)
    print(f"Occupations: {num_classes}  |  "
          f"Train: {len(train_texts):,}  |  Test: {len(test_texts):,}")

    if args.sanity_check:
        print("\n⚡ SANITY-CHECK MODE")
        train_texts   = train_texts[:500];   train_labels  = train_labels[:500]
        train_genders = train_genders[:500]; test_texts    = test_texts[:200]
        test_labels   = test_labels[:200];   test_genders  = test_genders[:200]
        args.epochs = 2

    # ── B. TOKENISE & CACHE ───────────────────────────────────────
    print("\n══════ TOKENISING ══════")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_ids, train_mask = tokenise_and_cache(
        train_texts, tokenizer, args.max_length,
        os.path.join(args.cache_dir, f"train_ml{args.max_length}.pt"),
    )
    test_ids, test_mask = tokenise_and_cache(
        test_texts, tokenizer, args.max_length,
        os.path.join(args.cache_dir, f"test_ml{args.max_length}.pt"),
    )

    train_ds = BiosDataset(train_ids, train_mask, train_labels, train_genders)
    test_ds  = BiosDataset(test_ids,  test_mask,  test_labels,  test_genders)

    loader_kwargs = dict(
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        pin_memory   = True,
        persistent_workers = args.num_workers > 0,
        prefetch_factor    = 2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── C. BUILD MODEL ────────────────────────────────────────────
    print("\n══════ BUILDING MODEL ══════")
    model = FrozenBertClassifier(num_classes=num_classes)

   # Class-weighted loss for occupational imbalance
    class_counts  = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion     = nn.CrossEntropyLoss(weight=weight_tensor)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler    = GradScaler()

    print(f"Trainable params: {sum(p.numel() for p in trainable):,}  "
          f"(classifier head only)")

    if n_gpu > 1:
        print(f"Wrapping in DataParallel ({n_gpu} GPUs)")
        model = nn.DataParallel(model)
    model.to(device)

    # ── D. TRAIN ──────────────────────────────────────────────────
    ckpt_path = os.path.join(args.save_dir, "best_bios_fp32.pt")

    print("\n══════ TRAINING (FP32 + AMP) ══════")
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, test_loader, criterion, device,
            desc=f"Epoch {epoch} val"
        )
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
              f"time={elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1  = val_f1
            raw_state = (_unwrap(model)).state_dict()
            torch.save(raw_state, ckpt_path)
            print(f"  ✓ Best saved (val_f1={val_f1:.4f}) → {ckpt_path}")

    print(f"\n[Training Complete] Best F1: {best_f1:.4f}")
    print(f"Model checkpoint saved → {ckpt_path}")
    print(f"\nTo run quantization comparison:")
    print(f"    python -m training.compare_quant_bios\n")


if __name__ == "__main__":
    main()
    p.add_argument("--data-dir",     default="/ssd_scratch/sai.teja/INLP_Project/data/bias_in_bios")
    p.add_argument("--save-dir",     default="/ssd_scratch/sai.teja/INLP_Project/models/bios",
                   help="Where to save checkpoints and plots")
    p.add_argument("--results-dir",  default="/ssd_scratch/sai.teja/INLP_Project/results/bios")
    p.add_argument("--cache-dir",    default="/ssd_scratch/sai.teja/INLP_Project/cache/bios",
                   help="Where to store pre-tokenised .pt files")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--max-length",   type=int,   default=128,   # FIX 2
                   help="128 covers >95%% of bios; 256 gives 4× more compute")
    p.add_argument("--num-workers",  type=int,   default=8,     # FIX 3
                   help="DataLoader workers; set to 4× num_GPUs")
    p.add_argument("--cka-samples",  type=int,   default=1000)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--sanity-check", action="store_true",
                   help="500 train / 200 test / 2 epochs — quick smoke test")
    p.add_argument("--skip-train",   action="store_true",
                   help="Load existing checkpoint, skip training")
    p.add_argument("--skip-quant",   action="store_true",
                   help="Skip quantization and representation analysis")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
# 11.  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.cache_dir,   exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs: {n_gpu}")

    # ── A. LOAD DATA ──────────────────────────────────────────────
    print("\n══════ LOADING DATA ══════")
    (train_texts, train_labels, train_genders,
     test_texts,  test_labels,  test_genders,
     occupations) = load_bias_in_bios(args.data_dir)

    num_classes = len(occupations)
    print(f"Occupations: {num_classes}  |  "
          f"Train: {len(train_texts):,}  |  Test: {len(test_texts):,}")

    if args.sanity_check:
        print("\n⚡ SANITY-CHECK MODE")
        train_texts   = train_texts[:500];   train_labels  = train_labels[:500]
        train_genders = train_genders[:500]; test_texts    = test_texts[:200]
        test_labels   = test_labels[:200];   test_genders  = test_genders[:200]
        args.epochs = 2; args.cka_samples = 100

    # ── B. TOKENISE & CACHE  (FIX 1) ─────────────────────────────
    print("\n══════ TOKENISING ══════")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_ids, train_mask = tokenise_and_cache(
        train_texts, tokenizer, args.max_length,
        os.path.join(args.cache_dir, f"train_ml{args.max_length}.pt"),
    )
    test_ids, test_mask = tokenise_and_cache(
        test_texts, tokenizer, args.max_length,
        os.path.join(args.cache_dir, f"test_ml{args.max_length}.pt"),
    )

    train_ds = BiosDataset(train_ids, train_mask, train_labels, train_genders)
    test_ds  = BiosDataset(test_ids,  test_mask,  test_labels,  test_genders)

    # FIX 3: num_workers=8, persistent_workers + prefetch for GPU feeding
    loader_kwargs = dict(
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        pin_memory   = True,
        persistent_workers = args.num_workers > 0,
        prefetch_factor    = 2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── C. BUILD MODEL ────────────────────────────────────────────
    print("\n══════ BUILDING MODEL ══════")
    model = FrozenBertClassifier(num_classes=num_classes)

    # Class-weighted loss for occupational imbalance
    class_counts  = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion     = nn.CrossEntropyLoss(weight=weight_tensor)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler    = GradScaler()   # FIX 4: AMP gradient scaler

    print(f"Trainable params: {sum(p.numel() for p in trainable):,}  "
          f"(classifier head only)")

    if n_gpu > 1:
        print(f"Wrapping in DataParallel ({n_gpu} GPUs)")
        model = nn.DataParallel(model)
    model.to(device)

    # ── D. TRAIN ──────────────────────────────────────────────────
    ckpt_path = os.path.join(args.save_dir, "best_bios_fp32.pt")

    if args.skip_train and os.path.exists(ckpt_path):
        print(f"\n══════ SKIP TRAINING — loading {ckpt_path} ══════")
    else:
        print("\n══════ TRAINING (FP32 + AMP) ══════")
        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device, epoch
            )
            _, val_acc, val_f1, _, _, _, _ = evaluate(
                model, test_loader, criterion, device,
                desc=f"Epoch {epoch} val"
            )
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{args.epochs}  "
                  f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
                  f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
                  f"time={elapsed:.1f}s")

            if val_acc > best_acc:
                best_acc  = val_acc
                raw_state = (_unwrap(model)).state_dict()
                torch.save(raw_state, ckpt_path)
                print(f"  ✓ Best saved (val_acc={val_acc:.4f}) → {ckpt_path}")

    # ── E. RELOAD BEST CHECKPOINT ─────────────────────────────────
    print("\n══════ LOADING BEST CHECKPOINT ══════")
    fp32_model = FrozenBertClassifier(num_classes=num_classes)
    fp32_model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )

    eval_model = copy.deepcopy(fp32_model)
    if n_gpu > 1:
        eval_model = nn.DataParallel(eval_model)
    eval_model.to(device)

    # ── F. FP32 EVALUATION ────────────────────────────────────────
    print("\n══════ FP32 EVALUATION ══════")
    all_results = {}
    all_results["fp32"] = full_eval(
        eval_model, test_loader, criterion, device, "FP32", num_classes
    )
    cka_fp16 = cka_int8 = None

    if not args.skip_quant:
        # ── G. FP16 ───────────────────────────────────────────────
        print("\n══════ FP16 QUANTIZATION & EVAL ══════")
        fp16_model = quantize_fp16(fp32_model)
        if n_gpu > 1:
            fp16_model = nn.DataParallel(fp16_model)
        fp16_model.to(device)
        all_results["fp16"] = full_eval(
            fp16_model, test_loader, criterion, device, "FP16", num_classes
        )

        # ── H. INT8 ───────────────────────────────────────────────
        print("\n══════ INT8 QUANTIZATION & EVAL ══════")
        int8_model    = quantize_int8(fp32_model)
        cpu_criterion = nn.CrossEntropyLoss(weight=weight_tensor.cpu())
        cpu_loader    = DataLoader(test_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers,
                                   pin_memory=False)
        all_results["int8"] = full_eval(
            int8_model, cpu_loader, cpu_criterion,
            torch.device("cpu"), "INT8", num_classes
        )

        # ── I. REPRESENTATION ANALYSIS ────────────────────────────
        print("\n══════ REPRESENTATION ANALYSIS ══════")
        repr_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                 num_workers=4)

        # FP32 hidden states
        fp32_repr = copy.deepcopy(fp32_model).to(device)
        h_fp32    = extract_hidden_states(fp32_repr, repr_loader,
                                          device, args.cka_samples)
        del fp32_repr
        np.save(os.path.join(args.results_dir, "hidden_fp32.npy"), h_fp32)

        # FP16 hidden states
        fp16_repr = quantize_fp16(fp32_model).to(device)
        h_fp16    = extract_hidden_states(fp16_repr, repr_loader,
                                          device, args.cka_samples)
        del fp16_repr
        np.save(os.path.join(args.results_dir, "hidden_fp16.npy"), h_fp16)

        # INT8 hidden states (CPU)
        fp8_repr = quantize_int8(fp32_model)
        h_int8   = extract_hidden_states(fp8_repr, repr_loader,
                                         torch.device("cpu"), args.cka_samples)
        del fp8_repr
        np.save(os.path.join(args.results_dir, "hidden_int8.npy"), h_int8)

        cka_fp16 = representation_analysis(h_fp32, h_fp16, "FP32_vs_FP16")
        cka_int8 = representation_analysis(h_fp32, h_int8, "FP32_vs_INT8")
        all_results["cka_fp16"] = cka_fp16
        all_results["cka_int8"] = cka_int8

        print(f"\n{'Layer':<6} {'FP16 CKA':>10} {'INT8 CKA':>10} "
              f"{'FP16 L2':>10} {'INT8 L2':>10}")
        print("─" * 50)
        for i in range(13):
            fl = cka_fp16["layers"][i]; il = cka_int8["layers"][i]
            print(f"{i:<6} {fl['cka']:>10.6f} {il['cka']:>10.6f} "
                  f"{fl['l2_distance']:>10.6f} {il['l2_distance']:>10.6f}")

    # ── J. PLOTS ──────────────────────────────────────────────────
    print("\n══════ GENERATING PLOTS ══════")
    plot_performance_comparison(all_results, args.results_dir, occupations)
    plot_eod_per_occupation(all_results, occupations, args.results_dir)
    plot_eod_delta(all_results, occupations, args.results_dir)
    plot_per_gender_accuracy(all_results, occupations, args.results_dir)
    if cka_fp16 and cka_int8:
        plot_representation_drift(cka_fp16, cka_int8, args.results_dir)
    plot_summary_dashboard(all_results, cka_fp16, cka_int8,
                           occupations, args.results_dir)

    # ── K. SAVE RESULTS ───────────────────────────────────────────
    results_path = os.path.join(args.results_dir, "bios_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults JSON saved → {results_path}")

    print_summary_table(all_results)
    print(f"\n[Done] All outputs in {args.results_dir}/\n")


if __name__ == "__main__":
    main()