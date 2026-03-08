"""
Bias in Bios — Frozen BERT Training Script
==========================================
Trains FP32 frozen BERT + trainable classifier head on Bias-in-Bios dataset.
Saves the best model checkpoint for later quantization and evaluation.

PERFORMANCE OPTIMIZATIONS:
  1. Pre-tokenization with disk caching
  2. max_length 128 (covers >95% of bios)
  3. num_workers 8 for multi-GPU feeding
  4. AMP (automatic mixed precision) training
  5. Frozen BERT wrapped in no_grad()

Usage:
    python train_bios.py                    # full training
    python train_bios.py --sanity-check     # quick test (500 samples, 2 epochs)
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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import load_bias_in_bios


# ════════════════════════════════════════════════════════════════
# 1.  MODEL
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
        cls_emb = outputs.last_hidden_state[:, 0, :].float()
        return self.classifier(self.dropout(cls_emb)), outputs.hidden_states


# ════════════════════════════════════════════════════════════════
# 2.  DATASET WITH PRE-TOKENIZATION CACHE
# ════════════════════════════════════════════════════════════════
def tokenise_and_cache(texts, tokenizer, max_length, cache_path):
    """Tokenize once, save to disk for instant loading on subsequent runs."""
    if os.path.exists(cache_path):
        print(f"  [cache] Loading tokenized data from {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        return data["input_ids"], data["attention_mask"]

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
    """Pre-tokenized dataset - no tokenization overhead at training time."""
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
    """AMP training with GradScaler."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        ids   = batch["input_ids"].to(device, non_blocking=True)
        mask  = batch["attention_mask"].to(device, non_blocking=True)
        lbls  = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits, _ = model(ids, mask)
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
            logits, _ = model(ids, mask)
            loss = criterion(logits, lbls)

        total_loss += loss.item() * lbls.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(lbls.cpu().tolist())

    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, acc, f1


# ════════════════════════════════════════════════════════════════
# 4.  MAIN
# ════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Bias-in-Bios frozen BERT training")
    p.add_argument("--data-dir",     default="data/bias_in_bios")
    p.add_argument("--save-dir",     default="models/bios")
    p.add_argument("--cache-dir",    default="cache/bios")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--max-length",   type=int,   default=128)
    p.add_argument("--num-workers",  type=int,   default=8)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--sanity-check", action="store_true",
                   help="500 train / 200 test / 2 epochs — quick smoke test")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir,  exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs: {n_gpu}")

    # Load data
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

    # Tokenize & cache
    print("\n══════ TOKENIZING ══════")
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

    # Build model
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

    # Train
    print("\n══════ TRAINING (FP32 + AMP) ══════")
    ckpt_path = os.path.join(args.save_dir, "best_bios_fp32.pt")
    best_acc = 0.0

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

        if val_acc > best_acc:
            best_acc  = val_acc
            # Unwrap DataParallel if present
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"  ✓ Best saved (val_acc={val_acc:.4f}) → {ckpt_path}")

    print(f"\n[Done] Training complete!")
    print(f"  Best validation accuracy: {best_acc:.4f}")
    print(f"  Saved model: {ckpt_path}")
    print(f"  Use 'compare_quant_bios.py' to quantize and evaluate fairness.\n")


if __name__ == "__main__":
    main()
