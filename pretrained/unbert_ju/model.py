"""
BERT Training Script for Toxicity Classification
================================================
Trains FP32 BERT baseline on Jigsaw Unintended Bias dataset.
Saves the trained model for later quantization and comparison.

Directory structure expected:
  data/jigsaw_data/
    train.csv

Install dependencies:
  pip install torch transformers pandas scikit-learn tqdm
"""

# ─────────────────────────────────────────────
# 0. IMPORTS & CONFIG
# ─────────────────────────────────────────────
import os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Reproducibility ──
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Paths ──
DATA_DIR        = "/ssd_scratch/sai.teja/INLP_Project/data/jigsaw_uni"
TRAIN_CSV       = os.path.join(DATA_DIR, "train.csv")
OUTPUT_DIR      = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters ──
MAX_LEN         = 128          # token sequence length
BATCH_SIZE      = 32
EPOCHS          = 5
LR              = 1e-3         # high LR — only training head, not BERT
TOXICITY_THRESH = 0.5          # binarise target >= 0.5 → toxic
TRAIN_SAMPLE    = 80_000       # subsample for speed (set None for full dataset)
VAL_SAMPLE      = 20_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_jigsaw(path, n_train=TRAIN_SAMPLE, n_val=VAL_SAMPLE, seed=SEED):
    """Load, binarise, and split the Jigsaw train CSV."""
    print(f"\n[Data] Loading {path} …")
    df = pd.read_csv(path)

    # Binarize target column (continuous toxicity score → binary)
    df["label"] = (df["target"] >= TOXICITY_THRESH).astype(int)

    # Keep only text and label
    df = df[["comment_text", "label"]]
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
# 2. MODEL ARCHITECTURE — Frozen BERT + MLP Head
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
            output_hidden_states=True,   # ← exposes all 12 transformer layers
        )
        # Freeze entire BERT encoder
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)   # only this trains

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token from final hidden layer  →  shape (batch, 768)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        logits  = self.classifier(self.dropout(cls_emb))
        return logits, outputs.hidden_states   # hidden_states: tuple of 13 tensors


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────

def train_model(model, train_df, val_df, tokenizer):
    print("\n[Train] Starting FP32 baseline training …")
    model = model.to(DEVICE)

    train_ds = JigsawDataset(train_df, tokenizer)
    val_ds   = JigsawDataset(val_df,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Class-weighted loss to handle severe imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01,
    )

    best_f1, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"  Epoch {epoch}/{EPOCHS} [train]", leave=False):
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            lbls  = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(ids, mask)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ── Validate ──
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"  Epoch {epoch}/{EPOCHS} [val]  ", leave=False):
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                logits, _ = model(ids, mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="macro")
        print(f"  Epoch {epoch}: loss={total_loss/len(train_dl):.4f}  acc={acc:.4f}  macro-F1={f1:.4f}")

        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(OUTPUT_DIR, "fp32_classifier_uni.pt")
    torch.save(best_state, ckpt_path)
    print(f"  Best macro-F1={best_f1:.4f}  → saved to {ckpt_path}")
    return model


# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BERT TOXICITY CLASSIFIER — TRAINING SCRIPT")
    print("=" * 60)
    
    # Load data
    train_df, val_df = load_jigsaw(TRAIN_CSV)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Train FP32 baseline
    fp32_model = FrozenBertClassifier(num_labels=2)
    fp32_model = train_model(fp32_model, train_df, val_df, tokenizer)

    print(f"\n[Done] Model training complete!")
    print(f"  Saved model: {os.path.join(OUTPUT_DIR, 'fp32_classifier.pt')}")
    print(f"  Use 'compare_quantized.py' to quantize and evaluate fairness.\n")


if __name__ == "__main__":
    main()