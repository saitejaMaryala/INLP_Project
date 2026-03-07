import os
import random
import torch
import numpy as np
import pandas as pd

from torch.optim import AdamW
from sklearn.model_selection import train_test_split

from models.frozen_bert import FrozenBertClassifier
from utils.metrics import compute_metrics

from transformers import BertTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from textdataset import TextDataset


# -----------------------------
# Create results directory
# -----------------------------
os.makedirs("results", exist_ok=True)


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/jigsaw_data/train.csv")

texts = df["comment_text"].astype(str).tolist()

label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

labels = df[label_cols].values.tolist()


# -----------------------------
# Train / Validation split
# -----------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=67
)


# -----------------------------
# Tokenizer + Dataset
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

collator = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collator
)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model
# -----------------------------
model = FrozenBertClassifier(num_labels=6).to(device)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)

# -----------------------------
# Optimizer + Loss
# -----------------------------
optimizer = AdamW(
    model.classifier.parameters(),
    lr=1e-3
)

criterion = torch.nn.BCEWithLogitsLoss()


# -----------------------------
# Training
# -----------------------------
EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()
    epoch_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits, _ = model(input_ids, attention_mask)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")


# -----------------------------
# Validation
# -----------------------------
model.eval()

all_logits = []
all_labels = []

with torch.no_grad():

    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].cpu().numpy()

        logits, _ = model(input_ids, attention_mask)

        logits = logits.cpu().numpy()

        all_logits.append(logits)
        all_labels.append(labels)


all_logits = np.concatenate(all_logits)
all_labels = np.concatenate(all_labels)


# -----------------------------
# Metrics
# -----------------------------
acc, macro_f1 = compute_metrics(all_logits, all_labels)

print("Validation Accuracy:", acc)
print("Validation Macro F1:", macro_f1)


# -----------------------------
# Save FP32 logits
# -----------------------------
np.save("results/jigsaw_logits.npy", all_logits)

print("Saved FP32 logits → results/jigsaw_logits.npy")


# -----------------------------
# Deterministic 1000-sample subset
# -----------------------------
random.seed(67)

sample_indices = random.sample(range(len(val_dataset)), 1000)

sample_texts = [val_texts[i] for i in sample_indices]
sample_labels = [val_labels[i] for i in sample_indices]

sample_dataset = TextDataset(sample_texts, sample_labels, tokenizer)

sample_loader = DataLoader(
    sample_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collator
)


# -----------------------------
# Extract hidden states
# -----------------------------
hidden_states_storage = [[] for _ in range(13)]

with torch.no_grad():

    for batch in sample_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        _, hidden_states = model(input_ids, attention_mask)

        for i in range(13):

            layer_states = hidden_states[i].cpu()

            hidden_states_storage[i].append(layer_states)


# Combine batches
for i in range(13):

    hidden_states_storage[i] = torch.cat(hidden_states_storage[i], dim=0)


# -----------------------------
# Save hidden states
# -----------------------------
torch.save(
    hidden_states_storage,
    "results/jigsaw_hidden_states.pt"
)

print("Saved hidden states → results/jigsaw_hidden_states.pt")