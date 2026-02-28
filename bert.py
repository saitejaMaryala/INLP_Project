import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch

from textdataset import TextDataset

df = pd.read_csv("data/jigsaw_data/train.csv")
texts = df["comment_text"].tolist()
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
labels = df[label_cols].values.tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

data_collator = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)