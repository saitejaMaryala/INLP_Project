from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def compute_metrics(logits, labels):

    preds = (logits > 0).astype(int)

    acc = accuracy_score(labels, preds)

    macro_f1 = f1_score(labels, preds, average="macro")

    return acc, macro_f1


