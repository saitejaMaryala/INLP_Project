

















"""
Dataset bootstrap utility.

What it does:
1) Bias in Bios:
   - Tries to download the real dataset from Hugging Face.
   - If download fails and --fallback-synthetic is enabled, creates a synthetic
     DatasetDict compatible with utils.data_loader.load_bias_in_bios.

2) Jigsaw Unintended Bias in Toxicity Classification:
     - Tries to download from Kaggle competition:
         jigsaw-unintended-bias-in-toxicity-classification
     - Stores real train.csv in data/jigsaw_bias_data (compatible with
         utils.data_loader.load_jigsaw_unintended_bias).
     - If Kaggle/API setup is not available and --fallback-synthetic is enabled,
         creates synthetic train/test/test_labels CSV files in data/jigsaw_data.

Usage examples:
  python data/download.py --dataset all --fallback-synthetic
  python data/download.py --dataset bias_in_bios --fallback-synthetic
  python data/download.py --dataset jigsaw --fallback-synthetic
"""

import argparse
import os
import shutil
import subprocess
import zipfile
from typing import List

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JIGSAW_DIR = os.path.join(BASE_DIR, "jigsaw_data")
JIGSAW_BIAS_DIR = os.path.join(BASE_DIR, "jigsaw_bias_data")
BIOS_DIR = os.path.join(BASE_DIR, "bias_in_bios")


OCCUPATIONS: List[str] = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician",
    "poet", "professor", "psychologist", "rapper", "software_engineer",
    "surgeon", "teacher", "yoga_teacher",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _ensure_bias_in_bios(fallback_synthetic: bool) -> None:
    if os.path.exists(BIOS_DIR):
        try:
            _ = load_from_disk(BIOS_DIR)
            print(f"[bias_in_bios] Found existing dataset at {BIOS_DIR}")
            return
        except Exception:
            print("[bias_in_bios] Existing folder is invalid. Rebuilding...")

    print("[bias_in_bios] Trying Hugging Face download: LabHC/bias_in_bios")
    try:
        dataset = load_dataset("LabHC/bias_in_bios")
        dataset.save_to_disk(BIOS_DIR)
        print(f"[bias_in_bios] Real dataset saved to {BIOS_DIR}")
        return
    except Exception as e:
        print(f"[bias_in_bios] Real download failed: {type(e).__name__}: {e}")

    if not fallback_synthetic:
        raise RuntimeError(
            "Bias in Bios download failed and synthetic fallback is disabled."
        )

    print("[bias_in_bios] Creating synthetic fallback dataset...")
    rng = np.random.default_rng(42)

    def build_split(n_per_occ: int) -> Dataset:
        rows = {"hard_text": [], "profession": [], "gender": []}
        for occ_idx, occ in enumerate(OCCUPATIONS):
            for i in range(n_per_occ):
                gender = "f" if (i % 2 == 0) else "m"
                pronoun = "She" if gender == "f" else "He"
                text = (
                    f"{pronoun} has worked as a {occ.replace('_', ' ')} for years. "
                    f"This biography sample #{i} is for synthetic fallback data."
                )
                if rng.random() < 0.15:
                    text += " They also mentor students and publish articles."
                rows["hard_text"].append(text)
                rows["profession"].append(occ_idx)
                rows["gender"].append(gender)
        return Dataset.from_dict(rows)

    dataset = DatasetDict(
        {
            "train": build_split(n_per_occ=30),  # 840 samples
            "test": build_split(n_per_occ=8),    # 224 samples
        }
    )
    _ensure_dir(BIOS_DIR)
    dataset.save_to_disk(BIOS_DIR)
    print(f"[bias_in_bios] Synthetic dataset saved to {BIOS_DIR}")


def _extract_if_zip(path: str, target_dir: str) -> None:
    if not path.lower().endswith(".zip"):
        return
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(target_dir)


def _download_jigsaw_unintended_bias() -> bool:
    """Download Kaggle competition files and extract train.csv into jigsaw_bias_data."""
    _ensure_dir(JIGSAW_BIAS_DIR)
    train_csv = os.path.join(JIGSAW_BIAS_DIR, "train.csv")
    if os.path.exists(train_csv):
        print(f"[jigsaw_unintended] Found existing train.csv at {train_csv}")
        return True

    try:
        kaggle_bin = shutil.which("kaggle")
        if kaggle_bin is None:
            raise RuntimeError("kaggle CLI not found in PATH")

        print("[jigsaw_unintended] Downloading from Kaggle competition...")
        cmd = [
            kaggle_bin,
            "competitions",
            "download",
            "-c",
            "jigsaw-unintended-bias-in-toxicity-classification",
            "-p",
            JIGSAW_BIAS_DIR,
        ]
        subprocess.run(cmd, check=True)

        for name in os.listdir(JIGSAW_BIAS_DIR):
            _extract_if_zip(os.path.join(JIGSAW_BIAS_DIR, name), JIGSAW_BIAS_DIR)

        if os.path.exists(train_csv):
            print(f"[jigsaw_unintended] Real dataset ready at {train_csv}")
            return True

        raise RuntimeError("Download completed but train.csv not found after extraction")
    except Exception as e:
        print(f"[jigsaw_unintended] Real download failed: {type(e).__name__}: {e}")
        return False


def _ensure_jigsaw(fallback_synthetic: bool) -> None:
    if _download_jigsaw_unintended_bias():
        return

    if not fallback_synthetic:
        raise RuntimeError(
            "Jigsaw Unintended Bias download failed and synthetic fallback is disabled. "
            "Configure Kaggle credentials (~/.kaggle/kaggle.json) and retry."
        )

    _ensure_dir(JIGSAW_DIR)
    req = [
        os.path.join(JIGSAW_DIR, "train.csv"),
        os.path.join(JIGSAW_DIR, "test.csv"),
        os.path.join(JIGSAW_DIR, "test_labels.csv"),
    ]
    if all(os.path.exists(p) for p in req):
        print(f"[jigsaw_data] Found existing synthetic fallback CSV files in {JIGSAW_DIR}")
        return

    print("[jigsaw_data] Creating synthetic fallback CSVs...")
    toxic_cols = [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
    ]

    # Synthetic train set
    train_rows = []
    for i in range(3000):
        is_toxic = int(i % 5 == 0)
        train_rows.append(
            {
                "id": i,
                "comment_text": (
                    "This is a toxic synthetic comment." if is_toxic
                    else "This is a neutral synthetic comment."
                ),
                "toxic": is_toxic,
                "severe_toxic": int(is_toxic and i % 2 == 0),
                "obscene": int(is_toxic and i % 3 == 0),
                "threat": int(is_toxic and i % 11 == 0),
                "insult": int(is_toxic and i % 4 == 0),
                "identity_hate": int(is_toxic and i % 13 == 0),
            }
        )

    train_df = pd.DataFrame(train_rows)

    # Synthetic test set + labels
    test_rows = []
    label_rows = []
    for i in range(1200):
        idx = 100000 + i
        is_toxic = int(i % 6 == 0)
        test_rows.append(
            {
                "id": idx,
                "comment_text": (
                    "Test toxic synthetic comment." if is_toxic
                    else "Test neutral synthetic comment."
                ),
            }
        )
        label_rows.append(
            {
                "id": idx,
                "toxic": is_toxic,
                "severe_toxic": int(is_toxic and i % 2 == 0),
                "obscene": int(is_toxic and i % 3 == 0),
                "threat": int(is_toxic and i % 11 == 0),
                "insult": int(is_toxic and i % 4 == 0),
                "identity_hate": int(is_toxic and i % 13 == 0),
            }
        )

    test_df = pd.DataFrame(test_rows)
    test_labels_df = pd.DataFrame(label_rows)

    train_df.to_csv(req[0], index=False)
    test_df.to_csv(req[1], index=False)
    test_labels_df.to_csv(req[2], index=False)

    print(f"[jigsaw_data] Synthetic CSV files saved to {JIGSAW_DIR}")
    print(f"[jigsaw_data] Columns: id, comment_text, {', '.join(toxic_cols)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap project datasets")
    p.add_argument(
        "--dataset",
        choices=["bias_in_bios", "jigsaw", "all"],
        default="all",
        help="Which dataset to prepare",
    )
    p.add_argument(
        "--fallback-synthetic",
        action="store_true",
        help="Create synthetic fallback data when real download is unavailable",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Data root: {BASE_DIR}")

    if args.dataset in ("bias_in_bios", "all"):
        _ensure_bias_in_bios(args.fallback_synthetic)

    if args.dataset in ("jigsaw", "all"):
        _ensure_jigsaw(args.fallback_synthetic)

    print("Dataset bootstrap complete.")


if __name__ == "__main__":
    main()
