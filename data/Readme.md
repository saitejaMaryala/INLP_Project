# Datasets

This directory contains the datasets used in this project.

## 1. Jigsaw Toxic Comment Classification Challenge
- **Location:** Stored locally in the `jigsaw_data/` folder.
- **Download Method:** Downloaded using the `kagglehub` library. 
  - *Example:* `kagglehub.competition_download("jigsaw-toxic-comment-classification-challenge")` (See `download.py` for reference).

## 2. Bias in Bios
- **Download Method:** Use the `download.py` script to fetch this dataset using the Hugging Face `datasets` library.
  - *Example:* `load_dataset("LabHC/bias_in_bios")` (See `download.py`).

## 3. Equity Evaluation Corpus (EEC)
- **Location:** Available as the file `Equity-Evaluation-Corpus.csv`.
- **Download Method:** Downloaded from Hugging Face.

## Scripts
- **`download.py`:** A utility script containing the code snippets used to download the Jigsaw and Bias in Bios datasets.