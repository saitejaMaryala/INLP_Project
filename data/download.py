import os
import zipfile
import shutil

import kagglehub
from datasets import load_dataset


os.makedirs("data/jigsaw_data", exist_ok=True)

# -------------------------
# Download Jigsaw dataset
# -------------------------
print("Downloading Jigsaw dataset...")

path = kagglehub.competition_download(
    "jigsaw-toxic-comment-classification-challenge"
)

zip_path = os.path.join(path, "train.csv.zip")

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall("data/jigsaw_data")

print("Jigsaw dataset ready at data/jigsaw_data/train.csv")


# -------------------------
# Download Bias in Bios
# -------------------------
print("Downloading Bias in Bios dataset...")

dataset = load_dataset("LabHC/bias_in_bios")

dataset.save_to_disk("data/bias_in_bios")

print("Bias in Bios saved to data/bias_in_bios")