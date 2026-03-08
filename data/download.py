import os
import zipfile

# import kagglehub
from datasets import load_dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JIGSAW_DIR = os.path.join(BASE_DIR, "jigsaw_data")
BIOS_DIR = os.path.join(BASE_DIR, "bias_in_bios")

os.makedirs(JIGSAW_DIR, exist_ok=True)


# # -------------------------
# # Download Jigsaw dataset
# # -------------------------
# print("Downloading Jigsaw dataset...")

# path = kagglehub.competition_download(
#     "jigsaw-toxic-comment-classification-challenge"
# )

# zip_path = os.path.join(path, "train.csv.zip")

# with zipfile.ZipFile(zip_path, "r") as z:
#     z.extractall(JIGSAW_DIR)

# print("Jigsaw dataset ready at", os.path.join(JIGSAW_DIR, "train.csv"))


# -------------------------
# Download Bias in Bios
# -------------------------
print("Downloading Bias in Bios dataset...")

dataset = load_dataset("LabHC/bias_in_bios")

dataset.save_to_disk(BIOS_DIR)

print("Bias in Bios saved to", BIOS_DIR)