# Bias in Bios Training Pipeline — Flow & Usage Guide

## What this script does (in order)

| Step | Section | Description |
|------|---------|-------------|
| 1 | **Data loading** | Calls `load_bias_in_bios()` → returns train/test texts, labels (0–27 occupations), genders (0=F, 1=M) |
| 2 | **Sanity trim** | If `--sanity-check`, slices to 500 train / 200 test / 2 epochs / 100 CKA samples |
| 3 | **Tokenization** | `BertTokenizer` pads/truncates to `--max-length` (default 256) |
| 4 | **Model build** | `FrozenBertClassifier`: frozen `bert-base-uncased` + trainable `Dropout → Linear(768→28)` |
| 5 | **DataParallel** | Wraps model in `nn.DataParallel` when >1 GPU detected |
| 6 | **Training loop** | AdamW on classifier only, class-weighted CE loss, tqdm progress bars |
| 7 | **Best model save** | Saves checkpoint with highest validation accuracy to `models/bios/best_bios_fp32.pt` |
| 8 | **FP32 eval** | Accuracy, Macro-F1, EOD (per-class TPR gap Female−Male) |
| 9 | **FP16 quant** | `model.bert.half()` — encoder to half precision, classifier stays FP32 |
| 10 | **FP16 eval** | Same metrics as step 8 |
| 11 | **INT8 quant** | `torch.quantization.quantize_dynamic` on BERT Linear layers (CPU only) |
| 12 | **INT8 eval** | Same metrics on CPU |
| 13 | **Hidden-state extraction** | CLS embeddings from all 13 BERT layers (FP32, FP16, INT8), saved as `.npy` |
| 14 | **CKA / L2 / Cosine** | Per-layer representation comparison: FP32↔FP16, FP32↔INT8 |
| 15 | **Delta summary** | Prints accuracy, F1, EOD differences across quantization levels |
| 16 | **Results save** | `results/bios/bios_results.json` with all metrics |

---

## Required file structure

```
INLP_Project/
├── training/
│   └── train_bios.py          ← this script
├── utils/
│   ├── __init__.py             ← can be empty (optional if running as module)
│   └── data_loaders.py         ← provides load_bias_in_bios()
├── data/
│   └── bias_in_bios/           ← HuggingFace dataset saved via save_to_disk
│       ├── dataset_dict.json
│       ├── train/
│       ├── test/
│       └── dev/
├── models/
│   └── bios/                   ← created automatically (saves best_bios_fp32.pt)
├── results/
│   └── bios/                   ← created automatically (saves .npy + .json)
└── requirements.txt
```

**If dataset is missing**, run from repo root:
```bash
python data/download.py
```

---

## How to run

### Full run (4 GPUs, 5 epochs)
```bash
python -m training.train_bios --epochs 5 --batch-size 32
```

### Sanity check (fast smoke test)
```bash
python -m training.train_bios --sanity-check
```

### Skip quantization (train + FP32 eval only)
```bash
python -m training.train_bios --epochs 5 --skip-quantization
```

### All CLI flags
| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/bias_in_bios` | Path to saved HuggingFace dataset |
| `--save-dir` | `models/bios` | Where to save model checkpoints |
| `--results-dir` | `results/bios` | Where to save `.npy` and `.json` results |
| `--epochs` | `5` | Training epochs |
| `--batch-size` | `32` | Batch size per GPU |
| `--lr` | `1e-3` | Learning rate (classifier only) |
| `--max-length` | `256` | Max token length |
| `--cka-samples` | `1000` | Number of test samples for CKA analysis |
| `--seed` | `42` | Random seed |
| `--sanity-check` | off | 500 train/200 test, 2 epochs |
| `--skip-quantization` | off | Skip FP16, INT8 and CKA phases |

---

## Outputs

| File | Contents |
|------|----------|
| `models/bios/best_bios_fp32.pt` | Best checkpoint (classifier + frozen BERT state dict) |
| `results/bios/bios_results.json` | All metrics: accuracy, F1, EOD, CKA per layer |
| `results/bios/bios_hidden_fp32.npy` | FP32 hidden states `(13, N, 768)` |
| `results/bios/bios_hidden_fp16.npy` | FP16 hidden states |
| `results/bios/bios_hidden_int8.npy` | INT8 hidden states |

---

## Dependencies

Everything in `requirements.txt`:
```
torch, transformers, datasets, scikit-learn, scipy, numpy, tqdm, fairlearn
```

Install: `pip install -r requirements.txt`
