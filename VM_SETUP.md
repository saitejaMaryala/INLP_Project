# VM Setup Guide - BERT Quantization Fairness Analysis

## Files to Copy to VM

Copy the following directory structure to your VM:

```
INLP_Project/
├── training/
│   └── train_frozen_bert.py          # Main training script
├── utils/
│   └── data_loaders.py                # Data loading utilities
├── data/
│   ├── jigsaw_data/
│   │   ├── train.csv                  # Jigsaw training data
│   │   ├── test.csv                   # Jigsaw test data
│   │   └── test_labels.csv            # Jigsaw test labels
│   ├── bias_in_bios/                  # Full HuggingFace dataset folder
│   │   ├── dataset_dict.json
│   │   ├── train/
│   │   ├── test/
│   │   └── dev/
│   └── Equity-Evaluation-Corpus.csv   # Counterfactual pairs
└── requirements.txt                    # Python dependencies
```

## Minimal Copy Command

If using `scp` from Windows to Linux VM:

```powershell
# From project root directory
scp -r training utils data requirements.txt user@vm-address:/path/to/destination/
```

Or create a zip file:

```powershell
# Compress required files
Compress-Archive -Path training,utils,data,requirements.txt -DestinationPath vm_upload.zip

# Copy to VM
scp vm_upload.zip user@vm-address:/path/to/destination/

# On VM, extract
unzip vm_upload.zip
```

## VM Setup Steps

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate   # On Windows

# Install packages
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU access
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 2. Run Sanity Check

Test the entire pipeline with a small dataset before full training:

```bash
# Quick test on Jigsaw (1000 samples, 2 epochs, no quantization)
python training/train_frozen_bert.py --sanity-check --task jigsaw

# Expected output:
# - Loads 1000 training samples
# - Trains for 2 epochs (~2-3 minutes on 4 GPUs)
# - Evaluates accuracy
# - Skips quantization and CKA
```

**What the sanity check does:**
- ✅ Verifies all imports work
- ✅ Tests data loading
- ✅ Confirms 4 GPUs are detected
- ✅ Trains small model quickly
- ✅ Validates pipeline end-to-end
- ❌ Skips quantization (faster)
- ❌ Skips CKA analysis (faster)

### 3. Run Full Training

Once sanity check passes, run full training:

#### Option A: Jigsaw Only (Toxicity Classification)

```bash
# Full training with FP32, FP16, and INT8
python training/train_frozen_bert.py \
    --task jigsaw \
    --epochs 10 \
    --batch-size 32 \
    --data-dir data \
    --save-dir models \
    --results-dir results

# Estimated time: 1-2 hours on 4 GPUs
```

#### Option B: Bias in Bios Only (Occupation Classification)

```bash
# Full training with 28-class occupation task
python training/train_frozen_bert.py \
    --task bios \
    --epochs 10 \
    --batch-size 16 \
    --bios-max-length 256 \
    --data-dir data \
    --save-dir models \
    --results-dir results

# Estimated time: 3-4 hours on 4 GPUs
```

#### Option C: Both Tasks

```bash
# Run both Jigsaw and Bias in Bios sequentially
python training/train_frozen_bert.py \
    --task both \
    --epochs 10 \
    --batch-size 32 \
    --data-dir data

# Estimated time: 4-6 hours on 4 GPUs
```

### 4. Advanced Options

```bash
# Custom learning rate and batch size
python training/train_frozen_bert.py \
    --task jigsaw \
    --epochs 15 \
    --batch-size 64 \
    --lr 5e-4

# Skip INT8, only do FP16
python training/train_frozen_bert.py \
    --task jigsaw \
    --quantization-types fp16

# Train on subset (e.g., 10,000 samples for faster iteration)
python training/train_frozen_bert.py \
    --task jigsaw \
    --sample-size 10000 \
    --epochs 5

# More CKA samples (higher resolution analysis)
python training/train_frozen_bert.py \
    --task jigsaw \
    --cka-samples 5000
```

## Expected Outputs

After successful training, you'll have:

### Models Directory (`models/`)

```
models/
├── jigsaw_fp32.pt      # Trained Jigsaw model (FP32)
└── bios_fp32.pt        # Trained Bios model (FP32)
```

### Results Directory (`results/`)

```
results/
├── fairness_analysis_results.json   # All fairness metrics
├── jigsaw_hidden_fp32.npy           # FP32 representations
├── jigsaw_hidden_fp16.npy           # FP16 representations
├── jigsaw_hidden_int8.npy           # INT8 representations
├── bios_hidden_fp32.npy
├── bios_hidden_fp16.npy
└── bios_hidden_int8.npy
```

### Console Logs

The script prints detailed progress:

- GPU detection and batch size calculations
- Data loading statistics
- Training progress (loss, accuracy per epoch)
- Fairness metrics (DPD, EOD, CFR)
- Layer-wise CKA scores
- Accuracy comparisons across precisions

## Monitoring GPU Usage

While training, monitor GPU utilization:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or check once
nvidia-smi
```

Expected GPU usage:
- **4 GPUs active:** Yes (via DataParallel)
- **Memory per GPU:** ~4-6 GB for batch_size=32
- **Utilization:** 80-100% during training

## Troubleshooting

### Issue: "Import torch could not be resolved"

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "CUDA out of memory"

```bash
# Reduce batch size
python training/train_frozen_bert.py --batch-size 16
```

### Issue: "FileNotFoundError: data/jigsaw_data/train.csv"

```bash
# Verify data copied correctly
ls -lh data/jigsaw_data/
ls -lh data/bias_in_bios/

# Re-download if needed
cd data
python download.py
```

### Issue: "Only 1 GPU detected instead of 4"

```bash
# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# If restricted, unset it
unset CUDA_VISIBLE_DEVICES

# Verify
python -c "import torch; print(torch.cuda.device_count())"
```

## Command Reference

| Command | Purpose | Time |
|---------|---------|------|
| `--sanity-check` | Quick pipeline test | 2-3 min |
| `--task jigsaw` | Train toxicity classifier | 1-2 hrs |
| `--task bios` | Train occupation classifier | 3-4 hrs |
| `--task both` | Train both tasks | 4-6 hrs |
| `--sample-size 1000` | Limit dataset for testing | Varies |
| `--epochs 10` | Set number of epochs | Varies |
| `--skip-quantization` | Skip FP16/INT8 (faster) | -50% |

## Example Workflow

```bash
# Step 1: Verify environment
source venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}, GPUs: {torch.cuda.device_count()}')"

# Step 2: Sanity check
python training/train_frozen_bert.py --sanity-check --task jigsaw

# Step 3: If successful, run full Jigsaw training
python training/train_frozen_bert.py --task jigsaw --epochs 10 --batch-size 32

# Step 4: Run Bias in Bios
python training/train_frozen_bert.py --task bios --epochs 10 --batch-size 16

# Step 5: Analyze results
python -c "import json; print(json.dumps(json.load(open('results/fairness_analysis_results.json')), indent=2))"
```

## Notes

- **Batch size per GPU:** The `--batch-size` parameter is per-GPU, so effective batch size = batch_size × num_gpus
- **Checkpointing:** Currently no mid-training checkpoints. Add if training takes >6 hours.
- **Reproducibility:** Random seed is set to 42 by default. Change with `--seed`.
- **VRAM:** ~24 GB minimum recommended for batch_size=32 on 4 GPUs.
