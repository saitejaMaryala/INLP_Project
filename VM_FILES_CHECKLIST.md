# Files to Copy to VM - Checklist

## ✅ Required Files

### Code (Python scripts)
- [ ] `training/train_frozen_bert.py` - Main training script
- [ ] `utils/data_loaders.py` - Data loading utilities
- [ ] `requirements.txt` - Python dependencies
- [ ] `setup_vm.sh` - Automated setup script (optional)

### Data Files
- [ ] `data/jigsaw_data/train.csv` (~66 MB)
- [ ] `data/jigsaw_data/test.csv` (~58 MB)
- [ ] `data/jigsaw_data/test_labels.csv` (~5 MB)
- [ ] `data/bias_in_bios/` (entire folder, ~500 MB)
  - [ ] `data/bias_in_bios/dataset_dict.json`
  - [ ] `data/bias_in_bios/train/` folder
  - [ ] `data/bias_in_bios/test/` folder
  - [ ] `data/bias_in_bios/dev/` folder
- [ ] `data/Equity-Evaluation-Corpus.csv` (~1 MB)

## 📦 Total Size
Approximately **630 MB** of data + code

## 🚀 Quick Copy Commands

### Option 1: SCP (from Windows to Linux VM)

```powershell
# From project root on your local machine
scp -r training utils data requirements.txt setup_vm.sh user@vm-ip:/home/user/INLP_Project/
```

### Option 2: Create Zip Archive

```powershell
# Create archive (Windows PowerShell)
Compress-Archive -Path training,utils,data,requirements.txt,setup_vm.sh -DestinationPath vm_files.zip

# Copy to VM
scp vm_files.zip user@vm-ip:/home/user/

# On VM, extract
unzip vm_files.zip
```

### Option 3: Rsync (most efficient for updates)

```bash
rsync -avz --progress training utils data requirements.txt setup_vm.sh user@vm-ip:/home/user/INLP_Project/
```

## ⚡ On the VM

After copying files:

```bash
# Make setup script executable
chmod +x setup_vm.sh

# Run automated setup
./setup_vm.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Run sanity check
python training/train_frozen_bert.py --sanity-check --task jigsaw
```

## ❓ Don't Copy (Not Needed)

- ❌ `.venv/` - Virtual environment (recreate on VM)
- ❌ `__pycache__/` - Python cache
- ❌ `docs/` - Documentation only
- ❌ `models/` - Will be generated
- ❌ `results/` - Will be generated
- ❌ `.git/` - Git repository
- ❌ `bert.py`, `textdataset.py` - Old scripts (not used)
- ❌ `datapre.ipynb` - Jupyter notebook (not needed)

## 📝 File Size Reference

| File/Folder | Size | Required |
|-------------|------|----------|
| `data/jigsaw_data/` | ~130 MB | Yes |
| `data/bias_in_bios/` | ~500 MB | Yes (for bios task) |
| `data/Equity-Evaluation-Corpus.csv` | ~1 MB | Yes (for CFR) |
| `training/train_frozen_bert.py` | ~30 KB | Yes |
| `utils/data_loaders.py` | ~15 KB | Yes |
| `requirements.txt` | 1 KB | Yes |

## 🎯 Minimal Copy (Jigsaw Only)

If you only want to run Jigsaw toxicity classification:

```powershell
scp -r training utils requirements.txt user@vm-ip:/home/user/INLP_Project/
scp -r data/jigsaw_data data/Equity-Evaluation-Corpus.csv user@vm-ip:/home/user/INLP_Project/data/
```

Then run with: `--task jigsaw`

## 🎯 Minimal Copy (Bios Only)

If you only want to run Bias in Bios:

```powershell
scp -r training utils requirements.txt user@vm-ip:/home/user/INLP_Project/
scp -r data/bias_in_bios user@vm-ip:/home/user/INLP_Project/data/
```

Then run with: `--task bios`
