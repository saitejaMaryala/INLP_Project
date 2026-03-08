#!/bin/bash
# Quick VM Setup Script
# Run after copying files to VM

echo "====================================================="
echo "BERT Quantization Fairness - VM Setup"
echo "====================================================="

# 1. Create virtual environment
echo -e "\n[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
echo -e "\n[2/5] Upgrading pip..."
pip install --upgrade pip

# 3. Install dependencies
echo -e "\n[3/5] Installing dependencies..."
pip install -r requirements.txt

# 4. Verify GPU access
echo -e "\n[4/5] Checking GPU availability..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ GPUs detected: {torch.cuda.device_count()}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')"

# 5. Verify data files
echo -e "\n[5/5] Verifying data files..."
if [ -f "data/jigsaw_data/train.csv" ]; then
    echo "✓ Jigsaw data found"
else
    echo "✗ Jigsaw data missing"
fi

if [ -d "data/bias_in_bios/train" ]; then
    echo "✓ Bias in Bios data found"
else
    echo "✗ Bias in Bios data missing"
fi

if [ -f "data/Equity-Evaluation-Corpus.csv" ]; then
    echo "✓ Equity Evaluation Corpus found"
else
    echo "✗ Equity Evaluation Corpus missing"
fi

# Create output directories
echo -e "\nCreating output directories..."
mkdir -p models results

echo -e "\n====================================================="
echo "Setup complete! Next steps:"
echo "====================================================="
echo "1. Run sanity check:"
echo "   python training/train_frozen_bert.py --sanity-check --task jigsaw"
echo ""
echo "2. If successful, run full training:"
echo "   python training/train_frozen_bert.py --task jigsaw --epochs 10"
echo "====================================================="
