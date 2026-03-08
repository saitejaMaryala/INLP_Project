# Training Pipeline Flow Documentation

## Overview

This document explains the complete flow of the BERT frozen encoder quantization fairness analysis pipeline.

---

## Architecture Summary

```
┌─────────────────────────────────────┐
│   BERT-base-uncased Encoder         │
│   (110M parameters)                 │
│   ✗ FROZEN - requires_grad=False    │  ← Never updated during training
│   ✓ Subject to quantization         │  ← FP32 → FP16 → INT8
└─────────────────────────────────────┘
                  ↓
              [CLS] token
           (768-dimensional)
                  ↓
┌─────────────────────────────────────┐
│   Dropout Layer (p=0.1)             │
│   ✓ TRAINABLE                       │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│   Linear Classifier                 │
│   (768 → num_classes)               │
│   ✓ TRAINABLE - requires_grad=True  │  ← Only this gets gradient updates
│   ✗ Always remains FP32             │  ← Never quantized
└─────────────────────────────────────┘
                  ↓
              Logits
```

---

## Phase 1: Baseline Training (FP32)

### What We're Training

**NOT training:** BERT encoder weights (frozen at pre-trained values)  
**ARE training:** Classification head only (Dropout + Linear layer)

### Task 1: Jigsaw Toxicity Classification

**Objective:** Binary classification (Toxic vs Non-Toxic)

**Before Training:**
- Load `bert-base-uncased` from HuggingFace
- Freeze all BERT parameters (`param.requires_grad = False`)
- Initialize random classifier head
- Compute class weights (toxic comments are minority)

**During Training (5-10 epochs):**
- Higher learning rate (1e-3) since only training classifier
- AdamW optimizer with weight decay
- Class-weighted Cross-Entropy Loss
- Multi-GPU support via DataParallel
- Monitor: Loss, Accuracy on validation set

**After Training:**
- Save FP32 model weights
- Extract validation metrics:
  - Accuracy
  - Macro F1-Score
  - Per-class precision/recall

### Task 2: Bias in Bios Occupation Classification

**Objective:** 28-class occupation prediction

**Before Training:**
- Same frozen BERT encoder
- New classifier head (768 → 28 classes)
- Compute class weights (occupations are imbalanced)

**During Training (5-10 epochs):**
- Same training setup as Jigsaw
- Higher complexity (28 classes vs 2)
- Longer sequences (256 tokens vs 128)

**After Training:**
- Save FP32 model weights
- Extract validation metrics

---

## Phase 2: Fairness Evaluation (FP32 Baseline)

### What Metrics We Calculate

**BEFORE quantization** - establish FP32 baseline fairness

#### Metric 1: Demographic Parity Difference (DPD)

**Dataset:** Jigsaw Toxicity  
**When:** After FP32 training  
**What:** Measures if different demographic groups get flagged as toxic at equal rates

```
DPD = max(P(Ŷ=toxic | group)) - min(P(Ŷ=toxic | group))
```

**Groups evaluated:**
- Race: black, white, asian, latino
- Religion: christian, muslim, jewish
- Sexual orientation: heterosexual, homosexual, bisexual
- Gender: male, female, transgender

**Ideal value:** 0.0 (perfect parity)  
**Interpretation:** Lower is better

#### Metric 2: Equal Opportunity Difference (EOD)

**Dataset:** Bias in Bios  
**When:** After FP32 training  
**What:** Measures if model correctly identifies occupations at equal rates for males vs females

```
EOD = TPR(female) - TPR(male)  [for each occupation]
```

**Example:** "Among actual surgeons, does the model identify female surgeons at the same rate as male surgeons?"

**Ideal value:** 0.0 (equal TPR)  
**Interpretation:** Measures performance gap

#### Metric 3: Counterfactual Flip Rate (CFR)

**Dataset:** Equity Evaluation Corpus  
**When:** After FP32 training  
**What:** Measures prediction stability when only demographic terms change

**Example pairs:**
- "Alonzo feels angry" ↔ "Adam feels angry"
- "She is a doctor" ↔ "He is a doctor"

```
CFR = (# of prediction flips) / (# of pairs)
```

**Ideal value:** 0.0% (no flips)  
**Interpretation:** Higher CFR = more bias sensitivity

---

## Phase 3: Representation Extraction (FP32)

### What We Extract

**Hidden states from all 13 BERT layers** for 1000 test samples

**Why:** To analyze how quantization affects internal representations

**Layers:**
- Layer 0: Token embeddings
- Layers 1-12: Transformer block outputs

**For each sample:**
- Extract [CLS] token embedding (768-dim)
- Store across all layers
- Save as numpy arrays for later CKA analysis

**Storage:**
```
results/jigsaw_hidden_fp32.npy     # Shape: (13, 1000, 768)
results/bios_hidden_fp32.npy       # Shape: (13, 1000, 768)
```

---

## Phase 4: Quantization (FP16)

### What We Quantize

**Quantize:** BERT encoder ONLY  
**Keep FP32:** Classifier head

### Process

1. **Load trained FP32 model** (with saved classifier weights)
2. **Convert BERT to FP16:** `model.bert.half()`
3. **Classifier stays FP32** (no conversion)

### Why FP16 First?

To map the **trajectory of degradation**:
- Is bias amplification linear or non-linear?
- Does FP16 show intermediate effects?
- Helps understand INT8 results in context

---

## Phase 5: Fairness Re-Evaluation (FP16)

### What We Calculate

**Same metrics as Phase 2, but with FP16 encoder:**

1. **Demographic Parity Difference (DPD)** - Jigsaw
2. **Equal Opportunity Difference (EOD)** - Bias in Bios  
3. **Counterfactual Flip Rate (CFR)** - Equity Evaluation Corpus

### Comparison

```
Fairness Deviation (FP16) = Metric(FP16) - Metric(FP32)
```

**Questions answered:**
- Did DPD increase? (worse fairness)
- Did EOD widen? (larger performance gap)
- Did CFR increase? (more unstable predictions)

---

## Phase 6: Representation Extraction (FP16)

### What We Extract

**Same 1000 samples, now through FP16 encoder**

- Extract hidden states from all 13 layers
- Store for comparison with FP32

**Storage:**
```
results/jigsaw_hidden_fp16.npy
results/bios_hidden_fp16.npy
```

---

## Phase 7: Quantization (INT8)

### What We Quantize

**Quantize:** BERT encoder to 8-bit integers  
**Keep FP32:** Classifier head

### Process

**Dynamic Range Quantization:**
```python
torch.quantization.quantize_dynamic(
    model.bert,
    {torch.nn.Linear},  # All linear layers
    dtype=torch.qint8
)
```

**What happens:**
- Weights stored as INT8 permanently
- Activations dynamically quantized during forward pass
- ~4x memory reduction
- 1.8-2.5x inference speedup

---

## Phase 8: Fairness Re-Evaluation (INT8)

### What We Calculate

**Complete fairness analysis with INT8 encoder:**

1. **DPD** - Demographic Parity Difference
2. **EOD** - Equal Opportunity Difference  
3. **CFR** - Counterfactual Flip Rate

### Critical Questions

**Has quantization amplified bias?**
```
ΔDPD = DPD(INT8) - DPD(FP32)
ΔEOD = EOD(INT8) - EOD(FP32)
ΔCFR = CFR(INT8) - CFR(FP32)
```

**Which groups are affected most?**
- Are minority demographics disproportionately impacted?
- Which occupations show largest gender gaps?

---

## Phase 9: Representation Analysis (CKA)

### What We Analyze

**Layer-wise comparison of FP32 vs FP16 vs INT8 representations**

### Metrics Computed

#### 1. L2 Distance
```
L2(layer) = mean(||h_fp32 - h_quant||₂)
```
**Interpretation:** Physical displacement in latent space

#### 2. Cosine Similarity
```
Cosine(layer) = mean(cos(h_fp32, h_quant))
```
**Interpretation:** Directional alignment (semantic preservation)

#### 3. Centered Kernel Alignment (CKA)
```
CKA(layer) = HSIC(H_fp32, H_quant) / sqrt(HSIC(H_fp32, H_fp32) × HSIC(H_quant, H_quant))
```
**Interpretation:** Structural similarity (1.0 = perfect, 0.0 = orthogonal)

### Why CKA is Critical

- **Invariant** to orthogonal transformations
- **Invariant** to isotropic scaling
- Measures **entire layer structure**, not just individual vectors
- Gold standard for representation comparison

### Analysis Per Layer

For each of 12 transformer layers:
```
Layer 1:  L2=X.XX, Cosine=0.XX, CKA=0.XX
Layer 2:  L2=X.XX, Cosine=0.XX, CKA=0.XX
...
Layer 12: L2=X.XX, Cosine=0.XX, CKA=0.XX
```

**Expected pattern:**
- Early layers: Small deviations
- Deep layers: Larger deviations (error accumulation)
- Correlation: CKA collapse ↔ Fairness degradation

---

## Phase 10: Correlation Analysis

### What We Correlate

**Representation collapse** (CKA scores) **with** **fairness degradation** (DPD, EOD, CFR)

### Key Hypothesis

> "Severe CKA degradation in final encoder layers forces the classifier to rely on demographic proxy features, causing measurable fairness deterioration"

### Evidence to Establish

1. **Layer-wise CKA decay** (deeper layers = worse)
2. **Fairness metric spikes** in INT8 vs FP32
3. **Statistical correlation** between the two

### Final Report Synthesis

**Proves:**
- Where quantization damage occurs (which layers)
- How it manifests as bias (which fairness metrics)
- Why it happens (representational collapse)

---

## Timeline Summary

| Day | Phase | What Happens |
|-----|-------|--------------|
| **1** | Data Prep | Download datasets, preprocess, create dataloaders |
| **2** | Baseline Training | Train FP32 models, evaluate fairness, extract hidden states |
| **3** | Quantization | Apply FP16/INT8, re-evaluate fairness |
| **4** | Analysis | Compute CKA, correlate with fairness, generate report |

---

## Output Files

### Models
```
models/jigsaw_fp32.pt          # Trained FP32 model (Jigsaw)
models/bios_fp32.pt            # Trained FP32 model (Bios)
```

### Hidden States
```
results/jigsaw_hidden_fp32.npy  # FP32 representations
results/jigsaw_hidden_fp16.npy  # FP16 representations
results/jigsaw_hidden_int8.npy  # INT8 representations
results/bios_hidden_fp32.npy
results/bios_hidden_fp16.npy
results/bios_hidden_int8.npy
```

### Metrics
```
results/fairness_analysis_results.json  # All fairness metrics
results/cka_analysis.json               # Representation analysis
```

### Structure of Results JSON
```json
{
  "jigsaw": {
    "fp32": {"dpd": 0.05, "eod": 0.03, "accuracy": 0.92},
    "fp16": {"dpd": 0.06, "eod": 0.04, "accuracy": 0.91},
    "int8": {"dpd": 0.12, "eod": 0.09, "accuracy": 0.88}
  },
  "bias_in_bios": {
    "fp32": {"eod": 0.08, "accuracy": 0.75},
    "fp16": {"eod": 0.10, "accuracy": 0.74},
    "int8": {"eod": 0.18, "accuracy": 0.70}
  },
  "counterfactual": {
    "fp32": {"cfr": 0.03},
    "fp16": {"cfr": 0.05},
    "int8": {"cfr": 0.14}
  },
  "representation_analysis": {
    "int8_vs_fp32": {
      "l2_distance": [0.5, 0.6, ..., 1.2],
      "cosine_similarity": [0.99, 0.98, ..., 0.92],
      "cka_score": [0.98, 0.95, ..., 0.78]
    }
  }
}
```

---

## Multi-GPU Strategy

### DataParallel Usage

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

**What this does:**
- Automatically splits batches across GPUs
- Replicates model on each GPU
- Synchronizes gradients after backward pass

### Effective Batch Size

With 4 GPUs:
```
actual_batch_size = per_gpu_batch_size × 4
```

**Example:**
- Per-GPU: 32 samples
- Total: 128 samples per iteration
- Faster training, same results

### Memory Distribution

- **BERT encoder (frozen):** Replicated on all GPUs (read-only)
- **Classifier (trainable):** Replicated with gradient sync
- **Each GPU:** Processes subset of batch independently

---

## Key Design Principles

### 1. Isolation of Variables
- **Only variable:** Quantization precision (FP32 → FP16 → INT8)
- **Constant:** BERT weights, classifier weights, evaluation data
- **Result:** Pure measurement of quantization effects

### 2. Frozen Encoder Rationale
- **Prevents:** Fine-tuning artifacts
- **Enables:** Direct attribution to quantization
- **Ensures:** Reproducibility

### 3. Classifier Always FP32
- **Why:** Isolate quantization to encoder only
- **Result:** Any observed bias is from encoder quantization
- **Not from:** Classifier numerical instability

### 4. Fixed Random Seeds
- **For:** Reproducible train/val splits
- **For:** Consistent hidden state extraction (same 1000 samples)
- **For:** Fair comparison across precision levels

---

## Scientific Contributions

This pipeline enables answering:

1. **Does post-training quantization amplify algorithmic bias?**
   - Measured by ΔDPD, ΔEOD, ΔCFR

2. **Which layers are most vulnerable to quantization?**
   - Identified via layer-wise CKA analysis

3. **Can we predict fairness degradation from representation collapse?**
   - Correlation analysis between CKA and fairness metrics

4. **Is the effect linear or exponential?**
   - FP16 intermediate baseline reveals trajectory

5. **Which demographic groups are most affected?**
   - Per-group fairness decomposition

---

## Next Steps After Pipeline Completion

1. **Visualizations:** Plot CKA scores, fairness metrics across precisions
2. **Statistical Tests:** Significance testing for metric differences
3. **Report Writing:** Synthesize findings with theoretical grounding
4. **Model Cards:** Document bias profiles for each precision level
