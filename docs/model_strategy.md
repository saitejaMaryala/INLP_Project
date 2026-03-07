# Model Selection Strategy for Quantization-Fairness Study

## ✅ RECOMMENDED APPROACH

### Base Encoder (Frozen)
**Model:** `bert-base-uncased`
- **Source:** HuggingFace Transformers
- **Status:** Pre-trained, frozen during your training
- **Why:** Exactly what your protocol specifies

```python
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT (this is YOUR frozen encoder)
encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Freeze all parameters
for param in encoder.parameters():
    param.requires_grad = False
```

### Classifier Heads (Trainable)
You MUST train these yourself for:

1. **Jigsaw Toxicity Classification**
   - Task: Binary classification (toxic vs non-toxic)
   - Input: 768-dim [CLS] embedding
   - Output: 2 classes
   - Dataset: `google/jigsaw_unintended_bias`

2. **Bias in Bios Occupation**
   - Task: Multi-class classification (28 occupations)
   - Input: 768-dim [CLS] embedding
   - Output: 28 classes
   - Dataset: `LabHC/bias_in_bios`

```python
import torch.nn as nn

class FrozenBertClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', 
                                               output_hidden_states=True)
        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Trainable classifier (ONLY this trains)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask)
        
        # Extract [CLS] token from last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Pass through trainable classifier
        x = self.dropout(cls_embedding)
        logits = self.classifier(x)
        
        return logits, outputs.hidden_states  # For CKA analysis
```

## ❌ WHAT NOT TO USE

### 1. Fully Fine-Tuned Models
**Examples:**
- `unitary/toxic-bert`
- `martin-ha/toxic-comment-model`
- Any model fine-tuned on these tasks

**❌ Why NOT:**
- Encoder weights have been updated
- Confounds quantization effects with fine-tuning
- Violates experimental design

### 2. Other Encoder Architectures
**❌ Don't use:**
- RoBERTa, ALBERT, DistilBERT (different architectures)
- GPT models (decoder-only)
- T5/BART (encoder-decoder)

**Why:** 
- Your protocol specifically requires `bert-base-uncased`
- Different architectures have different quantization behaviors
- Results wouldn't be comparable to existing literature

## ⚡ TRAINING EFFICIENCY TIPS

### Why Your Training Will Be Fast

1. **Small Parameter Count**
   - BERT encoder: 110M params (frozen, not trained)
   - Classifier head: ~1.5M params (binomial) or ~21K params (28-class)
   - **Only training 1-2% of total parameters!**

2. **Higher Learning Rate Possible**
   ```python
   optimizer = torch.optim.AdamW(
       model.classifier.parameters(),  # Only classifier params
       lr=1e-3,  # Much higher than fine-tuning (1e-5)
       weight_decay=0.01
   )
   ```

3. **Faster Convergence**
   - No catastrophic forgetting risk
   - Simpler optimization surface
   - 5-10 epochs typical

### Estimated Training Time

| Dataset | Samples | Epochs | Hardware | Est. Time |
|---------|---------|--------|----------|-----------|
| Jigsaw (subset) | 100K | 5 | T4 GPU | 30-60 min |
| Bias in Bios | 393K | 10 | T4 GPU | 2-3 hours |

**Total Day 2 training: ~3-4 hours** (well within one day)

## 🎯 COULD YOU USE SHORTCUTS? (Advanced)

### Option: Find a Frozen-Encoder Pre-trained Model

**Theoretical possibility:**
- Someone trained this exact architecture before
- Model available on HuggingFace Hub

**Reality check:**
- ⚠️ Extremely unlikely to exist
- Would need:
  - Exact frozen BERT base uncased
  - Trained on Jigsaw toxicity
  - Trained on Bias in Bios occupations
  - Public availability
  - Documented training procedure

**Verdict:** Don't count on this existing

### Option: Use as Baseline Comparison Only

You COULD use fully fine-tuned models for:
✅ Performance comparison
✅ Sanity checking your results
✅ Literature comparison

But NOT as your primary experimental model.

## 📋 IMPLEMENTATION CHECKLIST

- [ ] Load `bert-base-uncased` from HuggingFace
- [ ] Freeze all encoder parameters (`requires_grad=False`)
- [ ] Create classifier head (Dropout + Linear)
- [ ] Train on Jigsaw dataset (5-10 epochs)
- [ ] Train on Bias in Bios dataset (5-10 epochs)
- [ ] Save classifier weights (FP32 baseline)
- [ ] Extract 1000-sample hidden states for CKA
- [ ] Apply quantization to **frozen encoder only**
- [ ] Run CKA analysis comparing FP32 vs INT8/FP16

## 🔬 WHY THIS EXPERIMENTAL DESIGN MATTERS

From your project plan:
> "When a transformer model is fully fine-tuned for a specific downstream classification task, the backpropagation of errors alters the geometric distribution of the embeddings in the latent space across all transformer layers. This parameter updating process inherently confounds the measurement of quantization-induced noise, making it impossible to ascertain whether an observed spike in demographic bias is the result of the fine-tuning optimization surface or the reduced bit-width."

**The frozen encoder approach ensures:**
1. Pure, uncontaminated BERT representations
2. Quantization is the ONLY variable changed
3. Fairness degradation is ONLY from quantization
4. Results are scientifically valid and publishable

## 🎓 FINAL RECOMMENDATION

**Use:** `bert-base-uncased` (frozen) + Train your own classifier heads

**Why:**
- ✅ Scientifically sound
- ✅ Fast to train (3-4 hours total)
- ✅ Meets protocol requirements
- ✅ Publishable results
- ✅ Complete experimental control

**Don't:**
- ❌ Use fully fine-tuned models as primary experiment
- ❌ Skip training (compromises entire study)
- ❌ Use different encoder architectures
