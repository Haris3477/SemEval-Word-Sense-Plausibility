# SemEval 2026 Task 5: Notebook Guide

## 📋 Overview

This notebook implements an expert-level solution for **SemEval 2026 Task 5: Narrative Plausibility Prediction** using state-of-the-art techniques from the strategic plan document.

### Task Description
Predict human-perceived plausibility ratings (1.0 to 5.0) for word senses within narrative contexts.

### Performance Goals
- **Spearman Correlation (ρ)**: ≥ 0.80
- **Accuracy Within Standard Deviation**: ≥ 0.80

---

## 🏗️ Architecture

### Core Components

1. **Base Model**: DeBERTa-v3-large (304M parameters)
   - State-of-the-art for Natural Language Understanding
   - Disentangled attention mechanism
   - Enhanced mask decoder

2. **Parameter-Efficient Fine-Tuning**: LoRA (Low-Rank Adaptation)
   - Reduces trainable parameters from 304M to ~3-6M
   - Enables training on limited VRAM (8GB)
   - Acts as regularization for small datasets

3. **Ordinal Regression**: CORAL (COnsistent RAnk Logits)
   - Treats ratings as ordinal (1<2<3<4<5)
   - Transforms 5-class problem into 4 binary tasks
   - Optimizes for rank consistency (Spearman correlation)

4. **Training Strategy**:
   - Early stopping (patience=3) to prevent overfitting
   - Gradient accumulation for larger effective batch sizes
   - Low learning rate (6e-6) for stable fine-tuning
   - Comprehensive monitoring of train/val gaps

---

## 📂 Notebook Structure (17 Sections)

### Setup & Data (Sections 1-4)
1. **Environment Setup**: Import libraries, check GPU availability
2. **Configuration**: Hyperparameters and two-phase strategy
3. **Data Loading**: Load train.json and dev.json, visualize distributions
4. **Input Formatting**: Hierarchical context encoding (word → sentence → document)

### Model Architecture (Sections 5-8)
5. **CORAL Theory**: Explanation of ordinal regression framework
6. **Model Definition**: DeBERTa + LoRA + CORAL head
7. **Training Functions**: train_one_epoch, evaluate_model, metrics calculation
8. **Model Initialization**: Load model, apply LoRA, count parameters

### Training & Evaluation (Sections 9-15)
9. **DataLoaders**: Create PyTorch dataloaders
10. **Optimizer & Scheduler**: AdamW with linear warmup
11. **Training Loop**: Main training with early stopping
12. **Visualization**: Training/validation curves (loss, Spearman, ACC w/ SD)
13. **Final Evaluation**: Load best model, comprehensive metrics
14. **Prediction Analysis**: Scatter plots, error distribution
15. **Error Analysis**: Best/worst predictions, error by rating level

### Advanced Strategies (Sections 16-17)
16. **Phase 2 Improvements**: Data augmentation, ensemble methods
17. **Conclusion**: Summary and recommendations

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers peft scipy scikit-learn pandas numpy matplotlib seaborn tqdm
```

### Run the Notebook

1. Ensure `train.json` and `dev.json` are in the same directory
2. Open `semeval_2026_task5.ipynb`
3. Run cells in sequence (Shift+Enter)
4. Training will automatically:
   - Track train/val metrics
   - Apply early stopping
   - Save best model to `outputs/best_deberta_coral.pt`
   - Generate visualization plots

### Expected Runtime
- **RTX 4060 (8GB)**: ~2-3 hours for 5 epochs
- **RTX 4080**: ~1-1.5 hours for 5 epochs

---

## 🔄 Switching to RTX 4080 Setup

When you upgrade to RTX 4080, you can unlock additional capabilities and faster training.

### Configuration Changes

In **Section 2 (Configuration)**, modify these settings:

```python
config = {
    # ===== MODEL CONFIGURATION =====
    'model_name': 'microsoft/deberta-v3-large',
    'max_length': 512,
    'pooling': 'cls',
    
    # ===== LoRA CONFIGURATION (PEFT) =====
    'use_lora': True,
    'lora_r': 16,  # ✅ CHANGED: 8 → 16 (more capacity)
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'lora_target_modules': ['query_proj', 'key_proj', 'value_proj', 'dense'],
    
    # ===== ORDINAL REGRESSION =====
    'loss_type': 'coral',
    'num_classes': 5,
    'ordinal_weight': 1.0,
    
    # ===== TRAINING HYPERPARAMETERS =====
    'learning_rate': 6e-6,
    'epochs': 8,  # ✅ CHANGED: 5 → 8 (more training)
    'batch_size': 8,  # ✅ CHANGED: 4 → 8 (larger batches)
    'grad_accumulation_steps': 2,  # ✅ CHANGED: 4 → 2 (effective batch still 16)
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'scheduler_type': 'linear',
    
    # ===== REGULARIZATION =====
    'dropout': 0.3,
    'label_smoothing': 0.0,
    'gradient_clip': 1.0,
    
    # ===== EARLY STOPPING =====
    'early_stop_patience': 4,  # ✅ CHANGED: 3 → 4 (allow more exploration)
    'early_stop_metric': 'spearman',
    
    # ===== DATA PATHS =====
    'train_path': 'train.json',
    'dev_path': 'dev.json',
    
    # ===== OUTPUT =====
    'save_dir': 'outputs',
    'model_save_name': 'best_deberta_coral_4080.pt',  # ✅ CHANGED: different name
    'seed': 42,
}
```

### Summary of Changes for RTX 4080

| Parameter | RTX 4060 (8GB) | RTX 4080 (16GB) | Reason |
|-----------|----------------|-----------------|--------|
| `lora_r` | 8 | 16 | More trainable params (3M → 6M) for better capacity |
| `batch_size` | 4 | 8 | Larger batches for more stable gradients |
| `grad_accumulation_steps` | 4 | 2 | Keep effective batch size at 16 |
| `epochs` | 5 | 8 | More training iterations possible |
| `early_stop_patience` | 3 | 4 | Allow more exploration before stopping |

### Benefits of RTX 4080 Setup

1. **Faster Training**: 2x faster due to larger batch sizes
2. **Better Performance**: More capacity (r=16) can learn complex patterns
3. **Ensemble Training**: Can train multiple models in parallel
4. **Data Augmentation**: Can run LLMs for synthetic data generation

---

## 📊 Interpreting Results

### Training Curves (Section 12)

**1. Train/Val Loss Plot**
- Both should decrease
- If validation loss increases → overfitting
- Gap between train/val indicates overfitting severity

**2. Train/Val Spearman Plot**
- Target: Val Spearman ≥ 0.80
- **Overfitting Gap** = Train Spearman - Val Spearman
  - Gap < 0.08: Excellent generalization ✅
  - Gap 0.08-0.15: Moderate overfitting ⚠️
  - Gap > 0.15: High overfitting 🔥

**3. ACC Within SD Plot**
- Shows prediction precision
- Target: ≥ 0.80
- Should improve over epochs

### Prediction Analysis (Section 14)

**1. Scatter Plot (Predicted vs Actual)**
- Points near diagonal = good predictions
- Points colored by human stdev (red = high disagreement)

**2. Error Distribution**
- Centered at 0 = unbiased predictions
- Narrow distribution = precise predictions

**3. Error by Rating Level**
- Shows if model struggles with specific rating ranges
- Useful for targeted improvements

**4. Within SD Pie Chart**
- Visual representation of ACC within SD metric

---

## 🎯 Expected Performance

### Baseline (Phase 1 - RTX 4060)
- **Validation Spearman**: 0.65 - 0.75
- **ACC Within SD**: 0.70 - 0.80
- **Training Time**: 2-3 hours

### Improved (Phase 1 - RTX 4080)
- **Validation Spearman**: 0.70 - 0.80
- **ACC Within SD**: 0.75 - 0.85
- **Training Time**: 1-1.5 hours

### Advanced (Phase 2 - With Augmentation & Ensemble)
- **Validation Spearman**: 0.80 - 0.85+
- **ACC Within SD**: 0.80 - 0.90+
- **Training Time**: 4-6 hours (multiple models)

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution for RTX 4060:**
```python
config['batch_size'] = 2  # Reduce from 4
config['grad_accumulation_steps'] = 8  # Increase to maintain effective batch
```

**Solution for RTX 4080:**
```python
config['batch_size'] = 4  # Reduce from 8
config['grad_accumulation_steps'] = 4  # Adjust accordingly
```

### Issue: Severe Overfitting (Gap > 0.3)

**Solutions:**
1. Increase dropout: `config['dropout'] = 0.4`
2. Reduce LoRA rank: `config['lora_r'] = 4`
3. More aggressive early stopping: `config['early_stop_patience'] = 2`
4. Add label smoothing: `config['label_smoothing'] = 0.1`

### Issue: Val Spearman Not Improving

**Solutions:**
1. Increase LoRA rank: `config['lora_r'] = 16` (RTX 4080)
2. More epochs: `config['epochs'] = 10`
3. Try different pooling: `config['pooling'] = 'mean'`
4. Ensemble multiple models (Section 16)

### Issue: Training Too Slow

**Solutions:**
1. Enable multi-worker dataloading:
   ```python
   train_loader = DataLoader(..., num_workers=4)
   dev_loader = DataLoader(..., num_workers=4)
   ```
2. Reduce max_length: `config['max_length'] = 384`
3. Use mixed precision training (requires code modification)

---

## 📈 Phase 2 Improvements

If you don't hit the targets (Spearman ≥ 0.80) with Phase 1, try these:

### 1. Increase Model Capacity (RTX 4080 Required)

```python
config['lora_r'] = 32  # Even more capacity
config['lora_alpha'] = 64
```

### 2. Data Augmentation

**Paraphrasing** (maintain labels):
- Use back-translation (English → German → English)
- Use T5 paraphrasing models
- Target: +500-1000 augmented samples

**Counterfactual Generation** (flip labels):
- Use LLMs (GPT, Gemini) to generate plausibility-flipped examples
- High rating (4-5) → Low rating (1-2) by changing context
- Requires careful quality control

### 3. Ensemble Methods

Train 3-5 models with:
- Different random seeds (42, 123, 456, 789, 999)
- Different loss functions (CORAL, CORN, MSE)
- Different LoRA ranks (8, 16, 32)

**Simple Ensemble:**
```python
final_prediction = (model1_pred + model2_pred + model3_pred) / 3
```

Expected gain: +0.02 to +0.05 Spearman

### 4. Output Calibration

If predictions are systematically biased:
```python
from sklearn.isotonic import IsotonicRegression

# Fit on validation set
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(val_predictions, val_targets)

# Apply to test predictions
calibrated_predictions = calibrator.predict(test_predictions)
```

---

## 💾 Output Files

After running the notebook, you'll have:

| File | Description |
|------|-------------|
| `outputs/best_deberta_coral.pt` | Best model checkpoint |
| `data_exploration.png` | Rating & stdev distributions (4 plots) |
| `training_curves_coral.png` | Training curves with overfitting analysis (4 plots) |
| `prediction_analysis.png` | Error analysis & scatter plots (4 plots) |
| `dev_predictions.csv` | Detailed predictions for each dev sample |

---

## 📚 Key Concepts

### What is CORAL?

**Traditional Approach**: Treat ratings 1, 2, 3, 4, 5 as independent classes
- Problem: Model doesn't know 1 < 2 < 3 < 4 < 5
- Predicting 1 instead of 5 penalized same as predicting 4 instead of 5

**CORAL Approach**: Convert to 4 binary classification tasks
- Task 1: Is rating > 1? (Yes for 2, 3, 4, 5)
- Task 2: Is rating > 2? (Yes for 3, 4, 5)
- Task 3: Is rating > 3? (Yes for 4, 5)
- Task 4: Is rating > 4? (Yes for 5)

**Benefits**:
- Enforces ordinal structure
- Better optimizes for Spearman correlation (rank-based metric)
- More suitable for human rating prediction

### What is LoRA?

**Problem**: Fine-tuning 304M parameters on 2,280 samples = severe overfitting + high VRAM

**LoRA Solution**: Only train small low-rank matrices
- Injects trainable rank-r matrices into frozen pretrained weights
- r=8: ~3M trainable params (1% of total)
- r=16: ~6M trainable params (2% of total)

**Benefits**:
- Drastically reduces VRAM usage
- Acts as regularization (prevents overfitting on small data)
- Maintains most of pretrained knowledge
- Can be merged back into base model for deployment

---

## 🎓 Strategic Decisions Explained

### Why DeBERTa-v3-large?

1. **State-of-the-art NLU performance**: Best on GLUE, SQuAD 2.0
2. **Disentangled attention**: Better handles complex context
3. **Enhanced mask decoder**: Improved token representations
4. **Proven for narrative tasks**: Excellent at multi-sentence understanding

### Why Spearman Correlation?

- Task evaluates **ranking accuracy**, not absolute values
- If gold ratings are [1.2, 3.5, 4.8] and predictions are [1.0, 3.3, 4.9]
  - Spearman ρ ≈ 1.0 (perfect ranking!) ✅
  - MSE = 0.05 (decent but misleading)
- CORAL optimizes for rank consistency → high Spearman

### Why Early Stopping?

- Dataset: only 2,280 samples
- DeBERTa: 304M parameters (even with LoRA: 3-6M trainable)
- Risk: Model memorizes training data after few epochs
- Solution: Stop when validation Spearman stops improving

---

## 📖 References

1. **Strategic Plan**: "Expert Strategic Blueprint for SemEval 2026 Task 5"
2. **CORAL Paper**: Cao et al. "Rank Consistent Ordinal Regression for Neural Networks"
3. **LoRA Paper**: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
4. **DeBERTa Paper**: He et al. "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training" (2021)
5. **Ordinal Regression**: Gutierrez et al. "Ordinal Regression Methods: Survey and Experimental Study" (2016)

---

## 🤝 Support & Contribution

### Need Help?

1. Check **Troubleshooting** section above
2. Review **Section 16 (Phase 2 Strategy)** in notebook
3. Examine `dev_predictions.csv` for error patterns

### Next Steps

1. **Hit Targets?** → Submit to SemEval 2026!
2. **Close but Not There?** → Try RTX 4080 config + ensemble
3. **Far from Targets?** → Follow Phase 2 recommendations

### Iteration Strategy

```
1. Run Phase 1 (this notebook) → Get baseline
2. If Spearman < 0.75 → Switch to RTX 4080 config
3. If Spearman 0.75-0.80 → Train 3 models, ensemble
4. If still < 0.80 → Add data augmentation
5. If > 0.80 → Celebrate! 🎉
```

---

## 📝 License & Citation

If you use this notebook or achieve good results, please cite:

```
SemEval 2026 Task 5: Narrative Plausibility Prediction
Implementation based on "Expert Strategic Blueprint for SemEval 2026 Task 5"
Uses: DeBERTa-v3-large, LoRA, CORAL ordinal regression
```

---

**Good luck with SemEval 2026 Task 5!** 🚀




