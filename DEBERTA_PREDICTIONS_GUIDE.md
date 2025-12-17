# DeBERTa Predictions Guide

This guide explains how to generate predictions using the DeBERTa-v3-large model with CORAL loss and LoRA fine-tuning.

## Prerequisites

1. **Model Checkpoint**: Ensure you have the trained DeBERTa model checkpoint:
   - `outputs/best_deberta_coral.pt` (should be ~1.7GB)

2. **Dependencies**: Make sure you have the required packages:
   ```bash
   pip install torch transformers peft pandas numpy tqdm scipy scikit-learn
   ```

3. **Test Data**: Have your test data file ready (e.g., `data/dev.json`)

## Running Predictions

### Basic Command

```bash
python3 generate_predictions.py \
    --test_path data/dev.json \
    --model_path outputs/best_deberta_coral.pt \
    --model_name microsoft/deberta-v3-large \
    --output_file predictions_deberta.jsonl \
    --batch_size 2 \
    --pooling cls
```

### Parameters Explained

- `--test_path`: Path to your test JSON file (format: `{"id": {...}}`)
- `--model_path`: Path to the trained DeBERTa checkpoint (`outputs/best_deberta_coral.pt`)
- `--model_name`: HuggingFace model name (`microsoft/deberta-v3-large`)
- `--output_file`: Output predictions file name (`predictions_deberta.jsonl`)
- `--batch_size`: Batch size for inference (default: 16, use 2 for DeBERTa-large to avoid OOM)
- `--pooling`: Pooling method (`cls`, `mean`, or `weighted` - use `cls` for DeBERTa as per training config)
- `--max_length`: Maximum sequence length (default: 512)
- `--dropout`: Dropout rate (default: 0.3, should match training)

### Memory Considerations

- **DeBERTa-v3-large** is a large model (~1.3B parameters)
- Use `--batch_size 2` on Apple Silicon (MPS) or systems with limited GPU memory
- On systems with more memory, you can increase batch size (e.g., `--batch_size 4` or `--batch_size 8`)

## What the Script Does

1. **Loads the checkpoint** and detects:
   - CORAL model (checks for `coral_bias` in state dict)
   - LoRA configuration (checks for `lora` keys)
   - Model configuration from checkpoint metadata

2. **Applies LoRA** if detected:
   - Uses PEFT library to wrap the base model
   - Loads LoRA configuration from checkpoint (`r=8`, `alpha=32`, `dropout=0.1`)
   - Target modules: `['query_proj', 'key_proj', 'value_proj', 'dense']`

3. **Generates predictions**:
   - Processes test data in batches
   - Converts CORAL logits to ratings (1-5) using `coral_predict()` function
   - Outputs predictions in JSONL format

## Output Format

The script generates a `predictions.jsonl` file with one prediction per line:

```jsonl
{"id": "0", "prediction": 4}
{"id": "1", "prediction": 4}
{"id": "2", "prediction": 3}
...
```

Each prediction is an integer between 1 and 5.

## Creating Submission Zip

**Important**: The competition requires the file inside the zip to be named `predictions.jsonl` (not `predictions_deberta.jsonl`).

### Option 1: Rename and zip

```bash
# Copy the file with the correct name
cp predictions_deberta.jsonl predictions.jsonl

# Create zip file
zip predictions_deberta.zip predictions.jsonl
```

### Option 2: Direct zip with correct name

```bash
# Create zip with correct internal filename
cd /tmp
cp /path/to/predictions_deberta.jsonl predictions.jsonl
zip -j predictions_deberta.zip predictions.jsonl
mv predictions_deberta.zip /path/to/project/
```

### Verify Zip Contents

```bash
unzip -l predictions_deberta.zip
```

Should show:
```
Archive:  predictions_deberta.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
    18118  ...        predictions.jsonl
```

## Model Details

### Architecture
- **Base Model**: `microsoft/deberta-v3-large`
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
  - Rank (r): 8
  - Alpha: 32
  - Dropout: 0.1
- **Loss Function**: Hybrid (CORAL + MSE)
- **Pooling**: CLS token
- **Output**: CORAL ordinal regression (4 logits → ratings 1-5)

### Training Configuration
- Batch size: 2
- Gradient accumulation: 8 steps
- Learning rate: 8e-5
- Max length: 448 tokens
- Dropout: 0.3

## Troubleshooting

### Error: "CORAL model requires test3.py"
- **Solution**: The script now includes the CORAL model class directly. If you see this error, check that `generate_predictions.py` has the `CORALPlausibilityModel` class defined.

### Error: "MPS backend out of memory"
- **Solution**: Reduce batch size:
  ```bash
  --batch_size 1  # or even smaller
  ```

### Error: "PEFT not available"
- **Solution**: Install PEFT:
  ```bash
  pip install peft
  ```

### Error: "Not a valid predictions filepath"
- **Solution**: Ensure the zip file contains `predictions.jsonl` (not `predictions_deberta.jsonl`). See "Creating Submission Zip" section above.

### All predictions are 3
- **Cause**: Model not loading correctly or wrong model architecture
- **Solution**: 
  - Verify checkpoint has `coral_bias` key
  - Check that LoRA is being applied correctly
  - Ensure model outputs shape `(batch, 4)` for CORAL logits

## Example Output

```
Using device: mps (Apple Silicon GPU) 🚀

📂 Loading test data from: data/dev.json
   ✓ Loaded 588 test samples

🔤 Loading tokenizer: microsoft/deberta-v3-large
   ✓ Tokenizer loaded

📝 Creating text inputs...
   ✓ Created 588 text inputs

🤖 Loading model from: outputs/best_deberta_coral.pt
   ✓ Detected CORAL model (LoRA: True)
   ⚡ Applying LoRA (r=8, alpha=32)...
   ✓ LoRA applied
   ✓ Model loaded from checkpoint (model_state_dict)

🔮 Generating predictions...
   ✓ Detected CORAL model (output shape: torch.Size([2, 4]))
Predicting: 100%|██████████| 294/294 [04:20<00:00,  1.13it/s]

💾 Writing predictions to: predictions_deberta.jsonl

✅ Done! Generated 588 predictions
   File: predictions_deberta.jsonl
   Prediction range: 1 - 5
   Mean prediction: 3.33

📦 To submit: zip predictions_deberta.jsonl and upload to CodaBench
```

## Quick Reference

```bash
# Full command for DeBERTa predictions
python3 generate_predictions.py \
    --test_path data/dev.json \
    --model_path outputs/best_deberta_coral.pt \
    --model_name microsoft/deberta-v3-large \
    --output_file predictions_deberta.jsonl \
    --batch_size 2 \
    --pooling cls

# Create submission zip
cp predictions_deberta.jsonl predictions.jsonl
zip predictions_deberta.zip predictions.jsonl

# Verify
unzip -l predictions_deberta.zip
```

## Notes

- The DeBERTa model uses **CORAL ordinal regression**, which outputs 4 logits that are converted to ratings 1-5
- **LoRA** significantly reduces memory usage while maintaining performance
- The model was trained with **hybrid loss** (CORAL + MSE), but inference only uses CORAL predictions
- Use `--pooling cls` to match the training configuration
- The script automatically detects CORAL models and applies the correct conversion

