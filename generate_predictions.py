#!/usr/bin/env python
"""Generate predictions.jsonl file for SemEval 2026 Task 5 competition submission."""

import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from semeval_task5_main.py
from semeval_task5_main import (
    PlausibilityDataset,
    highlight_target,
    create_text_input,
    get_device,
    load_json_dataset,
    standardize_columns,
    PRED_MIN,
    PRED_MAX
)
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F

# Import PEFT for LoRA support
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠️  PEFT not available - LoRA models won't work")

# Try importing from test3.py for CORAL/LoRA models
try:
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("test3", "test3.py")
    test3_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test3_module)
    HAS_CORAL_MODEL = True
except Exception as e:
    HAS_CORAL_MODEL = False
    test3_module = None
    print(f"⚠️  Could not import test3.py: {e}")


def coral_predict(logits):
    """
    Convert CORAL logits to predicted ratings
    
    Args:
        logits: (batch, K-1) - raw CORAL outputs
    
    Returns:
        predictions: (batch,) - ratings in range [1, 5]
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)
    
    # Sum of probabilities + 1 gives the predicted rating
    predictions = 1 + probs.sum(dim=1)
    
    # Clamp to valid range
    predictions = torch.clamp(predictions, min=1.0, max=5.0)
    
    return predictions


class CORALPlausibilityModel(nn.Module):
    """CORAL model with LoRA support"""
    def __init__(self, model_name: str, dropout: float, pooling: str = 'cls', num_classes: int = 5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        self.num_classes = num_classes
        
        # Optional attention scorer for weighted pooling
        if pooling == 'weighted':
            self.attn = nn.Linear(hidden, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        # CORAL head: shared weight + K-1 biases
        self.coral_linear = nn.Linear(hidden, 1, bias=False)
        # Initialize biases to encourage diverse predictions
        initial_biases = torch.linspace(-1.0, 1.0, num_classes - 1)
        self.coral_bias = nn.Parameter(initial_biases)
    
    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        if self.pooling == 'cls':
            pooled = hidden_states[:, 0, :]
        elif self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (hidden_states * mask).sum(dim=1) / denom
        elif self.pooling == 'weighted':
            scores = self.attn(hidden_states).squeeze(-1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            weights = F.softmax(scores, dim=1)
            pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = hidden_states[:, 0, :]
        
        pooled = self.dropout(pooled)
        
        # CORAL logits: shared weight applied to pooled, then add individual biases
        logits = self.coral_linear(pooled)  # (batch, 1)
        logits = logits + self.coral_bias  # (batch, K-1) via broadcasting
        
        return logits  # (batch, num_classes-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions for competition")
    parser.add_argument('--test_path', default='data/dev.json', 
                       help='Path to test JSON file')
    parser.add_argument('--model_path', default='outputs/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_name', default='roberta-base',
                       help='Model name (must match training)')
    parser.add_argument('--output_file', default='predictions.jsonl',
                       help='Output predictions file')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--pooling', choices=['cls', 'mean', 'weighted'], default='weighted')
    parser.add_argument('--mark_homonym', action='store_true', default=True,
                       help='Mark homonym in sentence with [TGT] tags')
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    
    # Load test data
    print(f"\n📂 Loading test data from: {args.test_path}")
    test_df = load_json_dataset(args.test_path, source='test')
    test_df = standardize_columns(test_df)
    print(f"   ✓ Loaded {len(test_df)} test samples")
    
    # Load tokenizer
    print(f"\n🔤 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("   ✓ Tokenizer loaded")
    
    # Create text inputs
    print(f"\n📝 Creating text inputs...")
    texts = []
    for _, row in test_df.iterrows():
        text = create_text_input(row, mark_homonym=args.mark_homonym)
        texts.append(text)
    print(f"   ✓ Created {len(texts)} text inputs")
    
    # Create dataset (dummy scores/weights for test data)
    dummy_scores = [3.0] * len(test_df)  # Neutral score
    dummy_weights = [1.0] * len(test_df)
    test_dataset = PlausibilityDataset(
        texts=texts,
        scores=dummy_scores,
        weights=dummy_weights,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load checkpoint first to detect model type
    if not os.path.exists(args.model_path):
        print(f"   ❌ Model file not found: {args.model_path}")
        print(f"   Available models:")
        if os.path.exists('outputs'):
            for f in os.listdir('outputs'):
                if f.endswith('.pt'):
                    print(f"     - outputs/{f}")
        return
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    config = checkpoint.get('config', {})
    
    # Detect if model uses CORAL (has coral_bias key)
    use_coral_model = 'coral_bias' in state_dict
    has_lora = any('lora' in k for k in state_dict.keys())
    
    print(f"\n🤖 Loading model from: {args.model_path}")
    
    # Get config values
    model_config = {
        'model_name': config.get('model_name', args.model_name),
        'dropout': config.get('dropout', args.dropout),
        'pooling': config.get('pooling', args.pooling),
        'loss_type': config.get('loss_type', 'coral' if use_coral_model else 'mse'),
        'num_classes': config.get('num_classes', 5)
    }
    
    if use_coral_model:
        print(f"   ✓ Detected CORAL model (LoRA: {has_lora})")
        
        # Create base encoder
        base_encoder = AutoModel.from_pretrained(model_config['model_name'])
        
        # Apply LoRA if needed
        if has_lora and HAS_PEFT:
            print(f"   ⚡ Applying LoRA (r={config.get('lora_r', 8)}, alpha={config.get('lora_alpha', 32)})...")
            lora_config = LoraConfig(
                r=config.get('lora_r', 8),
                lora_alpha=config.get('lora_alpha', 32),
                lora_dropout=config.get('lora_dropout', 0.1),
                target_modules=config.get('lora_target_modules', ['query_proj', 'key_proj', 'value_proj', 'dense']),
                bias="none"
            )
            base_encoder = get_peft_model(base_encoder, lora_config)
            print("   ✓ LoRA applied")
        
        # Create CORAL model
        model = CORALPlausibilityModel(
            model_name=model_config['model_name'],
            dropout=model_config['dropout'],
            pooling=model_config['pooling'],
            num_classes=model_config['num_classes']
        )
        
        # Replace encoder with LoRA version if needed
        if has_lora:
            model.encoder = base_encoder
        
        model = model.to(device)
    else:
        from semeval_task5_main import PlausibilityModel
        model = PlausibilityModel(
            model_name=model_config['model_name'],
            dropout=model_config['dropout'],
            pooling=model_config['pooling']
        ).to(device)
    
    # Load checkpoint
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("   ✓ Model loaded from checkpoint (model_state_dict)")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    model.eval()
    
    # Generate predictions
    print(f"\n🔮 Generating predictions...")
    predictions = []
    ids = test_df['id'].tolist()
    
    # Detect if model uses CORAL (check first batch output shape)
    with torch.no_grad():
        # Test batch to detect output format
        test_batch = next(iter(test_loader))
        test_input_ids = test_batch['input_ids'].to(device)
        test_attention_mask = test_batch['attention_mask'].to(device)
        test_outputs = model(test_input_ids, test_attention_mask)
        
        # CORAL outputs have shape (batch, K-1) = (batch, 4)
        # Standard regression outputs have shape (batch,)
        if len(test_outputs.shape) == 2 and test_outputs.shape[1] == 4:
            print(f"   ✓ Detected CORAL model (output shape: {test_outputs.shape})")
            use_coral = True
        else:
            print(f"   ✓ Detected standard regression model (output shape: {test_outputs.shape})")
            use_coral = False
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            
            if use_coral:
                # Convert CORAL logits to predictions
                preds = coral_predict(outputs).cpu().numpy()
            else:
                # Model outputs are already in [1, 5] range (sigmoid scaled by PlausibilityModel)
                preds = outputs.cpu().numpy()
            
            predictions.extend(preds)
    
    # Round to nearest integer (competition requires integer predictions)
    predictions = np.round(predictions).astype(int)
    predictions = np.clip(predictions, 1, 5)  # Ensure valid range
    
    # Create predictions.jsonl file
    print(f"\n💾 Writing predictions to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        for id_val, pred in zip(ids, predictions):
            f.write(json.dumps({"id": str(id_val), "prediction": int(pred)}) + '\n')
    
    print(f"\n✅ Done! Generated {len(predictions)} predictions")
    print(f"   File: {args.output_file}")
    print(f"   Prediction range: {predictions.min()} - {predictions.max()}")
    print(f"   Mean prediction: {predictions.mean():.2f}")
    print(f"\n📦 To submit: zip {args.output_file} and upload to CodaBench")


if __name__ == '__main__':
    main()

