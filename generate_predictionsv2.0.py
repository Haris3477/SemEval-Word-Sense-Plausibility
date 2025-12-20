#!/usr/bin/env python
"""
Generate predictions.jsonl file for SemEval 2026 Task 5 competition submission.

This script matches the semeval_2026_task5.ipynb notebook setup exactly.
"""

import argparse
import json
import os
import re
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List

from transformers import AutoTokenizer, AutoModel

# Import PEFT for LoRA support
try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠️  PEFT not available - LoRA models won't work")


# ===== DATA LOADING (matches notebook cell 6) =====

def load_json_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    rows = []
    for key, item in data.items():
        row = item.copy()
        row['id'] = key
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ensure required columns exist (with defaults for test data)
    if 'average' not in df.columns:
        df['average'] = 3.0  # Neutral score for test data
    if 'stdev' not in df.columns:
        df['stdev'] = 1.0  # Default stdev for test data
    
    return df


# ===== TEXT FORMATTING (matches notebook cell 9) =====

def highlight_target(sentence: str, homonym: str) -> str:
    """Highlight target word with [TGT] tags"""
    if not sentence or '[TGT]' in sentence or not homonym:
        return sentence
    pattern = re.compile(rf"\b{re.escape(homonym)}\b", re.IGNORECASE)
    return pattern.sub(lambda m: f"[TGT]{m.group(0)}[/TGT]", sentence, count=1)


def create_narrative_input(row: pd.Series, mark_homonym: bool = True) -> str:
    """
    Create structured input for DeBERTa following the strategic plan.
    MUST match notebook Cell 9 exactly!
    
    Format: Provides hierarchical context from word-level to document-level
    - Ambiguous word + candidate sense
    - Sense tags (if available)
    - Dictionary example (word-level context)
    - Story context (sentence/paragraph-level)
    - Target sentence (where the word appears)
    - Ending (document-level closure)
    """
    sentence = row['sentence']
    if mark_homonym:
        sentence = highlight_target(sentence, row.get('homonym', ''))
    
    parts = [
        f"Ambiguous word: {row.get('homonym', '').strip()}.",
        f"Candidate sense: {row.get('judged_meaning', '').strip()}."
    ]
    
    # Add sense tags if available
    tags = row.get('sense_tags', '').strip() if row.get('sense_tags') else ''
    if tags:
        parts.append(f"Sense tags: {tags}.")
    
    # Add dictionary example (provides canonical usage)
    example = row.get('example_sentence', '').strip() if row.get('example_sentence') else ''
    if example:
        parts.append(f"Dictionary example: {example}")
    
    # Add story precontext (narrative background)
    precontext = row.get('precontext', '').strip() if row.get('precontext') else ''
    if precontext:
        parts.append(f"Story context: {precontext}")
    
    # Target sentence (most important)
    parts.append(f"Target sentence: {sentence}")
    
    # Add ending (narrative closure)
    ending = row.get('ending', '').strip() if row.get('ending') else ''
    if ending:
        parts.append(f"Ending: {ending}")
    
    return " ".join(parts)


# ===== DATASET CLASS (matches notebook cell 11) =====

class PlausibilityDataset(Dataset):
    """Dataset for narrative plausibility rating"""
    
    def __init__(self, texts: List[str], scores: List[float], stdevs: List[float], 
                 tokenizer, max_length: int):
        self.texts = texts
        self.scores = scores
        self.stdevs = stdevs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'score': torch.tensor(self.scores[idx], dtype=torch.float),
            'stdev': torch.tensor(self.stdevs[idx], dtype=torch.float)
        }


# ===== MODEL CLASS (matches notebook cell 13) =====

class CORALModel(nn.Module):
    """
    DeBERTa-v3-large with CORAL ordinal regression head
    
    Architecture:
    - DeBERTa encoder (with LoRA)
    - Pooling layer (CLS, mean, or weighted attention)
    - CORAL head: K-1 binary classifiers with shared weights + individual biases
    """
    
    def __init__(self, encoder, num_classes=5, pooling='cls', dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.pooling_type = pooling
        self.num_classes = num_classes
        
        hidden_size = encoder.config.hidden_size
        
        # Weighted attention pooling (optional)
        if pooling == 'weighted':
            self.attention_weights = nn.Linear(hidden_size, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # CORAL head: shared weight + K-1 biases
        self.coral_linear = nn.Linear(hidden_size, 1, bias=False)
        initial_biases = torch.linspace(-1.0, 1.0, num_classes - 1)
        self.coral_bias = nn.Parameter(initial_biases)
        
    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        
        # Pooling
        if self.pooling_type == 'cls':
            pooled = hidden_states[:, 0, :]
        elif self.pooling_type == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling_type == 'weighted':
            attn_scores = self.attention_weights(hidden_states).squeeze(-1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, torch.finfo(attn_scores.dtype).min)
            attn_weights = F.softmax(attn_scores, dim=1)
            pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = hidden_states[:, 0, :]
        
        pooled = self.dropout(pooled)
        
        # CORAL logits
        logits = self.coral_linear(pooled)
        logits = logits + self.coral_bias
        
        return logits


def coral_predict(logits):
    """
    Convert CORAL logits to predicted ratings
    
    Args:
        logits: (batch, K-1) - raw CORAL outputs
    
    Returns:
        predictions: (batch,) - ratings in range [1, 5]
    """
    probs = torch.sigmoid(logits)
    predictions = 1 + probs.sum(dim=1)
    predictions = torch.clamp(predictions, min=1.0, max=5.0)
    return predictions


# ===== DEVICE DETECTION =====

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# ===== ARGUMENT PARSING =====

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate predictions for SemEval 2026 Task 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument('--test_path', default='dev.json', 
                       help='Path to test/dev JSON file')
    parser.add_argument('--model_path', default='outputs/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_file', default='predictions.jsonl',
                       help='Output predictions file')
    
    # Model settings (should match training config)
    parser.add_argument('--model_name', default='microsoft/deberta-v3-large',
                       help='Base model name (must match training)')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pooling', choices=['cls', 'mean', 'weighted'], default='weighted')
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--num_classes', type=int, default=5)
    
    # LoRA settings
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    
    # Submission settings
    parser.add_argument('--create_zip', action='store_true', default=True,
                       help='Create submission zip file (default: True)')
    parser.add_argument('--zip_file', default=None,
                       help='Output zip filename (default: {output_file}.zip)')
    
    return parser.parse_args()


# ===== MAIN =====

def main():
    args = parse_args()
    device = get_device()
    print(f"\n🖥️  Using device: {device}")
    
    # ===== Load Test Data =====
    print(f"\n📂 Loading test data from: {args.test_path}")
    if not os.path.exists(args.test_path):
        print(f"   ❌ File not found: {args.test_path}")
        return
    
    test_df = load_json_dataset(args.test_path)
    print(f"   ✓ Loaded {len(test_df)} samples")
    
    # ===== Format Text Inputs =====
    print(f"\n📝 Formatting text inputs...")
    test_df['text'] = test_df.apply(
        lambda row: create_narrative_input(row, mark_homonym=True), 
        axis=1
    )
    print(f"   ✓ Formatted {len(test_df)} samples")
    
    # ===== Load Tokenizer =====
    print(f"\n🔤 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("   ✓ Tokenizer loaded")
    
    # ===== Create Dataset & DataLoader =====
    test_dataset = PlausibilityDataset(
        texts=test_df['text'].tolist(),
        scores=test_df['average'].tolist(),
        stdevs=test_df['stdev'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"   ✓ Created DataLoader with {len(test_loader)} batches")
    
    # ===== Load Model Checkpoint =====
    if not os.path.exists(args.model_path):
        print(f"\n❌ Model file not found: {args.model_path}")
        print(f"   Available models in outputs/:")
        if os.path.exists('outputs'):
            for f in sorted(os.listdir('outputs')):
                if f.endswith('.pt'):
                    print(f"     - outputs/{f}")
        return
    
    print(f"\n🤖 Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Use checkpoint config if available, otherwise use args
    model_name = config.get('model_name', args.model_name)
    pooling = config.get('pooling', args.pooling)
    dropout = config.get('dropout', args.dropout)
    num_classes = config.get('num_classes', args.num_classes)
    lora_r = config.get('lora_r', args.lora_r)
    lora_alpha = config.get('lora_alpha', args.lora_alpha)
    lora_dropout = config.get('lora_dropout', args.lora_dropout)
    use_lora = config.get('use_lora', True)
    
    print(f"   Model: {model_name}")
    print(f"   Pooling: {pooling}")
    print(f"   LoRA: r={lora_r}, alpha={lora_alpha}")
    
    # ===== Build Model =====
    print(f"\n⚙️  Building model architecture...")
    
    # Load base encoder
    base_encoder = AutoModel.from_pretrained(model_name)
    
    # Apply LoRA if used during training
    if use_lora and HAS_PEFT:
        print(f"   ⚡ Applying LoRA adapter...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=config.get('lora_target_modules', 
                                       ['query_proj', 'key_proj', 'value_proj', 'dense']),
            bias="none"
        )
        base_encoder = get_peft_model(base_encoder, lora_config)
        print(f"   ✓ LoRA applied")
    
    # Create CORAL model
    model = CORALModel(
        encoder=base_encoder,
        num_classes=num_classes,
        pooling=pooling,
        dropout=dropout
    )
    
    # Load weights with debugging
    try:
        # Check key matching
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        missing = model_keys - checkpoint_keys
        unexpected = checkpoint_keys - model_keys
        
        if missing:
            print(f"   ⚠️  Missing keys ({len(missing)}): {list(missing)[:5]}...")
        if unexpected:
            print(f"   ⚠️  Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}...")
        
        matched = model_keys & checkpoint_keys
        print(f"   📊 Key matching: {len(matched)}/{len(model_keys)} model keys matched")
        
        if len(matched) < len(model_keys) * 0.5:
            print(f"\n   ❌ CRITICAL: Less than 50% of keys matched!")
            print(f"   This usually means the model architecture doesn't match the checkpoint.")
            print(f"\n   Sample model keys: {list(model_keys)[:3]}")
            print(f"   Sample checkpoint keys: {list(checkpoint_keys)[:3]}")
        
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"   ⚠️  Load result - Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
        else:
            print(f"   ✓ All weights loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading weights: {e}")
    
    model = model.to(device)
    model.eval()
    
    # ===== Generate Predictions =====
    print(f"\n🔮 Generating predictions...")
    predictions = []
    ids = test_df['id'].tolist()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = coral_predict(logits)
            
            predictions.extend(preds.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # ===== Save Predictions =====
    print(f"\n💾 Writing predictions to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        for id_val, pred in zip(ids, predictions):
            f.write(json.dumps({"id": str(id_val), "prediction": float(pred)}) + '\n')
    
    # ===== Create Submission Zip =====
    if args.create_zip:
        # Determine zip filename
        if args.zip_file:
            zip_path = args.zip_file
        else:
            # Default: same name as output but with .zip extension
            base_name = os.path.splitext(args.output_file)[0]
            zip_path = f"{base_name}.zip"
        
        print(f"\n📦 Creating submission zip: {zip_path}")
        
        # The zip must contain a file named "predictions.jsonl" (competition requirement)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add the output file with the required name "predictions.jsonl"
            zf.write(args.output_file, arcname='predictions.jsonl')
        
        # Verify zip contents
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zip_contents = zf.namelist()
        
        print(f"   ✓ Zip created successfully!")
        print(f"   📋 Zip contents: {zip_contents}")
        
        if 'predictions.jsonl' in zip_contents:
            print(f"   ✅ Ready for CodaBench submission!")
        else:
            print(f"   ⚠️  Warning: predictions.jsonl not found in zip!")
    
    # ===== Summary =====
    print(f"\n" + "="*60)
    print(f"✅ PREDICTION COMPLETE")
    print(f"="*60)
    print(f"   Samples:     {len(predictions)}")
    print(f"   Output file: {args.output_file}")
    print(f"   Pred range:  [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"   Pred mean:   {predictions.mean():.3f}")
    print(f"   Pred std:    {predictions.std():.3f}")
    
    # If ground truth available, calculate metrics
    if 'average' in test_df.columns and test_df['average'].nunique() > 1:
        from scipy.stats import spearmanr
        targets = test_df['average'].values
        stdevs = test_df['stdev'].values
        
        # Spearman correlation
        spearman, _ = spearmanr(predictions, targets)
        
        # ACC within SD - IMPORTANT: clamp minimum stdev to 1.0 (matches notebook!)
        within_sd = np.abs(predictions - targets) <= np.maximum(stdevs, 1.0)
        acc_within_sd = np.mean(within_sd)
        
        # Additional metrics
        mae = np.mean(np.abs(predictions - targets))
        
        print(f"\n📊 Validation Metrics (matching notebook calculation):")
        print(f"   Spearman:     {spearman:.4f}")
        print(f"   ACC w/ SD:    {acc_within_sd:.4f}")
        print(f"   MAE:          {mae:.4f}")
    
    # Final instructions
    if args.create_zip:
        zip_name = args.zip_file if args.zip_file else f"{os.path.splitext(args.output_file)[0]}.zip"
        print(f"\n🚀 READY TO SUBMIT: Upload '{zip_name}' to CodaBench")
    else:
        print(f"\n📦 To submit: zip {args.output_file} as 'predictions.jsonl' and upload to CodaBench")


if __name__ == '__main__':
    main()
