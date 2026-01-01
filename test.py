#!/usr/bin/env python
"""SemEval 2026 Task 5 training pipeline with optional FEWS augmentation."""

import argparse
import json
import math
import os
import random
import re
import statistics
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

warnings_imported = False
try:
    import warnings
    warnings.filterwarnings("ignore")
    warnings_imported = True
except Exception:  # pragma: no cover
    warnings_imported = False

WSD_PATTERN = re.compile(r"<WSD>(.*?)</WSD>", re.IGNORECASE)
TARGET_TEMPLATE = "[TGT]{token}[/TGT]"
MIN_STD = 0.35
PRED_MIN, PRED_MAX = 1.0, 5.0


def parse_args():
    parser = argparse.ArgumentParser(description="SemEval Task 5 training pipeline")
    parser.add_argument('--train_path', default='data/train.json')
    parser.add_argument('--dev_path', default='data/dev.json')
    parser.add_argument('--model_name', default='roberta-base')
    parser.add_argument('--epochs', type=int, default=10)  # More epochs for exploration
    parser.add_argument('--batch_size', type=int, default=16)  # Slightly larger batches
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)  # Higher starting LR
    parser.add_argument('--weight_decay', type=float, default=0.15)  # Increased regularization
    parser.add_argument('--warmup_ratio', type=float, default=0.15)  # More warmup steps
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.35)  # Moderate dropout
    parser.add_argument('--pooling', choices=['cls', 'mean', 'weighted'], default='weighted')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--save_dir', default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_baseline', action='store_true')
    parser.add_argument('--skip_plots', action='store_true')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_dev_samples', type=int, default=None)

    parser.add_argument('--fews_dir', type=str, default=None)
    parser.add_argument('--fews_max_examples', type=int, default=60000)
    parser.add_argument('--fews_negatives', type=int, default=1)
    parser.add_argument('--fews_include_ext', action='store_true')
    parser.add_argument('--fews_weight', type=float, default=1.0)

    parser.add_argument('--disable_mark_homonym', dest='mark_homonym', action='store_false')
    parser.set_defaults(mark_homonym=True)
    parser.add_argument('--freeze_layers', type=int, default=0)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using device: mps (Apple Silicon GPU) 🚀\n")
        return torch.device('mps')
    if torch.cuda.is_available():
        print("Using device: cuda (NVIDIA GPU)\n")
        return torch.device('cuda')
    print("Using device: cpu\n")
    return torch.device('cpu')


def load_json_dataset(filepath: str, source: str) -> pd.DataFrame:
    with open(filepath, 'r') as f:
        data = json.load(f)
    rows = []
    for key, item in data.items():
        row = item.copy()
        row['id'] = key
        row['source'] = row.get('source', source)
        rows.append(row)
    df = pd.DataFrame(rows)
    df['source'] = df['source'].fillna(source)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = ['homonym', 'judged_meaning', 'precontext', 'sentence', 'ending',
                 'example_sentence', 'sense_tags', 'sense_synonyms']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('')
    for numeric in ['average', 'stdev']:
        if numeric not in df.columns:
            df[numeric] = np.nan
        df[numeric] = df[numeric].astype(float)
    if 'choices' not in df.columns:
        df['choices'] = [[] for _ in range(len(df))]
    if 'nonsensical' not in df.columns:
        df['nonsensical'] = [[] for _ in range(len(df))]
    return df


def highlight_target(sentence: str, homonym: str) -> str:
    """Highlight target word in sentence, handling case/inflection mismatches."""
    if not sentence or '[TGT]' in sentence or not homonym:
        return sentence
    
    # Try exact word boundary match first (case-insensitive)
    pattern = re.compile(rf"\b{re.escape(homonym)}\b", re.IGNORECASE)
    
    # Check if pattern matches before substituting
    if pattern.search(sentence):
        return pattern.sub(lambda m: TARGET_TEMPLATE.format(token=m.group(0)), sentence, count=1)
    
    # Fallback: try matching prefix (handles "running" when homonym is "run")
    # Only match if homonym is substantial (>= 3 chars) to avoid false positives
    if len(homonym) >= 3:
        prefix_pattern = re.compile(rf"\b{re.escape(homonym)}\w*\b", re.IGNORECASE)
        if prefix_pattern.search(sentence):
            return prefix_pattern.sub(lambda m: TARGET_TEMPLATE.format(token=m.group(0)), sentence, count=1)
    
    # Last resort: if no match found, just return original sentence
    # This prevents errors but means target won't be highlighted
    return sentence


def create_text_input(row: pd.Series, mark_homonym: bool) -> str:
    sentence = row['sentence']
    if mark_homonym:
        sentence = highlight_target(sentence, row.get('homonym', ''))
    parts = [
        f"Ambiguous word: {row.get('homonym', '').strip()}.",
        f"Candidate sense: {row.get('judged_meaning', '').strip()}."
    ]
    tags = row.get('sense_tags', '').strip()
    if tags:
        parts.append(f"Sense tags: {tags}.")
    example = row.get('example_sentence', '').strip()
    if example:
        parts.append(f"Dictionary example: {example}")
    precontext = row.get('precontext', '').strip()
    if precontext:
        parts.append(f"Story context: {precontext}")
    parts.append(f"Target sentence: {sentence}")
    ending = row.get('ending', '').strip()
    if ending:
        parts.append(f"Ending: {ending}")
    return " ".join([p for p in parts if p])


def compute_sample_weight(stdev: float, multiplier: float = 1.0) -> float:
    if math.isnan(stdev) or stdev <= 0:
        stdev = MIN_STD
    clipped = max(MIN_STD, stdev)
    return float((1.0 / (clipped + 0.2)) * multiplier)


# FEWS helpers

def load_fews_dataframe(
    fews_dir: str,
    max_examples: Optional[int],
    negatives: int,
    include_ext: bool,
    weight_multiplier: float,
    seed: int,
) -> pd.DataFrame:
    fews_root = Path(fews_dir)
    senses_path = fews_root / 'senses.txt'
    if not senses_path.exists():
        raise FileNotFoundError(f"senses.txt not found at {senses_path}")

    senses, lemma_to_senses = _load_fews_senses(senses_path)
    example_files = [fews_root / 'train' / 'train.txt']
    if include_ext:
        example_files.append(fews_root / 'train' / 'train.ext.txt')

    examples: List[Tuple[str, str]] = []
    for path in example_files:
        if path.exists():
            examples.extend(_load_fews_examples(path))
    if not examples:
        raise RuntimeError("No FEWS examples found")

    rng = random.Random(seed)
    if max_examples and max_examples < len(examples):
        examples = rng.sample(examples, max_examples)

    rows = []
    skipped_missing = 0
    skipped_empty_gloss = 0
    
    for idx, (raw_sentence, sense_id) in enumerate(tqdm(examples, desc="FEWS -> SemEval")):
        sense_meta = senses.get(sense_id)
        if not sense_meta:
            skipped_missing += 1
            continue
        
        # Extract bare lemma from sense_id (e.g., "potential.noun.0" -> "potential")
        # This ensures homonym matches AmbiStory format (no suffix)
        lemma = sense_id.split('.')[0] if '.' in sense_id else sense_id
        
        # Validate gloss exists and is non-empty
        gloss = sense_meta.get('gloss', '').strip()
        if not gloss:
            skipped_empty_gloss += 1
            continue
        
        sentence, target_token = _normalize_fews_sentence(raw_sentence)
        
        # If no explicit target found, use lemma for highlighting
        # Handle case/inflection: try exact match first, then case-insensitive
        if not target_token:
            sentence = highlight_target(sentence, lemma)
        
        rows.append(
            _build_fews_row(
                base_id=f"fews-{idx}",
                lemma=lemma,  # Use bare lemma (no .pos.sense suffix)
                sentence=sentence,
                sense_meta=sense_meta,
                sense_id=sense_id,  # Keep full sense_id for internal use only
                positive=True,
                rng=rng,
                weight_multiplier=weight_multiplier,
            )
        )
        
        # Generate negative examples with OTHER senses of same lemma
        lemma_senses = [sid for sid in lemma_to_senses.get(lemma, []) if sid != sense_id]
        if not lemma_senses:
            continue
        
        for neg in range(negatives):
            neg_id = rng.choice(lemma_senses)
            neg_meta = senses.get(neg_id)
            if not neg_meta:
                continue
            
            # Validate negative gloss exists
            neg_gloss = neg_meta.get('gloss', '').strip()
            if not neg_gloss:
                continue
            
            rows.append(
                _build_fews_row(
                    base_id=f"fews-{idx}-neg{neg}",
                    lemma=lemma,  # Same lemma, different sense
                    sentence=sentence,
                    sense_meta=neg_meta,
                    sense_id=neg_id,
                    positive=False,
                    rng=rng,
                    weight_multiplier=weight_multiplier,
                )
            )
    
    # Report statistics
    print(f"\n   FEWS Processing Stats:")
    print(f"     ✓ Successfully processed: {len(rows)} samples")
    print(f"     ⚠ Skipped (missing sense_id): {skipped_missing}")
    print(f"     ⚠ Skipped (empty gloss): {skipped_empty_gloss}")
    
    # Validation: print sample conversions for debugging
    if len(rows) > 0:
        print(f"\n   Sample FEWS -> AmbiStory conversions:")
        for i in range(min(3, len(rows))):
            sample = rows[i]
            print(f"\n   Example {i+1}:")
            print(f"     Homonym (bare lemma): '{sample['homonym']}'")
            print(f"     Sense ID (internal): '{sample['sense_id']}'")
            print(f"     Gloss: '{sample['judged_meaning'][:80]}...'")
            print(f"     Sentence: '{sample['sentence'][:100]}...'")
            # Verify no sense_id suffix in homonym
            if '.' in sample['homonym']:
                print(f"     ❌ WARNING: Homonym contains suffix!")
    
    return pd.DataFrame(rows)


def _load_fews_senses(path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    senses: Dict[str, Dict[str, str]] = {}
    lemma_to_senses: Dict[str, List[str]] = {}
    current: Dict[str, str] = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                if current and 'sense_id' in current:
                    sense_id = current['sense_id']
                    senses[sense_id] = current
                    lemma = current.get('word', '')
                    lemma_to_senses.setdefault(lemma, []).append(sense_id)
                current = {}
                continue
            if ':\t' not in line:
                continue
            key, value = line.split(':\t', 1)
            current[key.strip()] = value.strip()
    if current and 'sense_id' in current:
        sense_id = current['sense_id']
        senses[sense_id] = current
        lemma = current.get('word', '')
        lemma_to_senses.setdefault(lemma, []).append(sense_id)
    return senses, lemma_to_senses


def _load_fews_examples(path: Path) -> List[Tuple[str, str]]:
    records = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or '\t' not in line:
                continue
            sentence, label = line.strip().split('\t', 1)
            records.append((sentence.strip(), label.strip()))
    return records


def _normalize_fews_sentence(raw_sentence: str) -> Tuple[str, Optional[str]]:
    targets: List[str] = []

    def repl(match):
        token = match.group(1).strip()
        targets.append(token)
        return TARGET_TEMPLATE.format(token=token)

    normalized = WSD_PATTERN.sub(repl, raw_sentence)
    target = targets[0] if targets else None
    return normalized, target


def _build_fews_row(
    base_id: str,
    lemma: str,  # Changed from 'homonym' to 'lemma' for clarity
    sentence: str,
    sense_meta: Dict[str, str],
    sense_id: str,  # Full sense_id ONLY for internal tracking - NOT exposed to model
    positive: bool,
    rng: random.Random,
    weight_multiplier: float,
) -> Dict:
    # Make FEWS scores more realistic: wider distribution, less bimodal
    # Positive: 3.5-4.8 (instead of 4.4-4.9), Negative: 1.2-2.8 (instead of 1.0-1.6)
    if positive:
        base = rng.uniform(3.5, 4.8)
    else:
        base = rng.uniform(1.2, 2.8)
    
    votes = []
    for offset in [-0.4, -0.2, 0.0, 0.2, 0.4]:  # Wider variance
        jitter = rng.uniform(-0.3, 0.3)  # More noise
        score = max(PRED_MIN, min(PRED_MAX, base + offset + jitter))
        votes.append(int(round(score)))
    average = float(statistics.mean(votes))
    stdev = float(statistics.pstdev(votes)) if len(votes) > 1 else 0.6  # Higher variance
    
    # Extract metadata - ensure gloss exists
    synonyms = sense_meta.get('synonyms', '').strip()
    tags = sense_meta.get('tags', '').strip()
    gloss = sense_meta.get('gloss', '').strip()
    
    # CRITICAL: Ensure gloss is never empty (would harm model)
    if not gloss:
        gloss = f"A sense of {lemma}"  # Fallback
    
    # Build AmbiStory-compatible row
    # IMPORTANT: Use ONLY the bare lemma (no .pos.sense suffix) for homonym field
    # sense_id is stored but NEVER exposed in the text fields seen by the model
    return {
        'id': base_id,
        'sample_id': base_id,
        'homonym': lemma,  # ✓ BARE LEMMA ONLY (e.g., "potential" not "potential.noun.0")
        'judged_meaning': gloss,  # ✓ Validated non-empty gloss
        'precontext': '',  # FEWS has no story context
        'sentence': sentence,  # Already has [TGT] markers
        'ending': '',  # FEWS has no ending
        'choices': votes,
        'average': average,
        'stdev': stdev,
        'nonsensical': [False] * len(votes),
        'example_sentence': synonyms if synonyms else '',  # Use synonyms as example
        'sense_tags': tags if tags else '',  # POS/usage tags
        'sense_synonyms': synonyms if synonyms else '',
        'source': 'fews-positive' if positive else 'fews-negative',
        'sense_id': sense_id,  # ⚠️ INTERNAL ONLY - not used in create_text_input()
        'confidence_weight': compute_sample_weight(stdev, weight_multiplier),
    }


class PlausibilityDataset(Dataset):
    def __init__(self, texts: List[str], scores: List[float], weights: List[float], tokenizer, max_length: int):
        self.texts = texts
        self.scores = scores
        self.weights = weights
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
            'weight': torch.tensor(self.weights[idx], dtype=torch.float)
        }


class PlausibilityModel(nn.Module):
    """RoBERTa + LayerNorm + Small regression head with configurable pooling

    pooling: 'cls' | 'mean' | 'weighted'
    """
    def __init__(self, model_name: str, dropout: float, pooling: str = 'cls', freeze_layers: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        
        # Optional attention scorer for weighted pooling
        if pooling == 'weighted':
            self.attn = nn.Linear(hidden, 1)

        # Regression head with LayerNorm: 768 → LayerNorm → 192 → 1 (slightly larger)
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden),  # Stabilizes training
            nn.Linear(hidden, 192),  # Increased capacity
            nn.GELU(),  # Smoother activation than Tanh
            nn.Dropout(dropout),
            nn.Linear(192, 1)
        )
        self.dropout = nn.Dropout(dropout)
        
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, layers_to_freeze: int):
        encoder_layers = getattr(self.encoder, 'encoder', None)
        if encoder_layers is None:
            return
        layer_module = getattr(encoder_layers, 'layer', None)
        if not layer_module:
            return
        for layer in layer_module[:layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

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
            # attention scorer -> masked softmax -> weighted sum
            scores = self.attn(hidden_states).squeeze(-1)  # (batch, seq_len)
            # mask padding tokens
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            weights = F.softmax(scores, dim=1)
            pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        else:
            # fallback to cls
            pooled = hidden_states[:, 0, :]

        pooled = self.dropout(pooled)
        # Direct linear projection to score
        logits = self.regressor(pooled)

        # Clamp only at inference
        if not self.training:
            logits = torch.clamp(logits, min=PRED_MIN, max=PRED_MAX)

        return logits.squeeze(-1)


def calculate_metrics(predictions, targets, stdevs):
    spearman_corr, p_val = spearmanr(predictions, targets)
    within_sd = np.mean(np.abs(predictions - targets) <= np.maximum(stdevs, 1.0))
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    return {
        'spearman_correlation': float(spearman_corr),
        'spearman_pvalue': float(p_val),
        'accuracy_within_sd': float(within_sd),
        'mse': float(mse),
        'mae': float(mae)
    }


def print_metrics(metrics, label):
    print(f"\n=== {label} Performance ===")
    print(f"Spearman Correlation: {metrics['spearman_correlation']:.4f} (p={metrics['spearman_pvalue']:.4e})")
    print(f"Accuracy Within SD: {metrics['accuracy_within_sd']:.4f} ({metrics['accuracy_within_sd']*100:.2f}%)")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")


def check_data_distribution(df, label):
    """Diagnostic function to check if data has sufficient variance"""
    print(f"\n=== {label} Data Distribution ===")
    print(f"  Mean: {df['average'].mean():.3f}")
    print(f"  Std: {df['average'].std():.3f}")
    print(f"  Min: {df['average'].min():.3f}")
    print(f"  Max: {df['average'].max():.3f}")
    print(f"  Median: {df['average'].median():.3f}")
    
    if df['average'].std() < 0.5:
        print("  ⚠️  WARNING: Low variance in target values - model may predict mean!")


def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_accum, use_amp):
    """Training loop with Huber loss + variance penalty + label smoothing"""
    model.train()
    total_loss = 0.0
    # Huber loss: MSE for small errors, MAE for large errors
    # More robust than MSE, penalizes constant predictions more
    criterion = nn.HuberLoss(delta=1.0)
    label_smoothing = 0.01  # Reduced smoothing for clearer gradient signal
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)
        
        # Forward pass
        preds = model(input_ids, attention_mask)
        
        # Apply label smoothing: add small noise to targets for regularization
        if model.training and label_smoothing > 0:
            noise = torch.randn_like(scores) * label_smoothing
            scores_smoothed = torch.clamp(scores + noise, min=PRED_MIN, max=PRED_MAX)
        else:
            scores_smoothed = scores
        
        # Debug every 100 batches
        if step % 100 == 0:
            pred_std = preds.std().item()
            print(f"\nBatch {step}: Pred range [{preds.min().item():.3f}, {preds.max().item():.3f}], "
                  f"Mean: {preds.mean().item():.3f}, Std: {pred_std:.3f} | "
                  f"Actual range [{scores.min().item():.3f}, {scores.max().item():.3f}], "
                  f"Mean: {scores.mean().item():.3f}")
        
        # Huber loss for main prediction error
        loss = criterion(preds, scores_smoothed)
        
        # Add variance penalty: encourage diverse predictions
        # If all predictions are similar (low std), add penalty
        pred_std = preds.std()
        target_std = scores.std()
        variance_penalty = torch.relu(target_std - pred_std) * 0.15  # Reduced penalty
        loss = loss + variance_penalty
        
        total_loss += loss.item()
        
        # Backward pass
        loss = loss / grad_accum
        loss.backward()
        
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    # Handle remainder
    if len(dataloader) % grad_accum != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            outputs = model(input_ids, attention_mask)
            preds.append(outputs.cpu().numpy())
            labels.append(scores.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


def create_visualizations(train_df, dev_df, dev_targets, baseline_preds, roberta_preds,
                           roberta_targets, baseline_metrics, roberta_metrics):
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(train_df['average'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Training Rating Distribution')
    axes[1].hist(train_df['stdev'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_title('Training Rating StdDev Distribution')
    plt.tight_layout()
    plt.savefig('rating_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(dev_targets, baseline_preds, alpha=0.5, s=20)
    axes[0].plot([1, 5], [1, 5], 'r--', linewidth=2)
    axes[0].set_title(f"Baseline Spearman: {baseline_metrics['spearman_correlation']:.4f}")
    axes[1].scatter(roberta_targets, roberta_preds, alpha=0.5, s=20, color='green')
    axes[1].plot([1, 5], [1, 5], 'r--', linewidth=2)
    axes[1].set_title(f"Transformer Spearman: {roberta_metrics['spearman_correlation']:.4f}")
    for ax in axes:
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.set_xlabel('Actual Scores')
        ax.set_ylabel('Predicted Scores')
    plt.tight_layout()
    plt.savefig('predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    models = ['Baseline', 'Transformer']
    spearman_scores = [baseline_metrics['spearman_correlation'], roberta_metrics['spearman_correlation']]
    accuracy_scores = [baseline_metrics['accuracy_within_sd'], roberta_metrics['accuracy_within_sd']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(models, spearman_scores, color=['steelblue', 'green'])
    axes[0].set_title('Spearman Comparison')
    axes[1].bar(models, accuracy_scores, color=['steelblue', 'green'])
    axes[1].set_title('Accuracy Within SD Comparison')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    baseline_errors = baseline_preds - dev_targets
    roberta_errors = roberta_preds - roberta_targets
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(baseline_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title(f"Baseline Errors (MAE {baseline_metrics['mae']:.3f})")
    axes[1].hist(roberta_errors, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_title(f"Transformer Errors (MAE {roberta_metrics['mae']:.3f})")
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("SemEval 2026 Task 5: Rating Plausibility of Word Senses")
    print("=" * 80)
    print("\n1. Loading datasets...")

    train_df = standardize_columns(load_json_dataset(args.train_path, 'semeval-train'))
    dev_df = standardize_columns(load_json_dataset(args.dev_path, 'semeval-dev'))

    print(f"   Training set size (SemEval): {len(train_df)}")
    print(f"   Development set size: {len(dev_df)}")

    if args.max_train_samples:
        train_df = train_df.sample(args.max_train_samples, random_state=args.seed).reset_index(drop=True)
    if args.max_dev_samples:
        dev_df = dev_df.sample(args.max_dev_samples, random_state=args.seed).reset_index(drop=True)

    if args.fews_dir:
        fews_df = load_fews_dataframe(
            fews_dir=args.fews_dir,
            max_examples=args.fews_max_examples,
            negatives=args.fews_negatives,
            include_ext=args.fews_include_ext,
            weight_multiplier=args.fews_weight,
            seed=args.seed,
        )
        print(f"   FEWS synthetic samples: {len(fews_df)}")
        train_df = pd.concat([train_df, fews_df], ignore_index=True)

    for df in [train_df, dev_df]:
        df['sentence'] = df['sentence'].fillna('')
        df['text'] = df.apply(lambda row: create_text_input(row, args.mark_homonym), axis=1)

        def resolve_weight(row):
            # TEMPORARILY DISABLE WEIGHTING - testing if it's causing collapse
            return 1.0
            # confidence = row.get('confidence_weight', np.nan)
            # if not pd.isna(confidence):
            #     return float(confidence)
            # return compute_sample_weight(row['stdev'])

        df['weight'] = df.apply(resolve_weight, axis=1)

    if args.fews_dir:
        mask = train_df['source'].astype(str).str.startswith('fews')
        train_df.loc[mask, 'weight'] *= args.fews_weight

    # Check data distribution for potential issues
    check_data_distribution(train_df, "Training Set")
    check_data_distribution(dev_df, "Development Set")

    baseline_metrics = {
        'spearman_correlation': 0.0,
        'spearman_pvalue': 1.0,
        'accuracy_within_sd': 0.0,
        'mse': 0.0,
        'mae': 0.0
    }
    baseline_preds_dev = np.zeros(len(dev_df))

    if not args.skip_baseline:
        print("\n" + "=" * 80)
        print("3. BASELINE MODEL: TF-IDF + Ridge Regression")
        print("=" * 80)
        vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 3), min_df=2)
        X_train = vectorizer.fit_transform(train_df['text'])
        X_dev = vectorizer.transform(dev_df['text'])
        baseline_model = Ridge(alpha=1.0, solver='lsqr')
        baseline_model.fit(X_train, train_df['average'].fillna(3.0).values)
        baseline_preds_train = np.clip(baseline_model.predict(X_train), PRED_MIN, PRED_MAX)
        baseline_preds_dev = np.clip(baseline_model.predict(X_dev), PRED_MIN, PRED_MAX)
        baseline_metrics = calculate_metrics(baseline_preds_dev,
                                             dev_df['average'].values,
                                             dev_df['stdev'].fillna(1.0).values)
        print_metrics(baseline_metrics, "Baseline (Dev)")

    print("\n" + "=" * 80)
    print("4. TRANSFORMER MODEL")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = PlausibilityDataset(
        texts=train_df['text'].tolist(),
        scores=train_df['average'].fillna(3.0).tolist(),
        weights=train_df['weight'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    dev_dataset = PlausibilityDataset(
        texts=dev_df['text'].tolist(),
        scores=dev_df['average'].fillna(3.0).tolist(),
        weights=dev_df['weight'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    model = PlausibilityModel(
        model_name=args.model_name,
        dropout=args.dropout,
        pooling=args.pooling,  # Use the pooling argument from command line
        freeze_layers=args.freeze_layers
    ).to(device)

    # SIMPLE: Single learning rate for everything
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.grad_accumulation) * args.epochs
    
    # Standard warmup schedule
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )

    best_spearman = -1
    patience_counter = 0
    best_path = os.path.join(args.save_dir, 'best_model.pt')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum=args.grad_accumulation,
            use_amp=args.use_amp
        )
        print(f"   Training loss: {train_loss:.4f}")

        dev_preds, dev_actuals = evaluate_model(model, dev_loader, device)
        dev_metrics = calculate_metrics(
            dev_preds,
            dev_actuals,
            dev_df['stdev'].fillna(1.0).values
        )
        print_metrics(dev_metrics, f"Transformer (Dev) - Epoch {epoch + 1}")

        if dev_metrics['spearman_correlation'] > best_spearman:
            best_spearman = dev_metrics['spearman_correlation']
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"   ✓ Best model updated (Spearman {best_spearman:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print("   Early stopping triggered.")
                break

    print("\n" + "=" * 80)
    print("5. FINAL EVALUATION")
    print("=" * 80)
    model.load_state_dict(torch.load(best_path, map_location=device))

    train_preds, train_actuals = evaluate_model(model, train_loader, device)
    dev_preds, dev_actuals = evaluate_model(model, dev_loader, device)

    train_metrics = calculate_metrics(train_preds, train_actuals, train_df['stdev'].fillna(1.0).values)
    dev_metrics = calculate_metrics(dev_preds, dev_actuals, dev_df['stdev'].fillna(1.0).values)
    print_metrics(train_metrics, "Transformer Final (Train)")
    print_metrics(dev_metrics, "Transformer Final (Dev)")

    print("\n" + "=" * 80)
    print("6. RESULTS SUMMARY")
    print("=" * 80)
    results_summary = pd.DataFrame({
        'Model': ['Baseline', 'Transformer'],
        'Spearman Correlation': [baseline_metrics['spearman_correlation'], dev_metrics['spearman_correlation']],
        'Accuracy Within SD': [baseline_metrics['accuracy_within_sd'], dev_metrics['accuracy_within_sd']],
        'MAE': [baseline_metrics['mae'], dev_metrics['mae']],
        'MSE': [baseline_metrics['mse'], dev_metrics['mse']]
    })
    print("\n" + results_summary.to_string(index=False))
    results_summary.to_csv('results_summary.csv', index=False)

    if not args.skip_plots:
        create_visualizations(
            train_df=train_df,
            dev_df=dev_df,
            dev_targets=dev_df['average'].values,
            baseline_preds=baseline_preds_dev,
            roberta_preds=dev_preds,
            roberta_targets=dev_actuals,
            baseline_metrics=baseline_metrics,
            roberta_metrics=dev_metrics
        )

    print("\n" + "=" * 80)
    print("COMPLETE! All models trained and evaluated.")
    print("=" * 80)


if __name__ == '__main__':
    main()
