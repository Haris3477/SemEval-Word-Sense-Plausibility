"""
SemEval 2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences
Main implementation script
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
# from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, AdamW

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration - Support for Apple Silicon (M1/M2/M3) GPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Using device: mps (Apple Silicon GPU) 🚀\n")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: cuda (NVIDIA GPU)\n")
else:
    device = torch.device('cpu')
    print(f"Using device: cpu\n")


# ========================
# Data Loading Functions
# ========================

def load_data(filepath):
    """Load JSON data and convert to pandas DataFrame"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    data_list = []
    for key, item in data.items():
        item['id'] = key
        data_list.append(item)
    
    return pd.DataFrame(data_list)


def create_text_input(row):
    """Create full text input from all context fields"""
    parts = [
        f"Word sense: {row['judged_meaning']}.",
        f"Example: {row['example_sentence']}",
        f"Context: {row['precontext']}",
        f"Sentence: {row['sentence']}"
    ]
    
    if row['ending'] and str(row['ending']).strip():
        parts.append(f"Ending: {row['ending']}")
    
    return " ".join(parts)


# ========================
# Evaluation Functions
# ========================

def calculate_metrics(predictions, targets, stdevs):
    """Calculate Spearman correlation and Accuracy within SD"""
    # Spearman correlation
    spearman_corr, p_value = spearmanr(predictions, targets)
    
    # Accuracy within standard deviation
    within_sd = 0
    for pred, target, stdev in zip(predictions, targets, stdevs):
        if abs(pred - target) <= max(stdev, 1.0):
            within_sd += 1
    
    accuracy_within_sd = within_sd / len(predictions)
    
    # Additional metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    return {
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': p_value,
        'accuracy_within_sd': accuracy_within_sd,
        'mse': mse,
        'mae': mae
    }


def print_metrics(metrics, model_name="Model"):
    """Print metrics in a formatted way"""
    print(f"\n=== {model_name} Performance ===")
    print(f"Spearman Correlation: {metrics['spearman_correlation']:.4f} (p={metrics['spearman_pvalue']:.4e})")
    print(f"Accuracy Within SD: {metrics['accuracy_within_sd']:.4f} ({metrics['accuracy_within_sd']*100:.2f}%)")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")


# ========================
# PyTorch Dataset
# ========================

class PlausibilityDataset(Dataset):
    """PyTorch Dataset for plausibility rating prediction"""
    
    def __init__(self, texts, scores, tokenizer, max_length=512):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        score = self.scores[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'score': torch.tensor(score, dtype=torch.float)
        }


# ========================
# RoBERTa Model
# ========================

class RoBERTaPlausibilityModel(nn.Module):
    """RoBERTa-based model for plausibility prediction"""
    
    def __init__(self, model_name='roberta-base', dropout=0.3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.regressor(cls_output)
        output = torch.sigmoid(logits) * 4 + 1
        return output.squeeze(-1)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)
        
        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = nn.MSELoss()(predictions, scores)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            
            preds = model(input_ids, attention_mask)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(scores.cpu().numpy())
    
    return np.array(predictions), np.array(actuals)


# ========================
# Main Execution
# ========================

def main():
    print("="*80)
    print("SemEval 2026 Task 5: Rating Plausibility of Word Senses")
    print("="*80)
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_df = load_data('data/train.json')
    dev_df = load_data('data/dev.json')
    
    print(f"   Training set size: {len(train_df)}")
    print(f"   Development set size: {len(dev_df)}")
    
    # Create text inputs
    print("\n2. Creating text inputs...")
    train_df['text'] = train_df.apply(create_text_input, axis=1)
    dev_df['text'] = dev_df.apply(create_text_input, axis=1)
    
    # ========================
    # Baseline Model
    # ========================
    print("\n" + "="*80)
    print("3. BASELINE MODEL: TF-IDF + Ridge Regression")
    print("="*80)
    
    print("\n   Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(train_df['text'])
    X_dev_tfidf = vectorizer.transform(dev_df['text'])
    y_train = train_df['average'].values
    y_dev = dev_df['average'].values
    
    print(f"   TF-IDF feature shape: {X_train_tfidf.shape}")
    
    print("\n   Training Ridge Regression model...")
    baseline_model = Ridge(alpha=1.0, solver='lsqr')
    baseline_model.fit(X_train_tfidf, y_train)
    
    # Predictions
    train_preds_baseline = np.clip(baseline_model.predict(X_train_tfidf), 1, 5)
    dev_preds_baseline = np.clip(baseline_model.predict(X_dev_tfidf), 1, 5)
    
    # Evaluate
    train_metrics_baseline = calculate_metrics(train_preds_baseline, y_train, train_df['stdev'].values)
    dev_metrics_baseline = calculate_metrics(dev_preds_baseline, y_dev, dev_df['stdev'].values)
    
    print_metrics(train_metrics_baseline, "Baseline (Train)")
    print_metrics(dev_metrics_baseline, "Baseline (Dev)")
    
    # ========================
    # Transformer Model
    # ========================
    print("\n" + "="*80)
    print("4. TRANSFORMER MODEL: RoBERTa-based Plausibility Predictor")
    print("="*80)
    
    MODEL_NAME = 'roberta-base'
    print(f"\n   Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    print("\n   Creating PyTorch datasets...")
    train_dataset = PlausibilityDataset(
        train_df['text'].tolist(),
        train_df['average'].tolist(),
        tokenizer,
        max_length=512
    )
    
    dev_dataset = PlausibilityDataset(
        dev_df['text'].tolist(),
        dev_df['average'].tolist(),
        tokenizer,
        max_length=512
    )
    
    # Create dataloaders
    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Dev batches: {len(dev_loader)}")
    
    # Initialize model
    print("\n   Initializing RoBERTa model...")
    roberta_model = RoBERTaPlausibilityModel(MODEL_NAME).to(device)
    
    # Training configuration
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    optimizer = AdamW(roberta_model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print(f"\n   Training configuration:")
    print(f"     Epochs: {EPOCHS}")
    print(f"     Learning rate: {LEARNING_RATE}")
    print(f"     Batch size: {BATCH_SIZE}")
    
    # Training loop
    print("\n   Starting training...")
    best_dev_spearman = -1
    
    for epoch in range(EPOCHS):
        print(f"\n   {'='*60}")
        print(f"   Epoch {epoch + 1}/{EPOCHS}")
        print(f"   {'='*60}")
        
        # Train
        train_loss = train_epoch(roberta_model, train_loader, optimizer, scheduler, device)
        print(f"\n   Average training loss: {train_loss:.4f}")
        
        # Evaluate on dev set
        dev_preds, dev_actuals = evaluate_model(roberta_model, dev_loader, device)
        dev_metrics = calculate_metrics(dev_preds, dev_actuals, dev_df['stdev'].values)
        
        print_metrics(dev_metrics, f"RoBERTa (Dev) - Epoch {epoch+1}")
        
        # Save best model
        if dev_metrics['spearman_correlation'] > best_dev_spearman:
            best_dev_spearman = dev_metrics['spearman_correlation']
            torch.save(roberta_model.state_dict(), 'trained_models/best_roberta_model.pt')
            print(f"\n   ✓ Best model saved (Spearman: {best_dev_spearman:.4f})")
    
    # Load best model and final evaluation
    print("\n" + "="*80)
    print("5. FINAL EVALUATION")
    print("="*80)
    
    print("\n   Loading best model...")
    roberta_model.load_state_dict(torch.load('trained_models/best_roberta_model.pt'))
    
    train_preds_roberta, train_actuals = evaluate_model(roberta_model, train_loader, device)
    dev_preds_roberta, dev_actuals = evaluate_model(roberta_model, dev_loader, device)
    
    train_metrics_roberta = calculate_metrics(train_preds_roberta, train_actuals, train_df['stdev'].values)
    dev_metrics_roberta = calculate_metrics(dev_preds_roberta, dev_actuals, dev_df['stdev'].values)
    
    print_metrics(train_metrics_roberta, "RoBERTa Final (Train)")
    print_metrics(dev_metrics_roberta, "RoBERTa Final (Dev)")
    
    # ========================
    # Results Summary
    # ========================
    print("\n" + "="*80)
    print("6. RESULTS SUMMARY")
    print("="*80)
    
    results_summary = pd.DataFrame({
        'Model': ['Baseline (TF-IDF + Ridge)', 'RoBERTa'],
        'Spearman Correlation': [
            dev_metrics_baseline['spearman_correlation'],
            dev_metrics_roberta['spearman_correlation']
        ],
        'Accuracy Within SD': [
            dev_metrics_baseline['accuracy_within_sd'],
            dev_metrics_roberta['accuracy_within_sd']
        ],
        'MAE': [
            dev_metrics_baseline['mae'],
            dev_metrics_roberta['mae']
        ],
        'MSE': [
            dev_metrics_baseline['mse'],
            dev_metrics_roberta['mse']
        ]
    })
    
    print("\n" + results_summary.to_string(index=False))
    print("\n" + "="*80)
    
    # Save results
    results_summary.to_csv('results_summary.csv', index=False)
    print("\nResults saved to 'results_summary.csv'")
    
    # ========================
    # Visualizations
    # ========================
    print("\n7. Creating visualizations...")
    
    # Create visualizations
    create_visualizations(
        train_df, dev_df,
        y_dev, dev_preds_baseline, dev_preds_roberta, dev_actuals,
        dev_metrics_baseline, dev_metrics_roberta
    )
    
    print("\n" + "="*80)
    print("COMPLETE! All models trained and evaluated.")
    print("="*80)
    
    return {
        'baseline': dev_metrics_baseline,
        'roberta': dev_metrics_roberta,
        'results_df': results_summary
    }


def create_visualizations(train_df, dev_df, y_dev, dev_preds_baseline, 
                         dev_preds_roberta, dev_actuals, 
                         dev_metrics_baseline, dev_metrics_roberta):
    """Create all visualizations"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # 1. Distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(train_df['average'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Average Plausibility Rating', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Average Ratings (Training Set)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(train_df['stdev'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Standard Deviation of Ratings', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Rating Standard Deviations', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rating_distributions.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: rating_distributions.png")
    plt.close()
    
    # 2. Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].scatter(y_dev, dev_preds_baseline, alpha=0.5, s=20)
    axes[0].plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Plausibility Score', fontsize=12)
    axes[0].set_ylabel('Predicted Plausibility Score', fontsize=12)
    axes[0].set_title(f'Baseline Model (Dev Set)\nSpearman: {dev_metrics_baseline["spearman_correlation"]:.4f}', 
                     fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0.5, 5.5)
    axes[0].set_ylim(0.5, 5.5)
    
    axes[1].scatter(dev_actuals, dev_preds_roberta, alpha=0.5, s=20, color='green')
    axes[1].plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Plausibility Score', fontsize=12)
    axes[1].set_ylabel('Predicted Plausibility Score', fontsize=12)
    axes[1].set_title(f'RoBERTa Model (Dev Set)\nSpearman: {dev_metrics_roberta["spearman_correlation"]:.4f}', 
                     fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0.5, 5.5)
    axes[1].set_ylim(0.5, 5.5)
    
    plt.tight_layout()
    plt.savefig('predictions_scatter.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: predictions_scatter.png")
    plt.close()
    
    # 3. Model comparison
    models = ['Baseline', 'RoBERTa']
    spearman_scores = [
        dev_metrics_baseline['spearman_correlation'],
        dev_metrics_roberta['spearman_correlation']
    ]
    accuracy_scores = [
        dev_metrics_baseline['accuracy_within_sd'],
        dev_metrics_roberta['accuracy_within_sd']
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    bars1 = axes[0].bar(models, spearman_scores, color=['steelblue', 'green'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Spearman Correlation', fontsize=12)
    axes[0].set_title('Spearman Correlation Comparison (Dev Set)', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    bars2 = axes[1].bar(models, accuracy_scores, color=['steelblue', 'green'], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Accuracy Within SD', fontsize=12)
    axes[1].set_title('Accuracy Within SD Comparison (Dev Set)', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: model_comparison.png")
    plt.close()
    
    # 4. Error distributions
    baseline_errors = dev_preds_baseline - y_dev
    roberta_errors = dev_preds_roberta - dev_actuals
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(baseline_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Baseline Error Distribution\nMAE: {dev_metrics_baseline["mae"]:.4f}', 
                     fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(roberta_errors, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'RoBERTa Error Distribution\nMAE: {dev_metrics_roberta["mae"]:.4f}', 
                     fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: error_distribution.png")
    plt.close()


if __name__ == "__main__":
    results = main()

