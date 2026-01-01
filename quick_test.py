"""
Quick test version - Runs faster with reduced data/epochs
Use this to verify everything works before full training
"""

import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

print("="*60)
print("QUICK TEST MODE")
print("="*60)

# Load data
print("\n1. Loading data...")
with open('train.json', 'r') as f:
    train_data = json.load(f)
with open('dev.json', 'r') as f:
    dev_data = json.load(f)

# Convert to DataFrame
train_df = pd.DataFrame([{**v, 'id': k} for k, v in train_data.items()])
dev_df = pd.DataFrame([{**v, 'id': k} for k, v in dev_data.items()])

# Use only small subset for quick test
train_df = train_df.sample(500, random_state=42)
dev_df = dev_df.sample(200, random_state=42)

print(f"   Using {len(train_df)} training samples")
print(f"   Using {len(dev_df)} dev samples")

# Create text inputs
def create_text(row):
    parts = [
        f"Word sense: {row['judged_meaning']}.",
        f"Example: {row['example_sentence']}",
        f"Context: {row['precontext']}",
        f"Sentence: {row['sentence']}"
    ]
    if row['ending'] and str(row['ending']).strip():
        parts.append(f"Ending: {row['ending']}")
    return " ".join(parts)

print("\n2. Preprocessing...")
train_df['text'] = train_df.apply(create_text, axis=1)
dev_df['text'] = dev_df.apply(create_text, axis=1)

# Train baseline model
print("\n3. Training baseline model...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['text'])
X_dev = vectorizer.transform(dev_df['text'])

model = Ridge(alpha=1.0, solver='lsqr')
model.fit(X_train, train_df['average'].values)

# Evaluate
predictions = np.clip(model.predict(X_dev), 1, 5)
actuals = dev_df['average'].values

spearman = spearmanr(predictions, actuals)[0]
mae = mean_absolute_error(actuals, predictions)

# Accuracy within SD
within_sd = sum(abs(pred - actual) <= max(stdev, 1.0) 
                for pred, actual, stdev in zip(predictions, actuals, dev_df['stdev'])) / len(predictions)

print("\n" + "="*60)
print("RESULTS (Quick Test)")
print("="*60)
print(f"Spearman Correlation: {spearman:.4f}")
print(f"Accuracy Within SD: {within_sd:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print("\n✅ Everything works! Ready for full training.")
print("   Run: python semeval_task5_main.py")
print("="*60)

