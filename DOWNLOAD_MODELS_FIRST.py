"""
Quick script to pre-download DeBERTa models and avoid Jupyter progress bar issues.

Run this BEFORE running the notebook if you're experiencing download problems.
"""

import os

# Disable progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("=" * 80)
print("PRE-DOWNLOADING DeBERTa-v3-large MODEL & TOKENIZER")
print("=" * 80)

print("\n⏳ This will download:")
print("   • Tokenizer files (~2.5 MB)")
print("   • Model weights (~1.5 GB)")
print("   • Total download: ~1.5 GB")
print("\nThis may take 5-10 minutes depending on your internet speed...")
print("Files will be cached in ~/.cache/huggingface/\n")

from transformers import AutoModel, AutoTokenizer

model_name = 'microsoft/deberta-v3-large'

# Download tokenizer
print("\n[1/2] Downloading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print("✓ Tokenizer downloaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Retrying with cache_dir...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./hf_cache', use_fast=True)
    print("✓ Tokenizer downloaded to ./hf_cache/")

# Download model
print("\n[2/2] Downloading model weights (this is the big one - ~1.5GB)...")
try:
    model = AutoModel.from_pretrained(model_name)
    print("✓ Model downloaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Retrying with cache_dir...")
    model = AutoModel.from_pretrained(model_name, cache_dir='./hf_cache')
    print("✓ Model downloaded to ./hf_cache/")

print("\n" + "=" * 80)
print("✅ ALL DOWNLOADS COMPLETE!")
print("=" * 80)
print("\nYou can now run the notebook without download issues.")
print("The models are cached and will load instantly.")


