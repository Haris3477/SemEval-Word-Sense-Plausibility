#!/usr/bin/env python3
"""
Add synthetic precontext and ending to FEWS samples by borrowing from AmbiStory.

This makes FEWS samples structurally similar to AmbiStory so the model learns
to use story context on ALL training samples, not just the 2.3K AmbiStory ones.

Strategy:
1. Build a pool of (precontext, ending) pairs from AmbiStory train.json
2. For each FEWS sample, randomly attach one pair
3. Optionally filter by POS or use generic contexts

Usage:
    python add_context_to_fews.py --fews_file fews_train_10k_balanced.json --output fews_train_10k_with_context.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_ambistory_contexts(ambistory_path: str) -> List[Tuple[str, str]]:
    """Extract (precontext, ending) pairs from AmbiStory."""
    with open(ambistory_path, 'r') as f:
        data = json.load(f)
    
    contexts = []
    for key, sample in data.items():
        precontext = sample.get('precontext', '').strip()
        ending = sample.get('ending', '').strip()
        
        # Only use pairs where at least one exists
        if precontext or ending:
            contexts.append((precontext, ending))
    
    print(f"✓ Loaded {len(contexts)} context pairs from AmbiStory")
    return contexts


def add_contexts_to_fews(
    fews_path: str,
    contexts: List[Tuple[str, str]],
    seed: int = 42
) -> Dict:
    """Attach random AmbiStory contexts to FEWS samples."""
    with open(fews_path, 'r') as f:
        fews_data = json.load(f)
    
    rng = random.Random(seed)
    samples = list(fews_data.values())
    
    print(f"\n{'='*70}")
    print(f"Adding context to {len(samples)} FEWS samples...")
    print(f"{'='*70}")
    
    modified_count = 0
    for sample in samples:
        # Skip if already has context (shouldn't happen for pure FEWS)
        if sample.get('precontext', '').strip() or sample.get('ending', '').strip():
            continue
        
        # Randomly select a context pair
        precontext, ending = rng.choice(contexts)
        sample['precontext'] = precontext
        sample['ending'] = ending
        modified_count += 1
    
    print(f"✓ Added context to {modified_count} samples")
    
    # Rebuild dict with sequential keys
    output_data = {str(i): sample for i, sample in enumerate(samples)}
    return output_data


def validate_output(data: Dict, sample_count: int = 5):
    """Show sample outputs for validation."""
    print(f"\n{'='*70}")
    print(f"Sample outputs (first {sample_count}):")
    print(f"{'='*70}")
    
    samples = list(data.values())[:sample_count]
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. Word: '{sample['homonym']}'")
        print(f"   Sentence: {sample['sentence'][:80]}...")
        print(f"   Precontext: {sample.get('precontext', '')[:80]}...")
        print(f"   Ending: {sample.get('ending', '')[:80]}...")
        print(f"   Score: {sample['average']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Add AmbiStory context to FEWS samples")
    parser.add_argument('--fews_file', required=True, help='Input FEWS JSON file')
    parser.add_argument('--ambistory', default='data/train.json', help='AmbiStory train file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print("=" * 70)
    print("FEWS Context Augmentation")
    print("=" * 70)
    
    # Load AmbiStory contexts
    contexts = load_ambistory_contexts(args.ambistory)
    
    if not contexts:
        print("❌ Error: No contexts found in AmbiStory!")
        return
    
    # Add contexts to FEWS
    output_data = add_contexts_to_fews(args.fews_file, contexts, args.seed)
    
    # Validate
    validate_output(output_data)
    
    # Save
    print(f"\n{'='*70}")
    print(f"Saving to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved {len(output_data)} samples")
    print("=" * 70)


if __name__ == '__main__':
    main()
