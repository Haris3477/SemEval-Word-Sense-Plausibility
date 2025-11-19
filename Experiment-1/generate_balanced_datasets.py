#!/usr/bin/env python3
"""
Generate balanced datasets:
- Keep ALL positive samples from FEWS
- Generate negatives from best context samples (non-empty precontext+sentence+ending)
- Prioritize longer contexts for negative generation
"""

import sys
sys.path.append('Experiment-1')

# Import everything from the original script
from generate_clean_dataset import *

def score_context_quality(sample: dict) -> int:
    """Score sample by context richness. Higher = better for negatives."""
    precontext = sample.get('precontext', '').strip()
    sentence = sample.get('sentence', '').strip()
    ending = sample.get('ending', '').strip()
    
    # Must have all three non-empty
    if not precontext or not sentence or not ending:
        return 0
    
    # Score = total context length
    return len(precontext) + len(ending)

def generate_negative_samples(positive_samples: List[dict], num_negatives: int) -> List[dict]:
    """
    Generate negative samples from best context positive samples.
    Swaps sense definitions for same homonym.
    """
    # Score all samples by context quality
    scored = [(score_context_quality(s), s) for s in positive_samples]
    scored = [(score, s) for score, s in scored if score > 0]  # Filter out bad contexts
    scored.sort(key=lambda x: x[0], reverse=True)  # Best first
    
    print(f"   Found {len(scored):,} samples with full context")
    print(f"   Top context score: {scored[0][0] if scored else 0}")
    print(f"   Median context score: {scored[len(scored)//2][0] if scored else 0}")
    
    # Group by homonym for negative generation
    from collections import defaultdict
    homonym_samples = defaultdict(list)
    for score, sample in scored:
        homonym_samples[sample['homonym']].append(sample)
    
    # Generate negatives from best contexts
    negatives = []
    candidates = [s for score, s in scored]  # Already sorted by quality
    
    for candidate in candidates:
        if len(negatives) >= num_negatives:
            break
        
        homonym = candidate['homonym']
        
        # Find other samples with same homonym but different meaning
        alternatives = [s for s in homonym_samples[homonym] 
                       if s['judged_meaning'] != candidate['judged_meaning']]
        
        if not alternatives:
            continue
        
        # Create negative by swapping definition
        wrong_def = random.choice(alternatives)['judged_meaning']
        
        negative = candidate.copy()
        negative['judged_meaning'] = wrong_def
        negative['plausibility_rating'] = round(random.uniform(1.0, 2.5), 1)
        
        negatives.append(negative)
    
    print(f"   ✓ Generated {len(negatives):,} negative samples from best contexts")
    
    return negatives

def main():
    print("="*80)
    print("GENERATING BALANCED DATASETS (KEEP ALL POSITIVES)")
    print("="*80)
    
    # Load AmbiStory
    print("\n1. Loading AmbiStory...")
    with open('data/train.json', 'r', encoding='utf-8') as f:
        ambistory_data = json.load(f)
    
    ambistory_samples = []
    for key, value in ambistory_data.items():
        if isinstance(value, dict) and 'homonym' in value:
            ambistory_samples.append(value)
    
    amb_pos = sum(1 for s in ambistory_samples if s.get('average', 0) > 3.0)
    amb_neg = len(ambistory_samples) - amb_pos
    
    print(f"   ✓ Total: {len(ambistory_samples):,}")
    print(f"   ✓ Positive: {amb_pos:,}")
    print(f"   ✓ Negative: {amb_neg:,}")
    
    # Load FEWS senses
    print("\n2. Loading FEWS senses...")
    senses = load_senses('data/fews/fews/senses.txt')
    print(f"   ✓ Loaded {len(senses):,} definitions")
    
    # Extract ALL positive samples from FEWS
    print("\n3. Extracting ALL positive samples from FEWS raw/ files:")
    
    raw_files = {
        'quotations': 'data/fews/fews/raw/quotations.txt',
        'monosemous': 'data/fews/fews/raw/monosemous.txt',
        'examples': 'data/fews/fews/raw/examples.txt'
    }
    
    all_fews_positive = []
    
    for name, path in raw_files.items():
        print(f"\n   Processing {name}.txt...")
        samples = extract_from_fews_file(path, senses)
        print(f"   ✓ Extracted {len(samples):,} positive samples")
        all_fews_positive.extend(samples)
    
    print(f"\n   Total FEWS positive: {len(all_fews_positive):,}")
    
    # Generate negative samples from best contexts
    print("\n4. Generating negative samples from best context samples...")
    num_negatives_needed = len(all_fews_positive)  # Match positive count
    
    fews_negatives = generate_negative_samples(all_fews_positive, num_negatives_needed)
    
    # Create balanced FULL dataset
    print("\n5. Creating FULL balanced dataset...")
    full_dataset = ambistory_samples + all_fews_positive + fews_negatives
    random.shuffle(full_dataset)
    
    full_pos = sum(1 for s in full_dataset if s.get('average', s.get('plausibility_rating', 0)) > 3.0)
    full_neg = len(full_dataset) - full_pos
    
    print(f"   ✓ Total: {len(full_dataset):,}")
    print(f"   ✓ Positive: {full_pos:,} ({full_pos/len(full_dataset)*100:.1f}%)")
    print(f"   ✓ Negative: {full_neg:,} ({full_neg/len(full_dataset)*100:.1f}%)")
    
    # Save full
    result_full = {str(i): s for i, s in enumerate(full_dataset)}
    output_full = 'Experiment-1/final-ish/combined_clean_full_balanced.json'
    print(f"\n6. Saving FULL to {output_full}...")
    with open(output_full, 'w', encoding='utf-8') as f:
        json.dump(result_full, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved!")
    
    # Create balanced 10K subset
    print("\n7. Creating 10K balanced subset...")
    # Sample 4K positive and 4K negative from FEWS (+ 2280 AmbiStory)
    random.shuffle(all_fews_positive)
    random.shuffle(fews_negatives)
    
    fews_10k = all_fews_positive[:4000] + fews_negatives[:4000]
    dataset_10k = ambistory_samples + fews_10k
    random.shuffle(dataset_10k)
    
    pos_10k = sum(1 for s in dataset_10k if s.get('average', s.get('plausibility_rating', 0)) > 3.0)
    neg_10k = len(dataset_10k) - pos_10k
    
    print(f"   ✓ Total: {len(dataset_10k):,}")
    print(f"   ✓ Positive: {pos_10k:,} ({pos_10k/len(dataset_10k)*100:.1f}%)")
    print(f"   ✓ Negative: {neg_10k:,} ({neg_10k/len(dataset_10k)*100:.1f}%)")
    
    # Save 10K
    result_10k = {str(i): s for i, s in enumerate(dataset_10k)}
    output_10k = 'Experiment-1/final-ish/combined_clean_10k_balanced.json'
    print(f"\n8. Saving 10K to {output_10k}...")
    with open(output_10k, 'w', encoding='utf-8') as f:
        json.dump(result_10k, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved!")
    
    print("\n" + "="*80)
    print("✓ DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"\nFull dataset: {len(result_full):,} samples")
    print(f"10K dataset:  {len(result_10k):,} samples")

if __name__ == '__main__':
    main()
