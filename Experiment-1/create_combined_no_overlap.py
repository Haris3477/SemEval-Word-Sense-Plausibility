#!/usr/bin/env python3
"""
Create combined dataset WITHOUT using overlapping raw data.
Strategy: Only use train/dev/test files + monosemous.txt (which has <0.1% overlap)
EXCLUDE quotations.txt and examples.txt to avoid data leakage.
"""

import sys
sys.path.insert(0, 'Experiment-1')
from create_combined_multisent import *

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create combined dataset without raw overlap')
    parser.add_argument('--ambistory', default='data/train.json')
    parser.add_argument('--senses', default='data/fews/fews/senses.txt')
    parser.add_argument('--fews_dir', default='data/fews/fews')
    parser.add_argument('--output', default='Experiment-1/combined_no_overlap.json')
    parser.add_argument('--max_per_file', type=int, default=None)
    
    args = parser.parse_args()
    
    print("="*70)
    print("Creating Combined Dataset WITHOUT Raw File Overlap")
    print("="*70)
    print("EXCLUDING: quotations.txt (36.8% overlap with train)")
    print("EXCLUDING: examples.txt (82.7% overlap with train.ext)")
    print("INCLUDING: monosemous.txt (<0.1% overlap - safe to use)")
    print("="*70)
    
    # Load AmbiStory
    print(f"\nLoading AmbiStory data from {args.ambistory}...")
    with open(args.ambistory, 'r', encoding='utf-8') as f:
        ambistory_data = json.load(f)
    
    ambistory_samples = []
    if isinstance(ambistory_data, dict):
        for key, value in ambistory_data.items():
            if isinstance(value, dict):
                sample = {
                    "homonym": value.get("homonym", ""),
                    "precontext": value.get("precontext", ""),
                    "sentence": value.get("sentence", ""),
                    "ending": value.get("ending", ""),
                    "judged_meaning": value.get("judged_meaning", ""),
                    "plausibility_rating": value.get("average", 3.0)
                }
                ambistory_samples.append(sample)
    else:
        ambistory_samples = ambistory_data
    
    print(f"  Loaded {len(ambistory_samples)} AmbiStory samples")
    
    # Load FEWS senses
    print(f"\nLoading FEWS senses from {args.senses}...")
    senses = load_senses(args.senses)
    print(f"  Loaded {len(senses)} sense definitions")
    
    # Files to use (NO overlap issues)
    fews_files = [
        'train/train.ext.txt',          # 23.5% multi-sentence, 101K samples
        'raw/monosemous.txt',           # 21% multi-sentence, 132K samples, <0.1% overlap
        'test/test.few-shot.txt',       # 27.5% multi-sentence, 5K samples
        'test/test.zero-shot.txt',      # 23.5% multi-sentence, 5K samples
        'dev/dev.few-shot.txt',         # 18.5% multi-sentence, 5K samples
        'dev/dev.zero-shot.txt',        # 19.5% multi-sentence, 5K samples
    ]
    
    print(f"\n{'='*70}")
    print("Extracting multi-sentence samples...")
    print(f"{'='*70}")
    
    all_fews_samples = []
    for fews_file in fews_files:
        file_path = Path(args.fews_dir) / fews_file
        if not file_path.exists():
            print(f"  Warning: {file_path} not found, skipping")
            continue
        
        samples = extract_multisent_from_file(str(file_path), senses, args.max_per_file, single_word_only=True)
        all_fews_samples.extend(samples)
    
    print(f"\n{'='*70}")
    print("Dataset Summary:")
    print(f"{'='*70}")
    print(f"AmbiStory samples:     {len(ambistory_samples):6,}")
    print(f"FEWS multi-sent:       {len(all_fews_samples):6,}")
    print(f"{'='*70}")
    print(f"Total combined:        {len(ambistory_samples) + len(all_fews_samples):6,}")
    print(f"{'='*70}")
    
    combined = ambistory_samples + all_fews_samples
    random.shuffle(combined)
    
    # Convert to dict format
    output_dict = {}
    for idx, sample in enumerate(combined):
        output_dict[str(idx)] = sample
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == '__main__':
    main()
