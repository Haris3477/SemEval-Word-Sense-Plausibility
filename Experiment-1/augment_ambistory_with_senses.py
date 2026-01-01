#!/usr/bin/env python
"""
Augment AmbiStory dataset by generating sense-based variations.

This script takes the original AmbiStory training data and creates synthetic
variations by:
1. Looking up all dictionary senses for each homonym
2. Creating negative samples with wrong senses (low scores)
3. Creating positive samples with correct/plausible senses (high scores)

The output format matches AmbiStory exactly, so it can be used as additional
training data.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    print("⚠️  WordNet not available. Install with: pip install nltk")
    print("   Then run: python -c \"import nltk; nltk.download('wordnet')\"")


class AmbiStoryAugmenter:
    """Augments AmbiStory data with sense-based variations."""
    
    def __init__(self, fews_dir: str = None, seed: int = 42):
        """
        Initialize augmenter.
        
        Args:
            fews_dir: Optional path to FEWS directory (only uses senses.txt for definitions)
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.fews_senses = {}
        
        # Load FEWS sense inventory if available (richer than WordNet)
        if fews_dir and Path(fews_dir).exists():
            print(f"📖 Loading FEWS sense inventory from {fews_dir}")
            self._load_fews_senses(fews_dir)
            print(f"   ✓ Loaded {len(self.fews_senses)} unique lemmas")
            print(f"   ✓ Loaded {sum(len(v) for v in self.fews_senses.values())} total sense definitions")
        elif WORDNET_AVAILABLE:
            print("📖 Using WordNet for sense inventory")
        else:
            print("❌ No sense inventory available!")
            raise ValueError("Need either FEWS directory or WordNet installed")
    
    def _load_fews_senses(self, fews_dir: str):
        """Load FEWS senses.txt file for sense definitions."""
        senses_path = Path(fews_dir) / 'senses.txt'
        if not senses_path.exists():
            print(f"   ⚠️  senses.txt not found at {senses_path}")
            return
        
        current_entry = {}
        with open(senses_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Empty line signals end of entry
                if not line:
                    if current_entry and 'sense_id' in current_entry and 'gloss' in current_entry:
                        sense_id = current_entry['sense_id']
                        lemma = current_entry.get('word', sense_id.rsplit('.', 2)[0])
                        
                        if lemma not in self.fews_senses:
                            self.fews_senses[lemma] = []
                        
                        self.fews_senses[lemma].append({
                            'sense_id': sense_id,
                            'gloss': current_entry['gloss'],
                            'pos': sense_id.split('.')[-2] if '.' in sense_id else 'unknown'
                        })
                    current_entry = {}
                    continue
                
                # Parse key:value pairs
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    current_entry[key] = value
        
        # Handle last entry if file doesn't end with blank line
        if current_entry and 'sense_id' in current_entry and 'gloss' in current_entry:
            sense_id = current_entry['sense_id']
            lemma = current_entry.get('word', sense_id.rsplit('.', 2)[0])
            
            if lemma not in self.fews_senses:
                self.fews_senses[lemma] = []
            
            self.fews_senses[lemma].append({
                'sense_id': sense_id,
                'gloss': current_entry['gloss'],
                'pos': sense_id.split('.')[-2] if '.' in sense_id else 'unknown'
            })


    
    def get_senses(self, homonym: str) -> List[Dict[str, str]]:
        """
        Get all senses for a homonym.
        
        Returns:
            List of dicts with keys: sense_id, gloss, pos
        """
        # Try FEWS first (better coverage of rare words)
        if homonym.lower() in self.fews_senses:
            return self.fews_senses[homonym.lower()]
        
        # Fallback to WordNet
        if WORDNET_AVAILABLE:
            synsets = wn.synsets(homonym.lower())
            if not synsets:
                return []
            
            senses = []
            for idx, synset in enumerate(synsets):
                pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adj', 'r': 'adv', 's': 'adj'}
                pos = pos_map.get(synset.pos(), 'unknown')
                
                senses.append({
                    'sense_id': f"{homonym.lower()}.{pos}.{idx}",
                    'gloss': synset.definition(),
                    'pos': pos
                })
            return senses
        
        return []
    
    def score_sense_match(self, original_meaning: str, sense_gloss: str) -> float:
        """
        Estimate how well a sense gloss matches the original judged meaning.
        
        Returns:
            Similarity score 0.0-1.0 (higher = better match)
        """
        # Simple keyword overlap heuristic
        orig_words = set(original_meaning.lower().split())
        sense_words = set(sense_gloss.lower().split())
        
        # Remove common stopwords
        stopwords = {'a', 'an', 'the', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by'}
        orig_words -= stopwords
        sense_words -= stopwords
        
        if not orig_words or not sense_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(orig_words & sense_words)
        union = len(orig_words | sense_words)
        
        return intersection / union if union > 0 else 0.0
    
    def augment_sample(
        self,
        sample: Dict,
        negatives_per_sample: int = None,
        positives_per_sample: int = None,
        include_original: bool = True
    ) -> List[Dict]:
        """
        Create augmented versions of a single AmbiStory sample.
        
        Args:
            sample: Original AmbiStory sample
            negatives_per_sample: How many wrong-sense samples to generate (None = all available)
            positives_per_sample: How many similar-sense samples to generate (None = all available)
            include_original: Whether to include the original sample
        
        Returns:
            List of augmented samples (original + variations)
        """
        homonym = sample['homonym']
        original_meaning = sample['judged_meaning']
        original_score = sample['average']
        
        # Get all senses
        all_senses = self.get_senses(homonym)
        
        if not all_senses:
            # No senses found - just return original
            return [sample] if include_original else []
        
        # Score each sense by similarity to original meaning
        sense_scores = []
        for sense in all_senses:
            similarity = self.score_sense_match(original_meaning, sense['gloss'])
            sense_scores.append((sense, similarity))
        
        # Sort by similarity (best match first)
        sense_scores.sort(key=lambda x: x[1], reverse=True)
        
        augmented = []
        
        # Include original if requested
        if include_original:
            augmented.append(sample)
        
        # Generate POSITIVE samples from high-similarity senses
        # Take from top matches (excluding the very top which is likely the original)
        # Lower threshold: 0.1 (10% word overlap) is reasonable for dictionary definitions
        positive_candidates = [s for s, sim in sense_scores[1:] if sim >= 0.1]
        
        # Use ALL available positives if not limited
        num_positives = len(positive_candidates) if positives_per_sample is None else min(positives_per_sample, len(positive_candidates))
        
        if num_positives > 0:
            selected_positives = self.rng.sample(positive_candidates, num_positives) if positives_per_sample else positive_candidates
            
            for sense in selected_positives:
                pos_sample = sample.copy()
                pos_sample['judged_meaning'] = sense['gloss']
                
                # High score with some variance (similar to original)
                # Keep close to original score
                base_score = max(3.5, original_score - 0.5)
                jitter = self.rng.choice([-0.3, -0.15, 0, 0.15, 0.3])
                pos_sample['average'] = max(1.0, min(5.0, base_score + jitter))
                
                pos_sample['source'] = 'ambistory-augmented-positive'
                pos_sample['augmented_sense_id'] = sense['sense_id']
                
                augmented.append(pos_sample)
        
        # Generate NEGATIVE samples from poor-matching senses
        # Take from the bottom of the similarity ranking
        # Very low threshold: < 5% word overlap = clearly wrong sense
        negative_candidates = [s for s, sim in sense_scores if sim < 0.05]
        
        if len(negative_candidates) < 1:
            # Not enough clear negatives - use all non-top matches with low similarity
            negative_candidates = [s for s, sim in sense_scores[1:] if sim < 0.1]
        
        # BALANCE ENFORCEMENT: Cap negatives to prevent overwhelming imbalance
        # Default: max 2 negatives per sample (balanced dataset)
        if negatives_per_sample is None:
            num_negatives = min(2, len(negative_candidates))
        else:
            num_negatives = min(negatives_per_sample, len(negative_candidates))
        
        if num_negatives > 0:
            selected_negatives = self.rng.sample(negative_candidates, num_negatives) if num_negatives < len(negative_candidates) else negative_candidates
            
            for sense in selected_negatives:
                # Create negative sample with wrong sense
                neg_sample = sample.copy()
                neg_sample['judged_meaning'] = sense['gloss']
                
                # Assign low score with some variance
                base_score = self.rng.uniform(1.2, 2.8)
                jitter = self.rng.choice([-0.3, -0.15, 0, 0.15, 0.3])
                neg_sample['average'] = max(1.0, min(5.0, base_score + jitter))
                
                # Mark as synthetic
                neg_sample['source'] = 'ambistory-augmented-negative'
                neg_sample['augmented_sense_id'] = sense['sense_id']
                
                augmented.append(neg_sample)
        
        # Optionally add some "plausible but not quite right" samples (MODERATE)
        # These are senses with moderate similarity (0.05-0.1 range)
        moderate_candidates = [s for s, sim in sense_scores if 0.05 <= sim < 0.1]
        
        if moderate_candidates and self.rng.random() < 0.3:  # 30% chance
            sense = self.rng.choice(moderate_candidates)
            mod_sample = sample.copy()
            mod_sample['judged_meaning'] = sense['gloss']
            
            # Medium score (2.5-3.5 range)
            base_score = self.rng.uniform(2.5, 3.5)
            jitter = self.rng.choice([-0.2, -0.1, 0, 0.1, 0.2])
            mod_sample['average'] = max(1.0, min(5.0, base_score + jitter))
            
            mod_sample['source'] = 'ambistory-augmented-moderate'
            mod_sample['augmented_sense_id'] = sense['sense_id']
            
            augmented.append(mod_sample)
        
        return augmented
    
    def augment_dataset(
        self,
        input_path: str,
        output_path: str,
        negatives_per_sample: int = None,
        positives_per_sample: int = None,
        include_original: bool = True,
        max_samples: int = None
    ):
        """
        Augment entire AmbiStory dataset.
        
        Args:
            input_path: Path to original train.json
            output_path: Path to save augmented data
            negatives_per_sample: Negatives per original sample (None = all available)
            positives_per_sample: Additional positive samples per original (None = all available)
            include_original: Whether to include original samples
            max_samples: Optional limit on samples to process
        """
        print(f"\n{'='*80}")
        print(f"AmbiStory Augmentation")
        print(f"{'='*80}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Negatives per sample: {negatives_per_sample if negatives_per_sample is not None else 'ALL AVAILABLE'}")
        print(f"Positives per sample: {positives_per_sample if positives_per_sample is not None else 'ALL AVAILABLE'}")
        print(f"Include original: {include_original}")
        
        # Load original data
        with open(input_path, 'r', encoding='utf-8') as f:
            original_dict = json.load(f)
        
        # Convert dict to list of samples
        original_data = list(original_dict.values())
        
        if max_samples:
            original_data = original_data[:max_samples]
            print(f"Processing: {len(original_data)} samples (limited)")
        else:
            print(f"Processing: {len(original_data)} samples")
        
        # Augment each sample
        augmented_data = []
        skipped = 0
        
        for i, sample in enumerate(original_data):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(original_data)} samples processed...")
            
            variations = self.augment_sample(
                sample,
                negatives_per_sample=negatives_per_sample,
                positives_per_sample=positives_per_sample,
                include_original=include_original
            )
            
            if len(variations) == 1 and not include_original:
                # Only got original back, no senses found
                skipped += 1
            
            augmented_data.extend(variations)
        
        print(f"\n{'='*80}")
        print(f"Augmentation Complete")
        print(f"{'='*80}")
        print(f"Original samples: {len(original_data)}")
        print(f"Augmented samples: {len(augmented_data)}")
        print(f"Expansion factor: {len(augmented_data)/len(original_data):.2f}x")
        print(f"Skipped (no senses): {skipped}")
        
        # Analyze score distribution
        scores = [s['average'] for s in augmented_data]
        print(f"\nScore distribution:")
        print(f"  Mean: {sum(scores)/len(scores):.2f}")
        print(f"  Min: {min(scores):.2f}")
        print(f"  Max: {max(scores):.2f}")
        print(f"  Std: {(sum((x-sum(scores)/len(scores))**2 for x in scores)/len(scores))**0.5:.2f}")
        
        # Count sources
        sources = defaultdict(int)
        for s in augmented_data:
            sources[s.get('source', 'original')] += 1
        
        print(f"\nSample sources:")
        for source, count in sorted(sources.items()):
            print(f"  {source}: {count}")
        
        # Save augmented data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved to: {output_path}")
        
        # Print sample augmentations
        print(f"\n{'='*80}")
        print(f"Sample Augmentations (first 3 originals)")
        print(f"{'='*80}")
        
        shown = 0
        for sample in original_data[:10]:
            variations = self.augment_sample(
                sample, 
                negatives_per_sample, 
                positives_per_sample,
                include_original=False
            )
            
            if len(variations) > 0:
                print(f"\nOriginal:")
                print(f"  Homonym: '{sample['homonym']}'")
                print(f"  Original meaning: '{sample['judged_meaning'][:80]}...'")
                print(f"  Score: {sample['average']:.2f}")
                
                print(f"\n  Generated {len(variations)} variations:")
                for var in variations:
                    print(f"    → {var['source']}: '{var['judged_meaning'][:70]}...' (score: {var['average']:.2f})")
                
                shown += 1
                if shown >= 3:
                    break


def main():
    parser = argparse.ArgumentParser(
        description="Augment AmbiStory dataset with sense-based variations using FEWS sense inventory"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/train.json',
        help='Path to original AmbiStory train.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='train_augmented.json',
        help='Path to save augmented dataset'
    )
    parser.add_argument(
        '--fews_dir',
        type=str,
        default='data/fews/fews',
        help='Path to FEWS directory (uses senses.txt for sense definitions)'
    )
    parser.add_argument(
        '--negatives',
        type=int,
        default=2,  # Changed from None to 2 for balanced default
        help='Number of negative (wrong-sense) samples per original (default: 2 for balance)'
    )
    parser.add_argument(
        '--positives',
        type=int,
        default=None,
        help='Number of positive (similar-sense) samples per original (default: None = use all available)'
    )
    parser.add_argument(
        '--include_original',
        action='store_true',
        default=True,
        help='Include original samples in output'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Limit number of samples to process (for testing)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Initialize augmenter with FEWS directory (loads all files)
    fews_dir = args.fews_dir if Path(args.fews_dir).exists() else None
    augmenter = AmbiStoryAugmenter(fews_dir=fews_dir, seed=args.seed)
    
    # Run augmentation
    augmenter.augment_dataset(
        input_path=args.input,
        output_path=args.output,
        negatives_per_sample=args.negatives,
        positives_per_sample=args.positives,
        include_original=args.include_original,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
