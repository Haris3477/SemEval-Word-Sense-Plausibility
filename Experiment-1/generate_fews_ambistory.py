#!/usr/bin/env python
"""
Generate AmbiStory-style training data from FEWS sense inventory.

This script creates NEW training samples using FEWS words (not limited to AmbiStory vocabulary).
For each FEWS word with multiple senses:
1. Pick one sense as "correct" (high score 4-5)
2. Pick 2 other senses as "wrong" (low score 1-2)
3. Generate a simple context sentence using the word

This gives us potentially 100K+ training samples from FEWS's rich sense inventory.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


class FEWSAmbiStoryGenerator:
    """Generates AmbiStory-format data from FEWS sense inventory."""
    
    def __init__(self, fews_dir: str, seed: int = 42):
        self.rng = random.Random(seed)
        self.fews_senses = {}
        
        print(f"📖 Loading FEWS sense inventory from {fews_dir}")
        self._load_fews_senses(fews_dir)
        print(f"   ✓ Loaded {len(self.fews_senses)} unique lemmas")
        print(f"   ✓ Loaded {sum(len(v) for v in self.fews_senses.values())} total senses")
    
    def _load_fews_senses(self, fews_dir: str):
        """Load FEWS senses.txt file."""
        senses_path = Path(fews_dir) / 'senses.txt'
        if not senses_path.exists():
            raise FileNotFoundError(f"senses.txt not found at {senses_path}")
        
        current_entry = {}
        with open(senses_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    if current_entry and 'sense_id' in current_entry and 'gloss' in current_entry:
                        sense_id = current_entry['sense_id']
                        lemma = current_entry.get('word', sense_id.rsplit('.', 2)[0])
                        
                        if lemma not in self.fews_senses:
                            self.fews_senses[lemma] = []
                        
                        self.fews_senses[lemma].append({
                            'sense_id': sense_id,
                            'gloss': current_entry['gloss'],
                            'pos': sense_id.split('.')[-2] if '.' in sense_id else 'unknown',
                            'synonyms': current_entry.get('synonyms', ''),
                            'tags': current_entry.get('tags', '')
                        })
                    current_entry = {}
                    continue
                
                if ':' in line:
                    key, value = line.split(':', 1)
                    current_entry[key.strip()] = value.strip()
        
        # Handle last entry
        if current_entry and 'sense_id' in current_entry and 'gloss' in current_entry:
            sense_id = current_entry['sense_id']
            lemma = current_entry.get('word', sense_id.rsplit('.', 2)[0])
            
            if lemma not in self.fews_senses:
                self.fews_senses[lemma] = []
            
            self.fews_senses[lemma].append({
                'sense_id': sense_id,
                'gloss': current_entry['gloss'],
                'pos': sense_id.split('.')[-2] if '.' in sense_id else 'unknown',
                'synonyms': current_entry.get('synonyms', ''),
                'tags': current_entry.get('tags', '')
            })
    
    def generate_simple_sentence(self, word: str, pos: str) -> str:
        """Generate a simple sentence using the word."""
        # Simple templates based on POS
        templates = {
            'noun': [
                f"The {word} was important.",
                f"She looked at the {word}.",
                f"They found a {word}.",
                f"He talked about the {word}.",
            ],
            'verb': [
                f"She decided to {word}.",
                f"They will {word} soon.",
                f"He wants to {word}.",
                f"We should {word} now.",
            ],
            'adj': [
                f"It was very {word}.",
                f"The situation seemed {word}.",
                f"Everything felt {word}.",
                f"The result was {word}.",
            ],
            'adv': [
                f"She moved {word}.",
                f"He spoke {word}.",
                f"They acted {word}.",
                f"It happened {word}.",
            ],
        }
        
        pos_templates = templates.get(pos, templates['noun'])
        return self.rng.choice(pos_templates)
    
    def generate_samples(
        self,
        max_words: int = 10000,
        samples_per_word: int = 3,
        min_senses: int = 3
    ) -> List[Dict]:
        """
        Generate AmbiStory-format samples from FEWS.
        
        Args:
            max_words: Maximum number of words to process
            samples_per_word: How many samples per word (1 positive + N negatives)
            min_senses: Minimum number of senses required for a word
        
        Returns:
            List of AmbiStory-format samples
        """
        samples = []
        sample_id = 0
        
        # Filter words with enough senses AND single-word only (match AmbiStory)
        eligible_words = [
            (word, senses) for word, senses in self.fews_senses.items()
            if len(senses) >= min_senses and len(word.split()) == 1
        ]
        
        # Shuffle and limit
        self.rng.shuffle(eligible_words)
        eligible_words = eligible_words[:max_words]
        
        print(f"\n{'='*80}")
        print(f"Generating AmbiStory-format samples")
        print(f"{'='*80}")
        print(f"Eligible words (≥{min_senses} senses): {len(eligible_words)}")
        print(f"Samples per word: {samples_per_word}")
        print(f"Expected total: ~{len(eligible_words) * samples_per_word} samples")
        
        for word_idx, (word, senses) in enumerate(eligible_words):
            if (word_idx + 1) % 1000 == 0:
                print(f"  Progress: {word_idx+1}/{len(eligible_words)} words...")
            
            # Generate sentence
            pos = senses[0]['pos']
            sentence = self.generate_simple_sentence(word, pos)
            
            # Pick one sense as "correct"
            correct_sense = self.rng.choice(senses)
            
            # Create positive sample
            votes = [4, 5, 4, 5, 5]  # High scores
            jitter = [self.rng.choice([-0.2, -0.1, 0, 0.1, 0.2]) for _ in votes]
            votes_jittered = [max(1, min(5, v + j)) for v, j in zip(votes, jitter)]
            
            positive_sample = {
                'id': f'fews-{sample_id}',
                'sample_id': f'fews-{sample_id}',
                'homonym': word,
                'judged_meaning': correct_sense['gloss'],
                'precontext': '',
                'sentence': sentence,
                'ending': '',
                'choices': votes_jittered,
                'average': sum(votes_jittered) / len(votes_jittered),
                'stdev': 0.5,
                'nonsensical': [False] * len(votes_jittered),
                'example_sentence': correct_sense.get('synonyms', ''),
                'sense_tags': correct_sense.get('tags', ''),
                'source': 'fews-generated'
            }
            samples.append(positive_sample)
            sample_id += 1
            
            # Create negative samples (wrong senses)
            other_senses = [s for s in senses if s['sense_id'] != correct_sense['sense_id']]
            num_negatives = min(samples_per_word - 1, len(other_senses))
            
            if num_negatives > 0:
                selected_negatives = self.rng.sample(other_senses, num_negatives)
                
                for neg_sense in selected_negatives:
                    votes = [1, 2, 1, 2, 1]  # Low scores
                    jitter = [self.rng.choice([-0.2, -0.1, 0, 0.1, 0.2]) for _ in votes]
                    votes_jittered = [max(1, min(5, v + j)) for v, j in zip(votes, jitter)]
                    
                    negative_sample = {
                        'id': f'fews-{sample_id}',
                        'sample_id': f'fews-{sample_id}',
                        'homonym': word,
                        'judged_meaning': neg_sense['gloss'],
                        'precontext': '',
                        'sentence': sentence,
                        'ending': '',
                        'choices': votes_jittered,
                        'average': sum(votes_jittered) / len(votes_jittered),
                        'stdev': 0.5,
                        'nonsensical': [False] * len(votes_jittered),
                        'example_sentence': neg_sense.get('synonyms', ''),
                        'sense_tags': neg_sense.get('tags', ''),
                        'source': 'fews-generated'
                    }
                    samples.append(negative_sample)
                    sample_id += 1
        
        return samples
    
    def save_samples(self, samples: List[Dict], output_path: str):
        """Save samples in AmbiStory JSON format."""
        # Convert to dict format like AmbiStory
        output_dict = {str(i): sample for i, sample in enumerate(samples)}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"Generation Complete")
        print(f"{'='*80}")
        print(f"Total samples: {len(samples)}")
        
        # Analyze distribution
        sources = defaultdict(int)
        for s in samples:
            sources[s['source']] += 1
        
        print(f"\nSample sources:")
        for source, count in sorted(sources.items()):
            print(f"  {source}: {count}")
        
        print(f"\n✓ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate AmbiStory-format training data from FEWS sense inventory"
    )
    parser.add_argument(
        '--fews_dir',
        type=str,
        default='data/fews/fews',
        help='Path to FEWS directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='fews_ambistory_train.json',
        help='Path to save generated data'
    )
    parser.add_argument(
        '--max_words',
        type=int,
        default=10000,
        help='Maximum number of words to process'
    )
    parser.add_argument(
        '--samples_per_word',
        type=int,
        default=3,
        help='Samples per word (1 positive + N negatives)'
    )
    parser.add_argument(
        '--min_senses',
        type=int,
        default=3,
        help='Minimum senses required for a word'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = FEWSAmbiStoryGenerator(fews_dir=args.fews_dir, seed=args.seed)
    
    # Generate samples
    samples = generator.generate_samples(
        max_words=args.max_words,
        samples_per_word=args.samples_per_word,
        min_senses=args.min_senses
    )
    
    # Save
    generator.save_samples(samples, args.output)


if __name__ == '__main__':
    main()
