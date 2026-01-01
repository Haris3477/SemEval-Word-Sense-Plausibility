#!/usr/bin/env python3
"""
Generate clean dataset from scratch:
1. Load AmbiStory (2,280 samples)
2. Extract from FEWS raw/ files ONLY (quotations, monosemous, examples)
3. Apply all cleaning and filtering:
   - Multi-sentence validation
   - Remove URLs, HTML entities, references
   - Ignore citation semicolons
   - Filter single-word only (no underscores)
   - Fix text formatting
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

random.seed(42)

def clean_reference_artifacts(text: str) -> str:
    """Remove URLs, HTML entities, wiki markup, and citation markers"""
    # Remove URLs
    text = re.sub(r'https?://[^\s\]]+', '', text)
    text = re.sub(r'www\.[^\s\]]+', '', text)
    
    # Remove HTML entities
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    
    # Remove wiki/reference markup
    text = re.sub(r'&#91;\[[^\]]*\]&#93;', '', text)
    text = re.sub(r'\[\[[^\]]*\]\]', '', text)
    text = re.sub(r'\{\{[^\}]*\}\}', '', text)
    
    # Remove standalone brackets/citation markers
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    
    # Clean up multiple spaces
    text = ' '.join(text.split())
    
    return text.strip()

def has_multiple_sentences(text: str) -> bool:
    """
    Check if text has actual sentence boundaries (not citations).
    Remove brackets/parens first, THEN check for boundaries.
    """
    # Remove citations/references first
    text_clean = re.sub(r'\([^)]*\)', '', text)
    text_clean = re.sub(r'\[[^\]]*\]', '', text_clean)
    text_clean = ' '.join(text_clean.split())
    
    # Check for period, exclamation, question (with optional ellipsis)
    if re.search(r'[.!?][\s.]+[A-Z]', text_clean):
        return True
    
    # Check for semicolon followed by capital
    if re.search(r';\s+[A-Z]', text_clean):
        return True
    
    return False

def parse_fews_line(line: str) -> Optional[Tuple[str, str, str, str, str]]:
    """Parse FEWS line and return (before, target, after, sense_id, full_text)"""
    line = line.strip()
    if not line or '<WSD>' not in line:
        return None
    
    parts = line.split('\t')
    if len(parts) < 2:
        return None
    
    full_text_with_tags = parts[0]
    sense_id = parts[1].strip()
    
    # Find FIRST <WSD> tag
    first_match = re.search(r'<WSD>(.*?)</WSD>', full_text_with_tags)
    if not first_match:
        return None
    
    target = first_match.group(1).strip()
    target_start = first_match.start()
    target_end = first_match.end()
    
    before_with_tags = full_text_with_tags[:target_start]
    after_with_tags = full_text_with_tags[target_end:]
    
    # Remove ALL <WSD> tags
    before = re.sub(r'</?WSD>', '', before_with_tags).strip()
    after = re.sub(r'</?WSD>', '', after_with_tags).strip()
    full_text_clean = re.sub(r'</?WSD>', '', full_text_with_tags).strip()
    
    return before, target, after, sense_id, full_text_clean

def load_senses(senses_path: str) -> Dict[str, str]:
    """Load FEWS sense definitions"""
    senses = {}
    current_entry = {}
    
    with open(senses_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                if 'sense_id' in current_entry and 'gloss' in current_entry:
                    senses[current_entry['sense_id']] = current_entry['gloss']
                current_entry = {}
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                current_entry[key.strip()] = value.strip()
        
        # Handle last entry
        if 'sense_id' in current_entry and 'gloss' in current_entry:
            senses[current_entry['sense_id']] = current_entry['gloss']
    
    return senses

def split_context(before: str, after: str, target: str) -> Tuple[str, str, str]:
    """Split before/after into precontext, sentence, ending"""
    sentence_boundary_pattern = r'[.!?;]\s+'
    
    before_sentences = re.split(f'({sentence_boundary_pattern})', before)
    after_sentences = re.split(f'({sentence_boundary_pattern})', after)
    
    precontext = ""
    ending = ""
    target_sentence = ""
    
    if len(before_sentences) >= 3:
        precontext = ''.join(before_sentences[:-1]).strip()
        last_before_fragment = before_sentences[-1].strip()
        
        if len(last_before_fragment) < 20 and after and after[0].islower():
            target_sentence = f"{last_before_fragment} {target} {after}".strip()
            ending = ""
        elif after_sentences and len(after_sentences) > 0:
            first_after = after_sentences[0].strip()
            target_sentence = f"{last_before_fragment} {target} {first_after}".strip()
            if len(after_sentences) > 1:
                ending = ''.join(after_sentences[1:]).strip()
        else:
            target_sentence = f"{last_before_fragment} {target}".strip()
    
    elif len(after_sentences) >= 3:
        first_after = after_sentences[0].strip()
        if before:
            target_sentence = f"{before} {target} {first_after}".strip()
        else:
            target_sentence = f"{target} {first_after}".strip() if first_after else f"The {target}."
        
        ending = ''.join(after_sentences[1:]).strip()
        precontext = ""
    
    else:
        if before and after:
            target_sentence = f"{before} {target} {after}".strip()
        elif before:
            target_sentence = f"{before} {target}".strip()
        elif after:
            target_sentence = f"{target} {after}".strip() if after and after[0].islower() else f"The {target} {after}".strip()
        else:
            target_sentence = f"The {target}."
        
        precontext = ""
        ending = ""
    
    # Clean up
    target_sentence = target_sentence.lstrip('.!?;: ')
    if target_sentence and target_sentence[0].islower():
        target_sentence = target_sentence[0].upper() + target_sentence[1:]
    
    precontext = ' '.join(precontext.split())
    target_sentence = ' '.join(target_sentence.split())
    ending = ' '.join(ending.split())
    
    return precontext, target_sentence, ending

def create_sample(before: str, target: str, after: str, sense_id: str, sense_def: str) -> Optional[Dict]:
    """Create a cleaned sample"""
    precontext, sentence, ending = split_context(before, after, target)
    
    # Clean all text fields
    precontext = clean_reference_artifacts(precontext)
    sentence = clean_reference_artifacts(sentence)
    ending = clean_reference_artifacts(ending)
    sense_def = clean_reference_artifacts(sense_def)
    
    # Fix formatting
    precontext = precontext.replace('\\"', '"').replace("\\'", "'")
    sentence = sentence.replace('\\"', '"').replace("\\'", "'")
    ending = ending.replace('\\"', '"').replace("\\'", "'")
    
    # Extract lemma
    lemma = sense_id.split('.')[0] if '.' in sense_id else target
    
    # Validate
    full_text = (precontext + ' ' + sentence + ' ' + ending).strip()
    
    # Must have multiple sentences
    if not has_multiple_sentences(full_text):
        return None
    
    # Sentence shouldn't start with punctuation
    if sentence and sentence[0] in '.,;:!?':
        return None
    
    # Sentence shouldn't be too short
    if not sentence or len(sentence) < 5:
        return None
    
    score = random.uniform(4.0, 5.0)
    
    return {
        "homonym": lemma,
        "precontext": precontext,
        "sentence": sentence,
        "ending": ending,
        "judged_meaning": sense_def,
        "plausibility_rating": round(score, 1)
    }

def score_context_quality(sample: dict) -> int:
    """
    Score a sample based on context richness for generating negatives.
    Higher score = better candidate (has full context).
    """
    precontext = sample.get('precontext', '').strip()
    sentence = sample.get('sentence', '').strip()
    ending = sample.get('ending', '').strip()
    
    # Must have all three non-empty
    if not precontext or not sentence or not ending:
        return 0
    
    # Score based on total context length (precontext + ending)
    context_length = len(precontext) + len(ending)
    
    return context_length

def create_negative_from_positive(positive_sample: dict, all_positive_samples: List[dict]) -> Optional[Dict]:
    """
    Create a negative sample by swapping in an incorrect sense definition.
    Uses a different sense for the same homonym.
    """
    homonym = positive_sample['homonym']
    
    # Find other samples with the same homonym but different definition
    same_homonym = [s for s in all_positive_samples 
                    if s['homonym'] == homonym 
                    and s['judged_meaning'] != positive_sample['judged_meaning']]
    
    if not same_homonym:
        return None
    
    # Pick a random incorrect definition
    wrong_sample = random.choice(same_homonym)
    
    # Create negative sample with wrong definition
    negative = positive_sample.copy()
    negative['judged_meaning'] = wrong_sample['judged_meaning']
    negative['plausibility_rating'] = round(random.uniform(1.0, 2.5), 1)
    
    return negative

def extract_from_fews_file(file_path: str, senses: Dict[str, str]) -> List[Dict]:
    """Extract clean samples from a FEWS file and generate negative samples"""
    samples = []
    
    # Group senses by lemma for negative generation
    lemma_to_senses = defaultdict(list)
    for sense_id in senses.keys():
        lemma = sense_id.split('.')[0]
        if '_' not in lemma:  # Only single-word
            lemma_to_senses[lemma].append(sense_id)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            parsed = parse_fews_line(line)
            if not parsed:
                continue
            
            before, target, after, sense_id, full_text = parsed
            
            # Filter multi-word terms (underscores)
            lemma = sense_id.split('.')[0] if '.' in sense_id else target
            if '_' in lemma:
                continue
            
            # Get CORRECT sense definition
            correct_sense_def = senses.get(sense_id)
            if not correct_sense_def:
                continue
            
            # Create POSITIVE sample (correct sense)
            positive_sample = create_sample(before, target, after, sense_id, correct_sense_def)
            if positive_sample:
                samples.append(positive_sample)
                
                # Generate NEGATIVE sample (incorrect sense) 50% of the time
                if random.random() < 0.5 and lemma in lemma_to_senses:
                    # Get alternative senses for this lemma
                    alternative_senses = [s for s in lemma_to_senses[lemma] if s != sense_id]
                    
                    if alternative_senses:
                        # Pick a random incorrect sense
                        wrong_sense_id = random.choice(alternative_senses)
                        wrong_sense_def = senses[wrong_sense_id]
                        
                        # Create negative sample with wrong definition and low score
                        negative_sample = create_sample(before, target, after, wrong_sense_id, wrong_sense_def)
                        if negative_sample:
                            # Override score to be negative
                            negative_sample['plausibility_rating'] = round(random.uniform(1.0, 3.0), 1)
                            samples.append(negative_sample)
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
    
    return samples

def main():
    print("="*80)
    print("GENERATING CLEAN DATASET FROM SCRATCH")
    print("="*80)
    
    # Load AmbiStory
    print("\n1. Loading AmbiStory train.json...")
    with open('data/train.json', 'r', encoding='utf-8') as f:
        ambistory_data = json.load(f)
    
    ambistory_samples = []
    for key, value in ambistory_data.items():
        if isinstance(value, dict) and 'homonym' in value:
            ambistory_samples.append(value)
    
    print(f"   ✓ Loaded {len(ambistory_samples):,} AmbiStory samples")
    
    # Load FEWS senses
    print("\n2. Loading FEWS senses...")
    senses = load_senses('data/fews/fews/senses.txt')
    print(f"   ✓ Loaded {len(senses):,} sense definitions")
    
    # Extract from FEWS raw/ files ONLY
    print("\n3. Extracting from FEWS raw/ files:")
    
    raw_files = {
        'quotations': 'data/fews/fews/raw/quotations.txt',
        'monosemous': 'data/fews/fews/raw/monosemous.txt',
        'examples': 'data/fews/fews/raw/examples.txt'
    }
    
    all_fews_samples = []
    
    for name, path in raw_files.items():
        print(f"\n   Processing {name}.txt...")
        samples = extract_from_fews_file(path, senses)
        print(f"   ✓ Extracted {len(samples):,} clean multi-sentence samples")
        all_fews_samples.extend(samples)
    
    print(f"\n   Total FEWS samples: {len(all_fews_samples):,}")
    
    # Count positive/negative in FEWS
    fews_pos = sum(1 for s in all_fews_samples if s['plausibility_rating'] > 3.0)
    fews_neg = len(all_fews_samples) - fews_pos
    print(f"   ✓ Positive: {fews_pos:,} ({fews_pos/len(all_fews_samples)*100:.1f}%)")
    print(f"   ✓ Negative: {fews_neg:,} ({fews_neg/len(all_fews_samples)*100:.1f}%)")
    
    # Sample to create balanced 10K dataset
    print("\n4. Creating balanced 10K dataset...")
    # We have 2,280 AmbiStory (1,181 pos, 1,099 neg)
    # Need ~8,000 FEWS to reach 10K total
    # Target: ~50% positive overall = ~5,140 positive total
    # We have 1,181 pos from AmbiStory, need ~3,960 pos from FEWS
    # We have 1,099 neg from AmbiStory, need ~4,040 neg from FEWS
    
    fews_positive = [s for s in all_fews_samples if s['plausibility_rating'] > 3.0]
    fews_negative = [s for s in all_fews_samples if s['plausibility_rating'] <= 3.0]
    
    random.shuffle(fews_positive)
    random.shuffle(fews_negative)
    
    # Sample to balance
    fews_pos_sample = fews_positive[:3960]
    fews_neg_sample = fews_negative[:4040]
    fews_10k = fews_pos_sample + fews_neg_sample
    
    print(f"   ✓ Sampled {len(fews_pos_sample):,} positive FEWS")
    print(f"   ✓ Sampled {len(fews_neg_sample):,} negative FEWS")
    
    # Combine for 10K
    print("\n5. Combining for 10K dataset...")
    combined_10k = ambistory_samples + fews_10k
    random.shuffle(combined_10k)
    
    total_pos_10k = sum(1 for s in combined_10k if (s.get('average', s.get('plausibility_rating', 0)) > 3.0))
    total_neg_10k = len(combined_10k) - total_pos_10k
    
    print(f"   ✓ Total: {len(combined_10k):,}")
    print(f"   ✓ Positive: {total_pos_10k:,} ({total_pos_10k/len(combined_10k)*100:.1f}%)")
    print(f"   ✓ Negative: {total_neg_10k:,} ({total_neg_10k/len(combined_10k)*100:.1f}%)")
    
    # Save 10K
    result_10k = {str(i): s for i, s in enumerate(combined_10k)}
    output_10k = 'Experiment-1/final-ish/combined_clean_10k_balanced.json'
    print(f"\n6. Saving 10K to {output_10k}...")
    with open(output_10k, 'w', encoding='utf-8') as f:
        json.dump(result_10k, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved {len(result_10k):,} samples")
    
    # Create full balanced dataset
    print("\n7. Creating full balanced dataset...")
    # Target 50% positive overall
    # Need equal positive and negative
    max_samples = min(len(fews_positive), len(fews_negative)) * 2  # Balanced
    
    fews_pos_full = fews_positive[:len(fews_negative)]
    fews_neg_full = fews_negative[:len(fews_positive)]
    
    combined_full = ambistory_samples + fews_pos_full + fews_neg_full
    random.shuffle(combined_full)
    
    total_pos_full = sum(1 for s in combined_full if (s.get('average', s.get('plausibility_rating', 0)) > 3.0))
    total_neg_full = len(combined_full) - total_pos_full
    
    print(f"   ✓ Total: {len(combined_full):,}")
    print(f"   ✓ Positive: {total_pos_full:,} ({total_pos_full/len(combined_full)*100:.1f}%)")
    print(f"   ✓ Negative: {total_neg_full:,} ({total_neg_full/len(combined_full)*100:.1f}%)")
    
    # Save full
    result_full = {str(i): s for i, s in enumerate(combined_full)}
    output_full = 'Experiment-1/final-ish/combined_clean_full_balanced.json'
    print(f"\n8. Saving full to {output_full}...")
    with open(output_full, 'w', encoding='utf-8') as f:
        json.dump(result_full, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved {len(result_full):,} samples")
    
    print(f"   ✓ Saved {len(result_full):,} samples")
    
    # Show examples
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES FROM 10K:")
    print("="*80)
    import pprint
    pprint.pprint([result_10k[str(i)] for i in range(3)])
    
    print("\n✓ Dataset generation complete!")

if __name__ == '__main__':
    main()
