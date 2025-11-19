#!/usr/bin/env python3
"""
Generate balanced dataset:
1. Extract POSITIVE samples with full context from FEWS
2. Generate EQUAL number of NEGATIVE samples from same contexts
3. Combine with AmbiStory
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

random.seed(42)

# [Keep all the helper functions from before: clean_reference_artifacts, has_multiple_sentences, 
#  parse_fews_line, load_senses, split_context, create_sample - COPY FROM LINES 1-220 of backup]

def clean_reference_artifacts(text: str) -> str:
    """Remove URLs, HTML entities, wiki markup"""
    text = re.sub(r'https?://[^\s\]]+', '', text)
    text = re.sub(r'www\.[^\s\]]+', '', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'&#91;\[[^\]]*\]&#93;', '', text)
    text = re.sub(r'\[\[[^\]]*\]\]', '', text)
    text = re.sub(r'\{\{[^\}]*\}\}', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    return ' '.join(text.split()).strip()

def has_multiple_sentences(text: str) -> bool:
    """Check if text has sentence boundaries after removing citations and titles"""
    # Remove citations/references first
    text_clean = re.sub(r'\([^)]*\)', '', text)
    text_clean = re.sub(r'\[[^\]]*\]', '', text_clean)
    text_clean = ' '.join(text_clean.split())
    
    # Remove common abbreviations/titles that end with period before checking
    # This prevents "Mr. Smith" from being detected as two sentences
    abbreviations = [
        r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Ave|Blvd|Rd|Inc|Ltd|Corp|Co|vs|Vol|No|Fig|Rev|Hon|Capt|Lt|Gen|Col|Sgt|Maj)\.',
        r'\b(Ph\.D|B\.A|M\.A|B\.S|M\.S|D\.D\.S|M\.D|Ph|BA|MA|BS|MS)\.',
        r'\b([ap]\.m\.|[AP]\.M\.)',  # Time
        r'\b(etc|i\.e|e\.g|viz|cf|ca|c|approx|dept|govt|univ)\.',  # Common
    ]
    
    for pattern in abbreviations:
        # Replace "abbr. " with "abbr " to prevent false sentence boundaries
        text_clean = re.sub(pattern + r'\s+', lambda m: m.group(0).replace('. ', ' '), text_clean, flags=re.IGNORECASE)
    
    # Check for period, exclamation, question (with optional ellipsis)
    if re.search(r'[.!?][\s.]+[A-Z]', text_clean):
        return True
    
    # Check for semicolon followed by capital
    if re.search(r';\s+[A-Z]', text_clean):
        return True
    
    return False

def parse_fews_line(line: str) -> Optional[Tuple[str, str, str, str, str]]:
    """Parse FEWS line"""
    line = line.strip()
    if not line or '<WSD>' not in line:
        return None
    
    parts = line.split('\t')
    if len(parts) < 2:
        return None
    
    full_text_with_tags = parts[0]
    sense_id = parts[1].strip()
    
    first_match = re.search(r'<WSD>(.*?)</WSD>', full_text_with_tags)
    if not first_match:
        return None
    
    target = first_match.group(1).strip()
    target_start = first_match.start()
    target_end = first_match.end()
    
    before_with_tags = full_text_with_tags[:target_start]
    after_with_tags = full_text_with_tags[target_end:]
    
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
        
        if 'sense_id' in current_entry and 'gloss' in current_entry:
            senses[current_entry['sense_id']] = current_entry['gloss']
    
    return senses

def split_context(before: str, after: str, target: str) -> Tuple[str, str, str]:
    """Split before/after into precontext, sentence, ending"""
    # First, protect common abbreviations/titles by temporarily replacing them
    def protect_abbreviations(text: str) -> str:
        """Replace abbreviations with placeholders"""
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Rev|Hon|Lt|Col|Gen|Capt)\.\s+', r'\1<ABBR> ', text)
        text = re.sub(r'\b([ap]\.m)\.\s+', r'\1<ABBR> ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.\s+', r'\1<ABBR> ', text)
        text = re.sub(r'\b([A-Z])\.\s+', r'\1<ABBR> ', text)  # Single letter abbreviations
        return text
    
    def restore_abbreviations(text: str) -> str:
        """Restore abbreviations from placeholders"""
        return text.replace('<ABBR>', '.')
    
    # Protect abbreviations before splitting
    before = protect_abbreviations(before)
    after = protect_abbreviations(after)
    
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
    
    # Restore abbreviations
    precontext = restore_abbreviations(precontext)
    target_sentence = restore_abbreviations(target_sentence)
    ending = restore_abbreviations(ending)
    
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
    
    precontext = clean_reference_artifacts(precontext)
    sentence = clean_reference_artifacts(sentence)
    ending = clean_reference_artifacts(ending)
    sense_def = clean_reference_artifacts(sense_def)
    
    precontext = precontext.replace('\\"', '"').replace("\\'", "'")
    sentence = sentence.replace('\\"', '"').replace("\\'", "'")
    ending = ending.replace('\\"', '"').replace("\\'", "'")
    
    lemma = sense_id.split('.')[0] if '.' in sense_id else target
    
    full_text = (precontext + ' ' + sentence + ' ' + ending).strip()
    
    if not has_multiple_sentences(full_text):
        return None
    
    if sentence and sentence[0] in '.,;:!?':
        return None
    
    if not sentence or len(sentence) < 5:
        return None
    
    score = random.uniform(4.0, 5.0)
    
    return {
        "homonym": lemma,
        "precontext": precontext,
        "sentence": sentence,
        "ending": ending,
        "judged_meaning": sense_def,
        "plausibility_rating": round(score, 1),
        "_sense_id": sense_id  # Keep for negative generation
    }

def extract_positive_samples(file_path: str, senses: Dict[str, str]) -> List[Dict]:
    """Extract POSITIVE samples only"""
    samples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            parsed = parse_fews_line(line)
            if not parsed:
                continue
            
            before, target, after, sense_id, full_text = parsed
            
            lemma = sense_id.split('.')[0] if '.' in sense_id else target
            if '_' in lemma:
                continue
            
            sense_def = senses.get(sense_id)
            if not sense_def:
                continue
            
            sample = create_sample(before, target, after, sense_id, sense_def)
            if sample:
                samples.append(sample)
        
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def generate_negatives_for_sample(positive_sample: Dict, senses: Dict[str, str], lemma_to_senses: Dict[str, List[str]], max_negatives: int = 3) -> List[Dict]:
    """Generate multiple negative samples by swapping wrong definitions"""
    homonym = positive_sample['homonym']
    current_sense_id = positive_sample['_sense_id']
    
    # Use pre-built mapping
    if homonym not in lemma_to_senses:
        return []
    
    homonym_senses = [sid for sid in lemma_to_senses[homonym] if sid != current_sense_id]
    
    if not homonym_senses:
        return []
    
    # Generate multiple negatives (up to max_negatives or number of alternative senses)
    num_negatives = min(max_negatives, len(homonym_senses))
    selected_wrong_senses = random.sample(homonym_senses, num_negatives)
    
    negatives = []
    for wrong_sense_id in selected_wrong_senses:
        wrong_def = senses[wrong_sense_id]
        
        # Match AmbiStory's negative distribution more closely
        # AmbiStory has more low scores (1.0-2.0) and fewer medium scores (2.0-3.0)
        rand = random.random()
        if rand < 0.6:  # 60% very negative (1.0-2.0)
            score = round(random.uniform(1.0, 2.0), 1)
        else:  # 40% moderately negative (2.0-3.0)
            score = round(random.uniform(2.0, 3.0), 1)
        
        negative = {
            'homonym': positive_sample['homonym'],
            'precontext': positive_sample['precontext'],
            'sentence': positive_sample['sentence'],
            'ending': positive_sample['ending'],
            'judged_meaning': wrong_def,
            'plausibility_rating': score
        }
        
        # Clean escaped quotes immediately in negatives
        for field in ['precontext', 'sentence', 'ending', 'judged_meaning']:
            negative[field] = negative[field].replace('\\"', '"').replace("\\'", "'")
        
        negatives.append(negative)
    
    return negatives

def main():
    print("="*80)
    print("GENERATING BALANCED DATASET")
    print("="*80)
    
    # Load AmbiStory
    print("\n1. Loading AmbiStory...")
    with open('data/train.json', 'r') as f:
        ambistory_data = json.load(f)
    
    ambistory_samples = [v for v in ambistory_data.values() if isinstance(v, dict) and 'homonym' in v]
    print(f"   ✓ {len(ambistory_samples):,} samples")
    
    # Load FEWS senses
    print("\n2. Loading FEWS senses...")
    senses = load_senses('data/fews/fews/senses.txt')
    print(f"   ✓ {len(senses):,} definitions")
    
    # Build lemma -> sense_ids mapping for fast lookup
    print("   Building lemma index...", end='')
    lemma_to_senses = {}
    for sense_id in senses.keys():
        lemma = sense_id.split('.')[0]
        if lemma not in lemma_to_senses:
            lemma_to_senses[lemma] = []
        lemma_to_senses[lemma].append(sense_id)
    print(f" {len(lemma_to_senses):,} lemmas")
    
    # Extract POSITIVE samples from raw/ files
    print("\n3. Extracting POSITIVE samples from FEWS raw/...")
    
    raw_files = {
        'quotations': 'data/fews/fews/raw/quotations.txt',
        'monosemous': 'data/fews/fews/raw/monosemous.txt',
        'examples': 'data/fews/fews/raw/examples.txt'
    }
    
    all_positive = []
    for name, path in raw_files.items():
        print(f"   {name}...", end='')
        samples = extract_positive_samples(path, senses)
        print(f" {len(samples):,}")
        all_positive.extend(samples)
    
    print(f"\n   Total POSITIVE: {len(all_positive):,}")
    
    # Filter for FULL CONTEXT (non-empty precontext + sentence + ending) for 10K
    # Exclude samples where context is just "..." (ellipsis placeholder)
    print("\n4a. Filtering for FULL CONTEXT (strict: all 3 fields non-empty)...")
    full_context_positive = [
        s for s in all_positive
        if (s['precontext'].strip() and s['precontext'].strip() != '...') and 
           s['sentence'].strip() and 
           (s['ending'].strip() and s['ending'].strip() != '...')
    ]
    
    print(f"    ✓ Strict filter: {len(full_context_positive):,} with full context")
    
    # Filter for GOOD CONTEXT (at least precontext OR ending, plus sentence) for FULL dataset
    # Exclude samples where context is just "..." (ellipsis placeholder)
    print("\n4b. Filtering for GOOD CONTEXT (relaxed: precontext OR ending)...")
    good_context_positive = [
        s for s in all_positive
        if s['sentence'].strip() and (
            (s['precontext'].strip() and s['precontext'].strip() != '...') or
            (s['ending'].strip() and s['ending'].strip() != '...')
        )
    ]
    
    print(f"    ✓ Relaxed filter: {len(good_context_positive):,} with good context")
    
    # Score by context length for both
    for s in full_context_positive:
        s['_ctx_score'] = len(s['precontext']) + len(s['ending'])
    
    for s in good_context_positive:
        s['_ctx_score'] = len(s.get('precontext', '')) + len(s.get('ending', ''))
    
    full_context_positive.sort(key=lambda s: s['_ctx_score'], reverse=True)
    good_context_positive.sort(key=lambda s: s['_ctx_score'], reverse=True)
    
    if full_context_positive:
        print(f"    Full context - Best: {full_context_positive[0]['_ctx_score']}, Median: {full_context_positive[len(full_context_positive)//2]['_ctx_score']}")
    if good_context_positive:
        print(f"    Good context - Best: {good_context_positive[0]['_ctx_score']}, Median: {good_context_positive[len(good_context_positive)//2]['_ctx_score']}")
    
    print(f"\n5. Summary of extracted positives:")
    print(f"   Full context (strict): {len(full_context_positive):,} positives")
    print(f"   Good context (relaxed): {len(good_context_positive):,} positives")
    
    # Filter to ONLY multi-sense homonyms (can generate negatives)
    print(f"\n6. Filtering for multi-sense homonyms (required for negatives)...")
    
    full_context_multi = [
        s for s in full_context_positive 
        if s['homonym'] in lemma_to_senses and len(lemma_to_senses[s['homonym']]) > 1
    ]
    
    good_context_multi = [
        s for s in good_context_positive 
        if s['homonym'] in lemma_to_senses and len(lemma_to_senses[s['homonym']]) > 1
    ]
    
    print(f"   Full context multi-sense: {len(full_context_multi):,} (can generate negatives)")
    print(f"   Good context multi-sense: {len(good_context_multi):,} (can generate negatives)")
    
    print(f"\n7. Cleaning up _ctx_score (keeping _sense_id for negative generation)...")
    # Clean _ctx_score but KEEP _sense_id for negative generation
    for s in full_context_positive + good_context_positive:
        s.pop('_ctx_score', None)
        
        # Fix escaped quotes in all text fields  
        for field in ['precontext', 'sentence', 'ending', 'judged_meaning']:
            if field in s:
                s[field] = s[field].replace('\\"', '"').replace("\\'", "'")
    
    print(f"   ✓ Cleaned positive samples (kept _sense_id)")
    
    print(f"\n8. Now generating datasets...")
    
    # Create 10K balanced (equal pos/neg)
    print("\n   Creating 10K balanced dataset...")
    print(f"   Target: 2,280 AmbiStory + ~4,000 FEWS pos + ~4,000 FEWS neg")
    
    # Sample 4,000 from multi-sense positives
    target_n = min(4000, len(full_context_multi))
    fews_10k_pos = random.sample(full_context_multi, target_n)
    
    # Generate EQUAL negatives from these exact samples
    print(f"   Generating {target_n} negatives from {target_n} positives...")
    fews_10k_neg = []
    for pos in fews_10k_pos:
        negs = generate_negatives_for_sample(pos, senses, lemma_to_senses, max_negatives=1)
        fews_10k_neg.extend(negs)
    
    combined_10k = ambistory_samples + fews_10k_pos + fews_10k_neg
    random.shuffle(combined_10k)
    
    pos_10k = sum(1 for s in combined_10k if (s.get('average', s.get('plausibility_rating', 0)) > 3.0))
    neg_10k = len(combined_10k) - pos_10k
    print(f"   Total: {len(combined_10k):,}")
    print(f"   Positive: {pos_10k:,} ({pos_10k/len(combined_10k)*100:.1f}%)")
    print(f"   Negative: {neg_10k:,} ({neg_10k/len(combined_10k)*100:.1f}%)")
    print(f"   Ratio: {pos_10k/neg_10k:.2f}:1")
    
    result_10k = {str(i): s for i, s in enumerate(combined_10k)}
    with open('Experiment-1/final-ish/combined_clean_10k_balanced.json', 'w') as f:
        json.dump(result_10k, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved to combined_clean_10k_balanced.json")
    
    # Create FULL balanced (strict: all full-context)
    print("\n   Creating FULL balanced dataset (strict full context)...")
    # Generate EQUAL negatives for ALL full-context multi-sense positives
    print(f"   Generating {len(full_context_multi):,} negatives from {len(full_context_multi):,} positives...")
    full_strict_negatives = []
    for pos in full_context_multi:
        negs = generate_negatives_for_sample(pos, senses, lemma_to_senses, max_negatives=1)
        full_strict_negatives.extend(negs)
    
    combined_full_strict = ambistory_samples + full_context_multi + full_strict_negatives
    random.shuffle(combined_full_strict)
    
    pos_full_strict = sum(1 for s in combined_full_strict if (s.get('average', s.get('plausibility_rating', 0)) > 3.0))
    neg_full_strict = len(combined_full_strict) - pos_full_strict
    print(f"   Total: {len(combined_full_strict):,}")
    print(f"   Positive: {pos_full_strict:,} ({pos_full_strict/len(combined_full_strict)*100:.1f}%)")
    print(f"   Negative: {neg_full_strict:,} ({neg_full_strict/len(combined_full_strict)*100:.1f}%)")
    print(f"   Ratio: {pos_full_strict/neg_full_strict:.2f}:1")
    
    result_full_strict = {str(i): s for i, s in enumerate(combined_full_strict)}
    with open('Experiment-1/final-ish/combined_clean_full_balanced.json', 'w') as f:
        json.dump(result_full_strict, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved to combined_clean_full_balanced.json")
    
    # Create FULL RELAXED balanced (relaxed: precontext OR ending)
    print("\n   Creating FULL RELAXED balanced dataset (relaxed context)...")
    # Generate EQUAL negatives for ALL good-context multi-sense positives
    print(f"   Generating {len(good_context_multi):,} negatives from {len(good_context_multi):,} positives...")
    full_relaxed_negatives = []
    for pos in good_context_multi:
        negs = generate_negatives_for_sample(pos, senses, lemma_to_senses, max_negatives=1)
        full_relaxed_negatives.extend(negs)
    
    combined_full_relaxed = ambistory_samples + good_context_multi + full_relaxed_negatives
    random.shuffle(combined_full_relaxed)
    
    pos_full_relaxed = sum(1 for s in combined_full_relaxed if (s.get('average', s.get('plausibility_rating', 0)) > 3.0))
    neg_full_relaxed = len(combined_full_relaxed) - pos_full_relaxed
    print(f"   Total: {len(combined_full_relaxed):,}")
    print(f"   Positive: {pos_full_relaxed:,} ({pos_full_relaxed/len(combined_full_relaxed)*100:.1f}%)")
    print(f"   Negative: {neg_full_relaxed:,} ({neg_full_relaxed/len(combined_full_relaxed)*100:.1f}%)")
    print(f"   Ratio: {pos_full_relaxed/neg_full_relaxed:.2f}:1")
    
    result_full_relaxed = {str(i): s for i, s in enumerate(combined_full_relaxed)}
    with open('Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json', 'w') as f:
        json.dump(result_full_relaxed, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved to combined_clean_full_relaxed_balanced.json")
    
    # Final cleanup: Remove _sense_id and normalize rating field
    print("\n   Final cleanup: Removing _sense_id and normalizing rating field...")
    for dataset in [result_10k, result_full_strict, result_full_relaxed]:
        for sample in dataset.values():
            sample.pop('_sense_id', None)
            # Normalize AmbiStory's 'average' to 'plausibility_rating'
            if 'average' in sample and 'plausibility_rating' not in sample:
                sample['plausibility_rating'] = sample.pop('average')
    
    # Re-save cleaned datasets
    with open('Experiment-1/final-ish/combined_clean_10k_balanced.json', 'w') as f:
        json.dump(result_10k, f, indent=2, ensure_ascii=False)
    with open('Experiment-1/final-ish/combined_clean_full_balanced.json', 'w') as f:
        json.dump(result_full_strict, f, indent=2, ensure_ascii=False)
    with open('Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json', 'w') as f:
        json.dump(result_full_relaxed, f, indent=2, ensure_ascii=False)
    print(f"   ✓ All _sense_id fields removed")
    
    print("\n" + "="*80)
    print("✓ DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"\n1. combined_clean_10k_balanced.json:          {len(combined_10k):>7,} samples ({pos_10k:,} pos, {neg_10k:,} neg)")
    print(f"2. combined_clean_full_balanced.json:         {len(combined_full_strict):>7,} samples ({pos_full_strict:,} pos, {neg_full_strict:,} neg)")
    print(f"3. combined_clean_full_relaxed_balanced.json: {len(combined_full_relaxed):>7,} samples ({pos_full_relaxed:,} pos, {neg_full_relaxed:,} neg)")

if __name__ == '__main__':
    main()

