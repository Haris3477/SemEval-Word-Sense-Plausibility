#!/usr/bin/env python3
"""
Create combined AmbiStory + FEWS multi-sentence dataset.
Only includes FEWS samples that have actual multiple sentences (. + space + Capital).
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

random.seed(42)

def has_multiple_sentences(text: str) -> bool:
    """Check if text has actual sentence boundaries (. ! ? ; followed by space and capital)"""
    sentence_boundary_pattern = r'[.!?;]\s+[A-Z]'
    return bool(re.search(sentence_boundary_pattern, text))

def parse_fews_line(line: str) -> Optional[Tuple[str, str, str, str, str]]:
    """
    Parse FEWS line format: text with <WSD>word</WSD>\tsense_id\tmetadata
    Returns: (before_text, target_word, after_text, sense_id, full_text_clean) or None
    
    Note: Some lines have multiple <WSD> tags. We use the FIRST occurrence as the target,
    and remove ALL <WSD> tags from the text to get clean context.
    """
    line = line.strip()
    if not line or '<WSD>' not in line:
        return None
    
    parts = line.split('\t')
    if len(parts) < 2:
        return None
    
    full_text_with_tags = parts[0]
    sense_id = parts[1].strip()
    
    # Find the FIRST <WSD> tag - this is our target
    first_match = re.search(r'<WSD>(.*?)</WSD>', full_text_with_tags)
    if not first_match:
        return None
    
    target = first_match.group(1).strip()
    target_start = first_match.start()
    target_end = first_match.end()
    
    # Get text before and after the FIRST <WSD> tag (still has other tags)
    before_with_tags = full_text_with_tags[:target_start]
    after_with_tags = full_text_with_tags[target_end:]
    
    # Remove ALL <WSD> and </WSD> tags from before/after text
    before = re.sub(r'</?WSD>', '', before_with_tags).strip()
    after = re.sub(r'</?WSD>', '', after_with_tags).strip()
    
    # Clean full text (remove all tags)
    full_text_clean = re.sub(r'</?WSD>', '', full_text_with_tags).strip()
    
    return before, target, after, sense_id, full_text_clean

def load_senses(senses_path: str) -> Dict[str, str]:
    """Load FEWS sense definitions from key:value format"""
    senses = {}
    current_entry = {}
    
    with open(senses_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Empty line = end of entry
            if not line:
                if 'sense_id' in current_entry and 'gloss' in current_entry:
                    sense_id = current_entry['sense_id']
                    gloss = current_entry['gloss']
                    senses[sense_id] = gloss
                current_entry = {}
                continue
            
            # Parse key:value
            if ':' in line:
                key, value = line.split(':', 1)
                current_entry[key.strip()] = value.strip()
        
        # Handle last entry if file doesn't end with blank line
        if 'sense_id' in current_entry and 'gloss' in current_entry:
            sense_id = current_entry['sense_id']
            gloss = current_entry['gloss']
            senses[sense_id] = gloss
    
    return senses

def split_context(before: str, after: str, target: str) -> Tuple[str, str, str]:
    """
    Split before/after text into precontext, target_sentence, ending.
    Strategy: Find sentence boundaries and ensure target_sentence is grammatical.
    
    The target word should appear naturally in a complete sentence.
    """
    # Try to find sentence boundaries (period/semicolon + space + capital letter)
    sentence_boundary_pattern = r'[.!?;]\s+'
    
    before_sentences = re.split(f'({sentence_boundary_pattern})', before)
    after_sentences = re.split(f'({sentence_boundary_pattern})', after)
    
    precontext = ""
    ending = ""
    target_sentence = ""
    
    # Strategy: Find the most natural sentence containing the target word
    # This depends on where we have the most context
    
    # If we have substantial before text ending with a sentence boundary
    if len(before_sentences) >= 3:
        # Everything except last sentence fragment goes to precontext
        precontext = ''.join(before_sentences[:-1]).strip()
        last_before_fragment = before_sentences[-1].strip()
        
        # If the last before fragment is short and after starts with lowercase,
        # it's likely the middle of a sentence
        if len(last_before_fragment) < 20 and after and after[0].islower():
            # Combine: last_before + target + after
            target_sentence = f"{last_before_fragment} {target} {after}".strip()
            ending = ""
        elif after_sentences and len(after_sentences) > 0:
            # Take first part of after
            first_after = after_sentences[0].strip()
            target_sentence = f"{last_before_fragment} {target} {first_after}".strip()
            # Rest goes to ending
            if len(after_sentences) > 1:
                ending = ''.join(after_sentences[1:]).strip()
        else:
            target_sentence = f"{last_before_fragment} {target}".strip()
    
    # If we have substantial after text with sentence boundaries
    elif len(after_sentences) >= 3:
        # First part becomes target sentence
        first_after = after_sentences[0].strip()
        if before:
            target_sentence = f"{before} {target} {first_after}".strip()
        else:
            target_sentence = f"{target} {first_after}".strip() if first_after else f"The {target}."
        
        # Rest goes to ending
        ending = ''.join(after_sentences[1:]).strip()
        precontext = ""
    
    # Default: single long sentence or unclear boundaries
    else:
        if before and after:
            target_sentence = f"{before} {target} {after}".strip()
        elif before:
            target_sentence = f"{before} {target}".strip()
        elif after:
            target_sentence = f"{target} {after}".strip() if after[0].islower() else f"The {target} {after}".strip()
        else:
            target_sentence = f"The {target}."
        
        precontext = ""
        ending = ""
    
    # Clean up: ensure target_sentence doesn't start with punctuation
    target_sentence = target_sentence.lstrip('.!?;: ')
    
    # If target_sentence starts with lowercase, capitalize it
    if target_sentence and target_sentence[0].islower():
        target_sentence = target_sentence[0].upper() + target_sentence[1:]
    
    # Clean up any extra whitespace
    precontext = ' '.join(precontext.split())
    target_sentence = ' '.join(target_sentence.split())
    ending = ' '.join(ending.split())
    
    return precontext, target_sentence, ending

def create_ambistory_sample(
    before: str, 
    target: str, 
    after: str, 
    sense_id: str, 
    sense_def: str,
    is_positive: bool = True
) -> Dict:
    """Create an AmbiStory-format sample from FEWS data"""
    
    # Split into precontext, sentence, ending
    precontext, sentence, ending = split_context(before, after, target)
    
    # Extract lemma from sense_id (e.g., "free.adjective.0" -> "free")
    lemma = sense_id.split('.')[0] if '.' in sense_id else target
    
    # Generate plausibility score
    if is_positive:
        score = random.uniform(4.0, 5.0)  # High score for correct sense
    else:
        score = random.uniform(1.0, 2.5)  # Low score for incorrect sense
    
    return {
        "homonym": lemma,
        "precontext": precontext,
        "sentence": sentence,
        "ending": ending,
        "judged_meaning": sense_def,
        "plausibility_rating": round(score, 1)
    }

def extract_multisent_from_file(
    file_path: str, 
    senses: Dict[str, str],
    max_samples: Optional[int] = None,
    single_word_only: bool = True
) -> List[Dict]:
    """Extract only multi-sentence samples from a FEWS file"""
    
    samples = []
    print(f"\nProcessing {Path(file_path).name}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        multi_count = 0
        single_count = 0
        multi_word_skipped = 0
        
        for line in lines:
            parsed = parse_fews_line(line)
            if not parsed:
                continue
            
            before, target, after, sense_id, full_text = parsed
            
            # Check if this has multiple sentences
            if not has_multiple_sentences(full_text):
                single_count += 1
                continue
            
            # Filter multi-word terms (with underscores) if requested
            if single_word_only:
                lemma = sense_id.split('.')[0] if '.' in sense_id else target
                if '_' in lemma:
                    multi_word_skipped += 1
                    continue
            
            multi_count += 1
            
            # Get sense definition
            sense_def = senses.get(sense_id)
            if not sense_def:
                # Skip if no definition found
                continue
            
            # Only create positive samples (correct sense)
            # We can add negative sampling later if needed
            sample = create_ambistory_sample(
                before, target, after, sense_id, sense_def, is_positive=True
            )
            
            # FINAL VALIDATION: Verify the constructed sample still has multiple sentences
            final_text = (sample['precontext'] + ' ' + sample['sentence'] + ' ' + sample['ending']).strip()
            if not has_multiple_sentences(final_text):
                single_count += 1  # This got collapsed to single sentence in split_context
                multi_count -= 1
                continue
            
            # Also check that sentence doesn't start with punctuation
            if sample['sentence'] and sample['sentence'][0] in '.,;:!?':
                continue
            
            samples.append(sample)
            
            if max_samples and len(samples) >= max_samples:
                break
        
        skip_msg = f", {multi_word_skipped} multi-word skipped" if single_word_only else ""
        print(f"  Found {multi_count} multi-sentence samples ({single_count} single-sentence skipped{skip_msg})")
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
    
    return samples

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create combined AmbiStory + FEWS multi-sentence dataset')
    parser.add_argument('--ambistory', default='data/train.json', help='AmbiStory training data')
    parser.add_argument('--senses', default='data/fews/fews/senses.txt', help='FEWS senses file')
    parser.add_argument('--fews_dir', default='data/fews/fews', help='FEWS directory')
    parser.add_argument('--output', default='Experiment-1/combined_multisent.json', help='Output file')
    parser.add_argument('--max_per_file', type=int, default=None, help='Max samples per FEWS file')
    parser.add_argument('--include_examples', action='store_true', help='Include examples.txt (only 4.5% multi-sent)')
    parser.add_argument('--allow_multiword', action='store_true', help='Allow multi-word terms (with _). Default: single-word only')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Creating Combined AmbiStory + FEWS Multi-Sentence Dataset")
    print("="*70)
    
    # Load AmbiStory data
    print(f"\nLoading AmbiStory data from {args.ambistory}...")
    with open(args.ambistory, 'r', encoding='utf-8') as f:
        ambistory_data = json.load(f)
    
    # Handle dict format with numbered keys
    if isinstance(ambistory_data, dict):
        ambistory_samples = []
        for key, value in ambistory_data.items():
            if isinstance(value, dict):
                # Create sample with required fields
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
    
    # Define FEWS files to process (in order of multi-sentence percentage)
    fews_files = [
        'raw/quotations.txt',           # 29% multi-sentence
        'test/test.few-shot.txt',       # 27.5%
        'train/train.ext.txt',          # 23.5%
        'test/test.zero-shot.txt',      # 23.5%
        'raw/monosemous.txt',           # 21%
        'dev/dev.zero-shot.txt',        # 19.5%
        'dev/dev.few-shot.txt',         # 18.5%
    ]
    
    if args.include_examples:
        fews_files.append('raw/examples.txt')  # 4.5%
    
    # Extract multi-sentence samples from each file
    all_fews_samples = []
    single_word_only = not args.allow_multiword
    
    if single_word_only:
        print(f"\n📌 Filtering: Single-word terms only (matching AmbiStory format)")
    else:
        print(f"\n📌 Including multi-word terms (lemmas with underscores)")
    
    for fews_file in fews_files:
        file_path = Path(args.fews_dir) / fews_file
        if not file_path.exists():
            print(f"  Warning: {file_path} not found, skipping")
            continue
        
        samples = extract_multisent_from_file(str(file_path), senses, args.max_per_file, single_word_only)
        all_fews_samples.extend(samples)
    
    # Combine datasets
    print(f"\n{'='*70}")
    print("Dataset Summary:")
    print(f"{'='*70}")
    print(f"AmbiStory samples:     {len(ambistory_samples):6,}")
    print(f"FEWS multi-sent:       {len(all_fews_samples):6,}")
    print(f"{'='*70}")
    print(f"Total combined:        {len(ambistory_samples) + len(all_fews_samples):6,}")
    print(f"{'='*70}")
    
    combined = ambistory_samples + all_fews_samples
    
    # Shuffle
    random.shuffle(combined)
    
    # Convert to AmbiStory format (numbered dict keys)
    output_dict = {}
    for idx, sample in enumerate(combined):
        output_dict[str(idx)] = sample
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Done! Created {len(combined):,} samples")
    
    # Show some statistics
    fews_ratio = len(all_fews_samples) / len(combined) * 100
    print(f"\nDataset composition:")
    print(f"  AmbiStory: {len(ambistory_samples)/len(combined)*100:.1f}%")
    print(f"  FEWS:      {fews_ratio:.1f}%")

if __name__ == '__main__':
    main()
