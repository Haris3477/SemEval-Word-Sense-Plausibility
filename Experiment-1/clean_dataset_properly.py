#!/usr/bin/env python3
"""
Properly clean the dataset:
1. Remove reference artifacts (URLs, HTML entities, wiki markup, citation markers)
2. Ignore citation semicolons (in parentheses/brackets)
3. Re-validate multi-sentence after cleaning
4. Fix text formatting
"""

import json
import re

def clean_reference_artifacts(text: str) -> str:
    """Remove URLs, HTML entities, wiki markup, and citation markers"""
    # Remove URLs (http://, https://, www.)
    text = re.sub(r'https?://[^\s\]]+', '', text)
    text = re.sub(r'www\.[^\s\]]+', '', text)
    
    # Remove HTML entities
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    
    # Remove wiki/reference markup
    text = re.sub(r'&#91;\[[^\]]*\]&#93;', '', text)  # &#91;[...]&#93;
    text = re.sub(r'\[\[[^\]]*\]\]', '', text)  # [[...]]
    text = re.sub(r'\{\{[^\}]*\}\}', '', text)  # {{...}}
    
    # Remove standalone brackets/citation markers that are now empty
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    
    # Clean up multiple spaces
    text = ' '.join(text.split())
    
    return text.strip()

def has_multiple_sentences(text: str) -> bool:
    """
    Check if text has actual sentence boundaries (not citations or titles).
    - Period, exclamation, question mark followed by space + capital (may have ellipsis ...)
    - OR semicolon OUTSIDE parentheses/brackets followed by space + capital
    - Ignores common titles: Mr., Mrs., Dr., etc.
    
    NOTE: Remove brackets/parens first, THEN check for boundaries
    """
    # FIRST: Remove brackets and parentheses (citations, references)
    text_clean = re.sub(r'\([^)]*\)', '', text)
    text_clean = re.sub(r'\[[^\]]*\]', '', text_clean)
    text_clean = ' '.join(text_clean.split())  # Clean up spaces
    
    # Remove common abbreviations/titles that end with period
    titles_pattern = r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Ave|Blvd|Rd|Etc|Inc|Ltd|Corp|Co|vs|Vol|No|Fig|Ph\.D|B\.A|M\.A|Ph|BA|MA)\.\s+[A-Z]'
    text_no_titles = re.sub(titles_pattern, lambda m: m.group(0).replace('. ', ' '), text_clean)
    
    # THEN: Check for period, exclamation, question mark (with optional ellipsis)
    # Pattern: [.!?] optionally followed by more dots/spaces, then capital letter
    if re.search(r'[.!?][\s.]+[A-Z]', text_no_titles):
        return True
    
    # Check for semicolon followed by capital letter
    if re.search(r';\s+[A-Z]', text_no_titles):
        return True
    
    return False

def clean_text_formatting(text: str) -> str:
    """Fix escaped quotes and other formatting issues"""
    # Replace escaped quotes with regular quotes
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    
    # Fix multiple spaces
    text = ' '.join(text.split())
    
    return text.strip()

def clean_sample(sample: dict) -> dict:
    """Clean all text fields in a sample"""
    cleaned = {}
    
    for field in ['precontext', 'sentence', 'ending', 'judged_meaning']:
        if field in sample:
            # First remove reference artifacts
            text = clean_reference_artifacts(sample[field])
            # Then fix formatting
            text = clean_text_formatting(text)
            cleaned[field] = text
    
    # Copy non-text fields as-is
    cleaned['homonym'] = sample['homonym']
    cleaned['plausibility_rating'] = sample['plausibility_rating']
    
    return cleaned

def is_valid_sample(sample: dict) -> tuple[bool, str]:
    """
    Check if sample is valid after cleaning.
    Returns (is_valid, reason)
    """
    # Combine all text
    full_text = (sample['precontext'] + ' ' + sample['sentence'] + ' ' + sample['ending']).strip()
    
    # Must have some content
    if len(full_text) < 10:
        return False, "too_short"
    
    # Must have multiple sentences
    if not has_multiple_sentences(full_text):
        return False, "single_sentence"
    
    # Sentence shouldn't start with punctuation
    if sample['sentence'] and sample['sentence'][0] in '.,;:!?':
        return False, "bad_punctuation"
    
    # Sentence shouldn't be empty or too short
    if not sample['sentence'] or len(sample['sentence']) < 5:
        return False, "empty_sentence"
    
    return True, "valid"

def main():
    import argparse
    from collections import defaultdict
    
    parser = argparse.ArgumentParser(description='Properly clean the dataset')
    parser.add_argument('--input', required=True, help='Input dataset')
    parser.add_argument('--output', required=True, help='Output dataset')
    parser.add_argument('--show_examples', type=int, default=3, help='Show N example cleaned samples')
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original samples: {len(data)}")
    
    # Track statistics
    cleaned_samples = []
    removal_reasons = defaultdict(int)
    
    # Track examples of what was cleaned
    cleaning_examples = []
    
    print("\nProcessing samples...")
    for idx, sample in data.items():
        original_sample = sample.copy()
        
        # Clean the sample
        cleaned = clean_sample(sample)
        
        # Check if valid after cleaning
        is_valid, reason = is_valid_sample(cleaned)
        
        if not is_valid:
            removal_reasons[reason] += 1
            continue
        
        # Track examples where cleaning made a difference
        if len(cleaning_examples) < 5:
            original_text = (original_sample['precontext'] + ' ' + original_sample['sentence'])[:200]
            cleaned_text = (cleaned['precontext'] + ' ' + cleaned['sentence'])[:200]
            if original_text != cleaned_text:
                cleaning_examples.append({
                    'homonym': cleaned['homonym'],
                    'original': original_text,
                    'cleaned': cleaned_text
                })
        
        cleaned_samples.append(cleaned)
    
    print(f"\nFinal samples: {len(cleaned_samples)}")
    print(f"\nRemoved samples breakdown:")
    for reason, count in sorted(removal_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason:20} {count:6,} samples")
    
    # Show examples of cleaned samples
    if cleaning_examples:
        print(f"\n{'='*80}")
        print("EXAMPLES OF CLEANING:")
        print('='*80)
        for i, ex in enumerate(cleaning_examples, 1):
            print(f"\n{i}. Homonym: '{ex['homonym']}'")
            print(f"   Original: {ex['original']}...")
            print(f"   Cleaned:  {ex['cleaned']}...")
    
    # Convert to dict format with numbered keys
    result = {}
    for idx, sample in enumerate(cleaned_samples):
        result[str(idx)] = sample
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("Done!")
    
    # Show some final samples
    if len(result) > 0 and args.show_examples > 0:
        print(f"\n{'='*80}")
        print(f"FINAL CLEANED SAMPLES (first {args.show_examples}):")
        print('='*80)
        import pprint
        pprint.pprint([result[str(i)] for i in range(min(args.show_examples, len(result)))])

if __name__ == '__main__':
    main()
