#!/usr/bin/env python3
"""
Fix existing dataset:
1. Remove single-sentence samples (where combined text doesn't have multiple sentences)
2. Fix escaped quotes and other formatting issues
"""

import json
import re

def has_multiple_sentences(text: str) -> bool:
    """
    Check if text has actual sentence boundaries, not citations.
    - Requires . ! ? followed by space + capital letter
    - OR semicolon outside parentheses/brackets followed by space + capital letter
    """
    # First check for period, exclamation, question mark
    if re.search(r'[.!?]\s+[A-Z]', text):
        return True
    
    # For semicolons, need to be more careful - ignore those in citations
    # Remove content in parentheses and brackets (citations)
    text_no_citations = re.sub(r'\([^)]*\)', '', text)
    text_no_citations = re.sub(r'\[[^\]]*\]', '', text_no_citations)
    
    # Now check for semicolon followed by capital letter
    if re.search(r';\s+[A-Z]', text_no_citations):
        return True
    
    return False

def clean_text(text: str) -> str:
    """Clean text by removing URLs, HTML entities, wiki markup, and fixing quotes"""
    # Remove URLs (anything with http/https/www)
    text = re.sub(r'https?://[^\s\]]+', '', text)
    text = re.sub(r'www\.[^\s\]]+', '', text)
    
    # Remove HTML entities like &#91; &#93;
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    
    # Remove wiki-style links [[...]]
    text = re.sub(r'\[\[[^\]]+\]\]', '', text)
    
    # Remove standalone brackets that might be left from URL removal
    text = re.sub(r'\[\s*\]', '', text)
    
    # Replace escaped quotes with regular quotes
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    
    # Fix any double spaces and extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def is_valid_sample(sample: dict) -> bool:
    """Check if sample is valid (multi-sentence and properly formatted)"""
    # Clean the sample first
    cleaned_sample = clean_sample(sample)
    
    # Combine all text
    full_text = (cleaned_sample['precontext'] + ' ' + cleaned_sample['sentence'] + ' ' + cleaned_sample['ending']).strip()
    
    # Must have multiple sentences in combined text AFTER cleaning
    if not has_multiple_sentences(full_text):
        return False
    
    # Sentence shouldn't start with punctuation
    if cleaned_sample['sentence'] and cleaned_sample['sentence'][0] in '.,;:!?':
        return False
    
    return True

def clean_sample(sample: dict) -> dict:
    """Clean a sample by removing URLs, HTML entities, and fixing text formatting"""
    return {
        'homonym': clean_text(sample['homonym']),
        'precontext': clean_text(sample['precontext']),
        'sentence': clean_text(sample['sentence']),
        'ending': clean_text(sample['ending']),
        'judged_meaning': clean_text(sample['judged_meaning']),
        'plausibility_rating': sample['plausibility_rating']
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix dataset issues')
    parser.add_argument('--input', required=True, help='Input dataset')
    parser.add_argument('--output', required=True, help='Output dataset')
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original samples: {len(data)}")
    
    # Process samples
    cleaned_samples = []
    removed_single_sent = 0
    removed_bad_format = 0
    cleaned_text = 0
    
    for idx, sample in data.items():
        # Check if valid first
        if not is_valid_sample(sample):
            full_text = (sample['precontext'] + ' ' + sample['sentence'] + ' ' + sample['ending']).strip()
            if not has_multiple_sentences(full_text):
                removed_single_sent += 1
            else:
                removed_bad_format += 1
            continue
        
        # Clean the sample
        cleaned = clean_sample(sample)
        
        # Check if cleaning changed anything
        if cleaned != sample:
            cleaned_text += 1
        
        cleaned_samples.append(cleaned)
    
    print(f"\nRemoved:")
    print(f"  Single-sentence samples: {removed_single_sent}")
    print(f"  Bad formatting: {removed_bad_format}")
    print(f"Text cleaned: {cleaned_text}")
    print(f"Final samples: {len(cleaned_samples)}")
    
    # Convert to dict format with numbered keys
    result = {}
    for idx, sample in enumerate(cleaned_samples):
        result[str(idx)] = sample
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("Done!")
    
    # Show some examples
    if len(result) > 0:
        print("\nFirst 2 samples:")
        import pprint
        pprint.pprint([result[str(i)] for i in range(min(2, len(result)))])

if __name__ == '__main__':
    main()
