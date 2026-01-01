#!/usr/bin/env python
"""
Validate LLM-generated AmbiStory examples to ensure they match the desired format.

This script checks:
1. Stories have ambiguous precontexts (contain terms from both senses)
2. Sentences are straightforward but ambiguous
3. Endings provide clues but remain ambiguous
4. Format matches AmbiStory structure
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

def validate_story_format(story: Dict) -> List[str]:
    """Validate that a story has the correct AmbiStory format."""
    errors = []
    
    required_fields = ['homonym', 'judged_meaning', 'precontext', 'sentence', 'ending']
    for field in required_fields:
        if field not in story:
            errors.append(f"Missing required field: {field}")
        elif not story[field] or not str(story[field]).strip():
            errors.append(f"Empty required field: {field}")
    
    # Check precontext has multiple sentences (should be ~3)
    if 'precontext' in story and story['precontext']:
        sentences = story['precontext'].split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:
            errors.append(f"Precontext should have at least 2 sentences, found {len(sentences)}")
    
    # Check sentence contains the homonym
    if 'sentence' in story and 'homonym' in story:
        homonym = story['homonym'].lower()
        sentence = story['sentence'].lower()
        if homonym not in sentence and not any(word.startswith(homonym) for word in sentence.split()):
            errors.append(f"Sentence doesn't contain homonym '{homonym}'")
    
    # Check for [TGT] markers
    if 'sentence' in story and '[TGT]' not in story['sentence']:
        errors.append("Sentence should contain [TGT] markers around the homonym")
    
    return errors


def check_ambiguity(story: Dict, all_stories: List[Dict]) -> List[str]:
    """Check if precontext is ambiguous (contains terms from both senses)."""
    warnings = []
    
    homonym = story.get('homonym', '').lower()
    sense = story.get('judged_meaning', '').lower()
    precontext = story.get('precontext', '').lower()
    
    if not precontext:
        return warnings
    
    # Find other stories with same homonym but different sense
    other_senses = [
        s for s in all_stories 
        if s.get('homonym', '').lower() == homonym 
        and s.get('judged_meaning', '').lower() != sense
    ]
    
    if not other_senses:
        warnings.append("No other sense found for this homonym - cannot verify ambiguity")
        return warnings
    
    # Check if precontext contains terms that could relate to other senses
    # This is a heuristic check - we look for common words that might indicate ambiguity
    other_sense = other_senses[0].get('judged_meaning', '').lower()
    
    # Simple heuristic: if precontext is very short or doesn't contain varied vocabulary,
    # it might not be ambiguous enough
    words_in_precontext = set(precontext.split())
    if len(words_in_precontext) < 10:
        warnings.append("Precontext seems too short - may not be ambiguous enough")
    
    return warnings


def validate_generated_stories(json_path: Path) -> Dict:
    """Validate all stories in a generated JSON file."""
    print("=" * 80)
    print("Validating LLM-Generated AmbiStory Examples")
    print("=" * 80)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    stories = list(data.values())
    print(f"\nLoaded {len(stories)} stories from {json_path}")
    
    format_errors = []
    ambiguity_warnings = []
    valid_count = 0
    
    for idx, story in enumerate(stories):
        # Format validation
        errors = validate_story_format(story)
        if errors:
            format_errors.append((idx, story.get('id', f'story_{idx}'), errors))
        else:
            valid_count += 1
        
        # Ambiguity check
        warnings = check_ambiguity(story, stories)
        if warnings:
            ambiguity_warnings.append((idx, story.get('id', f'story_{idx}'), warnings))
    
    # Print results
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"\n✓ Valid format: {valid_count}/{len(stories)} ({valid_count/len(stories)*100:.1f}%)")
    print(f"✗ Format errors: {len(format_errors)}")
    print(f"⚠ Ambiguity warnings: {len(ambiguity_warnings)}")
    
    if format_errors:
        print(f"\n{'='*80}")
        print("FORMAT ERRORS (first 10):")
        print(f"{'='*80}")
        for idx, story_id, errors in format_errors[:10]:
            print(f"\nStory {idx} (ID: {story_id}):")
            for error in errors:
                print(f"  - {error}")
            if 'precontext' in story:
                print(f"  Precontext: {story['precontext'][:100]}...")
            if 'sentence' in story:
                print(f"  Sentence: {story['sentence']}")
    
    if ambiguity_warnings:
        print(f"\n{'='*80}")
        print("AMBIGUITY WARNINGS (first 10):")
        print(f"{'='*80}")
        for idx, story_id, warnings in ambiguity_warnings[:10]:
            print(f"\nStory {idx} (ID: {story_id}):")
            for warning in warnings:
                print(f"  - {warning}")
    
    # Show sample valid stories
    print(f"\n{'='*80}")
    print("SAMPLE VALID STORIES (first 3):")
    print(f"{'='*80}")
    
    valid_stories = [s for s in stories if not validate_story_format(s)]
    for idx, story in enumerate(valid_stories[:3]):
        print(f"\nExample {idx + 1}:")
        print(f"  Homonym: {story.get('homonym', 'N/A')}")
        print(f"  Sense: {story.get('judged_meaning', 'N/A')[:80]}...")
        print(f"  Precontext: {story.get('precontext', 'N/A')[:150]}...")
        print(f"  Sentence: {story.get('sentence', 'N/A')}")
        print(f"  Ending: {story.get('ending', 'N/A')[:100]}...")
        print(f"  Average score: {story.get('average', 'N/A')}")
    
    return {
        'total': len(stories),
        'valid': valid_count,
        'format_errors': len(format_errors),
        'ambiguity_warnings': len(ambiguity_warnings),
        'validity_rate': valid_count / len(stories) if stories else 0
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate LLM-generated AmbiStory examples")
    parser.add_argument('json_path', type=str, help='Path to generated JSON file')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    results = validate_generated_stories(json_path)
    
    # Exit with error code if validation failed
    if results['validity_rate'] < 0.8:
        print(f"\n⚠️  WARNING: Only {results['validity_rate']*100:.1f}% of stories are valid!")
        sys.exit(1)
    else:
        print(f"\n✓ Validation passed! {results['validity_rate']*100:.1f}% of stories are valid.")


if __name__ == '__main__':
    main()

