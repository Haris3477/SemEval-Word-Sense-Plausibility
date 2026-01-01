#!/usr/bin/env python
"""Quick test to validate FEWS integration fixes."""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main module
from semeval_task5_main import load_fews_dataframe, create_text_input, set_seed

def test_fews_integration():
    """Test FEWS dataset integration and check for common issues."""
    
    print("=" * 80)
    print("Testing FEWS Integration")
    print("=" * 80)
    
    # Load a small sample of FEWS data
    set_seed(42)
    fews_df = load_fews_dataframe(
        fews_dir='data/fews/fews',
        max_examples=100,  # Small sample for testing
        negatives=1,
        include_ext=False,
        weight_multiplier=1.0,
        seed=42
    )
    
    print(f"\n✓ Successfully loaded {len(fews_df)} FEWS samples")
    
    # Check 1: Ensure homonym has no suffixes
    print("\n" + "=" * 80)
    print("Check 1: Homonym Format (should be bare lemma, no .pos.sense)")
    print("=" * 80)
    
    bad_homonyms = []
    for idx, row in fews_df.head(20).iterrows():
        homonym = row['homonym']
        if '.' in homonym or '_' in homonym:
            bad_homonyms.append(homonym)
            print(f"  ❌ BAD: '{homonym}' (contains suffix)")
        else:
            print(f"  ✓ GOOD: '{homonym}'")
    
    if bad_homonyms:
        print(f"\n⚠️  Found {len(bad_homonyms)} homonyms with suffixes!")
    else:
        print(f"\n✓ All homonyms are clean bare lemmas!")
    
    # Check 2: Ensure gloss is non-empty
    print("\n" + "=" * 80)
    print("Check 2: Gloss/Judged Meaning (should be non-empty)")
    print("=" * 80)
    
    empty_glosses = 0
    for idx, row in fews_df.head(20).iterrows():
        gloss = row['judged_meaning']
        if not gloss or not gloss.strip():
            empty_glosses += 1
            print(f"  ❌ EMPTY gloss for '{row['homonym']}'")
        else:
            print(f"  ✓ '{row['homonym']}': '{gloss[:60]}...'")
    
    if empty_glosses > 0:
        print(f"\n⚠️  Found {empty_glosses} empty glosses!")
    else:
        print(f"\n✓ All glosses are non-empty!")
    
    # Check 3: Ensure sense_id is NOT in the visible text
    print("\n" + "=" * 80)
    print("Check 3: Sense ID Leakage (should NOT appear in model-visible text)")
    print("=" * 80)
    
    leakage_found = []
    for idx, row in fews_df.head(20).iterrows():
        text = create_text_input(row, mark_homonym=True)
        sense_id = row.get('sense_id', '')
        
        # Check if full sense_id appears in text
        if sense_id and sense_id in text:
            leakage_found.append((row['homonym'], sense_id))
            print(f"  ❌ LEAK: sense_id '{sense_id}' found in text!")
        else:
            print(f"  ✓ '{row['homonym']}' - no leakage")
    
    if leakage_found:
        print(f"\n⚠️  Found {len(leakage_found)} instances of sense_id leakage!")
    else:
        print(f"\n✓ No sense_id leakage detected!")
    
    # Check 4: Print sample model inputs
    print("\n" + "=" * 80)
    print("Check 4: Sample Model Inputs (first 3 examples)")
    print("=" * 80)
    
    for idx, row in fews_df.head(3).iterrows():
        text = create_text_input(row, mark_homonym=True)
        print(f"\nExample {idx + 1}:")
        print(f"  Homonym: '{row['homonym']}'")
        print(f"  Score: {row['average']:.2f} (source: {row['source']})")
        print(f"  Model Input (first 200 chars):")
        print(f"  {text[:200]}...")
    
    # Check 5: Verify [TGT] markers are present
    print("\n" + "=" * 80)
    print("Check 5: Target Markers (should contain [TGT]...[/TGT])")
    print("=" * 80)
    
    missing_markers = 0
    for idx, row in fews_df.head(20).iterrows():
        sentence = row['sentence']
        if '[TGT]' not in sentence or '[/TGT]' not in sentence:
            missing_markers += 1
            print(f"  ⚠️  '{row['homonym']}': Missing [TGT] markers")
        else:
            print(f"  ✓ '{row['homonym']}': Has [TGT] markers")
    
    if missing_markers > 0:
        print(f"\n⚠️  {missing_markers} sentences missing target markers!")
    else:
        print(f"\n✓ All sentences have [TGT] markers!")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_good = (
        len(bad_homonyms) == 0 and 
        empty_glosses == 0 and 
        len(leakage_found) == 0 and 
        missing_markers == 0
    )
    
    if all_good:
        print("✓ ✓ ✓ ALL CHECKS PASSED! FEWS integration looks good!")
    else:
        print("⚠️  Some issues detected - review output above")
        if bad_homonyms:
            print(f"  - {len(bad_homonyms)} homonyms with suffixes")
        if empty_glosses:
            print(f"  - {empty_glosses} empty glosses")
        if leakage_found:
            print(f"  - {len(leakage_found)} sense_id leakages")
        if missing_markers:
            print(f"  - {missing_markers} missing target markers")

if __name__ == '__main__':
    test_fews_integration()
