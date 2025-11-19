# Experiment 1: FEWS-Based Data Generation for AmbiStory

**Date:** November 18, 2025  
**Goal:** Generate large-scale training data from FEWS dictionary to improve model performance on AmbiStory task

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Problem Discovery](#problem-discovery)
3. [Solution Approach](#solution-approach)
4. [Implementation](#implementation)
5. [Generated Datasets](#generated-datasets)
6. [Training Instructions](#training-instructions)
7. [Key Findings](#key-findings)

---

## Overview

### Initial Baseline Performance
- **Train Spearman:** 0.854
- **Dev Spearman:** 0.497
- **Issue:** Severe overfitting due to small training set (2,280 samples, 220 words)

### Experiment Goal
Expand training data using FEWS (Few-shot Examples for Word Sense) dictionary to improve generalization.

---

## Problem Discovery

### Initial Augmentation Attempt
**Script:** `augment_ambistory_with_senses.py` (Old Approach)

**Strategy:**
- Take existing AmbiStory samples (2,280 samples, 219 words)
- Use Jaccard similarity to find similar FEWS sense definitions
- Generate additional samples for the same 219 words

**Problems Encountered:**
1. **FEWS Parser Failure:** Only loaded 1 lemma instead of 466K
   - Expected tab-separated format
   - Actual format: key:value pairs separated by blank lines
   
2. **Similarity Threshold Too Strict:** 
   - Set threshold at 0.4 but max similarity was only 0.032
   - Dictionary definitions rarely share exact words
   
3. **Stopword Leakage:**
   - Common phrases like "from which" matched everywhere
   - Generated false positives

4. **Limited Vocabulary:**
   - Still constrained to AmbiStory's 219 words
   - Result: 7,466 samples but low quality

---

## Solution Approach

### Breakthrough Insight
**Key Question:** "The word need not be in AmbiStory train, no? Like dribbling is in dev but not in train. Similarly we could do FEWS words not in AmbiStory but make similar format input file."

### New Strategy
Instead of augmenting existing AmbiStory samples:
1. **Use FEWS's full vocabulary** (466K homonyms, not just AmbiStory's 219)
2. **Generate NEW AmbiStory-format samples** from scratch
3. **Use simple templates** instead of similarity matching
4. **Maintain 1:1 positive:negative ratio** (balanced dataset)

### Advantages
- ✅ No similarity calculation needed (avoids stopword leakage)
- ✅ Massive vocabulary expansion: 219 → 73,113 words (333x increase!)
- ✅ Clean, balanced data with perfect 1:1 ratio
- ✅ Scalable: can generate 20K to 146K samples easily

---

## Implementation

### Script: `generate_fews_ambistory.py`

**Location:** `Experiment-1/generate_fews_ambistory.py`

**Key Features:**
1. **FEWS Parser** (Fixed from old version):
   ```python
   # Correctly parses key:value format with blank line separators
   current_entry = {}
   for line in f:
       if not line.strip():  # Blank line = end of entry
           if 'sense_id' in current_entry and 'gloss' in current_entry:
               # Process complete entry
       elif ':' in line:
           key, value = line.split(':', 1)
           current_entry[key.strip()] = value.strip()
   ```

2. **Single-Word Filter** (Matches AmbiStory format):
   ```python
   eligible_words = [
       (word, senses) for word, senses in self.fews_senses.items()
       if len(senses) >= min_senses and len(word.split()) == 1
   ]
   ```
   - AmbiStory uses 100% single-word entries
   - Filter ensures compatibility

3. **Template-Based Sentence Generation**:
   ```python
   templates = {
       'noun': ["The {word} was important.", "She found a {word}.", ...],
       'verb': ["She decided to {word}.", "They will {word} soon.", ...],
       'adj': ["It was very {word}.", "The {word} thing happened.", ...],
       'adv': ["She moved {word}.", "It happened {word}.", ...]
   }
   ```

4. **Sample Generation** (1 positive + 1 negative per word):
   - **Positive:** Random sense as "correct" → scores 4-5
   - **Negative:** Different random sense as "wrong" → scores 1-2
   - **Vote Jitter:** Adds realistic variation (±0.2) to votes

### Usage

```bash
# Generate 10K words (20K samples)
python Experiment-1/generate_fews_ambistory.py \
    --max_words 10000 \
    --samples_per_word 2 \
    --min_senses 2 \
    --output fews_train_10k_balanced.json

# Generate MAX words (146K samples)
python Experiment-1/generate_fews_ambistory.py \
    --max_words 73113 \
    --samples_per_word 2 \
    --min_senses 2 \
    --output fews_train_max_balanced.json
```

**Parameters:**
- `--max_words`: Number of unique words to sample from FEWS
- `--samples_per_word`: Samples per word (2 = 1 pos + 1 neg for 1:1 ratio)
- `--min_senses`: Minimum senses required (2 for balanced pairs)
- `--output`: Output JSON file path

---

## Generated Datasets

### FEWS-Only Datasets (Experiment-1/)

#### `fews_train_10k_balanced.json`
- **Total Samples:** 20,000
- **Unique Words:** 10,000
- **Positives:** 10,000 (50.0%)
- **Negatives:** 10,000 (50.0%)
- **Format:** 100% single-word entries
- **File Size:** ~11 MB

#### `fews_train_max_balanced.json`
- **Total Samples:** 146,226
- **Unique Words:** 73,113
- **Positives:** 73,113 (50.0%)
- **Negatives:** 73,113 (50.0%)
- **Format:** 100% single-word entries
- **File Size:** ~79 MB

### Combined Datasets (Root Folder)

#### `combined_train_10k.json`
- **Total Samples:** 22,280
  - AmbiStory: 2,280
  - FEWS: 20,000
- **Score Distribution:**
  - Positives (≥4.0): 10,741 (48.2%)
  - Negatives (<2.0): 10,432 (46.8%)
  - Moderate (2-4): 1,107 (5.0%)
- **Unique Words:** 10,193

#### `combined_train_max.json`
- **Total Samples:** 148,506
  - AmbiStory: 2,280
  - FEWS: 146,226
- **Score Distribution:**
  - Positives (≥4.0): 73,854 (49.7%)
  - Negatives (<2.0): 73,545 (49.5%)
  - Moderate (2-4): 1,107 (0.7%)
- **Unique Words:** 73,165

---

## Training Instructions

### Configuration Updates

**Batch Size Optimization:**
```python
# File: semeval_task5_main.py (line 50)
# Changed from 8 → 16 for RTX 4060 8GB
batch_size = 16  # Optimized for RTX 4060 8GB
```

**Current GPU Stats:**
- RTX 4060 8GB
- GPU Utilization: 100%
- VRAM Usage: 4.8GB / 8.2GB
- Temperature: 81°C

### Training Commands

#### Option 1: Train with 10K Combined (Recommended First)
```bash
python semeval_task5_main.py \
    --train_path combined_train_10k.json \
    --pooling weighted \
    --dropout 0.3 \
    --learning_rate 1e-5 \
    --epochs 5 \
    --batch_size 16 \
    --skip_baseline
```

**Rationale:**
- 22K samples (10x original AmbiStory)
- Faster training (~30 min vs 2+ hours)
- Validate approach before scaling up
- Lower dropout (0.3 vs 0.35) due to more data

#### Option 2: Train with MAX Combined (Full Scale)
```bash
python semeval_task5_main.py \
    --train_path combined_train_max.json \
    --pooling weighted \
    --dropout 0.25 \
    --learning_rate 1e-5 \
    --epochs 3 \
    --batch_size 16 \
    --skip_baseline
```

**Rationale:**
- 148K samples (65x original AmbiStory!)
- Maximum vocabulary coverage (73K words)
- Even lower dropout (0.25) due to massive data
- Fewer epochs (3 vs 5) to prevent overfitting

#### Option 3: FEWS-Only Training (Experimental)
```bash
python semeval_task5_main.py \
    --train_path Experiment-1/fews_train_10k_balanced.json \
    --pooling weighted \
    --dropout 0.3 \
    --learning_rate 1e-5 \
    --epochs 5 \
    --batch_size 16 \
    --skip_baseline
```

**Purpose:** Test pure FEWS generalization (no AmbiStory bias)

### Expected Improvements

**Baseline (AmbiStory only):**
- Train: 0.854
- Dev: 0.497
- Gap: 0.357 (severe overfitting)

**Target (10K Combined):**
- Train: 0.75-0.80 (expected drop)
- Dev: 0.55-0.60 (expected improvement)
- Gap: ~0.20 (reduced overfitting)

**Target (MAX Combined):**
- Train: 0.70-0.75 (further drop)
- Dev: 0.60-0.65 (best generalization)
- Gap: ~0.10 (minimal overfitting)

---

## Key Findings

### Data Quality Analysis

#### AmbiStory Format Requirements
- ✅ **100% single-word entries** (no multi-word phrases)
- ✅ Words like "track", "suit", "dribbling" (all single tokens)
- ✅ Dev set: 55 unique words, 588 samples

#### FEWS Vocabulary Coverage
- **Total FEWS lemmas:** 466,648
- **Single-word lemmas:** ~400,000
- **With ≥2 senses (usable):** 73,113
- **With ≥3 senses:** 33,831
- **With ≥4 senses:** 18,811

#### Data Expansion Comparison

| Approach | Words | Samples | Quality | Ratio |
|----------|-------|---------|---------|-------|
| Original AmbiStory | 220 | 2,280 | High | Mixed |
| Old Augmentation | 220 | 7,466 | Low | Imbalanced |
| FEWS 10K | 10,000 | 20,000 | High | 1:1 |
| FEWS MAX | 73,113 | 146,226 | High | 1:1 |

### Technical Lessons

1. **Parser Implementation Matters:**
   - Always inspect file format before assuming structure
   - FEWS uses key:value format, not TSV

2. **Similarity Metrics Fail on Definitions:**
   - Jaccard similarity too low for dictionary glosses
   - Stopword overlap creates false positives
   - Template generation more reliable

3. **Single-Word Constraint Critical:**
   - AmbiStory test set is 100% single-word
   - Multi-word phrases hurt generalization
   - Always filter to match test distribution

4. **Balanced Data Improves Stability:**
   - 1:1 ratio prevents bias
   - 2:1 (old) caused negative skew
   - Model learns both positive/negative equally

---

## Files Modified

### New Scripts
1. `Experiment-1/generate_fews_ambistory.py` (305 lines)
   - Main generation script
   - Fixed FEWS parser
   - Template-based sentence generation
   - Single-word filtering

2. `Experiment-1/augment_ambistory_with_senses.py` (deprecated)
   - Old augmentation approach
   - Kept for reference
   - Not recommended for use

### Modified Scripts
1. `semeval_task5_main.py`
   - Line 50: `batch_size = 8` → `batch_size = 16`
   - Comment added: "# Optimized for RTX 4060 8GB"

### Generated Data Files
1. `Experiment-1/fews_train_10k_balanced.json` (11 MB)
2. `Experiment-1/fews_train_max_balanced.json` (79 MB)
3. `combined_train_10k.json` (12 MB)
4. `combined_train_max.json` (80 MB)

---

## Next Steps

### Immediate Actions
1. ✅ Generate 10K balanced dataset
2. ✅ Generate MAX balanced dataset
3. ✅ Create combined datasets
4. ⏳ Train with combined_train_10k.json
5. ⏳ Evaluate Dev Spearman improvement
6. ⏳ Compare with MAX if needed

### Future Improvements
1. **Better Sentence Templates:**
   - Use Gemini/Ollama for context generation
   - Add AmbiStory-style precontext and endings
   - More natural variation

2. **Smart Word Selection:**
   - Prioritize words similar to AmbiStory dev set
   - Filter by word frequency
   - Balance POS tags (noun/verb/adj/adv)

3. **Hybrid Approaches:**
   - Combine template + LLM generation
   - Use FEWS for negatives, AmbiStory for positives
   - Active learning: generate samples for low-confidence dev words

4. **Evaluation:**
   - Analyze per-word performance on dev
   - Identify which FEWS words help most
   - Iteratively refine word selection

---

## Troubleshooting

### Common Issues

**Issue:** Multi-word entries in generated data
```bash
# Solution: Already fixed in latest version
# Filter added: len(word.split()) == 1
```

**Issue:** Imbalanced positive/negative ratio
```bash
# Solution: Use samples_per_word=2 with min_senses=2
# Generates exactly 1 positive + 1 negative per word
```

**Issue:** FEWS parser only loads 1 word
```bash
# Solution: Fixed in generate_fews_ambistory.py
# Uses key:value parsing instead of tab-separated
```

**Issue:** Out of memory during training
```bash
# Solution: Reduce batch size or use gradient accumulation
--batch_size 8  # Instead of 16
# Or add gradient accumulation (not implemented yet)
```

---

## References

- **FEWS Dataset:** `data/fews/fews/senses.txt`
- **AmbiStory Train:** `data/train.json`
- **AmbiStory Dev:** `data/dev.json`
- **Main Training Script:** `semeval_task5_main.py`
- **GPU:** NVIDIA RTX 4060 8GB (100% utilization at batch size 16)

---

## Summary Statistics

### Before Experiment 1
- Training data: 2,280 samples (220 words)
- Dev Spearman: 0.497
- Overfitting gap: 0.357

### After Experiment 1
- Training data options:
  - 22,280 samples (10,193 words) - 10K combined
  - 148,506 samples (73,165 words) - MAX combined
- Expected dev improvement: 0.55-0.65
- Expected overfitting reduction: gap ~0.10-0.20

### Data Generation Speed
- 10K words: ~20 seconds
- 73K words: ~3 minutes
- Highly scalable and reproducible

---

**Last Updated:** November 18, 2025  
**Next Evaluation:** After training with combined datasets

---

## Update: Context Analysis & Multi-Sentence Filtering (Nov 18, Evening)

### Problem Discovery: Template Approach Failed

**Initial Training Results:**
- Combined dataset (AmbiStory + FEWS templates): **Dev Spearman 0.045** (catastrophic failure!)
- Root cause: 90% of training data (FEWS templates) lacked precontext/ending
- Model learned to **ignore context** instead of using it

**Vocabulary Analysis:**
```
Train vocabulary: 220 words
Dev vocabulary:   55 words
Overlap:          0 words  ← 100% zero-shot vocabulary task!
```

This explains why the task requires learning generalizable sense reasoning, not word memorization.

### Context Format Investigation

Analyzed all FEWS .txt files for **true multi-sentence context** (multiple sentences with actual boundaries: `. ! ?` followed by space + capital letter).

**Analysis Method:**
```python
# Not just long text, but actual sentence boundaries
sentence_boundary_pattern = r'[.!?;]\s+[A-Z]'
```

**Results: Multi-Sentence Context Percentage**

| File               | Multi-Sent | Total | Percentage | Notes                    |
|--------------------|------------|-------|------------|--------------------------|
| quotations.txt     | 58/200     | 29.0% | ← Best source            |
| test.few-shot.txt  | 55/200     | 27.5% |                          |
| train.ext.txt      | 47/200     | 23.5% | ← Use this, not train.txt|
| test.zero-shot.txt | 47/200     | 23.5% |                          |
| monosemous.txt     | 42/200     | 21.0% |                          |
| dev.zero-shot.txt  | 39/200     | 19.5% |                          |
| dev.few-shot.txt   | 37/200     | 18.5% |                          |
| examples.txt       | 9/200      | 4.5%  | ← Skip this              |

**Total multi-sentence samples extracted:** ~102,000 (before filtering)

**Key Findings:**
- ❌ **No FEWS file has majority multi-sentence format** (70-80% are single long sentences)
- ✅ **But ~20-30% do have actual multiple sentences** - these match AmbiStory narrative structure better
- ✅ **train.ext.txt = train.txt but larger** (use ext version only)
- ⚠️ **FEWS contains multi-word terms** with underscores (e.g., `driving_force`, `light_up`)

### Multi-Word Term Analysis

**AmbiStory:** 0% multi-word terms (no underscores)

**FEWS Multi-Word Issues:**
```
Lemma: driving_force  →  In text: "driving forces"  (plural)
Lemma: light_up       →  In text: "lit up"         (conjugation)
Lemma: moon_shot      →  In text: "Moon shot"      (capitalization)
Lemma: donkey_kong    →  In text: "donkey kongs"   (plural)
```

**Decision:** Filter out multi-word terms to match AmbiStory format exactly.

### New Dataset: Combined Multi-Sentence Clean

**Script:** `Experiment-1/create_combined_multisent.py`

**Strategy:**
1. Extract ONLY samples with multiple sentences from all FEWS files
2. Filter OUT multi-word terms (with `_`) to match AmbiStory format
3. Split context intelligently into precontext/sentence/ending
4. Combine with original AmbiStory (2,280 samples)

**Generated Datasets:**

```bash
# Version 1: With multi-word terms (NOT RECOMMENDED)
python3 Experiment-1/create_combined_multisent.py \
    --output Experiment-1/combined_multisent.json
# Output: 104,481 samples (16.9% have multi-word mismatches)

# Version 2: Clean, single-word only (RECOMMENDED) ✅
python3 Experiment-1/create_combined_multisent.py \
    --output Experiment-1/combined_multisent_clean.json
# Output: 86,853 samples (0% multi-word, matches AmbiStory format)
```

**Final Dataset Breakdown:**
```
combined_multisent_clean.json:
- AmbiStory:       2,280 samples (2.6%)
- FEWS multi-sent: 84,573 samples (97.4%)
- Total:           86,853 samples

Files processed:
  quotations.txt:     41,971 samples (8,763 multi-word filtered)
  train.ext.txt:      18,319 samples (2,548 multi-word filtered)
  monosemous.txt:     20,312 samples (5,779 multi-word filtered)
  test.few-shot.txt:   1,034 samples (164 multi-word filtered)
  test.zero-shot.txt:    980 samples (98 multi-word filtered)
  dev.few-shot.txt:      983 samples (163 multi-word filtered)
  dev.zero-shot.txt:     974 samples (113 multi-word filtered)
```

**Context Splitting Algorithm:**
```python
# If multiple sentences in "before" text:
#   - Most of before → precontext
#   - Last sentence fragment + target + first part of after → target_sentence
#   - Rest of after → ending

# If multiple sentences in "after" text:
#   - Before + target + first sentence of after → target_sentence  
#   - Rest → ending
#   - Empty precontext

# Single long sentence:
#   - Entire thing → target_sentence
#   - Empty precontext and ending
```

### Dataset Comparison

| Dataset                        | Samples | Multi-word | Dev Spearman | Notes                        |
|--------------------------------|---------|------------|--------------|------------------------------|
| AmbiStory (original)           | 2,280   | 0%         | **0.497**    | Baseline                     |
| FEWS templates (10K)           | 20,000  | ?          | **0.045**    | Failed - no context          |
| Combined multi-sent (with _)   | 104,481 | 16.9%      | ?            | Has format mismatches        |
| **Combined multi-sent (clean)**| **86,853** | **0%**  | **?**        | **✅ RECOMMENDED**           |

### Why This Matters

**Context is Critical for Zero-Shot Generalization:**
- Task requires learning to use narrative context (not memorizing words)
- If 90% of training data lacks context, model learns "context = noise"
- Multi-sentence samples teach model to integrate precontext → sentence → ending

**Expected Improvement:**
- Previous combined (templates): Spearman **0.045** ❌
- Multi-sentence combined (clean): Expected **0.50-0.55+** ✅
- Provides 38x more data than AmbiStory alone
- All samples have genuine multi-sentence narrative structure
- Matches AmbiStory format exactly (single-word terms only)

### Training Command

```bash
# Train with clean multi-sentence dataset
python semeval_task5_main.py \
    --train_path Experiment-1/combined_multisent_clean.json \
    --dev_path data/dev.json \
    --model_name roberta-base \
    --pooling weighted \
    --dropout 0.35 \
    --learning_rate 8e-6 \
    --epochs 5 \
    --batch_size 16 \
    --skip_baseline

# Expected result: Dev Spearman > 0.50
```

### Files in Experiment-1

```
generate_fews_ambistory.py          # Original template approach (deprecated)
add_context_to_fews.py              # Synthetic context (creates nonsense)
create_combined_multisent.py        # ✅ Multi-sentence extraction script
combined_multisent.json             # With multi-word terms (104K samples)
combined_multisent_clean.json       # ✅ RECOMMENDED (87K samples, clean)
combined_train_10k.json             # Old: Templates without context
fews_train_10k_balanced.json        # Old: Templates without context
```

---

**Last Updated:** November 18, 2025 (Evening - Multi-Sentence Analysis & Filtering)  
**Next Steps:** 
1. ✅ Train with `combined_multisent_clean.json` 
2. Compare against:
   - AmbiStory baseline: 0.497
   - Template failure: 0.045
   - Target: >0.50 dev Spearman
3. If successful, consider two-stage training (pretrain on FEWS, fine-tune on AmbiStory)

---

## Critical Bug Fixes (Nov 19, Morning)

### Issue 1: Format Mismatch
**Problem:** Generated dataset used list format instead of dict format with numbered keys.

**AmbiStory format:**
```json
{
  "0": { "homonym": "potential", ... },
  "1": { "homonym": "drive", ... }
}
```

**Our initial format (WRONG):**
```json
[
  { "homonym": "potential", ... },
  { "homonym": "drive", ... }
]
```

**Fix:** Updated script to output dict with numbered string keys matching AmbiStory exactly.

---

### Issue 2: Missing Sense Definitions
**Problem:** FEWS sense definitions not loading - showing "Definition for X" instead of actual definitions.

**Root cause:** Only loaded 6 definitions instead of 663,730 from `senses.txt`

**Investigation:**
```python
# Old (wrong): Only matched exact sense_id in limited set
sense_def = senses.get(sense_id, f"Definition for {target}")

# Checking reveals:
# Loaded 6 sense definitions  ❌
# Should load: 663,730 definitions ✅
```

**Fix:** Updated `load_senses()` to properly parse the full `senses.txt` file.

**Result:** Now loads 663,730 sense definitions correctly.

---

### Issue 3: Multiple WSD Tags in Same Line
**Problem:** Some quotations have multiple `<WSD>` tags, causing incorrect parsing.

**Example:**
```
...attempting to enter the world of <WSD>audiation</WSD>. At this point, 
however, they are unable to begin to cope with <WSD>audiation</WSD>.
```

**Old parsing result:**
```json
{
  "sentence": ". audiation",  // ❌ WRONG - incomplete sentence
  "ending": "...unable to cope with <WSD>audiation</WSD>."  // ❌ Still has tags
}
```

**Fix Applied:**
1. Find the **FIRST** `<WSD>` tag as the target word
2. Remove **ALL** `<WSD>` and `</WSD>` tags from before/after text
3. Ensure clean text without any tag remnants

**Updated `parse_fews_line()` function:**
```python
# Find FIRST <WSD> tag
first_match = re.search(r'<WSD>(.*?)</WSD>', full_text_with_tags)
target = first_match.group(1).strip()

# Get text before/after FIRST tag
before_with_tags = full_text_with_tags[:first_match.start()]
after_with_tags = full_text_with_tags[first_match.end():]

# Remove ALL <WSD> tags from before/after
before = re.sub(r'</?WSD>', '', before_with_tags).strip()
after = re.sub(r'</?WSD>', '', after_with_tags).strip()
```

**New result:**
```json
{
  "sentence": "Private ' audiations ' are systematically insulated...",  // ✅ Clean
  "ending": "...unable to begin to cope with audiation."  // ✅ Tags removed
}
```

---

### Issue 4: Sentences Starting with Punctuation
**Problem:** Generated sentences starting with periods: `". audiation"`, `". just as"`

**Root cause:** Context splitting algorithm placing punctuation boundaries incorrectly.

**Fix Applied:**
1. Strip leading punctuation from target_sentence: `.lstrip('.!?;: ')`
2. Capitalize first letter if lowercase
3. Ensure grammatical sentence structure

**Updated `split_context()` logic:**
```python
# Clean up: ensure target_sentence doesn't start with punctuation
target_sentence = target_sentence.lstrip('.!?;: ')

# If target_sentence starts with lowercase, capitalize it
if target_sentence and target_sentence[0].islower():
    target_sentence = target_sentence[0].upper() + target_sentence[1:]
```

**Before:**
```
". audiation"  ❌
". just as core to their calling"  ❌
```

**After:**
```
"Private ' audiations ' are systematically insulated..."  ✅
"Improving the workings of the businesses..."  ✅
```

---

### Final Dataset Quality Check

**Script run results:**
```bash
python3 Experiment-1/create_combined_multisent.py \
    --output Experiment-1/combined_multisent_clean.json

Loading FEWS senses: 663,730 definitions ✅
Total samples: 88,104 ✅
Format: Dict with numbered keys ✅
Sentences starting with punctuation: 0 ✅
Missing definitions: 0 ✅
```

**Random sample quality:**
```
Sample 83810 - weevil:
  Sentence: "But you accuse other men of villainy with too easy a tongue, you weevil"
  Judged meaning: A loathsome person.
  ✅ Looks good

Sample 14592 - reverbate:
  Sentence: "The heavenly orbs heard the commanding voice reverbate from the mountains"
  Judged meaning: (rare) (reverberate)
  ✅ Looks good
```

---

### Dataset Version History

| Version                          | Samples | Format | Definitions | Sentence Quality | Status       |
|----------------------------------|---------|--------|-------------|------------------|--------------|
| combined_multisent.json          | 104,481 | List   | Missing     | Has issues       | ❌ Deprecated |
| combined_multisent_clean v1      | 86,853  | List   | Missing     | Has issues       | ❌ Deprecated |
| combined_multisent_clean_v2.json | 88,104  | Dict   | ✅ Loaded   | Has issues       | ❌ Deprecated |
| **combined_multisent_clean.json** | **88,104** | **Dict** | **✅ 663K** | **✅ Clean** | **✅ FINAL** |

**Recommended:** Use `Experiment-1/combined_multisent_clean.json` (final version)

---

**Last Updated:** November 19, 2025 (Morning - Critical Bug Fixes)  
**Status:** ✅ Dataset ready for training  
**Next Step:** Train model and compare performance against baselines

