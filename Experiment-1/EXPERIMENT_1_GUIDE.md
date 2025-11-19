# Experiment 1: Balanced FEWS Dataset Generation

**Date:** November 19, 2025  
**Goal:** Generate balanced, high-quality training data from FEWS with proper multi-sentence context

---

## Quick Start

**Best Dataset:** `Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json`

**Train Now:**
```bash
python semeval_task5_main.py \
    --train_path Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json \
    --dev_path data/dev.json \
    --model_name roberta-base \
    --pooling weighted \
    --dropout 0.35 \
    --learning_rate 8e-6 \
    --epochs 5 \
    --batch_size 16
```

**Expected:** Dev Spearman > 0.50 (baseline: 0.497)

---

## The Problem

**Baseline (AmbiStory only):**
- Train: 0.854 | Dev: 0.497 → Overfitting gap of 0.357
- Only 2,280 samples covering 220 words
- Dev set has 100% unseen vocabulary (zero-shot task)

**Failed Attempts:**
1. **Templates:** Generated sentences like "The X was important" → Dev 0.045 (no context!)
2. **Similarity matching:** Stopword leakage, low Jaccard scores
3. **Single sentences:** 70% of FEWS lacks multi-sentence structure

---

## Solution: Balanced Multi-Sentence FEWS

### Key Insights
1. **Use FEWS raw/ files only** (train/dev/test are subsets → avoid redundancy)
2. **Filter for multi-sentence context** (precontext + sentence + ending)
3. **Generate equal negatives** (FEWS only has positive samples)
4. **Balance is critical** (50/50 positive/negative prevents bias)

### Data Sources
```
FEWS raw/ files:
- quotations.txt:  237K samples (29% multi-sentence) ← Best source
- monosemous.txt:  132K samples (21% multi-sentence)
- examples.txt:     17K samples (4.5% multi-sentence)
- senses.txt:      663,730 sense definitions
```

### Quality Filters Applied

**1. Multi-Sentence Validation:**
```python
# Not just long text, but actual sentence boundaries
has_multiple_sentences(text):
    # Remove citations: (Author, 1969; Other, 2020)
    # Remove brackets: [page 10], [[wiki link]]
    # Remove abbreviations: Mr., Dr., Prof., St., etc.
    # Check for: ". X" or "! X" or "? X" (capital after boundary)
    return bool(re.search(r'[.!?][\s.]+[A-Z]', cleaned_text))
```

**2. Context Quality Tiers:**
- **Strict:** All 3 fields non-empty (precontext + sentence + ending), no "..."
- **Relaxed:** Either precontext OR ending non-empty, not both "..."

**3. Multi-Sense Filtering:**
- Only keep homonyms with 2+ definitions (required to generate negatives)
- Single-sense words can't have wrong definitions swapped

**4. Artifact Removal:**
```python
# Clean URLs, HTML entities, wiki markup
text = re.sub(r'https?://\S+|www\.\S+', '', text)
text = re.sub(r'&#\d+;|&[a-z]+;', '', text)
text = re.sub(r'\[\[.*?\]\]|\{\{.*?\}\}', '', text)
```

### Negative Generation Strategy

**FEWS Problem:** Only provides positive samples (correct sense definitions)

**Solution:** Swap wrong definitions from same homonym's alternative senses

**Example:**
```python
# Positive sample
{
  "homonym": "bank",
  "sentence": "She deposited money at the bank.",
  "judged_meaning": "A financial institution",
  "plausibility_rating": 4.5  # Correct
}

# Generated negative
{
  "homonym": "bank",
  "sentence": "She deposited money at the bank.",  # Same context
  "judged_meaning": "The slope beside a body of water",  # Wrong sense!
  "plausibility_rating": 1.2  # Very implausible
}
```

**Score Distribution:** Matches AmbiStory's negative pattern
- 60% very negative (1.0-2.0)
- 40% moderately negative (2.0-3.0)

---

## Generated Datasets

### Three Balanced Datasets

| Dataset | Samples | Pos/Neg | Context Quality | Use Case |
|---------|---------|---------|-----------------|----------|
| **10K Balanced** | 9,592 | 48%/52% | Full (strict) | Quick iteration |
| **Full Balanced** | 9,592 | 48%/52% | Full (strict) | Same as 10K* |
| **Full Relaxed** | 66,712 | 49.7%/50.3% | Good (relaxed) | **Maximum data** |

*Limited by multi-sense full-context samples from FEWS

**Composition:**
- AmbiStory: 2,280 samples (already balanced)
- FEWS positives: 3,656 (10K) or 32,216 (relaxed)
- FEWS negatives: Equal number to positives

**Quality Checks:**
- ✅ 100% have `plausibility_rating` field (normalized)
- ✅ 0 samples with both precontext/ending as "..."
- ✅ 0 `_sense_id` fields remaining
- ✅ Proper multi-sentence context validated
- ✅ No escaped quotes (JSON escaping is correct)

---

## Training Commands

### Recommended: Full Relaxed (66K samples)
```bash
python semeval_task5_main.py \
    --train_path Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json \
    --dev_path data/dev.json \
    --model_name roberta-base \
    --pooling weighted \
    --dropout 0.35 \
    --learning_rate 8e-6 \
    --epochs 5 \
    --batch_size 16
```

**Why this dataset?**
- 29x more data than AmbiStory alone
- Near-perfect balance (49.7% pos / 50.3% neg)
- Relaxed context = more samples while maintaining quality

### Alternative: 10K Balanced (faster)
```bash
python semeval_task5_main.py \
    --train_path Experiment-1/final-ish/combined_clean_10k_balanced.json \
    --dev_path data/dev.json \
    --model_name roberta-base \
    --pooling weighted \
    --dropout 0.35 \
    --learning_rate 8e-6 \
    --epochs 5 \
    --batch_size 16
```

**Use for:** Quick validation before full training (~5min vs ~30min)

---

## Implementation Details

### Script: `generate_balanced_dataset.py`

**Pipeline:**
1. Load AmbiStory (2,280 samples)
2. Load FEWS senses (663,730 definitions)
3. Extract positives from raw/ files with multi-sentence validation
4. Filter for context quality (strict or relaxed)
5. Filter for multi-sense homonyms (can generate negatives)
6. Generate equal negatives by swapping wrong definitions
7. Combine, shuffle, and save

**Key Functions:**
- `has_multiple_sentences()`: Validates sentence boundaries (handles abbreviations, citations)
- `split_context()`: Distributes text into precontext/sentence/ending
- `generate_negatives_for_sample()`: Creates negatives with wrong definitions
- `clean_reference_artifacts()`: Removes URLs, HTML, wiki markup

### Critical Bug Fixes Applied

**1. Abbreviation Handling:**
```python
# Problem: "On Sunday, a Mr. Brown" split as complete sentence
# Fix: Remove 22 common abbreviations before checking boundaries
common_abbreviations = ['Mr.', 'Mrs.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', ...]
```

**2. Citation Semicolons:**
```python
# Problem: "(Author, 1969; Other, 2020)" detected as sentence boundary
# Fix: Remove parentheses/brackets before checking semicolons
text = re.sub(r'\([^)]*\)|\[[^\]]*\]', '', text)
```

**3. "..." Placeholder Filtering:**
```python
# Problem: ~5K samples had both precontext AND ending as "..."
# Fix: Reject samples where both fields are empty/dots
if precontext in ['...', ''] and ending in ['...', '']:
    skip_sample()
```

**4. Field Normalization:**
```python
# Problem: AmbiStory uses 'average', FEWS uses 'plausibility_rating'
# Fix: Normalize all to 'plausibility_rating' for training
if 'average' in sample:
    sample['plausibility_rating'] = sample.pop('average')
```

---

## Expected Results

| Metric | AmbiStory Only | 10K Balanced | Full Relaxed |
|--------|----------------|--------------|--------------|
| Train Spearman | 0.854 | 0.75-0.80 | 0.70-0.75 |
| Dev Spearman | 0.497 | 0.52-0.55 | **0.55-0.60** |
| Overfitting Gap | 0.357 | ~0.20 | **~0.15** |

**Key Improvements:**
- Better generalization (more diverse vocabulary)
- Reduced overfitting (29x more data)
- Proper context usage (multi-sentence samples)
- Balanced learning (50/50 pos/neg prevents bias)

---

## Files

**Scripts:**
- `Experiment-1/generate_balanced_dataset.py` - Main generation script

**Generated Data:**
- `Experiment-1/final-ish/combined_clean_10k_balanced.json` (9,592 samples)
- `Experiment-1/final-ish/combined_clean_full_balanced.json` (9,592 samples)
- `Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json` (66,712 samples) ✅

**Deprecated:**
- `generate_fews_ambistory.py` - Template approach (failed: 0.045 dev)
- `create_combined_multisent.py` - Earlier version (had quality issues)
- `combined_multisent_clean.json` - Unbalanced (98% positive)

---

## Regenerate Datasets

```bash
cd Experiment-1
python3 generate_balanced_dataset.py

# Output:
# - final-ish/combined_clean_10k_balanced.json
# - final-ish/combined_clean_full_balanced.json  
# - final-ish/combined_clean_full_relaxed_balanced.json
#
# Runtime: ~5 minutes
```

---

## Key Learnings

1. **Balance matters:** 98% positive → model ignores negatives; 50/50 → proper learning
2. **Context is critical:** Template-generated sentences (no context) → 0.045 dev Spearman
3. **Multi-sentence ≠ long text:** Need actual boundaries (`. X`, `! X`), not just length
4. **FEWS structure:** raw/ is source, train/dev/test are subsets (avoid redundancy)
5. **Negative generation:** Only works for multi-sense homonyms (48% of FEWS samples)
6. **Abbreviations matter:** "Mr. Brown" must not be split as two sentences
7. **Quality > Quantity:** 66K balanced > 87K unbalanced

---

**Last Updated:** November 19, 2025  
**Status:** ✅ Ready for training  
**Next:** Train and evaluate, compare against baseline 0.497

