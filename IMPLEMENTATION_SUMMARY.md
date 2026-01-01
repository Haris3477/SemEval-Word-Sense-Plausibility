# Implementation Summary: LLM-Generated AmbiStory Few-Shot Examples

## What Was Built

A complete pipeline to generate AmbiStory-style few-shot training examples using LLM generation, specifically designed to teach models how to navigate **ambiguous contexts**.

## Key Problem Solved

Your teammate identified that previous FEWS data didn't have the ambiguous precontext structure that AmbiStory uses. This was causing the model to struggle with ambiguous contexts. The new generator creates proper AmbiStory-format examples with:

- **Ambiguous Precontexts**: 3 sentences that intentionally mix terms from BOTH senses
- **Straightforward Sentences**: Natural sentences with the homonym
- **Subtle Endings**: Clues that hint at the correct sense but remain ambiguous

## Files Created

### 1. `generate_ambistory_fews.py`
Main generation script that:
- Loads FEWS senses and finds homonym pairs
- Creates detailed prompts emphasizing ambiguous precontexts
- Generates stories using OpenAI API (GPT-4 recommended)
- Parses and formats output as AmbiStory JSON
- Creates both positive and negative examples

### 2. `validate_ambistory_generation.py`
Validation script that checks:
- Format correctness (all required fields)
- Precontext ambiguity (warns if too short or not ambiguous)
- Sentence structure (contains homonym, has [TGT] markers)
- Overall quality metrics

### 3. `quick_test_generation.py`
Quick test script to generate a small batch (5 pairs = 20 examples) for testing

### 4. `GENERATE_AMBISTORY_GUIDE.md`
Comprehensive usage guide with examples, troubleshooting, and best practices

## Integration with Training Pipeline

Updated `semeval_task5_main.py` to support LLM-generated stories:

**New Arguments:**
- `--llm_ambistory_path`: Path to generated JSON file
- `--llm_ambistory_weight`: Weight multiplier for LLM examples

**Usage:**
```bash
python semeval_task5_main.py \
    --train_path data/train.json \
    --dev_path data/dev.json \
    --fews_dir data/fews/fews \
    --llm_ambistory_path data/llm_generated_ambistory.json \
    --llm_ambistory_weight 1.0
```

## Example: What Makes a Good Ambiguous Precontext

**Homonym**: "bugs"
- Sense 1: insects
- Sense 2: software errors

**Good Precontext** (ambiguous):
> "Anna was having a tough week. Her room was a mess, and her computer kept crashing. Frustrated by everything going wrong, she called Jen."

Why it's good:
- "room was a mess" → could imply insects
- "computer kept crashing" → could imply software bugs
- Creates intentional confusion/ambiguity
- Doesn't make either sense obvious

**Bad Precontext** (too obvious):
> "Anna was debugging her code. She found several errors in the program. The software was malfunctioning."

Why it's bad:
- Only mentions software context
- Makes Sense 2 (software bugs) obvious
- No ambiguity

## Workflow

1. **Generate Stories**:
   ```bash
   python generate_ambistory_fews.py \
       --senses_path data/fews/fews/senses.txt \
       --output_path data/llm_generated_ambistory.json \
       --model gpt-4 \
       --max_stories 100
   ```

2. **Validate Quality**:
   ```bash
   python validate_ambistory_generation.py data/llm_generated_ambistory.json
   ```

3. **Train Model**:
   ```bash
   python semeval_task5_main.py \
       --llm_ambistory_path data/llm_generated_ambistory.json \
       --epochs 5
   ```

## Key Features

### 1. Ambiguity-Focused Prompting
The prompt specifically instructs the LLM to:
- Create precontexts that mention BOTH senses
- Intentionally create confusion
- Not make the correct sense obvious

### 2. Quality Control
- Format validation
- Ambiguity checking
- Sample review functionality

### 3. Flexible Integration
- Works alongside existing FEWS data
- Weighted training support
- Same format as SemEval data

## Expected Impact

By training on properly ambiguous precontexts, the model should:
- ✅ Better handle ambiguous contexts in test set
- ✅ Learn to use subtle clues from endings
- ✅ Improve accuracy on ambiguous samples
- ✅ Better correlate with human judgments

## Next Steps

1. **Test Generation**: Run `quick_test_generation.py` to generate a small batch
2. **Review Quality**: Check validation output and sample stories
3. **Generate More**: If quality is good, generate 100-500 stories
4. **Train & Evaluate**: Train model and compare dev set performance
5. **Iterate**: Adjust prompts if needed based on results

## Technical Details

### Dependencies Added
- `openai>=1.0.0` (for API access)

### API Compatibility
- Supports both old and new OpenAI API versions
- Handles API key from environment or argument
- Includes retry logic and error handling

### Data Format
Generated stories match SemEval format exactly:
- Same JSON structure
- Same field names
- Same scoring system (1-5 scale)
- Compatible with existing training pipeline

## Troubleshooting

See `GENERATE_AMBISTORY_GUIDE.md` for detailed troubleshooting, but common issues:

1. **No API key**: Set `OPENAI_API_KEY` environment variable
2. **Rate limiting**: Reduce `--max_stories` or add delays
3. **Poor quality**: Use GPT-4 instead of GPT-3.5, provide example stories
4. **Not ambiguous enough**: Review prompt and adjust emphasis on ambiguity

## Success Criteria

A successful generation should:
- ✅ Have >80% format validity
- ✅ Precontexts mention both senses
- ✅ Stories are natural and readable
- ✅ Endings provide subtle clues
- ✅ Improves model performance on dev set

---

**Ready to use!** Start with `quick_test_generation.py` to test the pipeline, then scale up generation based on results.

