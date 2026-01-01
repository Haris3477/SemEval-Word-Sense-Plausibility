# LLM-Generated AmbiStory Few-Shot Examples Guide

## Overview

This guide explains how to generate AmbiStory-style few-shot examples using LLM generation. The goal is to create training data that teaches the model to navigate **ambiguous contexts**, which is critical for the SemEval 2026 Task 5 challenge.

## Why This Matters

The AmbiStory dataset format is specifically designed to test a model's ability to handle ambiguity:

- **Precontext**: 3 sentences that are intentionally ambiguous, containing terms from BOTH senses
- **Sentence**: Straightforward sentence with the homonym (ambiguous on its own)
- **Ending**: Provides subtle clues but remains somewhat ambiguous

Your teammate identified that previous FEWS data didn't have this ambiguous precontext structure, which is why the model struggled with ambiguous contexts. This generator creates proper AmbiStory-format examples.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have an OpenAI API key. Set it as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Generate Stories

```bash
python generate_ambistory_fews.py \
    --senses_path data/fews/fews/senses.txt \
    --output_path data/llm_generated_ambistory.json \
    --model gpt-4 \
    --max_stories 100 \
    --example_stories_path data/sample_data.json
```

**Parameters:**
- `--senses_path`: Path to FEWS senses.txt file
- `--output_path`: Where to save generated stories (JSON format)
- `--model`: OpenAI model to use (`gpt-4`, `gpt-3.5-turbo`, etc.)
- `--max_stories`: Number of story pairs to generate (each pair generates 4 examples: 2 positive, 2 negative)
- `--example_stories_path`: Path to example AmbiStory stories for few-shot prompting
- `--api_key`: OpenAI API key (or set OPENAI_API_KEY env var)

### 3. Validate Generated Stories

```bash
python validate_ambistory_generation.py data/llm_generated_ambistory.json
```

This will check:
- Stories have correct format (precontext, sentence, ending)
- Precontexts are ambiguous (contain terms from both senses)
- Sentences contain the homonym with [TGT] markers
- Overall quality and validity

### 4. Use in Training

Add the generated stories to your training pipeline:

```bash
python semeval_task5_main.py \
    --train_path data/train.json \
    --dev_path data/dev.json \
    --fews_dir data/fews/fews \
    --llm_ambistory_path data/llm_generated_ambistory.json \
    --llm_ambistory_weight 1.0 \
    --epochs 5 \
    --batch_size 8
```

## How It Works

### Story Generation Process

1. **Extract Homonym Pairs**: Finds words in FEWS that have multiple senses
2. **Create Generation Prompt**: For each pair, creates a detailed prompt that:
   - Explains both senses
   - Emphasizes the need for ambiguous precontexts
   - Provides examples of good ambiguous precontexts
   - Requests two story versions (one for each sense)
3. **LLM Generation**: Uses GPT-4 (or other model) to generate stories
4. **Parse & Format**: Converts LLM output to AmbiStory JSON format
5. **Create Examples**: For each story, creates:
   - Positive example (correct sense matches story)
   - Negative example (wrong sense with same story)

### Example Output Structure

Each generated story follows this format:

```json
{
  "0": {
    "homonym": "potential",
    "judged_meaning": "the difference in electrical charge between two points",
    "precontext": "The old machine hummed in the corner. Clara examined its dusty dials. She wondered if it could be brought back to life.",
    "sentence": "The [TGT]potential[/TGT] couldn't be measured.",
    "ending": "She collected a battery reader and looked on earnestly.",
    "choices": [4, 5, 3, 4, 3],
    "average": 3.8,
    "stdev": 0.84,
    "source": "llm-generated-ambistory"
  }
}
```

## Key Features

### Ambiguous Precontext Generation

The generator specifically instructs the LLM to create precontexts that:
- Mention terms/contexts from BOTH senses
- Create intentional confusion/ambiguity
- Don't make it obvious which sense is correct

Example for "bugs":
- Sense 1: insects
- Sense 2: software errors
- Good precontext: "Anna was having a tough week. Her room was a mess, and her computer kept crashing."
  - "room was a mess" → could imply insects
  - "computer kept crashing" → could imply software bugs
  - Creates ambiguity!

### Quality Control

The validation script checks:
- ✅ Format correctness (all required fields present)
- ✅ Precontext length (should be ~3 sentences)
- ✅ Sentence contains homonym
- ✅ [TGT] markers present
- ⚠️ Ambiguity (warns if precontext seems too short or not ambiguous enough)

## Tips for Best Results

1. **Use GPT-4**: GPT-4 produces better ambiguous precontexts than GPT-3.5
2. **Provide Examples**: Use `--example_stories_path` to show the LLM good examples
3. **Start Small**: Generate 50-100 stories first, validate, then scale up
4. **Review Samples**: Always check the validation output and sample stories
5. **Iterate on Prompts**: If stories aren't ambiguous enough, adjust the prompt in `generate_ambistory_fews.py`

## Troubleshooting

### "OpenAI not available"
```bash
pip install openai
```

### "No API key found"
Set environment variable:
```bash
export OPENAI_API_KEY="your-key"
```

### Stories not ambiguous enough
- Check the validation warnings
- Review sample stories
- Consider adjusting the prompt to emphasize ambiguity more
- Try GPT-4 instead of GPT-3.5

### Rate limiting
The script includes a 0.5s delay between API calls. For large batches, you may need to:
- Use a higher-tier OpenAI plan
- Generate in smaller batches
- Use `--max_stories` to limit generation

## Integration with Training

The generated stories are automatically integrated into training when you use `--llm_ambistory_path`. They:
- Are loaded alongside FEWS and SemEval training data
- Use the same format as SemEval data
- Can be weighted with `--llm_ambistory_weight`
- Are included in all training metrics

## Expected Impact

By training on properly ambiguous precontexts, your model should:
- Better handle ambiguous contexts in the test set
- Learn to use subtle clues from endings
- Improve accuracy on samples with ambiguous precontexts
- Better correlate with human judgments

## Next Steps

1. Generate a small batch (50-100 stories)
2. Validate and review quality
3. Train model with generated stories
4. Compare dev set performance
5. Scale up generation if results are good
6. Iterate on prompt if needed

Good luck with SemEval 2026 Task 5! 🚀

