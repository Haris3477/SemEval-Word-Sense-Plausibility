# Using Google Gemini (Free) for AmbiStory Generation

## Quick Setup

### 1. Get a Free Gemini API Key

1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Install the Package

```bash
pip install google-generativeai
```

### 3. Generate Stories with Gemini

**Option A: Set environment variable**
```bash
export GEMINI_API_KEY="your-api-key-here"
python generate_ambistory_fews.py \
    --senses_path data/fews/fews/senses.txt \
    --output_path data/llm_generated_ambistory.json \
    --use_gemini \
    --max_stories 50 \
    --example_stories_path data/sample_data.json
```

**Option B: Pass API key directly**
```bash
python generate_ambistory_fews.py \
    --senses_path data/fews/fews/senses.txt \
    --output_path data/llm_generated_ambistory.json \
    --use_gemini \
    --api_key YOUR_GEMINI_API_KEY \
    --max_stories 50 \
    --example_stories_path data/sample_data.json
```

## Advantages of Gemini

✅ **Free tier available** - No credit card needed for basic usage
✅ **Generous rate limits** - Good for generating many stories
✅ **Good quality** - Gemini Pro produces high-quality creative text
✅ **No quota issues** - Unlike OpenAI free tier

## Model Options

- `gemini-pro` (default) - Best for creative writing
- `gemini-pro-vision` - If you need image support (not needed here)

## Rate Limits

Gemini free tier has generous limits:
- 60 requests per minute
- 1,500 requests per day

For 50 story pairs (200 examples), you'll need ~50 API calls, which is well within limits.

## Troubleshooting

**"Gemini API key not provided"**
- Make sure you set `GEMINI_API_KEY` environment variable OR pass `--api_key`
- Get your key from: https://makersuite.google.com/app/apikey

**"Rate limit exceeded"**
- Wait a minute and try again
- Reduce `--max_stories` to generate fewer at once

**"Model not found"**
- Use `gemini-pro` (default) or `gemini-pro-vision`
- Check Google's documentation for latest model names

## Example Output

The script will generate stories in AmbiStory format with:
- Ambiguous precontexts (mixing both senses)
- Straightforward sentences
- Subtle endings

Then validate with:
```bash
python validate_ambistory_generation.py data/llm_generated_ambistory.json
```

## Next Steps

1. Get your free API key
2. Run generation with `--use_gemini`
3. Validate the output
4. Use in training with `--llm_ambistory_path`

Happy generating! 🚀

