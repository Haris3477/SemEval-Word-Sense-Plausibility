#!/usr/bin/env python
"""
Generate AmbiStory-style few-shot examples using LLM generation.

This script creates training examples in the AmbiStory format:
- Precontext: 3 sentences with ambiguous terms from BOTH senses
- Sentence: Straightforward sentence containing the homonym
- Ending: Provides answer but remains somewhat ambiguous

The goal is to teach the model to navigate ambiguous contexts.
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import pandas as pd
from tqdm import tqdm

# Try to import OpenAI, but make it optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

# Try to import Google Gemini, but make it optional
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Gemini not available. Install with: pip install google-generativeai")

# Try to import local LLM support
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

WSD_PATTERN = re.compile(r"<WSD>(.*?)</WSD>", re.IGNORECASE)
TARGET_TEMPLATE = "[TGT]{token}[/TGT]"
PRED_MIN, PRED_MAX = 1.0, 5.0


def load_fews_senses(senses_path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    """Load FEWS senses and group by lemma."""
    senses: Dict[str, Dict[str, str]] = {}
    lemma_to_senses: Dict[str, List[str]] = {}
    current: Dict[str, str] = {}
    
    with open(senses_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                if current and 'sense_id' in current:
                    sense_id = current['sense_id']
                    senses[sense_id] = current
                    lemma = current.get('word', '')
                    lemma_to_senses.setdefault(lemma, []).append(sense_id)
                current = {}
                continue
            if ':\t' not in line:
                continue
            key, value = line.split(':\t', 1)
            current[key.strip()] = value.strip()
    
    if current and 'sense_id' in current:
        sense_id = current['sense_id']
        senses[sense_id] = current
        lemma = current.get('word', '')
        lemma_to_senses.setdefault(lemma, []).append(sense_id)
    
    return senses, lemma_to_senses


def get_homonym_pairs(lemma_to_senses: Dict[str, List[str]], senses: Dict[str, Dict[str, str]], 
                      min_senses: int = 2, max_pairs_per_lemma: int = 3) -> List[Tuple[str, str, str]]:
    """
    Get pairs of senses for homonyms that have multiple senses.
    Returns: List of (lemma, sense_id_1, sense_id_2) tuples
    """
    pairs = []
    
    for lemma, sense_ids in lemma_to_senses.items():
        if len(sense_ids) < min_senses:
            continue
        
        # Filter to senses with non-empty glosses
        valid_senses = [
            sid for sid in sense_ids 
            if senses.get(sid, {}).get('gloss', '').strip()
        ]
        
        if len(valid_senses) < min_senses:
            continue
        
        # Generate pairs (up to max_pairs_per_lemma)
        pair_count = 0
        for i in range(len(valid_senses)):
            if pair_count >= max_pairs_per_lemma:
                break
            for j in range(i + 1, len(valid_senses)):
                if pair_count >= max_pairs_per_lemma:
                    break
                pairs.append((lemma, valid_senses[i], valid_senses[j]))
                pair_count += 1
    
    return pairs


def create_generation_prompt(lemma: str, sense1_id: str, sense1_gloss: str, 
                            sense2_id: str, sense2_gloss: str, 
                            example_stories: Optional[List[Dict]] = None) -> str:
    """Create a prompt for LLM to generate AmbiStory-style story."""
    
    prompt = f"""You are creating a short story for a word sense disambiguation task. The word "{lemma}" has two different meanings:

Sense 1: {sense1_gloss}
Sense 2: {sense2_gloss}

Create a 5-sentence story in the following format:

1. PRECONTEXT (3 sentences): These sentences MUST be AMBIGUOUS and contain terms/contexts that could relate to BOTH senses. This is CRITICAL - the precontext should intentionally confuse the reader by mentioning things related to BOTH senses. 

   Example: For "bugs" (insects vs software errors), a good precontext might be:
   "Anna was having a tough week. Her room was a mess, and her computer kept crashing. Frustrated by everything going wrong, she called Jen."
   Notice how it mentions both "room was a mess" (could imply insects) AND "computer kept crashing" (could imply software bugs). This creates ambiguity.

2. SENTENCE (1 sentence): A straightforward sentence containing the word "{lemma}". This sentence should be ambiguous on its own - it could plausibly use either sense. It should NOT give away which sense is correct.

3. ENDING (1 sentence): This sentence should provide subtle clues that point toward ONE specific sense, but should still be somewhat ambiguous. It should make the correct sense more plausible without being completely obvious or explicit.

CRITICAL REQUIREMENTS FOR PRECONTEXT (MOST IMPORTANT):
- The precontext MUST intentionally mention things related to BOTH senses
- It should create confusion/ambiguity by mixing contexts
- For example, if one sense is about technology and another is about nature, mention BOTH technology and nature in the precontext
- The precontext should NOT make it obvious which sense is correct
- Write 3 complete sentences that together create this ambiguous context

CRITICAL REQUIREMENTS FOR SENTENCE:
- Should be straightforward and natural
- Should contain the word "{lemma}"
- Should be ambiguous - could work with either sense

CRITICAL REQUIREMENTS FOR ENDING:
- Should hint at the correct sense subtly
- Should NOT be completely obvious
- Should still allow some ambiguity

Generate TWO versions of the story:
1. One where Sense 1 is the correct interpretation (ending hints at Sense 1)
2. One where Sense 2 is the correct interpretation (ending hints at Sense 2)

IMPORTANT: For each version, the PRECONTEXT should be different and tailored to create ambiguity for that specific sense pair. The precontext should mention elements from BOTH senses to create confusion.

Format your response as JSON with this structure:
{{
  "sense1_story": {{
    "precontext": "Sentence 1. Sentence 2. Sentence 3.",
    "sentence": "Sentence with {lemma}.",
    "ending": "Ending sentence."
  }},
  "sense2_story": {{
    "precontext": "Sentence 1. Sentence 2. Sentence 3.",
    "sentence": "Sentence with {lemma}.",
    "ending": "Ending sentence."
  }}
}}

Generate the stories now:"""
    
    if example_stories:
        prompt += "\n\nHere are some examples of the desired style:\n\n"
        for ex in example_stories[:2]:  # Show 2 examples
            prompt += f"Example:\n"
            prompt += f"Homonym: {ex['homonym']}\n"
            prompt += f"Sense: {ex['judged_meaning']}\n"
            prompt += f"Precontext: {ex['precontext']}\n"
            prompt += f"Sentence: {ex['sentence']}\n"
            prompt += f"Ending: {ex['ending']}\n\n"
    
    return prompt


def generate_with_openai(prompt: str, model: str = "gpt-4", temperature: float = 0.7, 
                         max_retries: int = 3, api_key: Optional[str] = None) -> Optional[str]:
    """Generate story using OpenAI API."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not available. Install with: pip install openai")
    
    # Handle both old and new OpenAI API versions
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        use_new_api = True
    except ImportError:
        # Old API version
        use_new_api = False
        if api_key:
            openai.api_key = api_key
    
    for attempt in range(max_retries):
        try:
            if use_new_api:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a creative writer helping with a word sense disambiguation task."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1500
                )
                return response.choices[0].message.content.strip()
            else:
                # Old API
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a creative writer helping with a word sense disambiguation task."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1500
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None
    
    return None


def generate_with_gemini(prompt: str, api_key: Optional[str] = None, model: str = "gemini-2.5-flash", 
                         temperature: float = 0.7, max_retries: int = 3) -> Optional[str]:
    """Generate story using Google Gemini API (free tier available)."""
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Google Gemini not available. Install with: pip install google-generativeai")
    
    try:
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from environment
            import os
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
            else:
                raise RuntimeError("Gemini API key not provided. Set GEMINI_API_KEY env var or pass --api_key")
        
        # Map model names to correct API names (use full model path)
        model_map = {
            "gemini-pro": "models/gemini-2.5-flash",  # Default to stable flash
            "gemini-1.5-flash": "models/gemini-2.5-flash",
            "gemini-1.5-pro": "models/gemini-2.5-pro",
            "gemini-2.5-flash": "models/gemini-2.5-flash",
            "gemini-2.5-pro": "models/gemini-2.5-pro",
            "gemini-flash-latest": "models/gemini-flash-latest",
            "gemini-pro-latest": "models/gemini-pro-latest"
        }
        api_model_name = model_map.get(model, model)
        
        # Ensure model name starts with "models/" if not already
        if not api_model_name.startswith("models/"):
            api_model_name = f"models/{api_model_name}"
        
        # Create model instance
        model_instance = genai.GenerativeModel(api_model_name)
        
        for attempt in range(max_retries):
            try:
                # Generate content
                response = model_instance.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=2000,
                    )
                )
                
                if response and response.text:
                    return response.text.strip()
                else:
                    print(f"  Empty response from Gemini (attempt {attempt + 1})")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"  Failed after {max_retries} attempts: {e}")
                    return None
        
        return None
    except Exception as e:
        print(f"  Gemini generation failed: {e}")
        return None


def generate_with_local_llm(prompt: str, model_name: str = "gpt2", max_length: int = 1000) -> Optional[str]:
    """Generate story using local transformer model (fallback)."""
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers not available")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Note: This is a basic implementation. For better results, use a proper instruction-tuned model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_length=max_length, temperature=0.7, do_sample=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return generated_text
    except Exception as e:
        print(f"  Local LLM generation failed: {e}")
        return None


def parse_story_response(response: str) -> Optional[Dict]:
    """Parse LLM response into story structure."""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return data
    except json.JSONDecodeError:
        pass
    
    # Fallback: try to parse structured text
    try:
        # Look for sense1_story and sense2_story sections
        sense1_match = re.search(r'sense1_story[:\s]*\{([^}]+)\}', response, re.DOTALL | re.IGNORECASE)
        sense2_match = re.search(r'sense2_story[:\s]*\{([^}]+)\}', response, re.DOTALL | re.IGNORECASE)
        
        if sense1_match and sense2_match:
            # Try to extract fields
            def extract_field(text, field_name):
                match = re.search(rf'{field_name}[:\s]*"([^"]+)"', text, re.IGNORECASE)
                return match.group(1) if match else ""
            
            sense1_text = sense1_match.group(1)
            sense2_text = sense2_match.group(1)
            
            return {
                "sense1_story": {
                    "precontext": extract_field(sense1_text, "precontext"),
                    "sentence": extract_field(sense1_text, "sentence"),
                    "ending": extract_field(sense1_text, "ending")
                },
                "sense2_story": {
                    "precontext": extract_field(sense2_text, "precontext"),
                    "sentence": extract_field(sense2_text, "sentence"),
                    "ending": extract_field(sense2_text, "ending")
                }
            }
    except Exception as e:
        print(f"  Failed to parse response: {e}")
    
    return None


def highlight_target(sentence: str, homonym: str) -> str:
    """Highlight target word in sentence."""
    if not sentence or '[TGT]' in sentence or not homonym:
        return sentence
    
    pattern = re.compile(rf"\b{re.escape(homonym)}\b", re.IGNORECASE)
    if pattern.search(sentence):
        return pattern.sub(lambda m: TARGET_TEMPLATE.format(token=m.group(0)), sentence, count=1)
    
    if len(homonym) >= 3:
        prefix_pattern = re.compile(rf"\b{re.escape(homonym)}\w*\b", re.IGNORECASE)
        if prefix_pattern.search(sentence):
            return prefix_pattern.sub(lambda m: TARGET_TEMPLATE.format(token=m.group(0)), sentence, count=1)
    
    return sentence


def create_ambistory_example(lemma: str, sense_id: str, sense_gloss: str, 
                             story_data: Dict, base_id: str, rng: random.Random,
                             is_positive: bool = True) -> Dict:
    """Create an AmbiStory-format example from generated story."""
    
    # Generate realistic plausibility scores
    if is_positive:
        base = rng.uniform(3.5, 4.8)
    else:
        base = rng.uniform(1.2, 2.8)
    
    votes = []
    for offset in [-0.4, -0.2, 0.0, 0.2, 0.4]:
        jitter = rng.uniform(-0.3, 0.3)
        score = max(PRED_MIN, min(PRED_MAX, base + offset + jitter))
        votes.append(int(round(score)))
    
    average = float(statistics.mean(votes))
    stdev = float(statistics.pstdev(votes)) if len(votes) > 1 else 0.6
    
    sentence = story_data.get('sentence', '')
    sentence = highlight_target(sentence, lemma)
    
    return {
        'id': base_id,
        'sample_id': base_id,
        'homonym': lemma,
        'judged_meaning': sense_gloss,
        'precontext': story_data.get('precontext', ''),
        'sentence': sentence,
        'ending': story_data.get('ending', ''),
        'choices': votes,
        'average': average,
        'stdev': stdev,
        'nonsensical': [False] * len(votes),
        'example_sentence': '',
        'sense_tags': '',
        'sense_synonyms': '',
        'source': 'llm-generated-ambistory',
        'sense_id': sense_id,
    }


def generate_ambistory_stories(
    senses_path: Path,
    output_path: Path,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    max_stories: int = 1000,
    seed: int = 42,
    example_stories: Optional[List[Dict]] = None,
    use_local: bool = False,
    local_model: str = "gpt2",
    use_gemini: bool = False
):
    """Main function to generate AmbiStory-style stories."""
    
    print("=" * 80)
    print("AmbiStory-Style FEWS Story Generator")
    print("=" * 80)
    
    # Determine which API to use
    import os
    if use_gemini:
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Google Gemini not available. Install with: pip install google-generativeai")
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("Warning: No Gemini API key provided. Set GEMINI_API_KEY env var or use --api_key")
            print("Get a free API key at: https://makersuite.google.com/app/apikey")
    elif not use_local:
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI not available. Install with: pip install openai")
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY env var or use --api_key")
            print("Attempting to continue anyway (will fail if API key required)...")
    
    # Load FEWS senses
    print("\n1. Loading FEWS senses...")
    senses, lemma_to_senses = load_fews_senses(senses_path)
    print(f"   Loaded {len(senses)} senses for {len(lemma_to_senses)} lemmas")
    
    # Get homonym pairs
    print("\n2. Finding homonym pairs...")
    pairs = get_homonym_pairs(lemma_to_senses, senses, min_senses=2, max_pairs_per_lemma=3)
    print(f"   Found {len(pairs)} sense pairs")
    
    if max_stories:
        pairs = pairs[:max_stories]
        print(f"   Limiting to {max_stories} pairs")
    
    # Generate stories
    print("\n3. Generating AmbiStory-style stories...")
    rng = random.Random(seed)
    generated_examples = []
    failed_count = 0
    
    for idx, (lemma, sense1_id, sense2_id) in enumerate(tqdm(pairs, desc="Generating")):
        sense1_meta = senses.get(sense1_id, {})
        sense2_meta = senses.get(sense2_id, {})
        
        sense1_gloss = sense1_meta.get('gloss', '').strip()
        sense2_gloss = sense2_meta.get('gloss', '').strip()
        
        if not sense1_gloss or not sense2_gloss:
            continue
        
        # Create prompt
        prompt = create_generation_prompt(
            lemma, sense1_id, sense1_gloss, sense2_id, sense2_gloss, example_stories
        )
        
        # Generate with LLM
        if use_local:
            response = generate_with_local_llm(prompt, model_name=local_model)
        elif use_gemini:
            response = generate_with_gemini(prompt, api_key=api_key, model=model)
        else:
            response = generate_with_openai(prompt, model=model, api_key=api_key)
        
        if not response:
            failed_count += 1
            continue
        
        # Parse response
        story_data = parse_story_response(response)
        if not story_data:
            failed_count += 1
            continue
        
        # Create examples for both senses
        sense1_story = story_data.get('sense1_story', {})
        sense2_story = story_data.get('sense2_story', {})
        
        if sense1_story.get('sentence') and sense1_story.get('precontext'):
            example1 = create_ambistory_example(
                lemma, sense1_id, sense1_gloss, sense1_story,
                f"llm-ambistory-{idx}-sense1", rng, is_positive=True
            )
            generated_examples.append(example1)
            
            # Also create negative example with wrong sense
            example1_neg = create_ambistory_example(
                lemma, sense2_id, sense2_gloss, sense1_story,
                f"llm-ambistory-{idx}-sense1-neg", rng, is_positive=False
            )
            generated_examples.append(example1_neg)
        
        if sense2_story.get('sentence') and sense2_story.get('precontext'):
            example2 = create_ambistory_example(
                lemma, sense2_id, sense2_gloss, sense2_story,
                f"llm-ambistory-{idx}-sense2", rng, is_positive=True
            )
            generated_examples.append(example2)
            
            # Also create negative example with wrong sense
            example2_neg = create_ambistory_example(
                lemma, sense1_id, sense1_gloss, sense2_story,
                f"llm-ambistory-{idx}-sense2-neg", rng, is_positive=False
            )
            generated_examples.append(example2_neg)
        
        # Rate limiting for API calls
        if not use_local and idx < len(pairs) - 1:
            time.sleep(0.5)  # Small delay to avoid rate limits
    
    print(f"\n   Generated {len(generated_examples)} examples")
    print(f"   Failed generations: {failed_count}")
    
    # Save to JSON
    print(f"\n4. Saving to {output_path}...")
    output_dict = {str(i): ex for i, ex in enumerate(generated_examples)}
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\n✓ Complete! Generated {len(generated_examples)} AmbiStory-style examples")
    return generated_examples


def main():
    parser = argparse.ArgumentParser(description="Generate AmbiStory-style few-shot examples")
    parser.add_argument('--senses_path', type=str, default='data/fews/fews/senses.txt',
                       help='Path to FEWS senses.txt')
    parser.add_argument('--output_path', type=str, default='data/llm_generated_ambistory.json',
                       help='Output JSON file path')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', type=str, default='gpt-4',
                       help='OpenAI model to use (gpt-4, gpt-3.5-turbo, etc.)')
    parser.add_argument('--max_stories', type=int, default=100,
                       help='Maximum number of story pairs to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--example_stories_path', type=str, default='data/sample_data.json',
                       help='Path to example AmbiStory stories for few-shot prompting')
    parser.add_argument('--use_local', action='store_true',
                       help='Use local transformer model instead of API')
    parser.add_argument('--local_model', type=str, default='gpt2',
                       help='Local model name (if using --use_local)')
    parser.add_argument('--use_gemini', action='store_true',
                       help='Use Google Gemini API (free tier available) instead of OpenAI')
    
    args = parser.parse_args()
    
    # Load example stories for few-shot prompting
    example_stories = None
    if Path(args.example_stories_path).exists():
        with open(args.example_stories_path, 'r') as f:
            example_data = json.load(f)
            example_stories = list(example_data.values())[:10]  # Use first 10 as examples
        print(f"Loaded {len(example_stories)} example stories for few-shot prompting")
    
    # Default model for Gemini
    if args.use_gemini and args.model == 'gpt-4':
        args.model = 'gemini-2.5-flash'  # Stable, fast, and free
    
    generate_ambistory_stories(
        senses_path=Path(args.senses_path),
        output_path=Path(args.output_path),
        api_key=args.api_key,
        model=args.model,
        max_stories=args.max_stories,
        seed=args.seed,
        example_stories=example_stories,
        use_local=args.use_local,
        local_model=args.local_model,
        use_gemini=args.use_gemini
    )


if __name__ == '__main__':
    main()

