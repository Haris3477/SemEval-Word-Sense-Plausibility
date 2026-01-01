#!/usr/bin/env python
"""
Quick test script to generate a small batch of AmbiStory examples for testing.

This generates just 5 story pairs (20 examples total) to quickly test the pipeline.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 80)
    print("Quick Test: Generate Small Batch of AmbiStory Examples")
    print("=" * 80)
    
    # Check if OpenAI API key is set
    import os
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        print("   Or pass --api_key to the generation script")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Paths
    senses_path = Path('data/fews/fews/senses.txt')
    output_path = Path('data/llm_generated_ambistory_test.json')
    example_path = Path('data/sample_data.json')
    
    # Check if paths exist
    if not senses_path.exists():
        print(f"\n❌ Error: {senses_path} not found!")
        print("   Make sure you're in the project root directory.")
        sys.exit(1)
    
    if not example_path.exists():
        print(f"\n⚠️  Warning: {example_path} not found.")
        print("   Continuing without example stories...")
        example_path = None
    
    # Build command
    cmd = [
        sys.executable,
        'generate_ambistory_fews.py',
        '--senses_path', str(senses_path),
        '--output_path', str(output_path),
        '--model', 'gpt-4',  # Use GPT-4 for better quality
        '--max_stories', '5',  # Just 5 pairs for quick test
        '--seed', '42'
    ]
    
    if example_path:
        cmd.extend(['--example_stories_path', str(example_path)])
    
    print(f"\nGenerating 5 story pairs (20 examples total)...")
    print(f"Output will be saved to: {output_path}")
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Run generation
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Generation complete!")
        
        # Validate
        if output_path.exists():
            print("\n" + "=" * 80)
            print("Validating generated stories...")
            print("=" * 80)
            
            validate_cmd = [
                sys.executable,
                'validate_ambistory_generation.py',
                str(output_path)
            ]
            
            try:
                subprocess.run(validate_cmd, check=True)
                print("\n✓ Validation complete!")
            except subprocess.CalledProcessError:
                print("\n⚠️  Validation found some issues. Review the output above.")
        
        print(f"\n✓ Test complete! Check {output_path} for results.")
        print("\nNext steps:")
        print("  1. Review the generated stories")
        print("  2. If quality is good, generate more with --max_stories 100")
        print("  3. Use in training with --llm_ambistory_path")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Generation failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()

