#!/bin/bash
# Quick test training with minimal data to verify everything works

echo "Running quick test training (small dataset)..."
echo "=" * 60

python semeval_task5_main.py \
    --train_path data/train.json \
    --dev_path data/dev.json \
    --fews_dir data/fews/fews \
    --llm_ambistory_path data/llm_generated_ambistory.json \
    --llm_ambistory_weight 1.0 \
    --epochs 1 \
    --batch_size 8 \
    --max_train_samples 100 \
    --max_dev_samples 50 \
    --fews_max_examples 100 \
    --skip_baseline \
    --skip_plots

echo ""
echo "Test complete! Check results_summary.csv for results."

