#!/bin/bash
# Quick test with all improvements: new text format, Spearman loss, better architecture

echo "=" * 80
echo "QUICK TEST - Improved Model & Text Format"
echo "=" * 80
echo ""
echo "Testing:"
echo "  ✓ Improved text format (context-first)"
echo "  ✓ Spearman correlation loss"
echo "  ✓ Larger regression head (256→128→1)"
echo "  ✓ Higher LR for regression head"
echo "  ✓ Better initialization"
echo ""

python semeval_task5_main.py \
    --train_path data/train.json \
    --dev_path data/dev.json \
    --fews_dir data/fews/fews \
    --llm_ambistory_path data/llm_generated_ambistory.json \
    --llm_ambistory_weight 1.0 \
    --fews_weight 1.0 \
    --epochs 3 \
    --batch_size 8 \
    --max_train_samples 1000 \
    --max_dev_samples 200 \
    --fews_max_examples 500 \
    --learning_rate 3e-5 \
    --dropout 0.2 \
    --weight_decay 0.01 \
    --pooling weighted \
    --skip_baseline \
    --skip_plots

echo ""
echo "=" * 80
echo "Check results_summary.csv for Spearman correlation"
echo "Target: Spearman > 0.7"
echo "=" * 80

