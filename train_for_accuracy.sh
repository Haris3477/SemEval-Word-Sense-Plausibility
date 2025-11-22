#!/bin/bash
# Training script optimized for Accuracy Within SD > 70%

echo "=================================================================================="
echo "TRAINING FOR ACCURACY > 70%"
echo "=================================================================================="
echo ""
echo "Improvements:"
echo "  ✓ Stronger accuracy loss (5x weight)"
echo "  ✓ Smooth penalty for predictions outside margin"
echo "  ✓ Tightness bonus for accurate predictions"
echo "  ✓ Higher learning rate (5e-5)"
echo "  ✓ More epochs (15)"
echo "  ✓ Full dataset (no limits)"
echo ""

python semeval_task5_main.py \
    --train_path data/train.json \
    --dev_path data/dev.json \
    --fews_dir data/fews/fews \
    --llm_ambistory_path data/llm_generated_ambistory.json \
    --llm_ambistory_weight 1.0 \
    --fews_weight 1.0 \
    --epochs 15 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --dropout 0.15 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --pooling weighted \
    --skip_baseline \
    --skip_plots \
    --early_stop_patience 5

echo ""
echo "=================================================================================="
echo "Training complete! Check results_summary.csv"
echo "Target: Accuracy Within SD > 0.70 (70%)"
echo "=================================================================================="

