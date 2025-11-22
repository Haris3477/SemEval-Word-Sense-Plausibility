#!/bin/bash
# Training script optimized for Spearman correlation > 0.7

echo "Training for Spearman correlation > 0.7"
echo "Using FULL dataset + all FEWS + LLM-generated stories"
echo "=" * 80

python semeval_task5_main.py \
    --train_path data/train.json \
    --dev_path data/dev.json \
    --fews_dir data/fews/fews \
    --llm_ambistory_path data/llm_generated_ambistory.json \
    --llm_ambistory_weight 1.0 \
    --fews_weight 1.0 \
    --epochs 10 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --dropout 0.2 \
    --weight_decay 0.01 \
    --pooling weighted \
    --skip_baseline \
    --skip_plots \
    --early_stop_patience 3

echo ""
echo "Training complete! Check results_summary.csv for Spearman correlation."


