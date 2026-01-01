#!/bin/bash
# Two-Stage Training: FEWS Pretraining → AmbiStory Fine-tuning

set -e  # Exit on error

echo "======================================================================"
echo "STAGE 1: Pretraining on FEWS (Lexical Knowledge)"
echo "======================================================================"

python semeval_task5_main.py \
  --train_path Experiment-1/fews_train_10k_balanced.json \
  --dev_path data/dev.json \
  --model_name roberta-base \
  --pooling weighted \
  --dropout 0.3 \
  --learning_rate 2e-5 \
  --epochs 3 \
  --batch_size 16 \
  --save_dir outputs/stage1_fews \
  --skip_baseline

echo ""
echo "======================================================================"
echo "STAGE 1 COMPLETE - Checkpoint saved to outputs/stage1_fews/"
echo "======================================================================"
echo ""
echo "======================================================================"
echo "STAGE 2: Fine-tuning on AmbiStory (Context Reasoning)"
echo "======================================================================"
echo ""
echo "NOTE: You need to manually modify semeval_task5_main.py to load"
echo "      the checkpoint from stage 1. Current script doesn't support this."
echo ""
echo "Alternative: Just run AmbiStory training with the baseline model:"
echo ""
echo "python semeval_task5_main.py \\"
echo "  --train_path data/train.json \\"
echo "  --dev_path data/dev.json \\"
echo "  --model_name roberta-base \\"
echo "  --pooling weighted \\"
echo "  --dropout 0.35 \\"
echo "  --learning_rate 8e-6 \\"
echo "  --epochs 5 \\"
echo "  --batch_size 16"
echo ""
echo "======================================================================"
