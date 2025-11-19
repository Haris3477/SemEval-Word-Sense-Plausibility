#!/bin/bash
# Training commands for balanced FEWS + AmbiStory datasets

# Dataset 1: 10K Balanced (strict full context)
# 9,592 samples - 1:1 ratio
python semeval_task5_main.py \
    --train_path Experiment-1/final-ish/combined_clean_10k_balanced.json \
    --dev_path data/dev.json \
    --model_name roberta-base \
    --pooling weighted \
    --dropout 0.35 \
    --learning_rate 8e-6 \
    --epochs 5 \
    --batch_size 16 \
    --skip_baseline

# Dataset 2: Full Balanced (strict full context, same as 10K)
# 9,592 samples - 1:1 ratio
python semeval_task5_main.py \
    --train_path Experiment-1/final-ish/combined_clean_full_balanced.json \
    --dev_path data/dev.json \
    --model_name roberta-base \
    --pooling weighted \
    --dropout 0.35 \
    --learning_rate 8e-6 \
    --epochs 5 \
    --batch_size 16 \
    --skip_baseline

# Dataset 3: Full Relaxed Balanced (relaxed context)
# 66,712 samples - 1:1 ratio
# This will take ~7x longer to train (45-60 min per epoch)
python semeval_task5_main.py \
    --train_path Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json \
    --dev_path data/dev.json \
    --model_name roberta-base \
    --pooling weighted \
    --dropout 0.35 \
    --learning_rate 8e-6 \
    --epochs 5 \
    --batch_size 16 \
    --skip_baseline
