# SemEval 2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences

## Project Overview

This project implements a system for rating the plausibility of word senses in ambiguous sentences through narrative understanding. The task involves disambiguating homonyms in short stories by understanding contextual clues.

## Current SemEval 2026 Task 5 ranking
<img width="1600" height="874" alt="image" src="https://github.com/user-attachments/assets/af69da08-2d75-4dc9-8cd5-58b1d3933c69" />
SU NLP 29 - Rank 4

## Dataset

- **Training Set**: `train.json` - Large training dataset with annotated plausibility scores
- **Development Set**: `dev.json` - Development/validation dataset
- **Sample Data**: `sample_data.json` - Small sample for testing

Each data point contains:
- A homonym (ambiguous word)
- A judged meaning (specific word sense)
- Precontext (3 sentences of story setup)
- An ambiguous sentence containing the homonym
- Optional ending sentence
- Human plausibility ratings (1-5 scale)

## Installation

```bash
pip install -r requirements.txt
```

## Evaluation Metrics

- **Spearman Correlation**: Correlation between predicted and human scores
- **Accuracy Within Standard Deviation**: Percentage of predictions within 1 SD of human average


## Team Members

- Raamiz
- Musab
- Kiyan
- Haris

