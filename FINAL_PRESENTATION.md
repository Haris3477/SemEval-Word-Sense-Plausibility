---
title: "SemEval-2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences"
subtitle: "CS445 Final Presentation (Group Project)"
authors: "Raamiz • Musab • Kiyan • Haris"
repo_branch: "fews"
---

## Slide 1 — Title

**SemEval-2026 Task 5**: Rating Plausibility of Word Senses in Ambiguous Sentences  
**Theme**: narrative understanding + sense plausibility (1–5)

**Team**: Raamiz, Musab, Kiyan, Haris

**Best dev (our runs)**: **Spearman 0.7264**, **Within‑SD Acc 0.8350**  
Checkpoint: `outputs/deberta_coral_20251220_spear07264_acc08350.pt`

**Notes (speaker):**
- We focus on the main task: predict plausibility ratings for a *candidate* sense in a short narrative context.
- We’ll walk through dataset, model, augmentation attempts, and results.

---

## Slide 2 — Task in one sentence

Given:
- a short story context (`precontext`, `sentence`, `ending`)
- an ambiguous target word (`homonym`)
- a candidate meaning (`judged_meaning`)

Predict:
- a **plausibility rating** on **[1, 5]**

**Notes:**
- This is not classic WSD classification; it’s *graded* plausibility and ambiguity-sensitive.

---

## Slide 3 — Evaluation (what matters)

We optimize for the task’s evaluation style:
- **Spearman correlation (ρ)**: ranking agreement with human averages
- **Accuracy within 1 standard deviation**: prediction is “close enough” to human mean given annotator disagreement

**Notes:**
- Spearman rewards correct ordering even if absolute values aren’t perfect.
- Within‑SD is forgiving where humans disagree (large stdev), stricter where humans agree (small stdev).

---

## Slide 4 — Data (what we started with)

Official AmbiStory-format JSON:
- **Train**: 2,280 items (`data/train.json`)
- **Dev**: 588 items (`data/dev.json`)

Each example includes:
- `homonym`, `judged_meaning`
- `precontext` (3 sentences), `sentence` (target), `ending` (clue)
- `average` (gold), `stdev` (human disagreement)

**Notes:**
- Key challenge: dev is effectively “zero-shot” on many target lemmas.
- Small training size encourages overfitting and poor generalization.

---

## Slide 5 — Our core idea

**Treat plausibility as an ordinal prediction problem + maximize ranking quality**

We combined:
- **Strong encoder**: DeBERTa‑v3‑large
- **Parameter-efficient tuning**: LoRA
- **Ordinal head**: CORAL (K=5 → 4 logits)

**Notes:**
- Ordinal modeling aligns with “1 < 2 < 3 < 4 < 5”.
- LoRA lets us fine-tune large models on limited compute and reduces overfitting risk.

---

## Slide 6 — Input representation (what the encoder sees)

We format each example into one model input string:
- Ambiguous word + candidate sense (gloss)
- narrative context:
  - `precontext`
  - `sentence` with target marked as `[TGT]...[/TGT]`
  - `ending`

Implementation:
- `create_text_input()` in `semeval_task5_main.py`

**Notes:**
- Explicitly presenting “candidate sense” turns this into sense plausibility scoring rather than open-set WSD.
- Target marking helps the encoder focus attention.

---

## Slide 7 — Core architecture (DeBERTa + LoRA + CORAL)

**Encoder**: `microsoft/deberta-v3-large`  
**PEFT**: LoRA adapters (**r=32**, **alpha=128**, **dropout=0.1**)  
**Head**: CORAL ordinal regression (5 ratings → **4** logits)

Where it lives:
- Training notebook: `semeval_2026_task5.ipynb`
- Inference + submission: `generate_predictions.py` (auto-detects CORAL + LoRA)

**Notes:**
- LoRA lets us fine-tune a large encoder while keeping trainable params small.
- CORAL enforces ordinal consistency and aligns with Spearman.

---

## Slide 8 — Loss design (the big lever we iterated on)

From `semeval_2026_task5.ipynb`, our best-performing setup used:

- **CORAL loss**: optimizes ordinal ranking structure (Spearman-aligned)
- **Huber loss** on predicted rating: improves calibration (Within‑SD)
- **Dynamic, normalized weighting across epochs**:
  - CORAL weight decays (e.g., **1.0 → 0.85**)
  - Huber weight rises (e.g., **0.10 → 0.25**)
  - Total loss magnitude stays stable (prevents training instability)
- **Huber delta** tuned to **0.5** (more sensitive than 1.0)

**Notes:**
- This matches our git history: “added huber”, then “dynamic huber/coral weighting”, then improved results.

---

## Slide 9 — Training protocol (how we avoided overfitting)

Practices used across the final pipeline:
- **Early stopping on dev Spearman**
- **Gradient accumulation** (effective larger batch)
- Small batch sizes for DeBERTa‑large (memory constraints)
- Checkpoint naming includes metrics for easy model selection:
  - `outputs/deberta_coral_YYYYMMDD_spearXXXX_accXXXX.pt`

**Notes:**
- We tuned toward the competition metrics, not just loss curves.

---

## Slide 10 — Iteration timeline (from `fews` branch history)

Observed improvement path (commit subjects):
- **~0.50 corr / ~0.65 acc** (early working baseline, `da70ecf`)
- **0.66 / 0.82** (`4589bdc`, 2025‑12‑17)
- **0.69 / 0.83** (`cefd6ed`, 2025‑12‑18)
- **Loss engineering sprint** (12/18–12/20):
  - Huber loss variants (`e49379d`, `732eed1`)
  - dynamic CORAL↘ + Huber↗ weighting + analysis (`96ce7fa`)
- **0.71–0.72 / ~0.84–0.85** (notebook/model refinements: `37bb7cc`, `4107954`, `e97320c`)
- **0.73 / 0.84** best run + prediction tooling fixes (`36d0973`, plus prediction script commits)

**Notes:**
- In the talk: emphasize what changed each step (loss, stability, inference correctness).

---

## Slide 11 — Results summary (top checkpoints)

Top dev checkpoints (from filenames in `outputs/`):
- **Best Spearman**: **0.7264**, Within‑SD **0.8350**  
  `outputs/deberta_coral_20251220_spear07264_acc08350.pt`
- Next-best Spearman cluster:
  - 0.7198 / 0.8384
  - 0.7150 / 0.8367
  - 0.7101 / 0.8452
  - 0.7066 / 0.8231

**Notes:**
- Highlight stability: multiple runs around 0.71–0.72 Spearman.

---

## Slide 12 — Visual diagnostics (what we used to debug)

Key plots in repo:
- `training_curves_coral.png`
- `predictions_scatter.png`
- `prediction_analysis.png`
- `error_distribution.png`

**Notes:**
- These confirmed (1) ranking improved, (2) calibration improved, (3) we weren’t collapsing to the mean.

---

## Slide 13 — FEWS augmentation (data-side contribution)

Problem:
- dev split is close to zero-shot on lemmas; train is small (2,280).

Solution (Experiment‑1):
- filter FEWS raw text to **multi-sentence** contexts
- generate **balanced negatives** by swapping wrong sense definitions
- remove artifacts (citations, abbreviations, wiki/HTML)

Best curated dataset:
- `Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json` (~66K)

**Notes:**
- This is our “data engineering” contribution; it targets coverage + class balance.

---

## Slide 14 — LLM-based AmbiStory augmentation (built + blocker)

We built:
- `generate_ambistory_fews.py` + `validate_ambistory_generation.py`
- generates AmbiStory-style ambiguity (precontext mentions both senses) + subtle endings
- integrates into training via `--llm_ambistory_path`

Scaling blocker:
- **rate limiting (429s)** + **token/cost** from long prompts

**Notes:**
- We succeeded for small batches; scaling required a more “production” pipeline.

---

## Slide 15 — LLM augmentation: scaling strategy (engineering plan)

What we would implement next:
- **Two-stage generation** (cheap skeleton → expensive rewrite only if passes filters)
- **Prompt compression** (truncate glosses, fewer examples, strict schema)
- **Caching + resumable job queue** (avoid re-generation, safe restart after 429s)
- **Quality gating** (format + ambiguity filters; discard “too obvious” contexts)

**Notes:**
- This directly addresses the real constraints we hit (rate limits + token budget).

---

## Slide 16 — Submission / reproducibility (what to run)

Generate predictions for a split (or official test when available):
- `generate_predictions.py`

Example:
- load checkpoint (auto-detect CORAL + LoRA)
- output `predictions.jsonl` + zip for CodaBench

**Notes:**
- This is “demo-able”: we can show command and a few sample predictions.

---

## Slide 17 — What each teammate contributed (fill in)

- **Member A**: data pipeline + FEWS balancing / cleaning
- **Member B**: modeling experiments (losses, pooling, training sweeps)
- **Member C**: DeBERTa+LoRA+CORAL notebook + checkpointing + prediction script
- **Member D**: LLM augmentation pipeline + validation + integration

**Notes:**
- Replace “Member A/B/C/D” with your actual mapping before presenting.

---

## Slide 18 — Limitations

- Small official training size → overfitting risk
- FEWS domain/style mismatch (even after cleaning)
- LLM augmentation scale blocked by rate limits + token cost
- No official test set access during dev → careful about over-tuning on dev

**Notes:**
- We highlight honest limitations + why our engineering choices were reasonable.

---

## Slide 19 — Takeaways and next steps

**What worked**
- ordinal modeling + LoRA on a strong encoder
- disciplined evaluation + diagnostics
- data quality improvements (balanced negatives, multi-sentence filtering)

**Next steps**
- ship the scalable LLM-generation pipeline (Slide 15)
- ensemble 3–5 checkpoints for small Spearman gains
- calibrate outputs (isotonic/temperature) on dev for better Within‑SD

**Notes:**
- Close on: (1) impact, (2) readiness to submit, (3) future improvements.


