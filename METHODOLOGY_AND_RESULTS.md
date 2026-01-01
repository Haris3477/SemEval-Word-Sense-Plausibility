---
title: "SemEval-2026 Task 5 — Solution Methodology & Results (from `fews` branch)"
---

## Narrative summary (much more detailed, report-style)

### 1) Introduction / problem setting
SemEval‑2026 Task 5 asks systems to **rate the plausibility** of a *candidate* word sense in an ambiguous narrative. Each instance includes:
- `precontext`: 3 sentences designed to be ambiguous,
- `sentence`: a sentence containing the ambiguous target word (`homonym`),
- `ending`: a sentence that subtly provides evidence toward one sense,
- `judged_meaning`: a candidate meaning (gloss) for the homonym,
- `average` (gold rating) and `stdev` (annotator disagreement).

Unlike classic WSD, the system is not asked to choose a single label. Instead, it must output a **graded plausibility score in [1, 5]** for the proposed sense under the story evidence. This has two implications:
1) the model must learn **sense evaluation** (“how plausible is this gloss in this story?”), not only disambiguation, and  
2) evaluation emphasizes **rank consistency** (Spearman correlation) and a tolerance-aware correctness criterion (**Accuracy Within Standard Deviation**) that accounts for human disagreement.

### 2) Dataset and evaluation setup
We used the official data shipped in AmbiStory format:
- **Train**: 2,280 items (`data/train.json`)
- **Dev**: 588 items (`data/dev.json`)

We tracked two primary metrics throughout:
- **Spearman correlation (ρ)** between model predictions and human means (ranking agreement).
- **Accuracy Within SD**: the fraction of predictions that fall within a tolerance band around the mean, where the tolerance is based on `stdev` (with a minimum clamp used in our evaluation code).

Intuition:
- Spearman rewards correct ordering even if absolute scores are slightly shifted.
- Within‑SD rewards calibration (absolute closeness) but is more forgiving when annotators disagree.

### 3) Input formulation (turning the task into “evaluate this sense”)
We convert each instance into a single transformer input that explicitly contains:
1) the ambiguous target word (`homonym`),
2) the candidate sense gloss (`judged_meaning`),
3) the narrative context (`precontext`, `sentence`, `ending`).

We also mark the target token in the sentence using `[TGT]...[/TGT]`. This improves consistency for the model because:
- it makes the evaluated token unambiguous (even when the word appears multiple times),
- it provides a stable positional anchor for attention across examples,
- it reduces sensitivity to small tokenization/inflection differences.

### 4) Model architecture (DeBERTa + LoRA + CORAL)
Our strongest approach is an **ordinal regression** model built on a strong narrative encoder:

**Backbone encoder.**
- We use `microsoft/deberta-v3-large`, chosen for strong multi-sentence language understanding.

**Parameter-efficient fine-tuning (LoRA).**
Because the official training set is relatively small (2,280 items), full fine‑tuning of a large encoder risks unstable updates and overfitting. We therefore use LoRA adapters (configured in `semeval_2026_task5.ipynb`):
- rank **r=32**
- alpha **128**
- LoRA dropout **0.1**
- target modules include attention projections and dense components (as set in the notebook config).

**Ordinal prediction head (CORAL).**
The rating scale is ordinal (1 < 2 < 3 < 4 < 5). CORAL models this by predicting \(K-1\) ordered binary decisions. With \(K=5\), the model outputs 4 logits corresponding to:
- “is rating > 1?”, “> 2?”, “> 3?”, “> 4?”

At inference (as implemented in `generate_predictions.py`):
1) apply sigmoid to each logit to obtain probabilities,  
2) sum probabilities,  
3) add 1, producing a continuous prediction in [1, 5].

This design encourages monotonicity and directly supports rank-consistent behavior, which tends to help Spearman.

### 5) Optimization: hybrid objective (ranking + calibration)
During development, we found that plain MSE regression can produce undesirable behavior on small datasets (e.g., collapsing toward the mean). To address this, our final training objective combines:

1) **CORAL loss** (enforces ordinal structure / rank consistency)  
2) **Huber loss** on the predicted continuous rating (improves calibration; robust to noisy labels)

We use **dynamic, normalized weighting across epochs** (explicitly implemented in `semeval_2026_task5.ipynb`):
- CORAL weight decays from **1.0 → 0.85**
- Huber weight increases from **0.10 → 0.25**
- weights are scheduled so the overall loss scale remains stable
- Huber delta is tuned to **0.5** (more sensitive than delta=1.0)

Interpretation:
- early epochs prioritize learning the correct ordinal/ranking structure,
- later epochs increase pressure on absolute calibration (helping Within‑SD).

### 6) Training protocol (reproducible configuration)
Our final notebook configuration (from `semeval_2026_task5.ipynb`) uses:
- `max_length`: **512**
- `loss_type`: **hybrid**
- `learning_rate`: **8e-5**
- `epochs`: **10** (with early stopping patience configured in the notebook)
- `batch_size`: **2**
- `grad_accumulation_steps`: **8** → effective batch size **16**
- aggressive memory clearing between epochs (to avoid accumulation on constrained devices)

We continuously monitored:
- dev Spearman (primary model selection signal),
- dev Within‑SD accuracy (secondary),
- and qualitative diagnostics via plots (e.g., `training_curves_coral.png`, `predictions_scatter.png`, `error_distribution.png`).

### 7) Problems encountered and how we solved them (key engineering story)

This project was not “train once and done”; the final results came from repeatedly identifying concrete failure modes and designing targeted fixes.

#### Problem A — Severe overfitting and poor generalization (small official training set)
**Symptom.**
- The official training set is only 2,280 items, while dev contains largely unseen lemmas (zero-shot flavor).
- In early experiments (documented in `Experiment-1/EXPERIMENT_1_GUIDE.md`), a baseline could fit train well but lag on dev (large generalization gap).

**What we changed.**
- Used **parameter-efficient fine-tuning (LoRA)** instead of full fine-tuning to reduce overfitting pressure on a large encoder.
- Used **early stopping on dev Spearman** to prevent training past the point where ranking stops improving.
- Added **augmentation experiments** (FEWS / LLM) to increase coverage while trying to preserve the narrative structure of the task.

**Impact.**
- Enabled stable training of DeBERTa-large within constraints and reduced “memorize train” behavior.

#### Problem B — Data format mismatch: “more data” that does not match AmbiStory structure hurts
**Symptom.**
- Early attempts to generate additional training examples with simplistic templates produced very low dev performance (e.g., templates with no narrative context).
- FEWS contains many examples that are not multi-sentence narrative contexts; naïvely mixing them risks teaching the model the wrong structure.

**What we changed.**
- Built a **balanced multi-sentence FEWS pipeline** (Experiment‑1):
  - filtered to true multi-sentence contexts (not just long text),
  - cleaned citations/abbreviations/artifacts,
  - generated **equal negatives** by swapping wrong sense definitions into the same context,
  - produced a large balanced dataset artifact (~66K).

**Impact.**
- Augmentation became “structure-aligned” rather than just “more text”, reducing the chance of harming narrative reasoning.

#### Problem C — Training collapse toward mean predictions (ranking suffers)
**Symptom.**
- With small data and noisy labels, regression training can collapse toward predicting the global mean rating.
- This hurts Spearman (poor ordering) and also can hurt Within‑SD if predictions cluster too tightly.

**What we changed.**
- Switched away from pure MSE emphasis and adopted a **hybrid objective**:
  - CORAL to enforce ordinal structure / ranking consistency,
  - Huber to calibrate absolute scores robustly.
- Added **dynamic weighting** across epochs (CORAL emphasized early, Huber later) while normalizing loss scale (implemented in `semeval_2026_task5.ipynb`).
- Tuned **Huber delta=0.5** to increase sensitivity to errors.

**Impact.**
- This aligns with the branch history: a loss-engineering sprint moved results from ~0.66 to ~0.69 and then into the 0.71–0.72 range.

#### Problem D — Compute and memory constraints (DeBERTa-large is heavy)
**Symptom.**
- DeBERTa-v3-large requires small batch sizes on limited VRAM / Apple Silicon MPS.
- Out-of-memory issues can force unstable training or prevent experimentation entirely.

**What we changed.**
- Used **batch_size=2** with **gradient accumulation (8 steps)** to reach effective batch size 16 (from `semeval_2026_task5.ipynb` config).
- Kept **max_length=512** as a context/VRAM tradeoff and used aggressive memory clearing (notebook).
- In inference, documented the need for small batch sizes (`DEBERTA_PREDICTIONS_GUIDE.md` recommends batch_size=2, even 1 if needed).

**Impact.**
- Allowed consistent training and reproducible inference without requiring high-end GPUs.

#### Problem E — LLM augmentation did not scale (rate limits + token/cost)
**Symptom.**
- LLM-generated AmbiStory examples were valuable because they match task structure, but generation hit:
  - **rate limiting (429s)**,
  - high prompt length and output constraints (JSON formatting + two story versions).

**What we changed (engineering mitigations in code).**
- Implemented small delays between calls (`time.sleep(0.5)`).
- Implemented exponential backoff retries (`time.sleep(2 ** attempt)`).
- Bounded generation size (`max_tokens=1500` for OpenAI; `max_output_tokens=2000` for Gemini).

**Impact.**
- Enabled small-batch high-quality augmentation, but generation volume remained limited; we treated LLM data as a supplement rather than the primary scaling path.

### 7) Experimental progression (what changed, grounded in `fews` branch history)
Our development followed an iterative loop: implement → evaluate → fix the bottleneck.
From the `fews` branch commit subjects, the performance trajectory is:

- early working baseline around **0.5 correlation / 0.65 Within‑SD** (`da70ecf`)
- improved setup reaching **0.66 / 0.82** (`4589bdc`, 2025‑12‑17)
- loss-engineering sprint pushing to **0.69 / 0.83** (`cefd6ed`, 2025‑12‑18)
- addition of Huber variants (`e49379d`, `732eed1`, 2025‑12‑18)
- dynamic CORAL↘ + Huber↗ weighting + analysis plots (`96ce7fa`, 2025‑12‑18)
- refinements yielding **0.71–0.72** range (`37bb7cc`, `4107954`, `e97320c`, 2025‑12‑20)
- best run **~0.73 / ~0.84** (`36d0973`, 2025‑12‑21), followed by fixes ensuring error computations were correct (`f578e81`)

This history supports the narrative that the major gains came from **loss design + training stability**, not just swapping model backbones.

### 8) Results and stability analysis (dev set)
Our strongest dev performance is achieved by the DeBERTa+LoRA+CORAL hybrid training described above. The best checkpoint saved in `outputs/` is:
- **Spearman correlation**: **0.7264**
- **Accuracy Within SD**: **0.8350**
- file: `outputs/deberta_coral_20251220_spear07264_acc08350.pt`

We also observed stability near the best point: multiple checkpoints in `outputs/` on the same date achieve Spearman in the **0.71–0.72** range (e.g., 0.7198, 0.7150, 0.7101). This suggests the gains are robust to minor training noise and not a single lucky spike.

### 9) Supporting contributions: augmentation work (data-side engineering)
Because the official training set is small and dev is close to zero-shot on lemmas, we explored augmentation along two axes:

**(A) FEWS-based augmentation (structured, balanced).**
In `Experiment-1/EXPERIMENT_1_GUIDE.md` and scripts under `Experiment-1/`, we curated a large balanced dataset by:
- sourcing examples from FEWS `raw/` files (to avoid redundancy in FEWS train/dev/test subsets),
- enforcing **multi-sentence context** (not just long text) using a sentence-boundary heuristic that cleans citations/abbreviations,
- generating **equal negatives** by swapping an incorrect sense definition from the same homonym into the same context,
- removing artifacts (URLs, HTML entities, wiki markup).

The best curated artifact is:
- `Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json` (~66K balanced)

**(B) LLM-generated AmbiStory augmentation (structure-matched, but hard to scale).**
We built an AmbiStory-style generator in `generate_ambistory_fews.py` and a validator in `validate_ambistory_generation.py`. The generator:
- chooses homonym sense pairs from FEWS senses,
- prompts an LLM to create *ambiguous precontexts mentioning both senses* and a subtle ending,
- creates both positive and negative examples from each generated story.

In practice, scaling was constrained by:
- API rate limits (429s),
- high token/cost due to long prompts and JSON formatting requirements.

Engineering mitigations already present in the generator:
- small per-request delay (`time.sleep(0.5)`),
- exponential backoff retries (`time.sleep(2 ** attempt)`),
- OpenAI generation uses `max_tokens=1500`,
- Gemini path uses `max_output_tokens=2000`.

We therefore treated LLM augmentation as a **small-batch quality supplement** rather than a high-volume dataset.

## Repro / Submission (what we actually run)

### Generate predictions (CORAL + LoRA auto-detection)
Use `generate_predictions.py` with your best checkpoint:

```bash
python3 generate_predictions.py \
  --test_path data/dev.json \
  --model_path outputs/deberta_coral_20251220_spear07264_acc08350.pt \
  --model_name microsoft/deberta-v3-large \
  --output_file predictions_deberta.jsonl \
  --batch_size 2 \
  --pooling cls
```

### Package for submission (CodaBench expects `predictions.jsonl` inside zip)
```bash
cp predictions_deberta.jsonl predictions.jsonl
zip predictions_deberta.zip predictions.jsonl
```


