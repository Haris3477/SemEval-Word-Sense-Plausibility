## SemEval 2026 Task 5 – Current Model Approach, Results, and Improvement Plan

### 1. Goal

- **Task**: Rate the plausibility (1–5) of a word sense given an AmbiStory-style context.
- **Target metrics**:
  - **Spearman correlation** > **0.70**
  - **Accuracy Within Standard Deviation (Within SD)** > **70%**

---

### 2. Current Modeling Approach

#### 2.1 Data Sources

- **SemEval official data**
  - `data/train.json` (train) and `data/dev.json` (dev).
  - Already in AmbiStory format: `precontext`, `sentence`, `ending`, `homonym`, `judged_meaning`, `average`, `stdev`.
- **FEWS augmentation**
  - Loaded from `data/fews/fews/` via `load_fews_dataframe`.
  - FEWS senses are converted into **AmbiStory-compatible** rows using `_build_fews_row`.
  - Homonyms are normalized to **bare lemmas**.
  - Currently ~**930** FEWS-derived synthetic samples are added to training.
- **LLM-generated AmbiStory data**
  - Generated with `generate_ambistory_fews.py` into `data/llm_generated_ambistory.json`.
  - Validated using `validate_ambistory_generation.py`.
  - Loaded in `semeval_task5_main.py` when `--llm_ambistory_path` is provided (e.g. 96 examples in the latest run).

All sources are concatenated into a single training DataFrame with a `source` field to track origin and `weight` for per-source reweighting.

#### 2.2 Text Construction (`create_text_input`)

For each row, we build a single text string that the transformer sees:

- **Story context first**:
  - `precontext` (3 sentences, intentionally ambiguous between senses).
  - `sentence` with the homonym highlighted using `[TGT]...[/TGT]` via `highlight_target`.
  - `ending` (subtle clue towards one sense, still slightly ambiguous).
- **Sense evaluation prompt**:
  - A natural language instruction like:  
    **“Task: Rate how plausible the meaning '<sense gloss>' is for the word '<homonym>' in the story above.”**
  - Very long glosses are truncated to keep inputs manageable.
- **Extra semantic info** (if available):
  - `example_sentence` → prefixed with `"Example: …"`.
  - `sense_tags` → appended as `"Tags: …"`.

Recent improvements:

- Added a `normalize_text` helper to:
  - Strip and normalize whitespace.
  - Fix spacing around punctuation.
  - Ensure the final `text` field is clean and consistent.
- Ensured all text columns are filled and cast to `str` before building `text`.

#### 2.3 Model Architecture (`PlausibilityModel`)

- **Backbone**: `roberta-base` loaded via `AutoModel`.
- **Pooling**:
  - Configurable (`--pooling`), commonly using **`weighted` pooling**:
    - Learnable attention over token embeddings via a small linear layer.
- **Regression head**:
  - `LayerNorm(hidden)`
  - `Linear(hidden → 256)`
  - `GELU` activation
  - `Dropout` (reduced vs earlier versions)
  - `Linear(256 → 1)` to produce the plausibility score.
- **Initialization tweaks**:
  - Final layer bias initialized to **3.0** (approximate mean of target scores).
  - Final layer weights initialized with **`xavier_uniform_`** (gain ≈ 0.5) to encourage variance and avoid collapse to a constant prediction.
- **Parameter groups / learning rates**:
  - Encoder parameters: **`--learning_rate`** (e.g. `3e-5`).
  - Regressor head: **10× higher LR** than encoder (e.g. `3e-4`) for faster adaptation.

#### 2.4 Loss Functions and Optimization

For each batch, we compute a **composite loss**:

- **Primary regression loss**: `nn.MSELoss` between predictions and (optionally smoothed) targets.
- **Spearman loss**: `spearman_loss(preds, targets)`  
  - Differentiable approximation of **(1 – Spearman correlation)** to push the model towards higher rank correlation.
- **Accuracy Within SD loss**: `accuracy_within_sd_loss(preds, targets, stdevs)`
  - Computes margin = `max(stdev, 1.0)` per example.
  - Encourages predictions to fall **within this margin** of the human mean.
  - Adds a smooth quadratic penalty when outside the margin.
- **Variance penalty**:
  - Compares prediction std vs target std.
  - Penalizes prediction distributions that are too flat (collapse to mean) or too extreme.
- **Label smoothing**:
  - Adds small Gaussian noise (e.g. σ=0.02) to targets for robustness.

Current weighting (accuracy-focused run):

- **Total loss** ≈  
  `2.0 * accuracy_loss + 1.0 * spearman_loss + 0.5 * mse_loss + variance_penalty`

#### 2.5 Training Setup

- **Device**: Apple Silicon **MPS GPU** (`mps`), confirmed in logs.
- **Typical run (accuracy_test)**:
  - Epochs: **3**
  - Batch size: **8**
  - Max train samples: **1000** (subset of full training set) for faster experimentation.
  - Max dev samples: **200** (similarly subsampled).
  - FEWS examples: up to **500** sampled, then ~930 FEWS → AmbiStory rows after processing.
  - LLM AmbiStory examples: ~96 in the last test.
  - Optimizer: `AdamW` with linear warmup schedule.
  - Gradient accumulation: used to simulate larger effective batch size.
  - Sample weights: currently **neutralized to 1.0** per sample (to debug collapse), with optional multipliers per source (`--fews_weight`, `--llm_ambistory_weight`).

---

### 3. Current Results

#### 3.1 Data health check

From a recent diagnostic run on `data/train.json`:

- **No missing fields** for `homonym`, `judged_meaning`, `sentence`, `precontext`, `ending`, `average`, `stdev`.
- **Text lengths**:
  - Average `text` length ≈ **480 chars**.
  - All examples non-empty.
- **No duplicate `text` entries** with conflicting scores.
- **Score distribution**:
  - Mean ≈ **3.13**, std ≈ **1.18**, range **[1.0, 5.0]**, ~36 unique values.
- **Stdev distribution**:
  - Mean ≈ **0.95**, std ≈ **0.52**, range **[0.0, 2.19]**.
  - Many stdevs < 1.0 (we clamp margins to at least 1.0 in the loss/evaluation).

Conclusion: **The raw data is healthy**; the main limitations are on the modeling/optimization side.

#### 3.2 Latest Transformer Dev Performance

From `accuracy_test.log` and `results_summary.csv` (accuracy-focused 3‑epoch run with FEWS + LLM data):

- **Epoch 1 (dev)**:
  - Spearman correlation ≈ **0.039**
  - Accuracy Within SD ≈ **0.565** (56.5%)
  - MSE ≈ **1.43**
  - MAE ≈ **1.02**
- **Final summary (saved in `results_summary.csv`)**:
  - Transformer row:  
    - **Spearman** ≈ **0.009**  
    - **Accuracy Within SD** ≈ **0.54** (54%)  
    - **MAE** ≈ **1.05**  
    - **MSE** ≈ **1.49**

These numbers are **well below** the desired targets (**0.7+ Spearman, 70%+ Accuracy Within SD**).

Note: Baseline metrics in the CSV are placeholders (0 values) for this run, because baseline computation was skipped/disabled; they are not meaningful for comparison here.

---

### 4. Diagnosis – Why Performance Is Low

Based on the architecture, logs, and metrics, the main issues look like:

- **1. Undertraining / limited epochs**
  - Only **3 epochs** on a subset of data (1000 train / 200 dev), despite using a deep encoder.
  - Logs show the prediction distribution is still stabilizing (batch-wise Spearman varies widely).
- **2. Prediction variance still not matching target variance**
  - Batch logs show prediction std often much lower than target std.
  - This limits both Spearman and Accuracy Within SD (model hovers near the mean).
- **3. Mixed data sources without tuned weighting**
  - SemEval + FEWS + LLM AmbiStory are all combined with simple weights; optimal mix is not yet explored.
  - FEWS examples, while AmbiStory-like, may still differ stylistically from true SemEval AmbiStory data.
- **4. Loss weighting may not be optimal yet**
  - Strong emphasis on Accuracy Within SD and Spearman, but the exact coefficients may not be ideal.
  - Label smoothing plus composite losses can sometimes blur gradients if not well balanced.
- **5. Pooling and architecture choices are not fully explored**
  - Using only one pooling strategy at a time (`weighted` / `cls` / `mean`), no ablation results yet.
  - Only `roberta-base` has been tried so far.

Overall: **the data is not the bottleneck**; the primary opportunities are in **training regime**, **loss/weight tuning**, and potentially **architecture scaling**.

---

### 5. Concrete Improvement Plan

Below is a prioritized list of next steps to move towards **Spearman > 0.7** and **Accuracy Within SD > 70%**.

#### 5.1 Stabilize and Strengthen Training

- **Train longer on full data**:
  - Remove or increase `--max_train_samples` / `--max_dev_samples` (use full SemEval train+dev).
  - Increase epochs from **3 → 10–15** (with early stopping on dev Spearman).
- **Tune learning rates**:
  - Try slightly higher encoder LR (e.g. `5e-5`) while keeping head at 10× (e.g. `5e-4`).
  - Alternatively, use a small LR range test on a short run to find a better starting LR.
- **Adjust label smoothing**:
  - Experiment with **no smoothing** or smaller values (e.g. 0.0–0.01) to preserve target signal.

#### 5.2 Optimize Loss Balancing

- **Systematically sweep loss weights**:
  - Start from a simpler combination, e.g.:  
    `loss = mse + spearman_loss + accuracy_loss`  
    then add variance penalty once predictions stop collapsing.
  - Try different accuracy loss weights (e.g. 1.0, 2.0, 3.0, 5.0) and measure the effect on:
    - Dev Spearman
    - Dev Accuracy Within SD
    - Prediction std vs target std
- **Monitor batch-level stats** every N steps:
  - Predicted range/mean/std
  - Actual range/mean/std
  - Batch Accuracy Within SD
  - This is already partially implemented; use it to guide whether variance penalty or loss weights need adjustment.

#### 5.3 Data and Weighting Experiments

- **Curriculum-style training**:
  - Phase 1: Train on **SemEval only** until convergence.
  - Phase 2: Fine-tune with **SemEval + FEWS + LLM AmbiStory** for robustness.
- **Source weighting sweeps**:
  - Vary `--fews_weight` (e.g. 0.5, 1.0, 2.0) and `--llm_ambistory_weight` (e.g. 0.5, 1.0, 2.0).
  - Track dev metrics and see whether more/less synthetic data helps or hurts.
- **Tighten LLM data quality**:
  - Use `validate_ambistory_generation.py` to filter out any low-quality or non-ambiguous examples.
  - Optionally create a “high-confidence” subset of LLM stories and upweight only those.

#### 5.4 Architecture and Pooling Ablations

- **Pooling strategy ablation**:
  - Run short experiments comparing:
    - `--pooling cls`
    - `--pooling mean`
    - `--pooling weighted`
  - Pick the best-performing pooling before longer training runs.
- **Model scale** (if compute allows):
  - Try **`roberta-large`** with smaller batch size and maybe fewer epochs to test gains in Spearman/Accuracy.
  - If improvements are clear, schedule a longer `roberta-large` run.

#### 5.5 Evaluation and Monitoring

- **Always log and save**:
  - `results_summary.csv` for each major run.
  - Training logs (`*.log`) with batch-level diagnostics.
- **Plot diagnostics**:
  - Use existing plotting utilities in `semeval_task5_main.py` (when `--skip_plots` is disabled) to inspect:
    - Prediction vs. target scatter (`predictions_scatter.png`).
    - Error distributions (`error_distribution.png`, `rating_distributions.png`, etc.).
- **Targeted debugging**:
  - If certain score ranges (e.g. very low/high plausibility) are especially poor, consider:
    - Balanced sampling across score bins.
    - Small specialized fine-tunes focused on those ranges.

---

### 6. Summary

- **Current state**:
  - Strong, well-structured **data pipeline** (SemEval + FEWS + LLM AmbiStory).
  - A reasonable **RoBERTa-based regression model** with thoughtful initialization and composite losses.
  - **Dev performance** is currently around **0.01–0.04 Spearman** and **54–57% Accuracy Within SD**, below competition targets.
- **Key insight**: The **data is not the main problem**; the critical levers now are **training duration, loss weighting, source weighting, and minor architectural choices**.
- **Path forward**: Systematically extend training, tune the composite loss, experiment with data/source weights, and run small ablations on pooling and model size to climb towards **Spearman > 0.7** and **Accuracy Within SD > 70%**.


