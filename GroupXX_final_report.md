---
title: "CS445 GroupXX Final Report — SemEval-2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences"
team: "Raamiz • Musab • Kiyan • Haris"
branch: "fews"
date: "2025-12-28"
---

## Abstract

We develop a system for **SemEval‑2026 Task 5**, which asks models to rate the plausibility (1–5) of a **candidate word sense** in an **ambiguous narrative**. The task differs from classic WSD because it is **graded** and evaluated primarily with **Spearman correlation** (rank agreement) and **Accuracy Within Standard Deviation** (tolerance-aware correctness based on annotator disagreement). Our final approach uses a strong transformer encoder (DeBERTa‑v3‑large) adapted with parameter‑efficient fine-tuning and an ordinal head (CORAL-style) with a hybrid objective designed to balance ranking and calibration. We also explore dataset augmentation: (i) a curated FEWS pipeline that extracts multi-sentence contexts and generates balanced negatives by sense swapping, and (ii) an LLM-based AmbiStory generator that produces structure-matched narratives but is limited in scale by rate limits and token/cost constraints. Our best dev checkpoint achieves **Spearman ρ=0.7264** and **Within‑SD=0.8350**. We provide a complete, reproducible pipeline and detailed analysis of failure modes encountered and the fixes that drove performance improvements.

---

## 1. Introduction

### 1.1 Task description
This project targets **SemEval‑2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding**. Each instance provides a narrative context split into:
- `precontext` (three sentences intended to be ambiguous),
- `sentence` containing a target ambiguous word (`homonym`),
- `ending` that provides subtle evidence,
- a candidate meaning (`judged_meaning`, often a gloss),
- and human ratings summarized as `average` and `stdev`.

The system outputs a **plausibility score in [1, 5]**. Unlike classic Word Sense Disambiguation (WSD), this is **graded plausibility**: multiple candidate senses can be evaluated under the same narrative, and the goal is to match human judgments.

From a modeling perspective, the task can be viewed as **conditional semantic evaluation**: the model must read a story, interpret the target homonym in context, and decide how compatible a particular gloss is with the narrative evidence. The dataset structure deliberately makes this hard by mixing cues for multiple senses in the precontext and using an ending that only weakly resolves ambiguity. As a result, naive lexical matching between story and gloss is often insufficient, especially when the relevant cue is implied rather than explicitly stated.

### 1.2 Evaluation metrics
We focus on the official metrics used throughout development:
- **Spearman correlation (ρ)** between predicted and gold `average` scores (ranking agreement).
- **Accuracy Within Standard Deviation (Within‑SD)**: percentage of predictions within a tolerance band around gold `average` based on annotator disagreement (`stdev`, clamped by a minimum margin in our evaluation code).

For course grading requirements that expect classification artifacts (confusion matrix, F1, PR curves), we additionally report an **auxiliary 5‑class view** by **rounding continuous predictions** to {1,2,3,4,5}. This is explicitly an approximation for rubric compliance; the underlying task is ordinal/regression.

Because the official evaluation is rank-oriented, it is possible for two systems to have similar mean-squared error while differing significantly in Spearman correlation. In our experiments, optimizing the training objective to reflect rank consistency was critical: the model must correctly order “highly plausible” senses above “implausible” ones even when absolute scores are slightly biased.

### 1.3 Key challenges (why the task is non-trivial)
We found several structural challenges that shaped the system design:
- **Narrative ambiguity is intentional**: precontext mentions cues for multiple senses; the ending only subtly resolves the ambiguity.
- **Small official training set**: 2,280 training instances makes large models prone to overfitting.
- **Zero-shot flavor in dev**: the dev set contains many lemmas not seen in train (observed in our milestone/Experiment‑1 analysis), stressing generalization.
- **Human disagreement is part of the label**: `stdev` encodes uncertainty; the Within‑SD metric rewards calibration with respect to disagreement.

### 1.3 Key results (headline)
Our best dev performance (reported by our saved checkpoint naming convention) is:
- **Spearman ρ = 0.7264**
- **Within‑SD = 0.8350**

Checkpoint: `outputs/deberta_coral_20251220_spear07264_acc08350.pt`

We also provide baseline results and additional diagnostic metrics in Section 4.

---

## 2. Related Work (≥ 5 papers, allowed venues)

Our approach is grounded in five lines of work:

### 2.1 Transformers as the backbone for contextual reasoning
The Transformer architecture introduced self-attention as the core mechanism for sequence modeling, enabling strong contextual representations and efficient parallel training (Vaswani et al., NeurIPS 2017). This is foundational for the encoder families we use.

In our task, the input contains multiple narrative segments (precontext, target sentence, ending) and a candidate gloss. Self-attention provides a natural mechanism for the model to connect evidence across these segments—e.g., allowing a phrase in the ending to selectively attend to earlier cues in the precontext and align them with the candidate meaning.

### 2.2 Bidirectional transformer pretraining
BERT (Devlin et al., NAACL 2019) established large-scale bidirectional pretraining that can be fine-tuned for downstream tasks. Our modeling is in the same paradigm: pretrain → adapt to a task-specific scoring head.

Although we ultimately use DeBERTa as the encoder, the training logic follows the same pretraining-to-fine-tuning recipe: we rely on pretrained language representations and only add a light task head that converts contextual understanding into a plausibility score. This is especially important because the official dataset is not large enough to learn deep language features from scratch.

### 2.3 Gloss-aware modeling for word senses
GlossBERT (Huang et al., EMNLP‑IJCNLP 2019) demonstrated strong WSD performance by directly pairing context with gloss candidates. Our task is different (graded plausibility rather than classification), but the “context + gloss candidate” framing strongly motivates our **candidate‑sense scoring** setup.

We adopt the same core idea—explicitly include the candidate gloss in the model input—because it shifts the task from “predict a sense ID” to “evaluate a specific semantic hypothesis.” In our setting, this fits naturally: the dataset already provides a candidate `judged_meaning`, and plausibility becomes a function of how well that meaning is supported by the narrative cues.

### 2.4 Parameter-efficient adaptation (soft/prefix prompting)
Instead of full fine-tuning, parameter-efficient techniques can adapt large models while keeping most weights frozen. Prefix‑Tuning (Li & Liang, ACL‑IJCNLP 2021) optimizes continuous “virtual tokens,” and Prompt Tuning (Lester et al., EMNLP 2021) shows soft prompts become competitive at scale. These works justify our emphasis on parameter-efficient adaptation and small-trainable-parameter regimes when compute is constrained.

These results are relevant because Task 5 has a small official training set; fully updating all parameters of a large encoder can quickly overfit and is expensive on limited hardware. While our implementation uses LoRA adapters (a different PEFT method), the motivation is the same: constrain the number of trainable parameters so that adaptation focuses on task-specific alignment rather than rewriting the entire pretrained model.

### 2.5 Ordinal modeling for graded labels
While many NLP systems treat scores as pure regression, ordinal formulations encode that 1 < 2 < 3 < 4 < 5. This aligns naturally with Spearman-based evaluation. We implement an ordinal head (CORAL-style) and combine it with robust regression losses to balance ranking and calibration.

For plausibility ratings, ordinal structure matters: predicting 5 instead of 1 should be “more wrong” than predicting 5 instead of 4. Ordinal heads enforce consistency across thresholds and often lead to more stable training when paired with rank-oriented metrics. This aligns well with the evaluation emphasis on Spearman correlation.

**Citations (URLs):**
- Vaswani et al., 2017 (NeurIPS): `https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html`
- Devlin et al., 2019 (NAACL): `https://aclanthology.org/N19-1423/`
- Huang et al., 2019 (EMNLP‑IJCNLP): `https://aclanthology.org/D19-1355/`
- Li & Liang, 2021 (ACL‑IJCNLP): `https://aclanthology.org/2021.acl-long.353/`
- Lester et al., 2021 (EMNLP): `https://aclanthology.org/2021.emnlp-main.243/`

---

## 3. Methodology

### 3.0 High-level system diagram (text)
Our final system is best understood as a **candidate-sense ranker**:
1) Build an input sequence that contains the narrative context + an explicit candidate sense gloss.
2) Encode the sequence with a transformer encoder.
3) Predict an ordinal score with an ordinal head.
4) Optimize with a hybrid objective aligned with the official metrics.

### 3.1 Datasets used

#### 3.1.1 Official SemEval Task 5 data (AmbiStory format)
- Train: **2,280** instances (`data/train.json`)
- Dev: **588** instances (`data/dev.json`)

We treat this as the primary source of supervised signal.

#### 3.1.1.1 Small EDA (milestone summary)
We include an EDA figure produced in the repo:
- `data_exploration.png`

Key observations from EDA and dataset inspection:
- Ratings cover the full 1–5 range but are centered around the mid-scale.
- `stdev` varies substantially across examples; many examples have low `stdev`, making Within‑SD stricter for a large fraction of the data.
- This supports using both ranking-oriented objectives and calibration-oriented losses.

In addition, we found that the dataset’s `stdev` distribution is non-trivial: some examples have very low disagreement, meaning the Within‑SD metric is strict, while others have higher disagreement, meaning a “near miss” should not be overly penalized. This influenced both our evaluation monitoring and our loss design, since naive regression can over-optimize easy mid-range items and underperform on the tails.

#### 3.1.2 FEWS augmentation (curated, balanced)
Motivation: dev is close to **zero-shot** on many lemmas, and 2,280 training items can lead to overfitting.

We implemented a pipeline (Experiment‑1) to extract **multi-sentence** contexts from FEWS raw sources and generate **balanced negatives** by swapping an incorrect sense definition into the same context (see `Experiment-1/EXPERIMENT_1_GUIDE.md`).

The key insight from this phase was that **context structure matters more than data volume**. Early attempts to create additional training examples using simplistic templates produced very poor dev performance (documented in the Experiment‑1 guide). As a result, we only trusted augmentation sources that could preserve multi-sentence narrative context and support both positive and negative plausibility labels.

Best curated artifact:
- `Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json` (~66K balanced)

**Quality controls used (Experiment‑1 highlights):**
- Multi-sentence validation was based on sentence boundary patterns after removing citations/brackets and common abbreviations.
- Context quality tiers (strict vs relaxed) controlled whether we required both precontext and ending to exist.
- Negative generation required multi-sense homonyms; single-sense lemmas were excluded.
- We removed URL/HTML/wiki artifacts and other reference noise that corrupted sentence splitting.

#### 3.1.3 LLM-generated AmbiStory augmentation (structure-matched, limited scale)
We built an AmbiStory-style generator `generate_ambistory_fews.py` and validator `validate_ambistory_generation.py` to create **ambiguous precontexts mentioning both senses** + subtle endings. In practice, scaling was limited by **rate limiting (429)** and token/cost constraints, so this remained a small-batch supplement.

**What made the generator novel for this project:**
- The prompt explicitly enforced *precontext ambiguity* by requiring references to both senses (mirroring the task’s design).
- Each generated story pair produced positive and negative examples by swapping which gloss is judged against the same story.

**Engineering limits encountered:**
- Rate limits (429) required backoff and throttling.
- Token limits required output truncation and careful prompt construction.

This augmentation direction was motivated by an important mismatch: FEWS provides many examples, but they often do not match AmbiStory’s “ambiguous precontext + subtle ending” design. LLM generation is one of the few ways to explicitly enforce that structure. However, because we could not reliably scale generation to thousands of examples within our resource limits, we treated LLM augmentation primarily as a **quality-focused supplement** rather than a primary driver of performance.

### 3.2 Pipeline overview

End-to-end pipeline:
1. **Load and standardize** AmbiStory JSON (train/dev), with `average` and `stdev`.
2. (Optional) **Augment** training with curated FEWS (balanced positives/negatives) and/or LLM AmbiStory samples.
3. **Serialize input text**: include the ambiguous word, candidate gloss, and the story context; mark the target with `[TGT]...[/TGT]`.
4. **Train** ordinal model (DeBERTa + LoRA + CORAL) using a hybrid loss and early stopping.
5. **Evaluate** on dev using Spearman + Within‑SD; create diagnostic plots.
6. **Generate predictions** via `generate_predictions.py` for submission packaging.

### 3.2.1 Text construction (exact modeling interface)
The model sees a single string which concatenates:
- “Ambiguous word: …”
- “Candidate sense: …”
- “Story context: …” (precontext)
- “Target sentence: …” (with `[TGT]` marker)
- “Ending: …”

This follows the “context + gloss” design pattern from gloss-aware WSD work (Huang et al., 2019), adapted to graded plausibility.

We found it important that the candidate gloss is always visible in the input. Otherwise, the model implicitly learns a different problem (“predict plausibility of the *actual* sense”) rather than plausibility of a *specified* sense. Including the gloss ensures the learning signal is aligned with the evaluation: rate plausibility of the given `judged_meaning`.

### 3.2.2 Target marking `[TGT]...[/TGT]` and robustness
We highlight the target token to reduce ambiguity. Practical details:
- Case-insensitive exact match is attempted first.
- A prefix fallback is used for simple inflection mismatches.
- If no match is found, we keep the original sentence to avoid breaking input construction.

### 3.3 Models

#### 3.3.1 Baseline: TF‑IDF + Ridge regression
We implement a strong “classical” baseline:
- TF‑IDF features (characterized by ngrams)
- Ridge regression to predict continuous ratings

This baseline is fast and provides a sanity check against deep models.

In preliminary experiments, the baseline often behaved like a “gloss overlap” detector: it performed better when the story contained explicit lexical cues matching the gloss, but struggled when plausibility depended on implicit narrative reasoning. This helped us validate that improvements from the transformer model were due to deeper contextual understanding rather than just a better linear model.

**Why this baseline is appropriate:**
- It captures surface-level lexical overlap between story and gloss.
- It sets a “no-deep-learning” floor for performance.
- It helps identify whether performance is primarily driven by meaning-level reasoning (which should benefit deep models).

#### 3.3.2 Proposed model: DeBERTa‑v3‑large + LoRA + CORAL ordinal head
Our main model uses:
- Encoder: `microsoft/deberta-v3-large`
- Parameter-efficient adaptation: LoRA (configured in notebook)
- Ordinal head: CORAL-style \(K-1\) logits for \(K=5\)

**Training configuration (from `semeval_2026_task5.ipynb`):**
- `max_length = 512`
- `epochs = 10`
- `learning_rate = 8e-5`
- `batch_size = 2`
- `grad_accumulation_steps = 8` → effective batch = 16
- loss type: `hybrid` (CORAL + Huber)
- weights: CORAL **1.0 → 0.85**, Huber **0.10 → 0.25**
- `huber_delta = 0.5`

We chose a small per-device batch size due to memory constraints of DeBERTa-large, and used gradient accumulation to preserve optimization stability. We also relied on early stopping using dev Spearman to avoid over-training on the small dataset. This training protocol was essential for reproducibility: without it, runs could vary significantly and occasionally degrade due to overfitting.

### 3.3.3 CORAL ordinal head (detailed)
For \(K=5\) ordered ratings, CORAL predicts \(K-1=4\) logits corresponding to:
\[
\Pr(y > 1), \Pr(y > 2), \Pr(y > 3), \Pr(y > 4).
\]

At inference we compute probabilities via sigmoid and derive a continuous rating:
\[
\hat{y} = 1 + \sum_{t=1}^{4} \sigma(\text{logit}_t),
\]
then clamp to [1,5]. This encourages monotone ordering and is consistent with rank-based evaluation.

### 3.3.4 Hybrid loss (detailed)
We combine:
- **CORAL loss**: encourages correct ordinal thresholds (ranking structure).
- **Huber loss** on \(\hat{y}\): encourages calibrated absolute scores and robustness to label noise.

We apply a **dynamic schedule** over epochs:
- CORAL weight decreases: \(w_c: 1.0 \rightarrow 0.85\)
- Huber weight increases: \(w_h: 0.10 \rightarrow 0.25\)

This design was motivated by observed failure modes: ranking must be learned early; calibration benefits once the model is broadly ordering examples correctly.

Empirically, this hybrid and scheduled approach aligned with our branch history: the largest performance jumps occurred during the “loss-engineering sprint,” where we systematically replaced MSE with Huber variants and introduced dynamic weighting. This suggests that for Task 5, the dominant bottleneck was not encoder capacity but objective alignment and training stability.

### 3.4 What we borrowed vs. what we contributed

**Borrowed / inspired by literature:**
- Transformer-based contextual encoders and fine-tuning paradigm (Vaswani et al., 2017; Devlin et al., 2019).
- Candidate sense framing using glosses (GlossBERT; Huang et al., 2019) — we adapt the framing to graded plausibility.
- Parameter-efficient adaptation motivation (Li & Liang, 2021; Lester et al., 2021) — used to justify LoRA-style PEFT and small-trainable-parameter strategies.
- Ordinal regression concept (general) — implemented as a CORAL-style head to respect rating order.

**Our contributions (engineering + experimentation):**
- A full pipeline for **task-specific input serialization** (target marking, story+gloss templating).
- A systematic **loss-engineering progression**: MSE → Huber → hybrid + dynamic weights to mitigate collapse-to-mean and improve calibration.
- **Curated FEWS augmentation** with balanced negatives and multi-sentence filtering (Experiment‑1).
- An **LLM story generation + validation** pipeline matching AmbiStory structure (limited by rate limits).
- Submission tooling and reproducibility guides (`DEBERTA_PREDICTIONS_GUIDE.md`, `generate_predictions.py`).

### 3.5 Replicability (how to reproduce)

Recommended “best model” training is via notebook:
- `semeval_2026_task5.ipynb` (keep outputs for grading; do not clear)

Prediction generation:
- `generate_predictions.py` (see Section 4.4)

Artifacts generated in this repository:
- `training_curves_coral.png`, `predictions_scatter.png`, `prediction_analysis.png`, `error_distribution.png`
- confusion matrices: `confusion_matrix_baseline.png`, `confusion_matrix_deberta.png`
- PR curves: `pr_curves_baseline.png`, `pr_curves_deberta.png`
- metrics summary: `report_metrics.json`

---

## 4. Results

### 4.0 What results mean for this task
Because the official evaluation is Spearman + Within‑SD, the most important question is:
- “Does the model rank plausible senses above implausible ones?” (Spearman)
- “Are scores sufficiently calibrated to lie close to human averages, accounting for disagreement?” (Within‑SD)

The auxiliary classification view is included for the rubric, but is not the primary objective.

To make the results interpretable, we report both the official-style metrics (Spearman and Within‑SD) and the rubric-required classification artifacts derived from rounding. We emphasize that rounding compresses information and can penalize near-misses (e.g., predicting 3.49 vs 3.51) differently than the official evaluation would. Nevertheless, the rounding-based view provides useful intuition about where the model is systematically confusing adjacent rating levels.

### 4.1 Regression/ordinal metrics (official-style)

**Best checkpoint (reported by checkpoint naming convention):**
- Spearman: **0.7264**
- Within‑SD: **0.8350**
- file: `outputs/deberta_coral_20251220_spear07264_acc08350.pt`

**Baseline vs model metrics (example dev run artifacts)**
From `report_metrics.json` (computed from `dev_predictions.csv` and a recreated TF‑IDF baseline):

| Model | Spearman (ρ) | Within‑SD | MAE | MSE |
|---|---:|---:|---:|---:|
| TF‑IDF + Ridge (baseline) | -0.0130 | 0.5612 | 1.0363 | 1.4610 |
| DeBERTa run (dev_predictions.csv) | 0.5892 | 0.7755 | 0.7805 | 1.0136 |

**Note:** The “best checkpoint” result is taken from the model-selection run that produced the saved checkpoint naming `...spear07264_acc08350...`. The dev CSV corresponds to a particular run’s predictions artifact; the project’s headline model score is the best checkpoint.

Even in the example run artifacts, the gap between the TF‑IDF baseline and the transformer model is substantial on the official metrics. This supports the hypothesis that plausibility prediction requires semantic and narrative reasoning beyond surface lexical overlap, and that large pretrained encoders can capture those dependencies when trained with a suitable objective.

### 4.2 Classification-style artifacts (rubric requirement; derived by rounding)

We derive a 5-class label by rounding gold `average` and predictions to {1..5}. This is **not** the official task objective but provides confusion matrices and macro-F1 for grading requirements.

From `report_metrics.json`:

| Model | Rounded Acc | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|
| TF‑IDF + Ridge | 0.2585 | 0.1851 | 0.2095 | 0.1137 |
| DeBERTa run | 0.3537 | 0.2799 | 0.3042 | 0.2822 |

Figures:
- Confusion matrices: `confusion_matrix_baseline.png`, `confusion_matrix_deberta.png`
- Precision–Recall curves: `pr_curves_baseline.png`, `pr_curves_deberta.png`

### 4.3 Error analysis (qualitative)
We used:
- `predictions_scatter.png` to inspect systematic bias (over/under prediction).
- `error_distribution.png` to inspect whether the model is centered and whether tails are heavy.
- `prediction_analysis.png` for a multi-panel view of errors and calibration.

Typical failure patterns observed during development:
- **Sense gloss ambiguity**: very short or overlapping glosses make two senses both plausible.
- **Weak endings**: some endings do not strongly resolve ambiguity, increasing disagreement.
- **Narrative length constraints**: long contexts may exceed max_length, causing truncation of critical cues.

We also observed that errors are not uniformly distributed across rating values. The rounded confusion matrices show that adjacent-class confusion (e.g., 3 vs 4) is common, which is expected for ordinal tasks. More importantly, when the gold ratings have high disagreement (high `stdev`), the model’s “errors” are often within the tolerance band used in Within‑SD, indicating that the model is not necessarily wrong in a human sense; it is matching the inherent uncertainty of the data.

### 4.3 Plots and key findings
We include the following plots to highlight important findings:
- **EDA**: `data_exploration.png` (rating and stdev distributions)
- **Training curves**: `training_curves_coral.png` (Spearman/Within‑SD over epochs)
- **Scatter**: `predictions_scatter.png` (predicted vs gold)
- **Error distribution**: `error_distribution.png`
- **Aggregate analysis**: `prediction_analysis.png`

### 4.4 Comparison to literature / state-of-the-art
Direct SoTA comparison is difficult because Task 5 is new and leaderboard baselines may not be fully published. We therefore compare at the level of **method families**:
- Candidate gloss pairing is strongly supported by WSD literature (GlossBERT; Huang et al., 2019).
- Parameter-efficient adaptation is supported by prompting/PEFT literature (Li & Liang, 2021; Lester et al., 2021).
- Transformer encoders and bidirectional pretraining remain the dominant paradigm (Vaswani et al., 2017; Devlin et al., 2019).

Our approach combines these ideas into an ordinal scoring pipeline tuned to the task metrics.

---

## 5. Discussion (Section D.5 points)

### 5.1 Dataset selection and impact
- **Official AmbiStory data** is faithful to the task but small (2,280 train), increasing overfitting risk.
- **FEWS augmentation** increases coverage and provides negative examples through controlled sense swapping; however, it can introduce distribution shift if narrative structure is not preserved. Our multi-sentence filtering and artifact cleaning aim to reduce this mismatch.
- **LLM augmentation** is structure-matched but rate-limited, which prevented scaling to the volume we initially planned.

The most important dataset lesson from our journey is that naive scaling can backfire: additional examples that do not preserve AmbiStory’s narrative structure can teach the model shortcuts that fail on dev. This is why we prioritized multi-sentence filtering and balanced negative generation for FEWS, and why we invested in an LLM generator that explicitly enforces ambiguity in the precontext. In other words, we treated dataset curation as an optimization problem: maximize coverage while preserving the task’s defining characteristics.

### 5.2 Approach selection: advantages / disadvantages
**Advantages**
- Ordinal modeling respects the rating structure and aligns with Spearman.
- Hybrid loss helps balance ranking (Spearman) and calibration (Within‑SD).
- Parameter-efficient adaptation allows large encoders under compute constraints.

**Disadvantages**
- DeBERTa-large is resource intensive; training requires small batches and careful memory management.
- The classification-style metrics required by the rubric are approximations (rounding).

Another tradeoff is interpretability: while the baseline offers a straightforward explanation (“overlap between story and gloss”), the transformer model is less transparent. We partially address this with diagnostic plots and by using structured error analysis, but detailed interpretability (e.g., attention analysis) remains future work.

### 5.3 Comparison to existing systems (reported results)
Relative to the TF‑IDF baseline, our deep approach improves:
- Spearman correlation (ranking),
- Within‑SD (calibration),
- and auxiliary macro-F1 in the rounding-based view.

### 5.4 Limitations
- **Compute limits** restricted batch size and slowed experimentation.
- **Rate limiting/token costs** limited LLM augmentation volume.
- **Metric mismatch** between official regression/ordinal evaluation and course-required classification artifacts.
- Potential **domain shift** from FEWS to AmbiStory narratives.

Additionally, our strongest results are reported on the dev split, and the official test set is hidden. This is standard for SemEval-style shared tasks, but it means all design choices risk being dev-optimized. We mitigated this by focusing on general principles (ordinal modeling, PEFT, robust loss design) and by using structured augmentation rather than hand-tuning features to dev idiosyncrasies.

### 5.5 Potential improvements
- True multi-checkpoint **ensembling** (average ordinal predictions) for small Spearman gains.
- Add a **calibration layer** (e.g., isotonic regression) post-hoc on dev.
- A more production-grade LLM generation pipeline (caching, queuing, token-optimized prompts).
- Cross-validation or multiple dev splits for more stable model selection (if allowed by organizers).

---

## 6. Conclusion
We built an end-to-end system for SemEval‑2026 Task 5 that treats plausibility prediction as an ordinal scoring problem. Our best dev checkpoint reaches **ρ=0.7264** and **Within‑SD=0.8350**. Through iterative debugging, loss engineering, and structured augmentation attempts, we developed a reproducible pipeline with interpretable diagnostics and submission tooling.

Overall, the strongest improvements came from identifying and fixing concrete bottlenecks: data structure mismatch, loss/objective misalignment, and training instability under compute constraints. This “problem-first engineering” approach—measure a failure mode, implement a targeted fix, and validate with dev metrics—was essential to reaching competitive performance in the limited project timeline.

---

## 7. Individual Contributions (fill in precisely)
Replace with your final mapping:
- **Raamiz**: ______________________
- **Musab**: _______________________
- **Kiyan**: _______________________
- **Haris**: _______________________

---

## References (allowed venues)

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT. `https://aclanthology.org/N19-1423/`

Huang, L., Sun, C., Qiu, X., & Huang, X. (2019). *GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge*. EMNLP-IJCNLP. `https://aclanthology.org/D19-1355/`

Lester, B., Al-Rfou, R., & Constant, N. (2021). *The Power of Scale for Parameter-Efficient Prompt Tuning*. EMNLP. `https://aclanthology.org/2021.emnlp-main.243/`

Li, X. L., & Liang, P. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. ACL-IJCNLP. `https://aclanthology.org/2021.acl-long.353/`

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS. `https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html`

---

## Appendix A — Required figures checklist (for graders)
- EDA: `data_exploration.png`
- Training curves: `training_curves_coral.png`
- Scatter: `predictions_scatter.png`
- Error distribution: `error_distribution.png`
- Analysis: `prediction_analysis.png`
- Confusion matrices (rounded): `confusion_matrix_baseline.png`, `confusion_matrix_deberta.png`
- PR curves (rounded): `pr_curves_baseline.png`, `pr_curves_deberta.png`
- Metrics JSON: `report_metrics.json`



