---
title: "GroupXX Final Presentation — SemEval-2026 Task 5 (CS445)"
team: "Raamiz • Musab • Kiyan • Haris"
duration: "25 minutes talk + 5 minutes Q&A"
branch: "fews"
---

## Slide 1 — Title (All)
**SemEval‑2026 Task 5**: Rating Plausibility of Word Senses in Ambiguous Sentences  
GroupXX — CS445

Headline dev (best checkpoint): **Spearman 0.7264**, **Within‑SD 0.8350**  
Checkpoint: `outputs/deberta_coral_20251220_spear07264_acc08350.pt`

---

## Slide 2 — Agenda (All)
- Introduction + task definition
- Related work (≥5 citations)
- Dataset + EDA
- Methodology (pipeline + models)
- Results (official + rubric metrics)
- Discussion + limitations + next steps
- Individual contributions + submission status

---

## Slide 3 — Task definition (Raamiz)
- Input: (`precontext`, `sentence`, `ending`, `homonym`, `judged_meaning`)
- Output: plausibility rating in **[1,5]**
- Why hard: ambiguity + narrative cues + small training set + dev is near zero-shot on lemmas

---

## Slide 4 — Evaluation metrics (Raamiz)
- Official:
  - **Spearman ρ** (ranking)
  - **Accuracy Within SD** (tolerance based on `stdev`)
- Rubric-required extras (auxiliary):
  - confusion matrix + macro P/R/F1 via **rounding** predictions to 1..5
  - PR curves (one-vs-rest; confidence proxy)

---

## Slide 5 — Related work (Musab)
**(Explain how each influenced our approach)**
- Transformers (Vaswani et al., NeurIPS 2017)
- BERT (Devlin et al., NAACL 2019)
- Gloss-based WSD with context+gloss pairing (Huang et al., EMNLP-IJCNLP 2019)
- Parameter-efficient adaptation: Prefix-Tuning (Li & Liang, ACL-IJCNLP 2021)
- Parameter-efficient adaptation: Prompt Tuning (Lester et al., EMNLP 2021)

---

## Slide 6 — Dataset (Kiyan)
- Official data:
  - Train **2280**, Dev **588**
- Fields: `average` + `stdev` + narrative components
- Why `stdev` matters: humans disagree → Within‑SD metric

---

## Slide 7 — EDA (Kiyan)
Show `data_exploration.png`:
- rating distribution (train/dev)
- stdev distribution
- average vs stdev scatter

Talking points:
- mean around mid-scale; non-trivial variance
- many low stdev items → stricter Within‑SD requirement

---

## Slide 8 — Problems encountered (Haris)
**(Tell the story: symptom → fix → impact)**
- Overfitting / generalization gap (small train)
- Data mismatch from naive augmentation (templates)
- Collapse-to-mean in regression training
- Compute constraints (DeBERTa-large memory)
- LLM augmentation scaling blocked by 429 + token cost

---

## Slide 9 — Pipeline overview (Haris)
Diagram-style bullets:
1) load data → 2) build text input (story+gloss, `[TGT]`) → 3) train model → 4) evaluate → 5) generate predictions zip

Mention key repo files:
- `semeval_2026_task5.ipynb`
- `generate_predictions.py`
- `DEBERTA_PREDICTIONS_GUIDE.md`

---

## Slide 10 — Baseline model (Musab)
TF‑IDF + Ridge regression baseline:
- quick to run
- sanity-check performance
Report baseline metrics from `report_metrics.json`:
- Spearman **-0.013**, Within‑SD **0.561**

---

## Slide 11 — Main model architecture (Raamiz)
DeBERTa‑v3‑large + LoRA + CORAL ordinal head
- LoRA: r=32, alpha=128, dropout=0.1
- CORAL: 4 logits for 5-class ordinal scale

---

## Slide 12 — Loss & training strategy (Raamiz)
Hybrid objective:
- CORAL (ranking/ordinal)
- Huber (calibration)
- dynamic weighting across epochs (CORAL↓, Huber↑), delta=0.5

Training config (from notebook):
- max_length=512, lr=8e‑5
- batch=2, grad_accum=8 (effective 16)
- epochs=10 + early stopping

---

## Slide 13 — Augmentation attempts (Kiyan)
FEWS structured augmentation:
- multi-sentence filtering + balanced negatives
- artifact cleaning
- dataset artifact: `Experiment-1/final-ish/combined_clean_full_relaxed_balanced.json` (~66K)

LLM AmbiStory augmentation:
- generator + validator pipeline
- scaling blocked by 429 + token/cost; mitigations (delay + exponential backoff)

---

## Slide 14 — Results (official metrics) (Musab)
Headline (best checkpoint):
- Spearman **0.7264**
- Within‑SD **0.8350**

Also show example run metrics (from `report_metrics.json`):
- DeBERTa run Spearman **0.589**, Within‑SD **0.776**

Explain why: best checkpoint result from saved model-selection run; CSV corresponds to a specific run artifact.

---

## Slide 15 — Visual diagnostics (Haris)
Show:
- `training_curves_coral.png`
- `predictions_scatter.png`
- `error_distribution.png`

What to say:
- training stabilized after loss changes
- scatter closer to diagonal
- error distribution tighter

---

## Slide 16 — Confusion matrix & F1 (auxiliary) (Musab)
Explain: rubric-required classification view via rounding to 1..5

Show:
- `confusion_matrix_baseline.png`
- `confusion_matrix_deberta.png`

Report macro metrics (from `report_metrics.json`):
- baseline macro F1 **0.114**
- DeBERTa macro F1 **0.282**

---

## Slide 17 — Precision–Recall curves (auxiliary) (Musab)
Show:
- `pr_curves_baseline.png`
- `pr_curves_deberta.png`

Explain: one-vs-rest PR with confidence proxy score = -|pred-k|

---

## Slide 18 — Discussion (All, round-robin)
Cover required discussion points:
- dataset choice impact
- approach advantages/disadvantages
- comparison to cited methods/SoTA families
- limitations
- improvements if more time/resources (ensemble, calibration, better LLM pipeline)

---

## Slide 19 — Submission status + proof (Raamiz)
Fill one of:
- **Submitted**: screenshot/email + team name evidence
- **Not open yet**: mention deadline Jan 10; plan for submission proof ASAP

---

## Slide 20 — Individual contributions (All)
Fill with specifics (equal workload):
- Raamiz: ______
- Musab: ______
- Kiyan: ______
- Haris: ______

---

## Slide 21 — References (All)
- Vaswani et al., NeurIPS 2017 — `https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html`
- Devlin et al., NAACL 2019 — `https://aclanthology.org/N19-1423/`
- Huang et al., EMNLP-IJCNLP 2019 — `https://aclanthology.org/D19-1355/`
- Li & Liang, ACL-IJCNLP 2021 — `https://aclanthology.org/2021.acl-long.353/`
- Lester et al., EMNLP 2021 — `https://aclanthology.org/2021.emnlp-main.243/`

---

## Timing plan (enforced)
Target ~25 minutes total:
- Raamiz: Slides 3–4, 11–12, 19 (≈6–7 min)
- Musab: Slides 5, 10, 14, 16–17 (≈6–7 min)
- Kiyan: Slides 6–7, 13 (≈5–6 min)
- Haris: Slides 8–9, 15, 18–20 (≈6–7 min)


