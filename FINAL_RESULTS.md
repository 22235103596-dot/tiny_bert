# TinyBanglaBERT — Full Results Chapter (corrected, Nov 2026)

**Dataset:** BanglaBait (12,044 train / 1,506 val / 1,506 test, seed 42; 34.8% clickbait) • **Teacher:** `csebuetnlp/banglabert` (110.62M) • **Student:** 4-layer, 7.42M parameters • **Cross-domain:** BaitBuster-Bangla (10,000 headlines).

> All numbers here come from the re-run after the distillation fix (see §0). Figures in `experiments/figures/` must be regenerated with `python -m scripts.make_figures`.

## 📌 Headline numbers
- Teacher: **83.13%** accuracy, macro-F1 **0.8124**.
- Best student (R3, pretrained init + KD): **80.52 ± 0.37%**, macro-F1 **0.787 ± 0.004** (3 seeds) → **96.9% of teacher accuracy**.
- **6.7% of the parameters** (7.42M vs 110.62M), **11.6× faster** CPU inference (PyTorch, B=1).
- Pretrained initialisation: **+4.5 pts**; distillation: **+0.9 to +1.2 pts, not statistically significant** (p = 0.136 / 0.071).
- Teacher is significantly better than the student (McNemar p = 0.018). No cross-domain advantage for the student.

## 0. What was wrong in the old results
`DistillationTrainer` needs a `teacher_logits` column, but Hugging Face `Trainer` defaults to `remove_unused_columns=True` and silently dropped it, so training reduced to class-weighted cross-entropy; "distilled" and "supervised" runs were identical and the (T, α) grid was flat (0.6993). Fix: `remove_unused_columns=False` + a RuntimeError guard when logits are missing (`tests/test_kd_fixes.py`). The student is now initialised from pretrained `banglabert_small` layers [2, 5, 8, 11]; the PCA "teacher-projected" init was no better than random and was removed. The dataset has **0 unlabeled rows**, so no unlabeled distillation was done. Separately, cross-domain precision/recall and confusion matrices were swapped by an argument-order bug (fixed, `tests/test_metric_argument_order.py`).

## 1. In-domain results
| Model | Params | Acc (%) | Macro-F1 |
|---|---|---|---|
| Teacher | 110.62M | 83.13 | 0.8124 |
| Student R3 (mean ± std, 3 seeds) | 7.42M | 80.52 ± 0.37 | 0.787 ± 0.004 |
| Student R3 (seed 42; used for McNemar/efficiency/cross-domain) | 7.42M | 80.88 | 0.7914 |

Seed-42 student confusion matrix (rows = true, cols = predicted; [not-CB, CB]): [[826, 156], [132, 392]]. Distillation hyper-parameters: T = 1.0, α = 0.1 (best of a 5×5 validation grid, short runs). Training: lr 1e-4, batch 64, ≤10 epochs, early stopping on val macro-F1, seeds 42/43/44.

## 2. 2×2 ablation (mean ± std over 3 seeds)
| Row | Init | Distillation | Acc (%) | Macro-F1 | % of teacher acc |
|---|---|---|---|---|---|
| R0 | random | no | 75.19 ± 0.55 | 0.739 ± 0.006 | 90.4 |
| R1 | pretrained | no | 79.66 ± 0.43 | 0.782 ± 0.004 | 95.8 |
| R2 | random | yes | 76.38 ± 0.79 | 0.744 ± 0.004 | 91.9 |
| R3 | pretrained | yes | 80.52 ± 0.37 | 0.787 ± 0.004 | 96.9 |

**Reading:** initialisation is the main driver (≈ +4.5 pts); distillation gives ≈ +1 pt, about two seed-standard-deviations.

## 3. Efficiency (Kaggle CPU, B = 1, L = 128, 200 iterations)
| Model | Params | Size | CPU latency |
|---|---|---|---|
| Teacher PyTorch FP32 | 110.62M | 422.2 MB | 154.4 ms |
| Student PyTorch FP32 | 7.42M | 28.3 MB | 13.4 ms (**11.6×**) |
| Student ONNX FP32 | 7.42M | 28.4 MB | 7.5 ms |
| Student ONNX INT8 | 7.42M | 7.3 MB | 8.2 ms (smaller, not faster) |

Use the PyTorch-vs-PyTorch ratio as the headline; ONNX ratios compare different runtimes. Peak-memory numbers are unreliable and omitted. Measured on Kaggle, not an edge device.

## 4. Explainability (SHAP / LIME)
Re-run student (R3, seed 42) on 15 test headlines chosen to cover TP/TN/FP/FN (so this sample is *not* representative of accuracy):

| Metric (SHAP vs LIME) | Value |
|---|---|
| Mean top-3 token Jaccard overlap | 42.0% (median 50.0%) |
| Mean top-3 Jaccard, clickbait tokens | 33.4% |
| Sign (polarity) agreement | 54.9% |

**Reading:** agreement is only moderate, and sign agreement is close to the 50% expected by chance. We therefore do **not** claim that the two methods "converge"; explanations of this compact model should be treated as indicative, not definitive. Agreement varies by headline: for *"মূল্যায়ন পদ্ধতির পরিবর্তন জরুরী: শিক্ষামন্ত্রী"* (not clickbait, correctly predicted) both methods push towards not-clickbait with 100% sign agreement and top tokens *মূল্যায়ন, পদ্ধতির, পরিবর্তন*; for the viral-video headline *"বাঁশ দিয়ে ফাঁদ পেতে … ভিডিও তুমুল ভাইরাল !"* (clickbait, predicted 97.8%) both flag *ভাইরাল* as clickbait evidence but their top-3 sets do not overlap (SHAP: *পদ্ধতিতে, তুমুল, ধরার*; LIME: *ভাইরাল, ভিডিও, অভিনব*), Jaccard 0.00. Source: `results/explainability/` of the final-evaluation notebook (generated 2026-11-18). The generic "qualitative insights" paragraph in the auto-generated report is boilerplate, not derived from this data, and is not used.

## 5. McNemar exact tests (seed-42 models, N = 1,506)
| A vs B | A-only / B-only correct | p | Reading |
|---|---|---|---|
| Teacher vs R3 | 114 / 80 | 0.018 | teacher better |
| R3 vs R1 (KD, pretrained) | 74 / 56 | 0.136 | n.s. |
| R2 vs R0 (KD, random) | 101 / 76 | 0.071 | n.s. |
| R3 vs R2 (init, with KD) | 136 / 82 | < 0.001 | significant |
| R1 vs R0 (init, no KD) | 157 / 96 | < 0.001 | significant |

## 6. Cross-domain, zero-shot (BaitBuster-Bangla, 10,000: 5,644 CB / 4,356 non-CB)
| Model | Acc in → cross | Macro-F1 (cross) | CB precision | CB recall | TN / FP / FN / TP |
|---|---|---|---|---|---|
| Teacher | 83.13 → 77.03 (−6.1) | 0.769 | 0.970 | 0.612 | 4251 / 105 / 2192 / 3452 |
| Student R3 | 80.88 → 73.75 (−7.1) | 0.735 | 0.937 | 0.573 | 4139 / 217 / 2408 / 3236 |

Both models are **conservative** (high precision, low clickbait recall): they miss ~40% of BaitBuster clickbait, which suggests the two datasets label clickbait differently. The student does not generalise better than the teacher.

## 7. Caveats
Single teacher and split; one seed for McNemar / efficiency / cross-domain; distillation effect not significant at 3 seeds; Kaggle latency; label-definition mismatch across datasets.

## 8. Figures
`experiments/figures/fig1_ablation_accuracy.png` (ablation with error bars) and `fig2_efficiency.png`, produced by `python -m scripts.make_figures`.
