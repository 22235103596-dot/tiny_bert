# TinyBanglaBERT: A Lightweight Knowledge-Distilled Transformer for Explainable Bangla Clickbait Headline Detection

**IEEE-format paper draft (revised Nov 2026 with corrected, re-run results)**

*Authors: [AUTHOR NAMES] — Department of Computer Science and Engineering, Bangladesh University of Business and Technology (BUBT), Dhaka. Supervisor: [SUPERVISOR].*

---

## Abstract
Transformer models such as BanglaBERT (110.6M parameters) detect Bangla clickbait well but are too large and slow for browser or mobile deployment. We study how much of a fine-tuned BanglaBERT teacher can be retained by a 4-layer, 7.42M-parameter student trained with knowledge distillation (KD) on the BanglaBait corpus (12,044 / 1,506 / 1,506 train/val/test headlines). The teacher reaches 83.13% accuracy (macro-F1 0.812). The best student reaches **80.52 ± 0.37% accuracy (macro-F1 0.787 ± 0.004; mean ± std over three training seeds), i.e. 96.9% of the teacher's accuracy with 6.7% of its parameters**, and runs **11.6× faster** on CPU (13.4 ms vs 154.4 ms per headline, PyTorch FP32, batch size 1). A 2×2 factorial ablation (random vs. pretrained-subset initialisation × with vs. without distillation) shows that **initialising the student from the pretrained `banglabert_small` layers is the dominant factor (+4.5 accuracy points)**, while distillation adds a smaller gain (+0.9 points on pretrained init, +1.2 on random init) that is not statistically significant under exact McNemar tests (p = 0.136 and 0.071). The student remains significantly worse than the teacher (p = 0.018). In zero-shot transfer to 10,000 BaitBuster-Bangla headlines, accuracy drops by 6.1 points for the teacher and 7.1 for the student, so we find **no cross-domain advantage** for the compact model. We also document an implementation defect — distillation was silently disabled in our first experiments — and how it was found and fixed.

**Index Terms** — Bangla NLP, knowledge distillation, clickbait detection, model compression, explainability, ONNX.

---

## I. Introduction
Clickbait headlines use sensational or withholding language to attract clicks. Bangla has over 200 million speakers, and recent datasets (BanglaBait [3], BaitBuster-Bangla [4]) and pretrained models (BanglaBERT [1]) make supervised detection practical. Deployment is the open issue: a 110M-parameter encoder has a ~422 MB footprint and ~154 ms CPU latency per headline.

This paper investigates **how far knowledge distillation can compress a Bangla clickbait detector, and what actually drives the result.** Contributions:
1. A 4-layer, 7.42M-parameter student (TinyBanglaBERT) built from layers [2, 5, 8, 11] of `csebuetnlp/banglabert_small` and trained with temperature-scaled distillation from a fine-tuned BanglaBERT teacher.
2. A **three-seed 2×2 ablation** (initialisation × distillation) with exact McNemar tests, showing pretrained initialisation matters far more than distillation in this setting.
3. Efficiency measurements (PyTorch, ONNX, INT8) and a zero-shot cross-domain test on BaitBuster-Bangla, reported with an honest negative result on robustness.
4. An explainability protocol (SHAP/LIME with agreement analysis) and a **bug disclosure**: an early pipeline silently trained without distillation; we explain the cause (Hugging Face `remove_unused_columns`), the fix, and regression tests.

---

## II. Methodology

```
   Fine-tuned teacher: BanglaBERT (12 layers, 110.6M)
              │  logits z_T  (soft targets, temperature T)
              ▼
   labels y ─► L = α·CE(z_S, y) + (1−α)·T²·KL(σ(z_T/T) ‖ σ(z_S/T)) ◄─ student logits z_S
                                                         ▲
   Student: TinyBanglaBERT (4 layers, 7.42M) — init from banglabert_small layers [2,5,8,11]
```

### A. Data
BanglaBait: 15,056 labelled headlines (34.8% clickbait), split 12,044 / 1,506 / 1,506 (seed 42). The corpus also advertises ~65k unlabeled articles, but the copy used here contains **zero unlabeled rows**, so no unlabeled distillation was performed. NFC normalisation, max length 128.

### B. Teacher
`csebuetnlp/banglabert` (ELECTRA-base discriminator [1], [5]) fine-tuned for up to 5 epochs, lr 2e-5, batch 32, class-weighted cross-entropy, early stopping on validation macro-F1.

### C. Student and initialisation
Student = `banglabert_small` configuration truncated to 4 layers (7.42M parameters). Two initialisations: **pretrained subset** (embeddings + layers [2, 5, 8, 11] copied from `banglabert_small`, in the spirit of pre-trained compact students [7]) and **random** (same architecture, random weights). A previous PCA "teacher-projected" initialisation performed no better than random and was discarded.

### D. Distillation objective
$$\mathcal{L} = \alpha\,\mathcal{L}_{CE}(\sigma(z_S),y) + (1-\alpha)\,T^2\,\mathcal{L}_{KL}\big(\sigma(z_S/T)\,\|\,\sigma(z_T/T)\big)$$
following Hinton et al. [2]. A 5×5 validation grid over T ∈ {1,2,4,8,16}, α ∈ {0.1,0.3,0.5,0.7,0.9} (short runs) selected **T = 1.0, α = 0.1**. Students: lr 1e-4, batch 64, up to 10 epochs, early stopping on validation macro-F1. Seeds 42, 43, 44; data split fixed.

### E. Evaluation
Accuracy, macro-F1, exact McNemar tests [8] on the seed-42 models (N = 1,506), CPU latency (B = 1, L = 128, 200 iterations), zero-shot transfer to 10,000 BaitBuster-Bangla headlines, and SHAP [9] / LIME [10] attribution agreement.

---

## III. Results

### A. In-domain performance (BanglaBait test)
**TABLE I**
| Model | Params | Acc (%) | Macro-F1 | % of teacher acc |
|:---|:---:|:---:|:---:|:---:|
| Teacher (BanglaBERT) | 110.62M | 83.13 | 0.8124 | 100 |
| TinyBanglaBERT (R3, mean of 3 seeds) | 7.42M | 80.52 ± 0.37 | 0.787 ± 0.004 | 96.9 |
| TinyBanglaBERT (R3, seed 42) | 7.42M | 80.88 | 0.7914 | 97.3 |

### B. Factorial ablation (mean ± std, 3 seeds)
**TABLE II**
| Row | Init | Distil. | Acc (%) | Macro-F1 | % of teacher |
|:---|:---|:---:|:---:|:---:|:---:|
| R0 | random | no | 75.19 ± 0.55 | 0.739 ± 0.006 | 90.4 |
| R1 | pretrained | no | 79.66 ± 0.43 | 0.782 ± 0.004 | 95.8 |
| R2 | random | yes | 76.38 ± 0.79 | 0.744 ± 0.004 | 91.9 |
| R3 | pretrained | yes | **80.52 ± 0.37** | **0.787 ± 0.004** | 96.9 |

Initialisation effect ≈ +4.5 pts (R1−R0) / +4.1 pts (R3−R2); distillation effect ≈ +1.2 pts (R2−R0) / +0.9 pts (R3−R1). The distillation gains are about 2 standard deviations of seed noise, hence modest evidence.

### C. Statistical significance (exact McNemar, seed 42)
**TABLE III**
| Comparison | A-only / B-only correct | p-value | Reading |
|:---|:---:|:---:|:---|
| Teacher vs R3 | 114 / 80 | 0.018 | teacher significantly better |
| R3 vs R1 (KD, pretrained) | 74 / 56 | 0.136 | not significant |
| R2 vs R0 (KD, random) | 101 / 76 | 0.071 | not significant |
| R3 vs R2 (pretrained init, KD) | 136 / 82 | < 0.001 | significant |
| R1 vs R0 (pretrained init, no KD) | 157 / 96 | < 0.001 | significant |

### D. Efficiency (Kaggle CPU, B = 1, L = 128, 200 iterations)
**TABLE IV**
| Model | Params | Size | CPU latency | Speed-up |
|:---|:---:|:---:|:---:|:---:|
| Teacher PyTorch FP32 | 110.62M | 422.2 MB | 154.4 ms | 1× |
| Student PyTorch FP32 | 7.42M | 28.3 MB | 13.4 ms | **11.6×** |
| Student ONNX FP32 | 7.42M | 28.4 MB | 7.5 ms | (different runtime) |
| Student ONNX INT8 | 7.42M | 7.3 MB | 8.2 ms | smaller, not faster than ONNX FP32 |

The headline speed-up is the same-runtime PyTorch comparison; ONNX figures compare different runtimes and are given for deployment reference. Latency was measured on Kaggle hardware, not an edge device.

### E. Explainability (SHAP / LIME)
**TABLE V**
Re-run student (R3, seed 42) on 15 test headlines chosen to cover TP/TN/FP/FN (so this sample is *not* representative of accuracy):

| Metric (SHAP vs LIME) | Value |
|---|---|
| Mean top-3 token Jaccard overlap | 42.0% (median 50.0%) |
| Mean top-3 Jaccard, clickbait tokens | 33.4% |
| Sign (polarity) agreement | 54.9% |

**Reading:** agreement is only moderate, and sign agreement is close to the 50% expected by chance. We therefore do **not** claim that the two methods "converge"; explanations of this compact model should be treated as indicative, not definitive. Agreement varies by headline: for *"মূল্যায়ন পদ্ধতির পরিবর্তন জরুরী: শিক্ষামন্ত্রী"* (not clickbait, correctly predicted) both methods push towards not-clickbait with 100% sign agreement and top tokens *মূল্যায়ন, পদ্ধতির, পরিবর্তন*; for the viral-video headline *"বাঁশ দিয়ে ফাঁদ পেতে … ভিডিও তুমুল ভাইরাল !"* (clickbait, predicted 97.8%) both flag *ভাইরাল* as clickbait evidence but their top-3 sets do not overlap (SHAP: *পদ্ধতিতে, তুমুল, ধরার*; LIME: *ভাইরাল, ভিডিও, অভিনব*), Jaccard 0.00. Source: `results/explainability/` of the final-evaluation notebook (generated 2026-11-18). The generic "qualitative insights" paragraph in the auto-generated report is boilerplate, not derived from this data, and is not used.

### F. Zero-shot cross-domain transfer (BaitBuster-Bangla, 10,000 headlines: 5,644 clickbait / 4,356 not)
**TABLE VI**
| Model | Acc in-domain → cross | Macro-F1 cross | Clickbait precision | Clickbait recall | FP / FN |
|:---|:---:|:---:|:---:|:---:|:---:|
| Teacher | 83.13 → 77.03 (−6.1) | 0.769 | 0.970 | 0.612 | 105 / 2,192 |
| Student (R3, seed 42) | 80.88 → 73.75 (−7.1) | 0.735 | 0.937 | 0.573 | 217 / 2,408 |

Both models are **conservative** on the new domain: when they flag clickbait they are almost always right (precision 94–97%), but they miss 39–43% of clickbait, suggesting that BaitBuster annotators label more headlines as clickbait than BanglaBait does. The student degrades slightly more than the teacher; there is no evidence of a robustness advantage.

---

## IV. Discussion and Limitations
- **What worked:** pretrained initialisation; the student keeps ~97% of teacher accuracy at 6.7% of the size and ~11.6× lower CPU latency.
- **What did not:** the target of ~90% accuracy was not met (teacher itself is 83%); distillation's benefit is small and not significant at 3 seeds.
- **Limitations:** single teacher and dataset split; McNemar, efficiency and cross-domain use seed-42 models only; no unlabeled data was available; latency measured on Kaggle CPU; the cross-domain label definitions differ.
- **Disclosure:** the first experiments contained a pipeline defect (`teacher_logits` removed by `remove_unused_columns=True`), making "distilled" runs identical to supervised ones. It is fixed (`remove_unused_columns=False`, a fail-loud guard, `tests/test_kd_fixes.py`), and all numbers above come from re-runs. A separate cross-domain metric-argument-order defect (precision/recall swapped) was also corrected (`tests/test_metric_argument_order.py`).

## V. Conclusion
A 7.42M-parameter student distilled from BanglaBERT retains 96.9% of the teacher's accuracy (80.5% vs 83.1%) and is 11.6× faster on CPU. Pretrained initialisation, not distillation, explains most of the gain; the remaining gap to the teacher is statistically significant, and cross-domain robustness is not improved. Future work: more seeds, layer-wise / intermediate-representation distillation [6], a larger (6-layer) student, and additional teachers.

---

## References
1. A. Bhattacharjee, T. Hasan, W. U. Ahmad, K. Samin, M. S. Islam, A. Iqbal, M. S. Rahman, and R. Shahriyar, "BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla," in *Findings of the Association for Computational Linguistics: NAACL 2022*, pp. 1318–1327.
2. G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015.
3. M. Mahtab, M. Haque, M. Hasan, and F. Sadeque, "BanglaBait: Semi-Supervised Adversarial Approach for Clickbait Detection on Bangla Clickbait Dataset," in *Proc. RANLP 2023*, pp. 748–758.
4. A. Al Imran, M. S. H. Shovon, and M. F. Mridha, "BaitBuster-Bangla: A Comprehensive Dataset for Clickbait Detection in Bangla with Multi-Feature and Multimodal Analysis," arXiv:2310.11465, 2023. *(Verify final publication venue before submission.)*
5. K. Clark, M.-T. Luong, Q. V. Le, and C. D. Manning, "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators," in *ICLR*, 2020.
6. X. Jiao et al., "TinyBERT: Distilling BERT for Natural Language Understanding," in *Findings of EMNLP*, 2020.
7. I. Turc, M.-W. Chang, K. Lee, and K. Toutanova, "Well-Read Students Learn Better: On the Importance of Pre-training Compact Models," arXiv:1908.08962, 2019.
8. Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.
9. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *NeurIPS*, 2017.
10. M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in *ACM SIGKDD*, 2016.
11. V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT," arXiv:1910.01108, 2019.
