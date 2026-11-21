# TinyBanglaBERT – corrected results (Nov 2026)

## Why the old results were replaced
`DistillationTrainer` expects a `teacher_logits` column in every batch, but Hugging Face `Trainer` defaults to
`remove_unused_columns=True`, which silently drops columns not in the model's `forward()` signature. Training therefore fell back
to plain class-weighted cross-entropy: "distilled" and "supervised" runs were identical and the (T, alpha) grid was flat (0.6993).
Fix: `remove_unused_columns=False` plus a guard that raises if teacher logits are missing (`tests/test_kd_fixes.py`).
Other changes: student initialised from `csebuetnlp/banglabert_small` layers (`small_subset`); the old PCA "teacher-projected" init
was no better than random and was dropped. The unlabeled pool is empty in this dataset (0 rows) and was not used.

## Setup
BanglaBait, 12,044 / 1,506 / 1,506 (seed 42). Teacher `csebuetnlp/banglabert` fine-tuned, 83.13% acc, macro-F1 0.8124.
Students: 4 layers, 7.42M params. Distillation T=1.0, alpha=0.1 (best of a 5x5 validation grid, short runs). 3 training seeds (42,43,44), fixed split.

## Ablation (test accuracy %, mean ± std over 3 seeds)
| Row | Init | Distillation | Acc | Macro-F1 | % of teacher acc |
|---|---|---|---|---|---|
| R0 | random | no | 75.19 ± 0.55 | 0.739 ± 0.006 | 90.4 |
| R1 | pretrained | no | 79.66 ± 0.43 | 0.782 ± 0.004 | 95.8 |
| R2 | random | yes | 76.38 ± 0.79 | 0.744 ± 0.004 | 91.9 |
| R3 | pretrained | yes (final) | 80.52 ± 0.37 | 0.787 ± 0.004 | 96.9 |

## McNemar exact tests (seed-42 models, N=1,506)
| A vs B | A-only / B-only correct | p |
|---|---|---|
| Teacher vs R3 | 114 / 80 | 0.018 (teacher better) |
| R3 vs R1 (distillation, pretrained) | 74 / 56 | 0.136 |
| R2 vs R0 (distillation, random) | 101 / 76 | 0.071 |
| R3 vs R2 (pretrained init, with KD) | 136 / 82 | < 0.001 |
| R1 vs R0 (pretrained init, no KD) | 157 / 96 | < 0.001 |

## Efficiency (Kaggle hardware; B=1, L=128, 200 iterations)
| Model | Params | Size | CPU latency |
|---|---|---|---|
| Teacher PyTorch FP32 | 110.62M | 422.2 MB | 154.4 ms |
| Student PyTorch FP32 | 7.42M | 28.3 MB | 13.4 ms (11.6x faster) |
| Student ONNX FP32 | 7.42M | 28.4 MB | 7.5 ms |
| Student ONNX INT8 | 7.42M | 7.3 MB | 8.2 ms (smaller, not faster than ONNX FP32) |
Use the PyTorch-vs-PyTorch ratio as the headline speed-up; the ONNX ratios compare different runtimes. The "peak memory" column is unreliable.

## Cross-domain (BaitBuster-Bangla, 10,000 headlines, zero-shot)
Teacher 83.13 -> 77.03 acc (-6.1); student 80.88 -> 73.75 (-7.1). The student does not generalise better. Both are conservative there: clickbait precision 97.0% (teacher) / 93.7% (student) but recall only 61.2% / 57.3% (FN 2,192 / 2,408; FP 105 / 217). An earlier version of this file said they over-predict clickbait; that came from a precision/recall/confusion-matrix argument-order bug, now fixed.

## Caveats
Single teacher; seed-42 student used for McNemar/efficiency/cross-domain; distillation effect not significant at 3 seeds;
latency measured on Kaggle, not the original laptop. Figures in `experiments/figures/` are from the old run and need regenerating.
