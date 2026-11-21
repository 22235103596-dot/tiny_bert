# TinyBanglaBERT — Reproduction Manual (corrected, Nov 2026)

How to reproduce the corrected results in `FINAL_RESULTS.md`. The easiest route is the three Kaggle notebooks in `notebooks/`; the CLI route is given below. Expected numbers vary by ±0.5 accuracy points across runs/hardware.

## 1. Requirements
- Python 3.10+; a GPU is strongly recommended for training (Kaggle T4/P100 is enough; whole study ≈ 1–2 GPU hours). CPU is enough for the latency benchmark.
- Packages: `pip install -r requirements.txt` (torch, transformers, datasets, accelerate, scikit-learn, pandas, scipy, shap, lime, onnx, onnxruntime, matplotlib, pytest, pyyaml).

```bash
git clone <your-repo-url> && cd tiny_bert
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Option A — Kaggle notebooks (recommended)
| Notebook | Does |
|---|---|
| `notebooks/01_kaggle_teacher_and_first_rerun.ipynb` | data split, teacher fine-tuning, (T, α) grid, first student re-run |
| `notebooks/02_kaggle_ablation_multiseed.ipynb` | R0–R3 ablation, seeds 42/43/44, summary table |
| `notebooks/03_kaggle_final_evaluation.ipynb` | McNemar, efficiency (PyTorch/ONNX/INT8), cross-domain, SHAP/LIME |

Enable GPU + Internet, run top to bottom, and copy the outputs into `results/`.

## 3. Option B — command line
```bash
# 1. Data (BanglaBait, seed-42 split: 12,044 / 1,506 / 1,506)
python -m scripts.prepare_data --seed 42

# 2. Teacher (expected ≈ 83% acc, macro-F1 ≈ 0.81; ~5 epochs)
python -m scripts.train --config configs/teacher_baseline.yaml

# 3. (T, α) grid — KD is now really active, so the grid is no longer flat
python -m scripts.tune_kd --init small_subset --epochs 3

# 4. Ablation, 3 seeds each (use T=1.0, alpha=0.1)
for s in 42 43 44; do
  python -m scripts.train --config configs/student_kd_improved.yaml --init small_random  --alpha 1.0 --seed $s   # R0 random, no KD
  python -m scripts.train --config configs/student_kd_improved.yaml --init small_subset  --alpha 1.0 --seed $s   # R1 pretrained, no KD
  python -m scripts.train --config configs/student_kd_improved.yaml --init small_random  --temperature 1.0 --alpha 0.1 --seed $s  # R2
  python -m scripts.train --config configs/student_kd_improved.yaml --init small_subset  --temperature 1.0 --alpha 0.1 --seed $s  # R3 (final)
done
# --alpha 1.0 means pure cross-entropy (no distillation).

# 5. Efficiency, significance, cross-domain, explainability
python -m scripts.benchmark --n-iterations 200
python -c "from src.evaluation.mcnemar_test import run_full_mcnemar_suite; run_full_mcnemar_suite()"
python -c "from src.evaluation.crossdomain_eval import run_crossdomain_evaluation; run_crossdomain_evaluation('abdalimran/BaitBuster-Bangla')"
# (notebook 03 runs these with the exact final-model paths; use it if the defaults do not match your folders)
python -m scripts.explain --generate-thesis-samples --sample-size 15

# 6. Figures
python -m scripts.make_figures
```
Note: the `student_kd_improved.yaml` defaults (T=2, α=0.3, unlabeled pool) are *not* the values used in the paper; always pass `--temperature 1.0 --alpha 0.1` for R2/R3. `use_unlabeled` has no effect because the dataset contains no unlabeled rows.

## 4. Expected results (seed 42 / 3-seed mean)
| Item | Expected |
|---|---|
| Teacher | 83.13% acc, macro-F1 0.812 |
| R0 / R1 / R2 / R3 (acc %) | 75.2 / 79.7 / 76.4 / 80.5 (±0.4–0.8) |
| Teacher vs R3 McNemar | p ≈ 0.018 |
| CPU latency | teacher ≈ 154 ms, student ≈ 13 ms (Kaggle CPU) |
| Cross-domain acc | teacher ≈ 77.0%, student ≈ 73.8% |

## 5. Verify the code
```bash
pytest tests/ -v
```
`tests/test_kd_fixes.py` guards the distillation bug (teacher logits must reach the loss) and `tests/test_metric_argument_order.py` guards the cross-domain precision/recall orientation. `tests/test_ablation.py` is skipped (legacy).

## 6. Troubleshooting
- *"teacher_logits missing" RuntimeError:* intended guard — make sure `remove_unused_columns=False` and `experiments/teacher/train_soft_labels.pt` exist.
- *Flat (T, α) grid:* means KD is not running; re-check the guard/test above.
- Weights and ONNX files are git-ignored; regenerate them with the steps above.
- Old outputs from the buggy run are kept in `archive_old_buggy_run/` for reference only.
