# TinyBanglaBERT 🔬

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime%20Quantized-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**A Lightweight Knowledge-Distilled Transformer for Real-Time, Explainable Bangla Clickbait Detection on Edge Devices**

[**Research Paper (IEEE Format)**](experiments/IEEE_PAPER_REPORT.md) • [**Full Results Chapter**](experiments/FINAL_RESULTS.md) • [**Reproduction Manual**](REPRODUCE.md) • [**LaTeX Manuscript**](experiments/ieee_manuscript.tex)

</div>

---

## 📌 Executive Summary (updated Nov 2026)

> ⚠️ An earlier version of this README reported different numbers. A bug meant knowledge distillation never actually ran
> (the Hugging Face `Trainer` dropped `teacher_logits`). It is fixed, everything was re-run with 3 seeds, and the
> corrected results are below and in [`RESULTS_NOV2026.md`](RESULTS_NOV2026.md). `experiments/` still holds old, outdated outputs.

**TinyBanglaBERT** is a 4-layer ELECTRA student (7.42M parameters) initialised from layers of the pretrained
`csebuetnlp/banglabert_small` and distilled from a fine-tuned `csebuetnlp/banglabert` teacher (110.6M) for Bangla clickbait detection.
Data: BanglaBait (15,056 labelled headlines; 12,044 / 1,506 / 1,506 split, seed 42; no unlabeled data).

| Model | Test accuracy (mean ± std, 3 seeds) | % of teacher |
|---|---|---|
| Teacher (BanglaBERT) | 83.13 | 100 |
| Random init, no distillation | 75.19 ± 0.55 | 90.4 |
| Random init + distillation | 76.38 ± 0.79 | 91.9 |
| Pretrained init, no distillation | 79.66 ± 0.43 | 95.8 |
| **Pretrained init + distillation (final)** | **80.52 ± 0.37** | **96.9** |

**Findings**
1. The pretrained compact initialisation is the main gain (+~4 points, McNemar p < 0.001).
2. Distillation adds a small, consistent gain (+~1 point) that is **not** statistically significant (p = 0.07 and 0.14).
3. The teacher is still significantly better than the final student (McNemar p = 0.018).
4. 14.9× fewer parameters (7.42M vs 110.6M), 28.3 MB vs 422 MB (7.3 MB INT8), ~11.6× faster on CPU in PyTorch
   (13.4 ms vs 154.4 ms per headline; measured on Kaggle hardware).
5. Zero-shot on BaitBuster-Bangla both models drop (teacher 83.1→77.0, student 80.9→73.8); the student does **not** generalise better.

Notebooks that produce these results: [`notebooks/`](notebooks/).

---

## 📈 Visualizations & Figures

> ⚠️ The figures below come from the old (pre-fix) run and are outdated; they will be regenerated.

Publication-ready 300 DPI figures generated from empirical evaluation:

<div align="center">

| Figure 1: 4-Way Ablation Analysis | Figure 2: Efficiency-Accuracy Tradeoff |
|:---:|:---:|
| ![Ablation](experiments/figures/fig1_ablation_f1_accuracy.png) | ![Efficiency](experiments/figures/fig2_efficiency_tradeoff.png) |

| Figure 3: Latency & Speedup Breakdown | Figure 4: Cross-Domain Generalization |
|:---:|:---:|
| ![Latency](experiments/figures/fig3_latency_speedup_breakdown.png) | ![Generalization](experiments/figures/fig4_crossdomain_generalization.png) |

| Figure 5: SHAP vs. LIME Agreement | Figure 6: Test Confusion Matrices |
|:---:|:---:|
| ![Explainability](experiments/figures/fig5_explainability_agreement.png) | ![Confusion Matrices](experiments/figures/fig6_confusion_matrices.png) |

</div>

---

## 🔍 Explainability (SHAP & LIME)

TinyBanglaBERT includes first-class support for token attribution explainability via both **SHAP (KernelExplainer)** and **LIME (LimeTextExplainer)**.

```bash
# Run side-by-side SHAP and LIME token attribution on a headline
python -m scripts.explain --headline "ফেসবুকে এই ছবিটি দেখে চমকে যাবেন আপনিও!"
```

**Output:**
```
================================================================================
Headline: ফেসবুকে এই ছবিটি দেখে চমকে যাবেন আপনিও!
Predicted Class: CLICKBAIT (Confidence: 94.8%)
================================================================================
Top Attributing Tokens (SHAP):
  [+] চমকে      (+0.342)  --> Pushes toward Clickbait
  [+] যাবেন     (+0.218)  --> Pushes toward Clickbait
  [+] আপনিও    (+0.185)  --> Pushes toward Clickbait

Top Attributing Tokens (LIME):
  [+] চমকে      (+0.311)  --> Pushes toward Clickbait
  [+] যাবেন     (+0.194)  --> Pushes toward Clickbait
  [+] ছবিটি      (+0.102)  --> Pushes toward Clickbait

Cross-Method Token Agreement: Jaccard=0.67, Sign Agreement=100.0%
================================================================================
```

---

## 📂 Repository Structure

```
tinybanglabert/
├── configs/                                 # Strict YAML experiment configurations
│   ├── base.yaml                            # Shared architecture and dataset defaults
│   ├── teacher_baseline.yaml                # Teacher fine-tuning config
│   ├── student_teacher_init_distilled.yaml  # Winning distillation config
│   └── student_random_baseline.yaml         # Ablation baseline configs
├── data/
│   ├── raw/                                 # Untracked raw data (auto-downloaded)
│   └── processed/
│       ├── splits.json                      # Deterministic 80/10/10 split indices (tracked)
│       └── data_report.md                   # Dataset hygiene and NFC normalization audit
├── experiments/                             # Empirical results, reports, and publication artifacts
│   ├── figures/                             # 300 DPI high-resolution figures (fig1 to fig6)
│   ├── FINAL_RESULTS.md                     # Thesis Results Chapter package
│   ├── IEEE_PAPER_REPORT.md                 # IEEE formatted conference manuscript draft
│   ├── ieee_manuscript.tex                  # Ready-to-compile IEEEtran LaTeX source
│   ├── ablation_table.csv                   # 4-way ablation results
│   ├── efficiency_table.csv                 # PyTorch vs. ONNX INT8 benchmarks
│   ├── significance_table.csv               # McNemar exact binomial test matrix
│   ├── crossdomain_table.csv                # BaitBuster-Bangla 10k zero-shot benchmark
│   └── explainability/examples.md           # Qualitative SHAP/LIME explanation case studies
├── scripts/                                 # Executable CLI entry points
│   ├── prepare_data.py                      # Data acquisition, cleaning, and stratification
│   ├── train.py                             # Unified training & distillation pipeline
│   ├── evaluate.py                          # Multi-metric evaluation runner
│   ├── explain.py                           # SHAP & LIME CLI explainer
│   ├── benchmark.py                         # Efficiency, latency, and memory profiler
│   ├── tune_kd.py                           # Optuna hyperparameter tuning (T, alpha)
│   └── generate_all_figures.py              # Visual plot generator
├── src/                                     # Modular Python source library
│   ├── data/                                # Cleaners, loaders, splitters, crossdomain pipelines
│   ├── models/                              # Teacher, Student (TinyBanglaBERT), KD Loss
│   ├── training/                            # Baseline, Distillation, and Grid Search Trainers
│   ├── evaluation/                          # Metrics, McNemar exact tests, Ablation, Cross-Domain
│   ├── explainability/                      # SHAP runner, LIME runner, Jaccard agreement
│   ├── efficiency/                          # ONNX exporter, INT8 quantizer, Latency profiler
│   └── utils/                               # Seeding, Config loader, Experiment logger
├── tests/                                   # Complete Unit Test Suite (88 passing tests)
├── CITATION.cff                             # GitHub native citation metadata
├── CONTRIBUTING.md                          # Open-source contribution guidelines
├── LICENSE                                  # MIT License
├── pyproject.toml                           # PEP 517/518 build specification
├── README.md                                # Master project documentation
├── REPRODUCE.md                             # Step-by-step reproduction guide
└── requirements.txt                         # Pinned pip requirements
```

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-username/tinybanglabert.git
cd tinybanglabert

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux / macOS

# Install package in development mode
pip install -e ".[dev]"
```

### 2. Verify Installation (Unit Tests)

```bash
pytest tests/ -v
# Note: some tests need a trained teacher checkpoint; tests/test_kd_fixes.py runs on CPU without downloads
```

### 3. End-to-End Execution Pipeline

```bash
# Step 1: Prepare and clean dataset (NFC normalization + 80/10/10 split)
python -m scripts.prepare_data

# Step 2: Fine-tune Teacher model
python -m scripts.train --config configs/teacher_baseline.yaml

# Step 3: Distil TinyBanglaBERT Student
python -m scripts.tune_kd --config configs/student_kd_improved.yaml --init small_subset
python -m scripts.train --config configs/student_kd_improved.yaml --seed 42 --temperature 1.0 --alpha 0.1

# Step 4: Export to ONNX and run dynamic INT8 Quantization
python -m src.efficiency.export_onnx \
    --model_dir experiments/tinybanglabert_final \
    --output_dir experiments/tinybanglabert_final/onnx \
    --quantize

# Step 5: Benchmark CPU latency and memory
python -m scripts.benchmark --config configs/student_teacher_init_distilled.yaml

# Step 6: Run statistical McNemar significance testing
python -m src.evaluation.mcnemar_test

# Step 7: Evaluate zero-shot cross-domain generalization
python -m src.evaluation.crossdomain_eval

# Step 8: Generate all publication figures
python -m scripts.generate_all_figures
```

---

## 📜 Citation

If you use TinyBanglaBERT, its architecture, or distilled checkpoints in your research, please cite our work:

```bibtex
@article{rahman2026tinybanglabert,
  title   = {TinyBanglaBERT: A Lightweight Knowledge-Distilled Transformer for Real-Time Explainable Bangla Clickbait Detection on Edge Devices},
  author  = {Rahman, Mushfiqur},
  year    = {2026},
  journal = {arXiv preprint},
  url     = {https://github.com/mushfiqur-rahman/tinybanglabert}
}
```

---

## 📄 License

This repository is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
