"""Regression tests for the silent-no-distillation bug and the accuracy improvements.

Run with:  pytest tests/test_kd_fixes.py -v   (CPU only, no downloads)
"""
from __future__ import annotations

import tempfile

import pytest
import torch
import torch.nn.functional as F
from transformers import ElectraConfig, ElectraForSequenceClassification, TrainingArguments

from src.models.distillation import (
    IGNORE_INDEX,
    DistillationDataset,
    KnowledgeDistillationLoss,
    compute_teacher_logits,
    select_unlabeled_pool,
)
from src.models.student import config_from_pretrained_small, init_from_pretrained_small
from src.training.train_distill import DistillationTrainer


class _FakeTok:
    def __call__(self, texts, **kw):
        n = len(texts)
        g = torch.Generator().manual_seed(0)
        return type("E", (dict,), {"to": lambda self, d: self})(
            input_ids=torch.randint(1, 100, (n, 16), generator=g),
            attention_mask=torch.ones(n, 16, dtype=torch.long),
        )


def _tiny_cfg(layers=2, emb=8):
    return ElectraConfig(vocab_size=100, embedding_size=emb, hidden_size=16, num_hidden_layers=layers,
                         num_attention_heads=2, intermediate_size=32, num_labels=2)


def _run(remove_unused_columns: bool):
    seen = {"n_with": 0}

    class Probe(DistillationTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if model.training and "teacher_logits" in inputs:
                seen["n_with"] += 1
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    ds = DistillationDataset(["x"] * 32, [0, 1] * 16, torch.randn(32, 2), _FakeTok(), 16)
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), num_train_epochs=1, per_device_train_batch_size=16,
                             report_to="none", use_cpu=True, save_strategy="no",
                             remove_unused_columns=remove_unused_columns)
    Probe(kd_loss_fn=KnowledgeDistillationLoss(2.0, 0.1), model=ElectraForSequenceClassification(_tiny_cfg(1)),
          args=args, train_dataset=ds).train()
    return seen["n_with"]


def test_trainer_receives_teacher_logits_with_fix():
    assert _run(remove_unused_columns=False) == 2  # 32 samples / bs16


def test_trainer_fails_loudly_instead_of_silently_skipping_kd():
    # The original bug: default remove_unused_columns=True dropped teacher_logits.
    with pytest.raises(RuntimeError, match="teacher_logits"):
        _run(remove_unused_columns=True)


def test_loss_ignores_unlabeled_rows_in_hard_term():
    fn = KnowledgeDistillationLoss(temperature=2.0, alpha=0.5)
    s = torch.randn(4, 2, requires_grad=True)
    t = torch.randn(4, 2)
    labels = torch.tensor([0, 1, IGNORE_INDEX, IGNORE_INDEX])
    loss, b = fn(s, t, labels)
    expected_hard = F.cross_entropy(s[:2], labels[:2])
    assert torch.isclose(torch.tensor(b["hard_loss"]), expected_hard, atol=1e-5)
    loss.backward()
    assert torch.isfinite(s.grad).all()


def test_loss_all_unlabeled_batch_is_pure_soft_and_finite():
    fn = KnowledgeDistillationLoss(temperature=2.0, alpha=0.5)
    s = torch.randn(3, 2, requires_grad=True)
    t = torch.randn(3, 2)
    loss, b = fn(s, t, torch.full((3,), IGNORE_INDEX))
    assert torch.isfinite(loss) and b["hard_loss"] == 0.0
    ref, _ = KnowledgeDistillationLoss(2.0, 0.0)(s, t, None)
    assert torch.isclose(loss, ref, atol=1e-5)


def test_unlabeled_pool_never_contains_eval_headlines():
    pool = select_unlabeled_pool(["a", "b", "b", "c", "d"], exclude_headlines={"a", "d"})
    assert pool == ["b", "c"]
    assert len(select_unlabeled_pool([str(i) for i in range(100)], set(), max_n=10)) == 10


def test_compute_teacher_logits_shape():
    out = compute_teacher_logits(ElectraForSequenceClassification(_tiny_cfg()), _FakeTok(),
                                 ["x"] * 10, batch_size=4, device="cpu", fp16=False)
    assert out.shape == (10, 2)


def test_init_from_pretrained_small_copies_selected_layers():
    small = ElectraForSequenceClassification(_tiny_cfg(layers=6, emb=8))
    cfg = config_from_pretrained_small(small.config, num_hidden_layers=3)
    stu = init_from_pretrained_small(small, cfg)  # default map for 6->3 = [1, 3, 5]
    for i, j in enumerate([1, 3, 5]):
        a = stu.electra.encoder.layer[i].attention.self.query.weight
        b = small.electra.encoder.layer[j].attention.self.query.weight
        assert torch.equal(a, b)
    assert torch.equal(stu.electra.embeddings.word_embeddings.weight,
                       small.electra.embeddings.word_embeddings.weight)
    assert torch.equal(stu.electra.embeddings_project.weight, small.electra.embeddings_project.weight)


def test_init_from_pretrained_small_rejects_width_mismatch():
    small = ElectraForSequenceClassification(_tiny_cfg(layers=4))
    bad = config_from_pretrained_small(small.config, 2)
    bad.hidden_size = 32
    with pytest.raises(ValueError):
        init_from_pretrained_small(small, bad)
