"""Does DistillationTrainer actually receive teacher_logits? Tiny random model, CPU."""
import sys, tempfile, torch
sys.path.insert(0, sys.argv[1])
from transformers import ElectraConfig, ElectraForSequenceClassification, TrainingArguments
from src.training import train_distill as td
from src.models.distillation import KnowledgeDistillationLoss, DistillationDataset

class FakeTok:  # returns fixed-size encodings
    def __call__(self, texts, **kw):
        n = len(texts); g = torch.Generator().manual_seed(0)
        return {"input_ids": torch.randint(1, 100, (n, 16), generator=g),
                "attention_mask": torch.ones(n, 16, dtype=torch.long)}

seen = {"with_teacher": 0, "without": 0}
class Probe(td.DistillationTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.training:
            seen["with_teacher" if "teacher_logits" in inputs else "without"] += 1
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

def run(**extra):
    seen.update(with_teacher=0, without=0)
    cfg = ElectraConfig(vocab_size=100, embedding_size=16, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, num_labels=2)
    model = ElectraForSequenceClassification(cfg)
    ds = DistillationTrainer_ds = DistillationDataset(["x"]*64, [0,1]*32, torch.randn(64, 2), FakeTok(), 16)
    args = TrainingArguments(output_dir=tempfile.mkdtemp(), num_train_epochs=1, per_device_train_batch_size=16,
                             report_to="none", use_cpu=True, save_strategy="no", **extra)
    Probe(kd_loss_fn=KnowledgeDistillationLoss(2.0, 0.1), model=model, args=args, train_dataset=ds).train()
    return dict(seen)

print("repo default TrainingArguments :", run())
print("remove_unused_columns=False    :", run(remove_unused_columns=False))
