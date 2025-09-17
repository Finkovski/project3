from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from src.utils.metrics import compute_metrics, confusion
from src.data.preprocess import prepare_dataframe


@dataclass
class TrainResult:
    metrics: Dict[str, float]
    cm: np.ndarray


# ------------------------ helpers ------------------------ #
def _coerce_hparams(
    max_length: Any,
    train_size: Any,
    epochs: Any,
    batch_size: Any,
    lr: Any,
) -> Tuple[int, float, int, int, float]:
    """Coerce YAML/CLI values into correct dtypes (e.g., '5e-5' -> 5e-5)."""
    return int(max_length), float(train_size), int(epochs), int(batch_size), float(lr)


def _make_splits(texts, labels, train_size: float, seed: int = 42):
    """Stratified split when possible; fallback for tiny/imbalanced data."""
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, train_size=train_size, stratify=labels, random_state=seed
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, train_size=train_size, shuffle=True, random_state=seed
        )
    return X_tr, X_te, y_tr, y_te


def _layer_indices_from_names(param_names):
    """Detect encoder layer indices from names like '...encoder.layer.11...'."""
    idxs = []
    key = "encoder.layer."
    for n in param_names:
        if key in n:
            try:
                after = n.split(key, 1)[1]
                i = int(after.split(".", 1)[0])
                idxs.append(i)
            except Exception:
                pass
    return sorted(set(idxs))


# ------------------------ custom trainer ------------------------ #
class WeightedTrainer(Trainer):
    """Trainer that supports class-weighted CE loss (and accepts extra kwargs)."""
    def __init__(self, *args, class_weights_tensor: Optional[torch.Tensor] = None, **kwargs):
        self.class_weights_tensor = class_weights_tensor
        super().__init__(*args, **kwargs)

    # Newer HF passes extra kwargs (e.g., num_items_in_batch)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights_tensor is not None:
            weight = self.class_weights_tensor.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ------------------------ main API ------------------------ #
def train_and_eval(
    df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    train_size: float = 0.8,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 5e-5,
    freeze_base: bool = False,
    unfreeze_last_n: int = 0,
    use_class_weights: bool = False,
    label_smoothing: float = 0.0,
    save_dir: str = None,  # NEW: save fine-tuned model/tokenizer here if provided
) -> TrainResult:
    """
    Fine-tune a 3-class sentiment classifier and return metrics + confusion matrix.
    Labels: 0=negative, 1=neutral, 2=positive (set in preprocessing).
    """
    # Coerce potential stringy config values
    max_length, train_size, epochs, batch_size, lr = _coerce_hparams(
        max_length, train_size, epochs, batch_size, lr
    )

    # Prepare data
    df = prepare_dataframe(df)
    texts = df["review_text"].tolist()
    labels = df["sentiment"].astype(int).tolist()

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Freezing / unfreezing strategy
    param_names = [n for n, _ in model.named_parameters()]
    layer_idxs = _layer_indices_from_names(param_names)

    if freeze_base or unfreeze_last_n > 0:
        # freeze all base layers first
        if hasattr(model, "base_model"):
            for p in model.base_model.parameters():
                p.requires_grad = False
        else:
            for name, p in model.named_parameters():
                if "classifier" not in name:
                    p.requires_grad = False

    if unfreeze_last_n > 0 and layer_idxs:
        # unfreeze just the last N encoder layers
        keep = set(layer_idxs[-unfreeze_last_n:])
        for name, p in model.named_parameters():
            for i in keep:
                if f"encoder.layer.{i}." in name:
                    p.requires_grad = True
                    break
        # classifier head always trainable
        for name, p in model.named_parameters():
            if "classifier" in name:
                p.requires_grad = True

    # Report trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({(trainable/total):.1%})")

    # Datasets
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, xs, ys):
            enc = tokenizer(xs, truncation=True, padding=True, max_length=max_length)
            self.enc = {k: np.array(v) for k, v in enc.items()}
            self.labels = np.array(ys, dtype=np.int64)
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    X_train, X_test, y_train, y_test = _make_splits(texts, labels, train_size)
    train_ds = SimpleDataset(X_train, y_train)
    test_ds  = SimpleDataset(X_test,  y_test)

    # Class weights from TRAIN split (only if requested)
    class_weights_tensor = None
    if use_class_weights:
        counts = np.bincount(np.array(y_train), minlength=3).astype(float)
        inv = counts.sum() / (counts + 1e-9)      # inverse frequency
        inv = inv / inv.sum() * 3.0               # normalize to sum to #classes
        class_weights_tensor = torch.tensor(inv, dtype=torch.float)
        print("Class weights:", inv.round(3).tolist())

    # Training args (version-safe)
    try:
        args = TrainingArguments(
            output_dir="artifacts/classifier_runs",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            report_to=[],
            label_smoothing_factor=float(label_smoothing) if label_smoothing else 0.0,
        )
    except TypeError:
        args = TrainingArguments(
            output_dir="artifacts/classifier_runs",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            logging_steps=50,
        )

    # HF metrics adapter (handles different eval_pred shapes)
    def _hf_metrics(eval_pred) -> Dict[str, float]:
        if isinstance(eval_pred, tuple) and len(eval_pred) == 2:
            logits, labels_np = eval_pred
        else:
            logits, labels_np = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        return compute_metrics(labels_np, preds)

    # Choose trainer
    TrainerCls = WeightedTrainer if class_weights_tensor is not None else Trainer
    trainer = TrainerCls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=_hf_metrics,
        **({"class_weights_tensor": class_weights_tensor} if class_weights_tensor is not None else {}),
    )

    # Train
    trainer.train()

    # Final eval + confusion matrix
    pred_out = trainer.predict(test_ds)
    if isinstance(pred_out, tuple) and len(pred_out) >= 2:
        logits, label_ids = pred_out[0], pred_out[1]
    else:
        logits, label_ids = pred_out.predictions, pred_out.label_ids
    y_pred = np.argmax(logits, axis=-1)

    metrics = compute_metrics(label_ids, y_pred)
    cm = confusion(label_ids, y_pred)

    # Save fine-tuned model/tokenizer if requested
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.save_pretrained(save_dir)
        print(f"Saved fine-tuned model to: {save_dir}")

    return TrainResult(metrics=metrics, cm=cm)
