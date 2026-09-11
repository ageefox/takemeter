from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from takemeter.data import LABELS
from takemeter.models import ModelResult, score_predictions


@dataclass
class _EncodedDataset:
    encodings: dict[str, object]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def evaluate_distilbert(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    epochs: int = 3,
    seed: int = 42,
) -> tuple[ModelResult, list[dict[str, float]]]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            get_linear_schedule_with_warmup,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install the transformer dependencies with pip install -e '.[transformer]'"
        ) from error

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_name = "distilbert-base-uncased"
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(frame: pd.DataFrame) -> _EncodedDataset:
        encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=256,
        )
        return _EncodedDataset(
            encodings=encodings,
            labels=[label_to_id[label] for label in frame["label"]],
        )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(
        encode(train), batch_size=16, shuffle=True, collate_fn=collator
    )
    validation_loader = DataLoader(
        encode(validation), batch_size=32, shuffle=False, collate_fn=collator
    )
    test_loader = DataLoader(
        encode(test), batch_size=32, shuffle=False, collate_fn=collator
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(10, total_steps // 5),
        num_training_steps=total_steps,
    )

    def predict(loader: DataLoader) -> np.ndarray:
        model.eval()
        output: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch.pop("labels")
                logits = model(**batch).logits
                output.extend(logits.argmax(dim=-1).cpu().tolist())
                del labels
        return np.array([id_to_label[index] for index in output])

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            result = model(**batch)
            result.loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(result.loss.detach().cpu())

        validation_predictions = predict(validation_loader)
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": running_loss / len(train_loader),
                "validation_macro_f1": float(
                    f1_score(
                        validation["label"],
                        validation_predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    test_predictions = predict(test_loader)
    return (
        score_predictions(
            "distilbert",
            test["label"],
            test_predictions,
            {
                "model": model_name,
                "epochs": epochs,
                "learning_rate": 2e-5,
                "batch_size": 16,
                "max_length": 256,
                "seed": seed,
                "device": str(device),
            },
        ),
        history,
    )

