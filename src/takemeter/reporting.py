from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from takemeter.data import LABELS
from takemeter.models import ModelResult
from takemeter.splits import DatasetSplits


DISPLAY_NAMES = {
    "majority": "Majority baseline",
    "tfidf_logistic_regression": "TF–IDF + logistic regression",
    "distilbert": "DistilBERT",
    "distilbert_weighted": "DistilBERT + class weights",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_results(
    output_dir: Path,
    dataset_path: Path,
    splits: DatasetSplits,
    results: list[ModelResult],
    details: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(
        [
            {
                "model": result.name,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "weighted_f1": result.weighted_f1,
            }
            for result in results
        ]
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False, float_format="%.6f")

    predictions = splits.test[
        ["id", "source_site", "source_url", "source_context", "label"]
    ].copy()
    for result in results:
        predictions[result.name] = result.predictions
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    per_class = []
    for result in results:
        report = classification_report(
            splits.test["label"],
            result.predictions,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        )
        for label in LABELS:
            per_class.append({"model": result.name, "label": label, **report[label]})
    pd.DataFrame(per_class).to_csv(
        output_dir / "per_class_metrics.csv", index=False, float_format="%.6f"
    )

    run = {
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "rows": sum(len(frame) for frame in splits.as_dict().values()),
        },
        "split": {
            name: {
                "rows": len(frame),
                "source_threads": frame["source_url"].nunique(),
                "label_counts": frame["label"].value_counts().sort_index().to_dict(),
            }
            for name, frame in splits.as_dict().items()
        },
        "models": {result.name: result.parameters for result in results},
        "details": details,
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "matplotlib",
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "transformers",
            )
            if _package_is_installed(package)
        },
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    comparison = metrics.sort_values("macro_f1", ascending=True).copy()
    comparison["display_name"] = comparison["model"].map(DISPLAY_NAMES)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh(comparison["display_name"], comparison["macro_f1"], color="#4472C4")
    axis.set(xlabel="Macro F1", xlim=(0, 1), title="Unseen-thread test performance")
    for index, value in enumerate(comparison["macro_f1"]):
        axis.text(value + 0.015, index, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    for result in results:
        matrix = confusion_matrix(
            splits.test["label"], result.predictions, labels=list(LABELS)
        )
        fig, axis = plt.subplots(figsize=(7, 5.5))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        axis.set(
            xlabel="Predicted",
            ylabel="Actual",
            title=DISPLAY_NAMES[result.name],
        )
        for row in range(len(LABELS)):
            for column in range(len(LABELS)):
                axis.text(column, row, matrix[row, column], ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_matrix_{result.name}.png", dpi=160)
        plt.close(fig)


def _package_is_installed(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True
