from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


TFIDF_C_VALUES = (0.25, 1.0, 4.0)


@dataclass(frozen=True)
class ModelResult:
    name: str
    predictions: np.ndarray
    accuracy: float
    macro_f1: float
    weighted_f1: float
    parameters: dict[str, object]


def score_predictions(
    name: str,
    truth: pd.Series,
    predictions: np.ndarray,
    parameters: dict[str, object],
) -> ModelResult:
    return ModelResult(
        name=name,
        predictions=predictions,
        accuracy=float(accuracy_score(truth, predictions)),
        macro_f1=float(f1_score(truth, predictions, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(truth, predictions, average="weighted", zero_division=0)
        ),
        parameters=parameters,
    )


def evaluate_majority(train: pd.DataFrame, test: pd.DataFrame) -> ModelResult:
    majority_label = str(train["label"].mode().iloc[0])
    predictions = np.repeat(majority_label, len(test))
    return score_predictions(
        "majority",
        test["label"],
        predictions,
        {"majority_label": majority_label},
    )


def _tfidf_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_tfidf(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[ModelResult, list[dict[str, float]]]:
    candidates: list[tuple[float, float, Pipeline]] = []
    validation_results = []

    for c_value in TFIDF_C_VALUES:
        model = _tfidf_pipeline(c_value)
        model.fit(train["text"], train["label"])
        predictions = model.predict(validation["text"])
        macro_f1 = float(
            f1_score(validation["label"], predictions, average="macro", zero_division=0)
        )
        accuracy = float(accuracy_score(validation["label"], predictions))
        validation_results.append(
            {"C": c_value, "accuracy": accuracy, "macro_f1": macro_f1}
        )
        candidates.append((macro_f1, -c_value, model))

    _, _, selected_model = max(candidates, key=lambda item: item[:2])
    selected_c = float(selected_model.named_steps["classifier"].C)
    selected_model.fit(
        pd.concat([train["text"], validation["text"]], ignore_index=True),
        pd.concat([train["label"], validation["label"]], ignore_index=True),
    )
    test_predictions = selected_model.predict(test["text"])
    result = score_predictions(
        "tfidf_logistic_regression",
        test["label"],
        test_predictions,
        {
            "C": selected_c,
            "class_weight": "balanced",
            "ngram_range": [1, 2],
            "selection_metric": "validation_macro_f1",
        },
    )
    return result, validation_results
