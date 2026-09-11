from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def _representative_fold(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[pd.Index, pd.Index]:
    """Choose the fold closest to the full dataset's size and label mix."""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    overall = frame["label"].value_counts(normalize=True)
    target_fraction = 1 / n_splits

    candidates = []
    for fold_number, (remainder_idx, held_out_idx) in enumerate(
        splitter.split(frame, frame["label"], groups=frame["source_url"])
    ):
        held_out = frame.iloc[held_out_idx]
        distribution = held_out["label"].value_counts(normalize=True)
        distribution = distribution.reindex(overall.index, fill_value=0)
        label_distance = float((distribution - overall).abs().sum())
        size_distance = abs(len(held_out) / len(frame) - target_fraction)
        candidates.append(
            (label_distance + size_distance, fold_number, remainder_idx, held_out_idx)
        )

    _, _, remainder_idx, held_out_idx = min(candidates, key=lambda item: item[:2])
    return pd.Index(remainder_idx), pd.Index(held_out_idx)


def make_thread_held_out_splits(frame: pd.DataFrame) -> DatasetSplits:
    """Create deterministic train, validation, and test sets grouped by thread."""
    frame = frame.sort_values("id").reset_index(drop=True)
    train_validation_idx, test_idx = _representative_fold(
        frame,
        n_splits=6,
        random_state=42,
    )
    train_validation = frame.iloc[train_validation_idx].reset_index(drop=True)
    train_idx, validation_idx = _representative_fold(
        train_validation,
        n_splits=5,
        random_state=43,
    )

    splits = DatasetSplits(
        train=train_validation.iloc[train_idx].reset_index(drop=True),
        validation=train_validation.iloc[validation_idx].reset_index(drop=True),
        test=frame.iloc[test_idx].reset_index(drop=True),
    )
    _validate_split_integrity(splits)
    return splits


def _validate_split_integrity(splits: DatasetSplits) -> None:
    frames = splits.as_dict()
    all_ids: set[int] = set()
    all_sources: set[str] = set()

    for name, frame in frames.items():
        ids = set(frame["id"])
        sources = set(frame["source_url"])
        if all_ids.intersection(ids):
            raise ValueError(f"Rows overlap before the {name} split")
        if all_sources.intersection(sources):
            raise ValueError(f"Source threads overlap before the {name} split")
        all_ids.update(ids)
        all_sources.update(sources)
