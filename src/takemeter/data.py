from __future__ import annotations

from pathlib import Path

import pandas as pd


LABELS = (
    "technical_help",
    "showcase_reaction",
    "critique_feedback",
    "meta_opinion",
)
REQUIRED_COLUMNS = {
    "id",
    "text",
    "label",
    "source_site",
    "source_url",
    "source_context",
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the labeled forum-post dataset."""
    frame = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset contains no rows")
    if frame["id"].duplicated().any():
        raise ValueError("Dataset IDs must be unique")
    if frame["text"].isna().any() or frame["text"].str.strip().eq("").any():
        raise ValueError("Every row must contain text")
    if frame["source_url"].isna().any() or frame["source_url"].str.strip().eq("").any():
        raise ValueError("Every row must contain a source URL")

    unknown = set(frame["label"]).difference(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")
    if set(frame["label"]) != set(LABELS):
        raise ValueError("Dataset must contain every configured label")

    frame = frame.copy()
    frame["id"] = pd.to_numeric(frame["id"], errors="raise").astype(int)
    return frame.sort_values("id").reset_index(drop=True)
