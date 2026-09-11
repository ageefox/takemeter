import pandas as pd
import pytest

from takemeter.data import LABELS, load_data


def test_repository_dataset_satisfies_contract():
    frame = load_data("takemeter_dataset.csv")

    assert len(frame) == 212
    assert frame["id"].is_unique
    assert set(frame["label"]) == set(LABELS)
    assert frame["source_url"].nunique() == 28


def test_duplicate_ids_are_rejected(tmp_path):
    frame = pd.DataFrame(
        {
            "id": [1, 1, 3, 4],
            "text": ["text"] * 4,
            "label": list(LABELS),
            "source_site": ["site"] * 4,
            "source_url": [f"https://example.com/{i}" for i in range(4)],
            "source_context": ["thread"] * 4,
        }
    )
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="IDs must be unique"):
        load_data(path)
