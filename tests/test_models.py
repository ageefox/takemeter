import pandas as pd
import pytest

from takemeter.data import load_data
from takemeter.models import evaluate_majority, evaluate_tfidf
from takemeter.splits import make_thread_held_out_splits


def test_lightweight_benchmarks_return_one_prediction_per_test_row():
    splits = make_thread_held_out_splits(load_data("takemeter_dataset.csv"))

    majority = evaluate_majority(splits.train, splits.test)
    tfidf, validation_results = evaluate_tfidf(
        splits.train, splits.validation, splits.test
    )

    assert len(majority.predictions) == len(splits.test)
    assert len(tfidf.predictions) == len(splits.test)
    assert len(validation_results) == 3
    assert tfidf.parameters["C"] in {0.25, 1.0, 4.0}


def test_published_lightweight_metrics_match_current_code():
    splits = make_thread_held_out_splits(load_data("takemeter_dataset.csv"))
    actual_majority = evaluate_majority(splits.train, splits.test)
    actual_tfidf, _ = evaluate_tfidf(splits.train, splits.validation, splits.test)
    published = pd.read_csv("results/metrics.csv").set_index("model")

    for result in (actual_majority, actual_tfidf):
        assert published.loc[result.name, "accuracy"] == pytest.approx(
            result.accuracy, abs=1e-6
        )
        assert published.loc[result.name, "macro_f1"] == pytest.approx(
            result.macro_f1, abs=1e-6
        )
