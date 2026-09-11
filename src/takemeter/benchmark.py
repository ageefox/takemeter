from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from takemeter.data import load_data
from takemeter.models import evaluate_majority, evaluate_tfidf
from takemeter.reporting import write_results
from takemeter.splits import make_thread_held_out_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TakeMeter benchmark")
    parser.add_argument("--data", type=Path, default=Path("takemeter_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("majority", "tfidf", "distilbert", "distilbert_weighted"),
        default=("majority", "tfidf", "distilbert", "distilbert_weighted"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = make_thread_held_out_splits(load_data(args.data))
    results = []
    details: dict[str, object] = {}

    if "majority" in args.models:
        results.append(evaluate_majority(splits.train, splits.test))
    if "tfidf" in args.models:
        tfidf, validation_results = evaluate_tfidf(
            splits.train, splits.validation, splits.test
        )
        results.append(tfidf)
        details["tfidf_validation"] = validation_results
    if "distilbert" in args.models:
        from takemeter.transformer import evaluate_distilbert

        development = pd.concat(
            [splits.train, splits.validation], ignore_index=True
        )
        transformer, history = evaluate_distilbert(
            development,
            splits.test,
            epochs=args.epochs,
        )
        results.append(transformer)
        details["distilbert_training"] = history
    if "distilbert_weighted" in args.models:
        from takemeter.transformer import evaluate_distilbert

        development = pd.concat(
            [splits.train, splits.validation], ignore_index=True
        )
        transformer, history = evaluate_distilbert(
            development,
            splits.test,
            epochs=args.epochs,
            class_weighted=True,
        )
        results.append(transformer)
        details["distilbert_weighted_training"] = history

    write_results(args.output_dir, args.data, splits, results, details)
    for result in results:
        print(
            f"{result.name}: accuracy={result.accuracy:.3f}, "
            f"macro_f1={result.macro_f1:.3f}"
        )


if __name__ == "__main__":
    main()
