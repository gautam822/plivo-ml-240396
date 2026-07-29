"""Honest grouped evaluation for the exact shipped model recipes."""
from __future__ import annotations

import argparse
import csv
import os
import tempfile

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from dataset import load_folder
from model import make_model, train_weights
from score import score


def _write_predictions(keys, probabilities, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for (turn_id, pause_index), probability in zip(keys, probabilities):
            writer.writerow([turn_id, pause_index, f"{probability:.6f}"])


def evaluate_predictions(data_dir, keys, probabilities):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _write_predictions(keys, probabilities, path)
        return score(os.path.join(data_dir, "labels.csv"), path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def oof(language, data_dir, seed=None):
    d = load_folder(data_dir)
    probabilities = np.zeros(len(d["y"]), dtype=np.float64)
    splitter = (
        GroupKFold(n_splits=5)
        if seed is None
        else StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    )
    for train_idx, test_idx in splitter.split(d["X"], d["y"], d["turn_ids"]):
        clf = make_model(language)
        clf.fit(
            d["X"][train_idx],
            d["y"][train_idx],
            sample_weight=train_weights(
                d["y"][train_idx], d["hold_durations"][train_idx]
            ),
        )
        probabilities[test_idx] = clf.predict_proba(d["X"][test_idx])[:, 1]
    return evaluate_predictions(data_dir, d["keys"], probabilities)


def format_result(result):
    return (
        f"delay={result['latency'] * 1000:.0f} ms  "
        f"AUC={result['auc']:.3f}  cut={result['cutoff'] * 100:.1f}%  "
        f"threshold={result['threshold']:.2f}  action-delay={result['delay'] * 1000:.0f} ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="eot_data")
    parser.add_argument("--stress", action="store_true",
                        help="also run shuffled grouped folds (slower)")
    args = parser.parse_args()

    print("Fixed five-fold GroupKFold (directly comparable between iterations)")
    fixed = {}
    for language in ("english", "hindi"):
        path = os.path.join(args.data_dir, language)
        fixed[language] = oof(language, path)
        print(f"  {language:8s} {format_result(fixed[language])}")

    if args.stress:
        print("\nShuffled grouped-fold stress test")
        seeds_by_language = {"english": range(8), "hindi": range(12)}
        for language, seeds in seeds_by_language.items():
            path = os.path.join(args.data_dir, language)
            values = []
            for seed in seeds:
                result = oof(language, path, seed=seed)
                values.append((result["latency"] * 1000, result["auc"]))
            values = np.asarray(values)
            print(
                f"  {language:8s} median delay={np.median(values[:, 0]):.0f} ms  "
                f"mean delay={np.mean(values[:, 0]):.0f} ms  "
                f"median AUC={np.median(values[:, 1]):.3f}"
            )


if __name__ == "__main__":
    main()
