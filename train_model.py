"""Train the shipped model bundle and export it as portable JSON.

The bundle contains, for each language, a small ensemble of gradient-boosted
tree models (five seeds, averaged at prediction time - averaging removes the
run-to-run variance of subsampled boosting), plus an audio-based language
classifier used when file naming gives no language hint.

    python train_model.py --data_dir eot_data --out model.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from dataset import load_folder
from features import FEATURE_NAMES
from model import make_model, train_weights

N_SEEDS = 5


def _init_raw(clf) -> float:
    prior = float(clf.init_.class_prior_[1])
    return float(np.log(prior / (1.0 - prior)))


def export_boosting(clf) -> dict:
    trees = []
    for estimator in clf.estimators_[:, 0]:
        tree = estimator.tree_
        trees.append({
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "left": tree.children_left.tolist(),
            "right": tree.children_right.tolist(),
            "value": tree.value.reshape(-1).tolist(),
        })
    return {
        "init_score": _init_raw(clf),
        "learning_rate": float(clf.learning_rate),
        "trees": trees,
    }


def fit_language_ensemble(language, X, y, hold_durations):
    """Five differently-seeded models; predictions are averaged at inference."""
    blobs, fitted = [], []
    for seed in range(N_SEEDS):
        clf = make_model(language)
        clf.random_state = seed
        clf.fit(X, y, sample_weight=train_weights(y, hold_durations))
        blobs.append(export_boosting(clf))
        fitted.append(clf)
    return blobs, fitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="eot_data")
    parser.add_argument("--out", default="model.json")
    args = parser.parse_args()

    data = {}
    for language in ("english", "hindi"):
        path = os.path.join(args.data_dir, language)
        if os.path.isdir(path):
            data[language] = load_folder(path)
            print(f"{language}: {len(data[language]['y'])} pauses")
    if not data:
        raise SystemExit("No english/ or hindi/ folders found")

    ensembles, fitted = {}, {}
    for language, d in data.items():
        ensembles[language], fitted[language] = fit_language_ensemble(
            language, d["X"], d["y"], d["hold_durations"]
        )

    # Audio-based language classifier: prob(hindi) from the same causal
    # features. Used to softly blend the two language models whenever the
    # folder/file naming carries no language hint (hidden sets may be named
    # anything). 0 = english, 1 = hindi.
    lang_blob = None
    if len(data) == 2:
        Xl = np.vstack([data["english"]["X"], data["hindi"]["X"]])
        yl = np.concatenate([
            np.zeros(len(data["english"]["y"])),
            np.ones(len(data["hindi"]["y"])),
        ])
        lc = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0
        )
        lc.fit(Xl, yl)
        lang_blob = export_boosting(lc)
        fitted["_lang"] = (lc, Xl)

    payload = {
        "schema_version": 3,
        "feature_names": FEATURE_NAMES,
        "n_seeds": N_SEEDS,
        "routing": ("name pattern en__/hi__ or folder name -> that language's "
                    "ensemble; otherwise soft-blend the two ensembles by the "
                    "audio language classifier's probability"),
        "ensembles": ensembles,
        "language_classifier": lang_blob,
    }

    # Verify portable inference matches sklearn everywhere before saving.
    from predict import predict_proba_blob, predict_proba_ensemble
    max_diff = 0.0
    for language, d in data.items():
        p_json = predict_proba_ensemble(ensembles[language], d["X"])
        p_skl = np.mean(
            [m.predict_proba(d["X"])[:, 1] for m in fitted[language]], axis=0
        )
        max_diff = max(max_diff, float(np.max(np.abs(p_json - p_skl))))
    if lang_blob is not None:
        lc, Xl = fitted["_lang"]
        p_json = predict_proba_blob(lang_blob, Xl)
        p_skl = lc.predict_proba(Xl)[:, 1]
        max_diff = max(max_diff, float(np.max(np.abs(p_json - p_skl))))
    if max_diff >= 1e-6:
        raise RuntimeError(f"JSON inference mismatch: {max_diff}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"saved {args.out} ({size_mb:.1f} MB); "
          f"max JSON/sklearn difference={max_diff:.2e}")


if __name__ == "__main__":
    main()
