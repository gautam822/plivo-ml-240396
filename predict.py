"""Causal end-of-turn prediction on an unseen language folder.

    python predict.py --data_dir <folder> --out predictions.csv

Only turn_id, audio_file, pause_index, pause_start are required in labels.csv;
label/pause_end are never read (a live agent cannot see the future).

How a probability is produced for each pause:
  1. Causal features are extracted from audio strictly before the pause.
  2. If the file naming carries a language hint (en__/hi__ prefixes or the
     folder name), that language's five-seed model ensemble is used.
  3. Otherwise the two language ensembles are blended, weighted by an
     audio-based language classifier - so the model still routes correctly
     on a hidden set with unfamiliar file names.
Inference is pure numpy over trees exported to JSON; scikit-learn is not
imported here, so library-version differences on the grading machine cannot
break prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from features import CausalAnalyzer, FEATURE_NAMES, load_wav

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(path=None):
    path = path or os.path.join(HERE, "model.json")
    with open(path, encoding="utf-8") as f:
        model = json.load(f)
    if model.get("feature_names") != FEATURE_NAMES:
        raise ValueError("model.json feature list does not match features.py")
    return model


def _one_tree(tree, X):
    feature = np.asarray(tree["feature"], dtype=np.int32)
    threshold = np.asarray(tree["threshold"], dtype=np.float64)
    left = np.asarray(tree["left"], dtype=np.int32)
    right = np.asarray(tree["right"], dtype=np.int32)
    value = np.asarray(tree["value"], dtype=np.float64)

    node = np.zeros(len(X), dtype=np.int32)
    is_leaf = feature < 0
    active = ~is_leaf[node]
    while active.any():
        rows = np.flatnonzero(active)
        nodes = node[rows]
        go_left = X[rows, feature[nodes]] <= threshold[nodes]
        node[rows] = np.where(go_left, left[nodes], right[nodes])
        active = ~is_leaf[node]
    return value[node]


def predict_proba_blob(blob, X):
    """Probability from one exported boosting model (pure numpy)."""
    X = np.asarray(X, dtype=np.float64)
    raw = np.full(len(X), blob["init_score"], dtype=np.float64)
    for tree in blob["trees"]:
        raw += blob["learning_rate"] * _one_tree(tree, X)
    raw = np.clip(raw, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-raw))


def predict_proba_ensemble(blobs, X):
    """Average probability across the five differently-seeded models."""
    return np.mean([predict_proba_blob(b, X) for b in blobs], axis=0)


def name_language(turn_id, audio_file, data_dir):
    """Language from naming hints, or None if the names don't say."""
    text = f"{turn_id} {audio_file} {os.path.basename(os.path.normpath(data_dir))}".lower()
    if str(turn_id).lower().startswith("en__") or "english" in text:
        return "english"
    if str(turn_id).lower().startswith("hi__") or "hindi" in text:
        return "hindi"
    return None


def predict_folder(data_dir, model):
    with open(os.path.join(data_dir, "labels.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_audio = defaultdict(list)
    for i, row in enumerate(rows):
        by_audio[row["audio_file"]].append((i, row))

    features = [None] * len(rows)
    named = [None] * len(rows)
    keys = [None] * len(rows)

    for audio_file, indexed_rows in by_audio.items():
        try:
            x, sr = load_wav(os.path.join(data_dir, audio_file))
            analyzer = CausalAnalyzer(x, sr)
        except Exception:
            analyzer = None                      # corrupt/missing: neutral features
        for i, row in indexed_rows:
            pause_index = int(row["pause_index"])
            if analyzer is not None:
                try:
                    features[i] = analyzer.extract(float(row["pause_start"]), pause_index)
                except Exception:
                    features[i] = np.zeros(len(FEATURE_NAMES), np.float32)
            else:
                features[i] = np.zeros(len(FEATURE_NAMES), np.float32)
            named[i] = name_language(row["turn_id"], audio_file, data_dir)
            keys[i] = (row["turn_id"], pause_index)

    X = np.asarray(features, dtype=np.float32)
    probabilities = np.zeros(len(rows), dtype=np.float64)
    ensembles = model["ensembles"]

    # 1) name-routed rows -> that language's ensemble
    for language in ("english", "hindi"):
        idx = np.flatnonzero([n == language for n in named])
        if len(idx) and language in ensembles:
            probabilities[idx] = predict_proba_ensemble(ensembles[language], X[idx])

    # 2) unnamed rows -> soft blend weighted by the audio language classifier
    idx = np.flatnonzero([n is None for n in named])
    if len(idx):
        have_both = "english" in ensembles and "hindi" in ensembles
        if have_both and model.get("language_classifier") is not None:
            p_hi = predict_proba_blob(model["language_classifier"], X[idx])
            p_en_model = predict_proba_ensemble(ensembles["english"], X[idx])
            p_hi_model = predict_proba_ensemble(ensembles["hindi"], X[idx])
            probabilities[idx] = p_hi * p_hi_model + (1.0 - p_hi) * p_en_model
        else:
            only = next(iter(ensembles.values()))
            probabilities[idx] = predict_proba_ensemble(only, X[idx])

    return keys, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", default="predictions.csv")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    model = load_model(args.model)
    keys, probabilities = predict_folder(args.data_dir, model)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for (turn_id, pause_index), probability in zip(keys, probabilities):
            writer.writerow([turn_id, pause_index, f"{probability:.6f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
