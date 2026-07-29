"""
predict.py - the shipped model. Run by the graders on data they never showed us:

    python predict.py --data_dir <folder> --out predictions.csv

<folder> has the same layout as the provided data (a labels.csv listing pauses
and the wavs they point to). We output one row per pause:

    turn_id,pause_index,p_eot

p_eot is our probability that the turn is over at that pause.

Design choices that keep this robust on a machine we don't control:
  * We load the model from model.json (plain numbers) and run the tree math in
    pure numpy. predict.py does NOT import scikit-learn, so a version mismatch
    on the grader's machine cannot break us.
  * We resolve model.json relative to THIS file, not the working directory, so
    it is found no matter where the grader runs us from.
  * We only require the columns a live agent would have: turn_id, audio_file,
    pause_index, pause_start. If the file also has label / pause_end we ignore
    them (a live agent cannot see the future, and neither do we).
  * Any single unreadable pause falls back to a neutral probability instead of
    crashing the whole run.
"""
import argparse
import csv
import json
import os

import numpy as np

from features import load_wav, extract_features

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(path=None):
    """Load the exported model. Defaults to model.json next to this file."""
    if path is None:
        path = os.path.join(HERE, "model.json")
    with open(path) as f:
        return json.load(f)


def _one_tree(tree, X):
    """Run every row of X down one decision tree, return each row's leaf value.

    Standard tree walk: at each node compare feature[node] to threshold[node];
    threshold == -2 marks a leaf (sklearn's convention), where we read the value.
    Vectorised over rows so it stays fast.
    """
    feat = np.asarray(tree["feature"])
    thr = np.asarray(tree["threshold"])
    left = np.asarray(tree["left"])
    right = np.asarray(tree["right"])
    val = np.asarray(tree["value"])

    node = np.zeros(len(X), dtype=int)              # everyone starts at the root
    is_leaf = feat < 0                              # sklearn uses -2 for leaves
    active = ~is_leaf[node]
    while active.any():
        rows = np.where(active)[0]
        n = node[rows]
        go_left = X[rows, feat[n]] <= thr[n]
        node[rows] = np.where(go_left, left[n], right[n])
        active = ~is_leaf[node]
    return val[node]


def predict_proba_json(model, X):
    """Probability of eot for each row of X, using the JSON model in pure numpy.

    Boosting sums the trees: raw = init + lr * sum(tree outputs); then sigmoid.
    This mirrors GradientBoostingClassifier exactly (verified in train_model.py).
    """
    X = np.asarray(X, dtype=np.float64)
    raw = np.full(len(X), model["init_score"], dtype=np.float64)
    lr = model["learning_rate"]
    for tree in model["trees"]:
        raw += lr * _one_tree(tree, X)
    return 1.0 / (1.0 + np.exp(-raw))               # sigmoid -> probability


def predict_folder(data_dir, model):
    """Compute (turn_id, pause_index, p_eot) for every pause in a folder."""
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"),
                                    encoding="utf-8")))
    audio_cache = {}
    feats, keys = [], []
    for r in rows:
        wav_path = os.path.join(data_dir, r["audio_file"])
        try:
            if wav_path not in audio_cache:
                audio_cache[wav_path] = load_wav(wav_path)
            x, sr = audio_cache[wav_path]
            f = extract_features(x, sr, float(r["pause_start"]))
        except Exception:
            # Never let one bad file sink the whole run: emit neutral features,
            # which the model maps to a middling probability.
            f = np.zeros(len(model["feature_names"]), dtype=np.float32)
        feats.append(f)
        keys.append((r["turn_id"], r["pause_index"]))

    probs = predict_proba_json(model, np.array(feats))
    return keys, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=None, help="path to model.json")
    args = ap.parse_args()

    model = load_model(args.model)
    keys, probs = predict_folder(args.data_dir, model)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, probs):
            w.writerow([tid, pi, f"{p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
