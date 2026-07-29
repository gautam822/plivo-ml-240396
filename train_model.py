"""
train_model.py - fit the final model and save it as portable JSON.

    python train_model.py --data_dir eot_data --out model.json

Why train on BOTH languages together: the hidden test set is "unseen turns,
mostly Hindi". Our features are speaker-relative (pitch measured against each
speaker's own median, energy against their own speech baseline), so combining
English and Hindi gives the model more examples of the same underlying cues
without the absolute-scale differences between languages confusing it. Our
cross-language checks in eval.py confirmed this generalises.

Why export to JSON instead of pickling the sklearn model: the graders run our
predict.py on THEIR machine, with a possibly different scikit-learn version. A
pickle can refuse to load across versions. So we read the trees out of the
fitted model and write them as plain numbers. predict.py then re-implements the
(very simple) tree math in pure numpy - it does not import sklearn at all. We
verify the JSON reproduces sklearn's probabilities exactly before saving.
"""
import argparse
import json
import os

import numpy as np

from dataset import load_folder
from model import make_model, train_weights


def export_boosting_to_json(clf, feature_names):
    """Pull the decision trees and coefficients out of a fitted
    GradientBoostingClassifier into a plain dict of Python numbers.

    A gradient-boosted classifier makes a prediction by:
      raw = init_score + learning_rate * sum(each tree's output)
      probability = sigmoid(raw)
    Each tree is a set of nodes; at each node we compare one feature to a
    threshold and go left or right until we reach a leaf value. That is all the
    math predict.py needs, so that is all we save.
    """
    trees = []
    for tree_arr in clf.estimators_[:, 0]:          # binary clf -> one tree per round
        t = tree_arr.tree_
        trees.append({
            "feature": t.feature.tolist(),          # which feature each node tests
            "threshold": t.threshold.tolist(),      # the value it compares against
            "left": t.children_left.tolist(),       # node index if test is True
            "right": t.children_right.tolist(),     # node index if test is False
            "value": t.value.reshape(-1).tolist(),  # leaf outputs
        })
    return {
        "feature_names": list(feature_names),
        "init_score": _init_raw(clf),
        "learning_rate": float(clf.learning_rate),
        "trees": trees,
    }


def _init_raw(clf):
    """The model's starting log-odds before any tree is added."""
    # GradientBoostingClassifier stores the initial prediction as a log-odds.
    prior = clf.init_.class_prior_[1]
    return float(np.log(prior / (1.0 - prior)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="eot_data",
                    help="folder containing english/ and hindi/ subfolders")
    ap.add_argument("--out", default="model.json")
    args = ap.parse_args()

    # Gather every language subfolder that actually exists.
    langs = [d for d in ("english", "hindi")
             if os.path.isdir(os.path.join(args.data_dir, d))]
    if not langs:
        raise SystemExit(f"no english/ or hindi/ folder under {args.data_dir}")

    X_parts, y_parts, hd_parts = [], [], []
    for lang in langs:
        d = load_folder(os.path.join(args.data_dir, lang))
        X_parts.append(d["X"])
        y_parts.append(d["y"])
        hd_parts.append(d["hold_durations"])
        print(f"  {lang}: {len(d['y'])} pauses")
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    hd = np.concatenate(hd_parts)
    print(f"training on {len(y)} pauses from {len(langs)} language(s)")

    from features import FEATURE_NAMES
    clf = make_model()
    clf.fit(X, y, sample_weight=train_weights(y, hd))

    # Export to JSON and VERIFY it matches sklearn before trusting it.
    blob = export_boosting_to_json(clf, FEATURE_NAMES)
    from predict import predict_proba_json      # reuse the exact inference code
    p_json = predict_proba_json(blob, X)
    p_sklearn = clf.predict_proba(X)[:, 1]
    max_diff = float(np.max(np.abs(p_json - p_sklearn)))
    assert max_diff < 1e-6, f"JSON export mismatch! max diff = {max_diff}"
    print(f"JSON inference matches sklearn (max diff {max_diff:.2e})")

    with open(args.out, "w") as f:
        json.dump(blob, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
