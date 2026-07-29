"""
eval.py - measure a model HONESTLY, the way the hidden test will.

Two checks, both run from the command line:

1. Out-of-fold (OOF) score within a language.
   We split turns into 5 folds, and for each fold we predict it using a model
   trained on the OTHER four folds. So every pause is scored by a model that
   never saw its turn. We then hand those predictions to the official score.py.
   This is the number we trust - unlike refitting on all data and scoring the
   same data, which flatters itself.

2. Cross-language transfer.
   Train on English, score on Hindi (and vice-versa). The hidden test set is
   "unseen turns, mostly Hindi", so a feature that helps within a language but
   hurts across languages is a trap. This check catches that.

    python eval.py            # runs both checks on english + hindi
"""
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.model_selection import GroupKFold

from dataset import load_folder
from model import make_model, train_weights


def _write_pred(keys, probs, path):
    """Write predictions in the exact format score.py expects."""
    with open(path, "w", newline="") as f:
        f.write("turn_id,pause_index,p_eot\n")
        for (tid, pi), p in zip(keys, probs):
            f.write(f"{tid},{pi},{p:.4f}\n")


def _official_score(data_dir, pred_path):
    """Call the untouched official scorer and pull out the delay in ms."""
    out = subprocess.run(
        [sys.executable, "score.py", "--data_dir", data_dir, "--pred", pred_path],
        capture_output=True, text=True,
    ).stdout
    delay = auc = None
    for line in out.splitlines():
        if "mean response delay" in line:
            delay = float(line.split(":")[1].strip().split()[0])
        if "AUC=" in line:
            auc = float(line.split("AUC=")[1].split()[0])
    return delay, auc


def oof_score(data_dir, n_folds=5):
    """Out-of-fold delay for one language folder."""
    d = load_folder(data_dir)
    X, y, groups, keys, hd = d["X"], d["y"], d["turn_ids"], d["keys"], d["hold_durations"]

    probs = np.zeros(len(y), dtype=np.float32)
    gkf = GroupKFold(n_splits=n_folds)
    for tr, te in gkf.split(X, y, groups):
        model = make_model()
        model.fit(X[tr], y[tr], sample_weight=train_weights(y[tr], hd[tr]))
        probs[te] = model.predict_proba(X[te])[:, 1]

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        _write_pred(keys, probs, f.name)
        return _official_score(data_dir, f.name)


def cross_language(train_dir, test_dir):
    """Train on one language, score on the other."""
    tr = load_folder(train_dir)
    te = load_folder(test_dir)
    model = make_model()
    model.fit(tr["X"], tr["y"], sample_weight=train_weights(tr["y"], tr["hold_durations"]))
    probs = model.predict_proba(te["X"])[:, 1]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        _write_pred(te["keys"], probs, f.name)
        return _official_score(test_dir, f.name)


def main():
    print("=" * 58)
    print("OUT-OF-FOLD (honest within-language score)")
    print("=" * 58)
    for lang in ["english", "hindi"]:
        delay, auc = oof_score(f"eot_data/{lang}")
        print(f"  {lang:<8}  delay = {delay:>6.0f} ms   AUC = {auc:.3f}")

    print()
    print("=" * 58)
    print("CROSS-LANGUAGE (proxy for the mostly-Hindi hidden set)")
    print("=" * 58)
    d, a = cross_language("eot_data/english", "eot_data/hindi")
    print(f"  train english -> score hindi :  delay = {d:>6.0f} ms   AUC = {a:.3f}")
    d, a = cross_language("eot_data/hindi", "eot_data/english")
    print(f"  train hindi   -> score english: delay = {d:>6.0f} ms   AUC = {a:.3f}")


if __name__ == "__main__":
    main()
