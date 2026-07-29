"""
model.py - the classifier and how we weight its training samples.

Kept tiny and in one place so the two decisions that matter are easy to see:

  make_model()     -> which classifier we use and why.
  train_weights()  -> our key trick: teach the model to care most about the
                      mistakes that actually cost points in the scorer.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


def make_model():
    """A small gradient-boosted tree classifier.

    Why boosting: our features are a handful of tabular numbers with nonlinear,
    interacting effects (e.g. falling pitch matters more when energy is also
    dropping). Trees capture those interactions without us hand-coding them,
    and a shallow, modest ensemble stays fast on CPU and does not overfit
    ~250 pauses. Logistic regression was our sanity baseline; boosting beat it.
    """
    return GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,        # a little randomness -> less overfitting
        random_state=0,
    )


def train_weights(y, hold_durations):
    """Weight each training pause by how much a mistake on it would cost.

    This is the heart of what we add beyond an off-the-shelf classifier, and it
    comes straight from reading the scorer:

      * A HOLD pause only causes a "false cutoff" if the agent's silence delay
        is SHORTER than the pause. Long holds are the dangerous ones: the user
        goes quiet long enough that a hasty agent barges in. Short holds are
        almost harmless. So we weight a hold by its length -> the model works
        hardest to avoid firing on the long holds that actually lose points.

      * EOT pauses all matter equally (each is one chance to respond quickly),
        so they get a flat weight.

    Causality note: pause length is used ONLY as a training weight, never as a
    model input. At prediction time the model never sees it. Weighting training
    samples by label-derived information is standard and allowed - the causality
    rule governs what the model may look at when making a live decision.

    Args:
        y              array of 0/1 labels (1 = eot).
        hold_durations array the same length as y; for hold rows it holds the
                       pause length in seconds, for eot rows the value is ignored.
    """
    w = np.ones(len(y), dtype=np.float32)
    w[y == 1] = 1.0                       # eot rows: uniform weight
    # hold rows: weight grows with pause length, capped so no single long hold
    # dominates. A 0.1 s hold ~ weight 1; a >=1.5 s hold ~ weight 3.
    hold = y == 0
    w[hold] = 1.0 + 2.0 * np.clip(hold_durations[hold] / 1.5, 0.0, 1.0)
    return w
