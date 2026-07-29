"""Language-specific gradient boosting recipes and scorer-aware weights."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


def make_model(language: str):
    """Return the model recipe used for a language.

    English benefits from smaller leaves; Hindi is more stable with stronger
    leaf regularisation. Both are deliberately small CPU models.
    """
    language = language.lower()
    if language == "english":
        return GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            min_samples_leaf=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=0,
        )
    if language == "hindi":
        return GradientBoostingClassifier(
            n_estimators=120,
            max_depth=3,
            min_samples_leaf=15,
            learning_rate=0.05,
            subsample=0.8,
            random_state=0,
        )
    return GradientBoostingClassifier(
        n_estimators=130,
        max_depth=3,
        min_samples_leaf=10,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
    )


def train_weights(y, hold_durations, target_delay=0.60):
    """Approximate the official objective during training.

    A hold shorter than the target action delay cannot create a cutoff, so it
    gets little weight. A hold longer than the target delay is dangerous and
    gets much more weight. EOT examples retain unit weight.
    """
    y = np.asarray(y)
    hold_durations = np.asarray(hold_durations)
    w = np.ones(len(y), dtype=np.float64)
    hold = y == 0
    w[hold] = np.where(hold_durations[hold] > target_delay, 3.0, 0.1)
    return w
