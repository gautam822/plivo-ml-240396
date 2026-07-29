"""
dataset.py - load a data folder into a feature matrix.

Shared by train_model.py (fits the model) and eval.py (checks it honestly),
so both see EXACTLY the same features. One source of truth avoids the classic
bug where training and evaluation compute features slightly differently.
"""
import csv
import os

import numpy as np

from features import load_wav, extract_features


def load_folder(data_dir, need_labels=True):
    """Read one language folder (english/ or hindi/) into arrays.

    Returns a dict with:
        X            feature matrix, shape (n_pauses, n_features)
        y            1 for eot, 0 for hold  (None if labels are absent)
        turn_ids     the turn each pause belongs to (for grouped splitting)
        keys         (turn_id, pause_index) pairs, for writing predictions

    `need_labels=False` is what predict.py uses on the hidden test set, where
    the `label` column may not exist. We only ever require the columns a live
    agent would actually have: turn_id, audio_file, pause_index, pause_start.
    """
    labels_path = os.path.join(data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path, encoding="utf-8")))

    audio_cache = {}                      # load each wav once, reuse for its pauses
    X, y, turn_ids, keys, hold_dur = [], [], [], [], []
    for r in rows:
        wav_path = os.path.join(data_dir, r["audio_file"])
        if wav_path not in audio_cache:
            audio_cache[wav_path] = load_wav(wav_path)
        x, sr = audio_cache[wav_path]

        X.append(extract_features(x, sr, float(r["pause_start"])))
        turn_ids.append(r["turn_id"])
        keys.append((r["turn_id"], r["pause_index"]))
        if need_labels:
            y.append(1 if r["label"] == "eot" else 0)
            # pause length is used ONLY to weight training (see model.py). It is
            # never a feature. We read it here because the labels file has it.
            hold_dur.append(float(r["pause_end"]) - float(r["pause_start"]))

    return {
        "X": np.array(X, dtype=np.float32),
        "y": np.array(y) if need_labels else None,
        "turn_ids": np.array(turn_ids),
        "keys": keys,
        "hold_durations": np.array(hold_dur, dtype=np.float32) if need_labels else None,
    }
