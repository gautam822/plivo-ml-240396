"""Load one language folder into a causal feature matrix."""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

from features import CausalAnalyzer, load_wav


def load_folder(data_dir: str, need_labels: bool = True) -> dict:
    labels_path = os.path.join(data_dir, "labels.csv")
    with open(labels_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Process one WAV at a time: frame descriptors are computed once and then
    # reused for all pauses in that turn. This is much faster and lower-memory
    # than recomputing the full prefix for every pause.
    by_audio: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_audio[row["audio_file"]].append((i, row))

    X = [None] * len(rows)
    y = np.zeros(len(rows), dtype=np.int8) if need_labels else None
    turn_ids = np.empty(len(rows), dtype=object)
    keys = [None] * len(rows)
    hold_durations = np.zeros(len(rows), dtype=np.float64) if need_labels else None

    for audio_file, indexed_rows in by_audio.items():
        x, sr = load_wav(os.path.join(data_dir, audio_file))
        analyzer = CausalAnalyzer(x, sr)
        for i, row in indexed_rows:
            pause_index = int(row["pause_index"])
            X[i] = analyzer.extract(float(row["pause_start"]), pause_index)
            turn_ids[i] = row["turn_id"]
            keys[i] = (row["turn_id"], pause_index)
            if need_labels:
                y[i] = 1 if row["label"] == "eot" else 0
                hold_durations[i] = float(row["pause_end"]) - float(row["pause_start"])

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": y,
        "turn_ids": np.asarray(turn_ids),
        "keys": keys,
        "hold_durations": hold_durations,
    }
