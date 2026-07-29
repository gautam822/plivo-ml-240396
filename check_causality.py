"""Mechanical test that changing future audio cannot change current features."""
from __future__ import annotations

import csv
import os

import numpy as np

from features import CausalAnalyzer, FEATURE_NAMES, load_wav


def test_pause(data_dir, row, rng):
    path = os.path.join(data_dir, row["audio_file"])
    x, sr = load_wav(path)
    pause_start = float(row["pause_start"])
    pause_index = int(row["pause_index"])
    before = CausalAnalyzer(x, sr).extract(pause_start, pause_index)

    changed = x.copy()
    cut = int(pause_start * sr)
    if cut < len(changed):
        changed[cut:] = rng.normal(0.0, 0.4, len(changed) - cut).astype(np.float32)
    after = CausalAnalyzer(changed, sr).extract(pause_start, pause_index)
    return float(np.max(np.abs(before - after))), before, after


def main():
    rng = np.random.default_rng(7)
    tested = 0
    maximum = 0.0
    for language in ("english", "hindi"):
        data_dir = os.path.join("eot_data", language)
        with open(os.path.join(data_dir, "labels.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Early, middle, and late pauses exercise different prefix lengths.
        for row in (rows[0], rows[len(rows) // 2], rows[-1]):
            diff, before, after = test_pause(data_dir, row, rng)
            tested += 1
            maximum = max(maximum, diff)
            if diff != 0.0:
                print(f"FAIL: {language} {row['turn_id']} pause {row['pause_index']}")
                for name, a, b in zip(FEATURE_NAMES, before, after):
                    if a != b:
                        print(f"  {name}: {a} != {b}")
                raise SystemExit(1)
    print(f"PASS: {tested} pauses tested; maximum future-audio effect={maximum:.2e}")


if __name__ == "__main__":
    main()
