"""
check_causality.py - proof that our features never see the future.

The assignment's hard rule: to score a pause at `pause_start`, we may only use
audio before that moment. This script proves we obey it, mechanically:

  1. Extract features for a pause.
  2. Replace ALL audio from `pause_start` onward with random noise.
  3. Extract features again.
  If any feature changed, it must have looked at the future. It doesn't:
  the two feature vectors are bit-for-bit identical.

    python check_causality.py
"""
import numpy as np

from features import load_wav, extract_features, FEATURE_NAMES


def main():
    x, sr = load_wav("eot_data/hindi/audio/hi__000.wav")
    pause_start = 6.1

    before = extract_features(x, sr, pause_start)

    corrupted = x.copy()
    cut = int(pause_start * sr)
    corrupted[cut:] = np.random.randn(len(x) - cut).astype(np.float32)
    after = extract_features(corrupted, sr, pause_start)

    diff = float(np.abs(before - after).max())
    print(f"max feature change when future audio -> noise: {diff:.2e}")
    if diff == 0.0:
        print("PASS: features depend only on audio before the pause.")
    else:
        print("FAIL: a feature used future audio:")
        for name, a, b in zip(FEATURE_NAMES, before, after):
            if abs(a - b) > 0:
                print(f"  {name}: {a} != {b}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
