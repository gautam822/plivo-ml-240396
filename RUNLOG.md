# RUNLOG — End-of-Turn Detection

Every entry: **hypothesis → what changed → score → conclusion.** The score is
the official `score.py` metric (mean response delay in ms at ≤5% false-cutoff
rate, **lower is better**), measured **out-of-fold**: every pause is scored by a
model that never saw its turn. AUC is a ranking diagnostic.

Baseline to beat (silence-only, given): **~1600 ms**.

---

### Run 0 — silence-only baseline
- **Hypothesis:** silence alone can't tell a mid-turn pause from a real ending.
- **Change:** ran the provided `baseline.py` (every pause `p_eot = 1.0`).
- **Score:** english **1600 ms** (AUC 0.514), hindi **850 ms** (AUC 0.501).
- **Conclusion:** AUC ≈ 0.5 confirms silence carries no information about
  *which* pause ends the turn. Hindi's lower delay is luck (shorter holds on
  average), not signal.

### Run 1 — first prosodic features + boosting + duration-weighted holds
- **Hypothesis:** endings show falling pitch, trailing energy, and
  final-syllable lengthening in the audio *before* the pause; long holds
  deserve more training weight because only they can cause a false cutoff.
- **Change:** 10 causal features; GradientBoosting; holds weighted by length.
- **Score (OOF):** english 1376 ms (AUC 0.549), hindi 888 ms (AUC 0.643).
- **Conclusion:** real signal on Hindi, weak on English. Fix features next,
  not the model.

### Run 2 — fix two broken features found by error analysis
- **Hypothesis:** the pitch-slope feature read ≈0.00 for everyone (bad
  normalisation crushed it); the syllable-length ratio had 10x outliers.
- **Change:** slope in **semitones/second**; length ratio clipped.
- **Score (OOF):** english 1348 ms (0.552), hindi 902 ms (**0.673**).
- **Conclusion:** the semitone fix revived the strongest cue on Hindi.

### Run 3 — longer pitch window; a real cross-language finding
- **Change:** pitch slope over the last ~700 ms of voiced frames.
- **Score (OOF):** english 1362 ms (0.574), hindi 857 ms (**0.699**).
- **Finding:** with the clean window, **Hindi endings fall in pitch (eot slope
  negative, hold positive; separation 0.45) but English endings do not
  (separation 0.05)**. English marks completion with energy and timing here.
  We keep pitch for Hindi and let English lean on energy/timing.

### Run 4 — spectral tilt (voice quality)
- **Change:** log(low/high-frequency energy) over the final voiced frames.
- **Score (OOF):** english 1405 ms (0.570), hindi **817 ms (0.704)**.
- **Conclusion:** best Hindi delay so far; kept.

### Run 5 — rebuilt feature extractor (cached, adaptive, richer) + per-language models
- **Hypothesis (three parts):** (a) recomputing all frames per pause is
  wastefully slow — cache per file; (b) a fixed energy threshold mislabels
  speech activity across different recording levels — make it adaptive per
  prefix; (c) endings also live in fine spectral detail — add band energies,
  centroid, flatness, flux, pitch confidence, segment-rhythm timing.
- **Change:** `CausalAnalyzer` computes frame descriptors once per file and
  slices causally per pause; adaptive activity threshold from prefix
  percentiles; 38 features; **separate English and Hindi models** (different
  leaf regularisation); sharper hold weighting (holds shorter than the target
  action delay get 0.1x — they cannot cause a cutoff; longer get 3x).
- **Score (OOF):** english **1150 ms (AUC 0.690)**, hindi **726 ms (AUC 0.759)**.
- **Conclusion:** the biggest jump. Per-language models beat one shared model
  because the two languages use different ending cues (Run 3's finding).

### Run 6 — seed ensembling
- **Hypothesis:** subsampled boosting has run-to-run variance; averaging five
  differently-seeded models should give a small, reliable gain.
- **Change:** 5-seed ensembles per language, probabilities averaged.
- **Score (OOF):** english **1138 ms**, hindi **698 ms** (AUC 0.759).
- **Conclusion:** kept — cheap and principled variance reduction.

### Run 7 — weight-parameter sweep (negative result, reported honestly)
- **Hypothesis:** the 0.60 s cutoff / 3.0-vs-0.1 hold weights were guesses;
  sweeping them might help.
- **Change:** swept cutoff ∈ {0.45, 0.60, 0.75} and weights on fixed folds,
  then **stress-tested the "winners" across 5 shuffled grouped splits**.
- **Result:** every apparent gain vanished under shuffled splits (english
  wl=5 median 1215 vs wl=3 1226 — noise; hindi ws=0.3 was *worse*, 850 vs
  774). The fixed-fold numbers had been flattering.
- **Conclusion:** no change adopted. On 100 turns the delay metric swings
  ±60 ms between splits; anything not robust across splits is overfitting.

### Run 8 — audio-based language routing for the hidden set
- **Hypothesis:** `predict.py` routed by file naming (`en__`/`hi__`). A hidden
  set with different names would fall to a combined fallback model that costs
  **850 ms vs 698 ms** on Hindi — a 150 ms penalty for a filename.
- **Change:** trained a language classifier on the same causal features
  (81.5% turn-level accuracy); unnamed data now gets a **soft blend** of the
  two language ensembles weighted by prob(hindi). Named data still routes
  directly.
- **Score (OOF, simulating unknown naming):** hindi **742 ms** (vs 850
  before), english **1116 ms**. Named routing unchanged (698 / 1138).
- **Conclusion:** the shipped model no longer depends on the graders' file
  naming. Kept.

## Final honest results

| language | silence baseline | shipped model (OOF) | unknown-naming path |
|----------|------------------|---------------------|---------------------|
| english  | 1600 ms          | **1138 ms** (AUC 0.69) | 1116 ms |
| hindi    | 850 ms (AUC .50) | **698 ms** (AUC 0.76) | 742 ms |

Scoring the final model on its own training turns gives ~AUC 0.99 /
~250 ms — that is train-on-test leakage and is **not** claimed anywhere.
The numbers above are from models that never saw the turns they score.
