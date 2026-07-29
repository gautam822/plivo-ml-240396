# NOTES

1. **Signal used:** for each pause we read only the audio *before* it and
   measure prosodic end-of-turn cues — pitch slope (in semitones, speaker-
   relative), energy trailing off vs the speaker's own speech baseline, final-
   syllable lengthening, breathy/creaky voice quality (spectral tilt), how long
   the turn has run, and how many times the speaker already paused.
2. **Why speaker-relative:** absolute pitch and loudness differ by speaker and
   language, so every cue is measured against that speaker's own running
   statistics — this is what lets an English-trained signal transfer to Hindi.
3. **Model:** a small gradient-boosted tree classifier (150 shallow trees) that
   combines these cues; it is exported to plain JSON and run with pure numpy at
   prediction time so it cannot break across library versions.
4. **Key trick:** training samples are weighted by hold length, because the
   scorer only penalises a false cutoff when the agent fires *before* a long
   hold ends — so the model learns to be most careful exactly where mistakes
   cost points.
5. **Best honest result (out-of-fold):** Hindi — the hidden-test language —
   improved from a coin-flip (AUC 0.50, ~850 ms) to **AUC 0.70 at ~817 ms**,
   roughly halving the silence baseline's delay within the 5% interruption
   budget; English improved from 1600 ms to ~1405 ms.
6. **Where it fails:** English turn-ends are *not* pitch-marked in this data
   (verified: eot vs hold pitch-slope separation ≈ 0.05), so English leans on
   energy and timing and gains less than Hindi.
7. It also struggles on very early pauses (little audio context) and on
   hesitant speakers who trail off mid-thought exactly like they do at a real
   ending.
8. **Evaluation honesty:** we report out-of-fold numbers; scoring the final
   model on its own training turns gives a misleading ~250 ms / AUC 0.98, which
   we deliberately do not claim.
9. **With one more day:** add a lightweight per-speaker pitch-range calibration,
   a short sequence model over the pause history within a turn, and a
   discourse-marker energy feature for English.
10. Everything runs on a laptop CPU in seconds (~56 ms per pause end to end),
    matching a real-time voice agent's latency budget.
