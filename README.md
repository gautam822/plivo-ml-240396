# End-of-Turn Detection

Predict, for each pause in a phone-call turn, the probability `p_eot` that the
speaker has finished — so a voice agent responds fast without talking over
people. CPU-only, no pretrained models.

## Setup

```bash
pip install numpy scipy scikit-learn
```

(scikit-learn is only needed to retrain; `predict.py` runs on numpy + scipy.)

## Predict on a data folder

```bash
python predict.py --data_dir eot_data/english --out predictions_english.csv
python predict.py --data_dir eot_data/hindi   --out predictions_hindi.csv
```

Works on unseen folders with the same schema; only `turn_id, audio_file,
pause_index, pause_start` are read. Output: `turn_id,pause_index,p_eot`.

## Score (official scorer, unchanged)

```bash
python score.py --data_dir eot_data/english --pred predictions_english.csv
python score.py --data_dir eot_data/hindi   --pred predictions_hindi.csv
```

## Reproduce from scratch

```bash
python train_model.py --data_dir eot_data --out model.json   # ~1 min
python eval.py                    # honest out-of-fold scores
python eval.py --stress           # + shuffled-split stress test
python check_causality.py         # proves no future audio is used
```

## Files

| file | what it is |
|------|------------|
| `features.py`        | causal feature extraction (`CausalAnalyzer`) — the core |
| `model.py`           | per-language model recipes + scorer-aware weighting |
| `dataset.py`         | folder → feature matrix |
| `train_model.py`     | trains 5-seed ensembles + language classifier → `model.json` |
| `predict.py`         | **the deliverable** — pure-numpy inference, name-agnostic routing |
| `eval.py`            | out-of-fold + stress evaluation |
| `check_causality.py` | mechanical proof features ignore future audio |
| `model.json`         | trained model bundle (plain numbers) |
| `score.py`, `baseline.py` | provided scorer and baseline (unchanged) |
| `RUNLOG.md`, `NOTES.md`, `SUMMARY.html` | write-ups |
