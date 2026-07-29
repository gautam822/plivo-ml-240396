# End-of-Turn Detection

Predict, for each pause in a phone-call turn, the probability `p_eot` that the
speaker has actually finished — so a voice agent can respond quickly without
talking over people. Everything runs on a laptop CPU in seconds.

## Setup

```bash
pip install numpy scipy scikit-learn
```

(`scikit-learn` is only needed to *train*. Prediction uses pure numpy.)

## Run the model on a data folder

The data folder must contain a `labels.csv` and the wavs it references
(`english/` and `hindi/` are laid out this way).

```bash
python predict.py --data_dir eot_data/english --out predictions_english.csv
python predict.py --data_dir eot_data/hindi   --out predictions_hindi.csv
```

Output columns: `turn_id,pause_index,p_eot`.

## Score it (official scorer, unchanged)

```bash
python score.py --data_dir eot_data/english --pred predictions_english.csv
python score.py --data_dir eot_data/hindi   --pred predictions_hindi.csv
```

## Reproduce everything from scratch

```bash
python train_model.py --data_dir eot_data --out model.json   # ~3 seconds
python eval.py                                                # honest out-of-fold scores
python check_causality.py                                     # proves no future audio is used
```

## Files

| file | what it is |
|------|------------|
| `features.py`        | causal prosodic features (the core of the solution) |
| `model.py`           | classifier + duration-based sample weighting |
| `dataset.py`         | loads a folder into a feature matrix |
| `train_model.py`     | fits the final model, exports `model.json` |
| `predict.py`         | **the deliverable** — pure-numpy inference on unseen folders |
| `eval.py`            | out-of-fold + cross-language honest evaluation |
| `check_causality.py` | proof that features never see audio after the pause |
| `model.json`         | the trained model, as plain numbers |
| `score.py`, `baseline.py` | provided scorer and baseline (unchanged) |
| `RUNLOG.md`, `NOTES.md`, `SUMMARY.html` | write-ups |
