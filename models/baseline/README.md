# Baseline snapshot — DO NOT OVERWRITE

Immutable record of the four custom models **as published before hyperparameter
tuning began** (2026-07-15). Every tuned result is compared against this.

## Published baseline headline (from `model_report.json`, generated 2026-07-15 02:25)

| Model    | Headline metric            | Value  |
|----------|----------------------------|--------|
| Blink    | classification accuracy    | 0.9817 |
| Voice    | classification accuracy    | 0.9196 |
| Gaze     | PCK @ 5° (normalized patch)| 0.9348 |
| Head pose| PCK @ 15° (AFLW2000)       | 0.8766 |

Full per-model metric suites are in each `*.metrics.json` and in `model_report.json`.
The published reports are preserved under `reports/baseline/`.

## Provenance of each file (honest accounting)

- `gaze_cnn.pt`, `headpose_cnn.pt`, `voice_kws.pt` (+ their `.metrics.json`,
  `voice_kws.labels.json`): the **exact untouched published models** — copied
  straight from `models/custom/` before any retrain overwrote them.
- `blink_cnn.pt` + `blink_cnn.metrics.json`: the **2026-07-13 blink model
  (accuracy 0.9815)**, a near-identical sibling of the published 0.9817 model.
  The exact published `.pt` was lost early in the tuning session — a `--smoke`
  run and two throwaway `--epochs` runs overwrote `models/custom/blink_cnn.pt`
  before this snapshot was taken (the bug that let `--smoke` write to the
  production path is now fixed in the trainers). The published **number**
  (0.9817) survives in `model_report.json` and `reports/baseline/`; only the
  weight file is the 0.9815 sibling.
- `model_report.json`: the published aggregate (headline 0.9817 / 0.9196 /
  0.9348 / 0.8766).

## IMPORTANT — why these numbers are NOT the fair comparison baseline

These published figures were computed under the **old, leaky** protocol:
- Blink/gaze/headpose used a **random row split**, so near-duplicate frames of
  the same subject/participant/face sat on both sides of the split — the scores
  reward memorisation.
- The gaze 0.9348 (PCK@5°) is the **training-domain** number: it was measured on
  the same normalized MPIIGaze patches the model trained on. It does not
  transfer to the live FaceMesh pipeline (~9–10° there).

The tuning work re-runs everything under a **subject/participant/face-disjoint
3-way split** with a held-out test set touched once. To isolate the *effect of
tuning* (not the effect of fixing the leak), the final report also carries a
**fair baseline**: the original hyperparameters retrained under the new honest
protocol. So three columns are reported per model:
1. Published baseline (here) — old protocol, for the record.
2. Fair baseline — original hyperparameters, honest protocol.
3. Tuned — tuned hyperparameters, honest protocol.
