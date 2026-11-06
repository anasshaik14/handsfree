# Custom models — training

Four models, each trained **from scratch** (own architecture, not a
fine-tuned backbone) on a public dataset. Each becomes the **primary**
provider for its input; the classical providers stay as automatic fallbacks.

| Input | Script | Dataset | Model input | Output | Saved weights |
|---|---|---|---|---|---|
| Blink | `train_blink.py` | [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) | 32×32 grayscale eye crop | open / closed | `models/custom/blink_cnn.pt` |
| Gaze | `train_gaze.py` | [MPIIGaze](https://www.mpi-inf.mpg.de/mpiigaze) (Normalized) | 36×60 grayscale eye patch | gaze pitch/yaw (rad) | `models/custom/gaze_cnn.pt` |
| Head pose | `train_headpose.py` | [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) / AFLW2000 | 64×64 RGB face crop | yaw/pitch/roll (deg) | `models/custom/headpose_cnn.pt` |
| Voice | `train_voice.py` | Google Speech Commands v2 (auto-downloaded) | log-mel spectrogram | keyword class | `models/custom/voice_kws.pt` |

## 1. Install the training/runtime deps

```
pip install torch torchvision torchaudio scikit-learn
```

The app runs **without** these — every `custom` provider falls back to its
classical counterpart until torch **and** the trained `.pt` weights are both
present. So you can train the models one at a time and the rest keep working.

## 2. Prove the pipeline first (no download needed)

Every script has a `--smoke` mode that trains on random data and writes a
(useless but valid) `.pt`, so you can confirm training → save → the app
picking up the model end-to-end before committing to the big downloads:

```
python scripts/train_blink.py    --smoke
python scripts/train_gaze.py     --smoke
python scripts/train_headpose.py --smoke
python scripts/train_voice.py    --smoke
```

## 3. Real training

```
# Blink — download + unzip mrlEyes_2018_01.zip, then:
python scripts/train_blink.py --data path/to/mrlEyes_2018_01

# Voice — Speech Commands auto-downloads (~2.3 GB first run):
python scripts/train_voice.py --epochs 8

# Head pose — download + unzip 300W-LP and AFLW2000:
python scripts/train_headpose.py --data path/to/300W_LP --eval path/to/AFLW2000

# Gaze — MPIIGaze "Data/Normalized":
python scripts/train_gaze.py --data path/to/MPIIGaze/Data/Normalized
```

Each writes its `.pt` (TorchScript) plus a `*.metrics.json` next to it. On a
CPU-only box start with a small `--limit`/`--epochs` to sanity-check speed.

## 3b. Hyperparameter tuning

Every trainer takes `--search N` to run an N-trial random search
(`scripts/tuning.py:sample_cfg` over optimizer, LR, weight decay, dropout,
conv/fc width, LR schedule, augmentation, batch size). Trial 0 is always the
original config, so the search is directly comparable to what shipped; every
trial is logged to `reports/tuning/<model>_trials.csv`, and selection is on the
**validation** split only.

```
# examples used in the study (small subsets for a CPU box):
python scripts/train_gaze.py     --data "MPIIGaze_DATASET/Data/Normalized" --search 10 --limit 10000 --epochs 5
python scripts/train_blink.py    --data "mrlEyes_2018_01_DATASET"          --search 20 --limit 20000 --epochs 8
python scripts/train_headpose.py --data "300W_LP_DATASET"                  --search 10 --limit 15000 --epochs 6
python scripts/train_voice.py                                              --search 10 --epochs 6
```

Then retrain the winning config to completion and read the held-out **test**
set once with `--config '<json>' --test`. Splits are **group-disjoint** by
subject / participant / face identity (`tuning.group_split`) — a random row
split leaks identity across train/test and scores memorisation, which is why
the original "published" numbers were inflated. Weight init is seeded
(`tuning.seed_all`) so a trial is reproducible from its CSV row.

The full tuning study — the leakage fix, three-numbers-per-model framing
(published → fair baseline → tuned), per-model results, the train/serve gap
that keeps gaze on its classical provider, and the compute caveats — is written
up in **`reports/tuning/FINAL_REPORT.md`**.

## 4. Use them

Select `custom` for any input from the dashboard provider switch to run your
trained model, and A/B it against the classical providers live. The current
`config.json` defaults are `custom` for **voice** (it measures best on the live
pipeline) and classical for the rest — `iris_polynomial` (gaze), `ear_standard`
(blink), `mp_geometry` (head pose) — reflecting what wins on real input today.
Change a default by setting that modality's `provider`/`engine` in
`config.json`. Any `custom` provider falls back to its classical counterpart
until `torch` **and** the trained `.pt` weights are both present.

Per-input knobs (in `config.json`):
- `gaze.custom.gain_x` / `gain_y` — angle→screen sensitivity (the calibration
  knob the physical screen distance needs; also use "set center" while
  looking at screen centre).
- `voice.custom.base_engine` — `vosk` / `whisper`, the ASR that transcribes
  words the keyword model doesn't cover.
- `app/headpose/custom_cnn.py: AXIS_SIGN` — flip an axis to `-1` if your
  trained head-pose model steers a gesture the wrong way (dataset sign
  convention).

## 5. Evaluation reports

```
# custom CNNs: full metric suite -> models/custom/model_report.json
python scripts/model_report.py

# pretrained/classical providers on the SAME datasets, scored by the SAME
# functions (solvepnp + landmark_heuristic on AFLW2000, vosk on Speech
# Commands) -> models/pretrained_report.json
python scripts/eval_pretrained.py     # run model_report.py first

# the full HTML evaluation report (model quality + runtime sessions)
python -m app.main --report

# every metric, every provider, in ONE plain sheet sectioned by modality
# (quality from the JSONs above; runtime re-aggregated from sessions/, and
# unlike the HTML report it includes the retired fusion_nn provider)
# -> reports/handsfree_metrics.xlsx
python scripts/export_metrics_xlsx.py [out.xlsx]
```

Both scripts share their metric code (`unified_metrics` / `reg_metrics` /
`facing_angular_deg` in `model_report.py`), so a custom model and a classical
provider in the same table are never scored by different formulas. The HTML
report and the xlsx export likewise share `report.enrich_custom_report` (which
merges measured inference time and the AUC/top-k scores from the per-model
metrics files into the aggregate report) plus the same labels and rating
thresholds, so the two can't disagree.

Two providers can't be dataset-scored and say so in the report rather than
showing a fake number: `mp_geometry` (head pose) only outputs pose relative to
a per-user neutral face cloud, and the classical blink detectors (EAR) need
full-face landmarks, which the MRL Eye dataset's cropped eye images don't have.

Self-checks: `python -m pytest scripts/test_metrics.py scripts/test_model_report.py`

## Notes / caveats

- **Gaze → screen**: the CNN predicts gaze *direction*; turning that into a
  screen point depends on your screen size/distance, so it needs the light
  affine calibration above. Absolute cursor accuracy is only as good as that
  calibration.
- **MPIIGaze `.mat` layout** varies between redistributions; if
  `train_gaze.py` finds no samples, adjust `_iter_eyes()` to your copy's
  field names.
- **Single dataset domain**: models trained on these datasets carry their
  domain (camera, lighting, demographics). Expect to fine-tune on your own
  captures if accuracy in your setup matters — state this as a limitation.
