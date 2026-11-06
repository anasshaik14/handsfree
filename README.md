# HandsFree Control

**A hands-free control mechanism for accessible computing and mitigating
Carpal Tunnel Syndrome** — control the entire laptop with your eyes, blinks,
head and voice. Two provider families per modality run behind one interface:
**classical** (MediaPipe FaceLandmarker, Vosk/Whisper, geometry and signal
processing — no training, works out of the box) and **custom CNNs** trained
from scratch on public datasets (blink, gaze, head pose, voice), each
hyperparameter-tuned and evaluated under a leak-free protocol. You can A/B any
two live from the dashboard.

## The paper

**MAPLE: A leakage-audited and explainable framework for real-time multimodal
hands-free interaction** --- Mohammad Anas Shaik, Vaibhav Agarwal,
Gayathri Karthick, Sahar Ahmadzadeh. Department of Computer and Data Science,
York St John University, London.

The compiled manuscript is [`PAPER.pdf`](PAPER.pdf) (116 pages); the LaTeX
source is in [`paper1_work/`](paper1_work/), formatted with the Elsevier CAS
template for *Machine Learning with Applications*.

The paper's central claim is a methodological one, and it is what this
repository exists to support. Group-disjoint partitioning is the standard
remedy for identity leakage, and it repairs each training pipeline in
isolation --- but it leaves the **comparison between two pipelines**
contaminated whenever the compared models were developed on overlapping,
group-structured corpora and the evaluation sample spans at least one model's
partition boundary. The paper formalises that condition as *asymmetric
exposure*, shows the established leakage taxonomies have no category for it
because they are organised per pipeline, and introduces **GERA** (Group
Error-Ratio Audit): a detector that needs neither model's training data, only
the two prediction streams and a group label per sample.

On this four-modality system, asymmetric exposure inflated one modality's
apparent accuracy by 1.905 degrees, of which 22--56 % is training familiarity
rather than intrinsic difficulty. Correcting the protocol **reverses an
architectural decision the uncorrected protocol had supported** --- which is
the practical reason any of this matters.

| Modality | Result under the corrected protocol |
|---|---|
| Blink | 93.67 % |
| Gaze | 6.439 degrees |
| Head pose | 7.367 degrees |
| Voice | 94.53 % |

Four networks totalling 7.84 MB, 6.68 ms per frame on one CPU thread.

## Quick start

```bat
run.bat
```

That starts the perception pipeline and opens the dashboard at
<http://127.0.0.1:8765>. First-time setup (already done if `.venv` exists):

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m app.main --selftest   :: verify camera/models
run.bat
```

## The four input modalities

| Modality | What it does | Models (runtime-switchable) |
|---|---|---|
| **Eye cursor** | Cursor follows your gaze, professionally stabilized (One Euro filter → deadzone hysteresis → slew limiter) | `iris_geometric`, `iris_polynomial` (9-point calibration, current default), `pupil_classical` (GazeTracking-style CV), `custom` (trained CNN) |
| **Blink clicks** | Only *intentional* blinks click — hold ≥ 420 ms = left click, ≥ 1 s = right click, ≥ 1.7 s = double click. Natural blinks never fire. | `ear_standard` (current default), `ear_adaptive`, `ela` (eyelid angle), `custom` (trained CNN) |
| **Head gestures** | Turn head left/right = switch apps, tilt up/down = switch browser tabs (mapping swappable in the UI). Deliberate-only: dwell + return-to-neutral + cooldown. | `solvepnp`, `mp_geometry` (Kabsch, current default), `landmark_heuristic`, `custom` (trained CNN) |
| **Voice assistant** | Natural commands: *"open chrome"*, *"search for X"*, *"scroll down"*, *"type hello"*, *"start dictation"*, *"volume up"*, *"pause control"*… say **"help"** for more. | `custom` (trained keyword CNN, current default), `vosk` (offline), `whisper` (faster-whisper, optional), `webspeech` (in-browser) |

All four run **simultaneously** off a single FaceLandmarker inference per frame.

## Using it well

1. Sit comfortably, face the camera, decent lighting.
2. Click **Calibrate gaze (9-point)** in the dashboard and follow the dots —
   this switches you to the accurate polynomial model and reports your
   accuracy/precision in pixels.
3. Click **Set neutral pose** while looking straight at the screen.
4. Tune sliders live: *Smoothness* (lower = steadier), *Responsiveness*,
   *Deadzone* for the cursor; hold time for blinks; thresholds for gestures.

**Safety built in:** grabbing the physical mouse instantly pauses gaze
control for 2 s; say *"pause control"* / *"resume control"* any time; the
big **Pause control** button always works.

## Evaluation & research metrics

Every second the app logs FPS, per-stage latency (mean/p95), CPU, memory,
fixation jitter, calibration accuracy/precision and all events, tagged with
the active model of every modality, to `sessions/*.jsonl`.

- **Generate report** in the UI (or `run.bat --report`) builds
  `reports/<timestamp>/index.html` with comparison tables and charts per
  modality/model.
- To compare models: use the app for a few minutes with each model selected
  (switching creates naturally-tagged data), then generate the report.

## Custom models & hyperparameter tuning

Four CNNs are trained from scratch (not fine-tuned backbones) as an alternative
provider per modality — see [`scripts/README.md`](scripts/README.md) to train
them and [`reports/tuning/FINAL_REPORT.md`](reports/tuning/FINAL_REPORT.md) for
the full write-up. Headline points a reviewer should know:

- **Honest evaluation.** All four trainers originally used random row splits,
  which leak subject/participant/face identity across train and test (the
  datasets have thousands of near-duplicate frames per subject). Splits are now
  **group-disjoint** by identity (`scripts/tuning.py:group_split`), and each
  model reports three numbers — *published* (old leaky), *fair baseline*
  (original settings, honest split), *tuned* (searched settings, honest split)
  — so the leak fix and the tuning gain are never conflated.
- **Tuning results** (tuned vs fair baseline, honest test set):

  | Model | Result | Transfers to the live app? |
  |---|---|---|
  | **Voice** | 94.5% accuracy (+9.8pp) | ✅ matches Vosk's accuracy at ~10× the speed |
  | **Head pose** | 7.37° MAE on AFLW2000 (−11% rel.) | ✅ beats both classical providers |
  | **Blink** | 93.7% accuracy (+1.3pp) | same crop pipeline as the app |
  | **Gaze** | small lab gain only | ❌ on live webcam crops the classical iris providers still win — tuning in progress |

- **Why classical is still the default for gaze/blink/head pose:** the current
  `config.json` selects `iris_polynomial` / `ear_standard` / `mp_geometry` and
  `custom` only for voice, reflecting what measures best on the live pipeline
  today. Switch any modality to `custom` from the dashboard to compare.

## Tests

```bat
.venv\Scripts\python -m pytest tests -q
```

111 unit tests cover the stabilization chain, blink state machine, gesture
state machine, intent parser, calibration math, pose math, config and metrics.

## Layout

```
app/
  core/        types + provider interfaces (Strategy pattern)
  vision/      camera thread, shared FaceLandmarker, landmark constants
  gaze/        3 gaze providers, calibration, smoothing, cursor controller
  blink/       3 blink detectors + intentional-blink FSM
  headpose/    3 pose providers + deliberate-gesture FSM
  voice/       3 speech engines + intent parser + action executor
  system/      Windows input (ctypes SendInput), app launcher
  metrics/     collector, session logs, report generator
  server/      FastAPI + WebSocket + dashboard (static/)
  pipeline.py  orchestrator
scripts/       CNN training + hyperparameter search + evaluation (train_*.py,
               tuning.py, model_report.py, eval_pretrained.py)
models/        custom/ (trained .pt weights) + baseline/ (frozen pre-tuning)
reports/       tuning/ (search logs, trials CSVs, FINAL_REPORT.md), metrics xlsx
docs/          architecture & evaluation notes
```

See `docs/ARCHITECTURE.md` for design details and
`reports/tuning/FINAL_REPORT.md` for the tuning study.

## Datasets are not included here

The four corpora the paper uses are **not redistributed in this repository**.
Each carries licence terms that restrict redistribution, and MPIIGaze requires
registering with MPI directly. Obtain them from the original providers:

| Corpus | Used for | Grouping key |
|---|---|---|
| MRL Eye Dataset | blink | subject |
| MPIIGaze | gaze | participant |
| 300W-LP / AFLW2000 | head pose | face identity |
| Google Speech Commands v2 | voice | speaker |

The grouping key column is the point: every split in this work is
group-disjoint on that key, and the training scripts in `scripts/` expect the
corpora laid out as the paths at the top of each `train_*.py` describe.

The pre-trained third-party models (MediaPipe FaceLandmarker, Vosk,
faster-whisper) are likewise fetched at setup rather than committed.

## What is in here

The trained networks are the research artefact: `models/custom/` holds the
four tuned models the paper reports, `models/baseline/` the frozen
pre-tuning versions used as the internal control, and
`models/custom/_aug80k_20260803/` the augmentation experiment of Section 5.2.
Per-model `.metrics.json` files carry the measured figures.

`reports/figures/` holds the 63 figures of the paper and
`reports/paper/` the markdown the manuscript was written from.

## Citation

```bibtex
@unpublished{shaik2026maple,
  author = {Shaik, Mohammad Anas and Agarwal, Vaibhav and
            Karthick, Gayathri and Ahmadzadeh, Sahar},
  title  = {{MAPLE}: A leakage-audited and explainable framework for
            real-time multimodal hands-free interaction},
  note   = {Manuscript under preparation},
  year   = {2026},
}
```

## Licence

Code and trained models in this repository: see [`LICENSE`](LICENSE).
The datasets and the pre-trained third-party models remain under their own
licences and are not covered by it.
