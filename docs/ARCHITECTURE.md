# Architecture

## Design principles

- **Clean architecture / SOLID.** The pipeline depends only on abstract
  interfaces (`app/core/interfaces.py`); every concrete model is a Strategy
  behind `GazeProvider`, `BlinkProvider`, `HeadPoseProvider` or `VoiceEngine`.
  Swapping a model at runtime replaces one object under a lock — nothing else
  changes (Open/Closed, Liskov).
- **One inference, many consumers.** MediaPipe FaceLandmarker (478 landmarks,
  iris included) runs once per frame; gaze, blink and head pose all consume
  the same `FaceResult`. This is why all modalities can run simultaneously in
  real time on CPU (~5 ms/frame measured).
- **Two provider families behind one interface.** Every modality ships a
  *classical* provider (pretrained perception + geometry + signal processing +
  rule-based NLU — no training, always available) and a *custom CNN* trained
  from scratch on a public dataset. Both implement the same Strategy interface,
  so they are interchangeable at runtime and scored by the same metric code.
  The custom CNNs are optional: without `torch` or the `.pt` weights, each
  `custom` provider transparently falls back to its classical counterpart.

## Dataflow

```
Camera thread (latest-frame slot, never blocks the pipeline)
   └─> Pipeline thread
         ├─ FaceLandmarker  ──────────────── FaceResult (478 x 3 px coords)
         ├─ HeadPoseProvider ─ HeadPose ──> GestureFSM ──> Alt+Tab / Ctrl+Tab
         ├─ BlinkProvider ── BlinkFrame ──> BlinkFSM ────> SendInput clicks
         ├─ GazeProvider ─── GazeSample ──> Stabilizer ──> SetCursorPos
         └─ MetricsCollector (per-stage timers, 1 Hz session log)
Audio thread (Vosk/Whisper) ─ transcript ─> IntentParser ─> ActionExecutor
Browser (Web Speech API) ────── WebSocket ┘
FastAPI/WS server ── dashboard UI, runtime switching, config, MJPEG preview
```

## Eye cursor stability (the hard requirement)

Raw webcam gaze is inherently noisy (~5–10 px landmark jitter). The cursor
chain removes it without adding sluggishness:

1. **One Euro filter** (Casiez et al. 2012) — adaptive low-pass: cutoff
   `f_c = mincutoff + beta * |velocity|`. At fixation velocity≈0 → heavy
   smoothing (no jitter); during a saccade velocity is huge → cutoff rises →
   near-zero lag. `beta` is calibrated for *pixel* units (0.004); a unit
   test guards this tuning.
2. **Deadzone hysteresis** (9 px) — output moves only when the filtered
   target escapes a radius, and then *trails it by the radius*, so motion is
   continuous with zero popping. Kills 100% of residual tremor at rest.
3. **Slew limiter** (2600 px/s) — a single-frame estimation glitch can move
   the cursor at most ~87 px in one frame instead of teleporting.
4. **Freeze windows** — the cursor holds position while the eyes are closing
   /closed (blink landmarks corrupt gaze), during head gestures
   (`guard_cursor`), and after the user grabs the physical mouse
   (safety override).

## Gaze models

| Model | Idea | Trade-off |
|---|---|---|
| `iris_geometric` | iris position inside the eye box, per-user center + gains, optional head-pose assist | instant, no calibration; medium accuracy |
| `iris_polynomial` | ridge-regressed 2nd-order polynomial `(hx, hy, yaw, pitch) → screen`, fitted from a 9-point calibration | best accuracy; tolerant to small head motion because pose is a feature |
| `pupil_classical` | GazeTracking-style classical CV: darkest-blob pupil segmentation (blur → erode → percentile threshold → contour centroid) in the eye ROI | no neural iris; the classical baseline for comparison |
| `custom` | trained CNN regressing gaze pitch/yaw from a 36×60 eye patch (MPIIGaze), + light affine calibration to screen | strong on the training domain; on live FaceMesh crops it does **not** yet beat the iris providers (train/serve gap) |

Note on GazeTracking (OpenCV+Dlib): the original library uses Dlib's 68-point
detector *only to locate the eye region*, then finds the pupil classically.
Dlib has no reliable prebuilt wheels on Windows/Python 3.11, so
`pupil_classical` reproduces the classical pupil-segmentation method on eye
ROIs from the already-running FaceLandmarker — the *comparison-relevant* part
of the method is faithful.

Calibration metrics follow gaze-research convention: **accuracy** = mean
Euclidean error to targets (px); **precision** = RMS dispersion of samples
around their own centroid (px).

## Blink models

- `ear_standard` — Soukupova & Cech Eye Aspect Ratio with a fixed threshold
  (the literature baseline).
- `ear_adaptive` — EAR with a rolling per-eye median baseline of *open*
  samples; threshold = 0.70 × baseline. Adapts to eye shape, glasses,
  lighting.
- `ela` — Eye Lid Angle: the angle at the outer eye corner between upper- and
  lower-lid chords, with the same adaptive baseline scheme.
- `custom` — trained CNN classifying a 32×32 eye crop as open/closed (MRL Eye
  Dataset). 93.7% subject-disjoint accuracy; feeds the same intentional-blink
  FSM as the classical detectors.

**Intentional-blink FSM:** natural blinks last 100–300 ms; only closures held
≥ `hold_click_ms` (420 ms) act, on reopen: ≥420 ms → left click, ≥1 s →
right click, ≥1.7 s → double click, ≥3 s → cancelled (eye rest). Refractory
period prevents double fires; audible pips announce which action will fire.

## Head-pose models

All providers share one sign convention derived from the face-forward vector
(no Euler-order ambiguity): yaw>0 = turned to the user's left, pitch>0 = up.
Neutral is auto-captured from the first second and re-settable.

- `solvepnp` — classic 6-point PnP against a generic 3D face model.
- `mp_geometry` — Kabsch/SVD rigid alignment of a 15-landmark rigid subset
  against the *user's own* neutral cloud (personalized; no generic-model
  anthropometric error).
- `landmark_heuristic` — pure 2D ratios (nose offset between cheeks, nose
  height between eyes and chin, eye-line angle). The cheapest baseline.
- `custom` — trained CNN regressing yaw/pitch/roll from a 64×64 face crop
  (300W-LP, tested on AFLW2000). 7.37° MAE, the best of any head-pose provider
  and the cleanest tuning win of the four models.

**Gesture FSM:** threshold + dwell (260 ms) + fire-once + return-to-neutral
hysteresis + cooldown (750 ms); dominant-axis arbitration prevents diagonal
double-fires. Quick glances shorter than the dwell can never trigger.

## Voice

- `vosk` — offline streaming Kaldi, default.
- `whisper` — faster-whisper (CTranslate2, no PyTorch) with energy-VAD
  utterance segmentation. Optional install.
- `webspeech` — browser-side Web Speech API; transcripts arrive over the
  WebSocket (works when the dashboard is open in Chrome/Edge).
- `custom` — trained keyword-spotting CNN over log-mel spectrograms (Google
  Speech Commands v2, 6-word command vocab), with a `base_engine` (vosk/whisper)
  transcribing anything outside the keyword set. 94.5% accuracy at ~10× Vosk's
  speed — the current default voice engine.

Transcripts feed a deterministic regex-grammar intent parser (~60 intents
with slots) and a single `ActionExecutor` that performs everything through
`SendInput`/shell. Dictation mode types everything until "stop dictation".
App launching resolves: alias table → URL protocols → PATH → fuzzy Start-Menu
shortcut search.

## Custom CNNs — training, tuning & honest evaluation

The four `custom` providers are trained by `scripts/train_*.py` (each a small
from-scratch CNN saved as self-contained TorchScript, loaded by
`app/ml/runtime.py:load_scripted`). Shared training plumbing lives in
`scripts/tuning.py`. The methodology matters more than the architectures:

- **Leak-free splits.** Rows in these datasets are not independent (MRL Eye has
  ~2,300 frames/subject, MPIIGaze thousands/participant, 300W-LP a median of 34
  synthesized views per face). A random split memorises identity. All four use
  `group_split` — 3-way disjoint by subject / participant / face identity — so
  val/test measure a *new* person, matching real use.
- **Three numbers per model.** *Published* (old leaky protocol, frozen in
  `models/baseline/`), *fair baseline* (original hyperparameters, honest split),
  *tuned* (searched hyperparameters, honest split). The published→fair gap is
  the leak; the fair→tuned gap is the tuning. They are reported separately and
  never merged.
- **Random hyperparameter search** (`sample_cfg`): optimizer, LR, weight decay,
  dropout, conv/fc width, LR schedule, augmentation, batch size. Trial 0 is
  always the original config so the search is directly comparable; every trial
  is appended to `reports/tuning/<model>_trials.csv`. Weight init is seeded
  (`seed_all`) for reproducibility. Hyperparameters are selected on validation;
  the test set is read exactly once, at the end.
- **Train/serve honesty gate.** `scripts/eval_pretrained.py` re-scores gaze and
  the other modalities on the *live app pipeline* (FaceMesh crops from raw
  frames), not the pre-normalised training patches. This is where gaze's lab
  gain disappears — the tuned gaze CNN (8.24° angular) trails the classical
  iris providers (~7.6°) on real webcam input, so gaze remains classical by
  default. Head pose and voice transfer cleanly and win.

Full study, per-model tables and caveats: `reports/tuning/FINAL_REPORT.md`.

## Metrics & evaluation

`MetricsCollector` timestamps every stage each frame and logs a 1 Hz summary
(JSONL) stamped with the active provider of each modality, plus every event
(blink, gesture, voice, calibration). `app/metrics/report.py` aggregates all
sessions into per-model comparison tables (FPS, stage latency mean/p95,
pipeline latency, CPU, memory, fixation jitter RMS, calibration
accuracy/precision, counts) and renders charts + a self-contained HTML report.

## Windows integration

`app/system/win_input.py` uses raw `ctypes` `SendInput`/`SetCursorPos` with
per-monitor DPI awareness — no third-party input libraries, lowest available
latency. Window management via `EnumWindows`/`SetForegroundWindow`.

## Security posture

The server binds to `127.0.0.1` only. Nothing leaves the machine (Vosk and
MediaPipe are fully offline; Web Speech uses the browser's own service only
if that engine is selected).
