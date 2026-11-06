# HandsFree — Complete User Guide

**Author:** Mohammed Anas Shaik · York St John University

HandsFree lets you control your Windows PC with your **eyes**, **blinks**, **head**, and **voice** — all at once, from one webcam and one microphone.

---

## Starting the app

```
cd D:\handsfree
.venv\Scripts\python.exe -m app.main
```

The dashboard opens at **http://127.0.0.1:8765** (use Chrome or Edge). Each of the four input panels has an ON/OFF switch — the app remembers your toggles between restarts.

**Safety first:** grabbing the physical mouse always pauses gaze control for 2 seconds (mouse override). Say **"pause control"** / **"resume control"**, or press the PAUSE CONTROL button, to stop/start cursor control entirely.

---

## 1 · Eye cursor (gaze tracking)

Your iris position steers the mouse cursor.

### The models (Tracking Model dropdown)

| Model | Calibration | When to use |
|---|---|---|
| `iris_geometric` | None — works instantly | Quick start, demos. Look at screen center for ~1 s so it auto-centers, or say "set center". |
| `iris_polynomial` ⭐ default | **9-point calibration required** | Daily use — by far the most accurate after calibration. Falls back to geometric until calibrated. |
| `pupil_classical` | None | Comparison baseline (classical computer vision, no neural iris). Least accurate — for evaluation only. |
| `custom` | Light affine (use "set center") | Trained CNN (MPIIGaze). Strong on lab data, but on live webcam crops it currently trails the iris models — for evaluation/comparison. Needs `torch` + trained weights, else falls back to geometric. |

### How to use it well
1. Sit at your normal distance, face evenly lit (avoid a bright window behind you).
2. Click **CALIBRATE GAZE · 9-PT**. Follow the dot **with your eyes only — keep your head completely still**.
3. If it says "calibration not good enough", that's the quality gate protecting you — just run it again.
4. Fine-tune in *calibration trim*: **Smoothness** (lower = steadier, slower), **Deadzone** (bigger = rock-still when staring), **Reach** (raise if corners are hard to hit).

**Important:** if you ever change the Pose Model (Head Gestures panel), recalibrate gaze — the calibration is tied to the pose model that was active when you calibrated.

---

## 2 · Blink clicks

Deliberate long blinks click the mouse. Normal blinking (~0.1–0.3 s) never clicks.

### How to click
Close both eyes and hold, then open. The hold time picks the action — beeps guide you while your eyes are closed:

| Hold for | Action | Sound while held | Sound when it fires |
|---|---|---|---|
| ~0.4 s | **Left click** | 700 Hz beep | single beep |
| ~1.0 s | **Right click** | 950 Hz beep | low-high double beep |
| ~1.7 s | **Double click** | 1200 Hz beep | two quick high beeps |
| ~3.0 s+ | **Cancelled** — rest your eyes, nothing fires | — | — |

The cursor freezes while your eyes are closed, so the click lands exactly where you were looking.

### The detectors (Detector dropdown)

| Model | How it works | When to use |
|---|---|---|
| `ear_standard` ⭐ default | Fixed eye-openness threshold | Robust, no learning period; the current default |
| `ear_adaptive` | Learns *your* eye shape as a rolling baseline | Handles glasses, lighting, camera angle once its baseline settles |
| `ela` | Eyelid angle instead of aspect ratio | Alternative if EAR is unreliable for your eye shape (e.g. narrow eyes) |
| `custom` | Trained CNN on a 32×32 eye crop (open/closed) | 93.7% accuracy; feeds the same click FSM. Needs `torch` + trained weights, else falls back to EAR |

Tuning: **Click hold ms** (raise if you accidentally click; lower for faster clicking), **Adaptive ratio** (lower = must close eyes more fully to register).

---

## 3 · Head gestures

Turn your head fully and hold to switch apps/tabs. Small natural movements are ignored by design: a gesture needs a **~30° turn held for ~0.65 s**, then return to center; 1.2 s cooldown between gestures.

### The pose models (Pose Model dropdown)

| Model | How it works | When to use |
|---|---|---|
| `solvepnp` | Generic 3D face model geometry | Classic baseline — tends to exaggerate angles |
| `mp_geometry` ⭐ default | Aligns against *your own* face captured at startup | Personalized to you; the current default for gesture control |
| `landmark_heuristic` | Simple landmark ratios | Fallback/comparison only |
| `custom` | Trained CNN regressing yaw/pitch/roll from a face crop | Most accurate on the AFLW2000 benchmark (7.37° MAE, beats the classical providers). Needs `torch` + trained weights, else falls back to geometry |

### How to use
- On startup it captures your neutral pose automatically (~1 s of sitting normally). If your camera is far off-center, click **SET NEUTRAL** once while facing your screen.
- The **Mapping** dropdown chooses what gestures do: `← → apps, ↑ ↓ tabs` or `← → tabs, ↑ ↓ apps`.
- The neutral reference slowly follows your posture, so slouching won't cause false triggers.
- Tuning: raise **Yaw/Pitch threshold** or **Dwell ms** if it still triggers too easily.

---

## 4 · Voice assistant

Speak commands naturally. Replies are **spoken back through Windows speech** (works with every engine — toggle with "Speak replies").

### The engines (Engine dropdown)

| Engine | Internet? | Accuracy | Latency | Notes |
|---|---|---|---|---|
| `custom` ⭐ default | No — fully offline | 94.5% on the command vocab | Fast (~10× Vosk) | Trained keyword CNN; a `base_engine` (vosk/whisper) handles words outside the command set. Needs `torch` + trained weights, else falls back to that base engine |
| `vosk` | No — fully offline | Modest | Instant (streaming) | Weakest with accented English |
| `whisper` | No (after first model download) | Good | ~1–2 s after you stop speaking | Uses faster-whisper `base.en`; change `whisper_model` to `small.en` in config.json for better accuracy |
| `webspeech` | **Yes** | Best (Google, set to Indian English) | Fast | Runs in the browser — the **dashboard tab must stay open** with mic permission allowed |

**webspeech checklist:** Chrome or Edge · click "Allow" on the microphone prompt · keep the tab open (minimized is fine).

### Command reference

- **Wake phrase:** "**HELLO ONE**" — always works, opens the dashboard and greets you
- **Clicking:** "click" · "double click" · "right click" · "middle click"
- **Scrolling:** "scroll down / up" · "page down / up" · "go to top / bottom"
- **Browser:** "go back / forward" · "refresh" · "new tab" · "close tab" · "reopen tab" · "next / previous tab" · "zoom in / out"
- **Windows/apps:** "switch to chrome" · "open notepad" · "next / previous app" · "minimize / maximize" · "close window" · "snap left / right" · "show desktop"
- **Web:** "search for cats" · "open the website github.com"
- **Typing:** "type hello world" · "press enter" · "select all" · "copy / paste / cut" · "undo / redo" · "save" · "delete"
- **Dictation:** "start dictation" → everything you say is typed → "stop dictation"
- **Media:** "volume up / down" · "mute" · "play / pause" · "next song"
- **System:** "take a screenshot" · "lock screen"
- **App control:** "pause control" · "resume control" · "calibrate" · "set center" · "set neutral" · "generate report" · "help"

Slightly garbled recognition still works ("scroll town" → scroll down). The app never transcribes its own spoken replies.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Cursor doesn't move | Check the Eye Cursor toggle is ON (toggles persist!), face chip is green, and control isn't paused |
| Cursor inaccurate / drifts | Recalibrate (head still, eyes only). Recalibrate after changing Pose Model |
| Cursor won't reach corners | Raise the **Reach** slider |
| No voice replies | Check "Speak replies" is ticked and system volume up |
| webspeech not hearing | Dashboard tab must be open in Chrome/Edge with mic allowed — reload the page if you denied it once |
| whisper slow | Expected — it transcribes after you finish speaking. Use webspeech for instant response |
| Head gestures fire too easily / never | Raise / lower Yaw threshold and Dwell ms |
| Camera black / "recovering" | Close other apps using the webcam; the app auto-recovers wedged cameras |
| Everything seems dead | Look at the four ON/OFF toggles — they save to disk, so an OFF stays OFF until you re-enable it |
