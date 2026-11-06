# User Guide

## Starting

Double-click `run.bat` (or `python -m app.main`). The dashboard opens at
<http://127.0.0.1:8765>. Give it ~2 seconds to warm up; the header chips show
camera / face / FPS / CPU at a glance.

> **Camera black or 1 FPS?** Open the webcam privacy shutter / check
> Windows Settings → Privacy → Camera. The self-test
> (`run.bat --selftest`) verifies everything.

## Eye cursor

- Works immediately in **geometric** mode. For real precision, click
  **Calibrate gaze (9-point)** and follow the dots (~20 s). Your accuracy and
  precision in pixels are reported and shown in the dashboard.
- Look where you want the cursor. It's intentionally steady at fixation —
  the deadzone means tiny eye tremor moves nothing.
- **Set center**: look at the middle of the screen and click it (or say
  "set center") to re-center the geometric mapping.
- Grabbing the real mouse pauses gaze control for 2 s automatically.
- Sliders: **Smoothness** lower = calmer cursor, **Responsiveness** higher =
  snappier saccades, **Deadzone** = anti-tremor radius.

## Blink clicks

Blink normally — nothing happens. To click, **close both eyes and hold**:

| Hold | Action | Cue |
|---|---|---|
| ~0.45 s | left click | 1st beep |
| ~1 s | right click | 2nd beep |
| ~1.7 s | double click | 3rd beep |
| > 3 s | nothing (rest your eyes) | — |

Open your eyes when you hear the beep for the action you want. The progress
bar in the Blink card shows the same thing visually.

## Head gestures

Return to a comfortable straight pose, then make a **deliberate** turn/tilt
and hold it a beat:

- **Turn left / right** → previous / next app (Alt+Shift+Tab / Alt+Tab)
- **Tilt up / down** → next / previous browser tab (Ctrl+Tab / Ctrl+Shift+Tab)

You must return to neutral before the next gesture (no accidental repeats).
Swap the mapping (apps ↔ tabs) in the Head gestures card. Click **Set neutral
pose** (or say "set neutral") if gestures feel off-center.

## Voice assistant

Always listening (offline, Vosk). Examples:

- **Apps:** "open chrome", "open spotify", "launch calculator",
  "switch to notepad", "open youtube", "close this window"
- **Web:** "search for accessible computing", "new tab", "next tab",
  "go back", "refresh", "scroll down", "scroll up a lot", "zoom in"
- **Clicking:** "click", "double click", "right click"
- **Typing:** "type hello world", "press enter", "select all", "copy",
  "paste", "undo", "save"
- **Windows:** "minimize", "maximize", "show desktop", "snap left",
  "next app"
- **Media:** "volume up", "mute", "play", "pause", "next song"
- **System:** "take a screenshot", "lock the computer"
- **Dictation:** "start dictation" → everything you say is typed →
  "stop dictation"
- **Control:** "pause control", "resume control", "calibrate my eyes",
  "set neutral", "generate report", "help"

Engines: **custom** (trained keyword CNN, offline, current default — 94.5%
accuracy at ~10× Vosk's speed) · **vosk** (offline) · **whisper** (better
accuracy, install `faster-whisper`) · **webspeech** (browser-based, requires
the dashboard open in Chrome/Edge and internet). Switch in the Voice card. A
wake word can be set in `config.json` (`voice.wake_word`) if you don't want
always-on.

> Each modality also has a **`custom`** option in its card — a CNN trained from
> scratch on a public dataset. Voice and head pose use it well; gaze and blink
> default to their classical providers today (see
> `reports/tuning/FINAL_REPORT.md`). Custom providers need `torch` plus trained
> weights installed, otherwise they fall back to the classical provider
> automatically.

## Research workflow (model comparison)

1. Use the system a few minutes per model (switch models in the UI — every
   logged second is tagged with the active models).
2. Click **Generate report**: comparison tables + charts for FPS, latency
   (mean/p95), CPU, memory, jitter, accuracy/precision, per modality.
3. **New session** starts a fresh log file for a clean experiment block.
4. Raw data: `sessions/*.jsonl` (1 Hz samples + every event) — easy to load
   into pandas for the dissertation.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Cursor drifts | Recalibrate; or "set center"; ensure steady lighting |
| Clicks fire from normal blinks | Raise *Click hold ms* slider (up to 600) |
| Gestures too sensitive / dull | Adjust yaw/pitch threshold sliders |
| Voice misses commands | Speak naturally but clearly; try the whisper engine; check mic in Windows sound settings |
| Cursor won't move | Check Pause state (header button), and that a face is detected (chip) |
| Everything slow | Close other camera apps; check CPU chip |
