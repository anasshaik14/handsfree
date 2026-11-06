"""Evaluate the PRETRAINED / classical providers on the same public datasets
as the custom CNNs, so the report can show honest accuracy/precision/recall
for every model — not just the trained ones.

  head pose : solvepnp + landmark_heuristic on AFLW2000 (MediaPipe FaceMesh
              landmarks -> provider -> yaw/pitch/roll vs .mat ground truth).
              mp_geometry is skipped: it only outputs pose RELATIVE to a
              per-user neutral cloud, so it has no absolute pose to score.
  voice     : vosk (small en-US) on the Speech Commands v2 validation split,
              limited-vocabulary grammar over the six command words — the
              same keyword-spotting task the custom CNN solves.

Classical pose providers are always used neutral-relative in the app, so a
per-axis constant offset (the "neutral capture") is removed before scoring;
axis sign is aligned to the dataset convention once, globally.

Writes models/pretrained_report.json (rendered by app/metrics/report.py).

Usage:  python scripts/eval_pretrained.py [--limit-voice 150] [--aflw AFLW2000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# the same metric functions the custom CNNs are scored with, so every column
# in the report is computed by identical code
from model_report import (CLS_ORDER, facing_angular_deg,  # noqa: E402
                          reg_metrics, unified_metrics)

KWS_WORDS = ["down", "go", "no", "stop", "up", "yes"]
REJECT = "unrecognized"
GAZE_ORDER = ("iris_geometric", "iris_polynomial", "pupil_classical")


def _gaze_angular(true_py, pred_py):
    """Angle (deg) between 3D gaze vectors. Inputs are (N,2) [pitch, yaw] in
    degrees; reuses the head-pose helper, which reads [yaw, pitch, roll] and
    ignores roll — same vector formula, one definition."""
    def ypr(a):
        return np.stack([a[:, 1], a[:, 0], np.zeros(len(a))], 1)
    return facing_angular_deg(ypr(true_py), ypr(pred_py))


# ------------------------------------------------------------------ head pose
def _face_landmarker(confidence: float = 0.5):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=confidence,
        min_face_presence_confidence=confidence)
    return mp, vision.FaceLandmarker.create_from_options(options)


def eval_headpose(aflw_root: Path, max_angle: float = 99.0) -> dict:
    import cv2
    from scipy.io import loadmat

    from app.core.types import FaceResult
    from app.headpose.providers import HeuristicHeadPose, SolvePnPHeadPose

    mp, lm = _face_landmarker()
    providers = {"solvepnp": SolvePnPHeadPose(), "landmark_heuristic": HeuristicHeadPose()}
    gt, preds = [], {n: [] for n in providers}
    n_total = n_detected = 0

    for img_path in sorted(aflw_root.glob("*.jpg")):
        mat_path = img_path.with_suffix(".mat")
        if not mat_path.exists():
            continue
        pose = loadmat(str(mat_path), variable_names=["Pose_Para"])["Pose_Para"][0][:3]
        pitch, yaw, roll = np.degrees(pose)
        if max(abs(yaw), abs(pitch), abs(roll)) > max_angle:
            continue
        n_total += 1
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            continue
        pts = np.array([(p.x * w, p.y * h, p.z * w) for p in res.face_landmarks[0]])
        face = FaceResult(t=0.0, landmarks=pts, width=w, height=h)
        row = {}
        for name, prov in providers.items():
            if prov.process(face) is None:
                break
            row[name] = prov._last          # absolute (yaw, pitch, roll)
        if len(row) != len(providers):
            continue
        n_detected += 1
        gt.append((yaw, pitch, roll))       # trainer convention: [yaw, pitch, roll]
        for name, angles in row.items():
            preds[name].append(angles)

    lm.close()
    true = np.asarray(gt, np.float64)
    out_models = {}
    for name in providers:
        pred = np.asarray(preds[name], np.float64)
        # solvePnP's mirrored solution: forward lands on -z here, so yaw sits
        # near ±180°. Map atan2(x,-z) -> atan2(x,z) once for the whole set.
        if np.median(np.abs(pred[:, 0])) > 90:
            pred[:, 0] = np.sign(pred[:, 0]) * 180.0 - pred[:, 0]
        # one global sign alignment per axis (convention fix, not a fit) ...
        signs = [1.0 if np.corrcoef(true[:, i], pred[:, i])[0, 1] >= 0 else -1.0
                 for i in range(3)]
        pred = pred * np.asarray(signs)
        # ... then the neutral capture: remove each axis's constant offset
        pred = pred - np.median(pred - true, axis=0)
        m = reg_metrics(true, pred, ["yaw", "pitch", "roll"])
        m["angular_mean_deg"] = round(float(facing_angular_deg(true, pred).mean()), 4)
        out_models[name] = {
            "title": name, "dataset": "AFLW2000", "n_samples": len(true),
            "metrics": m,
        }
        print(f"  {name:20s} MAE {m['mae_deg']:.2f}°  "
              f"angular {m['angular_mean_deg']:.2f}°  "
              f"PCK@15° {m['acc_within_15deg'] * 100:.1f}%")
    print(f"  faces detected: {n_detected}/{n_total}")

    # comparison column: the custom CNN, already evaluated on the same set
    mr = json.loads((ROOT / "models" / "custom" / "model_report.json").read_text())
    hp = mr["reg"]["models"].get("headpose")
    models = {}
    if hp:
        models["custom"] = {"title": "custom (our CNN)", "dataset": hp["dataset"],
                            "n_samples": hp["n_samples"], "metrics": hp["metrics"]}
    models.update(out_models)
    return {
        "title": "Head pose — all providers on AFLW2000",
        "note": (f"FaceMesh detected {n_detected}/{n_total} faces; classical "
                 "providers are scored on those (the CNN needs no landmarks). "
                 "A per-axis constant offset was removed for the classical "
                 "providers, mirroring the app's neutral-pose capture, and "
                 "axis conventions were aligned to the dataset's once, "
                 "globally (no per-sample fitting)."),
        "order": ["acc_within_5deg", "acc_within_10deg", "acc_within_15deg",
                  "mae_deg", "rmse_deg", "median_ae_deg", "p90_ae_deg",
                  "angular_mean_deg", "r2_best", "r2_worst",
                  "corr_best", "corr_worst"],
        "models": models,
    }


# ---------------------------------------------------------------------- gaze
def _mpii_truth(line: str):
    """(gaze_pitch_deg, gaze_yaw_deg) from one Data/Original annotation line.

    Fields (1-indexed, per MPIIGaze ReadMe): 27~29 = 3D gaze target in camera
    coords, 36~38 / 39~41 = 3D right / left eye centre. Gaze runs from the eye
    midpoint to the target; the dataset's own angle convention is
    pitch = asin(-y), yaw = atan2(-x, -z), which is the same one already used
    for the custom CNN in model_report.gaze_eval.
    """
    v = np.fromstring(line, sep=" ")
    if v.size != 41:
        return None
    eye = (v[35:38] + v[38:41]) / 2.0
    d = v[26:29] - eye
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    d = d / n
    return (float(np.degrees(np.arcsin(-d[1]))),
            float(np.degrees(np.arctan2(-d[0], -d[2]))))


def _fit_apply(feats_cal, tgt_cal, feats_test):
    """Least-squares fit on the calibration split, applied to the held-out
    split. `feats_*` are (N, k) design matrices (a bias column included by the
    caller)."""
    coef, *_ = np.linalg.lstsq(feats_cal, tgt_cal, rcond=None)
    return feats_test @ coef


def eval_gaze(mpii_root: Path, per_participant: int) -> dict:
    """Every gaze provider on MPIIGaze's own evaluation subset, through the
    app's real front end (FaceMesh on the original image)."""
    import cv2

    from app.core.types import FaceResult
    from app.gaze.calibration import PolynomialModel, feature_vector
    from app.gaze.custom_cnn import CustomGazeCNN
    from app.gaze.iris_geometric import IrisGeometricGaze
    from app.gaze.iris_polynomial import IrisPolynomialGaze
    from app.gaze.pupil_classical import PupilClassicalGaze
    from app.headpose.providers import SolvePnPHeadPose

    cfg = lambda k, d=None: d          # noqa: E731 - provider defaults
    # MPIIGaze is real laptop-webcam capture, much of it in near-dark rooms
    # (mean pixel ~2/255, a screen-lit face on black). At the 0.5 default
    # FaceMesh finds only ~58% of the faces. Dropping to 0.15 finds ~97% and
    # was verified NOT to be scraping the barrel: measured against MPIIGaze's
    # own manually annotated eye corners, the recovered detections land within
    # 3.6 px median / 6.4 px p90 on a 1280x720 image, versus 3.3 px / 5.0 px at
    # the default. The extra samples are good data, so keeping them is what
    # avoids silently scoring only the brightest third of the set.
    mp, lm = _face_landmarker(confidence=0.15)
    classical = {"iris_geometric": IrisGeometricGaze(cfg),
                 "pupil_classical": PupilClassicalGaze(cfg),
                 "iris_polynomial": IrisPolynomialGaze(cfg)}
    custom = CustomGazeCNN(cfg)
    pose_p = SolvePnPHeadPose()

    orig = mpii_root / "Data" / "Original"
    subset = mpii_root / "Evaluation Subset" / "annotation for face image"
    rows: list[dict] = []             # one per usable image
    n_total = n_detected = 0

    for plist in sorted(subset.glob("p*.txt")):
        pid = plist.stem
        want = [l.split()[0] for l in plist.read_text().split("\n") if l.strip()]
        seen, picks = set(), []
        for rel in want:              # de-dup, keep dataset order
            if rel not in seen:
                seen.add(rel)
                picks.append(rel)
        if per_participant and len(picks) > per_participant:
            step = len(picks) / per_participant
            picks = [picks[int(i * step)] for i in range(per_participant)]

        ann_cache: dict[str, list[str]] = {}
        for rel in picks:
            day, fname = rel.split("/")
            if day not in ann_cache:
                p = orig / pid / day / "annotation.txt"
                ann_cache[day] = p.read_text().splitlines() if p.exists() else []
            idx = int(Path(fname).stem) - 1        # 0001.jpg -> line 0
            lines = ann_cache[day]
            if not (0 <= idx < len(lines)):
                continue
            truth = _mpii_truth(lines[idx])
            if truth is None:
                continue
            n_total += 1
            img = cv2.imread(str(orig / pid / day / fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(
                                         cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            if not res.face_landmarks:
                continue
            pts = np.array([(q.x * w, q.y * h, q.z * w) for q in res.face_landmarks[0]])
            face = FaceResult(t=0.0, landmarks=pts, width=w, height=h)
            face.frame = img

            if pose_p.process(face) is None:
                continue
            hyaw, hpitch, _ = pose_p._last                  # absolute head pose
            if abs(hyaw) > 90:      # solvePnP's mirrored branch (see eval_headpose)
                hyaw = np.sign(hyaw) * 180.0 - hyaw

            row = {"pid": pid, "pitch": truth[0], "yaw": truth[1],
                   "hyaw": hyaw, "hpitch": hpitch}
            ok = True
            for name, prov in classical.items():
                prov.reset()        # independent stills: no temporal smoothing
                s = prov.process(face, None, frame=img)
                if s is None:
                    ok = False
                    break
                row[name] = (s.raw_nx, s.raw_ny)           # the provider's feature
            if not ok:
                continue
            ang = custom._predict_angles(face)             # (pitch, yaw) rad
            if ang is None:
                continue
            row["custom"] = (float(np.degrees(ang[0])), float(np.degrees(ang[1])))
            n_detected += 1
            rows.append(row)
        print(f"    {pid}: {len([r for r in rows if r['pid'] == pid])} usable",
              flush=True)
    lm.close()

    # ---- per-participant calibration, evaluated on a held-out split --------
    truth_all, preds = [], {k: [] for k in list(classical) + ["custom"]}
    for pid in sorted({r["pid"] for r in rows}):
        pr = [r for r in rows if r["pid"] == pid]
        if len(pr) < 20:
            continue
        cal = [r for i, r in enumerate(pr) if i % 2 == 0]
        test = [r for i, r in enumerate(pr) if i % 2 == 1]
        t_test = np.array([(r["pitch"], r["yaw"]) for r in test])
        t_cal = np.array([(r["pitch"], r["yaw"]) for r in cal])
        truth_all.append(t_test)

        for name in list(classical) + ["custom"]:
            if name == "iris_polynomial":
                # its real model: the app's 8-term feature vector (iris + head
                # pose + quadratics), fitted exactly as run_calibration does
                def design(rs):
                    return np.stack([feature_vector(r[name][0], r[name][1],
                                                    r["hyaw"], r["hpitch"])
                                     for r in rs])
            else:
                # Every other provider emits a 2D signal that the app turns into
                # a gaze estimate with a per-user affine map: gain * signal +
                # pose_assist * head angle + centre. That is literally what
                # GeometricMapper.map does for iris_geometric / pupil_classical,
                # and what gaze.custom.gain_{x,y} + set_center do to the CNN's
                # angles -- so the CNN is fitted the same way rather than being
                # read as absolute. It has to be: the CNN is trained on MPIIGaze's
                # NORMALISED patches, whose labels live in a warped frame that
                # cancels head rotation, so its raw output is not in the camera
                # frame this ground truth uses. Scoring it unfitted would measure
                # that frame mismatch, not the model.
                def design(rs, _n=name):
                    return np.stack([[r[_n][0], r[_n][1], r["hyaw"], r["hpitch"], 1.0]
                                     for r in rs])
            p = _fit_apply(design(cal), t_cal, design(test))
            preds[name].append(p)

    true = np.concatenate(truth_all)
    models = {}
    for name in ("custom",) + GAZE_ORDER:
        pred = np.concatenate(preds[name])
        # same angular error + angular-based PCK as the custom gaze table
        ang = _gaze_angular(true, pred)
        m = reg_metrics(true, pred, ["pitch", "yaw"], angular=ang)
        title = "custom (our CNN)" if name == "custom" else name
        models[name] = {"title": title, "dataset": "MPIIGaze (Original)",
                        "n_samples": len(true), "metrics": m}
        print(f"  {title:20s} angular {m['angular_mean_deg']:.2f}°  "
              f"MAE {m['mae_deg']:.2f}°  PCK@5° {m['acc_within_5deg'] * 100:.1f}%")
    print(f"  faces detected: {n_detected}/{n_total}")

    return {
        "title": "Gaze — all providers on MPIIGaze (Original)",
        "note": (f"FaceMesh detected {n_detected}/{n_total} faces on MPIIGaze's own "
                 "evaluation subset (much of it captured in near-dark rooms, so "
                 "detection confidence is lowered to 0.15; the recovered faces "
                 "were checked against MPIIGaze's manually annotated eye corners "
                 "and land within 3.6 px median on a 1280x720 image, so they are "
                 "real detections rather than noise). Every provider is scored on "
                 "those same images through the app's real front end. The three classical "
                 "providers output a screen-space signal rather than an angle, so "
                 "each gets the per-user calibration the app itself performs "
                 "(gain + head-pose assist + centre for the geometric/pupil "
                 "mappers, the full 8-term polynomial for iris_polynomial), "
                 "fitted per participant on half their samples and scored on the "
                 "held-out half. The custom CNN is fitted the same way, because "
                 "the app does the same thing to it (gaze.custom.gain_x/gain_y "
                 "plus set_center) and because its training labels live in "
                 "MPIIGaze's normalised frame, not this camera frame. "
                 "IMPORTANT: this table is a harness for comparing providers "
                 "against each other on equal terms, and its custom column is NOT "
                 "comparable to the 93.5% / 2.43° in the regressor table above — "
                 "that number scores the CNN on MPIIGaze's pre-normalised eye "
                 "patches (its training domain, and the standard protocol the "
                 "literature reports), whereas this one crops live from the "
                 "original image via FaceMesh exactly as the running app does. "
                 "The gap between the two is a genuine train/serve difference, "
                 "not a scoring error."),
        "order": ["acc_within_5deg", "acc_within_10deg", "acc_within_15deg",
                  "mae_deg", "rmse_deg", "median_ae_deg", "p90_ae_deg",
                  "angular_mean_deg", "r2_best", "r2_worst",
                  "corr_best", "corr_worst"],
        "models": models,
    }


# --------------------------------------------------------------------- voice
def _time_custom_kws(base: Path, per_word: dict) -> float:
    """ms/clip for the custom KWS engine's full runtime path (log-mel +
    forward + confidence gate) on the same clips vosk is timed on, so the two
    speed numbers are measured the same way on the same machine."""
    from scipy.io import wavfile

    from app.voice.custom_kws import CustomVoiceKWS
    eng = CustomVoiceKWS(base_engine="vosk")
    if not eng.available:
        return 0.0
    clips = [wavfile.read(base / rel)[1].astype(np.float32) / 32768.0
             for rels in per_word.values() for rel in rels]
    for c in clips[:10]:
        eng.classify(c)                       # warm up lazy melspec + jit
    t0 = time.perf_counter()
    for c in clips:
        eng.classify(c)
    return round((time.perf_counter() - t0) / max(len(clips), 1) * 1000, 1)


def eval_voice(limit_per_word: int) -> dict:
    import vosk
    from scipy.io import wavfile
    from sklearn import metrics as skm

    base = ROOT / "data" / "speech_commands" / "SpeechCommands" / "speech_commands_v0.02"
    val = [l for l in (base / "validation_list.txt").read_text().split()
           if l.split("/")[0] in KWS_WORDS]
    per_word: dict[str, list[str]] = {w: [] for w in KWS_WORDS}
    for rel in val:
        w = rel.split("/")[0]
        if len(per_word[w]) < limit_per_word:
            per_word[w].append(rel)

    vosk.SetLogLevel(-1)
    model = vosk.Model(str(ROOT / "models" / "vosk-model-small-en-us-0.15"))
    grammar = json.dumps(KWS_WORDS + ["[unk]"])
    y_true, y_pred = [], []
    t0 = time.perf_counter()
    for word, rels in per_word.items():
        for rel in rels:
            sr, audio = wavfile.read(base / rel)
            rec = vosk.KaldiRecognizer(model, sr, grammar)
            rec.AcceptWaveform(audio.tobytes())
            text = json.loads(rec.FinalResult()).get("text", "").strip()
            y_true.append(KWS_WORDS.index(word))
            # index 6 = declined to answer (empty / out-of-grammar transcript)
            y_pred.append(KWS_WORDS.index(text) if text in KWS_WORDS else 6)
    n = len(y_true)
    dt = (time.perf_counter() - t0) / max(n, 1) * 1000
    cm = skm.confusion_matrix(y_true, y_pred, labels=list(range(7)))
    # score the 6 real words; the reject column is never a ground-truth class
    m = unified_metrics(cm, score_labels=list(range(6)))
    m["inference_ms"] = round(dt, 1)
    rejects = int(cm[:, 6].sum())
    print(f"  vosk               accuracy {m['accuracy'] * 100:.1f}%  "
          f"F1 {m['f1_macro']:.3f}  ({n} clips, {rejects} rejected, {dt:.0f} ms/clip)")

    # comparison column: the custom KWS CNN on the same validation split, its
    # 18 metrics recomputed from its saved confusion matrix by the same function
    vm = json.loads((ROOT / "models" / "custom" / "voice_kws.metrics.json").read_text())
    custom = unified_metrics(vm["confusion_matrix"])
    custom["inference_ms"] = _time_custom_kws(base, per_word)
    print(f"  custom (our CNN)   accuracy {custom['accuracy'] * 100:.1f}%  "
          f"F1 {custom['f1_macro']:.3f}  ({vm['n_samples']} clips, "
          f"{custom['inference_ms']} ms/clip)")
    return {
        "title": "Voice — engines on Speech Commands v2 (six command words)",
        "note": ("vosk runs with a limited-vocabulary grammar over the same six "
                 "words (its keyword-spotting mode). Both engines are scored by "
                 "the same function as the custom CNNs above. One asymmetry to "
                 f"note: vosk can decline to answer, and did so on {rejects} of "
                 f"{n} clips — each counts as a miss (it is scored WITH its "
                 "reject option, while the CNN column is pure argmax with its "
                 "0.80 runtime confidence gate switched off, so if anything vosk "
                 "is handicapped here). Per-class averages cover the six real "
                 "words; accuracy counts every clip."),
        "order": list(CLS_ORDER) + ["inference_ms"],
        "confusion": {"labels": KWS_WORDS + [REJECT], "matrix": cm.tolist(),
                      "title": "Voice — vosk confusion matrix"},
        "models": {
            "custom": {"title": "custom (our CNN)", "dataset": "Speech Commands v2",
                       "n_samples": vm["n_samples"], "metrics": custom},
            "vosk": {"title": "vosk (pretrained ASR)", "dataset": "Speech Commands v2",
                     "n_samples": n, "metrics": m},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aflw", default=str(ROOT / "AFLW2000"))
    ap.add_argument("--mpii", default=str(ROOT / "MPIIGaze"))
    ap.add_argument("--limit-voice", type=int, default=150,
                    help="validation clips per word for the vosk eval")
    ap.add_argument("--limit-gaze", type=int, default=200,
                    help="images per participant for the gaze eval (15 people)")
    args = ap.parse_args()

    print("gaze: evaluating all providers on MPIIGaze (Original)…", flush=True)
    g_section = eval_gaze(Path(args.mpii), args.limit_gaze)
    print("head pose: evaluating classical providers on AFLW2000…", flush=True)
    hp_section = eval_headpose(Path(args.aflw))
    print("voice: evaluating vosk on Speech Commands…", flush=True)
    v_section = eval_voice(args.limit_voice)

    rep = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sections": [g_section, hp_section, v_section],
        "notes": [
            "mp_geometry (head pose) reports pose relative to a per-user neutral "
            "face cloud, so it has no absolute output to score against a dataset; "
            "see the runtime tables for its live performance.",
            "The classical blink detectors (EAR standard/adaptive, ELA) are "
            "geometric rules over FaceMesh eye landmarks, so scoring them needs "
            "full-FACE images labelled open/closed. Every eye dataset on this "
            "machine (mrlEyes_2018_01 and both archive copies) contains cropped "
            "eye images with no surrounding face, which FaceMesh cannot process, "
            "and no full-face set here (300W-LP, AFLW2000, MPIIGaze) carries "
            "eye-state labels. They are therefore compared in the runtime tables "
            "only; a labelled face set such as CEW (Closed Eyes in the Wild) "
            "would be needed to score them against the blink CNN's MRL-Eye "
            "numbers. This gap is stated rather than filled with a proxy metric.",
        ],
    }
    out = ROOT / "models" / "pretrained_report.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
