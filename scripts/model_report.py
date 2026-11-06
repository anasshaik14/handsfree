"""Build one quality report for all four trained custom CNNs.

Two model families, each shown with the metrics that actually fit it:

  CLASSIFIERS  blink (open/closed), voice (command keywords)
      -> the famous classification suite (accuracy, precision, recall, F-scores,
         MCC, kappa, ...), all pure functions of the saved confusion matrix.

  REGRESSORS   gaze (pitch/yaw), headpose (yaw/pitch/roll)
      -> regression error (MAE/RMSE/R2) PLUS tolerance / PCK accuracy: the
         fraction of predictions within N degrees of ground truth. For a
         regressor THAT is "accuracy" (eye-tracking / pose papers report it
         this way); binning a 2.5-degree-accurate gaze model into a 9-zone grid
         and calling the boundary flips "errors" understated it badly.

Every model still gets one comparable headline **Accuracy** number. Writes
models/custom/model_report.json (read by app/metrics/report.py) + a standalone
reports/model_metrics.html.

Usage:  python scripts/model_report.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
CUSTOM = ROOT / "models" / "custom"

# ---- classification suite (blink, voice) -------------------------------------
CLS_ORDER = [
    "accuracy", "balanced_accuracy", "error_rate",
    "precision_macro", "precision_weighted",
    "recall_macro", "recall_weighted", "specificity_macro",
    "f1_macro", "f1_weighted", "fhalf_macro", "f2_macro", "jaccard_macro",
    "mcc", "cohen_kappa", "fpr_macro", "fnr_macro", "fdr_macro",
]
CLS_LABELS = {
    "accuracy": ["Accuracy", "higher is better"],
    "balanced_accuracy": ["Balanced accuracy", "per-class mean recall"],
    "error_rate": ["Error rate", "lower is better"],
    "precision_macro": ["Precision (macro)", "higher is better"],
    "precision_weighted": ["Precision (weighted)", "support-weighted"],
    "recall_macro": ["Recall / Sensitivity (macro)", "higher is better"],
    "recall_weighted": ["Recall (weighted)", "support-weighted"],
    "specificity_macro": ["Specificity / TNR (macro)", "higher is better"],
    "f1_macro": ["F1-score (macro)", "higher is better"],
    "f1_weighted": ["F1-score (weighted)", "support-weighted"],
    "fhalf_macro": ["F0.5-score (macro)", "precision-weighted"],
    "f2_macro": ["F2-score (macro)", "recall-weighted"],
    "jaccard_macro": ["Jaccard / IoU (macro)", "higher is better"],
    "mcc": ["Matthews corr. (MCC)", "-1..1, higher"],
    "cohen_kappa": ["Cohen's kappa", "chance-adjusted"],
    "fpr_macro": ["False-positive rate (macro)", "lower is better"],
    "fnr_macro": ["False-negative rate (macro)", "lower is better"],
    "fdr_macro": ["False-discovery rate (macro)", "lower is better"],
}

# ---- regression suite (gaze, headpose) ---------------------------------------
PCK = (5, 10, 15)                       # tolerance thresholds shared by both regressors
REG_ORDER = [
    "acc_within_5deg", "acc_within_10deg", "acc_within_15deg",
    "mae_deg", "rmse_deg", "median_ae_deg", "p90_ae_deg", "angular_mean_deg",
    "r2_best", "r2_worst", "corr_best", "corr_worst",
]
REG_LABELS = {
    "acc_within_5deg": ["Accuracy within 5°", "PCK, higher is better"],
    "acc_within_10deg": ["Accuracy within 10°", "PCK, higher is better"],
    "acc_within_15deg": ["Accuracy within 15°", "PCK, higher is better"],
    "mae_deg": ["Mean abs error (°)", "lower is better"],
    "rmse_deg": ["RMSE (°)", "lower is better"],
    "median_ae_deg": ["Median abs error (°)", "lower is better"],
    "p90_ae_deg": ["P90 abs error (°)", "lower is better"],
    "angular_mean_deg": ["Mean angular error (°)", "3D gaze vector"],
    "r2_best": ["R² (best axis)", "higher is better"],
    "r2_worst": ["R² (worst axis)", "higher is better"],
    "corr_best": ["Correlation (best axis)", "higher is better"],
    "corr_worst": ["Correlation (worst axis)", "higher is better"],
}


def unified_metrics(cm, score_labels=None) -> dict:
    """All 18 classification metrics from a KxK confusion matrix (expanded to
    label arrays so sklearn gives the exact standard values).

    `score_labels` restricts the per-class averages (macro/weighted/specificity
    and the FPR/FNR/FDR family) to those class indices. That's for an engine
    with a REJECT option: vosk can decline to answer, so its matrix carries an
    extra "unrecognized" column that is never a ground-truth class, and macro-
    averaging over it would invent a phantom 0%-precision class. Accuracy and
    the chance-corrected metrics still see every sample — a rejected clip is a
    wrong answer, not an excused one.
    """
    cm = np.asarray(cm, dtype=np.int64)
    K = cm.shape[0]
    yt, yp = [], []
    for i in range(K):
        for j in range(K):
            n = int(cm[i, j])
            if n:
                yt.extend([i] * n)
                yp.extend([j] * n)
    yt, yp = np.array(yt), np.array(yp)
    lab = list(range(K)) if score_labels is None else list(score_labels)
    sel = np.asarray(lab)
    diag = np.diag(cm).astype(float)
    tp = diag[sel]
    fp = (cm.sum(0) - diag)[sel]
    fn = (cm.sum(1) - diag)[sel]
    tn = cm.sum() - tp - fp - fn
    spec = np.divide(tn, tn + fp, out=np.zeros(len(sel)), where=(tn + fp) > 0)
    acc = skm.accuracy_score(yt, yp)
    kw = {"labels": lab, "zero_division": 0}
    m = {
        "accuracy": acc,
        "balanced_accuracy": skm.balanced_accuracy_score(yt, yp),
        "error_rate": 1.0 - acc,
        "precision_macro": skm.precision_score(yt, yp, average="macro", **kw),
        "precision_weighted": skm.precision_score(yt, yp, average="weighted", **kw),
        "recall_macro": skm.recall_score(yt, yp, average="macro", **kw),
        "recall_weighted": skm.recall_score(yt, yp, average="weighted", **kw),
        "specificity_macro": float(spec.mean()),
        "f1_macro": skm.f1_score(yt, yp, average="macro", **kw),
        "f1_weighted": skm.f1_score(yt, yp, average="weighted", **kw),
        "fhalf_macro": skm.fbeta_score(yt, yp, beta=0.5, average="macro", **kw),
        "f2_macro": skm.fbeta_score(yt, yp, beta=2.0, average="macro", **kw),
        "jaccard_macro": skm.jaccard_score(yt, yp, average="macro", **kw),
        "mcc": skm.matthews_corrcoef(yt, yp),
        "cohen_kappa": skm.cohen_kappa_score(yt, yp),
        "fpr_macro": float((fp / np.maximum(fp + tn, 1)).mean()),
        "fnr_macro": float((fn / np.maximum(fn + tp, 1)).mean()),
        "fdr_macro": float((fp / np.maximum(fp + tp, 1)).mean()),
    }
    return {k: round(float(v), 4) for k, v in m.items()}


def facing_angular_deg(true_ypr, pred_ypr):
    """Per-sample angle (deg) between the true and predicted face-FORWARD
    directions, from (yaw, pitch, roll) in degrees.

    The head-pose counterpart of the gaze angular error, and needed for the
    same reason: yaw and pitch errors don't compose linearly, so per-axis MAE
    can't tell you how far off the face is actually pointing. Roll is excluded
    on purpose — it spins the face about the forward axis without moving it.
    """
    def vec(yaw, pitch):
        y, p = np.radians(yaw), np.radians(pitch)
        return np.stack([-np.sin(y) * np.cos(p), -np.sin(p),
                         -np.cos(y) * np.cos(p)], -1)
    vt = vec(true_ypr[:, 0], true_ypr[:, 1])
    vp = vec(pred_ypr[:, 0], pred_ypr[:, 1])
    return np.degrees(np.arccos(np.clip((vt * vp).sum(-1), -1.0, 1.0)))


def reg_metrics(true, pred, axes, angular=None) -> dict:
    """Regression + PCK. true/pred in degrees, (N, len(axes)). `angular` is the
    per-sample 3D angular error (deg) when it applies (gaze); otherwise PCK uses
    the per-sample mean abs error across axes."""
    err = np.abs(pred - true)
    flat = err.reshape(-1)
    samp = angular if angular is not None else err.mean(1)
    r2 = [skm.r2_score(true[:, i], pred[:, i]) for i in range(len(axes))]
    corr = [float(np.corrcoef(true[:, i], pred[:, i])[0, 1]) for i in range(len(axes))]
    d = {
        "mae_deg": float(flat.mean()),
        "rmse_deg": float(np.sqrt((flat ** 2).mean())),
        "median_ae_deg": float(np.median(flat)),
        "p90_ae_deg": float(np.percentile(flat, 90)),
        "r2_best": max(r2), "r2_worst": min(r2),
        "corr_best": max(corr), "corr_worst": min(corr),
    }
    for t in PCK:
        d[f"acc_within_{t}deg"] = float((samp <= t).mean())
    if angular is not None:
        d["angular_mean_deg"] = float(angular.mean())
    return {k: round(float(v), 4) for k, v in d.items()}


def _predict(model_path, X, batch=256):
    import torch
    model = torch.jit.load(str(model_path))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(X[i:i + batch]).cpu().numpy())
    return np.concatenate(out)


def gaze_eval(data_root, limit=0):
    """Score the gaze CNN on the HELD-OUT TEST participants only.

    Was scoring `_load_dataset(root, 8000)` — the whole dataset, which the model
    trained on, so the headline was a train-set number. Rebuild the exact same
    participant-disjoint split the trainer uses and evaluate the test partition.
    """
    import torch
    import train_gaze
    import tuning
    ds = train_gaze.MPIIGaze(str(data_root), limit)
    _, _, idx_te = tuning.group_split(ds.parts, (0.7, 0.15, 0.15), seed=0)
    X = torch.stack([ds.imgs_tensor(i) for i in idx_te])
    true = np.degrees(ds.targets[idx_te])              # (N,2) pitch,yaw deg
    predr = _predict(CUSTOM / "gaze_cnn.pt", X)        # (N,2) pitch,yaw rad
    pred = np.degrees(predr)
    tr, pr = np.radians(true), predr

    def vec(p, y):
        return np.stack([-np.sin(y) * np.cos(p), -np.sin(p), -np.cos(y) * np.cos(p)], -1)
    vt, vp = vec(tr[:, 0], tr[:, 1]), vec(pr[:, 0], pr[:, 1])
    ang = np.degrees(np.arccos(np.clip((vt * vp).sum(-1), -1, 1)))
    return reg_metrics(true, pred, ["pitch", "yaw"], angular=ang), len(true)


def headpose_eval(eval_root):
    import torch
    import train_headpose
    crops, tgts, ids = train_headpose._load_pairs(str(eval_root), 0, tag="aflw2000")
    ds = train_headpose.HeadPoseSet(crops, tgts, ids)
    true = ds.targets                                  # (N,3) yaw,pitch,roll deg
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    pred = _predict(CUSTOM / "headpose_cnn.pt", X)
    m = reg_metrics(true, pred, ["yaw", "pitch", "roll"])
    # Added separately, NOT via reg_metrics(angular=...), which would also
    # redefine PCK off the angular error and silently move the headline number.
    m["angular_mean_deg"] = round(float(facing_angular_deg(true, pred).mean()), 4)
    return m, len(true)


def _saved(name):
    return json.loads((CUSTOM / f"{name}.metrics.json").read_text())


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze", default=str(ROOT / "MPIIGaze" / "Data" / "Normalized"))
    ap.add_argument("--aflw", default=str(ROOT / "AFLW2000"))
    ap.add_argument("--gaze-limit", type=int, default=8000)
    args = ap.parse_args()

    blink, voice = _saved("blink_cnn"), _saved("voice_kws")
    print("blink/voice: classification metrics from saved confusion matrices", flush=True)
    print("gaze: evaluating regressor (MAE + PCK tolerance accuracy)...", flush=True)
    g_m, g_n = gaze_eval(args.gaze, args.gaze_limit)
    print("headpose: evaluating regressor (MAE + PCK tolerance accuracy)...", flush=True)
    h_m, h_n = headpose_eval(args.aflw)

    cls_models = {
        "blink": {
            "title": "Blink", "classes": f"{blink['n_classes']} — open / closed",
            "dataset": "MRL Eye", "n_samples": blink["n_samples"],
            "metrics": unified_metrics(blink["confusion_matrix"]),
        },
        "voice": {
            "title": "Voice", "classes": f"{voice['n_classes']} — command words",
            "dataset": "Speech Commands v2", "n_samples": voice["n_samples"],
            "metrics": unified_metrics(voice["confusion_matrix"]),
        },
    }
    reg_models = {
        "gaze": {
            "title": "Gaze", "dataset": "MPIIGaze", "n_samples": g_n,
            "metrics": g_m, "headline_key": "acc_within_5deg",
            "headline_note": "within 5°",
        },
        "headpose": {
            "title": "Head pose", "dataset": "AFLW2000", "n_samples": h_n,
            "metrics": h_m, "headline_key": "acc_within_15deg",
            "headline_note": "within 15°",
        },
    }
    headline = {}
    for name, mm in cls_models.items():
        headline[name] = {"title": mm["title"], "accuracy": mm["metrics"]["accuracy"],
                          "kind": "classification accuracy"}
    for name, mm in reg_models.items():
        headline[name] = {"title": mm["title"],
                          "accuracy": mm["metrics"][mm["headline_key"]],
                          "kind": f"PCK accuracy ({mm['headline_note']})"}

    rep = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "headline": headline,
        "cls": {"order": CLS_ORDER, "labels": CLS_LABELS, "models": cls_models},
        "reg": {"order": REG_ORDER, "labels": REG_LABELS, "models": reg_models},
    }
    (CUSTOM / "model_report.json").write_text(json.dumps(rep, indent=2))

    from app.metrics import report as rpt
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>HandsFree - Model Metrics</title><style>"
        "body{font-family:Segoe UI,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1a1a2e}"
        "h1{font-size:1.5rem}h2{font-size:1.15rem;margin-top:2rem}"
        "h3{font-size:1rem;margin-top:1.6rem}"
        "table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.4rem 0}"
        "th,td{border:1px solid #d5d5e0;padding:.45rem .6rem;text-align:center}"
        "th{background:#eef0f7}td.m{text-align:left;font-weight:600}"
        "td.m span{display:block;font-weight:400;color:#888;font-size:.75rem}"
        "td span.of{color:#999;font-size:.78rem}"
        "td span{display:block;font-weight:400;color:#888;font-size:.72rem}"
        ".rate{display:block;font-size:.72rem;font-weight:600;padding:0 .4rem;margin-top:.15rem}"
        ".rate.ok{color:#0a7a3d}.rate.mid{color:#a6690a}.rate.bad{color:#b02a2a}"
        ".cm td.good{background:#e8f7ee;font-weight:600}.cm td.bad{background:#fdeaea}"
        "p.note{color:#555;font-size:.85rem}"
        "</style></head><body><h1>HandsFree &mdash; Custom Model Metrics</h1>"
        + rpt.model_quality_html(rep) + "</body></html>")
    (ROOT / "reports" / "model_metrics.html").write_text(html, encoding="utf-8")

    for n, hd in headline.items():
        print(f"  {hd['title']:10s} accuracy={hd['accuracy']:.4f}  ({hd['kind']})")
    print(f"wrote {CUSTOM / 'model_report.json'} and reports/model_metrics.html")


if __name__ == "__main__":
    main()
