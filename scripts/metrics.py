"""Shared metric computation for the 4 custom-model training scripts.

One function per task family, both backed by sklearn so the numbers match
the standard definitions instead of hand-rolled formulas. Each training
script calls `collect_classification`/`collect_regression` once on its best
model, then feeds the result to `classification_metrics`/`regression_metrics`
to get the full ~15-20-field dict written to `*.metrics.json`.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn import metrics as skm


def collect_classification(model, loader, device):
    """Run model over loader, return (y_true, y_pred, y_prob) numpy arrays."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            prob = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(prob.argmax(1))
            y_prob.append(prob)
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)


def collect_regression(model, loader, device):
    """Run model over loader, return (y_true, y_pred) numpy arrays."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            y_pred.append(model(x.to(device)).cpu().numpy())
            y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """y_true/y_pred: 1D int arrays. y_prob: (n, n_classes) softmax, optional."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    out = {
        "n_samples": len(y_true),
        "n_classes": n_classes,
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
        "precision_macro": skm.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": skm.recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": skm.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": skm.precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": skm.recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": skm.f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": skm.matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": skm.cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
        "per_class_f1": skm.f1_score(y_true, y_pred, average=None,
                                      labels=list(range(n_classes)), zero_division=0).tolist(),
    }
    if n_classes == 2:
        tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["false_positive_rate"] = fp / max(1, fp + tn)
        out["false_negative_rate"] = fn / max(1, fn + tp)
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if n_classes == 2:
                out["roc_auc"] = skm.roc_auc_score(y_true, y_prob[:, 1])
                out["pr_auc"] = skm.average_precision_score(y_true, y_prob[:, 1])
                out["brier_score"] = skm.brier_score_loss(y_true, y_prob[:, 1])
            else:
                out["roc_auc_ovr_macro"] = skm.roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(n_classes)))
                out["top5_accuracy"] = _top_k_accuracy(y_true, y_prob, k=min(5, n_classes))
            out["log_loss"] = skm.log_loss(y_true, y_prob, labels=list(range(n_classes)))
        except ValueError:
            pass  # a class missing from this split — skip prob-based metrics rather than crash
    return out


def _top_k_accuracy(y_true, y_prob, k: int) -> float:
    topk = np.argsort(-y_prob, axis=1)[:, :k]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def regression_metrics(y_true_deg, y_pred_deg, axis_names) -> dict:
    """y_true_deg/y_pred_deg: (n, len(axis_names)) arrays, already in degrees."""
    y_true = np.asarray(y_true_deg, dtype=np.float64)
    y_pred = np.asarray(y_pred_deg, dtype=np.float64)
    err = y_pred - y_true
    abs_err = np.abs(err)
    out = {
        "n_samples": len(y_true),
        "mae_deg": float(abs_err.mean()),
        "rmse_deg": float(np.sqrt((err ** 2).mean())),
        "median_ae_deg": float(np.median(abs_err)),
        "max_ae_deg": float(abs_err.max()),
        "p90_ae_deg": float(np.percentile(abs_err, 90)),
        "p95_ae_deg": float(np.percentile(abs_err, 95)),
    }
    for i, name in enumerate(axis_names):
        out[f"mae_deg_{name}"] = float(abs_err[:, i].mean())
        out[f"rmse_deg_{name}"] = float(np.sqrt((err[:, i] ** 2).mean()))
        out[f"bias_deg_{name}"] = float(err[:, i].mean())
        out[f"std_deg_{name}"] = float(err[:, i].std())
        out[f"r2_{name}"] = float(skm.r2_score(y_true[:, i], y_pred[:, i]))
        out[f"corr_{name}"] = float(np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1])
    return out


def gaze_angular_error_deg(pitch_yaw_true_rad, pitch_yaw_pred_rad) -> dict:
    """Standard MPIIGaze metric: angle (deg) between the 3D unit gaze vectors —
    NOT the same number as per-axis MAE, since pitch/yaw errors don't compose
    linearly into true angular distance. Inputs in radians, (n, 2) [pitch, yaw]."""
    def to_vec(pitch, yaw):
        v = np.stack([-np.sin(yaw) * np.cos(pitch),
                      -np.sin(pitch),
                      -np.cos(yaw) * np.cos(pitch)], axis=-1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    t, p = np.asarray(pitch_yaw_true_rad), np.asarray(pitch_yaw_pred_rad)
    vt, vp = to_vec(t[:, 0], t[:, 1]), to_vec(p[:, 0], p[:, 1])
    ang = np.degrees(np.arccos(np.clip((vt * vp).sum(-1), -1.0, 1.0)))
    return {
        "angular_error_deg_mean": float(ang.mean()),
        "angular_error_deg_median": float(np.median(ang)),
        "angular_error_deg_p90": float(np.percentile(ang, 90)),
    }
