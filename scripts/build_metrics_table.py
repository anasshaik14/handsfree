"""Compute the SAME full metric suite for the pretrained models as for the custom models,
on the matched per-sample data, so every number is directly comparable.

Uses the identical metrics.py functions the custom training used. Writes:
  reports/metrics_custom_vs_pretrained.xlsx  (one sheet per input, custom | pretrained | Δ)
  reports/metrics_aligned.json               (for the PDF table pages)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import metrics as M

PT = ROOT / "reports" / "tuning" / "pretrained"


def reg(true, pred, axes, gaze=False):
    d = M.regression_metrics(true, pred, axes)
    if gaze:
        d.update(M.gaze_angular_error_deg(np.radians(true), np.radians(pred)))
    return d


def cls(y, prob):
    return M.classification_metrics(y, prob.argmax(1), prob)


def build():
    out = {}

    # Held-out participants only. The 15-participant matched file contains nine of the
    # custom model's own TRAINING participants, which understates its error by ~1.9 deg.
    _ho = PT / "matched_gaze_heldout.npz"
    g = np.load(_ho if _ho.exists() else PT / "matched_gaze.npz")
    out["gaze"] = {
        "kind": "regression", "axes": ["pitch", "yaw"], "N": int(len(g["true"])),
        "custom": reg(g["true"], g["pred_custom"], ["pitch", "yaw"], gaze=True),
        "pretrained": reg(g["true"], g["pred_pt"], ["pitch", "yaw"], gaze=True),
        "custom_name": "Custom CNN", "pretrained_name": "MPIIGaze ResNet (pretrained)",
        "note": "Matched per-sample on MPIIGaze normalized eyes, restricted to the custom "
                "model's HELD-OUT participants (p05, p07) so no training identity is scored. "
                "The pretrained baseline was itself trained on MPIIGaze and very likely saw "
                "these participants, so its figure is a lower bound. Lower is better (degrees).",
    }

    h = np.load(PT / "matched_headpose.npz")
    out["headpose"] = {
        "kind": "regression", "axes": ["yaw", "pitch", "roll"], "N": int(len(h["true"])),
        "custom": reg(h["true"], h["pred_custom"], ["yaw", "pitch", "roll"]),
        "pretrained": reg(h["true"], h["pred_pt"], ["yaw", "pitch", "roll"]),
        "custom_name": "Custom CNN", "pretrained_name": "6DRepNet (pretrained)",
        "note": "Matched per-sample on AFLW2000 (1,969 faces). Lower is better (degrees).",
    }

    b = np.load(PT / "matched_blink.npz", allow_pickle=True)
    yb = (b["label"] == 1).astype(int)
    out["blink"] = {
        "kind": "classification", "N": int(len(yb)),
        "custom": cls(yb, b["prob"][:, [0, 1]]),
        "pretrained": None,
        "custom_name": "Custom CNN", "pretrained_name": "Classical EAR (maths rule)",
        "note": "Custom scored per-sample on MRL test split (12,927). The EAR alternative is a "
                "landmark maths rule, not a neural net, and cannot be scored per-sample on 32x32 "
                "eye crops — so no pretrained column exists for blink.",
    }

    v = np.load(PT / "matched_voice.npz", allow_pickle=True)
    vosk = json.load(open(PT / "voice_vosk.metrics.json"))
    out["voice"] = {
        "kind": "classification", "N": int(len(v["label"])),
        "custom": cls(v["label"].astype(int), v["prob"]),
        "pretrained": vosk,
        "custom_name": "Custom CNN", "pretrained_name": "Vosk (pretrained)",
        "note": "Custom scored per-sample on Speech Commands test split (2,468). Vosk returns a hard "
                "transcript (no probabilities), so its probability-based metrics (ROC-AUC, log-loss, "
                "Top-5) are N/A; all label-based metrics use the identical formulas.",
    }
    return out


REG_ORDER = ["n_samples", "mae_deg", "rmse_deg", "median_ae_deg", "p90_ae_deg", "p95_ae_deg", "max_ae_deg",
             "angular_error_deg_mean", "angular_error_deg_median", "angular_error_deg_p90"]
REG_PER = ["mae_deg", "rmse_deg", "bias_deg", "std_deg", "r2", "corr"]
CLS_ORDER = ["n_samples", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
             "precision_weighted", "recall_weighted", "f1_weighted", "mcc", "cohen_kappa",
             "false_positive_rate", "false_negative_rate", "roc_auc", "pr_auc", "brier_score",
             "roc_auc_ovr_macro", "top5_accuracy", "log_loss"]

PRETTY = {
    "n_samples": "Samples (N)", "mae_deg": "MAE (deg)", "rmse_deg": "RMSE (deg)",
    "median_ae_deg": "Median AE (deg)", "p90_ae_deg": "P90 AE (deg)", "p95_ae_deg": "P95 AE (deg)",
    "max_ae_deg": "Max AE (deg)", "angular_error_deg_mean": "Angular error mean (deg)",
    "angular_error_deg_median": "Angular error median (deg)", "angular_error_deg_p90": "Angular error P90 (deg)",
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)", "precision_weighted": "Precision (weighted)",
    "recall_weighted": "Recall (weighted)", "f1_weighted": "F1 (weighted)", "mcc": "Matthews corr. (MCC)",
    "cohen_kappa": "Cohen's kappa", "false_positive_rate": "False-positive rate",
    "false_negative_rate": "False-negative rate", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC (avg precision)",
    "brier_score": "Brier score", "roc_auc_ovr_macro": "ROC-AUC (OvR macro)",
    "top5_accuracy": "Top-5 accuracy", "log_loss": "Log loss",
}


def rows_for(entry):
    """Return list of (label, custom_str, pretrained_str) for display."""
    c, p = entry["custom"], entry["pretrained"]
    rows = []
    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, int) or (isinstance(v, float) and abs(v) >= 100):
            return f"{v:,.0f}"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.3f}"
    if entry["kind"] == "regression":
        for k in REG_ORDER:
            if k in c:
                rows.append((PRETTY.get(k, k), fmt(c.get(k)), fmt(p.get(k)) if p else "N/A"))
        rows.append(("— per axis —", "", ""))
        for ax in entry["axes"]:
            for base in REG_PER:
                k = f"{base}_{ax}"
                if k in c:
                    lbl = f"{PRETTY.get(base, base)} · {ax}" if base in PRETTY else f"{base} · {ax}"
                    lbl = {"mae_deg": "MAE", "rmse_deg": "RMSE", "bias_deg": "Bias", "std_deg": "Std",
                           "r2": "R²", "corr": "Correlation"}[base] + f" · {ax}"
                    rows.append((lbl, fmt(c.get(k)), fmt(p.get(k)) if p else "N/A"))
    else:
        for k in CLS_ORDER:
            if k in c or (p and k in p):
                cv = fmt(c.get(k)) if k in c else "—"
                pv = fmt(p.get(k)) if (p and p.get(k) is not None) else ("N/A" if p else "N/A")
                rows.append((PRETTY.get(k, k), cv, pv))
    return rows


def write_xlsx(data):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = Workbook(); wb.remove(wb.active)
    hdr = Font(bold=True, color="FFFFFF"); hf = PatternFill("solid", fgColor="2A78D6")
    for key in ("gaze", "headpose", "voice", "blink"):
        e = data[key]; ws = wb.create_sheet(key.capitalize())
        ws["A1"] = f"{key.capitalize()} — full metric suite (custom vs pretrained)"; ws["A1"].font = Font(bold=True, size=13)
        ws["A2"] = e["note"]; ws["A2"].font = Font(italic=True, size=9, color="666666")
        r = 4
        for col, txt in enumerate(["Metric", e["custom_name"], e["pretrained_name"]], 1):
            c = ws.cell(r, col, txt); c.font = hdr; c.fill = hf; c.alignment = Alignment(horizontal="center")
        r += 1
        for lbl, cv, pv in rows_for(e):
            ws.cell(r, 1, lbl); ws.cell(r, 2, cv); ws.cell(r, 3, pv)
            if lbl.startswith("—"):
                ws.cell(r, 1).font = Font(italic=True, color="999999")
            r += 1
        ws.column_dimensions["A"].width = 26; ws.column_dimensions["B"].width = 18; ws.column_dimensions["C"].width = 26
    out = ROOT / "reports" / "metrics_custom_vs_pretrained.xlsx"
    wb.save(out); print("wrote", out)


if __name__ == "__main__":
    data = build()
    (ROOT / "reports" / "metrics_aligned.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("wrote reports/metrics_aligned.json")
    write_xlsx(data)
    # quick console preview
    for k in ("gaze", "headpose", "voice", "blink"):
        print(f"\n=== {k} ===")
        for lbl, cv, pv in rows_for(data[k])[:8]:
            print(f"  {lbl:28s} custom={cv:>10s}   pretrained={pv:>12s}")
