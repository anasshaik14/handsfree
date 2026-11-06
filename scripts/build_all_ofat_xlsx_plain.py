"""Build PLAIN (no colours, no styling) .xlsx versions of the 10-table adaptive
OFAT sweeps for all four models. Same content as the styled workbooks — 10
parameter tables, every hyperparameter per trial + the full metric suite, and
a plain-text 'best' marker on each table's winning trial — but with no fills,
fonts, borders, or freeze panes. Writes *_ofat_10tables_plain.xlsx; does not
touch the styled *_ofat_10tables.xlsx files.

  python scripts/build_all_ofat_xlsx_plain.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parent.parent
TUN = ROOT / "reports" / "tuning"

TABLES = [
    ("P1_learning_rate", "P1 - Learning Rate", "Learning Rate"),
    ("P2_optimizer", "P2 - Optimizer", "Optimizer"),
    ("P3_dropout", "P3 - Dropout", "Dropout"),
    ("P4_augmentation", "P4 - Augmentation", "Augmentation"),
    ("P5_batch_size", "P5 - Batch Size", "Batch size"),
    ("P6_lr_schedule", "P6 - LR Schedule", "LR Schedule"),
    ("P7_weight_decay", "P7 - Weight Decay", "Weight Decay"),
    ("P8_epochs", "P8 - Epochs", "Epochs"),
    ("P9_conv_width", "P9 - Conv Width", "Conv Width"),
    ("P10_fc_width", "P10 - FC Width", "FC Width"),
]
PARAMS = [
    ("Learning Rate", "lr"), ("Batch size", "batch"), ("Dropout", "dropout"),
    ("Weight Decay", "weight_decay"), ("Epochs", "epochs"), ("Optimizer", "opt"),
    ("LR Schedule", "sched"), ("Augmentation", "aug"), ("Conv Width", "width"),
    ("FC Width", "fc"),
]

CLS_METRICS = [
    ("Accuracy", "accuracy"), ("Balanced Acc", "balanced_accuracy"),
    ("Precision (macro)", "precision_macro"), ("Recall (macro)", "recall_macro"),
    ("F1 (macro)", "f1_macro"), ("Precision (wtd)", "precision_weighted"),
    ("Recall (wtd)", "recall_weighted"), ("F1 (wtd)", "f1_weighted"),
    ("MCC", "mcc"), ("Cohen Kappa", "cohen_kappa"),
    ("FPR", "false_positive_rate"), ("FNR", "false_negative_rate"),
    ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
    ("Brier", "brier_score"), ("Log Loss", "log_loss"),
]
VOICE_METRICS = [
    ("Accuracy", "accuracy"), ("Balanced Acc", "balanced_accuracy"),
    ("Precision (macro)", "precision_macro"), ("Recall (macro)", "recall_macro"),
    ("F1 (macro)", "f1_macro"), ("Precision (wtd)", "precision_weighted"),
    ("Recall (wtd)", "recall_weighted"), ("F1 (wtd)", "f1_weighted"),
    ("MCC", "mcc"), ("Cohen Kappa", "cohen_kappa"),
    ("ROC-AUC (OvR)", "roc_auc_ovr_macro"), ("Top-5 Acc", "top5_accuracy"),
    ("Log Loss", "log_loss"),
]
GAZE_METRICS = [
    ("MAE deg", "mae_deg"), ("RMSE deg", "rmse_deg"), ("Median AE deg", "median_ae_deg"),
    ("Max AE deg", "max_ae_deg"), ("P90 AE deg", "p90_ae_deg"), ("P95 AE deg", "p95_ae_deg"),
    ("MAE pitch", "mae_deg_pitch"), ("RMSE pitch", "rmse_deg_pitch"),
    ("Bias pitch", "bias_deg_pitch"), ("Std pitch", "std_deg_pitch"),
    ("R2 pitch", "r2_pitch"), ("Corr pitch", "corr_pitch"),
    ("MAE yaw", "mae_deg_yaw"), ("RMSE yaw", "rmse_deg_yaw"),
    ("Bias yaw", "bias_deg_yaw"), ("Std yaw", "std_deg_yaw"),
    ("R2 yaw", "r2_yaw"), ("Corr yaw", "corr_yaw"),
    ("Angular mean", "angular_error_deg_mean"), ("Angular median", "angular_error_deg_median"),
    ("Angular p90", "angular_error_deg_p90"),
]
HEAD_METRICS = [
    ("MAE deg", "mae_deg"), ("RMSE deg", "rmse_deg"), ("Median AE deg", "median_ae_deg"),
    ("Max AE deg", "max_ae_deg"), ("P90 AE deg", "p90_ae_deg"), ("P95 AE deg", "p95_ae_deg"),
    ("MAE yaw", "mae_deg_yaw"), ("RMSE yaw", "rmse_deg_yaw"),
    ("Bias yaw", "bias_deg_yaw"), ("Std yaw", "std_deg_yaw"),
    ("R2 yaw", "r2_yaw"), ("Corr yaw", "corr_yaw"),
    ("MAE pitch", "mae_deg_pitch"), ("RMSE pitch", "rmse_deg_pitch"),
    ("Bias pitch", "bias_deg_pitch"), ("Std pitch", "std_deg_pitch"),
    ("R2 pitch", "r2_pitch"), ("Corr pitch", "corr_pitch"),
    ("MAE roll", "mae_deg_roll"), ("RMSE roll", "rmse_deg_roll"),
    ("Bias roll", "bias_deg_roll"), ("Std roll", "std_deg_roll"),
    ("R2 roll", "r2_roll"), ("Corr roll", "corr_roll"),
]

MODELS = [
    dict(key="blink", prefix="ofat_blink_adaptive", out="blink_ofat_10tables_plain.xlsx",
         title="Blink CNN - Adaptive OFAT Hyperparameter Sweep",
         metrics=CLS_METRICS, objective="accuracy", lower_better=False, capped=()),
    dict(key="gaze", prefix="ofat_gaze_adaptive", out="gaze_ofat_10tables_plain.xlsx",
         title="Gaze CNN - Adaptive OFAT Hyperparameter Sweep (objective: angular error deg, lower better)",
         metrics=GAZE_METRICS, objective="angular_error_deg_mean", lower_better=True, capped=()),
    dict(key="headpose", prefix="ofat_headpose_adaptive", out="headpose_ofat_10tables_plain.xlsx",
         title="Head Pose CNN - Adaptive OFAT Hyperparameter Sweep (objective: MAE deg, lower better)",
         metrics=HEAD_METRICS, objective="mae_deg", lower_better=True,
         capped=("P9_conv_width", "P10_fc_width")),
    dict(key="voice", prefix="ofat_voice_adaptive", out="voice_ofat_10tables_plain.xlsx",
         title="Voice KWS CNN - Adaptive OFAT Hyperparameter Sweep",
         metrics=VOICE_METRICS, objective="accuracy", lower_better=False,
         capped=("P9_conv_width", "P10_fc_width")),
]


def fmt(key, val):
    if val in ("", None):
        return ""
    if key == "aug":
        return "On" if str(val).strip() in ("1", "1.0", "True") else "Off"
    if key in ("epochs", "batch", "width", "fc", "seed", "trial"):
        try:
            return int(float(val))
        except ValueError:
            return val
    if key in ("opt", "sched"):
        return str(val)
    try:
        f = float(val)
    except ValueError:
        return val
    if key == "lr":
        return round(f, 6)
    if key == "weight_decay":
        return float(f"{f:.3e}")
    if key == "dropout":
        return round(f, 4)
    return round(f, 5)


def build(m):
    metrics = m["metrics"]
    headers = ["Trial", "Seed"] + [p[0] for p in PARAMS] + ["METRICS:"] + [d for d, _ in metrics] + ["Winner"]

    wb = Workbook()
    ws = wb.active
    ws.title = f"{m['key']} - 10 OFAT tables"

    r = 1
    ws.cell(r, 1, m["title"])
    r += 2

    obj = m["objective"]
    total = 0
    for suffix, title, swept in TABLES:
        path = TUN / f"{m['prefix']}_{suffix}_trials.csv"
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        total += len(rows)
        win_idx = (min if m["lower_better"] else max)(
            range(len(rows)), key=lambda i: float(rows[i][obj]))

        note = "  [capacity table - capped 6 epochs]" if suffix in m["capped"] else ""
        ws.cell(r, 1, f"{title}  -  swept: {swept}  ({len(rows)} trials){note}")
        r += 1
        for c, h in enumerate(headers, 1):
            ws.cell(r, c, h)
        r += 1
        for i, row in enumerate(rows):
            vals = [fmt("trial", row.get("trial")), fmt("seed", row.get("seed", ""))]
            vals += [fmt(k, row.get(k, "")) for _, k in PARAMS]
            vals += [""]
            vals += [fmt(k, row.get(k, "")) for _, k in metrics]
            vals += ["best" if i == win_idx else ""]
            for c, v in enumerate(vals, 1):
                ws.cell(r, c, v)
            r += 1
        r += 2

    out = TUN / m["out"]
    wb.save(out)
    print(f"wrote {out}  ({len(TABLES)} tables, {total} trial rows)")


def main():
    for m in MODELS:
        build(m)


if __name__ == "__main__":
    main()
