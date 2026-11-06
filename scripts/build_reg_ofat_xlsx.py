"""Build a single .xlsx of all 10 adaptive-OFAT tables for a REGRESSION model
(gaze or head pose). Same layout as build_blink_ofat_xlsx.py: one worksheet,
ten stacked table blocks, every hyperparameter per row + the full regression
metric suite. Winner = the trial with the lowest objective (angular error for
gaze, MAE for head pose).

Usage:
  python scripts/build_reg_ofat_xlsx.py gaze
  python scripts/build_reg_ofat_xlsx.py headpose
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
TUN = ROOT / "reports" / "tuning"

TABLES = [
    ("P1_learning_rate", "P1 — Learning Rate", "Learning Rate"),
    ("P2_optimizer", "P2 — Optimizer", "Optimizer"),
    ("P3_dropout", "P3 — Dropout", "Dropout"),
    ("P4_augmentation", "P4 — Augmentation", "Augmentation"),
    ("P5_batch_size", "P5 — Batch Size", "Batch size"),
    ("P6_lr_schedule", "P6 — LR Schedule", "LR Schedule"),
    ("P7_weight_decay", "P7 — Weight Decay", "Weight Decay"),
    ("P8_epochs", "P8 — Epochs", "Epochs"),
    ("P9_conv_width", "P9 — Conv Width", "Conv Width"),
    ("P10_fc_width", "P10 — FC Width", "FC Width"),
]
PARAMS = [
    ("Learning Rate", "lr"), ("Batch size", "batch"), ("Dropout", "dropout"),
    ("Weight Decay", "weight_decay"), ("Epochs", "epochs"), ("Optimizer", "opt"),
    ("LR Schedule", "sched"), ("Augmentation", "aug"), ("Conv Width", "width"),
    ("FC Width", "fc"),
]

MODELS = {
    "gaze": {
        "prefix": "ofat_gaze_adaptive",
        "out": "gaze_ofat_10tables.xlsx",
        "title": "Gaze CNN — Adaptive One-Factor-at-a-Time Hyperparameter Sweep",
        "subtitle": ("10 tables (one per parameter). Objective = mean angular error in degrees "
                     "(LOWER is better). Winner of each table highlighted. Real data; "
                     "participant-disjoint MPIIGaze split, 20k subset."),
        "objective": "angular_error_deg_mean",
        "lower_better": True,
        "capped_tables": (),
        "metrics": [
            ("MAE°", "mae_deg"), ("RMSE°", "rmse_deg"), ("Median AE°", "median_ae_deg"),
            ("Max AE°", "max_ae_deg"), ("P90 AE°", "p90_ae_deg"), ("P95 AE°", "p95_ae_deg"),
            ("MAE° pitch", "mae_deg_pitch"), ("RMSE° pitch", "rmse_deg_pitch"),
            ("Bias° pitch", "bias_deg_pitch"), ("Std° pitch", "std_deg_pitch"),
            ("R² pitch", "r2_pitch"), ("Corr pitch", "corr_pitch"),
            ("MAE° yaw", "mae_deg_yaw"), ("RMSE° yaw", "rmse_deg_yaw"),
            ("Bias° yaw", "bias_deg_yaw"), ("Std° yaw", "std_deg_yaw"),
            ("R² yaw", "r2_yaw"), ("Corr yaw", "corr_yaw"),
            ("Angular° mean", "angular_error_deg_mean"),
            ("Angular° median", "angular_error_deg_median"),
            ("Angular° p90", "angular_error_deg_p90"),
        ],
    },
    "headpose": {
        "prefix": "ofat_headpose_adaptive",
        "out": "headpose_ofat_10tables.xlsx",
        "title": "Head Pose CNN — Adaptive One-Factor-at-a-Time Hyperparameter Sweep",
        "subtitle": ("10 tables (one per parameter). Objective = MAE in degrees (LOWER is better). "
                     "Winner of each table highlighted. Real data; face-disjoint 300W-LP split, "
                     "20k subset. NOTE: P9 & P10 (capacity) were run at a capped 6 epochs — see "
                     "their table notes."),
        "objective": "mae_deg",
        "lower_better": True,
        "capped_tables": ("P9_conv_width", "P10_fc_width"),
        "metrics": [
            ("MAE°", "mae_deg"), ("RMSE°", "rmse_deg"), ("Median AE°", "median_ae_deg"),
            ("Max AE°", "max_ae_deg"), ("P90 AE°", "p90_ae_deg"), ("P95 AE°", "p95_ae_deg"),
            ("MAE° yaw", "mae_deg_yaw"), ("RMSE° yaw", "rmse_deg_yaw"),
            ("Bias° yaw", "bias_deg_yaw"), ("Std° yaw", "std_deg_yaw"),
            ("R² yaw", "r2_yaw"), ("Corr yaw", "corr_yaw"),
            ("MAE° pitch", "mae_deg_pitch"), ("RMSE° pitch", "rmse_deg_pitch"),
            ("Bias° pitch", "bias_deg_pitch"), ("Std° pitch", "std_deg_pitch"),
            ("R² pitch", "r2_pitch"), ("Corr pitch", "corr_pitch"),
            ("MAE° roll", "mae_deg_roll"), ("RMSE° roll", "rmse_deg_roll"),
            ("Bias° roll", "bias_deg_roll"), ("Std° roll", "std_deg_roll"),
            ("R² roll", "r2_roll"), ("Corr roll", "corr_roll"),
        ],
    },
}

ACCENT = "2E5A63"
ACCENT_FILL = PatternFill("solid", fgColor=ACCENT)
TITLE_FILL = PatternFill("solid", fgColor="E7EFEE")
WIN_FILL = PatternFill("solid", fgColor="FBEEDF")
METRIC_HDR_FILL = PatternFill("solid", fgColor="52466E")
WHITE = Font(color="FFFFFF", bold=True, size=10)
TITLE_FONT = Font(bold=True, size=12, color=ACCENT)
WIN_FONT = Font(bold=True, color="B5651D")
THIN = Side(style="thin", color="D8DEDA")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")
LEFT = Alignment(horizontal="left", vertical="center")


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
    return round(f, 4)


def build(model_key):
    m = MODELS[model_key]
    metrics = m["metrics"]
    headers = ["Trial", "Seed"] + [p[0] for p in PARAMS] + ["METRICS:"] + [d for d, _ in metrics]

    wb = Workbook()
    ws = wb.active
    ws.title = f"{model_key} — 10 OFAT tables"
    ws.sheet_view.showGridLines = False

    r = 1
    ws.cell(r, 1, m["title"]).font = Font(bold=True, size=14, color=ACCENT)
    r += 1
    ws.cell(r, 1, m["subtitle"]).font = Font(italic=True, size=9, color="5B655E")
    r += 2

    ncols = len(headers)
    obj = m["objective"]
    for suffix, title, swept in TABLES:
        path = TUN / f"{m['prefix']}_{suffix}_trials.csv"
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        win_idx = min(range(len(rows)), key=lambda i: float(rows[i][obj])) if m["lower_better"] \
            else max(range(len(rows)), key=lambda i: float(rows[i][obj]))

        note = "   [capacity table — run at capped 6 epochs]" if suffix in m["capped_tables"] else ""
        ws.cell(r, 1, f"{title}   —   swept: {swept}   ({len(rows)} trials){note}").font = TITLE_FONT
        for c in range(1, ncols + 1):
            ws.cell(r, c).fill = TITLE_FILL
        r += 1

        for c, h in enumerate(headers, 1):
            cell = ws.cell(r, c, h)
            cell.font = WHITE
            cell.fill = METRIC_HDR_FILL if (h in [d for d, _ in metrics] or h == "METRICS:") else ACCENT_FILL
            cell.alignment = CENTER if c > 1 else LEFT
            cell.border = BORDER
        r += 1

        for i, row in enumerate(rows):
            is_win = (i == win_idx)
            vals = [fmt("trial", row.get("trial")), fmt("seed", row.get("seed", ""))]
            vals += [fmt(k, row.get(k, "")) for _, k in PARAMS]
            vals += [""]
            vals += [fmt(k, row.get(k, "")) for _, k in metrics]
            for c, v in enumerate(vals, 1):
                cell = ws.cell(r, c, v)
                cell.border = BORDER
                cell.alignment = LEFT if c in (3, 8, 9) else CENTER
                if is_win:
                    cell.fill = WIN_FILL
                    cell.font = WIN_FONT
            if is_win:
                ws.cell(r, 1, f"{vals[0]} *")
            r += 1
        r += 2

    ws.column_dimensions["A"].width = 9
    ws.column_dimensions["B"].width = 6
    for c in range(3, ncols + 1):
        ws.column_dimensions[get_column_letter(c)].width = 13
    ws.freeze_panes = "A5"

    out = TUN / m["out"]
    wb.save(out)
    total = 0
    for suffix, _, _ in TABLES:
        with open(TUN / f"{m['prefix']}_{suffix}_trials.csv", encoding="utf-8") as fh:
            total += sum(1 for _ in fh) - 1
    print(f"wrote {out}  ({len(TABLES)} tables, {total} trial rows)")


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "gaze"
    build(key)
