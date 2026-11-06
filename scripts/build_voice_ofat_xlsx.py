"""Build a single .xlsx of all 10 adaptive-OFAT tables for the voice model.

Same layout as build_blink_ofat_xlsx.py, but voice is a 6-class classifier, so
the metric suite is the multiclass one (ROC-AUC one-vs-rest macro + Top-5
accuracy instead of the binary FPR/FNR/PR-AUC/Brier). Winner = highest accuracy.
P9 & P10 (capacity) were run at a capped 6 epochs and are annotated as such.

Reads ofat_voice_adaptive_P*_trials.csv. Writes voice_ofat_10tables.xlsx.
"""
from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
TUN = ROOT / "reports" / "tuning"
OUT = TUN / "voice_ofat_10tables.xlsx"
PREFIX = "ofat_voice_adaptive"
CAPPED = ("P9_conv_width", "P10_fc_width")

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
METRICS = [
    ("Accuracy", "accuracy"), ("Balanced Acc", "balanced_accuracy"),
    ("Precision (macro)", "precision_macro"), ("Recall (macro)", "recall_macro"),
    ("F1 (macro)", "f1_macro"), ("Precision (wtd)", "precision_weighted"),
    ("Recall (wtd)", "recall_weighted"), ("F1 (wtd)", "f1_weighted"),
    ("MCC", "mcc"), ("Cohen Kappa", "cohen_kappa"),
    ("ROC-AUC (OvR)", "roc_auc_ovr_macro"), ("Top-5 Acc", "top5_accuracy"),
    ("Log Loss", "log_loss"),
]
HEADERS = ["Trial", "Seed"] + [p[0] for p in PARAMS] + ["METRICS:"] + [m[0] for m in METRICS]

ACCENT = "6B3F6E"
ACCENT_FILL = PatternFill("solid", fgColor=ACCENT)
TITLE_FILL = PatternFill("solid", fgColor="F0E8F1")
WIN_FILL = PatternFill("solid", fgColor="FBEEDF")
METRIC_HDR_FILL = PatternFill("solid", fgColor="2E5A63")
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
    return round(f, 5)


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "Voice — 10 OFAT tables"
    ws.sheet_view.showGridLines = False

    r = 1
    ws.cell(r, 1, "Voice KWS CNN — Adaptive One-Factor-at-a-Time Hyperparameter Sweep").font = \
        Font(bold=True, size=14, color=ACCENT)
    r += 1
    ws.cell(r, 1, "10 tables (one per parameter). 6-class keyword spotting (up/down/stop/go/yes/no). "
                  "Full dataset. Winner = highest accuracy, highlighted. NOTE: P9 & P10 (capacity) "
                  "were run at a capped 6 epochs — see their table notes.").font = \
        Font(italic=True, size=9, color="5B655E")
    r += 2

    ncols = len(HEADERS)
    total = 0
    for suffix, title, swept in TABLES:
        path = TUN / f"{PREFIX}_{suffix}_trials.csv"
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        total += len(rows)
        win_idx = max(range(len(rows)), key=lambda i: float(rows[i]["accuracy"]))

        note = "   [capacity table — capped 6 epochs]" if suffix in CAPPED else ""
        ws.cell(r, 1, f"{title}   —   swept: {swept}   ({len(rows)} trials){note}").font = TITLE_FONT
        for c in range(1, ncols + 1):
            ws.cell(r, c).fill = TITLE_FILL
        r += 1

        for c, h in enumerate(HEADERS, 1):
            cell = ws.cell(r, c, h)
            cell.font = WHITE
            cell.fill = METRIC_HDR_FILL if (h in [m[0] for m in METRICS] or h == "METRICS:") else ACCENT_FILL
            cell.alignment = CENTER if c > 1 else LEFT
            cell.border = BORDER
        r += 1

        for i, row in enumerate(rows):
            is_win = (i == win_idx)
            vals = [fmt("trial", row.get("trial")), fmt("seed", row.get("seed", ""))]
            vals += [fmt(k, row.get(k, "")) for _, k in PARAMS]
            vals += [""]
            vals += [fmt(k, row.get(k, "")) for _, k in METRICS]
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
    wb.save(OUT)
    print(f"wrote {OUT}  ({len(TABLES)} tables, {total} trial rows)")


if __name__ == "__main__":
    main()
