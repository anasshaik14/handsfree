"""Build a single .xlsx of all 10 adaptive-OFAT tables for the blink model.

Layout matches the requested template: one worksheet, ten stacked table blocks
(one per swept parameter, P1..P10). Each row is one trial and carries EVERY
hyperparameter (not just the swept one) plus the full 16-metric classification
suite. The winning trial of each table is highlighted.

Reads reports/tuning/ofat_blink_adaptive_P*_trials.csv (the real sweep output).
Writes reports/tuning/blink_ofat_10tables.xlsx.
"""
from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
TUN = ROOT / "reports" / "tuning"
OUT = TUN / "blink_ofat_10tables.xlsx"

# (file suffix, display title, swept-parameter display name)
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

# parameter columns (display name -> csv key)
PARAMS = [
    ("Learning Rate", "lr"), ("Batch size", "batch"), ("Dropout", "dropout"),
    ("Weight Decay", "weight_decay"), ("Epochs", "epochs"), ("Optimizer", "opt"),
    ("LR Schedule", "sched"), ("Augmentation", "aug"), ("Conv Width", "width"),
    ("FC Width", "fc"),
]
# metric columns (display -> csv key), the full 16-field classification suite
METRICS = [
    ("Accuracy", "accuracy"), ("Balanced Acc", "balanced_accuracy"),
    ("Precision (macro)", "precision_macro"), ("Recall (macro)", "recall_macro"),
    ("F1 (macro)", "f1_macro"), ("Precision (wtd)", "precision_weighted"),
    ("Recall (wtd)", "recall_weighted"), ("F1 (wtd)", "f1_weighted"),
    ("MCC", "mcc"), ("Cohen Kappa", "cohen_kappa"),
    ("FPR", "false_positive_rate"), ("FNR", "false_negative_rate"),
    ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
    ("Brier", "brier_score"), ("Log Loss", "log_loss"),
]

HEADERS = ["Trial", "Seed"] + [p[0] for p in PARAMS] + ["METRICS →"] + [m[0] for m in METRICS]

# styling
ACCENT = "52466E"
ACCENT_FILL = PatternFill("solid", fgColor=ACCENT)
TITLE_FILL = PatternFill("solid", fgColor="EDEAF3")
WIN_FILL = PatternFill("solid", fgColor="FBEEDF")
METRIC_HDR_FILL = PatternFill("solid", fgColor="2E5A63")
WHITE = Font(color="FFFFFF", bold=True, size=10)
TITLE_FONT = Font(bold=True, size=12, color=ACCENT)
BOLD = Font(bold=True)
WIN_FONT = Font(bold=True, color="B5651D")
THIN = Side(style="thin", color="D8DEDA")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")
LEFT = Alignment(horizontal="left", vertical="center")


def fmt(key, val):
    """Human-friendly cell value."""
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
    if key in ("dropout",):
        return round(f, 4)
    return round(f, 5)


def load(suffix):
    path = TUN / f"ofat_blink_adaptive_{suffix}_trials.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "Blink — 10 OFAT tables"
    ws.sheet_view.showGridLines = False

    r = 1
    # top banner
    ws.cell(r, 1, "Blink CNN — Adaptive One-Factor-at-a-Time Hyperparameter Sweep")
    ws.cell(r, 1).font = Font(bold=True, size=14, color=ACCENT)
    r += 1
    ws.cell(r, 1, "10 tables (one per parameter). Each trial shows every hyperparameter + the full 16-metric "
                  "suite. Winner of each table highlighted. Real data; subject-disjoint MRL Eye split, 20k subset.")
    ws.cell(r, 1).font = Font(italic=True, size=9, color="5B655E")
    r += 2

    ncols = len(HEADERS)
    for suffix, title, swept in TABLES:
        rows = load(suffix)
        # winner = max accuracy in this table
        win_idx = max(range(len(rows)), key=lambda i: float(rows[i]["accuracy"]))

        # title bar
        ws.cell(r, 1, f"{title}   —   swept parameter: {swept}   ({len(rows)} trials)")
        ws.cell(r, 1).font = TITLE_FONT
        for c in range(1, ncols + 1):
            ws.cell(r, c).fill = TITLE_FILL
        r += 1

        # header row
        for c, h in enumerate(HEADERS, 1):
            cell = ws.cell(r, c, h)
            cell.font = WHITE
            cell.fill = METRIC_HDR_FILL if (h in [m[0] for m in METRICS] or h == "METRICS →") else ACCENT_FILL
            cell.alignment = CENTER if c > 1 else LEFT
            cell.border = BORDER
        r += 1

        # data rows
        for i, row in enumerate(rows):
            is_win = (i == win_idx)
            vals = [fmt("trial", row.get("trial")), fmt("seed", row.get("seed", ""))]
            vals += [fmt(k, row.get(k, "")) for _, k in PARAMS]
            vals += [""]  # METRICS separator col
            vals += [fmt(k, row.get(k, "")) for _, k in METRICS]
            for c, v in enumerate(vals, 1):
                cell = ws.cell(r, c, v)
                cell.border = BORDER
                cell.alignment = LEFT if c in (3, 8, 9) else CENTER  # LR/Optimizer/Schedule left
                if is_win:
                    cell.fill = WIN_FILL
                    cell.font = WIN_FONT
            if is_win:
                ws.cell(r, 1, f"{vals[0]} ★")
            r += 1
        r += 2  # gap between tables

    # column widths
    ws.column_dimensions["A"].width = 9
    ws.column_dimensions["B"].width = 6
    for c in range(3, len(HEADERS) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 13
    ws.freeze_panes = "A5"

    wb.save(OUT)
    print(f"wrote {OUT}")
    print(f"  {len(TABLES)} tables, {sum(len(load(s)) for s,_,_ in TABLES)} total trial rows")


if __name__ == "__main__":
    main()
