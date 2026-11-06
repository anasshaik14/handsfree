"""Export every evaluation metric — all four modalities, every provider — into
ONE plain Excel sheet, organised into a section per modality.

Model-quality figures are read from the same JSONs the HTML report renders from
(models/custom/model_report.json, models/pretrained_report.json, and the
per-model models/custom/*.metrics.json). Runtime figures are re-aggregated from
every session in sessions/ at export time, using the report's own aggregator.

fusion_nn no longer exists in the app, so the report excludes it; it is included
here (runtime only — it was never scored on a dataset) from the historical logs.

    python scripts/export_metrics_xlsx.py [out.xlsx]
"""
from __future__ import annotations

import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.metrics import report  # noqa: E402

FONT = "Arial"
NA = "n/a"

# 0..1 fractions the report treats as percentages; everything else that is a
# 0..1 / -1..1 score stays a score, and degrees/ms stay in their own unit.
PERCENT_KEYS = {
    "accuracy", "balanced_accuracy", "error_rate", "precision_macro",
    "precision_weighted", "recall_macro", "recall_weighted", "specificity_macro",
    "f1_macro", "f1_weighted", "fhalf_macro", "f2_macro", "fpr_macro",
    "fnr_macro", "fdr_macro", "top5_accuracy", "acc_within_5deg",
    "acc_within_10deg", "acc_within_15deg",
}

CLS_COLS = [
    ("accuracy", "Accuracy (%)"),
    ("balanced_accuracy", "Balanced accuracy (%)"),
    ("error_rate", "Error rate (%)"),
    ("precision_macro", "Precision macro (%)"),
    ("precision_weighted", "Precision weighted (%)"),
    ("recall_macro", "Recall / sensitivity macro (%)"),
    ("recall_weighted", "Recall weighted (%)"),
    ("specificity_macro", "Specificity / TNR macro (%)"),
    ("f1_macro", "F1-score macro (%)"),
    ("f1_weighted", "F1-score weighted (%)"),
    ("fhalf_macro", "F0.5-score macro (%)"),
    ("f2_macro", "F2-score macro (%)"),
    ("jaccard_macro", "Jaccard / IoU macro (0-1)"),
    ("mcc", "Matthews corr. MCC (-1 to 1)"),
    ("cohen_kappa", "Cohen's kappa (-1 to 1)"),
    ("fpr_macro", "False-positive rate macro (%)"),
    ("fnr_macro", "False-negative rate macro (%)"),
    ("fdr_macro", "False-discovery rate macro (%)"),
    ("roc_auc", "AUC-ROC (0-1)"),
    ("pr_auc", "AUC-PR (0-1)"),
    ("roc_auc_ovr_macro", "AUC-ROC one-vs-rest macro (0-1)"),
    ("top5_accuracy", "Top-5 accuracy (%)"),
    ("inference_ms", "Model inference time (ms/frame)"),
]

REG_COLS = [
    ("acc_within_5deg", "Accuracy within 5° / PCK (%)"),
    ("acc_within_10deg", "Accuracy within 10° / PCK (%)"),
    ("acc_within_15deg", "Accuracy within 15° / PCK (%)"),
    ("mae_deg", "MAE (°)"),
    ("rmse_deg", "RMSE (°)"),
    ("median_ae_deg", "Median abs error (°)"),
    ("p90_ae_deg", "P90 abs error (°)"),
    ("angular_mean_deg", "Mean angular error (°)"),
    ("r2_best", "R² best axis (0-1)"),
    ("r2_worst", "R² worst axis (0-1)"),
    ("corr_best", "Correlation best axis (0-1)"),
    ("corr_worst", "Correlation worst axis (0-1)"),
    ("inference_ms", "Model inference time (ms/frame)"),
]

RUNTIME_COLS = [
    ("samples", "Live data (s)"),
    ("fps_mean", "FPS"),
    ("stage_ms_mean", "Stage latency (ms)"),
    ("stage_ms_p95", "Stage p95 latency (ms)"),
    ("pipeline_ms_mean", "Pipeline latency (ms)"),
    ("cpu_percent", "CPU (% of one core)"),
    ("memory_mb", "Memory (MB)"),
]
GAZE_RUNTIME_COLS = [
    ("jitter_px", "Fixation jitter (px RMS)"),
    ("calib_error_px", "Calibration error (px)"),
    ("calib_spread_px", "Calibration spread (px)"),
]

LEAD_COLS = ["Provider", "Model type", "Eval dataset", "Eval samples (N)"]

CUSTOM = "custom-trained CNN"
CLASSICAL = "classical / pretrained"

# (section title, kind, [(row label, runtime provider key, quality ref, type)])
# quality ref: ("reg"|"cls", key) for the custom report, ("pre", section, model)
# for the pretrained report, or None where the provider was never dataset-scored.
SECTIONS = [
    ("GAZE TRACKING", "reg", [
        ("iris_geometric", "iris_geometric", ("pre", "gaze", "iris_geometric"), CLASSICAL),
        ("iris_polynomial", "iris_polynomial", ("pre", "gaze", "iris_polynomial"), CLASSICAL),
        ("pupil_classical", "pupil_classical", ("pre", "gaze", "pupil_classical"), CLASSICAL),
        ("fusion_nn", "fusion_nn", None, "fusion (removed from app)"),
        ("custom", "custom", ("reg", "gaze"), CUSTOM),
        ("custom (app-pipeline scoring, see note 1)", None, ("pre", "gaze", "custom"), CUSTOM),
    ]),
    ("BLINK DETECTION", "cls", [
        ("ear_adaptive", "ear_adaptive", None, CLASSICAL),
        ("fusion_nn", "fusion_nn", None, "fusion (removed from app)"),
        ("custom", "custom", ("cls", "blink"), CUSTOM),
        ("ear_standard", "ear_standard", None, CLASSICAL),
    ]),
    ("HEAD POSE ESTIMATION", "reg", [
        ("solvepnp", "solvepnp", ("pre", "head pose", "solvepnp"), CLASSICAL),
        ("mp_geometry", "mp_geometry", None, CLASSICAL),
        ("fusion_nn", "fusion_nn", None, "fusion (removed from app)"),
        ("custom", "custom", ("reg", "headpose"), CUSTOM),
        ("landmark_heuristic (not a live provider)", None,
         ("pre", "head pose", "landmark_heuristic"), CLASSICAL),
    ]),
    ("VOICE COMMAND", "cls", [
        ("off", "off", None, "baseline (voice disabled)"),
        ("vosk", "vosk", ("pre", "voice", "vosk"), CLASSICAL),
        ("webspeech", "webspeech", None, CLASSICAL),
        ("whisper", "whisper", None, CLASSICAL),
        ("fusion_nn", "fusion_nn", None, "fusion (removed from app)"),
        ("custom", "custom", ("cls", "voice"), CUSTOM),
    ]),
]

NOTES = [
    "1. Gaze / custom: the two custom rows score the SAME CNN under two different "
    "protocols and are not comparable. The 'custom' row uses MPIIGaze's "
    "pre-normalised eye patches (its training domain, and the protocol the "
    "literature reports) — this is the headline 93.48% / 1.56°. The "
    "'app-pipeline scoring' row crops live from the original image via FaceMesh "
    "exactly as the running app does, and its runtime cost is the 'custom' row's. "
    "The gap between them is a genuine train/serve difference, not a scoring error.",
    "2. fusion_nn no longer exists in the app. Its runtime figures come from "
    "historical session logs (65-122 seconds of live data each, far less than the "
    "other providers), and it was never scored on a dataset, so it has no quality "
    "metrics.",
    "3. Blank quality cells are providers that cannot be dataset-scored rather "
    "than providers that scored zero. mp_geometry reports head pose relative to a "
    "per-user neutral face cloud, so it has no absolute output to score. The "
    "classical blink detectors (ear_standard, ear_adaptive) are geometric rules "
    "over FaceMesh eye landmarks, so scoring them needs full-face images labelled "
    "open/closed; every eye dataset on this machine holds cropped eyes with no "
    "surrounding face, and no full-face set here carries eye-state labels. "
    "webspeech and whisper are cloud/large ASR engines not run over the offline "
    "eval set.",
    "4. Runtime columns (live data through memory) are re-aggregated from every "
    "session in sessions/ at export time; model quality columns come from the "
    "fixed evaluation JSONs. Runtime averages include sessions recorded before "
    "the 2026-07-14 performance fix (single-threaded torch, throttled hand "
    "tracking, async object detection), so the custom providers' CPU and FPS here "
    "are worse than the current build achieves.",
    "5. Model inference time is re-benchmarked on this machine at export time "
    "(100 single-frame CPU passes), so it moves slightly between exports. It is "
    "the model's own cost; the runtime columns are the whole pipeline's.",
    "6. Gaze classical providers output a screen-space signal rather than an "
    "angle, so each gets the per-user calibration the app performs, fitted per "
    "participant on half their samples and scored on the held-out half. FaceMesh "
    "detected 2268/3000 faces on MPIIGaze (much of it near-dark) at a lowered 0.15 "
    "confidence; on AFLW2000 it detected 1749/1969, which is why the classical "
    "head-pose rows show N=1,749 against the CNN's 1,969.",
]


def val(key: str, m: dict):
    """One metric value in the unit its column header advertises."""
    v = m.get(key)
    if v is None:
        return NA
    v = float(v)
    return round(v * 100, 2) if key in PERCENT_KEYS else round(v, 4)


def main(out: str = "reports/handsfree_metrics.xlsx") -> Path:
    custom = report.enrich_custom_report(report._load_model_report())
    pre = report._load_pretrained_report()
    pre_sections = {s["title"].split(" — ")[0].lower(): s["models"]
                    for s in pre["sections"]}

    # ponytail: the report hides fusion_nn because the app dropped it; this export
    # wants it, and a module-level swap is enough in a one-shot script.
    report.EXCLUDE_PROVIDERS = set()
    agg = report._aggregate(report._load_samples(Path("sessions")))

    wb = Workbook()
    ws = wb.active
    ws.title = "All metrics"
    row = 1

    ws.cell(row=row, column=1,
            value="Hands-Free Control — evaluation metrics, all modalities, all providers")
    row += 1
    ws.cell(row=row, column=1, value=(
        f"Model quality: custom CNNs generated {custom.get('generated', '')}, "
        f"pretrained/classical generated {pre.get('generated', '')}. "
        "Runtime: re-aggregated from all sessions at export time. "
        "Blank/n/a = not measurable for that provider, not a zero — see notes at the bottom."))
    row += 2

    for title, kind, rows in SECTIONS:
        qcols = REG_COLS if kind == "reg" else CLS_COLS
        rcols = RUNTIME_COLS + (GAZE_RUNTIME_COLS if title.startswith("GAZE") else [])
        headers = LEAD_COLS + [h for _k, h in qcols] + [h for _k, h in rcols]

        ws.cell(row=row, column=1, value=title).font = Font(name=FONT, bold=True)
        row += 1
        for c, h in enumerate(headers, start=1):
            ws.cell(row=row, column=c, value=h).font = Font(name=FONT, bold=True)
        row += 1

        for label, rt_key, qref, mtype in rows:
            if qref is None:
                q, ds, n = {}, "", ""
            elif qref[0] == "pre":
                m = pre_sections[qref[1]][qref[2]]
                q, ds, n = m["metrics"], m.get("dataset", ""), m.get("n_samples", "")
            else:
                m = custom[qref[0]]["models"][qref[1]]
                q, ds, n = m["metrics"], m.get("dataset", ""), m.get("n_samples", "")
            rt = agg.get({"GAZE TRACKING": "gaze", "BLINK DETECTION": "blink",
                          "HEAD POSE ESTIMATION": "headpose",
                          "VOICE COMMAND": "voice"}[title], {}).get(rt_key or "", {})

            line = [label, mtype, ds or NA, n if n != "" else NA]
            line += [val(k, q) for k, _h in qcols]
            for k, _h in rcols:
                v = rt.get(k)
                line.append(NA if v is None else
                            int(v) if k == "samples" else round(float(v), 2))
            for c, v in enumerate(line, start=1):
                ws.cell(row=row, column=c, value=v)
            row += 1
        row += 1

    ws.cell(row=row, column=1, value="NOTES").font = Font(name=FONT, bold=True)
    row += 1
    for n in NOTES:
        ws.cell(row=row, column=1, value=n)
        row += 1

    for r in ws.iter_rows():
        for cell in r:
            if cell.font is None or not cell.font.bold:
                cell.font = Font(name=FONT)
    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 24
    for c in range(4, len(LEAD_COLS) + len(CLS_COLS) + len(RUNTIME_COLS)
                   + len(GAZE_RUNTIME_COLS) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 16
    ws.freeze_panes = "B1"

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


if __name__ == "__main__":
    p = main(*sys.argv[1:2])
    print(f"wrote {p.resolve()}")
