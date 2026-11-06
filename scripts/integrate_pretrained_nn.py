"""Fold the two genuinely-pretrained NN models (6DRepNet head pose, MPIIGaze
resnet-preact gaze) into the website's pretrained report AND a standalone Excel,
scoring them with the SAME canonical reg_metrics() the custom CNNs use.

Reads the raw (true,pred) arrays saved by the two eval scripts:
  reports/tuning/pretrained/headpose_6drepnet.arrays.npz
  reports/tuning/pretrained/gaze_mpiigaze.arrays.npz

- patches models/pretrained_report.json  (adds a row to the head-pose & gaze sections)
- writes reports/pretrained_models_metrics.xlsx  (pretrained NN vs custom, full suites)

Voice is already covered by the existing 'vosk' row in the report, so it is only
mirrored into the Excel, not re-added.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from model_report import facing_angular_deg, reg_metrics  # noqa: E402

PRE = ROOT / "reports" / "tuning" / "pretrained"
REPORT = ROOT / "models" / "pretrained_report.json"


def gaze_angular_deg(true_py_deg, pred_py_deg):
    """Per-sample 3D gaze angle (deg) from (N,2) [pitch,yaw] in degrees
    (ptgaze gaze_angles_to_vector convention — matches the 2.82° validation)."""
    def vec(a):
        p, y = np.radians(a[:, 0]), np.radians(a[:, 1])
        return -np.stack([np.cos(p) * np.sin(y), np.sin(p), np.cos(p) * np.cos(y)], 1)
    vt, vp = vec(true_py_deg), vec(pred_py_deg)
    return np.degrees(np.arccos(np.clip((vt * vp).sum(1), -1.0, 1.0)))


def headpose_metrics():
    a = np.load(PRE / "headpose_6drepnet.arrays.npz")
    t, p = a["true_deg"], a["pred_deg"]                 # (N,3) yaw,pitch,roll
    m = reg_metrics(t, p, ["yaw", "pitch", "roll"])     # PCK on per-axis mean err (matches section)
    m["angular_mean_deg"] = round(float(facing_angular_deg(t, p).mean()), 4)
    return {"title": "6DRepNet (pretrained NN)", "dataset": "AFLW2000",
            "n_samples": int(len(t)), "metrics": m}


def gaze_metrics():
    a = np.load(PRE / "gaze_mpiigaze.arrays.npz")
    t, p = a["true_deg"], a["pred_deg"]                 # (N,2) pitch,yaw
    ang = gaze_angular_deg(t, p)
    m = reg_metrics(t, p, ["pitch", "yaw"], angular=ang)  # PCK on angular err (matches custom gaze)
    return {"title": "MPIIGaze ResNet-preact (pretrained NN)",
            "dataset": "MPIIGaze (Normalized eye patches)",
            "n_samples": int(len(t)), "metrics": m}


def patch_report(hp, gz):
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    for sec in rep["sections"]:
        low = sec["title"].lower()
        if low.startswith("head pose"):
            sec["models"]["6drepnet"] = hp
        elif low.startswith("gaze"):
            sec["models"]["mpiigaze_resnet"] = gz
    REPORT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"patched {REPORT.name}: +6drepnet (head pose), +mpiigaze_resnet (gaze)")


def build_excel(hp, gz):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    custom = json.loads((ROOT / "models" / "custom" / "model_report.json").read_text())
    pre = json.loads(REPORT.read_text(encoding="utf-8"))
    pre_sec = {s["title"].lower().split(" ")[0]: s["models"] for s in pre["sections"]}

    REG = [("acc_within_5deg", "PCK@5° (%)"), ("acc_within_10deg", "PCK@10° (%)"),
           ("acc_within_15deg", "PCK@15° (%)"), ("mae_deg", "MAE (°)"),
           ("rmse_deg", "RMSE (°)"), ("median_ae_deg", "Median AE (°)"),
           ("p90_ae_deg", "P90 AE (°)"), ("angular_mean_deg", "Mean angular err (°)"),
           ("r2_best", "R² best"), ("r2_worst", "R² worst"),
           ("corr_best", "Corr best"), ("corr_worst", "Corr worst")]
    CLS = [("accuracy", "Accuracy (%)"), ("balanced_accuracy", "Balanced acc (%)"),
           ("precision_macro", "Precision macro (%)"), ("recall_macro", "Recall macro (%)"),
           ("f1_macro", "F1 macro (%)"), ("f1_weighted", "F1 weighted (%)"),
           ("mcc", "MCC"), ("cohen_kappa", "Cohen κ"),
           ("fpr_macro", "FPR macro (%)"), ("fnr_macro", "FNR macro (%)")]
    PCT = {k for k, _ in REG if "acc_within" in k} | {
        "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted", "fpr_macro", "fnr_macro"}

    def cell(k, m):
        v = m.get(k)
        if v is None:
            return "n/a"
        return round(float(v) * 100, 2) if k in PCT else round(float(v), 4)

    wb = Workbook(); ws = wb.active; ws.title = "Pretrained vs custom"
    B = Font(name="Arial", bold=True); N = Font(name="Arial")
    r = 1
    ws.cell(r, 1, "HandsFree — pretrained NN models vs custom CNNs (same metric code)").font = B
    r += 2

    def section(title, cols, rows):
        nonlocal r
        ws.cell(r, 1, title).font = B; r += 1
        head = ["Model", "Type", "Dataset", "N"] + [h for _, h in cols]
        for c, h in enumerate(head, 1):
            ws.cell(r, c, h).font = B
        r += 1
        for label, mtype, ds, n, met in rows:
            line = [label, mtype, ds, n] + [cell(k, met) for k, _ in cols]
            for c, v in enumerate(line, 1):
                ws.cell(r, c, v).font = N
            r += 1
        r += 1

    # head pose
    hc = custom["reg"]["models"]["headpose"]
    section("HEAD POSE (AFLW2000)", REG, [
        ("6DRepNet", "pretrained NN", hp["dataset"], hp["n_samples"], hp["metrics"]),
        ("custom CNN", "custom", hc["dataset"], hc["n_samples"], hc["metrics"]),
        ("solvepnp", "classical", pre_sec["head"]["solvepnp"]["dataset"],
         pre_sec["head"]["solvepnp"]["n_samples"], pre_sec["head"]["solvepnp"]["metrics"]),
    ])
    # gaze
    gc = custom["reg"]["models"]["gaze"]
    section("GAZE", REG, [
        ("MPIIGaze ResNet-preact", "pretrained NN", gz["dataset"], gz["n_samples"], gz["metrics"]),
        ("custom CNN", "custom", gc["dataset"], gc["n_samples"], gc["metrics"]),
    ])
    # voice
    vc = custom["cls"]["models"]["voice"]
    vosk = pre_sec["voice"]["vosk"]
    section("VOICE (Speech Commands v2)", CLS, [
        ("Vosk", "pretrained ASR", vosk["dataset"], vosk["n_samples"], vosk["metrics"]),
        ("custom CNN", "custom", vc["dataset"], vc["n_samples"], vc["metrics"]),
    ])

    ws.cell(r, 1, "Note: head pose (6DRepNet vs custom) is a true apples-to-apples test on AFLW2000. "
                  "Gaze pretrained is scored on MPIIGaze normalized eye patches (its and the custom "
                  "CNN's shared training domain); the classical gaze providers in the website report "
                  "use a different FaceMesh-crop protocol and are not directly comparable here.").font = N
    ws.column_dimensions["A"].width = 26
    for c in range(2, 4 + len(REG) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 15
    ws.freeze_panes = "A4"
    out = ROOT / "reports" / "pretrained_models_metrics.xlsx"
    wb.save(out)
    print(f"wrote {out}")


def main():
    hp, gz = headpose_metrics(), gaze_metrics()
    print("head pose 6DRepNet:", hp["metrics"]["mae_deg"], "° MAE, angular",
          hp["metrics"]["angular_mean_deg"], "° N", hp["n_samples"])
    print("gaze MPIIGaze:", gz["metrics"]["angular_mean_deg"], "° angular, PCK@5°",
          gz["metrics"]["acc_within_5deg"], "N", gz["n_samples"])
    patch_report(hp, gz)
    build_excel(hp, gz)


if __name__ == "__main__":
    main()
