"""Automatic evaluation reports: charts + comparison tables from session logs.

Reads every sessions/*.jsonl, groups samples by active provider, aggregates
FPS / latency / CPU / memory / jitter / event counts, renders matplotlib
charts and a self-contained HTML report under reports/<timestamp>/.

Every quality metric is shown with its range and a plain-language rating
(e.g. "98.2% — Excellent") so a value can never masquerade as something it
isn't (a 256 px calibration error is labelled as an error in pixels, not as
"accuracy").
"""
from __future__ import annotations

import json
import logging
import statistics
import time
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MODALITY_KEYS = {
    "gaze": ("gaze_provider", "gaze"),
    "blink": ("blink_provider", "blink"),
    "headpose": ("headpose_provider", "headpose"),
    "voice": ("voice_engine", "voice"),
}

# providers that no longer exist in the app; old session logs still mention them
EXCLUDE_PROVIDERS = {"fusion_nn"}

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- formatting
def _rate(pct: float) -> str:
    """Qualitative rating for a 0-100 'higher is better' percentage."""
    return ("Outstanding" if pct >= 97 else "Excellent" if pct >= 90 else
            "Good" if pct >= 80 else "Fair" if pct >= 65 else
            "Poor" if pct >= 50 else "Weak")


def _chip(pct: float) -> str:
    cls = ("ok" if pct >= 90 else "mid" if pct >= 65 else "bad")
    return f"<span class='rate {cls}'>{_rate(pct)}</span>"


def _cell(kind: str, v) -> str:
    """One formatted table cell: value in its natural unit + range + rating."""
    if v is None:
        return "<td>—</td>"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return f"<td>{v}</td>"
    if kind == "pct":          # 0..1 fraction, higher is better -> %
        return f"<td><b>{v * 100:.1f}%</b>{_chip(v * 100)}</td>"
    if kind == "pct_low":      # 0..1 fraction, lower is better (error rates)
        return f"<td><b>{v * 100:.1f}%</b>{_chip(100 - v * 100)}</td>"
    if kind == "unit":         # 0..1 score (F1, AUC, IoU)
        return f"<td><b>{v:.3f}</b> <span class='of'>/ 1.0</span>{_chip(v * 100)}</td>"
    if kind == "signed":       # -1..1 (MCC, kappa)
        return f"<td><b>{v:.3f}</b> <span class='of'>/ 1.0</span>{_chip(max(v, 0) * 100)}</td>"
    if kind == "deg":
        return f"<td><b>{v:.2f}°</b></td>"
    if kind == "ms":
        return f"<td><b>{v:.2f} ms</b></td>"
    if kind == "score":        # plain 0..1-ish score, no rating (R², corr)
        return f"<td><b>{v:.3f}</b></td>"
    return f"<td>{v:g}</td>"


# label, range/meaning note, kind — for every model-quality metric key
QUALITY_METRICS = {
    "accuracy": ("Accuracy", "0–100%, higher is better", "pct"),
    "balanced_accuracy": ("Balanced accuracy", "0–100%, per-class mean recall", "pct"),
    "error_rate": ("Error rate", "0–100%, lower is better", "pct_low"),
    "precision_macro": ("Precision (macro)", "0–100%, higher is better", "pct"),
    "precision_weighted": ("Precision (weighted)", "0–100%, support-weighted", "pct"),
    "recall_macro": ("Recall / Sensitivity (macro)", "0–100%, higher is better", "pct"),
    "recall_weighted": ("Recall (weighted)", "0–100%, support-weighted", "pct"),
    "specificity_macro": ("Specificity / TNR (macro)", "0–100%, higher is better", "pct"),
    "f1_macro": ("F1-score (macro)", "0–1, harmonic mean of precision & recall", "unit"),
    "f1_weighted": ("F1-score (weighted)", "0–1, support-weighted", "unit"),
    "fhalf_macro": ("F0.5-score (macro)", "0–1, precision-weighted", "unit"),
    "f2_macro": ("F2-score (macro)", "0–1, recall-weighted", "unit"),
    "jaccard_macro": ("Jaccard / IoU (macro)", "0–1, higher is better", "unit"),
    "mcc": ("Matthews corr. (MCC)", "−1–1, higher is better", "signed"),
    "cohen_kappa": ("Cohen's kappa", "−1–1, chance-adjusted", "signed"),
    "fpr_macro": ("False-positive rate (macro)", "0–100%, lower is better", "pct_low"),
    "fnr_macro": ("False-negative rate (macro)", "0–100%, lower is better", "pct_low"),
    "fdr_macro": ("False-discovery rate (macro)", "0–100%, lower is better", "pct_low"),
    "roc_auc": ("AUC-ROC", "0–1, higher is better", "unit"),
    "roc_auc_ovr_macro": ("AUC-ROC (one-vs-rest macro)", "0–1, higher is better", "unit"),
    "pr_auc": ("AUC-PR", "0–1, higher is better", "unit"),
    "top5_accuracy": ("Top-5 accuracy", "0–100%, higher is better", "pct"),
    "inference_ms": ("Inference time", "ms per frame, CPU, lower is better", "ms"),
    "acc_within_5deg": ("Accuracy within 5°", "0–100% of predictions (PCK)", "pct"),
    "acc_within_10deg": ("Accuracy within 10°", "0–100% of predictions (PCK)", "pct"),
    "acc_within_15deg": ("Accuracy within 15°", "0–100% of predictions (PCK)", "pct"),
    "mae_deg": ("Mean abs error", "degrees, 0° is perfect", "deg"),
    "rmse_deg": ("RMSE", "degrees, 0° is perfect", "deg"),
    "median_ae_deg": ("Median abs error", "degrees, 0° is perfect", "deg"),
    "p90_ae_deg": ("P90 abs error", "degrees, 0° is perfect", "deg"),
    "angular_mean_deg": ("Mean angular error", "degrees, 3D direction vector", "deg"),
    "r2_best": ("R² (best axis)", "1.0 is perfect fit", "score"),
    "r2_worst": ("R² (worst axis)", "1.0 is perfect fit", "score"),
    "corr_best": ("Correlation (best axis)", "1.0 is perfect", "score"),
    "corr_worst": ("Correlation (worst axis)", "1.0 is perfect", "score"),
}


def _load_samples(session_dir: Path) -> list[dict]:
    samples = []
    for f in sorted(session_dir.glob("*.jsonl")):
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        if d.get("type") == "sample":
                            samples.append(d)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return samples


def _mean(vals: list[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(statistics.fmean(vals), 2) if vals else None


def _aggregate(samples: list[dict]) -> dict[str, dict[str, dict]]:
    """{modality: {provider: {metric: value-or-None}}}"""
    out: dict[str, dict[str, dict]] = {}
    for modality, (ctx_key, stage_key) in MODALITY_KEYS.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            prov = s.get("ctx", {}).get(ctx_key)
            if prov and prov not in EXCLUDE_PROVIDERS:
                groups[prov].append(s)
        table: dict[str, dict] = {}
        for prov, rows in groups.items():
            fps = [r.get("fps", 0) for r in rows if r.get("fps", 0) > 0]
            stage_mean = [r["stages_ms"][stage_key]["mean"] for r in rows
                          if stage_key in r.get("stages_ms", {})]
            stage_p95 = [r["stages_ms"][stage_key]["p95"] for r in rows
                         if stage_key in r.get("stages_ms", {})]
            total_mean = [r["stages_ms"]["total"]["mean"] for r in rows
                          if "total" in r.get("stages_ms", {})]
            cpu = [r.get("cpu_percent") for r in rows if r.get("cpu_percent", 0) > 0]
            mem = [r.get("memory_mb") for r in rows if r.get("memory_mb", 0) > 0]
            jitter = [r["gauges"]["jitter_px"] for r in rows
                      if r.get("gauges", {}).get("jitter_px", -1) >= 0]
            entry = {
                "samples": len(rows),
                "fps_mean": _mean(fps),
                "stage_ms_mean": _mean(stage_mean),
                "stage_ms_p95": _mean(stage_p95),
                "pipeline_ms_mean": _mean(total_mean),
                "cpu_percent": _mean(cpu),
                "memory_mb": _mean(mem),
            }
            if modality == "gaze":
                entry["jitter_px"] = _mean(jitter)
                acc = [r["gauges"]["calib_accuracy_px"] for r in rows
                       if r.get("gauges", {}).get("calib_accuracy_px", 0) > 0]
                prec = [r["gauges"]["calib_precision_px"] for r in rows
                        if r.get("gauges", {}).get("calib_precision_px", 0) > 0]
                entry["calib_error_px"] = _mean(acc)
                entry["calib_spread_px"] = _mean(prec)
            table[prov] = entry
        if table:
            out[modality] = table
    return out


# ------------------------------------------------------------- model quality
def _load_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_model_report() -> Optional[dict]:
    """The trained-CNN quality table, written by scripts/model_report.py."""
    return _load_json(ROOT / "models" / "custom" / "model_report.json")


def _load_pretrained_report() -> Optional[dict]:
    """Classical/pretrained provider metrics, written by scripts/eval_pretrained.py."""
    return _load_json(ROOT / "models" / "pretrained_report.json")


def _bench_inference_ms() -> dict[str, float]:
    """Measure each custom model's single-frame CPU inference time (ms)."""
    from app.ml import preprocess, runtime
    shapes = {
        "blink": (1, *preprocess.EYE_SIZE),
        "gaze": (1, *preprocess.GAZE_EYE_SIZE),
        "headpose": (3, *preprocess.FACE_SIZE),
        "voice": (1, 40, 81),
    }
    files = {"blink": "blink_cnn.pt", "gaze": "gaze_cnn.pt",
             "headpose": "headpose_cnn.pt", "voice": "voice_kws.pt"}
    out: dict[str, float] = {}
    if not runtime.torch_available():
        return out
    import numpy as np
    for name, fn in files.items():
        model = runtime.load_scripted(runtime.MODELS_DIR / fn)
        if model is None:
            continue
        x = np.zeros(shapes[name], dtype=np.float32)
        for _ in range(10):
            runtime.infer(model, x)
        t0 = time.perf_counter()
        n = 100
        for _ in range(n):
            runtime.infer(model, x)
        out[name] = round((time.perf_counter() - t0) / n * 1000.0, 2)
    return out


def enrich_custom_report(rep: dict) -> dict:
    """Merge the metrics that live outside the aggregate report into it: measured
    inference time, and the AUC/top-k scores saved in the per-model metrics
    files. Extends each order list to match. Mutates and returns rep; idempotent."""
    bench = _bench_inference_ms()
    cls = rep.get("cls", {})
    for c, fn in (("blink", "blink_cnn"), ("voice", "voice_kws")):
        if c not in cls.get("models", {}):
            continue
        m = cls["models"][c]["metrics"]
        m.setdefault("inference_ms", bench.get(c))
        saved = _load_json(ROOT / "models" / "custom" / f"{fn}.metrics.json") or {}
        for k in ("roc_auc", "pr_auc", "roc_auc_ovr_macro", "top5_accuracy"):
            if k in saved:
                m.setdefault(k, saved[k])
    if cls.get("models"):
        extra = ["roc_auc", "pr_auc", "roc_auc_ovr_macro", "top5_accuracy",
                 "inference_ms"]
        cls["order"] = list(cls["order"]) + [k for k in extra
                                             if k not in cls["order"]]
    reg = rep.get("reg", {})
    for c in reg.get("models", {}):
        reg["models"][c]["metrics"].setdefault("inference_ms", bench.get(c))
    if reg.get("models") and "inference_ms" not in reg["order"]:
        reg["order"] = list(reg["order"]) + ["inference_ms"]
    return rep


def _metric_rows(order: list[str], models: dict, cols: list[str]) -> str:
    rows = ""
    for k in order:
        if not any(models[c]["metrics"].get(k) is not None for c in cols):
            continue
        lab, note, kind = QUALITY_METRICS.get(k, (k, "", "raw"))
        cells = "".join(_cell(kind, models[c]["metrics"].get(k)) for c in cols)
        rows += f"<tr><td class='m'>{lab}<span>{note}</span></td>{cells}</tr>"
    return rows


def _confusion_binary(cm: list, classes: list[str]) -> str:
    """2x2 confusion matrix as TP/FP/TN/FN (positive class = classes[1])."""
    (tn, fp), (fn, tp) = cm
    pos, neg = classes[1], classes[0]
    return f"""
<table class='cm'>
<tr><th></th><th>Predicted {neg}</th><th>Predicted {pos}</th></tr>
<tr><td class='m'>Actual {neg}</td><td class='good'>TN {tn:,}</td><td class='bad'>FP {fp:,}</td></tr>
<tr><td class='m'>Actual {pos}</td><td class='bad'>FN {fn:,}</td><td class='good'>TP {tp:,}</td></tr>
</table>
<p class='note'>True Positives {tp:,} · False Positives {fp:,} ·
True Negatives {tn:,} · False Negatives {fn:,}
(positive class = “{pos}”)</p>"""


def _confusion_multi(cm: list, labels: list[str]) -> str:
    head = "".join(f"<th>{l}</th>" for l in labels)
    rows = ""
    for i, l in enumerate(labels):
        cells = "".join(
            f"<td class='{'good' if i == j else ('bad' if cm[i][j] else '')}'>{cm[i][j]}</td>"
            for j in range(len(labels)))
        rows += f"<tr><td class='m'>{l}</td>{cells}</tr>"
    return (f"<table class='cm'><tr><th>actual \\ predicted</th>{head}</tr>{rows}</table>"
            "<p class='note'>Diagonal = correct predictions; everything else is a confusion.</p>")


def model_quality_html(rep: Optional[dict] = None) -> str:
    """Render the custom-CNN quality tables (percentages + ratings + ranges),
    confusion matrices, AUC and measured inference time. Classifiers (blink,
    voice) get the classification suite; regressors (gaze, headpose) get
    regression error plus tolerance/PCK accuracy. Returns "" when no report
    exists yet."""
    if rep is None:
        rep = _load_model_report()
    if not rep or "headline" not in rep:
        return ""
    enrich_custom_report(rep)

    hl = rep["headline"]
    hcols = [c for c in ("blink", "gaze", "headpose", "voice") if c in hl]
    hhead = "".join(f"<th>{hl[c]['title']}</th>" for c in hcols)
    hrow = "".join(
        f"<td><b>{hl[c]['accuracy'] * 100:.1f}%</b>{_chip(hl[c]['accuracy'] * 100)}"
        f"<span>{hl[c]['kind']}</span></td>" for c in hcols)
    headline_tbl = (f"<table><tr><th>Headline</th>{hhead}</tr>"
                    f"<tr><td class='m'>Accuracy<span>0–100%, higher is better</span></td>"
                    f"{hrow}</tr></table>")

    out = ["<h2>Model quality — custom-trained CNNs</h2>",
           "<p class='note'>Blink and voice are classifiers (accuracy = fraction of "
           "correct predictions). Gaze and head pose are regressors, so their accuracy "
           "is <b>PCK</b> — the fraction of predictions within a degree tolerance of "
           "ground truth (the standard accuracy metric for gaze/pose). "
           f"Generated {rep.get('generated', '')}.</p>",
           headline_tbl]

    cls = rep.get("cls", {})
    ccols = [c for c in ("blink", "voice") if c in cls.get("models", {})]
    if ccols:
        head = "".join(f"<th>{cls['models'][c]['title']}</th>" for c in ccols)
        sub = "".join(f"<td>{cls['models'][c]['classes']}<span>"
                      f"{cls['models'][c]['n_samples']:,} val · {cls['models'][c]['dataset']}</span></td>"
                      for c in ccols)
        order = cls["order"]
        out.append("<h3>Classifiers — full metric suite</h3>")
        out.append(f"<table><tr><th>Metric</th>{head}</tr>"
                   f"<tr><td class='m'>Classes<span>what it outputs</span></td>{sub}</tr>"
                   + _metric_rows(order, cls["models"], ccols) + "</table>")

    reg = rep.get("reg", {})
    rcols = [c for c in ("gaze", "headpose") if c in reg.get("models", {})]
    if rcols:
        head = "".join(f"<th>{reg['models'][c]['title']}</th>" for c in rcols)
        sub = "".join(f"<td>{reg['models'][c]['n_samples']:,} val"
                      f"<span>{reg['models'][c]['dataset']}</span></td>" for c in rcols)
        order = reg["order"]
        out.append("<h3>Regressors — error &amp; tolerance accuracy</h3>")
        out.append(f"<table><tr><th>Metric</th>{head}</tr>"
                   f"<tr><td class='m'>Eval set<span></span></td>{sub}</tr>"
                   + _metric_rows(order, reg["models"], rcols) + "</table>")

    # confusion matrices from the saved per-model metrics
    blink_m = _load_json(ROOT / "models" / "custom" / "blink_cnn.metrics.json")
    if blink_m and "confusion_matrix" in blink_m:
        classes = sorted(blink_m.get("classes", {"open": 0, "closed": 1}),
                         key=blink_m.get("classes", {"open": 0, "closed": 1}).get)
        out.append("<h3>Blink — confusion matrix</h3>")
        out.append(_confusion_binary(blink_m["confusion_matrix"], classes))
    voice_m = _load_json(ROOT / "models" / "custom" / "voice_kws.metrics.json")
    if voice_m and "confusion_matrix" in voice_m:
        out.append("<h3>Voice — confusion matrix</h3>")
        out.append(_confusion_multi(voice_m["confusion_matrix"],
                                    voice_m.get("labels", [])))
    return "".join(out)


def pretrained_quality_html(rep: Optional[dict] = None) -> str:
    """Render the pretrained/classical provider quality tables (same formatting
    as the custom models, evaluated on the same public datasets)."""
    if rep is None:
        rep = _load_pretrained_report()
    if not rep:
        return ""
    out = ["<h2>Model quality — pretrained &amp; classical providers</h2>",
           "<p class='note'>The classical/pretrained alternatives, scored on the "
           "same public datasets as the custom CNNs so the columns are directly "
           f"comparable. Generated {rep.get('generated', '')}.</p>"]
    for section in rep.get("sections", []):
        models = section["models"]
        cols = list(models)
        head = "".join(f"<th>{models[c]['title']}</th>" for c in cols)
        sub = "".join(f"<td>{models[c].get('n_samples', 0):,} val"
                      f"<span>{models[c].get('dataset', '')}</span></td>" for c in cols)
        out.append(f"<h3>{section['title']}</h3>")
        if section.get("note"):
            out.append(f"<p class='note'>{section['note']}</p>")
        out.append(f"<table><tr><th>Metric</th>{head}</tr>"
                   f"<tr><td class='m'>Eval set<span></span></td>{sub}</tr>"
                   + _metric_rows(section["order"], models, cols) + "</table>")
        cm = section.get("confusion")
        if cm:
            out.append(f"<h3>{cm['title']}</h3>")
            out.append(_confusion_multi(cm["matrix"], cm["labels"]))
    for note in rep.get("notes", []):
        out.append(f"<p class='note'>{note}</p>")
    return "".join(out)


# ------------------------------------------------------------------- charts
def _bar_chart(path: Path, title: str, ylabel: str,
               labels: list[str], values: list[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=110)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][:len(labels)]
    ax.bar(labels, values, color=colors, width=0.55)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _timeseries_chart(path: Path, samples: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = samples[-600:]
    if not rows:
        return
    t0 = rows[0]["ts"]
    ts = [(r["ts"] - t0) / 60.0 for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), dpi=110, sharex=True)
    axes[0].plot(ts, [r.get("fps", 0) for r in rows], color="#4C72B0", lw=1.2)
    axes[0].set_ylabel("FPS", fontsize=9)
    axes[1].plot(ts, [r.get("cpu_percent", 0) for r in rows], color="#DD8452", lw=1.2)
    axes[1].set_ylabel("CPU %", fontsize=9)
    axes[1].set_xlabel("minutes", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Session timeline", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# label, note, unit suffix for the runtime (live session) tables
_RUNTIME_LABELS = {
    "samples": ("Samples", "seconds of live data", " s"),
    "fps_mean": ("FPS", "frames per second, higher is better", ""),
    "stage_ms_mean": ("Stage latency", "ms per frame for this stage, lower is better", " ms"),
    "stage_ms_p95": ("Stage p95 latency", "ms, 95th percentile, lower is better", " ms"),
    "pipeline_ms_mean": ("Pipeline latency", "ms per full frame, lower is better", " ms"),
    "cpu_percent": ("CPU", "% of one core, lower is better", "%"),
    "memory_mb": ("Memory", "MB resident, lower is better", " MB"),
    "jitter_px": ("Fixation jitter", "px RMS while looking at one spot, lower is better", " px"),
    "calib_error_px": ("Calibration error", "px between gaze point and target, lower is better", " px"),
    "calib_spread_px": ("Calibration spread", "px repeatability of fixations, lower is better", " px"),
}


def generate(session_dir: str | Path = "sessions",
             out_root: str | Path = "reports",
             open_in_browser: bool = True) -> Path:
    session_dir = Path(session_dir)
    out = Path(out_root) / time.strftime("report_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(session_dir)
    agg = _aggregate(samples)

    charts: list[str] = []
    for modality, table in agg.items():
        provs = list(table)
        for metric in ("fps_mean", "stage_ms_mean", "cpu_percent", "jitter_px",
                       "calib_error_px"):
            vals = [table[p].get(metric) or 0 for p in provs]
            if not any(vals):
                continue
            label, note, _unit = _RUNTIME_LABELS[metric]
            fname = f"{modality}_{metric}.png"
            _bar_chart(out / fname, f"{modality}: {label} ({note})", label,
                       provs, vals)
            charts.append(fname)
    _timeseries_chart(out / "timeline.png", samples)
    if (out / "timeline.png").exists():
        charts.append("timeline.png")

    # ---- HTML ----------------------------------------------------------------
    def table_html(modality: str, table: dict) -> str:
        provs = list(table)
        metrics = [m for m in _RUNTIME_LABELS if any(m in table[p] for p in provs)]
        head = "".join(f"<th>{p}</th>" for p in provs)
        rows = ""
        for m in metrics:
            label, note, unit = _RUNTIME_LABELS[m]
            cells = ""
            for p in provs:
                v = table[p].get(m)
                cells += "<td>—</td>" if v is None else f"<td><b>{v:g}{unit}</b></td>"
            rows += f"<tr><td class='m'>{label}<span>{note}</span></td>{cells}</tr>"
        return (f"<h3>{modality.title()} — provider comparison</h3>"
                f"<table><tr><th>Metric</th>{head}</tr>{rows}</table>")

    runtime_body = "".join(table_html(m, t) for m, t in agg.items())
    if runtime_body:
        runtime_body = ("<h2>Runtime performance — live providers</h2>"
                        "<p class='note'>Measured on this machine from real usage "
                        "sessions. These are speed/resource numbers, not model "
                        "quality — see the model-quality tables above for "
                        "accuracy. Averages include sessions recorded before the "
                        "2026-07-14 performance fix (single-threaded torch, "
                        "throttled hand tracking, async object detection) — the "
                        "custom providers' historical CPU/FPS here are far worse "
                        "than the current build; click NEW SESSION on the "
                        "dashboard and use the app to measure the fixed build.</p>"
                        + runtime_body)
    else:
        runtime_body = "<p>No session data yet — run the app for a while first.</p>"
    body = model_quality_html() + pretrained_quality_html() + runtime_body
    imgs = "".join(f'<img src="{c}" alt="{c}">' for c in charts)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Hands-Free Control — Evaluation Report</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:2rem auto;
      padding:0 1rem;color:#1a1a2e}}
 h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:2.2rem}}
 h3{{font-size:1rem;margin-top:1.6rem}}
 table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:.4rem 0}}
 th,td{{border:1px solid #d5d5e0;padding:.45rem .6rem;text-align:center}}
 th{{background:#eef0f7}} td.m{{text-align:left;font-weight:600}}
 td.m span{{display:block;font-weight:400;color:#888;font-size:.75rem}}
 td span.of{{color:#999;font-size:.78rem}}
 td span{{display:block;font-weight:400;color:#888;font-size:.72rem}}
 .rate{{display:block;font-size:.72rem;font-weight:600;border-radius:8px;
       padding:0 .4rem;margin-top:.15rem}}
 .rate.ok{{color:#0a7a3d}} .rate.mid{{color:#a6690a}} .rate.bad{{color:#b02a2a}}
 .cm td.good{{background:#e8f7ee;font-weight:600}} .cm td.bad{{background:#fdeaea}}
 p.note{{color:#555;font-size:.85rem}}
 img{{max-width:100%;margin:.8rem 0;border:1px solid #e2e2ee;border-radius:6px}}
</style></head><body>
<h1>Hands-Free Control Mechanism — Evaluation Report</h1>
<p>Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from
{len(samples)} one-second samples across {len(list(session_dir.glob('*.jsonl')))} sessions.</p>
{body}
<h2>Charts</h2>{imgs}
</body></html>"""
    index = out / "index.html"
    index.write_text(html, encoding="utf-8")
    log.info("report written to %s", index)
    if open_in_browser:
        webbrowser.open(index.resolve().as_uri())
    return index
