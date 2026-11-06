"""Evaluate the pretrained Vosk speech engine on our 6-keyword Speech Commands
TEST split, producing the confusion-matrix-based classification metrics used for
our custom voice CNN, so the two are directly comparable.

Vosk outputs a hard transcript (a decoded word), NOT a probability distribution
over our 6 classes, so probability-based metrics (ROC-AUC, log loss, Brier,
Top-5) cannot be computed for it — those are reported as N/A. Everything derived
from the confusion matrix (accuracy, precision, recall, F1, MCC, kappa, etc.) is
computed identically to the custom model.

Writes reports/tuning/pretrained/voice_vosk.metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "tuning" / "pretrained"
WORDS = ["down", "go", "no", "stop", "up", "yes"]   # sorted, matches custom labels


def main():
    import vosk
    from scipy.io import wavfile

    base = ROOT / "data" / "speech_commands" / "SpeechCommands" / "speech_commands_v0.02"
    # use the official TEST split (touched once) over the 6 keywords
    test = [l for l in (base / "testing_list.txt").read_text().split()
            if l.split("/")[0] in WORDS]
    print(f"vosk on {len(test)} test clips ({len(WORDS)} keywords)...", flush=True)

    vosk.SetLogLevel(-1)
    model = vosk.Model(str(ROOT / "models" / "vosk-model-small-en-us-0.15"))
    grammar = json.dumps(WORDS + ["[unk]"])

    y_true, y_pred = [], []
    t0 = time.perf_counter()
    for i, rel in enumerate(test):
        w = rel.split("/")[0]
        sr, audio = wavfile.read(base / rel)
        rec = vosk.KaldiRecognizer(model, sr, grammar)
        rec.AcceptWaveform(audio.tobytes())
        text = json.loads(rec.FinalResult()).get("text", "").strip()
        y_true.append(WORDS.index(w))
        y_pred.append(WORDS.index(text) if text in WORDS else 6)   # 6 = reject
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(test)}", flush=True)
    ms = (time.perf_counter() - t0) / max(len(y_true), 1) * 1000

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels6 = list(range(6))
    # score the 6 real classes (reject col 6 is never a ground-truth label)
    out = {
        "n_samples": int(len(y_true)),
        "n_classes": 6,
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(skm.precision_score(y_true, y_pred, labels=labels6, average="macro", zero_division=0)),
        "recall_macro": float(skm.recall_score(y_true, y_pred, labels=labels6, average="macro", zero_division=0)),
        "f1_macro": float(skm.f1_score(y_true, y_pred, labels=labels6, average="macro", zero_division=0)),
        "precision_weighted": float(skm.precision_score(y_true, y_pred, labels=labels6, average="weighted", zero_division=0)),
        "recall_weighted": float(skm.recall_score(y_true, y_pred, labels=labels6, average="weighted", zero_division=0)),
        "f1_weighted": float(skm.f1_score(y_true, y_pred, labels=labels6, average="weighted", zero_division=0)),
        "mcc": float(skm.matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(skm.cohen_kappa_score(y_true, y_pred)),
        # probability-based metrics are undefined for a hard decoder:
        "roc_auc_ovr_macro": None,
        "top5_accuracy": None,
        "log_loss": None,
        "rejected": int((y_pred == 6).sum()),
        "inference_ms": round(ms, 1),
        "model": "Vosk small en-US (pretrained)",
        "test_set": "Speech Commands v2 (6-keyword test split)",
        "note": "Vosk gives a hard transcript, not class probabilities; ROC-AUC/log-loss/Top-5 are N/A.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "voice_vosk.metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nVosk: accuracy {out['accuracy']*100:.1f}%  F1 {out['f1_macro']:.3f}  "
          f"({out['n_samples']} clips, {out['rejected']} rejected, {out['inference_ms']} ms/clip)")
    print(f"wrote {OUT / 'voice_vosk.metrics.json'}")


if __name__ == "__main__":
    main()
