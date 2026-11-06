"""Regenerate custom-CNN per-sample probabilities for the two classifiers (blink, voice)
on their exact reported test splits, so the fascinating figures can use real per-sample
data (calibration, ROC/PR, margins, confusion flow). Pretrained providers here (EAR maths
for blink, Vosk transcript for voice) expose no per-sample probabilities, so they enter the
figures as summary reference only — this script covers the custom side.

Writes reports/tuning/pretrained/matched_blink.npz  (prob (N,2), label (N,), classes)
       reports/tuning/pretrained/matched_voice.npz  (prob (N,6), label (N,), labels)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from app.ml import preprocess
import torch
torch.set_num_threads(1)
OUT = ROOT / "reports" / "tuning" / "pretrained"


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def blink():
    import tuning
    z = np.load(ROOT / "data/cache/mrl_eyes.npz", allow_pickle=True)
    imgs, labels, subjects = z["imgs"], z["labels"], z["subjects"]
    _, _, idx_te = tuning.group_split(subjects, (0.7, 0.15, 0.15), seed=0)
    x = preprocess.norm_gray(imgs[idx_te].astype(np.float32) / 255.0)[:, None]  # (N,1,32,32)
    net = torch.jit.load(str(ROOT / "models/custom/blink_cnn.pt"), map_location="cpu").eval()
    logits = []
    with torch.no_grad():
        xt = torch.from_numpy(np.ascontiguousarray(x))
        for i in range(0, len(xt), 4096):
            logits.append(net(xt[i:i + 4096]).numpy())
    prob = _softmax(np.concatenate(logits))
    lab = labels[idx_te].astype(int)
    classes = json.load(open(ROOT / "models/custom/blink_cnn.metrics.json"))["classes"]
    np.savez(OUT / "matched_blink.npz", prob=prob, label=lab, classes=np.array(classes, dtype=object))
    acc = (prob.argmax(1) == lab).mean()
    print(f"[blink] N={len(lab)} test acc={acc:.4f} classes={classes}", flush=True)


def voice():
    z = np.load(ROOT / "data/cache/voice_kwudsgyn_test.npz")
    X, Y = z["X"], z["Y"]                                   # (N,1,n_mels,T), (N,)
    net = torch.jit.load(str(ROOT / "models/custom/voice_kws.pt"), map_location="cpu").eval()
    logits = []
    with torch.no_grad():
        xt = torch.from_numpy(X.astype(np.float32))
        for i in range(0, len(xt), 1024):
            logits.append(net(xt[i:i + 1024]).numpy())
    prob = _softmax(np.concatenate(logits))
    lab = Y.astype(int)
    labels = json.load(open(ROOT / "models/custom/voice_kws.labels.json"))
    if isinstance(labels, dict):
        labels = [labels[str(i)] if str(i) in labels else labels[i] for i in range(len(labels))]
    np.savez(OUT / "matched_voice.npz", prob=prob, label=lab, labels=np.array(labels, dtype=object))
    acc = (prob.argmax(1) == lab).mean()
    print(f"[voice] N={len(lab)} test acc={acc:.4f} labels={list(labels)}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "blink"):
        blink()
    if which in ("all", "voice"):
        voice()
    print("done")
