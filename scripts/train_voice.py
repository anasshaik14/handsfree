"""Train the custom voice keyword-spotting CNN on Google Speech Commands v2.

SPLIT PROTOCOL: the OFFICIAL Speech Commands split — validation_list.txt and
testing_list.txt are already speaker-aware (a speaker's clips never straddle
splits), so no custom grouping is needed. Hyperparameters are selected on the
validation split; testing_list.txt is read once, at the end (--test).

Dataset is downloaded automatically by torchaudio (~2.3GB the first time).

Usage:
  python scripts/train_voice.py --keywords up,down,stop,go,yes,no
  python scripts/train_voice.py --keywords ... --search 20
  python scripts/train_voice.py --keywords ... --test
  python scripts/train_voice.py --smoke

Saves models/custom/voice_kws.pt + voice_kws.labels.json + voice_kws.metrics.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset, TensorDataset  # noqa: E402

from app.ml.torch_models import VoiceKWSCNN  # noqa: E402
from app.voice.custom_kws import (LABELS_PATH, N_MELS, SAMPLE_RATE,  # noqa: E402
                                  WEIGHTS_PATH)
import metrics  # noqa: E402
import tuning  # noqa: E402

MODELS_DIR = WEIGHTS_PATH.parent
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

SPACE = {
    "opt": ["adam", "adamw", "sgd"],
    "lr": ("loguniform", 1e-4, 3e-3),
    "weight_decay": ("loguniform", 1e-6, 1e-3),
    "dropout": ("uniform", 0.1, 0.5),
    "width": [16, 32],
    "fc": [64, 128],
    "sched": ["step", "cosine", "none"],
    "aug": [0, 1],
    "batch": [64, 128],
}
BASELINE = {"opt": "adam", "lr": 1e-3, "weight_decay": 0.0, "dropout": 0.3,
            "width": 16, "fc": 64, "sched": "step", "aug": 0, "batch": 128}


def _base_dir(root):
    return Path(root) / "SpeechCommands" / "speech_commands_v0.02"


def _split_paths(root, which):
    base = _base_dir(root)
    val = set((base / "validation_list.txt").read_text().split())
    test = set((base / "testing_list.txt").read_text().split())
    if which == "validation":
        rel = sorted(val)
    elif which == "testing":
        rel = sorted(test)
    else:
        allw = {f"{p.parent.name}/{p.name}" for p in base.glob("*/*.wav")
                if not p.parent.name.startswith("_")}
        rel = sorted(allw - val - test)
    return [str(base / r) for r in rel]


def _labels_from_dirs(root):
    return sorted(d.name for d in _base_dir(root).iterdir()
                  if d.is_dir() and not d.name.startswith("_"))


def _logmel_fn():
    import torchaudio
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=400, hop_length=200, n_mels=N_MELS)

    def logmel(wav: torch.Tensor) -> torch.Tensor:
        if wav.shape[-1] < SAMPLE_RATE:
            wav = torch.nn.functional.pad(wav, (0, SAMPLE_RATE - wav.shape[-1]))
        else:
            wav = wav[..., :SAMPLE_RATE]
        return torch.log(melspec(wav) + 1e-6)   # (1, n_mels, 81)

    return logmel


def _precompute(paths, label_idx, tag=""):
    """Read each wav once, cache its log-mel. Persisted to .npz per split/tag so
    a search doesn't re-decode the wavs on every trial."""
    cache = CACHE_DIR / f"voice_{tag}.npz" if tag else None
    if cache and cache.exists():
        z = np.load(cache)
        return TensorDataset(torch.from_numpy(z["X"]), torch.from_numpy(z["Y"]))
    from scipy.io import wavfile
    logmel = _logmel_fn()
    X = torch.empty(len(paths), 1, N_MELS, 81)
    Y = torch.empty(len(paths), dtype=torch.long)
    for i, p in enumerate(paths):
        _sr, data = wavfile.read(p)
        wav = torch.from_numpy(data.astype(np.float32) / 32768.0)
        if wav.ndim > 1:
            wav = wav.mean(1)
        X[i] = logmel(wav.unsqueeze(0))
        Y[i] = label_idx[Path(p).parent.name]
        if (i + 1) % 10000 == 0:
            print(f"  preprocessed {i + 1}/{len(paths)}", flush=True)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X.numpy(), Y=Y.numpy())
    return TensorDataset(X, Y)


class _AugWrap(Dataset):
    """SpecAugment on log-mels: time & frequency masking + small time shift.
    Applied to the TRAIN tensors only; masks fill with the batch's log-mel
    floor. Off during eval."""

    def __init__(self, base: TensorDataset):
        self.base = base
        self.aug = False

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if not self.aug:
            return x, y
        x = x.clone()
        _, n_mels, t = x.shape
        floor = float(x.min())
        if random.random() < 0.5:                        # time shift
            s = random.randint(-8, 8)
            x = torch.roll(x, shifts=s, dims=-1)
        if random.random() < 0.5:                        # freq mask
            f = random.randint(0, 8)
            f0 = random.randint(0, max(0, n_mels - f))
            x[:, f0:f0 + f, :] = floor
        if random.random() < 0.5:                        # time mask
            tm = random.randint(0, 12)
            t0 = random.randint(0, max(0, t - tm))
            x[:, :, t0:t0 + tm] = floor
        return x, y


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return correct / max(1, total)


def train_one(cfg, train_base, val_dl, n_classes, device, epochs, seed=0, patience=4, verbose=False):
    tuning.seed_all(seed)
    train_wrap = _AugWrap(train_base)
    train_dl = DataLoader(train_wrap, batch_size=int(cfg["batch"]), shuffle=True)
    model = VoiceKWSCNN(n_classes, width=int(cfg["width"]), fc=int(cfg["fc"]),
                        dropout=float(cfg["dropout"])).to(device)
    opt = tuning.make_optimizer(cfg["opt"], model.parameters(), float(cfg["lr"]),
                                float(cfg["weight_decay"]))
    sched = tuning.make_sched(cfg["sched"], opt, epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state, bad = 0.0, None, 0
    for epoch in range(epochs):
        train_wrap.aug = bool(cfg["aug"])
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
        if sched:
            sched.step()
        acc = evaluate(model, val_dl, device)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  val_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


def _build_splits(args, rng):
    """Return (train_base TensorDataset, val_base, test_base|None, labels)."""
    all_words = _labels_from_dirs(args.root)
    kws = [w.strip() for w in args.keywords.split(",") if w.strip()] if args.keywords else all_words
    labels = sorted(kws) + (["unknown"] if (args.keywords and args.unknown) else [])
    if args.keywords:
        unk = labels.index("unknown") if args.unknown else None
        label_idx = {w: (labels.index(w) if w in kws else unk) for w in all_words}

        def subset(split):
            paths = _split_paths(args.root, split)
            kw_p = [p for p in paths if Path(p).parent.name in kws]
            if not args.unknown:
                return kw_p
            other = [p for p in paths if Path(p).parent.name not in kws]
            rng.shuffle(other)
            cap = max(1, len(kw_p) // len(kws))
            return kw_p + other[:cap]
        tag = "kw" + "".join(w[0] for w in kws) + ("_u" if args.unknown else "")
        tr_paths, va_paths = subset("training"), subset("validation")
        te_paths = subset("testing") if args.test else None
    else:
        label_idx = {w: i for i, w in enumerate(labels)}
        tag = "all"
        tr_paths = _split_paths(args.root, "training")
        va_paths = _split_paths(args.root, "validation")
        te_paths = _split_paths(args.root, "testing") if args.test else None

    rng.shuffle(tr_paths)
    if args.limit:
        tr_paths = tr_paths[:args.limit]
    print(f"preprocessing {len(tr_paths)} train + {len(va_paths)} val wavs "
          f"({len(labels)} classes: {labels})...", flush=True)
    tr = _precompute(tr_paths, label_idx, tag=f"{tag}_train{'' if not args.limit else '_'+str(args.limit)}")
    va = _precompute(va_paths, label_idx, tag=f"{tag}_val")
    te = _precompute(te_paths, label_idx, tag=f"{tag}_test") if te_paths else None
    return tr, va, te, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/speech_commands")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--search", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--keywords", type=str, default="",
                    help="comma list to train only these words; empty = all 35 classes")
    ap.add_argument("--unknown", action="store_true",
                    help="with --keywords, add an 'unknown' reject class")
    ap.add_argument("--patience", type=int, default=4, help="early-stopping patience")
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        labels = ["up", "down", "stop", "go", "yes", "no"]
        rng = np.random.default_rng(0)
        Xtr = torch.from_numpy(rng.normal(0, 1, (256, 1, N_MELS, 81)).astype(np.float32))
        Ytr = torch.from_numpy(rng.integers(0, len(labels), 256))
        Xva = torch.from_numpy(rng.normal(0, 1, (64, 1, N_MELS, 81)).astype(np.float32))
        Yva = torch.from_numpy(rng.integers(0, len(labels), 64))
        train_base, val_base, test_base = TensorDataset(Xtr, Ytr), TensorDataset(Xva, Yva), None
    else:
        train_base, val_base, test_base, labels = _build_splits(args, random.Random(args.seed))

    val_dl = DataLoader(val_base, batch_size=256)

    if args.search:
        rng = random.Random(args.seed)
        best = None
        for t in range(args.search):
            cfg = BASELINE.copy() if t == 0 else tuning.sample_cfg(rng, SPACE)
            _, acc = train_one(cfg, train_base, val_dl, len(labels), device, args.epochs, seed=args.seed)
            tuning.log_trial("voice", {"trial": t, **cfg}, {"val_acc": round(acc, 5)})
            print(f"trial {t + 1}/{args.search} val_acc={acc:.4f}  {cfg}", flush=True)
            if best is None or acc > best[1]:
                best = (cfg, acc)
        print(f"\nBEST val_acc={best[1]:.4f}\n{json.dumps(best[0])}")
        return

    cfg = json.loads(args.config) if args.config else BASELINE.copy()
    model, val_acc = train_one(cfg, train_base, val_dl, len(labels), device, args.epochs,
                               seed=args.seed, patience=args.patience, verbose=True)

    report_dl = DataLoader(test_base, batch_size=256) if (args.test and test_base) else val_dl
    y_true, y_pred, y_prob = metrics.collect_classification(model, report_dl, device)
    full_metrics = metrics.classification_metrics(y_true, y_pred, y_prob)
    full_metrics["labels"] = labels
    full_metrics["split"] = "test" if (args.test and test_base) else "val"
    full_metrics["protocol"] = "official Speech Commands v2 speaker-aware split"
    full_metrics["val_accuracy"] = val_acc
    full_metrics["hyperparameters"] = cfg

    out = tuning.out_path(WEIGHTS_PATH, args.smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros(1, 1, N_MELS, 81)
    torch.jit.trace(model.cpu(), example).save(str(out))
    (out.parent / LABELS_PATH.name).write_text(json.dumps(labels), encoding="utf-8")
    out.with_suffix(".metrics.json").write_text(json.dumps(full_metrics, indent=2))
    print(f"saved {out}  ({len(labels)} classes, {full_metrics['split']}_acc={full_metrics['accuracy']:.4f})")


if __name__ == "__main__":
    main()
