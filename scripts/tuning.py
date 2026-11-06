"""Shared plumbing for the hyperparameter search: seeding, subject-disjoint
splits, and trial logging.

Why this exists (all three are correctness fixes, not conveniences):

1. SEEDING. Weight init was unseeded, so two runs of the same config differed
   by more than the effect we're trying to measure — the search was invalid
   before it started. `seed_all` makes a trial reproducible from its CSV row.

2. SPLITS. Every trainer used `random_split` over rows. Rows are not
   independent in any of these datasets: MRL has ~2k frames per subject,
   MPIIGaze thousands per participant, 300W-LP a median of 34 synthesized
   views of each of its 3,837 faces. A random row split therefore puts
   near-duplicates of the same face on both sides and scores memorisation.
   `group_split` splits on the group key (subject/participant/face) instead,
   into three disjoint parts, so selection and reporting stay honest.

3. TRIAL LOG. Every trial appends its config + validation score to
   reports/tuning/<model>_trials.csv so the search is reproducible/citable.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

TRIALS_DIR = Path(__file__).resolve().parent.parent / "reports" / "tuning"


def seed_all(seed: int) -> None:
    """Seed python/numpy/torch so a config -> result mapping is reproducible."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def group_split(groups, fracs=(0.7, 0.15, 0.15), seed: int = 0):
    """Split row indices into 3 GROUP-disjoint parts.

    `groups` is a per-row key (subject id, participant, face id). Groups are
    shuffled then dealt into train/val/test by target row-count, so the parts
    hit `fracs` by rows while no group ever spans two parts.

    Returns (train_idx, val_idx, test_idx) as int arrays.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(groups)
    targets = [f * n for f in fracs]
    buckets: list[list] = [[], [], []]
    counts = [0.0, 0.0, 0.0]
    # Deal the biggest groups first into whichever bucket is furthest below its
    # target — greedy, but it keeps the row-fractions close even when group
    # sizes are lopsided (MRL subjects range ~700..10k frames).
    sizes = {g: int((groups == g).sum()) for g in uniq}
    for g in sorted(uniq, key=lambda g: -sizes[g]):
        i = int(np.argmax([t - c for t, c in zip(targets, counts)]))
        buckets[i].append(g)
        counts[i] += sizes[g]

    out = []
    for b in buckets:
        mask = np.isin(groups, b)
        out.append(np.nonzero(mask)[0])
    return tuple(out)


def log_trial(model: str, cfg: dict, scores: dict) -> None:
    """Append one trial row to reports/tuning/<model>_trials.csv."""
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRIALS_DIR / f"{model}_trials.csv"
    row = {**cfg, **scores}
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def sample_cfg(rng: random.Random, space: dict) -> dict:
    """One random config. `space` maps name -> list (choice) or
    ('loguniform'|'uniform', lo, hi)."""
    cfg = {}
    for k, v in space.items():
        if isinstance(v, list):
            cfg[k] = rng.choice(v)
        elif v[0] == "loguniform":
            cfg[k] = float(np.exp(rng.uniform(np.log(v[1]), np.log(v[2]))))
        elif v[0] == "uniform":
            cfg[k] = float(rng.uniform(v[1], v[2]))
        else:
            raise ValueError(f"bad space entry {k}={v!r}")
    return cfg


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    import torch
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimiser {name!r}")


def make_sched(name: str, opt, epochs: int):
    import torch
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)
    return None


def out_path(weights_path, smoke: bool):
    """Where a run may write its .pt. A --smoke run trains on random noise, so
    it must never land on the real weights the app loads — it did, and a smoke
    test silently replaced a trained model. Redirect it to a scratch file."""
    if not smoke:
        return Path(weights_path)
    scratch = Path(weights_path).parent / "_smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch / Path(weights_path).name


def cpu_threads() -> None:
    """Training-time thread setup.

    Pinning torch to 1 thread is right for the real-time RUNTIME
    (app/ml/runtime.py, tiny per-frame inference where extra threads only
    spin) but wrong for training at torch's DEFAULT thread count (~8 on this
    box): a background process on this machine (claude-mem's chroma-mcp)
    competes for CPU unpredictably, and 8-way torch intra-op parallelism
    across only 8 logical cores turns into kernel-dominated thread thrash
    the moment something else wants a core (one trial: 16s -> 30+ min).

    2 threads is the middle ground: enough parallelism to beat 1 thread when
    the box is quiet, small enough that extra threads don't add more
    context-switch overhead than compute when it isn't."""
    import torch
    torch.set_num_threads(2)
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass


def demo() -> None:
    """Self-check: the split is group-disjoint and roughly hits its fractions."""
    groups = np.repeat(np.arange(40), 100)          # 40 groups x 100 rows
    tr, va, te = group_split(groups, (0.7, 0.15, 0.15), seed=0)
    assert len(tr) + len(va) + len(te) == len(groups), "split must cover every row"
    assert not (set(tr) & set(va) | set(va) & set(te) | set(tr) & set(te)), "row overlap"
    gt, gv, ge = groups[tr], groups[va], groups[te]
    assert not set(gt) & set(gv), "train/val share a group"
    assert not set(gv) & set(ge), "val/test share a group"
    assert not set(gt) & set(ge), "train/test share a group"
    assert abs(len(tr) / len(groups) - 0.7) < 0.05, f"train frac off: {len(tr)/len(groups)}"

    # lopsided group sizes (like MRL subjects) must still split disjointly
    lop = np.concatenate([np.full(s, i) for i, s in enumerate([10000, 5000, 700, 700, 700, 700, 700])])
    tr, va, te = group_split(lop, (0.7, 0.15, 0.15), seed=1)
    assert not set(lop[tr]) & set(lop[va]), "lopsided: train/val share a group"
    assert not set(lop[tr]) & set(lop[te]), "lopsided: train/test share a group"

    rng = random.Random(0)
    cfg = sample_cfg(rng, {"lr": ("loguniform", 1e-4, 3e-3), "opt": ["adam", "sgd"]})
    assert 1e-4 <= cfg["lr"] <= 3e-3 and cfg["opt"] in ("adam", "sgd")
    print("tuning.py self-check OK")


if __name__ == "__main__":
    demo()
