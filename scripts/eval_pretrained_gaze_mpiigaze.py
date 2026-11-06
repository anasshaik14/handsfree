"""Evaluate the pretrained MPIIGaze-trained gaze model (hysts resnet_preact, the
in-domain replacement for the cross-dataset L2CS attempt) on MPIIGaze, producing
the SAME regression/angular metric suite used for our custom gaze CNN.

This is a TRUE apples-to-apples test: the model was trained on MPIIGaze normalized
eye patches + 2D head pose -- exactly the input our custom gaze CNN uses -- so we
feed it the official MPIIGaze Normalized .mat data directly (no re-normalization,
no cross-domain gap). Preprocessing/conventions are taken verbatim from ptgaze's
own inference code (transforms.ToTensor, left-eye convention, angle<->vector).

Writes reports/tuning/pretrained/gaze_mpiigaze.metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import metrics  # noqa: E402

OUT = ROOT / "reports" / "tuning" / "pretrained"
NORM = ROOT / "MPIIGaze_DATASET" / "Data" / "Normalized"
WEIGHTS = Path.home() / ".cache/huggingface/hub/models--hysts--ptgaze-mpiigaze-resnet-preact"


def _weights_path():
    hits = list(WEIGHTS.rglob("model.safetensors"))
    if not hits:
        raise FileNotFoundError("run hf_hub_download for hysts/ptgaze-mpiigaze-resnet-preact first")
    return hits[0]


def load_model():
    import importlib.util
    import torch
    from safetensors.torch import load_file
    src = ROOT / "app/gaze/mpiigaze_resnet_preact.py"
    spec = importlib.util.spec_from_file_location("rp", str(src))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.Model(None)  # config is ignored by __init__ (hardcoded depth=8)
    sd = load_file(str(_weights_path()))
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}", flush=True)
    net.eval()
    return net


def convert_pose(vecs):
    """(N,3) rotvec -> (N,2) [pitch, yaw] of the z-axis (ptgaze _normalize_head_pose)."""
    from scipy.spatial.transform import Rotation
    z = Rotation.from_rotvec(vecs).as_matrix()[:, :, 2]      # (N,3)
    return np.stack([np.arcsin(z[:, 1]), np.arctan2(z[:, 0], z[:, 2])], 1)


def convert_gaze(vecs):
    """(N,3) gaze vector -> (N,2) [pitch, yaw] (ptgaze gaze_vector_to_angles)."""
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    return np.stack([np.arcsin(-y), np.arctan2(-x, -z)], 1)


def main(per_participant=None):
    import torch
    torch.set_num_threads(1)

    net = load_model()
    gt, pred = [], []
    t0 = time.time()

    import scipy.io as sio
    for pdir in sorted(NORM.glob("p*")):
        pid = pdir.stem
        imgs, heads, gzs, is_right = [], [], [], []
        for mat in sorted(pdir.glob("day*.mat")):
            d = sio.loadmat(str(mat), struct_as_record=False, squeeze_me=True)["data"]
            for side, mirror in (("left", False), ("right", True)):
                s = getattr(d, side)
                im = np.asarray(s.image, np.uint8).reshape(-1, 36, 60)
                hp = convert_pose(np.asarray(s.pose, np.float64).reshape(-1, 3))
                gz = convert_gaze(np.asarray(s.gaze, np.float64).reshape(-1, 3))
                if mirror:                                   # right eye -> left-eye convention (input only)
                    im = im[:, :, ::-1]
                    hp = hp * np.array([1, -1])
                imgs.append(im); heads.append(hp); gzs.append(gz)
                is_right.append(np.full(len(im), mirror))
        im = np.concatenate(imgs).astype(np.float32) / 255.0  # ToTensor: /255
        hp = np.concatenate(heads).astype(np.float32)
        gz = np.concatenate(gzs)
        rmask = np.concatenate(is_right)
        if per_participant and len(im) > per_participant:
            sel = np.linspace(0, len(im) - 1, per_participant).astype(int)
            im, hp, gz, rmask = im[sel], hp[sel], gz[sel], rmask[sel]

        x = torch.from_numpy(np.ascontiguousarray(im))[:, None]  # (N,1,36,60)
        y = torch.from_numpy(hp)
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                outs.append(net(x[i:i + 4096], y[i:i + 4096]).numpy())
        p = np.concatenate(outs)                              # (N,2) [pitch,yaw], left-eye convention
        p[rmask, 1] *= -1                                     # flip predicted yaw back for right eyes
        gt.append(gz); pred.append(p)
        print(f"  {pid}: {len(p)} eyes ({time.time()-t0:.0f}s)", flush=True)

    gt = np.concatenate(gt)
    pred = np.concatenate(pred)

    gt_deg = np.degrees(gt)
    pred_deg = np.degrees(pred)
    full = metrics.regression_metrics(gt_deg, pred_deg, ["pitch", "yaw"])
    full.update(metrics.gaze_angular_error_deg(gt, pred))
    # save raw arrays (degrees) so the report/Excel can be built with the SAME
    # canonical reg_metrics() the custom CNN uses (acc_within, r2_best, ...)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "gaze_mpiigaze.arrays.npz", true_deg=gt_deg, pred_deg=pred_deg)
    full["model"] = "MPIIGaze ResNet-preact (pretrained, hysts)"
    full["test_set"] = "MPIIGaze (official Normalized .mat, both eyes) -- IN-DOMAIN"
    full["n_samples"] = int(len(gt))
    full["output_order"] = ["pitch", "yaw"]
    full["note"] = ("MPIIGaze-trained model on official normalized eye patches + 2D head pose; "
                    "same input domain as our custom gaze CNN (true apples-to-apples).")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gaze_mpiigaze.metrics.json").write_text(json.dumps(full, indent=2))
    print(f"\nMPIIGaze model ({len(gt)} eyes): angular={full['angular_error_deg_mean']:.3f} deg "
          f"(custom CNN was ~6.1 deg)")
    print(f"wrote {OUT / 'gaze_mpiigaze.metrics.json'}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(per_participant=n)
