"""Runtime helpers for loading and running the custom TorchScript models.

Torch is imported lazily so the whole app (and this module) imports fine
without torch installed — the custom providers call `torch_available()` and
`load_scripted()` and quietly fall back to their classical counterpart when
either torch or the trained weights file is missing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path("models/custom")

_torch_configured = False


def _configure_torch(torch) -> None:
    """Pin torch to one thread. The custom CNNs are tiny (32x32..64x64
    inputs); intra-op threading buys nothing but OpenMP busy-spin that
    saturates every core (the observed ~400% CPU at 3-4 fps). Single
    thread is faster for these models AND leaves the CPU to the rest of
    the pipeline."""
    global _torch_configured
    if _torch_configured:
        return
    _torch_configured = True
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass  # raises if parallel work already started — intra=1 still holds


def torch_available() -> bool:
    try:
        import torch
        _configure_torch(torch)
        return True
    except Exception:
        return False


def load_scripted(path: "str | Path"):
    """Load a TorchScript model in eval mode on CPU, or None if the file is
    missing / torch is unavailable / it fails to load."""
    p = Path(path)
    if not p.exists() or not torch_available():
        return None
    try:
        import torch
        model = torch.jit.load(str(p), map_location="cpu")
        model.eval()
        return model
    except Exception:
        log.exception("failed to load custom model %s", p)
        return None


def infer(model, x: np.ndarray) -> np.ndarray:
    """Run a forward pass on a numpy array, return numpy logits/outputs.
    `x` must already have the model's expected shape minus the batch dim."""
    import torch
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
        return model(t).squeeze(0).cpu().numpy()
