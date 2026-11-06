"""Polynomial gaze calibration: fit + persistence + on-screen wizard.

A 9-point calibration collects (eye features, head pose) -> (screen point)
pairs and fits a regularized 2nd-order polynomial per screen axis:

  screen = w0 + w1*hx + w2*hy + w3*yaw + w4*pitch + w5*hx*hy + w6*hx^2 + w7*hy^2

Including yaw/pitch makes the model tolerant to small head motion — the main
failure mode of iris-only calibration.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

FEATURE_DIM = 8
# A model whose training error is worse than this is unusable — predictions
# would fling the cursor into screen corners. Refuse to save or load it.
MAX_SANE_ERROR_PX = 250.0
# Precision = prediction scatter at a fixed target. Worse than this and the
# cursor visibly wanders/jitters no matter how much output smoothing runs.
MAX_SANE_PRECISION_PX = 130.0


def feature_vector(hx: float, hy: float, yaw: float, pitch: float) -> np.ndarray:
    return np.array([1.0, hx, hy, yaw / 30.0, pitch / 30.0,
                     hx * hy, hx * hx, hy * hy], dtype=np.float64)


class PolynomialModel:
    def __init__(self, wx: Optional[np.ndarray] = None,
                 wy: Optional[np.ndarray] = None) -> None:
        self.wx = wx
        self.wy = wy
        self.mean_error_px: float = -1.0
        self.precision_px: float = -1.0
        # Held-out accuracy from the post-fit quality check (Upgrade 10): the
        # error on random targets the fit never saw. Unlike mean_error_px
        # (measured on training samples, which flatters the model), this is
        # what the user will actually feel. -1 = not validated.
        self.validation_error_px: float = -1.0
        self.fitted_at: float = 0.0
        # The model's yaw/pitch terms are scaled to the head-pose provider
        # active during calibration; a different provider at runtime feeds
        # mis-scaled angles and drags the cursor away from the gaze.
        self.pose_provider: str = ""
        # RLS inverse-covariance for online/dynamic calibration (Upgrade 4).
        # Set by fit(); lazily cold-started if a model was loaded without it.
        self.P: Optional[np.ndarray] = None

    @property
    def ready(self) -> bool:
        return self.wx is not None and self.wy is not None

    def predict(self, hx: float, hy: float, yaw: float, pitch: float) -> Tuple[float, float]:
        f = feature_vector(hx, hy, yaw, pitch)
        return float(f @ self.wx), float(f @ self.wy)

    # -- dynamic calibration (Upgrade 4) ----------------------------------
    def online_update(self, hx: float, hy: float, yaw: float, pitch: float,
                      tx: float, ty: float, forget: float = 0.9995) -> None:
        """One recursive-least-squares step nudging the fit toward a fresh
        (features -> known screen target) pair, e.g. captured the instant the
        user commits a click. `forget` < 1 slowly discounts old data so the
        model tracks drift (posture, lighting) without a full recalibration.

        Standard RLS with a forgetting factor — mathematically the same fit
        as batch least squares, just applied one sample at a time, so it can
        only refine the existing weights, never diverge from them wildly."""
        if not self.ready:
            return
        f = feature_vector(hx, hy, yaw, pitch)
        if self.P is None:
            # Cold start for a model loaded from disk without covariance:
            # moderate prior so early updates stay gentle.
            self.P = np.eye(FEATURE_DIM, dtype=np.float64)
        Pf = self.P @ f
        denom = forget + float(f @ Pf)
        if denom < 1e-9:
            return
        k = Pf / denom
        self.wx = self.wx + k * (tx - float(f @ self.wx))
        self.wy = self.wy + k * (ty - float(f @ self.wy))
        self.P = (self.P - np.outer(k, Pf)) / forget

    # -- fitting ----------------------------------------------------------
    @staticmethod
    def fit(samples: List[dict], ridge: float = 1e-4,
            trim: float = 0.2) -> "PolynomialModel":
        """samples: [{hx, hy, yaw, pitch, tx, ty}] with targets in normalized [0..1].

        Robust two-pass fit: solve, drop the `trim` fraction of samples with
        the largest residuals (blinks, glances away from the dot), re-solve.
        """
        def _solve(subset: List[dict]):
            X = np.stack([feature_vector(s["hx"], s["hy"], s["yaw"], s["pitch"])
                          for s in subset])
            tx = np.array([s["tx"] for s in subset])
            ty = np.array([s["ty"] for s in subset])
            A = X.T @ X + ridge * np.eye(FEATURE_DIM)
            return (np.linalg.solve(A, X.T @ tx),
                    np.linalg.solve(A, X.T @ ty), X, tx, ty)

        wx, wy, X, tx, ty = _solve(samples)
        if trim > 0 and len(samples) >= 30:
            # Iterative trimming: each pass drops the worst residuals and
            # refits, so outliers can't hide behind a fit they skewed.
            subset = samples
            for _ in range(3):
                err = np.hypot(X @ wx - tx, X @ wy - ty)
                keep = np.argsort(err)[:int(len(subset) * (1.0 - trim / 3))]
                if len(keep) == len(subset):
                    break
                subset = [subset[i] for i in keep]
                wx, wy, X, tx, ty = _solve(subset)
        model = PolynomialModel(wx, wy)
        # Seed the RLS covariance from the batch fit so dynamic calibration
        # starts as confident as the calibration that produced it.
        model.P = np.linalg.inv(X.T @ X + ridge * np.eye(FEATURE_DIM))
        model.fitted_at = time.time()
        return model

    def evaluate(self, samples: List[dict], screen_w: int, screen_h: int) -> None:
        """Gaze-research metrics: accuracy = mean error to target (px),
        precision = RMS sample dispersion around each target's own centroid."""
        if not samples:
            return
        errors, dispersions = [], []
        by_target: dict = {}
        for s in samples:
            px, py = self.predict(s["hx"], s["hy"], s["yaw"], s["pitch"])
            ex = (px - s["tx"]) * screen_w
            ey = (py - s["ty"]) * screen_h
            errors.append(float(np.hypot(ex, ey)))
            by_target.setdefault((s["tx"], s["ty"]), []).append((px * screen_w, py * screen_h))
        for pts in by_target.values():
            arr = np.array(pts)
            if len(arr) > 1:
                centroid = arr.mean(axis=0)
                dispersions.append(float(np.sqrt(((arr - centroid) ** 2).sum(axis=1).mean())))
        self.mean_error_px = float(np.mean(errors))
        self.precision_px = float(np.mean(dispersions)) if dispersions else 0.0

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "wx": self.wx.tolist(), "wy": self.wy.tolist(),
            "mean_error_px": self.mean_error_px,
            "precision_px": self.precision_px,
            "validation_error_px": self.validation_error_px,
            "fitted_at": self.fitted_at,
            "pose_provider": self.pose_provider,
            "P": self.P.tolist() if self.P is not None else None,
        }), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> Optional["PolynomialModel"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            m = PolynomialModel(np.array(d["wx"]), np.array(d["wy"]))
            m.mean_error_px = d.get("mean_error_px", -1.0)
            m.precision_px = d.get("precision_px", -1.0)
            m.validation_error_px = d.get("validation_error_px", -1.0)
            m.fitted_at = d.get("fitted_at", 0.0)
            m.pose_provider = d.get("pose_provider", "")
            if d.get("P") is not None:
                m.P = np.array(d["P"], dtype=np.float64)
            if m.mean_error_px > MAX_SANE_ERROR_PX:
                log.warning("ignoring saved calibration (error %.0fpx > %.0fpx) — "
                            "recalibrate", m.mean_error_px, MAX_SANE_ERROR_PX)
                return None
            if m.precision_px > MAX_SANE_PRECISION_PX:
                log.warning("ignoring saved calibration (precision %.0fpx > %.0fpx: "
                            "cursor would wander) — recalibrate",
                            m.precision_px, MAX_SANE_PRECISION_PX)
                return None
            return m
        except Exception:
            log.exception("failed to load calibration")
            return None


# 9-point grid in normalized screen coords (margin keeps dots visible)
def calibration_points(margin: float = 0.08, grid: Optional[int] = None
                       ) -> List[Tuple[float, float]]:
    """Target dots in normalized screen coords.

    Default (grid=None) is the classic center-first 9-point layout. Pass an
    NxN `grid` (Upgrade 1: multi-point) for a denser calibration — 4->16,
    5->25, 6->36 points — laid out in a serpentine so the dot never leaps
    across the screen between captures.
    """
    if grid is None:
        lo, hi = margin, 1.0 - margin
        mid = 0.5
        return [(mid, mid), (lo, lo), (hi, lo), (lo, hi), (hi, hi),
                (mid, lo), (mid, hi), (lo, mid), (hi, mid)]
    n = max(2, int(grid))
    coords = [margin + (1.0 - 2 * margin) * i / (n - 1) for i in range(n)]
    pts: List[Tuple[float, float]] = []
    for row, y in enumerate(coords):
        xs = coords if row % 2 == 0 else list(reversed(coords))
        pts.extend((x, y) for x in xs)
    return pts


def validation_points(n: int = 5, margin: float = 0.14,
                      seed: Optional[int] = None) -> List[Tuple[float, float]]:
    """Random held-out targets for the post-fit quality check (Upgrade 10).

    Deliberately random (not on the calibration grid) so the check measures
    generalization, not memorization. Kept away from the extreme edges where
    every model is worst and a fail would be unfairly harsh."""
    rng = np.random.default_rng(seed)
    lo, hi = margin, 1.0 - margin
    return [(float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi)))
            for _ in range(max(1, n))]
