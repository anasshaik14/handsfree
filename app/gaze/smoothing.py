"""Signal stabilization for the gaze cursor.

The chain that produces a professional, jitter-free cursor:

  raw gaze -> One Euro filter -> deadzone hysteresis -> slew limiter -> screen

* One Euro (Casiez et al., CHI 2012): adaptive low-pass whose cutoff rises
  with speed — heavy smoothing at fixation (no jitter), minimal lag during
  saccades (no rubber-banding).
* Deadzone hysteresis: the cursor only moves once the target escapes a small
  radius, absorbing residual tremor completely at rest.
* Slew limiter: caps px/s so single-frame estimation glitches can never
  teleport the cursor.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple


class LowPass:
    def __init__(self) -> None:
        self._y: Optional[float] = None

    def filter(self, x: float, alpha: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    @property
    def last(self) -> Optional[float]:
        return self._y

    def reset(self) -> None:
        self._y = None


class OneEuro:
    """Scalar One Euro filter."""

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.0,
                 dcutoff: float = 1.0) -> None:
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x = LowPass()
        self._dx = LowPass()
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        if self._t is None:
            self._t = t
            self._dx.filter(0.0, 1.0)
            return self._x.filter(x, 1.0)
        dt = max(t - self._t, 1e-6)
        self._t = t
        prev = self._x.last
        dx = (x - prev) / dt if prev is not None else 0.0
        edx = self._dx.filter(dx, self._alpha(self.dcutoff, dt))
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self._x.filter(x, self._alpha(cutoff, dt))

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._t = None


class OneEuro2D:
    def __init__(self, mincutoff: float = 1.0, beta: float = 0.0,
                 dcutoff: float = 1.0) -> None:
        self.fx = OneEuro(mincutoff, beta, dcutoff)
        self.fy = OneEuro(mincutoff, beta, dcutoff)

    def filter(self, x: float, y: float, t: float) -> Tuple[float, float]:
        return self.fx.filter(x, t), self.fy.filter(y, t)

    def tune(self, mincutoff: float, beta: float, dcutoff: float) -> None:
        for f in (self.fx, self.fy):
            f.mincutoff, f.beta, f.dcutoff = mincutoff, beta, dcutoff

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()


class EMA2D:
    def __init__(self, alpha: float = 0.25) -> None:
        self.alpha = alpha
        self._x = LowPass()
        self._y = LowPass()

    def filter(self, x: float, y: float, t: float) -> Tuple[float, float]:
        return self._x.filter(x, self.alpha), self._y.filter(y, self.alpha)

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()


class Deadzone:
    """Positional hysteresis: output stays put until input escapes `radius`."""

    def __init__(self, radius: float = 8.0) -> None:
        self.radius = radius
        self._ox: Optional[float] = None
        self._oy: Optional[float] = None

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        if self._ox is None:
            self._ox, self._oy = x, y
            return x, y
        dx, dy = x - self._ox, y - self._oy
        dist = math.hypot(dx, dy)
        if dist > self.radius:
            # Follow the input but stay `radius` behind it -> continuous motion,
            # no popping when crossing the boundary.
            k = (dist - self.radius) / dist
            self._ox += dx * k
            self._oy += dy * k
        return self._ox, self._oy

    def reset(self) -> None:
        self._ox = self._oy = None


class SlewLimiter:
    """Caps output speed (px/s) to suppress single-frame glitches."""

    def __init__(self, max_speed: float = 2600.0) -> None:
        self.max_speed = max_speed
        self._x: Optional[float] = None
        self._y: Optional[float] = None
        self._t: Optional[float] = None

    def filter(self, x: float, y: float, t: float) -> Tuple[float, float]:
        if self._x is None:
            self._x, self._y, self._t = x, y, t
            return x, y
        dt = max(t - self._t, 1e-6)
        self._t = t
        dx, dy = x - self._x, y - self._y
        dist = math.hypot(dx, dy)
        max_step = self.max_speed * dt
        if dist > max_step:
            k = max_step / dist
            dx, dy = dx * k, dy * k
        self._x += dx
        self._y += dy
        return self._x, self._y

    def reset(self) -> None:
        self._x = self._y = self._t = None


class Stabilizer:
    """The full production chain, configured from the app config."""

    def __init__(self, cfg: dict) -> None:
        self.kind = cfg.get("filter", "one_euro")
        self.euro = OneEuro2D(cfg.get("mincutoff", 0.6),
                              cfg.get("beta", 0.004),
                              cfg.get("dcutoff", 1.0))
        self.ema = EMA2D(cfg.get("ema_alpha", 0.25))
        self.deadzone = Deadzone(cfg.get("deadzone_px", 9.0))
        self.slew = SlewLimiter(cfg.get("max_speed_px", 2600.0))

    def retune(self, cfg: dict) -> None:
        self.kind = cfg.get("filter", self.kind)
        self.euro.tune(cfg.get("mincutoff", 0.6), cfg.get("beta", 0.004),
                       cfg.get("dcutoff", 1.0))
        self.ema.alpha = cfg.get("ema_alpha", 0.25)
        self.deadzone.radius = cfg.get("deadzone_px", 9.0)
        self.slew.max_speed = cfg.get("max_speed_px", 2600.0)

    def process(self, x: float, y: float, t: float) -> Tuple[float, float]:
        if self.kind == "one_euro":
            x, y = self.euro.filter(x, y, t)
        elif self.kind == "ema":
            x, y = self.ema.filter(x, y, t)
        x, y = self.deadzone.filter(x, y)
        x, y = self.slew.filter(x, y, t)
        return x, y

    def reset(self) -> None:
        self.euro.reset()
        self.ema.reset()
        self.deadzone.reset()
        self.slew.reset()
