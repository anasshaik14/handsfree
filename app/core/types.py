"""Core value types shared across the whole pipeline.

Every modality produces small immutable-ish dataclasses so that providers are
interchangeable (Liskov substitution) and the orchestrator/metrics layers can
treat them uniformly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


def now() -> float:
    """Monotonic timestamp used everywhere in the pipeline."""
    return time.perf_counter()


@dataclass
class GazeSample:
    """A single gaze estimate in normalized screen space [0..1]."""
    t: float
    nx: float
    ny: float
    quality: float = 1.0     # 0..1 confidence
    valid: bool = True
    raw_nx: float = 0.0      # pre-mapping feature (for calibration/diagnostics)
    raw_ny: float = 0.0


@dataclass
class BlinkFrame:
    """Per-frame eye-openness measurement from a blink provider."""
    t: float
    left: float                 # openness measure (EAR or lid angle)
    right: float
    closed_left: bool
    closed_right: bool
    threshold_left: float
    threshold_right: float
    measure: str = "ear"        # 'ear' | 'angle'

    @property
    def closed(self) -> bool:
        return self.closed_left and self.closed_right

    @property
    def openness(self) -> float:
        return (self.left + self.right) / 2.0


@dataclass
class BlinkEvent:
    """Emitted by the blink state machine when a closure ends (or progresses)."""
    kind: str        # 'natural' | 'click' | 'right_click' | 'double_click' | 'cancelled'
    duration: float  # seconds the eyes were closed
    t: float


@dataclass
class HeadPose:
    """Neutral-relative head orientation in degrees.

    Conventions (non-mirrored camera frame):
      yaw   > 0  : head turned to the user's LEFT
      pitch > 0  : head tilted UP
      roll  > 0  : head tilted toward the user's left shoulder
    """
    t: float
    yaw: float
    pitch: float
    roll: float
    valid: bool = True


@dataclass
class GestureEvent:
    name: str    # 'head_left' | 'head_right' | 'head_up' | 'head_down'
    t: float
    angle: float = 0.0


@dataclass
class VoiceCommand:
    text: str
    engine: str
    t: float
    intent: str = ""
    args: dict = field(default_factory=dict)
    handled: bool = False
    reply: str = ""


@dataclass
class FaceResult:
    """Output of the shared FaceMesh stage for one frame."""
    t: float
    landmarks: Optional["object"] = None   # np.ndarray (478, 3) in pixel coords (x, y, z*w)
    width: int = 0
    height: int = 0
    frame: Optional["object"] = None       # BGR image the landmarks came from
                                           # (custom CNN providers crop from it)

    @property
    def ok(self) -> bool:
        return self.landmarks is not None
