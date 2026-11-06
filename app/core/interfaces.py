"""Abstract interfaces every provider implements (Strategy pattern).

The pipeline only talks to these ABCs, which is what makes runtime model
switching trivial: swap the concrete instance under a lock, nothing else
changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.core.types import BlinkFrame, FaceResult, GazeSample, HeadPose


class GazeProvider(ABC):
    """Maps face landmarks to a normalized screen position."""

    name: str = "gaze"
    needs_calibration: bool = False

    @abstractmethod
    def process(self, face: FaceResult, pose: Optional[HeadPose],
                frame=None) -> Optional[GazeSample]:
        """`frame` is the BGR image (only classical CV providers need it)."""

    def reset(self) -> None:  # noqa: B027 - optional hook
        pass


class BlinkProvider(ABC):
    """Produces per-frame eye openness + closed/open classification."""

    name: str = "blink"

    @abstractmethod
    def process(self, face: FaceResult) -> Optional[BlinkFrame]:
        ...

    def reset(self) -> None:  # noqa: B027
        pass


class HeadPoseProvider(ABC):
    """Estimates neutral-relative yaw/pitch/roll in degrees."""

    name: str = "headpose"

    @abstractmethod
    def process(self, face: FaceResult) -> Optional[HeadPose]:
        ...

    @abstractmethod
    def set_neutral(self, face: FaceResult) -> bool:
        """Capture the current pose as neutral. Returns success."""

    def reset(self) -> None:  # noqa: B027
        pass


class VoiceEngine(ABC):
    """Continuously converts speech to text and hands transcripts to a callback."""

    name: str = "voice"

    @abstractmethod
    def start(self, on_final, on_partial=None) -> bool:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @property
    @abstractmethod
    def running(self) -> bool:
        ...

    @staticmethod
    def available() -> bool:
        return True
