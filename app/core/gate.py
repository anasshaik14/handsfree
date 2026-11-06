"""Hysteretic confidence gate, generalized from the gaze quality gate.

A flickering confidence signal around one threshold would otherwise
freeze/reopen a modality in a stutter every frame — this needs two
thresholds: close below `low`, only reopen at/above `high`.
"""
from __future__ import annotations


class HysteresisGate:
    def __init__(self, low: float, high: float, start_open: bool = True) -> None:
        self.low = low
        self.high = high
        self.is_open = start_open

    def update(self, confidence: float) -> bool:
        if self.is_open:
            if confidence < self.low:
                self.is_open = False
        elif confidence >= self.high:
            self.is_open = True
        return self.is_open
