"""Intentional-blink state machine.

Natural blinks last ~100-300 ms; the FSM only fires actions for closures held
past a configurable threshold (default 420 ms), so normal blinking never
clicks. Hold durations select the action:

    closure >= hold_click_ms   -> left click   (on reopen)
    closure >= hold_right_ms   -> right click
    closure >= hold_double_ms  -> double click
    closure >= cancel_ms       -> cancelled (rest your eyes, no action)

The FSM also reports live progress (0..1 toward each stage) so the UI and a
beep can tell the user which action will fire when they reopen.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from app.core.types import BlinkEvent, BlinkFrame


class BlinkFSM:
    OPEN, CLOSED, REFRACTORY = "open", "closed", "refractory"

    def __init__(self, cfg_get, on_event: Optional[Callable[[BlinkEvent], None]] = None) -> None:
        self._cfg = cfg_get
        self._on_event = on_event
        self.state = self.OPEN
        self._closed_since: float = 0.0
        self._refractory_until: float = 0.0
        self._last_stage = ""
        self.counts = {"natural": 0, "click": 0, "right_click": 0,
                       "double_click": 0, "cancelled": 0}

    # ordered stages: (config key, event kind)
    _STAGES: List[tuple[str, str]] = [
        ("blink.hold_click_ms", "click"),
        ("blink.hold_right_ms", "right_click"),
        ("blink.hold_double_ms", "double_click"),
        ("blink.cancel_ms", "cancelled"),
    ]

    def _stage_for(self, duration_ms: float) -> str:
        """Highest stage reached for a closure of this duration."""
        kind = "natural"
        for key, k in self._STAGES:
            if duration_ms >= float(self._cfg(key, 1e12)):
                kind = k
        return kind

    def update(self, frame: BlinkFrame) -> Optional[BlinkEvent]:
        t = frame.t
        if self.state == self.REFRACTORY:
            if t >= self._refractory_until:
                self.state = self.OPEN
            else:
                return None

        if self.state == self.OPEN:
            if frame.closed:
                self.state = self.CLOSED
                self._closed_since = t
                self._last_stage = "natural"
            return None

        # state == CLOSED
        duration_ms = (t - self._closed_since) * 1000.0
        stage = self._stage_for(duration_ms)
        event: Optional[BlinkEvent] = None

        if not frame.closed:  # reopened -> decide
            self.state = self.OPEN
            kind = stage
            self.counts[kind] = self.counts.get(kind, 0) + 1
            event = BlinkEvent(kind=kind, duration=duration_ms / 1000.0, t=t)
            if kind not in ("natural",):
                self._refractory_until = t + float(
                    self._cfg("blink.refractory_ms", 400)) / 1000.0
                self.state = self.REFRACTORY
            if self._on_event:
                self._on_event(event)
        elif stage == "cancelled" and self._last_stage != "cancelled":
            # eyes still closed past cancel point: notify once (rest mode)
            self._last_stage = "cancelled"

        self._last_stage = stage if frame.closed else ""
        return event

    def progress(self, t: float) -> dict:
        """Live progress for UI feedback while eyes are closed."""
        if self.state != self.CLOSED:
            return {"closed": False, "duration_ms": 0, "stage": ""}
        duration_ms = (t - self._closed_since) * 1000.0
        return {"closed": True, "duration_ms": round(duration_ms),
                "stage": self._stage_for(duration_ms)}

    def reset(self) -> None:
        self.state = self.OPEN
        self._refractory_until = 0.0
