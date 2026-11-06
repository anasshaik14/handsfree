"""Turns per-frame gesture labels (open_palm/thumbs_up/thumbs_down/peace/"")
into debounced, deliberate actions.

Debounce: a label only becomes "current" after N consecutive frames agree
(single-frame misclassifications from motion blur / partial occlusion don't
fire anything). Once current:
  - open_palm is LEVEL-triggered: `.paused` is true for as long as it's shown
    (read directly by the pipeline to gate cursor + blink clicks).
  - thumbs_up / thumbs_down are EDGE-triggered with auto-repeat while held
    (like the head-gesture repeat mode), so one deliberate lift bumps volume
    once and holding it keeps going.
  - peace requires a short deliberate hold (dwell) before firing once, then
    must drop out and come back to fire again — a peace sign quits the whole
    program, so an accidental single-frame flash must never trigger it.
"""
from __future__ import annotations

from typing import Callable, List, Optional


class HandGestureFSM:
    REPEATABLE = {"thumbs_up", "thumbs_down"}

    def __init__(self, cfg_get) -> None:
        self._cfg = cfg_get
        self._raw_streak_label = ""
        self._raw_streak_n = 0
        self.current = ""            # debounced label, "" = no gesture
        self._current_since = 0.0
        self._last_repeat = 0.0
        self._quit_armed_since: Optional[float] = None
        self._quit_fired = False

    @property
    def paused(self) -> bool:
        return self.current == "open_palm"

    def update(self, label: str, t: float) -> List[str]:
        """Feed one frame's classified label; returns fired event names."""
        debounce_n = int(self._cfg("hands.debounce_frames", 3))
        if label == self._raw_streak_label:
            self._raw_streak_n += 1
        else:
            self._raw_streak_label = label
            self._raw_streak_n = 1

        events: List[str] = []
        stable = self._raw_streak_n >= debounce_n
        if stable and label != self.current:
            self.current = label
            self._current_since = t
            self._last_repeat = t
            self._quit_armed_since = None
            self._quit_fired = False
            if label in self.REPEATABLE:
                events.append(label)

        if self.current in self.REPEATABLE and self._raw_streak_label == self.current:
            repeat_s = float(self._cfg("hands.repeat_interval_ms", 600)) / 1000.0
            if (t - self._last_repeat) >= repeat_s and t != self._current_since:
                self._last_repeat = t
                events.append(self.current)

        if self.current == "peace":
            dwell_s = float(self._cfg("hands.quit_dwell_ms", 700)) / 1000.0
            if self._quit_armed_since is None:
                self._quit_armed_since = t
            elif not self._quit_fired and (t - self._quit_armed_since) >= dwell_s:
                self._quit_fired = True
                events.append("quit")
        else:
            self._quit_armed_since = None
            self._quit_fired = False

        return events

    def reset(self) -> None:
        self.__init__(self._cfg)
