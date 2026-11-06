"""Deliberate head-gesture state machine.

A gesture fires only when the head crosses the ON threshold and STAYS there
for the dwell time (deliberateness), then nothing more can fire until the
head returns inside the OFF threshold (return-to-neutral hysteresis) and the
cooldown has elapsed. Optional repeat mode re-fires while the pose is held.

Axes are independent: yaw drives head_left/head_right, pitch drives
head_up/head_down. The dominant axis wins to avoid diagonal double-fires.
"""
from __future__ import annotations

from typing import Callable, Optional

from app.core.types import GestureEvent, HeadPose


class _AxisFSM:
    IDLE, PENDING, FIRED = "idle", "pending", "fired"

    def __init__(self, pos_name: str, neg_name: str) -> None:
        self.pos_name = pos_name
        self.neg_name = neg_name
        self.state = self.IDLE
        self._direction = 0
        self._entered_at = 0.0
        self._last_fire = float("-inf")   # never fired -> no startup cooldown

    def update(self, value: float, t: float, on: float, off: float,
               dwell_s: float, cooldown_s: float, repeat: bool,
               repeat_s: float) -> Optional[str]:
        direction = 1 if value >= on else (-1 if value <= -on else 0)
        inside_neutral = abs(value) <= off

        if self.state == self.IDLE:
            if direction != 0 and (t - self._last_fire) >= cooldown_s:
                self.state = self.PENDING
                self._direction = direction
                self._entered_at = t
            return None

        if self.state == self.PENDING:
            if direction != self._direction:
                self.state = self.IDLE
                return None
            if (t - self._entered_at) >= dwell_s:
                self.state = self.FIRED
                self._last_fire = t
                return self.pos_name if self._direction > 0 else self.neg_name
            return None

        # FIRED: wait for return-to-neutral (or auto-repeat while held)
        if inside_neutral:
            self.state = self.IDLE
        elif repeat and direction == self._direction and \
                (t - self._last_fire) >= repeat_s:
            self._last_fire = t
            return self.pos_name if self._direction > 0 else self.neg_name
        return None

    def reset(self) -> None:
        self.state = self.IDLE
        self._direction = 0


class GestureFSM:
    SMOOTH_ALPHA = 0.35   # EMA on the angles: estimation spikes can't dwell

    def __init__(self, cfg_get,
                 on_gesture: Optional[Callable[[GestureEvent], None]] = None) -> None:
        self._cfg = cfg_get
        self._on_gesture = on_gesture
        # yaw > 0 = turned LEFT  -> head_left
        self._yaw_fsm = _AxisFSM("head_left", "head_right")
        # pitch > 0 = UP -> head_up
        self._pitch_fsm = _AxisFSM("head_up", "head_down")
        self._sy: Optional[float] = None    # smoothed yaw/pitch
        self._sp: Optional[float] = None
        self.counts = {"head_left": 0, "head_right": 0, "head_up": 0, "head_down": 0}
        self.last_gesture = ""

    def update(self, pose: HeadPose) -> Optional[GestureEvent]:
        if not pose.valid:
            return None
        a = self.SMOOTH_ALPHA
        self._sy = pose.yaw if self._sy is None else a * pose.yaw + (1 - a) * self._sy
        self._sp = pose.pitch if self._sp is None else a * pose.pitch + (1 - a) * self._sp
        pose = HeadPose(t=pose.t, yaw=self._sy, pitch=self._sp,
                        roll=pose.roll, valid=pose.valid)
        g = self._cfg
        dwell = float(g("headpose.dwell_ms", 260)) / 1000.0
        cooldown = float(g("headpose.cooldown_ms", 750)) / 1000.0
        repeat = bool(g("headpose.repeat", False))
        repeat_s = float(g("headpose.repeat_interval_ms", 900)) / 1000.0

        # Dominant-axis arbitration: the axis with more headroom vs its
        # threshold gets priority; the other is held in reset.
        yaw_ratio = abs(pose.yaw) / max(float(g("headpose.yaw_on_deg", 18.0)), 1e-6)
        pitch_ratio = abs(pose.pitch) / max(float(g("headpose.pitch_on_deg", 14.0)), 1e-6)

        name: Optional[str] = None
        if yaw_ratio >= pitch_ratio:
            name = self._yaw_fsm.update(
                pose.yaw, pose.t, float(g("headpose.yaw_on_deg", 18.0)),
                float(g("headpose.yaw_off_deg", 8.0)), dwell, cooldown,
                repeat, repeat_s)
            if pitch_ratio < 0.6:
                self._pitch_fsm.reset()
        else:
            name = self._pitch_fsm.update(
                pose.pitch, pose.t, float(g("headpose.pitch_on_deg", 14.0)),
                float(g("headpose.pitch_off_deg", 6.0)), dwell, cooldown,
                repeat, repeat_s)
            if yaw_ratio < 0.6:
                self._yaw_fsm.reset()

        if name is None:
            return None
        self.counts[name] = self.counts.get(name, 0) + 1
        self.last_gesture = name
        event = GestureEvent(name=name, t=pose.t,
                             angle=pose.yaw if "left" in name or "right" in name
                             else pose.pitch)
        if self._on_gesture:
            self._on_gesture(event)
        return event

    @property
    def engaged(self) -> bool:
        """True while a gesture is in progress (used to guard the cursor)."""
        return self._yaw_fsm.state != _AxisFSM.IDLE or \
            self._pitch_fsm.state != _AxisFSM.IDLE

    def reset(self) -> None:
        self._yaw_fsm.reset()
        self._pitch_fsm.reset()
        self._sy = self._sp = None
