"""Threaded webcam capture that always serves the latest frame.

Capturing in its own thread decouples camera I/O (blocking, ~33 ms) from the
processing loop, so a slow frame never stalls the cursor and the pipeline
never processes stale frames.

Includes health monitoring with automatic recovery: some UVC webcams (seen
with a Logitech C270) can enter a bad USB state where they deliver black /
corrupt frames at ~1 fps until the device is closed and reopened. The capture
loop detects that signature (starved or near-black frames for several
seconds) and transparently reopens the device, cycling capture backends.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# health thresholds
_DARK_MEAN = 4.0          # mean pixel value below this = black/corrupt frame
_SLOW_DT = 0.35           # seconds between frames that counts as starving
_BAD_SCORE_LIMIT = 12     # consecutive bad signals before reopening
_RECOVER_COOLDOWN = 6.0   # min seconds between reopen attempts


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 fps: int = 30) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps_requested = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_t: float = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps_actual = 0.0
        self._backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        self._backend_i = 0
        self.healthy = True
        self.recoveries = 0

    # -- device handling ---------------------------------------------------
    def _open(self) -> bool:
        backend = self._backends[self._backend_i % len(self._backends)]
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps_requested)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return True

    def _reopen(self) -> None:
        """Recover from a wedged device by closing and reopening it."""
        self.recoveries += 1
        log.warning("camera unhealthy (black/starved frames) — reopening "
                    "(attempt %d)", self.recoveries)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        time.sleep(0.6)   # let the driver actually release the device
        # try each backend until one opens
        for _ in range(len(self._backends)):
            if self._open():
                return
            self._backend_i += 1
        log.error("camera reopen failed on all backends")

    def start(self) -> bool:
        opened = False
        for _ in range(len(self._backends)):
            if self._open():
                opened = True
                break
            self._backend_i += 1
        if not opened:
            log.error("camera %s could not be opened", self.index)
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return True

    # -- capture loop ----------------------------------------------------------
    def _loop(self) -> None:
        last = time.perf_counter()
        ema = 0.0
        bad_score = 0
        last_recover = 0.0
        while not self._stop.is_set():
            if self._cap is None:
                if time.perf_counter() - last_recover > _RECOVER_COOLDOWN:
                    last_recover = time.perf_counter()
                    self._reopen()
                else:
                    time.sleep(0.2)
                continue
            ok, frame = self._cap.read()
            t = time.perf_counter()
            dt = t - last
            if not ok or frame is None:
                bad_score += 3
                time.sleep(0.02)
            else:
                last = t
                if dt > 0:
                    ema = 0.9 * ema + 0.1 * (1.0 / dt) if ema else 1.0 / dt
                # cheap corruption signature check (~0.2 ms)
                dark = float(frame.mean()) < _DARK_MEAN
                slow = dt > _SLOW_DT
                if dark or slow:
                    bad_score = min(bad_score + 1, _BAD_SCORE_LIMIT + 4)
                else:
                    bad_score = max(bad_score - 2, 0)
                with self._lock:
                    self._frame = frame
                    self._frame_t = t
                    self._seq += 1
                    self._fps_actual = ema

            self.healthy = bad_score < _BAD_SCORE_LIMIT
            if not self.healthy and \
                    time.perf_counter() - last_recover > _RECOVER_COOLDOWN:
                last_recover = time.perf_counter()
                self._reopen()
                bad_score = 0
                last = time.perf_counter()
                ema = 0.0

    # -- consumers ------------------------------------------------------------
    def latest(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Returns (frame, capture_time, sequence). Frame is shared, do not mutate."""
        with self._lock:
            return self._frame, self._frame_t, self._seq

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps_actual

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
