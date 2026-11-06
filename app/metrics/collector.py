"""Runtime metrics: FPS, per-stage latency, CPU/RAM, jitter, event counts.

Every pipeline stage is timed each frame; a background sampler adds process
CPU/memory once a second; one summary line per second is appended to the
session JSONL so reports can compare providers on identical measures.
"""
from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class StageTimer:
    """Context manager: with metrics.stage('gaze'): ..."""

    def __init__(self, collector: "MetricsCollector", name: str) -> None:
        self._c = collector
        self._name = name

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._c.record_stage(self._name, (time.perf_counter() - self._t0) * 1000.0)
        return False


class MetricsCollector:
    def __init__(self, session_dir: str | Path = "sessions",
                 log_enabled: bool = True) -> None:
        self._stages: dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        self._frame_times: deque = deque(maxlen=300)
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()
        self._cpu = 0.0
        self._mem_mb = 0.0
        self._stop = threading.Event()
        self._session_path: Optional[Path] = None
        self._session_file = None
        self._log_enabled = log_enabled
        self._context: dict[str, str] = {}   # active provider names etc.
        self._sys_thread = threading.Thread(target=self._sample_system,
                                            name="metrics-sys", daemon=True)
        self._sys_thread.start()
        if log_enabled:
            self.new_session()
        self._writer = threading.Thread(target=self._write_loop,
                                        name="metrics-writer", daemon=True)
        self._writer.start()

    # ---- session -----------------------------------------------------------
    def new_session(self, tag: str = "") -> Path:
        d = Path("sessions")
        d.mkdir(exist_ok=True)
        name = time.strftime("session_%Y%m%d_%H%M%S")
        if tag:
            name += f"_{tag}"
        with self._lock:
            if self._session_file:
                self._session_file.close()
            self._session_path = d / f"{name}.jsonl"
            self._session_file = open(self._session_path, "a", encoding="utf-8")
        return self._session_path

    def set_context(self, **kwargs: str) -> None:
        """Record which providers are active; stamped onto every log line."""
        with self._lock:
            self._context.update({k: str(v) for k, v in kwargs.items()})

    # ---- recording -----------------------------------------------------------
    def stage(self, name: str) -> StageTimer:
        return StageTimer(self, name)

    def record_stage(self, name: str, ms: float) -> None:
        with self._lock:
            self._stages[name].append(ms)

    def frame_done(self) -> None:
        with self._lock:
            self._frame_times.append(time.perf_counter())

    def count(self, name: str, inc: int = 1) -> None:
        with self._lock:
            self._counters[name] += inc

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def log_event(self, kind: str, data: dict) -> None:
        line = {"type": "event", "kind": kind, "ts": time.time(), **data}
        with self._lock:
            ctx = dict(self._context)
            f = self._session_file
        line["ctx"] = ctx
        if f:
            try:
                f.write(json.dumps(line) + "\n")
            except Exception:
                pass

    # ---- system sampler --------------------------------------------------
    def _sample_system(self) -> None:
        try:
            import psutil
            proc = psutil.Process()
            proc.cpu_percent()  # prime
            while not self._stop.is_set():
                time.sleep(1.0)
                with self._lock:
                    self._cpu = proc.cpu_percent()
                    self._mem_mb = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            log.exception("system sampler stopped")

    # ---- periodic session writer ------------------------------------------
    def _write_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(1.0)
            if not self._log_enabled:
                continue
            snap = self.snapshot()
            snap["type"] = "sample"
            snap["ts"] = time.time()
            with self._lock:
                f = self._session_file
            if f:
                try:
                    f.write(json.dumps(snap) + "\n")
                    f.flush()
                except Exception:
                    pass

    # ---- reading ------------------------------------------------------------
    @staticmethod
    def _stats(values) -> dict:
        vals = list(values)
        if not vals:
            return {"mean": 0.0, "p95": 0.0}
        vals.sort()
        return {
            "mean": round(statistics.fmean(vals), 3),
            "p95": round(vals[min(int(len(vals) * 0.95), len(vals) - 1)], 3),
        }

    def fps(self) -> float:
        with self._lock:
            times = list(self._frame_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return round((len(times) - 1) / span, 1) if span > 0 else 0.0

    def snapshot(self) -> dict:
        with self._lock:
            stages = {k: self._stats(v) for k, v in self._stages.items()}
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            cpu, mem = self._cpu, self._mem_mb
            ctx = dict(self._context)
        return {
            "fps": self.fps(),
            "stages_ms": stages,
            "counters": counters,
            "gauges": {k: round(v, 3) for k, v in gauges.items()},
            "cpu_percent": round(cpu, 1),
            "memory_mb": round(mem, 1),
            "ctx": ctx,
        }

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._session_file:
                self._session_file.close()
                self._session_file = None
