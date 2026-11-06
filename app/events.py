"""Tiny thread-safe publish/subscribe event bus.

Topics used across the app:
  gaze.sample, cursor.state, blink.frame, blink.event, head.pose, head.gesture,
  voice.partial, voice.command, action.executed, system.speak, system.toast,
  metrics.tick, control.paused, control.resumed
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[str, Any], None]]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, topic: str, fn: Callable[[str, Any], None]) -> None:
        """Subscribe to a topic; use '*' to receive every event."""
        with self._lock:
            self._subs[topic].append(fn)

    def unsubscribe(self, topic: str, fn: Callable[[str, Any], None]) -> None:
        with self._lock:
            if fn in self._subs.get(topic, []):
                self._subs[topic].remove(fn)

    def publish(self, topic: str, payload: Any = None) -> None:
        with self._lock:
            handlers = list(self._subs.get(topic, [])) + list(self._subs.get("*", []))
        for fn in handlers:
            try:
                fn(topic, payload)
            except Exception:  # noqa: BLE001 - a bad handler must not kill the pipeline
                log.exception("event handler failed for topic %s", topic)


# A single process-wide bus keeps wiring trivial without hidden globals in modules.
bus = EventBus()
