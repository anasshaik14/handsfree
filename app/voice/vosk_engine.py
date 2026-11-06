"""Voice engine 1: Vosk (Kaldi) — fully offline streaming recognition.

Low latency, small footprint, runs continuously on CPU. The default engine.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

from app.core.interfaces import VoiceEngine

log = logging.getLogger(__name__)


class VoskEngine(VoiceEngine):
    name = "vosk"

    def __init__(self, model_path: str, device: Optional[int] = None,
                 samplerate: int = 16000) -> None:
        self.model_path = model_path
        self.device = device
        self.samplerate = samplerate
        self._running = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._q: queue.Queue = queue.Queue()

    @staticmethod
    def available() -> bool:
        try:
            import vosk  # noqa: F401
            import sounddevice  # noqa: F401
            return True
        except Exception:
            return False

    def start(self, on_final: Callable[[str], None],
              on_partial: Optional[Callable[[str], None]] = None) -> bool:
        if self._running:
            return True
        if not Path(self.model_path).exists():
            log.error("vosk model not found at %s", self.model_path)
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(on_final, on_partial),
            name="vosk", daemon=True)
        self._thread.start()
        self._running = True
        return True

    def _loop(self, on_final, on_partial) -> None:
        try:
            import sounddevice as sd
            import vosk
            vosk.SetLogLevel(-1)
            model = vosk.Model(self.model_path)
            rec = vosk.KaldiRecognizer(model, self.samplerate)
            rec.SetWords(False)

            def callback(indata, frames, t, status):
                if status:
                    log.debug("audio status: %s", status)
                self._q.put(bytes(indata))

            with sd.RawInputStream(samplerate=self.samplerate, blocksize=4000,
                                   device=self.device, dtype="int16",
                                   channels=1, callback=callback):
                while not self._stop.is_set():
                    try:
                        data = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if rec.AcceptWaveform(data):
                        text = json.loads(rec.Result()).get("text", "").strip()
                        if text:
                            on_final(text)
                    elif on_partial:
                        partial = json.loads(rec.PartialResult()).get("partial", "")
                        if partial:
                            on_partial(partial)
        except Exception:
            log.exception("vosk engine crashed")
        finally:
            self._running = False

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._running = False

    @property
    def running(self) -> bool:
        return self._running
