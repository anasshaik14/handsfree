"""Voice engine 2: Whisper via faster-whisper (CTranslate2, no PyTorch).

Higher accuracy than Vosk at the cost of latency: audio is segmented with an
energy-based VAD and each utterance is transcribed as a chunk.
Optional dependency — install with:  pip install faster-whisper
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np

from app.core.interfaces import VoiceEngine

log = logging.getLogger(__name__)


class WhisperEngine(VoiceEngine):
    name = "whisper"

    def __init__(self, model_name: str = "base.en", device: Optional[int] = None,
                 samplerate: int = 16000) -> None:
        self.model_name = model_name
        self.device = device
        self.samplerate = samplerate
        self._running = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def available() -> bool:
        try:
            import faster_whisper  # noqa: F401
            import sounddevice  # noqa: F401
            return True
        except Exception:
            return False

    def start(self, on_final: Callable[[str], None],
              on_partial: Optional[Callable[[str], None]] = None) -> bool:
        if self._running:
            return True
        if not self.available():
            log.error("faster-whisper not installed (pip install faster-whisper)")
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(on_final,),
                                        name="whisper", daemon=True)
        self._thread.start()
        self._running = True
        return True

    def _loop(self, on_final) -> None:
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
            model = WhisperModel(self.model_name, device="cpu",
                                 compute_type="int8")
            q: queue.Queue = queue.Queue()

            def callback(indata, frames, t, status):
                q.put(indata.copy())

            block = int(self.samplerate * 0.1)   # 100 ms blocks
            # Energy VAD state
            speech: list[np.ndarray] = []
            silence_blocks = 0
            noise_floor = 0.01

            with sd.InputStream(samplerate=self.samplerate, blocksize=block,
                                device=self.device, dtype="float32",
                                channels=1, callback=callback):
                while not self._stop.is_set():
                    try:
                        chunk = q.get(timeout=0.2).flatten()
                    except queue.Empty:
                        continue
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    noise_floor = 0.995 * noise_floor + 0.005 * min(rms, 0.02)
                    is_speech = rms > max(noise_floor * 3.0, 0.012)
                    if is_speech:
                        speech.append(chunk)
                        silence_blocks = 0
                    elif speech:
                        speech.append(chunk)
                        silence_blocks += 1
                        # ~600 ms of silence ends the utterance
                        if silence_blocks >= 6:
                            audio = np.concatenate(speech)
                            speech, silence_blocks = [], 0
                            if len(audio) >= self.samplerate * 0.4:
                                segments, _ = model.transcribe(
                                    audio, language="en", beam_size=1,
                                    vad_filter=True)
                                text = " ".join(s.text for s in segments).strip()
                                if text:
                                    on_final(text)
        except Exception:
            log.exception("whisper engine crashed")
        finally:
            self._running = False

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._running = False

    @property
    def running(self) -> bool:
        return self._running


class WebSpeechEngine(VoiceEngine):
    """Voice engine 3: Web Speech API.

    Recognition happens in the browser dashboard (Chrome/Edge); final
    transcripts arrive over the WebSocket and are injected via `inject()`.
    This class only tracks engine state so switching works uniformly.
    """

    name = "webspeech"

    def __init__(self) -> None:
        self._running = False
        self._on_final: Optional[Callable[[str], None]] = None

    @staticmethod
    def available() -> bool:
        return True  # availability is decided by the browser

    def start(self, on_final: Callable[[str], None],
              on_partial: Optional[Callable[[str], None]] = None) -> bool:
        self._on_final = on_final
        self._running = True
        return True

    def inject(self, text: str) -> None:
        """Called by the web server when the browser sends a transcript."""
        if self._running and self._on_final and text.strip():
            self._on_final(text.strip())

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running
