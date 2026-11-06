"""Feedback tones (blink-click beeps) that survive Bluetooth outputs.

winsound.Beep writes ~70 ms of audio and returns; Bluetooth outputs (e.g.
earbuds) take 200-500 ms to wake their audio link, so short beeps — and even
short TTS replies — are swallowed before the link starts streaming. Two
counter-measures:

* a persistent, silent output stream keeps the sink's audio link awake, so
  every short sound (ours AND the SAPI TTS) is actually delivered;
* tones are rendered into that same always-open stream, so they never race
  the wake-up.

Falls back to winsound.Beep when no output stream can be opened (also covers
the default device disappearing mid-session, e.g. earbuds switched off).
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

_lock = threading.Lock()
_stream = None                 # persistent sounddevice.OutputStream
_samplerate = 48000
_pending = np.zeros(0, np.float32)   # samples the callback still has to play
_next_retry = 0.0              # rate-limit stream reopen attempts


def _callback(outdata, frames, _time, _status) -> None:
    global _pending
    with _lock:
        n = min(frames, len(_pending))
        outdata[:n, 0] = _pending[:n]
        outdata[n:, 0] = 0.0
        _pending = _pending[n:]


def _ensure_stream() -> bool:
    """Open (or reopen) the persistent output stream on the default device."""
    global _stream, _samplerate, _next_retry
    if _stream is not None:
        return True
    if time.monotonic() < _next_retry:
        return False
    _next_retry = time.monotonic() + 5.0
    try:
        import sounddevice as sd
        dev = sd.default.device[1]
        _samplerate = int(sd.query_devices(dev)["default_samplerate"])
        stream = sd.OutputStream(samplerate=_samplerate, channels=1,
                                 dtype="float32", callback=_callback)
        stream.start()
        _stream = stream
        log.info("audio feedback stream open (%s @ %d Hz)",
                 sd.query_devices(dev)["name"], _samplerate)
        return True
    except Exception:
        log.warning("audio feedback stream unavailable — falling back to "
                    "winsound", exc_info=True)
        return False


def start() -> None:
    """Open the keep-alive stream up front (call once at pipeline start)."""
    _ensure_stream()


def stop() -> None:
    global _stream
    stream, _stream = _stream, None
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


def play(seq: list[tuple[int, int]], volume: float = 0.4) -> None:
    """Queue a tone sequence [(freq_hz, duration_ms), ...]; non-blocking."""
    if not seq:
        return
    if not _ensure_stream():
        threading.Thread(target=_winsound_fallback, args=(list(seq),),
                         daemon=True).start()
        return
    global _pending
    sr = _samplerate
    parts = []
    gap = np.zeros(int(sr * 0.02), np.float32)
    for freq, ms in seq:
        t = np.arange(int(sr * ms / 1000.0), dtype=np.float32) / sr
        tone = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        edge = int(sr * 0.005)  # 5 ms fade in/out so tones don't click
        if len(tone) > 2 * edge:
            ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
            tone[:edge] *= ramp
            tone[-edge:] *= ramp[::-1]
        parts.append(tone)
        parts.append(gap)
    samples = np.concatenate(parts)
    with _lock:
        _pending = np.concatenate([_pending, samples]) \
            if len(_pending) else samples


def _winsound_fallback(seq: list[tuple[int, int]]) -> None:
    try:
        import winsound
        for freq, ms in seq:
            winsound.Beep(freq, ms)
    except Exception:
        pass
