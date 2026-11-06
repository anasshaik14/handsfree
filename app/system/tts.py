"""Offline text-to-speech via Windows SAPI (System.Speech through PowerShell).

Zero extra dependencies: uses the .NET SpeechSynthesizer that ships with every
Windows install. One persistent PowerShell worker process speaks all
utterances — spawning PowerShell per phrase added ~2 s of latency and, on
Bluetooth outputs, short replies died entirely while the audio link woke up
(see app.system.audio_feedback, whose keep-alive stream also protects speech).

A single worker thread serializes utterances, and `is_speaking()` lets the
voice pipeline drop transcripts of our own voice (speaker -> microphone echo
would otherwise trigger command loops).
"""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time

log = logging.getLogger(__name__)

_q: queue.Queue = queue.Queue()
_speaking = threading.Event()
_started = False
_start_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_proc_out: queue.Queue | None = None

# Reads one line per utterance; re-binds to the CURRENT default audio device
# before each Speak so the voice follows output-device changes mid-session.
_PS_WORKER = (
    "$ErrorActionPreference = 'Continue'; "
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Rate = 1; "
    "while ($true) { "
    "  $line = [Console]::In.ReadLine(); "
    "  if ($null -eq $line) { break } "
    "  if ($line -ne '') { "
    "    try { $s.SetOutputToDefaultAudioDevice(); $s.Speak($line) } catch {} "
    "  } "
    "  [Console]::Out.WriteLine('__DONE__'); "
    "}"
)


def _ensure_proc() -> subprocess.Popen | None:
    global _proc, _proc_out
    if _proc is not None and _proc.poll() is None:
        return _proc
    try:
        _proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", _PS_WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # a blocking readline could wedge the tts thread (and with it the
        # is_speaking() echo gate) forever — read via a queue instead
        _proc_out = queue.Queue()
        threading.Thread(target=_drain_stdout, args=(_proc, _proc_out),
                         daemon=True, name="sapi-tts-out").start()
        log.info("SAPI tts worker started (pid %d)", _proc.pid)
    except Exception:
        log.exception("SAPI tts worker failed to start")
        _proc = None
    return _proc


def _drain_stdout(proc: subprocess.Popen, out: queue.Queue) -> None:
    try:
        for line in proc.stdout:
            out.put(line)
    except Exception:
        pass
    out.put(None)   # EOF marker: the worker died


def _kill_proc() -> None:
    global _proc
    proc, _proc = _proc, None
    if proc is not None:
        try:
            proc.kill()
        except Exception:
            pass


def _speak_once(text: str) -> bool:
    """Send one utterance to the worker and wait until it finishes speaking."""
    proc = _ensure_proc()
    if proc is None:
        return False
    # single line only: the worker reads exactly one line per utterance
    safe = " ".join(text.split())
    # plain ASCII survives the console-codepage pipe on every locale
    safe = safe.encode("ascii", "ignore").decode()
    out = _proc_out
    try:
        proc.stdin.write(safe + "\n")
        proc.stdin.flush()
    except Exception:
        return False
    # generous per-utterance deadline; a hung worker gets killed and rebuilt
    deadline = time.monotonic() + 10.0 + 0.1 * len(safe)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_proc()
            return False
        try:
            line = out.get(timeout=remaining)
        except queue.Empty:
            continue
        if line is None:        # worker died
            return False
        if line.strip() == "__DONE__":
            return True


def _worker() -> None:
    while True:
        text = _q.get()
        if not text:
            continue
        _speaking.set()
        try:
            if not _speak_once(text):
                _kill_proc()
                if not _speak_once(text):     # one retry with a fresh worker
                    log.warning("SAPI tts failed to speak %r", text)
        except Exception:
            log.exception("SAPI tts failed")
        finally:
            # grace period: the mic may still catch the tail of our voice
            time.sleep(0.3)
            if _q.empty():
                _speaking.clear()


def speak(text: str) -> None:
    """Queue a phrase; returns immediately (speech happens on the worker)."""
    global _started
    if not text or not text.strip():
        return
    with _start_lock:
        if not _started:
            threading.Thread(target=_worker, daemon=True, name="sapi-tts").start()
            _started = True
    _q.put(text.strip())


def is_speaking() -> bool:
    return _speaking.is_set()
