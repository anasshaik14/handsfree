"""Voice engine "custom": our own keyword-spotting CNN, trained from scratch
on Google Speech Commands v2 (log-mel spectrogram -> word class). For the
words it knows it produces the command directly (fast, no ASR); for anything
else — and for the free-text commands that need real transcription (open X,
type X, search X) — it forwards the utterance to the configured base ASR
(vosk / whisper), which becomes the fallback.

The Speech Commands vocabulary only partly overlaps the app's command set;
SC_TO_PHRASE maps the useful words to canonical phrases the intent parser
already understands. Everything else routes to the base engine.

Train it with:  python scripts/train_voice.py
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from app.core.interfaces import VoiceEngine
from app.ml import runtime
from app.voice.whisper_engine import WebSpeechEngine

log = logging.getLogger(__name__)

WEIGHTS_PATH = runtime.MODELS_DIR / "voice_kws.pt"
LABELS_PATH = runtime.MODELS_DIR / "voice_kws.labels.json"
CONFIDENCE_THRESHOLD = 0.80
SAMPLE_RATE = 16000
N_MELS = 40

# Speech Commands word -> canonical app phrase (must match a phrase the intent
# parser / fuzzy table recognizes). Words not listed fall back to full ASR.
SC_TO_PHRASE = {
    "up": "scroll up", "down": "scroll down",
    "stop": "pause control", "go": "resume control",
    "yes": "enter", "no": "escape",
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


class CustomVoiceKWS(VoiceEngine):
    name = "custom"

    def __init__(self, base_engine: str = "vosk", cfg_get=None,
                 vosk_model_path: str = "", whisper_model: str = "base.en",
                 device: Optional[int] = None, samplerate: int = SAMPLE_RATE) -> None:
        self._cfg = cfg_get
        self._base_engine_name = base_engine
        self._vosk_model_path = vosk_model_path
        self._whisper_model_name = whisper_model
        self.device = device
        self.samplerate = samplerate
        self._running = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._webspeech: Optional[WebSpeechEngine] = None
        self._vosk_model = None
        self._whisper_model = None
        self._melspec = None

        self._model = runtime.load_scripted(WEIGHTS_PATH)
        self._labels: List[str] = []
        if LABELS_PATH.exists():
            try:
                self._labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
            except Exception:
                log.exception("failed to read %s", LABELS_PATH)

    @property
    def available(self) -> bool:
        return self._model is not None and bool(self._labels)

    @staticmethod
    def available_static() -> bool:
        return True

    # -- keyword spotting -------------------------------------------------
    def _logmel(self, audio: np.ndarray):
        import torch
        import torchaudio
        if self._melspec is None:
            self._melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.samplerate, n_fft=400, hop_length=200, n_mels=N_MELS)
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
        mel = self._melspec(wav)
        return torch.log(mel + 1e-6)   # (1, n_mels, T)

    def classify(self, audio: np.ndarray) -> Optional[str]:
        """Canonical phrase for a recognized command word, else None."""
        if not self.available:
            return None
        import torch
        with torch.no_grad():
            logits = self._model(self._logmel(audio).unsqueeze(0)).squeeze(0).cpu().numpy()
        probs = _softmax(logits)
        i = int(probs.argmax())
        if float(probs[i]) < CONFIDENCE_THRESHOLD:
            return None
        return SC_TO_PHRASE.get(self._labels[i])

    def _transcribe_fallback(self, audio: np.ndarray) -> str:
        if self._base_engine_name == "whisper":
            from faster_whisper import WhisperModel
            if self._whisper_model is None:
                self._whisper_model = WhisperModel(self._whisper_model_name,
                                                   device="cpu", compute_type="int8")
            segments, _ = self._whisper_model.transcribe(
                audio, language="en", beam_size=1, vad_filter=True)
            return " ".join(s.text for s in segments).strip()
        import json as _json

        import vosk
        if self._vosk_model is None:
            vosk.SetLogLevel(-1)
            self._vosk_model = vosk.Model(self._vosk_model_path)
        rec = vosk.KaldiRecognizer(self._vosk_model, self.samplerate)
        rec.SetWords(False)
        rec.AcceptWaveform((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return _json.loads(rec.FinalResult()).get("text", "").strip()

    # -- engine lifecycle -------------------------------------------------
    def start(self, on_final: Callable[[str], None],
              on_partial: Optional[Callable[[str], None]] = None) -> bool:
        if self._running:
            return True
        if self._base_engine_name == "webspeech":
            self._webspeech = WebSpeechEngine()
            self._running = self._webspeech.start(on_final, on_partial)
            return self._running
        if self._base_engine_name == "vosk" and not Path(self._vosk_model_path or "").exists():
            log.error("vosk model not found at %s", self._vosk_model_path)
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(on_final,),
                                        name="custom_voice", daemon=True)
        self._thread.start()
        self._running = True
        return True

    def inject(self, text: str) -> None:
        if self._webspeech is not None:
            self._webspeech.inject(text)

    def _loop(self, on_final) -> None:
        try:
            import sounddevice as sd
            q: queue.Queue = queue.Queue()

            def callback(indata, frames, t, status):
                q.put(indata.copy())

            block = int(self.samplerate * 0.1)
            speech: List[np.ndarray] = []
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
                    if rms > max(noise_floor * 3.0, 0.012):
                        speech.append(chunk)
                        silence_blocks = 0
                    elif speech:
                        speech.append(chunk)
                        silence_blocks += 1
                        if silence_blocks >= 6:
                            audio = np.concatenate(speech)
                            speech, silence_blocks = [], 0
                            if len(audio) >= self.samplerate * 0.4:
                                text = self.classify(audio) or self._transcribe_fallback(audio)
                                if text:
                                    on_final(text)
        except Exception:
            log.exception("custom voice engine crashed")
        finally:
            self._running = False

    def stop(self) -> None:
        if self._webspeech is not None:
            self._webspeech.stop()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._running = False

    @property
    def running(self) -> bool:
        return self._running
