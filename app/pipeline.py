"""The orchestrator: one camera loop drives all four modalities at once.

Frame flow (single FaceMesh inference feeds everything):

    camera -> FaceMesh -> head pose -> gesture FSM -> app/tab switching
                       -> blink     -> blink FSM   -> clicks
                       -> gaze      -> stabilizer  -> OS cursor

Voice runs in its own audio thread. The pipeline owns provider registries and
supports hot-swapping any provider at runtime under a lock.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.core.gate import HysteresisGate
from app.core.types import BlinkEvent, FaceResult, GestureEvent, now
from app.events import bus
from app.blink.blink_fsm import BlinkFSM
from app.blink.custom_cnn import CustomBlinkCNN
from app.blink.detectors import AdaptiveEARBlink, ELABlink, StandardEARBlink
from app.fatigue.monitor import FatigueMonitor
from app.gaze.calibration import PolynomialModel, calibration_points
from app.gaze.cursor_controller import CursorController
from app.gaze.custom_cnn import CustomGazeCNN
from app.gaze.iris_geometric import IrisGeometricGaze
from app.gaze.iris_polynomial import IrisPolynomialGaze
from app.gaze.pupil_classical import PupilClassicalGaze
from app.hands.gesture_classify import classify as classify_hand
from app.hands.hand_gesture_fsm import HandGestureFSM
from app.hands.hand_landmarker import HandLandmarkerService
from app.headpose.custom_cnn import CustomHeadPoseCNN
from app.headpose.gesture_fsm import GestureFSM
from app.headpose.nod_shake import NodShakeDetector
from app.headpose.providers import (HeuristicHeadPose, MPGeometryHeadPose,
                                    SolvePnPHeadPose)
from app.system import audio_feedback, win_input
from app.vision.camera import Camera
from app.vision.face_mesh import FaceMeshService
from app.vision.object_detector import ObjectDetectorService
from app.voice.actions import ActionExecutor
from app.voice.custom_kws import CustomVoiceKWS
from app.voice.vosk_engine import VoskEngine
from app.voice.whisper_engine import WebSpeechEngine, WhisperEngine

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg, metrics) -> None:
        self.cfg = cfg
        self.metrics = metrics
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        cam = cfg.get("camera", {})
        self.camera = Camera(cam.get("index", 0), cam.get("width", 640),
                             cam.get("height", 480), cam.get("fps", 30))
        self.mesh: Optional[FaceMeshService] = None  # built on start (heavy)
        self.hand_mesh: Optional[HandLandmarkerService] = None
        self.object_detector: Optional[ObjectDetectorService] = None

        self.cursor = CursorController(cfg)
        self.executor = ActionExecutor(cfg, cursor=self.cursor)
        self.executor.register_hook("calibrate", self.run_calibration)
        self.executor.register_hook("set_center", self.set_center)
        self.executor.register_hook("set_neutral", self.set_neutral)
        self.executor.register_hook("generate_report", self._report_hook)
        self.executor.register_hook("open_dashboard", self._open_dashboard_hook)

        # ---- provider registries (Strategy pattern) ----
        # "custom" = our own model trained from scratch on a public dataset
        # (the primary), the rest are the classical/pretrained fallbacks.
        self.GAZE = {
            "custom": lambda: CustomGazeCNN(cfg.get),
            "iris_geometric": lambda: IrisGeometricGaze(cfg.get),
            "iris_polynomial": lambda: IrisPolynomialGaze(cfg.get),
            "pupil_classical": lambda: PupilClassicalGaze(cfg.get),
        }
        self.BLINK = {
            "custom": lambda: CustomBlinkCNN(cfg.get),
            "ear_standard": lambda: StandardEARBlink(cfg.get),
            "ear_adaptive": lambda: AdaptiveEARBlink(cfg.get),
            "ela": lambda: ELABlink(cfg.get),
        }
        self.HEADPOSE = {
            "custom": lambda: CustomHeadPoseCNN(cfg.get),
            "solvepnp": SolvePnPHeadPose,
            "mp_geometry": MPGeometryHeadPose,
            "landmark_heuristic": HeuristicHeadPose,
        }
        self.VOICE = {
            "custom": lambda: CustomVoiceKWS(
                base_engine=cfg.get("voice.custom.base_engine", "vosk"), cfg_get=cfg.get,
                vosk_model_path=cfg.get("voice.vosk_model"),
                whisper_model=cfg.get("voice.whisper_model", "base.en"),
                device=cfg.get("voice.device")),
            "vosk": lambda: VoskEngine(cfg.get("voice.vosk_model"),
                                       cfg.get("voice.device")),
            "whisper": lambda: WhisperEngine(cfg.get("voice.whisper_model", "base.en"),
                                             cfg.get("voice.device")),
            "webspeech": WebSpeechEngine,
        }

        self.gaze = self.GAZE[self._valid("gaze.provider", self.GAZE,
                                          "custom", "iris_geometric")]()
        self.blink = self.BLINK[self._valid("blink.provider", self.BLINK,
                                            "custom", "ear_adaptive")]()
        self.headpose = self.HEADPOSE[self._valid("headpose.provider", self.HEADPOSE,
                                                  "custom", "mp_geometry")]()
        self.voice = None
        self.blink_fsm = BlinkFSM(cfg.get, on_event=self._on_blink_event)
        self.gesture_fsm = GestureFSM(cfg.get, on_gesture=self._on_gesture)
        self.hand_fsm = HandGestureFSM(cfg.get)
        self.nod_shake = NodShakeDetector(cfg.get, on_gesture=self._on_nod_shake)
        self.fatigue = FatigueMonitor(cfg.get, on_alert=self._on_fatigue_alert)

        # shared state for UI / calibration
        self._last_face: Optional[FaceResult] = None
        self._last_pose = None
        self._last_blink = None
        self._last_gaze = None
        self._last_voice_text = ""
        self._last_voice_reply = ""
        self._last_action = ""
        self._calibrating = False
        self._calib_status = ""
        self._last_calib = ""   # last calibration outcome, shown in the UI
        self._preview_jpeg: Optional[bytes] = None
        self._preview_at = 0.0
        self._preview_last = 0.0
        self._frame_i = 0
        self._object_busy = threading.Event()
        self._gaze_gate = HysteresisGate(low=0.12, high=0.20)
        self._hand_paused = False
        self._last_objects: list = []
        self._last_object_scan = 0.0
        self._shutdown_requested = threading.Event()
        self._screen = win_input.screen_size()   # for dynamic calibration
        self.started = False
        self.camera_ok = False

        bus.subscribe("voice.command", self._on_voice_command)
        bus.subscribe("blink.event", self._on_blink_for_fatigue)
        self._update_context()

    @property
    def shutdown_requested(self) -> bool:
        """True once a peace-sign hold has asked the process to exit."""
        return self._shutdown_requested.is_set()

    def _valid(self, key: str, table: dict, preferred: str,
              fallback: Optional[str] = None) -> str:
        """`preferred` is the requested default (e.g. "custom"); a `fallback`
        distinct from `preferred` marks it as an availability-gated choice --
        granted only if the instantiated provider reports `.available` True
        (its trained weights loaded and torch is importable). No trained
        model on disk -> not available -> silently keeps `fallback`, so a
        fresh clone runs on the classical providers exactly as before.
        """
        fallback = fallback or preferred
        v = self.cfg.get(key, preferred)
        if v not in table:
            return fallback
        if v == "custom" and fallback != "custom":
            try:
                available = bool(getattr(table[v](), "available", False))
            except Exception:
                log.exception("custom-model availability check failed for %s", key)
                available = False
            if not available:
                return fallback
        return v

    # ------------------------------------------------------------------ start
    def start(self) -> bool:
        self.camera_ok = self.camera.start()
        if not self.camera_ok:
            log.error("camera failed to start — vision modalities disabled")
        self.mesh = FaceMeshService()
        if self.cfg.get("hands.enabled", True):
            try:
                self.hand_mesh = HandLandmarkerService()
            except Exception:
                log.exception("hand landmarker unavailable — hand gestures disabled")
        if self.cfg.get("objects.enabled", True):
            try:
                self.object_detector = ObjectDetectorService(
                    confidence=float(self.cfg.get("objects.confidence", 0.5)))
            except Exception:
                log.exception("object detector unavailable — object detection disabled")
        if self.cfg.get("audio.keepalive", True):
            # keeps Bluetooth outputs awake so beeps/TTS aren't swallowed
            audio_feedback.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="pipeline",
                                        daemon=True)
        self._thread.start()
        self.start_voice(self._valid("voice.engine", self.VOICE, "custom", "vosk"))
        self.started = True
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self.voice:
            self.voice.stop()
        self.camera.stop()
        if self.mesh:
            self.mesh.close()
        if self.hand_mesh:
            self.hand_mesh.close()
        if self.object_detector:
            self.object_detector.close()
        audio_feedback.stop()

    # ------------------------------------------------------------- switching
    def switch(self, kind: str, name: str) -> bool:
        with self._lock:
            if kind == "gaze" and name in self.GAZE:
                self.gaze = self.GAZE[name]()
                self.cursor.stab.reset()
                self.cfg.set("gaze.provider", name)
            elif kind == "blink" and name in self.BLINK:
                self.blink = self.BLINK[name]()
                self.blink_fsm.reset()
                self.cfg.set("blink.provider", name)
            elif kind == "headpose" and name in self.HEADPOSE:
                self.headpose = self.HEADPOSE[name]()
                self.gesture_fsm.reset()
                self.cfg.set("headpose.provider", name)
            elif kind == "voice" and name in self.VOICE:
                self.start_voice(name)
                self.cfg.set("voice.engine", name)
            else:
                return False
        self._update_context()
        bus.publish("system.toast", {"text": f"{kind} switched to {name}"})
        return True

    def start_voice(self, name: str) -> bool:
        if self.voice:
            self.voice.stop()
            self.voice = None
        if not self.cfg.get("voice.enabled", True):
            return False
        factory = self.VOICE.get(name)
        if factory is None:
            return False
        engine = factory()
        ok = engine.start(
            on_final=lambda text: self._on_voice_final(text, engine.name),
            on_partial=self._on_voice_partial,
        )
        if ok:
            self.voice = engine
        else:
            bus.publish("system.toast",
                        {"text": f"voice engine '{name}' unavailable"})
        self._update_context()
        return ok

    def _update_context(self) -> None:
        self.metrics.set_context(
            gaze_provider=getattr(self.gaze, "name", "?"),
            blink_provider=getattr(self.blink, "name", "?"),
            headpose_provider=getattr(self.headpose, "name", "?"),
            voice_engine=getattr(self.voice, "name", "off") if self.voice else "off",
        )

    # ------------------------------------------------------------ event sinks
    def _on_voice_partial(self, text: str) -> None:
        bus.publish("voice.partial", {"text": text})
        if self.executor.dictation and self.cfg.get("voice.autocomplete", True):
            from app.voice import completion
            suggestions = completion.suggest(text)
            if suggestions:
                bus.publish("voice.suggestions", {"suggestions": suggestions})

    def _on_voice_final(self, text: str, engine: str) -> None:
        from app.system import tts
        if tts.is_speaking():
            return  # that's our own TTS voice echoing into the microphone
        self._last_voice_text = text
        t0 = now()
        cmd = self.executor.handle_transcript(text, engine)
        self.metrics.record_stage("voice", (now() - t0) * 1000.0)
        self.metrics.count(f"voice_{'ok' if cmd.handled else 'miss'}")
        self.metrics.log_event("voice", {"text": text, "intent": cmd.intent,
                                         "handled": cmd.handled})

    def _on_voice_command(self, _topic, cmd) -> None:
        reply = getattr(cmd, "reply", "")
        self._last_voice_reply = reply
        # Speak replies from the backend (Windows SAPI): works for every
        # engine, with or without a dashboard tab open.
        if reply and self.cfg.get("voice.tts_feedback", True):
            from app.system import tts
            tts.speak(reply)

    def _on_blink_event(self, ev: BlinkEvent) -> None:
        self.metrics.count(f"blink_{ev.kind}")
        self.metrics.log_event("blink", {"kind": ev.kind,
                                         "duration_ms": round(ev.duration * 1000)})
        if ev.kind == "natural" or ev.kind == "cancelled":
            bus.publish("blink.event", ev)
            return
        if not self.cfg.get("blink.enabled", True):
            return
        if self._hand_paused:
            return   # open palm shown: suspend blink clicks
        # Fire the click exactly where the (frozen) cursor sits.
        self.cursor.freeze(0.25)
        action = {"click": lambda: win_input.click("left"),
                  "right_click": lambda: win_input.click("right"),
                  "double_click": win_input.double_click}.get(ev.kind)
        if action:
            action()
            if self.cfg.get("blink.sound_feedback", True):
                self._click_sound(ev.kind)   # non-blocking (queued samples)
            self._maybe_dynamic_calibrate()  # Upgrade 4: learn from this click
            self._last_action = ev.kind
            bus.publish("action.executed", {"source": "blink", "action": ev.kind,
                                            "t": ev.t})
        bus.publish("blink.event", ev)

    def _maybe_dynamic_calibrate(self) -> None:
        """Dynamic calibration (Upgrade 4): treat a deliberate click as a
        fixation whose true target is the click point, and nudge the model
        toward it. Gated on high gaze confidence so a stray blink-click while
        looking away can't poison the fit."""
        if not self.cfg.get("gaze.calibration.dynamic", True):
            return
        gaze = self.gaze
        updater = getattr(gaze, "online_update", None)
        if updater is None or not getattr(gaze, "calibrated", False):
            return
        sample = self._last_gaze
        if sample is None or sample.quality < \
                float(self.cfg.get("gaze.calibration.dynamic_min_quality", 0.5)):
            return
        sw, sh = self._screen
        cx, cy = win_input.cursor_pos()
        tx, ty = cx / max(sw, 1), cy / max(sh, 1)
        if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
            return
        pose = self._last_pose
        yaw = pose.yaw if pose and pose.valid else 0.0
        pitch = pose.pitch if pose and pose.valid else 0.0
        try:
            # raw_nx/raw_ny carry the (hx, hy) iris features behind this sample
            updater(sample.raw_nx, sample.raw_ny, yaw, pitch, tx, ty)
        except Exception:
            log.exception("dynamic calibration update failed")

    def _on_gesture(self, ev: GestureEvent) -> None:
        self.metrics.count(f"gesture_{ev.name}")
        self.metrics.log_event("gesture", {"name": ev.name,
                                           "angle": round(ev.angle, 1)})
        if not self.cfg.get("headpose.enabled", True):
            return
        if self.cfg.get("headpose.guard_cursor", True):
            self.cursor.freeze(0.6)
        action = self.executor.run_gesture(ev.name)
        self._last_action = f"{ev.name} → {action}"
        bus.publish("head.gesture", ev)

    def _on_blink_for_fatigue(self, _topic, ev: BlinkEvent) -> None:
        self.fatigue.on_blink_event(ev)

    def _on_hand_gesture(self, kind: str) -> None:
        self.metrics.count(f"hand_{kind}")
        self.metrics.log_event("hand", {"kind": kind})
        if kind == "thumbs_up":
            win_input.press("volume_up")
            bus.publish("system.toast", {"text": "Volume up (thumbs up)"})
        elif kind == "thumbs_down":
            win_input.press("volume_down")
            bus.publish("system.toast", {"text": "Volume down (thumbs down)"})
        elif kind == "quit":
            bus.publish("system.toast",
                        {"text": "Peace sign held — shutting down HandsFree"})
            from app.system import tts
            tts.speak("Peace sign detected. Shutting down.")
            self._shutdown_requested.set()
            return
        self._last_action = f"hand:{kind}"
        bus.publish("action.executed", {"source": "hand", "action": kind, "t": now()})

    def _on_nod_shake(self, kind: str) -> None:
        self.metrics.count(f"headgesture_{kind}")
        self.metrics.log_event("nod_shake", {"kind": kind})
        # Nod = yes -> press Enter (confirm); shake = no -> press Escape (cancel).
        if kind == "nod":
            text, topic, key = "Yes", "gesture.confirm", "enter"
        else:
            text, topic, key = "No", "gesture.cancel", "escape"
        win_input.press(key)
        bus.publish(topic, {"t": now()})
        bus.publish("system.toast", {"text": f"{kind} detected — {text} ({key})"})
        if self.cfg.get("voice.tts_feedback", True):
            from app.system import tts
            tts.speak(text)
        self._last_action = f"head:{kind} → {key}"

    def _on_fatigue_alert(self, score: float) -> None:
        self.metrics.count("fatigue_alert")
        self.metrics.log_event("fatigue", {"score": round(score, 2)})
        bus.publish("system.toast", {"text": "You seem tired. Take a short break."})
        from app.system import tts
        tts.speak("You seem tired. Take a short break.")

    def _report_hook(self) -> None:
        from app.metrics import report
        report.generate()

    def _open_dashboard_hook(self) -> None:
        import webbrowser
        host = self.cfg.get("server.host", "127.0.0.1")
        port = self.cfg.get("server.port", 8765)
        webbrowser.open(f"http://{host}:{port}/?greet=1")

    # ---------------------------------------------------------------- actions
    def set_center(self) -> bool:
        with self._lock:
            fn = getattr(self.gaze, "set_center", None)
        ok = bool(fn and fn())
        if ok:
            bus.publish("system.toast", {"text": "gaze center set"})
        return ok

    def set_neutral(self) -> bool:
        with self._lock:
            face = self._last_face
            if face is None or not face.ok:
                return False
            ok = self.headpose.set_neutral(face)
        if ok:
            self.gesture_fsm.reset()
            bus.publish("system.toast", {"text": "neutral head pose set"})
        return ok

    # -------------------------------------------------------------- main loop
    def _loop(self) -> None:
        last_seq = -1
        freeze_ms = float(self.cfg.get("gaze.freeze_on_blink_ms", 180))
        while not self._stop.is_set():
            frame, t_cap, seq = self.camera.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq
            t_total0 = time.perf_counter()

            with self.metrics.stage("mesh"):
                face = self.mesh.process(frame, t=t_cap)
            if face is not None:
                face.frame = frame   # BGR image, for custom CNN providers to crop
            self._last_face = face

            with self._lock:
                gaze_p, blink_p, pose_p = self.gaze, self.blink, self.headpose

            # ---- head pose (first: gaze may use it) ----
            pose = None
            if self.cfg.get("headpose.enabled", True) or True:
                # pose always runs: it also feeds gaze pose-assist & calibration
                with self.metrics.stage("headpose"):
                    pose = pose_p.process(face)
            self._last_pose = pose
            if pose is not None and self.cfg.get("headpose.enabled", True):
                self.gesture_fsm.update(pose)
                if self.cfg.get("headpose.guard_cursor", True) and \
                        self.gesture_fsm.engaged:
                    self.cursor.freeze(0.15)
                if self.cfg.get("headpose.nod_shake.enabled", True):
                    self.nod_shake.update(pose)

            # ---- hand gestures (open palm / thumbs up-down / peace) ----
            # Run every Nth frame: the hand landmarker costs as much as the
            # face mesh, and gestures are held for many frames anyway.
            self._frame_i += 1
            stride = max(1, int(self.cfg.get("hands.frame_stride", 2)))
            if self.cfg.get("hands.enabled", True) and self.hand_mesh is not None \
                    and self._frame_i % stride == 0:
                with self.metrics.stage("hands"):
                    hand = self.hand_mesh.process(frame, t=t_cap)
                label = classify_hand(hand.landmarks) if hand.ok else ""
                for kind in self.hand_fsm.update(label, t_cap):
                    self._on_hand_gesture(kind)
                self._hand_paused = self.hand_fsm.paused
            elif not self.cfg.get("hands.enabled", True) or self.hand_mesh is None:
                self._hand_paused = False

            # ---- blink ----
            blink_frame = None
            if self.cfg.get("blink.enabled", True):
                with self.metrics.stage("blink"):
                    blink_frame = blink_p.process(face)
                if blink_frame is not None:
                    if blink_frame.closed:
                        self.cursor.freeze(freeze_ms / 1000.0)
                    self.blink_fsm.update(blink_frame)
            self._last_blink = blink_frame

            # ---- gaze ----
            if self.cfg.get("gaze.enabled", True) and not self._calibrating:
                with self.metrics.stage("gaze"):
                    sample = gaze_p.process(face, pose, frame=frame)
                if sample is not None:
                    self._last_gaze = sample
                    # Nearly-closed eyes give garbage iris positions — hold the
                    # cursor rather than let it wander. Hysteretic gate:
                    # quality flickering around one threshold would otherwise
                    # freeze/jump the cursor in a stutter. Thresholds must stay
                    # LOW: looking at the bottom of the screen necessarily
                    # half-closes the lids (they follow the iris), so a strict
                    # gate freezes the cursor exactly when the user aims down.
                    if self._gaze_gate.update(sample.quality) and not self._hand_paused:
                        self.cursor.update(sample)

            # ---- metrics ----
            jitter = self.cursor.jitter_px()
            if jitter >= 0:
                self.metrics.gauge("jitter_px", jitter)
            model = getattr(gaze_p, "model", None)
            if model is not None and getattr(model, "ready", False):
                self.metrics.gauge("calib_accuracy_px", model.mean_error_px)
                self.metrics.gauge("calib_precision_px", model.precision_px)
            self.metrics.record_stage("total",
                                      (time.perf_counter() - t_total0) * 1000.0)
            self.metrics.frame_done()

            # ---- fatigue (blink rate/duration fed via bus; rest here) ----
            if self.cfg.get("fatigue.enabled", True):
                self.fatigue.update(pose, max(jitter, 0.0), face, t_cap)

            # ---- object detection (throttled AND off-thread: EfficientDet
            # takes 100-300 ms — inline it would stall the whole loop) ----
            if self.cfg.get("objects.enabled", True) and self.object_detector is not None:
                interval_s = float(self.cfg.get("objects.interval_s", 2.5))
                if time.perf_counter() - self._last_object_scan >= interval_s \
                        and not self._object_busy.is_set():
                    self._last_object_scan = time.perf_counter()
                    self._object_busy.set()
                    threading.Thread(target=self._scan_objects, args=(frame,),
                                     name="objects", daemon=True).start()

            # ---- preview (max ~12 fps, only if someone recently asked) ----
            if time.perf_counter() - self._preview_at < 3.0 and \
                    time.perf_counter() - self._preview_last >= 1 / 12:
                self._preview_last = time.perf_counter()
                self._render_preview(frame, face)

    def _scan_objects(self, frame: np.ndarray) -> None:
        try:
            with self.metrics.stage("objects"):
                self._last_objects = self.object_detector.detect(frame)
        except Exception:
            log.exception("object scan failed")
        finally:
            self._object_busy.clear()

    # ------------------------------------------------------------ feedback
    @staticmethod
    def _click_sound(kind: str) -> None:
        """Confirmation beeps the instant a blink click fires — the count
        tells the actions apart by ear: 1 = left click, 2 = double click,
        3 = right click (same pitch so they're easy to count)."""
        beep = (1000, 90)
        seq = {"click": [beep],
               "double_click": [beep, beep],
               "right_click": [beep, beep, beep]}.get(kind, [])
        audio_feedback.play(seq)

    # ------------------------------------------------------------- preview
    def request_preview(self) -> Optional[bytes]:
        self._preview_at = time.perf_counter()
        return self._preview_jpeg

    def _render_preview(self, frame: np.ndarray, face: FaceResult) -> None:
        img = cv2.flip(frame, 1)  # selfie view
        h, w = img.shape[:2]
        if face.ok:
            pts = face.landmarks
            from app.vision import landmarks as L
            mesh_xy = [(w - int(x), int(y)) for x, y, _ in pts]
            for (x, y) in mesh_xy:
                cv2.circle(img, (x, y), 1, (140, 255, 190), -1)
            for idx in L.RIGHT_EYE_EAR + L.LEFT_EYE_EAR:
                x, y = pts[idx][:2]
                cv2.circle(img, (w - int(x), int(y)), 1, (0, 255, 170), -1)
            for idx in (L.RIGHT_IRIS_CENTER, L.LEFT_IRIS_CENTER):
                x, y = pts[idx][:2]
                cv2.circle(img, (w - int(x), int(y)), 3, (60, 170, 255), -1)
        pose = self._last_pose
        if pose is not None and pose.valid:
            cv2.putText(img, f"yaw {pose.yaw:+5.1f}  pitch {pose.pitch:+5.1f}",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        blink = self._last_blink
        if blink is not None:
            color = (0, 80, 255) if blink.closed else (0, 220, 120)
            cv2.putText(img, f"{blink.measure} L {blink.left:.2f} R {blink.right:.2f}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
        gaze = self._last_gaze
        if gaze is not None:
            gx = int(min(max(gaze.nx, 0), 1) * (w - 1))
            gy = int(min(max(gaze.ny, 0), 1) * (h - 1))
            cv2.drawMarker(img, (gx, gy), (80, 120, 255),
                           cv2.MARKER_CROSS, 16, 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self._preview_jpeg = buf.tobytes()

    # ------------------------------------------------------------ calibration
    # grid density (# dots) -> NxN argument for calibration_points()
    _GRID = {9: None, 16: 4, 25: 5, 36: 6}

    def run_calibration(self, points: Optional[int] = None) -> dict:
        """Multi-point on-screen calibration (blocking; run from a worker).

        `points` (9/16/25/36) overrides the configured grid density for this
        run; None uses gaze.calibration.points."""
        if self._calibrating:
            return {"ok": False, "error": "already calibrating"}
        if not self.camera_ok:
            self._last_calib = "no camera — cannot calibrate"
            bus.publish("system.toast", {"text": self._last_calib})
            return {"ok": False, "error": "no camera"}
        self._calibrating = True
        was_paused = self.cursor.paused
        self.cursor.pause()
        try:
            result = self._calibrate_inner(points)
            # Surface every outcome (a daemon thread discards this return
            # value, so a silent failure just looks like "no calibration").
            if result.get("ok"):
                self._last_calib = f"calibrated ✓ ({result.get('accuracy_px')}px)"
            else:
                self._last_calib = f"calibration failed: {result.get('error', '?')}"
            return result
        except Exception as e:
            log.exception("calibration crashed")
            self._last_calib = f"calibration error: {e}"
            bus.publish("system.toast", {"text": self._last_calib})
            return {"ok": False, "error": str(e)}
        finally:
            self._calibrating = False
            self._calib_status = ""
            if not was_paused:
                self.cursor.resume()

    def _capture_dot(self, win, cw: int, ch: int, tx: float, ty: float,
                     smoother, settle_s: float, target_frames: int,
                     max_collect_s: float, label: str) -> tuple[list[dict], bool]:
        """Show a dot at (tx,ty), let the eye settle, then collect up to
        `target_frames` smoothed feature samples (Upgrade 2). Returns
        (samples, cancelled). Shared by calibration and the quality check."""
        from app.gaze.features import iris_features
        samples: list[dict] = []
        for phase in ("settle", "collect"):
            budget = settle_s if phase == "settle" else max_collect_s
            t_end = time.perf_counter() + budget
            while time.perf_counter() < t_end:
                if phase == "collect" and len(samples) >= target_frames:
                    break
                canvas = np.zeros((ch, cw, 3), np.uint8)
                cx, cy = int(tx * cw), int(ty * ch)
                pulse = 14 if phase == "settle" else 9
                cv2.circle(canvas, (cx, cy), pulse, (60, 170, 255), -1)
                cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), -1)
                if phase == "collect" and target_frames > 0:
                    frac = min(len(samples) / target_frames, 1.0)
                    cv2.ellipse(canvas, (cx, cy), (22, 22), -90, 0,
                                int(360 * frac), (90, 220, 255), 2)
                cv2.putText(canvas, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imshow(win, canvas)
                if cv2.waitKey(15) & 0xFF == 27:
                    return samples, True
                face, pose = self._last_face, self._last_pose
                if face is not None and face.ok:
                    feats = iris_features(face.landmarks, face.width, face.height)
                    if feats is not None and feats[2] >= 0.5:
                        # keep the smoother warm through settle; record on collect
                        hx, hy = smoother.apply(feats[0], feats[1], face.t)
                        if phase == "collect":
                            samples.append({
                                "hx": hx, "hy": hy,
                                "yaw": pose.yaw if pose and pose.valid else 0.0,
                                "pitch": pose.pitch if pose and pose.valid else 0.0,
                                "tx": tx, "ty": ty,
                            })
        return samples, False

    def _calibrate_inner(self, points: Optional[int]) -> dict:
        from app.gaze.features import FeatureSmoother
        from app.gaze.calibration import (MAX_SANE_ERROR_PX, MAX_SANE_PRECISION_PX,
                                          validation_points)
        g = self.cfg.get
        n_points = int(points if points is not None else g("gaze.calibration.points", 16))
        grid = self._GRID.get(n_points, 4)
        per_point = int(g("gaze.calibration.samples_per_point", 60))
        # Collect until we hit the target frame count, with a time cap so a
        # low-fps machine finishes instead of hanging (it just gathers fewer).
        max_collect = max(1.5, per_point / 15.0 + 1.0)

        sw, sh = win_input.screen_size()
        win = "HandsFree Calibration"
        cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        samples: list[dict] = []
        smoother = FeatureSmoother()
        try:
            pts = calibration_points(grid=grid)
            for i, (tx, ty) in enumerate(pts):
                self._calib_status = f"point {i + 1}/{len(pts)}"
                got, cancelled = self._capture_dot(
                    win, sw, sh, tx, ty, smoother, 0.8, per_point, max_collect,
                    f"Follow the dot  ({i + 1}/{len(pts)})  —  ESC to cancel")
                samples.extend(got)
                if cancelled:
                    return {"ok": False, "error": "cancelled"}
            if len(samples) < 40:
                return {"ok": False,
                        "error": f"only {len(samples)} samples — is your face visible?"}
            model = PolynomialModel.fit(samples)
            model.evaluate(samples, sw, sh)
            model.pose_provider = self.cfg.get("headpose.provider", "")
            if model.mean_error_px > MAX_SANE_ERROR_PX or \
                    model.precision_px > MAX_SANE_PRECISION_PX:
                bus.publish("system.toast", {
                    "text": f"calibration failed (accuracy "
                            f"{model.mean_error_px:.0f}px, precision "
                            f"{model.precision_px:.0f}px) — keep your head "
                            "still, follow the dot with your EYES, try again"})
                return {"ok": False,
                        "error": f"calibration not good enough (accuracy "
                                 f"{model.mean_error_px:.0f}px, precision "
                                 f"{model.precision_px:.0f}px) — try again"}

            # ---- Upgrade 10: quality check on random held-out targets ----
            if g("gaze.calibration.validate", True):
                self._calib_status = "quality check"
                verrs: list[float] = []
                vpts = validation_points(int(g("gaze.calibration.validation_points", 5)))
                vsmoother = FeatureSmoother()
                for j, (vx, vy) in enumerate(vpts):
                    vgot, cancelled = self._capture_dot(
                        win, sw, sh, vx, vy, vsmoother, 0.7, 25, 3.0,
                        f"Quality check  ({j + 1}/{len(vpts)})  —  look at the dot")
                    if cancelled:
                        break   # skip the rest of the check, keep the fit
                    for s in vgot:
                        px, py = model.predict(s["hx"], s["hy"], s["yaw"], s["pitch"])
                        verrs.append(float(np.hypot((px - vx) * sw, (py - vy) * sh)))
                if verrs:
                    model.validation_error_px = float(np.mean(verrs))
                    if model.validation_error_px > MAX_SANE_ERROR_PX:
                        bus.publish("system.toast", {
                            "text": f"quality check failed (held-out error "
                                    f"{model.validation_error_px:.0f}px) — "
                                    "recalibrate, following the dot with your eyes"})
                        return {"ok": False,
                                "error": f"quality check failed "
                                         f"({model.validation_error_px:.0f}px held-out "
                                         "error) — try again"}

            with self._lock:
                if not isinstance(self.gaze, IrisPolynomialGaze):
                    self.switch("gaze", "iris_polynomial")
                if isinstance(self.gaze, IrisPolynomialGaze):
                    self.gaze.set_model(model)
            self.cursor.stab.reset()
            self.metrics.log_event("calibration", {
                "points": len(pts), "samples": len(samples),
                "accuracy_px": round(model.mean_error_px, 1),
                "precision_px": round(model.precision_px, 1),
                "validation_px": round(model.validation_error_px, 1)})
            vtxt = (f", validated {model.validation_error_px:.0f}px"
                    if model.validation_error_px >= 0 else "")
            bus.publish("system.toast", {
                "text": f"calibrated ({len(pts)}-pt) — accuracy "
                        f"{model.mean_error_px:.0f}px, precision "
                        f"{model.precision_px:.0f}px{vtxt}"})
            return {"ok": True, "points": len(pts), "samples": len(samples),
                    "accuracy_px": round(model.mean_error_px, 1),
                    "precision_px": round(model.precision_px, 1),
                    "validation_px": round(model.validation_error_px, 1)}
        finally:
            cv2.destroyWindow(win)

    # ---------------------------------------------------------------- state
    def snapshot(self) -> dict:
        pose = self._last_pose
        blink = self._last_blink
        gaze = self._last_gaze
        t = now()
        model = getattr(self.gaze, "model", None)
        return {
            "camera_ok": self.camera_ok,
            "camera_healthy": self.camera.healthy,
            "camera_recoveries": self.camera.recoveries,
            "camera_fps": round(self.camera.fps, 1),
            "face": bool(self._last_face and self._last_face.ok),
            "providers": {
                "gaze": getattr(self.gaze, "name", "?"),
                "blink": getattr(self.blink, "name", "?"),
                "headpose": getattr(self.headpose, "name", "?"),
                "voice": getattr(self.voice, "name", "off") if
                         (self.voice and self.voice.running) else "off",
            },
            "available": {
                "gaze": list(self.GAZE), "blink": list(self.BLINK),
                "headpose": list(self.HEADPOSE), "voice": list(self.VOICE),
            },
            "enabled": {
                "gaze": self.cfg.get("gaze.enabled", True),
                "blink": self.cfg.get("blink.enabled", True),
                "headpose": self.cfg.get("headpose.enabled", True),
                "voice": self.cfg.get("voice.enabled", True),
            },
            "cursor": self.cursor.state(),
            "gaze_sample": {"nx": round(gaze.nx, 4), "ny": round(gaze.ny, 4),
                            "quality": round(gaze.quality, 2)} if gaze else None,
            "calibrated": bool(model and getattr(model, "ready", False)),
            "calibrating": self._calibrating,
            "calib_status": self._calib_status if self._calibrating else "",
            "calib_last": self._last_calib,
            "calib_accuracy_px": round(model.mean_error_px, 1)
                if model and getattr(model, "mean_error_px", -1) >= 0 else None,
            "calib_validation_px": round(model.validation_error_px, 1)
                if model and getattr(model, "validation_error_px", -1) >= 0 else None,
            "blink": {
                "left": round(blink.left, 3), "right": round(blink.right, 3),
                "thr_left": round(blink.threshold_left, 3),
                "thr_right": round(blink.threshold_right, 3),
                "closed": blink.closed, "measure": blink.measure,
                "progress": self.blink_fsm.progress(t),
                "counts": dict(self.blink_fsm.counts),
            } if blink else None,
            "pose": {"yaw": round(pose.yaw, 1), "pitch": round(pose.pitch, 1),
                     "roll": round(pose.roll, 1), "valid": pose.valid,
                     "counts": dict(self.gesture_fsm.counts)} if pose else None,
            "voice": {
                "last_text": self._last_voice_text,
                "last_reply": self._last_voice_reply,
                "dictation": self.executor.dictation,
            },
            "last_action": self._last_action,
            "hand_gesture": self.hand_fsm.current,
            "hand_paused": self._hand_paused,
            "fatigue_score": round(self.fatigue.last_score, 2),
            "objects": [{"label": d.label, "score": d.score, "bbox": d.bbox}
                       for d in self._last_objects],
            "metrics": self.metrics.snapshot(),
        }
