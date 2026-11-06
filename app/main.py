"""HandsFree Control — entry point.

Usage:
    python -m app.main                 # start everything + open dashboard
    python -m app.main --selftest      # verify environment, camera, model
    python -m app.main --no-browser    # don't auto-open the dashboard
    python -m app.main --report        # generate an evaluation report and exit
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import webbrowser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handsfree")


def selftest() -> int:
    print("== HandsFree self-test ==")
    ok = True
    for mod in ("numpy", "cv2", "mediapipe", "fastapi", "uvicorn", "vosk",
                "sounddevice", "psutil", "matplotlib"):
        try:
            __import__(mod)
            print(f"  [ok] import {mod}")
        except Exception as e:
            print(f"  [FAIL] import {mod}: {e}")
            ok = False
    from app.config import Config
    cfg = Config()
    from pathlib import Path
    model = Path(cfg.get("voice.vosk_model", ""))
    print(f"  [{'ok' if model.exists() else 'FAIL'}] vosk model at {model}")
    ok = ok and model.exists()

    from app.vision.camera import Camera
    cam = Camera(cfg.get("camera.index", 0))
    if cam.start():
        time.sleep(1.5)
        frame, _, seq = cam.latest()
        if frame is not None:
            print(f"  [ok] camera: {frame.shape[1]}x{frame.shape[0]} "
                  f"@ ~{cam.fps:.0f} fps ({seq} frames)")
            from app.vision.face_mesh import FaceMeshService
            mesh = FaceMeshService()
            face = mesh.process(frame)
            print(f"  [ok] face mesh runs (face detected: {face.ok})")
            mesh.close()
        else:
            print("  [WARN] camera opened but no frames yet")
        cam.stop()
    else:
        print("  [FAIL] camera could not be opened")
        ok = False
    print("== self-test", "PASSED" if ok else "FAILED", "==")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="HandsFree Control")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--camera", type=int, default=None)
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if args.report:
        from app.metrics import report
        report.generate()
        return 0

    # cv2 defaults to one worker per logical core; its jobs here (cvtColor,
    # resize, JPEG encode on 640x480) are too small to parallelize and the
    # extra threads just fight mediapipe/torch for cores.
    import cv2
    cv2.setNumThreads(2)

    from app.config import Config
    cfg = Config()
    if args.port:
        cfg.set("server.port", args.port, save=False)
    if args.camera is not None:
        cfg.set("camera.index", args.camera, save=False)

    from app.metrics.collector import MetricsCollector
    metrics = MetricsCollector(log_enabled=cfg.get("metrics.log_sessions", True))

    from app.pipeline import Pipeline
    pipeline = Pipeline(cfg, metrics)
    pipeline.start()

    from app.server.web import run_server
    host = cfg.get("server.host", "127.0.0.1")
    port = cfg.get("server.port", 8765)
    run_server(pipeline, cfg, metrics, host, port)
    url = f"http://{host}:{port}"
    log.info("dashboard: %s", url)
    if not args.no_browser and cfg.get("server.open_browser", True):
        webbrowser.open(url)

    try:
        while True:
            if pipeline.shutdown_requested:   # peace-sign hand gesture
                log.info("shutdown requested by hand gesture…")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("shutting down…")
    finally:
        pipeline.stop()
        metrics.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
