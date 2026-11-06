"""Local web server: dashboard UI, WebSocket state stream, REST control API.

Bound to 127.0.0.1 — the dashboard is a local control panel, not exposed to
the network. The same UI is what a public deployment would ship (the Web
Speech engine already runs fully in-browser).
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.events import bus

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


def create_app(pipeline, cfg, metrics) -> FastAPI:
    app = FastAPI(title="HandsFree Control", docs_url=None, redoc_url=None)
    clients: set[WebSocket] = set()
    loop_holder: dict[str, Optional[asyncio.AbstractEventLoop]] = {"loop": None}

    # ---- push bus events to all websocket clients -------------------------
    def _relay(topic: str, payload) -> None:
        loop = loop_holder["loop"]
        if loop is None or not clients:
            return
        try:
            data = payload.__dict__ if hasattr(payload, "__dict__") else payload
            msg = json.dumps({"type": "event", "topic": topic, "data": data},
                             default=str)
        except Exception:
            return
        for ws in list(clients):
            asyncio.run_coroutine_threadsafe(_safe_send(ws, msg), loop)

    async def _safe_send(ws: WebSocket, msg: str) -> None:
        try:
            await ws.send_text(msg)
        except Exception:
            clients.discard(ws)

    for topic in ("blink.event", "head.gesture", "voice.command", "voice.partial",
                  "voice.suggestions", "action.executed", "system.toast",
                  "control.paused", "control.resumed", "gesture.confirm",
                  "gesture.cancel"):
        bus.subscribe(topic, _relay)

    # ---- websocket ------------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        loop_holder["loop"] = asyncio.get_running_loop()
        clients.add(ws)
        try:
            sender = asyncio.create_task(_state_sender(ws))
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await _handle_client_msg(msg)
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)
            sender.cancel()

    async def _state_sender(ws: WebSocket) -> None:
        while True:
            try:
                snap = pipeline.snapshot()
                await ws.send_text(json.dumps({"type": "state", "data": snap},
                                              default=str))
            except Exception:
                return
            await asyncio.sleep(0.12)

    async def _handle_client_msg(msg: dict) -> None:
        kind = msg.get("type")
        if kind == "switch":
            pipeline.switch(msg.get("kind", ""), msg.get("name", ""))
        elif kind == "set":
            path, value = msg.get("path", ""), msg.get("value")
            if path:
                cfg.set(path, value)
        elif kind == "toggle":
            modality, on = msg.get("modality", ""), bool(msg.get("on", True))
            if modality in ("gaze", "blink", "headpose", "voice"):
                cfg.set(f"{modality}.enabled", on)
                if modality == "voice":
                    if on:
                        pipeline.start_voice(cfg.get("voice.engine", "vosk"))
                    elif pipeline.voice:
                        pipeline.voice.stop()
        elif kind == "voice_transcript":
            # Web Speech API transcript from the browser
            engine = pipeline.voice
            text = msg.get("text", "")
            if engine is not None and getattr(engine, "name", "") == "webspeech":
                engine.inject(text)
        elif kind == "action":
            name = msg.get("name", "")
            if name == "calibrate":
                points = msg.get("points")   # 9/16/25/36, or None for config default
                threading.Thread(target=pipeline.run_calibration,
                                 kwargs={"points": points}, daemon=True).start()
            elif name == "set_center":
                pipeline.set_center()
            elif name == "set_neutral":
                pipeline.set_neutral()
            elif name == "pause":
                pipeline.cursor.pause()
            elif name == "resume":
                pipeline.cursor.resume()
            elif name == "report":
                from app.metrics import report
                threading.Thread(target=report.generate, daemon=True).start()
            elif name == "new_session":
                metrics.new_session()

    # ---- REST -----------------------------------------------------------
    @app.get("/health")
    def health():
        return {"ok": True, "started": pipeline.started,
                "camera": pipeline.camera_ok}

    @app.get("/api/config")
    def get_config():
        return cfg.data()

    @app.post("/api/config")
    async def set_config(data: dict):
        cfg.replace(data)
        return {"ok": True}

    @app.get("/api/state")
    def get_state():
        return pipeline.snapshot()

    # ---- MJPEG preview -------------------------------------------------
    @app.get("/preview.mjpg")
    async def preview():
        boundary = "handsfreeframe"

        async def gen():
            while True:
                jpeg = pipeline.request_preview()
                if jpeg:
                    yield (f"--{boundary}\r\n"
                           f"Content-Type: image/jpeg\r\n"
                           f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + \
                        jpeg + b"\r\n"
                await asyncio.sleep(0.08)

        return StreamingResponse(
            gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

    # ---- UI --------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    return app


def run_server(pipeline, cfg, metrics, host: str, port: int) -> threading.Thread:
    import uvicorn
    app = create_app(pipeline, cfg, metrics)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="webserver", daemon=True)
    thread.start()
    return thread
