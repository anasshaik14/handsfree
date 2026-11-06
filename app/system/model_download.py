"""Shared on-demand model download (same pattern as app.vision.face_mesh)."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def ensure_model(path: Path, url: str) -> Path:
    if path.exists():
        return path
    log.info("downloading model to %s …", path)
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed https URL
    return path
