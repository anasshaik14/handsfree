"""Application launching and window management for Windows.

Launch resolution order:
  1. known alias table (chrome, notepad, spotify, ...)
  2. URL protocols (ms-settings:, https://...)
  3. executable on PATH
  4. fuzzy search of Start Menu shortcuts (.lnk) — finds anything installed
"""
from __future__ import annotations

import difflib
import logging
import os
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

ALIASES: dict[str, str] = {
    "chrome": "chrome", "google chrome": "chrome", "browser": "chrome",
    "edge": "msedge", "microsoft edge": "msedge",
    "firefox": "firefox",
    "notepad": "notepad", "note pad": "notepad",
    "calculator": "calc", "calc": "calc",
    "paint": "mspaint",
    "word": "winword", "microsoft word": "winword",
    "excel": "excel", "microsoft excel": "excel",
    "powerpoint": "powerpnt", "power point": "powerpnt",
    "outlook": "outlook",
    "file explorer": "explorer", "explorer": "explorer", "files": "explorer",
    "settings": "ms-settings:",
    "task manager": "taskmgr",
    "command prompt": "cmd", "cmd": "cmd",
    "terminal": "wt", "windows terminal": "wt", "powershell": "powershell",
    "vs code": "code", "visual studio code": "code", "code": "code",
    "spotify": "spotify",
    "whatsapp": "whatsapp",
    "camera": "microsoft.windows.camera:",
    "store": "ms-windows-store:", "microsoft store": "ms-windows-store:",
    "control panel": "control",
    "snipping tool": "snippingtool",
}

WEBSITES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "stack overflow": "https://stackoverflow.com",
    "wikipedia": "https://www.wikipedia.org",
    "netflix": "https://www.netflix.com",
    "twitter": "https://twitter.com", "x": "https://x.com",
    "reddit": "https://www.reddit.com",
    "linkedin": "https://www.linkedin.com",
    "chatgpt": "https://chat.openai.com",
    "claude": "https://claude.ai",
}


def _start_menu_dirs() -> list[Path]:
    dirs = []
    for env, sub in [("PROGRAMDATA", r"Microsoft\Windows\Start Menu\Programs"),
                     ("APPDATA", r"Microsoft\Windows\Start Menu\Programs")]:
        base = os.environ.get(env)
        if base:
            p = Path(base) / sub
            if p.exists():
                dirs.append(p)
    return dirs


def _find_start_menu_shortcut(name: str) -> Optional[Path]:
    """Fuzzy-match a Start Menu .lnk by its display name."""
    candidates: dict[str, Path] = {}
    for d in _start_menu_dirs():
        for lnk in d.rglob("*.lnk"):
            candidates.setdefault(lnk.stem.lower(), lnk)
    if not candidates:
        return None
    # exact / substring first
    if name in candidates:
        return candidates[name]
    subs = [k for k in candidates if name in k]
    if subs:
        return candidates[min(subs, key=len)]
    close = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.75)
    return candidates[close[0]] if close else None


def _shell_start(target: str) -> bool:
    try:
        subprocess.Popen(f'start "" "{target}"', shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        log.exception("shell start failed for %s", target)
        return False


def launch(name: str) -> tuple[bool, str]:
    """Open an app or site by natural name. Returns (ok, what_was_opened)."""
    name = name.strip().lower()
    if name in WEBSITES:
        webbrowser.open(WEBSITES[name])
        return True, WEBSITES[name]
    if name in ALIASES:
        target = ALIASES[name]
        return _shell_start(target), target
    if name.startswith(("http://", "https://", "www.")) or \
            ("." in name and " " not in name):
        url = name if name.startswith("http") else f"https://{name}"
        webbrowser.open(url)
        return True, url
    lnk = _find_start_menu_shortcut(name)
    if lnk is not None:
        try:
            os.startfile(str(lnk))
            return True, lnk.stem
        except Exception:
            log.exception("startfile failed for %s", lnk)
    # last resort: let the shell resolve it
    return _shell_start(name), name


def open_url(url: str) -> bool:
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url.replace(" ", "")
    webbrowser.open(url)
    return True


def search_web(query: str) -> bool:
    from urllib.parse import quote_plus
    webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}")
    return True
