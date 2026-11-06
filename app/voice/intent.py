"""Natural-language intent parser for voice control.

Rule/grammar-based NLU (no model training): a prioritized list of regex
patterns with slot extraction. Deterministic, instant, fully offline, and
easy to extend — appropriate for a command-and-control assistant.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Intent:
    name: str
    args: dict


_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "once": 1, "twice": 2,
}


def _num(text: Optional[str], default: int = 1) -> int:
    if not text:
        return default
    text = text.strip()
    if text.isdigit():
        return int(text)
    return _NUM_WORDS.get(text, default)


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\.\-:/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    # strip polite fillers / wake-word remnants
    text = re.sub(r"^(hey |ok |okay |please |can you |could you |computer )+", "", text)
    return text.strip()


# (intent, compiled pattern) — first match wins, so specific rules come first.
_RULES: list[tuple[str, re.Pattern]] = [(n, re.compile(p)) for n, p in [
    # -- wake phrase: always recognized, bypasses the configurable wake_word gate --
    # ("hello won"/"hello 1" are common speech-recognition spellings of it)
    ("open_dashboard", r"^hello (one|won|wan|1)[.!?]?$"),

    # -- dictation & control --
    ("stop_dictation", r"^(stop|end|finish) (dictation|dictating|typing)$"),
    ("start_dictation", r"^(start |begin )?(dictation|dictate|dictating)$"),
    ("pause_control", r"^(pause|stop|disable|suspend) (eye |gaze |cursor |hands ?free )?(control|tracking|cursor)$"),
    ("resume_control", r"^(resume|start|enable|continue) (eye |gaze |cursor |hands ?free )?(control|tracking|cursor)$"),
    ("calibrate", r"^(re)?calibrat\w*( (my )?(gaze|eyes?|eye tracking))?$"),
    ("set_center", r"^(set |mark )?(center|centre)( gaze| cursor)?$|^i am looking at the (center|centre)$"),
    ("set_neutral", r"^(set|reset) (neutral|head)( pose| position)?$"),
    ("generate_report", r"^(generate|create|make|build) (a |the )?report$"),
    ("help", r"^(help|what can i say|show commands)$"),

    # -- clicking --
    ("double_click", r"^double( |-)?click( here| that)?$"),
    ("right_click", r"^right( |-)?click( here| that)?$"),
    ("middle_click", r"^middle( |-)?click$"),
    ("click", r"^(left )?click( here| that| it)?$|^select (this|that|it)$"),

    # -- scrolling & navigation --
    ("scroll", r"^scroll (?P<dir>up|down)( (?P<amt>a lot|a little|\w+))?$"),
    ("page", r"^page (?P<dir>up|down)$"),
    ("top", r"^(go to |scroll to )?(the )?top( of (the )?page)?$"),
    ("bottom", r"^(go to |scroll to )?(the )?bottom( of (the )?page)?$"),
    ("back", r"^go back(ward)?$|^back$|^previous page$"),
    ("forward", r"^go forward$|^forward$|^next page$"),
    ("refresh", r"^(refresh|reload)( the)?( page)?$"),
    ("zoom_in", r"^zoom in$"),
    ("zoom_out", r"^zoom out$"),

    # -- tabs & windows & apps --
    ("new_tab", r"^(open |create )?(a )?new tab$"),
    ("close_tab", r"^close (this |the |current )?tab$"),
    ("reopen_tab", r"^reopen( closed)? tab$"),
    ("tab_next", r"^(next|switch) tab$|^tab (right|next)$"),
    ("tab_prev", r"^(previous|last) tab$|^tab (left|back|previous)$"),
    ("app_next", r"^(next|switch) (app|application|window)s?$|^switch$"),
    ("app_prev", r"^(previous|last) (app|application|window)$"),
    ("minimize", r"^minimi[sz]e( this| the)?( window| app)?$"),
    ("maximize", r"^maximi[sz]e( this| the)?( window| app)?$"),
    ("show_desktop", r"^(show|go to)( the)? desktop$|^minimi[sz]e (everything|all)$"),
    ("close_window", r"^close (this |the |current )?(window|app|application|program)$"),
    ("snap_left", r"^snap (window )?left$"),
    ("snap_right", r"^snap (window )?right$"),
    ("switch_to", r"^(switch|go) to (?P<name>.+)$"),

    # -- launching & web --
    ("search_web", r"^(search|google|look up)( the web| google)?( for)? (?P<q>.+)$"),
    ("open_url", r"^(open|go to)( the)? (website|site|url) (?P<url>.+)$"),
    ("open_app", r"^(open|launch|start|run) (?P<name>.+)$"),

    # -- typing & keys --
    ("type_text", r"^(type|write|enter text) (?P<text>.+)$"),
    ("press_key", r"^(press|hit|push)( the)? (?P<key>.+?)( key)?$"),
    ("select_all", r"^select (all|everything)$"),
    ("copy", r"^copy( that| this| it)?$"),
    ("paste", r"^paste( that| this| it)?$"),
    ("cut", r"^cut( that| this| it)?$"),
    ("undo", r"^undo( that)?$"),
    ("redo", r"^redo( that)?$"),
    ("save", r"^save( this| the)?( file| document)?$"),
    ("delete", r"^delete( that| this)?$"),

    # -- media & volume --
    ("volume_up", r"^(turn )?(the )?volume up$|^(increase|raise) (the )?volume$|^louder$"),
    ("volume_down", r"^(turn )?(the )?volume down$|^(decrease|lower) (the )?volume$|^quieter$"),
    ("mute", r"^(mute|unmute)( the)?( sound| volume| audio)?$"),
    ("play_pause", r"^(play|pause|resume)( the)?( music| video| media| song)?$"),
    ("media_next", r"^next (song|track|video)$|^skip( this)?( song| track)?$"),
    ("media_prev", r"^(previous|last) (song|track)$"),

    # -- system --
    ("screenshot", r"^(take|capture)( a)? (screenshot|screen shot)$"),
    ("lock_screen", r"^lock( the)?( screen| computer| pc)?$"),
    ("press_enter", r"^(enter|confirm|submit|okay|ok)$"),
    ("press_escape", r"^(escape|cancel|dismiss|never ?mind)$"),
]]


# Canonical argless phrases for fuzzy recovery: when speech recognition is
# slightly off ("scroll town", "knew tab"), the closest phrase above the
# cutoff still triggers. Commands that carry free-text slots (open X,
# search for X, type X) are deliberately excluded — guessing those is worse
# than asking the user to repeat.
_FUZZY: dict[str, tuple[str, dict]] = {
    "hello one": ("open_dashboard", {}),
    "click": ("click", {}),
    "double click": ("double_click", {}),
    "right click": ("right_click", {}),
    "scroll down": ("scroll", {"dir": "down", "clicks": 3}),
    "scroll up": ("scroll", {"dir": "up", "clicks": 3}),
    "go back": ("back", {}),
    "go forward": ("forward", {}),
    "refresh": ("refresh", {}),
    "new tab": ("new_tab", {}),
    "close tab": ("close_tab", {}),
    "next tab": ("tab_next", {}),
    "previous tab": ("tab_prev", {}),
    "next app": ("app_next", {}),
    "pause control": ("pause_control", {}),
    "resume control": ("resume_control", {}),
    "start dictation": ("start_dictation", {}),
    "stop dictation": ("stop_dictation", {}),
    "calibrate": ("calibrate", {}),
    "set center": ("set_center", {}),
    "volume up": ("volume_up", {}),
    "volume down": ("volume_down", {}),
    "select all": ("select_all", {}),
    "copy": ("copy", {}),
    "paste": ("paste", {}),
    "undo": ("undo", {}),
    "enter": ("press_enter", {}),
    "escape": ("press_escape", {}),
    "help": ("help", {}),
}


def parse(text: str) -> Optional[Intent]:
    text = normalize(text)
    if not text:
        return None
    for name, pattern in _RULES:
        m = pattern.match(text)
        if m:
            args = {k: v for k, v in m.groupdict().items() if v is not None}
            if name == "scroll":
                amt = args.get("amt", "")
                clicks = {"a lot": 10, "a little": 1}.get(amt, _num(amt, 3))
                args = {"dir": args["dir"], "clicks": clicks}
            return Intent(name=name, args=args)
    close = difflib.get_close_matches(text, _FUZZY.keys(), n=1, cutoff=0.75)
    if close:
        name, args = _FUZZY[close[0]]
        return Intent(name=name, args=dict(args))
    return None
