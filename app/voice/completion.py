"""Phrase auto-completion for dictation.

ponytail: this is prefix matching against a curated phrase table, not a
trained language model — there's no LLM already in this project's
dependencies, and pulling one in (llama.cpp + a multi-GB GGUF file, or a
hosted API) is a much bigger install than "suggest a few common endings" needs.
Good enough for the requested "can you ple" -> "please send/call/help" case
and similar everyday phrases. Upgrade path: swap `suggest()` for a call to a
local GGUF model via llama-cpp-python, or a hosted completion API, if
free-form generative completion is actually needed.
"""
from __future__ import annotations

# Common phrase continuations, longest-useful-match first isn't required —
# suggest() ranks by shortest total phrase (closest, least presumptuous
# completion) among those that start with the user's partial text.
PHRASES = [
    "please send", "please call", "please help", "please wait", "please stop",
    "please open", "please close", "please repeat", "please confirm",
    "can you help me with", "can you please send", "can you please call",
    "can you please open", "can you send this to", "can you please repeat that",
    "thank you so much", "thank you for your time", "thanks a lot",
    "i would like to", "i need to", "i want to schedule a", "i am running late",
    "i am not sure", "i will get back to you", "i will send it over",
    "let me know if", "let me know when you", "let me check and get back",
    "see you soon", "see you tomorrow", "talk to you later",
    "sorry for the delay", "sorry for the inconvenience", "sorry, i missed that",
    "could you please", "could you send me", "could you let me know",
    "looking forward to", "looking forward to hearing from you",
    "just following up on", "just wanted to check in",
    "how are you doing", "how is everything going",
    "what time works for you", "what time is the meeting",
    "give me a moment", "give me a second", "give me a call",
    "call me back", "call me when you", "text me when you",
    "meeting is at", "meeting has been rescheduled",
    "attached is the", "attached please find",
    "as discussed", "as per our conversation", "as mentioned earlier",
]


def suggest(partial: str, limit: int = 3) -> list[str]:
    """Return up to `limit` phrase continuations whose text starts with the
    lowercased, trimmed `partial`. Empty/very short input yields no noise."""
    text = " ".join(partial.lower().split())
    if len(text) < 3:
        return []
    matches = [p for p in PHRASES if p.startswith(text) and p != text]
    matches.sort(key=len)
    return matches[:limit]
