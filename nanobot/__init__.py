"""
nanobot - A lightweight AI agent framework
"""

import sys

__version__ = "0.1.0"

# Use a safe fallback logo on Windows when the console can't handle emoji
def _safe_logo() -> str:
    try:
        "🐈".encode(sys.stdout.encoding or "utf-8")
        return "🐈"
    except (UnicodeEncodeError, LookupError):
        return "[nanobot]"

__logo__ = _safe_logo()
