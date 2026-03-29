import sys
import traceback
from pathlib import Path

# Mock profile
from writer_profile import get_persona
from phase10_legacy_synthesis import compose_legacy_page

try:
    title = "TEST"
    body = ["The quick brown fox jumps over the lazy dog."]
    out = Path("tmp_debug.jpg")
    profile = get_persona("Architect")
    compose_legacy_page(title, body, out, profile=profile)
    print("SUCCESS")
except Exception:
    traceback.print_exc()
