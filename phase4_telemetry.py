"""
Phase 4: Telemetry Simulation for Multimodal Forensics
Objective: Simulate human keystroke dynamics while transcribing the augmented
text, producing a paired telemetry log for digital forensics research.

Keystroke features captured:
  - Inter-key interval (IKI) per character
  - Words-per-minute (WPM) sampled from U[30, 120]
  - 10% backspace/correction event probability
  - Pause events (word boundaries, punctuation, sentence ends)
  - Dwell time simulation for held keys (shift, ctrl)

Output:
  - Telemetry JSON log  (per-keystroke metadata)
  - Typed document file (final transcription result)

Note: PyAutoGUI requires a graphical display. Set HEADLESS=True to run in
      simulation-only mode (no actual keyboard events fired).
"""

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── PyAutoGUI import guard ────────────────────────────────────────────────────
try:
    import pyautogui

    pyautogui.FAILSAFE = True  # move mouse to corner to abort
    pyautogui.PAUSE = 0.0  # we handle timing ourselves
    PYAUTOGUI_AVAILABLE = True
except (ImportError, Exception):
    PYAUTOGUI_AVAILABLE = False
    print("[WARN] pyautogui not available — running in simulation-only mode.")

# ── Headless override (set True to disable actual key presses) ────────────────
HEADLESS = not PYAUTOGUI_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WPM_MIN = 30
WPM_MAX = 120
CORRECTION_PROB = 0.10  # 10% chance of backspace + retype
WORD_PAUSE_FACTOR = 1.8  # IKI multiplier at word boundary
SENTENCE_PAUSE_SEC = 0.35  # extra pause after sentence-ending punctuation
COMMA_PAUSE_SEC = 0.12  # extra pause after comma
SHIFT_DWELL_MS = 45  # milliseconds shift held for capital letter
NEWLINE_PAUSE_SEC = 0.30  # pause before pressing Enter

# Fatigue model: IKI increases slightly over time
FATIGUE_ONSET_CHARS = 300  # chars before fatigue starts
FATIGUE_RATE = 0.0004  # seconds added per char after onset


# Psycholinguistic Digraphs (th, he, in, er, an) — Ref: Research 2025
COMMON_DIGRAPHS = {"th", "he", "in", "er", "an", "re", "on", "at", "en"}
DIGRAPH_SPEEDUP = 0.25  # 25% IKI reduction for common digraphs

# Behavourial models
THINK_PAUSE_PROB = 0.02  # 2% chance before words > 8 chars
SAVING_CYCLE_CHARS = 150  # chars between periodic 'ctrl+s' simulations
DWELL_MS_MU = 85  # mean dwell time (ms)
DWELL_MS_SIGMA = 12  # jitter (ms)

# ── DATA STRUCTURES ─────────────────────────────────────────────────────────────


@dataclass
class KeyEvent:
    """Single keystroke event for forensic telemetry."""

    seq_id: int
    char: str
    key_code: str
    event_type: str  # 'press' | 'backspace' | 'correction_retype' | 'hotkey'
    timestamp_ms: float  # wall-clock ms since session start
    iki_ms: float  # inter-key interval from previous event (ms)
    dwell_ms: float  # duration key was held down (ms)
    wpm_at_event: float
    is_correction: bool
    word_position: int  # character index within current word
    line_number: int
    word_number: int


@dataclass
class TelemetrySession:
    """Full session metadata wrapper."""

    session_id: str
    start_utc: str
    target_text: str
    wpm_min: int = WPM_MIN
    wpm_max: int = WPM_MAX
    correction_prob: float = CORRECTION_PROB
    events: list[KeyEvent] = field(default_factory=list)
    total_chars_typed: int = 0
    total_corrections: int = 0
    total_backspaces: int = 0
    session_duration_sec: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# WPM → IKI CONVERSION
# ─────────────────────────────────────────────────────────────────────────────


def wpm_to_iki_sec(wpm: float, chars_per_word: float = 5.0) -> float:
    """Convert WPM to mean inter-key interval in seconds."""
    cps = (wpm * chars_per_word) / 60.0
    return 1.0 / max(cps, 0.1)


def sample_wpm(rng: random.Random) -> float:
    """Sample current WPM from a log-normal centred on the uniform [30, 120] range."""
    mu = (WPM_MAX + WPM_MIN) / 2.0
    sigma = (WPM_MAX - WPM_MIN) / 6.0
    return max(WPM_MIN, min(WPM_MAX, rng.gauss(mu, sigma)))


# ─────────────────────────────────────────────────────────────────────────────
# INTER-KEY INTERVAL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────


class IKIGenerator:
    """
    Generates realistic, variable inter-key intervals by combining:
      - WPM-derived mean IKI
      - Gaussian jitter (σ = 15% of mean)
      - Word/punctuation boundary pauses
      - Fatigue drift
      - Burst-typing episodes (transient WPM spikes)
      - Digraph speedup (Ref: Psycholinguistic 2025)
      - Think-pauses (Ref: Cognitive 2024)
    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self._char_count = 0
        self._burst_mode = False
        self._burst_rem = 0

    def next_iki(
        self,
        char: str,
        last_char: str,
        at_word_boundary: bool,
        at_sentence_end: bool,
        at_comma: bool,
    ) -> float:
        """Return IKI (seconds) for the next character."""
        self._char_count += 1

        # ── Digraph Speedup ───────────────────────────────────────────────
        digraph = (last_char + char).lower()
        is_digraph = digraph in COMMON_DIGRAPHS

        # ── Burst typing ──────────────────────────────────────────────────
        if self._burst_mode:
            wpm = min(WPM_MAX, sample_wpm(self.rng) * 1.3)
            self._burst_rem -= 1
            if self._burst_rem <= 0:
                self._burst_mode = False
        else:
            wpm = sample_wpm(self.rng)
            if self.rng.random() < 0.04:
                self._burst_mode = True
                self._burst_rem = self.rng.randint(5, 15)

        mean_iki = wpm_to_iki_sec(wpm)

        # ── Gaussian jitter ───────────────────────────────────────────────
        jitter = self.rng.gauss(0, mean_iki * 0.15)
        iki = max(0.015, mean_iki + jitter)

        if is_digraph:
            iki *= 1.0 - DIGRAPH_SPEEDUP

        # ── Boundary pauses ───────────────────────────────────────────────
        if at_sentence_end:
            iki += SENTENCE_PAUSE_SEC + self.rng.gauss(0, 0.06)
        elif at_comma:
            iki += COMMA_PAUSE_SEC
        elif at_word_boundary:
            iki *= WORD_PAUSE_FACTOR
            # Think pause
            if self.rng.random() < THINK_PAUSE_PROB:
                iki += self.rng.uniform(0.4, 1.2)

        # ── Fatigue drift ─────────────────────────────────────────────────
        if self._char_count > FATIGUE_ONSET_CHARS:
            iki += FATIGUE_RATE * (self._char_count - FATIGUE_ONSET_CHARS)

        return iki

    def sample_dwell(self) -> float:
        """Sample key dwell time (ms)."""
        return self.rng.gauss(DWELL_MS_MU, DWELL_MS_SIGMA)


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION MODEL
# ─────────────────────────────────────────────────────────────────────────────


def should_make_correction(rng: random.Random, prob: float = CORRECTION_PROB) -> bool:
    return rng.random() < prob


def correction_delay(rng: random.Random) -> float:
    """
    Realisation delay before user notices the error and hits backspace.
    Typically 0.3–1.5 s after the erroneous keystroke.
    """
    return rng.uniform(0.30, 1.50)


# ─────────────────────────────────────────────────────────────────────────────
# KEYBOARD INTERFACE  (real or simulated)
# ─────────────────────────────────────────────────────────────────────────────


def press_key(char: str, headless: bool = HEADLESS) -> None:
    """Press a single character key (or special key name)."""
    if headless or not PYAUTOGUI_AVAILABLE:
        return
    if char == "\n":
        pyautogui.press("enter")
    elif char == "\t":
        pyautogui.press("tab")
    elif char.isupper():
        with pyautogui.hold("shift"):
            pyautogui.press(char.lower())
    else:
        pyautogui.press(char)


def save_hotkey(headless: bool = HEADLESS) -> None:
    """Simulate Ctrl+S periodic save."""
    if headless or not PYAUTOGUI_AVAILABLE:
        return
    pyautogui.hotkey("ctrl", "s")


def press_backspace(n: int = 1, headless: bool = HEADLESS) -> None:
    if headless or not PYAUTOGUI_AVAILABLE:
        return
    for _ in range(n):
        pyautogui.press("backspace")
        time.sleep(0.04)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────


def simulate_transcription(
    text: str,
    session_id: str = "ses_001",
    focus_window_title: Optional[str] = None,
    output_dir: Path = Path("output/phase4"),
    headless: bool = HEADLESS,
) -> TelemetrySession:
    """
    Simulate human transcription of `text` with realistic keystroke dynamics.

    Parameters
    ----------
    text               : The augmented text to transcribe
    session_id         : Unique identifier for this forensic session
    focus_window_title : Window title to focus before typing (optional)
    output_dir         : Directory to write logs
    headless           : If True, no actual keyboard events are fired

    Returns
    -------
    TelemetrySession with all keystroke events logged
    """
    from datetime import datetime, timezone

    rng = random.Random(99)
    iki_gen = IKIGenerator(rng)
    session = TelemetrySession(
        session_id=session_id,
        start_utc=datetime.now(timezone.utc).isoformat(),
        target_text=text[:200] + "…" if len(text) > 200 else text,
    )

    # ── Pre-flight: focus target window ──────────────────────────────────
    if not headless and PYAUTOGUI_AVAILABLE and focus_window_title:
        print(f"[Phase 4] Focusing window: '{focus_window_title}' ...")
        time.sleep(3.0)  # give user time to switch to target window

    wall_start = time.monotonic()
    timestamp_ms = 0.0
    seq_id = 0
    word_pos = 0
    line_num = 1
    word_num = 0
    words_typed = []  # accumulates correctly typed output

    chars = list(text)
    i = 0
    last_ch = ""

    while i < len(chars):
        ch = chars[i]

        # ── Periodic Save simulation ──────────────────────────────────────
        if session.total_chars_typed > 0 and session.total_chars_typed % SAVING_CYCLE_CHARS == 0:
            save_hotkey(headless)
            session.events.append(
                KeyEvent(
                    seq_id=seq_id,
                    char="CTRL+S",
                    key_code="s",
                    event_type="hotkey",
                    timestamp_ms=timestamp_ms,
                    iki_ms=250.0,
                    dwell_ms=120.0,
                    wpm_at_event=0.0,
                    is_correction=False,
                    word_position=word_pos,
                    line_number=line_num,
                    word_number=word_num,
                )
            )
            seq_id += 1
            timestamp_ms += 300.0

        # ── Detect boundary context ───────────────────────────────────────
        at_sentence_end = i > 0 and chars[i - 1] in ".!?"
        at_comma = i > 0 and chars[i - 1] == ","
        at_word_boundary = ch == " " or ch == "\n"

        if at_word_boundary:
            word_pos = 0
            word_num += 1
        if ch == "\n":
            line_num += 1

        # ── Compute IKI + sleep ───────────────────────────────────────────
        iki = iki_gen.next_iki(ch, last_ch, at_word_boundary, at_sentence_end, at_comma)
        iki_ms = iki * 1000.0
        timestamp_ms += iki_ms

        if not headless:
            time.sleep(iki)

        dwell_ms = iki_gen.sample_dwell()

        # ── Correction event (10% probability) ───────────────────────────
        if should_make_correction(rng) and ch.isalpha():
            # Type a wrong character first
            wrong_char = rng.choice("abcdefghijklmnopqrstuvwxyz")
            wrong_iki = iki * rng.uniform(0.8, 1.2)

            # Log wrong keypress
            session.events.append(
                KeyEvent(
                    seq_id=seq_id,
                    char=wrong_char,
                    key_code=wrong_char,
                    event_type="press",
                    timestamp_ms=timestamp_ms,
                    iki_ms=wrong_iki * 1000,
                    dwell_ms=dwell_ms,
                    wpm_at_event=sample_wpm(rng),
                    is_correction=False,
                    word_position=word_pos,
                    line_number=line_num,
                    word_number=word_num,
                )
            )
            press_key(wrong_char, headless)
            session.total_chars_typed += 1
            seq_id += 1

            # Realisation delay
            delay = correction_delay(rng)
            timestamp_ms += delay * 1000
            if not headless:
                time.sleep(delay)

            # Backspace event
            session.events.append(
                KeyEvent(
                    seq_id=seq_id,
                    char="<BS>",
                    key_code="backspace",
                    event_type="backspace",
                    timestamp_ms=timestamp_ms,
                    iki_ms=delay * 1000,
                    dwell_ms=50.0,
                    wpm_at_event=sample_wpm(rng),
                    is_correction=True,
                    word_position=word_pos,
                    line_number=line_num,
                    word_number=word_num,
                )
            )
            press_backspace(1, headless)
            session.total_backspaces += 1
            session.total_corrections += 1
            seq_id += 1
            timestamp_ms += 60.0  # small post-backspace pause

        # ── Correct keystroke ─────────────────────────────────────────────
        event_type = (
            "correction_retype"
            if (session.events and session.events[-1].event_type == "backspace")
            else "press"
        )

        session.events.append(
            KeyEvent(
                seq_id=seq_id,
                char=ch if ch not in ("\n", "\t") else repr(ch),
                key_code="enter" if ch == "\n" else ("tab" if ch == "\t" else ch),
                event_type=event_type,
                timestamp_ms=timestamp_ms,
                iki_ms=iki_ms,
                dwell_ms=dwell_ms,
                wpm_at_event=round(sample_wpm(rng), 1),
                is_correction=(event_type == "correction_retype"),
                word_position=word_pos,
                line_number=line_num,
                word_number=word_num,
            )
        )
        press_key(ch, headless)
        session.total_chars_typed += 1
        seq_id += 1
        word_pos += 1
        words_typed.append(ch)
        last_ch = ch
        i += 1

    # ── Session summary ───────────────────────────────────────────────────
    session.session_duration_sec = time.monotonic() - wall_start

    # ── Persist telemetry log (JSON & CSV) ───────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"{session_id}_telemetry.json"
    csv_path = output_dir / f"{session_id}_telemetry.csv"
    doc_path = output_dir / f"{session_id}_typed_document.txt"

    # Serialise JSON
    serial = asdict(session)
    serial["events"] = [asdict(e) for e in session.events]
    log_path.write_text(json.dumps(serial, indent=2, ensure_ascii=False), encoding="utf-8")

    # Export CSV (Ref: Forensic ingestion standard 2025)
    import csv as csv_lib

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_lib.DictWriter(f, fieldnames=KeyEvent.__annotations__.keys())
        writer.writeheader()
        for e in session.events:
            writer.writerow(asdict(e))

    doc_path.write_text("".join(words_typed), encoding="utf-8")

    print(f"  [Phase 4] Telemetry log     -> {log_path}")
    print(f"  [Phase 4] Typed document    -> {doc_path}")
    print(f"  [Phase 4] Total chars typed : {session.total_chars_typed}")
    print(f"  [Phase 4] Total corrections : {session.total_corrections}")
    print(f"  [Phase 4] Total backspaces  : {session.total_backspaces}")
    simulated_rate = session.total_corrections / max(session.total_chars_typed, 1) * 100
    print(f"  [Phase 4] Actual correction rate: {simulated_rate:.1f}%")

    return session


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    aug_text_path = Path("output/phase1/augmented_text.txt")

    if aug_text_path.exists():
        target_text = aug_text_path.read_text(encoding="utf-8")
    else:
        target_text = (
            "Because, the informations provided comprises of various softwares. "
            "Likewise, he are responsible for all feedbacks. Hence, kindly do the "
            "needful and oblige. Moreover, the staffs was informed about the "
            "updation of records accordingly."
        )
        print("[WARN] Phase 1 output not found. Using built-in sample text.")

    print("\n======================================================")
    print("  Phase 4 - Keystroke Telemetry Simulation")
    if HEADLESS:
        print("  Mode: SIMULATION ONLY (no keyboard events fired)")
    else:
        print("  Mode: LIVE  (keyboard events will be fired in 3 s)")
        print("  Switch to your target text editor NOW.")
    print("======================================================\n")

    session = simulate_transcription(
        text=target_text,
        session_id="pake_ocr_ses_001",
        focus_window_title=None,  # set to window title for live mode
        output_dir=Path("output/phase4"),
        headless=HEADLESS,
    )

    print("\n======================================================")
    print("  Phase 4 - Telemetry Simulation Complete")
    print(f"  Session duration : {session.session_duration_sec:.2f}s")
    print("======================================================")
