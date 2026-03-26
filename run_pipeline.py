"""
run_pipeline.py — Master execution script for the PakE OCR Corpus Pipeline
Chains Phases 1 → 2 → 3 → 4 in sequence with full error handling and
per-phase timing telemetry.

Usage
-----
    python run_pipeline.py                        # use built-in sample text
    python run_pipeline.py path/to/input.txt      # use custom text file
    python run_pipeline.py --headless             # skip PyAutoGUI live typing

Pipeline Output Tree
--------------------
output/
├── phase1/
│   ├── augmented_text.txt
│   └── augmentation_metadata.json
├── phase2/
│   ├── rendered_page.png
│   └── render_metadata.json
├── phase3/
│   ├── degraded_page.png
│   └── degradation_metadata.json
└── phase4/
    ├── pake_ocr_ses_001_telemetry.json
    └── pake_ocr_ses_001_typed_document.txt
"""

import sys
import time
import json
import traceback
from pathlib import Path

# ── Ensure scripts directory is importable ────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

SAMPLE_TEXT = """
The committee comprises several members who stress the importance of sharing
information with all staff. Research indicates that software development requires
adequate equipment and furniture in office spaces. They are responsible for
providing feedback on the work submitted by all departments.

The team was asked to contact the department and discuss the matter in detail.
He is confident that the knowledge gained will benefit everyone involved in the
project. We are planning to update the records by next week as per the schedule.

The manager explained the new policy to all employees present in the meeting.
It is essential that advice from experts is followed carefully. Please reply to
this email at the earliest convenience and do the needful to proceed further.

Because the data entry requires special attention, hence all operators should
ensure accuracy. Likewise, the software updates must be installed on all machines.
Moreover, the informations regarding new procedures will be circulated shortly.
"""


def run_phase(name: str, func, *args, **kwargs):
    """Run a single pipeline phase with timing and error handling."""
    print(f"\n{'═' * 56}")
    print(f"  ► {name}")
    print(f"{'═' * 56}")
    t0 = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  ✓ {name} completed in {elapsed:.2f}s")
        return result, elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  ✗ {name} FAILED after {elapsed:.2f}s")
        print(f"    Error: {exc}")
        traceback.print_exc()
        return None, elapsed


def main():
    # ── CLI arguments ─────────────────────────────────────────────────────
    args     = sys.argv[1:]
    headless = "--headless" in args
    txt_args = [a for a in args if not a.startswith("--")]

    if txt_args:
        raw_text = Path(txt_args[0]).read_text(encoding="utf-8")
        print(f"[INFO] Input text loaded from: {txt_args[0]}")
    else:
        raw_text = SAMPLE_TEXT.strip()
        print("[INFO] Using built-in sample text.")

    timings = {}
    pipeline_start = time.perf_counter()

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1 — NLP Augmentation
    # ─────────────────────────────────────────────────────────────────────
    from phase1_nlp_augmentation import augment

    (aug_result, t1) = run_phase(
        "Phase 1 — NLP Augmentation (PakE Dialect Modeling)",
        augment, raw_text
    )
    timings["phase1"] = t1

    if aug_result is None:
        print("[FATAL] Phase 1 failed. Aborting pipeline.")
        sys.exit(1)

    augmented_text, p1_meta = aug_result

    # Save Phase 1 output for downstream phases
    out1 = Path("output/phase1")
    out1.mkdir(parents=True, exist_ok=True)
    (out1 / "augmented_text.txt").write_text(augmented_text, encoding="utf-8")
    (out1 / "augmentation_metadata.json").write_text(
        json.dumps(p1_meta, indent=2), encoding="utf-8"
    )

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2 — Digital Ink Synthesis
    # ─────────────────────────────────────────────────────────────────────
    from phase2_ink_synthesis import render_page

    (cv_img, t2) = run_phase(
        "Phase 2 — Digital Ink Synthesis (Handwriting Rendering)",
        render_page,
        title         = "PakE OCR Corpus — Synthetic Document",
        body_text     = augmented_text,
        output_path   = Path("output/phase2/rendered_page.png"),
        metadata_path = Path("output/phase2/render_metadata.json"),
    )
    timings["phase2"] = t2

    if cv_img is None:
        print("[FATAL] Phase 2 failed. Aborting pipeline.")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 3 — Environmental Degradation
    # ─────────────────────────────────────────────────────────────────────
    from phase3_degradation import degrade_image

    severity = "standard"
    if "--mild" in args: severity = "mild"
    if "--heavy" in args: severity = "heavy"

    (degraded, t3) = run_phase(
        "Phase 3 — Environmental Degradation (Augraphy Pipeline)",
        degrade_image,
        input_path    = Path("output/phase2/rendered_page.png"),
        output_path   = Path("output/phase3/degraded_page.png"),
        metadata_path = Path("output/phase3/degradation_metadata.json"),
        severity      = severity
    )
    timings["phase3"] = t3

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 4 — Telemetry Simulation
    # ─────────────────────────────────────────────────────────────────────
    from phase4_telemetry import simulate_transcription

    (session, t4) = run_phase(
        "Phase 4 — Keystroke Telemetry Simulation (Forensics)",
        simulate_transcription,
        text       = augmented_text,
        session_id = "pake_ocr_ses_001",
        output_dir = Path("output/phase4"),
        headless   = headless,
    )
    timings["phase4"] = t4

    # ─────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    total = time.perf_counter() - pipeline_start

    print(f"\n{'═' * 56}")
    print("  PIPELINE COMPLETE — Summary")
    print(f"{'═' * 56}")
    for phase, t in timings.items():
        status = "✓" if t > 0 else "✗"
        print(f"  {status}  {phase:<12}  {t:>6.2f}s")
    print(f"{'─' * 56}")
    print(f"     Total runtime : {total:.2f}s")
    print(f"{'═' * 56}")
    print("""
  Output files:
    output/phase1/augmented_text.txt
    output/phase1/augmentation_metadata.json
    output/phase2/rendered_page.png
    output/phase2/render_metadata.json
    output/phase3/degraded_page.png
    output/phase3/degradation_metadata.json
    output/phase4/pake_ocr_ses_001_telemetry.json
    output/phase4/pake_ocr_ses_001_typed_document.txt
""")


if __name__ == "__main__":
    main()
