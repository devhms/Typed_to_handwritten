#!/usr/bin/env bash
# =============================================================================
# EXECUTION COMMANDS  —  PakE OCR Corpus Synthetic Dataset Pipeline
# =============================================================================
#
# PREREQUISITES
# -------------
# Python >= 3.11, pip, (optionally) a graphical display for Phase 4 live mode.
# Place any .ttf handwriting fonts (Caveat, DancingScript, etc.) into ./fonts/
#
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Environment setup
# ─────────────────────────────────────────────────────────────────────────────

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Run individual phases (for testing / inspection)
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1: NLP augmentation on a custom input file
python phase1_nlp_augmentation.py path/to/raw_text.txt

# Phase 1: Built-in sample text
python phase1_nlp_augmentation.py

# Phase 2: Render augmented text to handwriting image
#   (requires Phase 1 output at output/phase1/augmented_text.txt)
python phase2_ink_synthesis.py

# Phase 3: Apply environmental degradation
#   (requires Phase 2 output at output/phase2/rendered_page.png)
python phase3_degradation.py

# Phase 4: Simulate keystroke telemetry (headless — no keyboard events fired)
#   (requires Phase 1 output at output/phase1/augmented_text.txt)
python phase4_telemetry.py


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Run full pipeline (recommended)
# ─────────────────────────────────────────────────────────────────────────────

# Using built-in sample text, headless telemetry (safe, no keyboard events):
python run_pipeline.py --headless

# Using a custom input text file, headless:
python run_pipeline.py path/to/raw_text.txt --headless

# Using a custom input text file, LIVE telemetry
# (switch to a text editor within 3 s of running this command):
python run_pipeline.py path/to/raw_text.txt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Batch dataset generation (N pages)
# ─────────────────────────────────────────────────────────────────────────────

# Generate 100 degraded page images from a corpus directory:
for i in $(seq 1 100); do
    python run_pipeline.py corpus/page_${i}.txt --headless
    mv output/phase3/degraded_page.png dataset/degraded/page_${i}.png
    mv output/phase4/pake_ocr_ses_001_telemetry.json \
       dataset/telemetry/page_${i}_telemetry.json
done


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Verify outputs
# ─────────────────────────────────────────────────────────────────────────────

# List all generated files:
find output/ -type f | sort

# Quick image inspection (requires imagemagick):
display output/phase2/rendered_page.png   # clean render
display output/phase3/degraded_page.png   # degraded output

# View telemetry stats:
python -c "
import json
data = json.load(open('output/phase4/pake_ocr_ses_001_telemetry.json'))
print('Total chars typed :', data['total_chars_typed'])
print('Total corrections :', data['total_corrections'])
print('Correction rate   :', round(data['total_corrections']/data['total_chars_typed']*100,2), '%')
print('Session duration  :', round(data['session_duration_sec'], 2), 's')
print('Total events      :', len(data['events']))
"


# ─────────────────────────────────────────────────────────────────────────────
# NOTES
# ─────────────────────────────────────────────────────────────────────────────
# • augraphy may require: pip install numba (for JIT acceleration)
# • pyautogui requires a display; on headless servers use Xvfb:
#       sudo apt-get install xvfb
#       Xvfb :99 &  &&  export DISPLAY=:99
# • For best handwriting visuals, download free OFL fonts:
#       Caveat:        https://fonts.google.com/specimen/Caveat
#       DancingScript: https://fonts.google.com/specimen/Dancing+Script
#   and place them in ./fonts/
