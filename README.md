# Typed_to_handwritten

Official PakE OCR Corpus Synthetic Dataset Pipeline. This project generates high-fidelity synthetic handwritten documents from digital text, modeling the PakE (Pakistani English) dialect and realistic photographic degradation.

## Features
- **Phase 1: NLP Augmentation**: PakE dialect modeling.
- **Phase 2: Ink Synthesis**: Realistic handwriting rendering.
- **Phase 3: Environmental Degradation**: Augraphy-based document aging and photo-realism.
- **Phase 4: Telemetry Simulation**: Forensic keystroke dynamics capture.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py waterfall_summary.txt --headless
```
