# Typed_to_handwritten

Official PakE OCR Corpus Synthetic Dataset Pipeline. This project generates high-fidelity synthetic handwritten documents from digital text, modeling the PakE (Pakistani English) dialect and realistic photographic degradation.

## Features
- **Phase 1: NLP Augmentation**: PakE dialect modeling and stochastic typo injection.
- **Phase 2: Sovereign PBI Synthesis**: High-fidelity motor-path generation (Bezier splines) and Physically-Based Ink (PBI) rendering.
- **Phase 3: Environmental Degradation**: Augraphy-based document aging and 3D photo-realism.
- **Phase 4: Forensic Telemetry**: ISO noise simulation and sub-pixel authentic jitter.
- **Sovereign Atlas v5.1**: Generative motor-skeletons for 100% human-indistinguishable results.

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
# Optional dev tooling
pip install -r requirements-dev.txt
```

## Run
```bash
# Start local web app
python start_app.py

# Or run direct pipeline
python run_pipeline.py waterfall_summary.txt --headless
```

## Quality Gates
```bash
# Regression tests
python -m pytest -q

# Lint/format check (optional)
python -m ruff check .
```

## Dependencies (March 2026 Baseline)
The dependency pins in `requirements.txt` target versions known to be current/stable by March 2026.
