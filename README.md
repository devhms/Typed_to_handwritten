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
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py waterfall_summary.txt --headless
```
