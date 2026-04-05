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

## Production Runbook

Use these commands before shipping changes:

```bash
# 1) Fast safety checks
python -m ruff check notebook_renderer.py server.py run_pipeline.py tools/ocr_benchmark.py tools/ocr_technique_sweep.py tests
python -m pytest -q tests/test_top_clipping_regression.py tests/test_render_api_compat.py tests/test_pipeline_cli.py tests/test_ocr_benchmark_stats.py

# 2) End-to-end pipeline smoke
python run_pipeline.py --headless --severity mild

# 3) OCR benchmark smoke with uncertainty
python tools/ocr_benchmark.py --runs 3 --mode preview --style neat --preprocess competitor_v4 --bootstrap-iterations 300 --label smoke

# 4) Technique sweep on generated report
python tools/ocr_technique_sweep.py --report "output/ocr_benchmark/report_smoke.json" --profile quick --max-samples 3
```

### Interpreting OCR metrics
- `CER` and `WER` in `tools/ocr_benchmark.py` are Levenshtein-based normalized error rates.
- Reported confidence intervals are percentile bootstrap intervals on the mean error rates.
- Compare techniques by both mean and CI width; prefer lower means with tighter intervals.

### Operational notes
- `run_pipeline.py` supports `--severity` and legacy `--mild/--heavy` flags for compatibility.
- `render_notebook_page` supports both block-model input and legacy `body_text` + `title` usage.
- `phase3_degradation.py` falls back to OpenCV if Augraphy fails at runtime; details are recorded in degradation metadata.
- Keep `.firecrawl/` and `augraphy_cache/` out of commits (already ignored).

## Vercel Operations

### Health endpoint
- `GET /health` returns deployment health metadata (service, runtime, version).
- Use this for uptime probes and quick post-deploy sanity checks.

### Function runtime budget
- `vercel.json` sets `server.py` limits:
  - `maxDuration: 60`
  - `memory: 1024`
- This keeps the generation endpoint bounded and predictable under load.

Note: Vercel function budgets are applied to `api/**/*.py` entrypoints.

### Post-deploy smoke test
```bash
python scripts/post_deploy_smoke.py --base-url https://typed-to-handwritten.vercel.app
```

Smoke test validates:
- `/health` returns `success=true` and `status=ok`
- `/generate` accepts a preview payload and returns image output

## Dependencies (March 2026 Baseline)
The dependency pins in `requirements.txt` target versions known to be current/stable by March 2026.
