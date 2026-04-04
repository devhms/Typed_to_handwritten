"""
Phase 3: Environmental Degradation — Augraphy Pipeline
Objective: Apply layered physical degradation to the rendered handwriting
image, simulating real-world document ageing, scanning artefacts, and
binarisation challenges for OCR robustness testing.

References:
  [11] Klieber et al. (2022) - Augraphy: A Data Augmentation Library
       for Document Images
  [12] Journet et al. (2017) - Document image binarisation
  [13] Stamatopoulos et al. (2011) - Goal-oriented rectification of
       camera-based document images
  [14] Shafait et al. (2008) - Efficient implementation of local
       adaptive thresholding techniques

Pipeline order follows Augraphy's three-phase architecture:
    Paper Phase  →  Ink Phase  →  Post Phase
"""

import cv2
import json
import sys
import numpy as np
from pathlib import Path

# ── Augraphy import guard ─────────────────────────────────────────────────────
AugraphyPipeline = None
ColorPaper = None
PaperFactory = None
PageBorder = None
InkBleed = None
Markup = None
InkMottling = None
Letterpress = None
Geometric = None
BadPhotoCopy = None
Dithering = None
Folding = None
SectionShift = None
ShadowCast = None
LightingGradient = None
DirtyDrum = None
OneOf = None
LowInkRandomLines = None
LowInkPeriodicLines = None

try:
    from augraphy import (
        AugraphyPipeline,
        # Paper Phase
        ColorPaper,
        PaperFactory,
        PageBorder,
        # Ink Phase
        InkBleed,
        Markup,
        InkMottling,
        Letterpress,
        # Post Phase
        Geometric,
        BadPhotoCopy,
        Dithering,
        Folding,
        SectionShift,
        ShadowCast,
        LightingGradient,
        DirtyDrum,
        # Stochastic selection
        OneOf,
        LowInkRandomLines,
        LowInkPeriodicLines,
    )

    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False
    print("[WARN] augraphy not installed — using OpenCV-only fallback degradation.")
    print("       Install with: pip install augraphy")


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Paper Phase
PAPER_TEXTURE_TYPE = "normal"  # 80 gsm fibrous paper simulation
PAPER_TEXTURE_SIGMA = (3, 5)  # Gaussian noise σ range for fibre texture

# Ink Phase — InkBleed
INKBLEED_SEVERITY = (0.4, 0.6)  # capillary spread intensity
INKBLEED_KERNEL_SIZE = (3, 5)  # bleed kernel (pixels)

# Ink Phase — BleedThrough
BLEEDTHROUGH_ALPHA = (0.15, 0.35)  # rear-page transparency range
BLEEDTHROUGH_OFFSET_X = (-10, 10)
BLEEDTHROUGH_OFFSET_Y = (-10, 10)

# Ink Phase — Markup (strikethrough)
MARKUP_TYPE = "strikethrough"
MARKUP_COVERAGE_RATIO = 0.02  # 2% of text lines (as specified)
MARKUP_THICKNESS = (1, 2)  # pixel thickness of strikethrough

# Post Phase — Geometric (perspective warp)
GEOMETRIC_ROTATE_RANGE = (-4, 4)  # degrees (increased for handheld photo look)
GEOMETRIC_SKEW_RANGE = (-0.03, 0.03)  # stronger keystone distortion for photo look

# Post Phase — BadPhotoCopy
BADPHOTOCOPY_NOISE = (0.05, 0.15)  # noise injection proportion
BADPHOTOCOPY_BLUR = (1, 2)  # Gaussian blur σ (slight camera defocus)
BADPHOTOCOPY_THRESH = (160, 240)  # wider threshold range


# ─────────────────────────────────────────────────────────────────────────────
# AUGRAPHY PIPELINE BUILDER
# ─────────────────────────────────────────────────────────────────────────────


def build_augraphy_pipeline(severity: str = "standard"):
    """
    Construct the three-phase Augraphy degradation pipeline with severity presets.

    severity: "mild", "standard", or "heavy"
    """

    if severity == "mild":
        mult = 0.5
    elif severity == "heavy":
        mult = 2.0
    else:
        mult = 1.0

    rotate_extent = max(1, int(round(1.5 * mult)))

    # ── Paper Phase ───────────────────────────────────────────────────────
    # NOTE: TextureGenerator is intentionally excluded because some Augraphy
    # versions pass mask/keypoints kwargs that TextureGenerator does not accept.
    paper_phase = [
        PaperFactory(p=0.6),
        ColorPaper(hue_range=(0, 10), saturation_range=(0, 10), p=0.3),
        PageBorder(p=0.4),
    ]

    # ── Ink Phase ─────────────────────────────────────────────────────────
    ink_phase = [
        InkBleed(intensity_range=(0.2 * mult, 0.4 * mult), p=0.7),
        InkMottling(p=0.3),
        Letterpress(p=0.2),
        OneOf(
            [
                LowInkRandomLines(p=1.0),
                LowInkPeriodicLines(p=1.0),
            ],
            p=0.3 * mult,
        ),
        Markup(num_lines_range=(1, int(3 * mult)), p=0.02 * mult),
    ]

    # ── Post Phase ────────────────────────────────────────────────────────
    post_phase = [
        Geometric(rotate_range=(-rotate_extent, rotate_extent), p=0.8),
        OneOf(
            [
                Folding(p=1.0),
                SectionShift(p=1.0),
            ],
            p=0.3 * mult,
        ),  # Reduced from 0.4
        ShadowCast(p=0.4 * mult),  # Reduced from 0.5
        LightingGradient(p=0.5 * mult),  # Reduced from 0.6
        DirtyDrum(p=0.2 * mult),  # Reduced from 0.3
        BadPhotoCopy(p=0.4 * mult),
        Dithering(p=0.1),
    ]

    return AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        save_outputs=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OPENCV-ONLY FALLBACK DEGRADATION
# (used when augraphy is not installed)
# ─────────────────────────────────────────────────────────────────────────────


def opencv_fallback_degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Pure-OpenCV degradation chain that approximates the Augraphy pipeline
    for environments where augraphy cannot be installed.
    """
    out = img.copy()

    # 1. Fibrous paper texture (additive Gaussian noise)
    sigma = rng.uniform(*PAPER_TEXTURE_SIGMA)
    noise = rng.normal(0, sigma, out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. InkBleed approximation — slight morphological dilation on dark pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.inRange(out, (0, 0, 0), (80, 80, 80))
    bleed = cv2.dilate(dark, kernel, iterations=1)
    out[bleed > 0] = np.clip(out[bleed > 0].astype(int) - 20, 0, 255).astype(np.uint8)

    # 3. BleedThrough — [DISABLED] for single-page assignment realism
    pass  # Removed to fix ghosting artifact reported in audit

    # 4. Markup — random horizontal strikethroughs over ~2% of lines
    h, w = out.shape[:2]
    n_lines = max(1, int(h / 90 * MARKUP_COVERAGE_RATIO))
    for _ in range(n_lines):
        y = int(rng.integers(50, h - 50))
        x_start = int(rng.integers(80, 200))
        x_end = int(rng.integers(w // 2, w - 80))
        thick = int(rng.integers(*MARKUP_THICKNESS))
        cv2.line(out, (x_start, y), (x_end, y), (20, 20, 20), thick)

    # 5. Geometric perspective warp (1–2°)
    angle_deg = float(rng.uniform(*GEOMETRIC_ROTATE_RANGE))
    cx, cy = w / 2, h / 2
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    out = cv2.warpAffine(out, M_rot, (w, h), borderValue=(255, 255, 255))

    # 6. BadPhotoCopy — uneven lighting + scanner noise
    noise_layer = rng.normal(0, 10, out.shape).astype(np.int16)  # Increased noise
    out = np.clip(out.astype(np.int16) + noise_layer, 0, 255).astype(np.uint8)
    # Uneven illumination gradient (simulates flash or room lighting)
    gradient = np.linspace(0.70, 1.15, w, dtype=np.float32)  # Stronger gradient
    gradient = np.tile(gradient, (h, 1))[:, :, np.newaxis]
    out = np.clip(out.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    # 7. Add Vignette (darker edges, simulates camera lens)
    rows, cols = np.indices((h, w))
    center_y, center_x = h / 2, w / 2
    dist = np.sqrt((rows - center_y) ** 2 + (cols - center_x) ** 2)
    max_dist = np.sqrt(center_y**2 + center_x**2)
    vig = 1.0 - (dist / max_dist) * 0.4  # Fade factor
    out = (out.astype(np.float32) * vig[:, :, np.newaxis]).astype(np.uint8)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEGRADATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


def degrade_image(
    input_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
    severity: str = "standard",
) -> np.ndarray:
    """
    Apply the full degradation pipeline to a rendered handwriting image.

    Parameters
    ----------
    input_path    : Path to Phase 2 rendered PNG
    output_path   : Destination for degraded image
    metadata_path : Optional JSON for degradation telemetry
    severity      : "mild", "standard", or "heavy"

    Returns
    -------
    degraded : np.ndarray (BGR, uint8)
    """
    # ── Load image ────────────────────────────────────────────────────────
    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    rng = np.random.default_rng(seed=7)

    augraphy_runtime_error = None

    # ── Apply pipeline ────────────────────────────────────────────────────
    if AUGRAPHY_AVAILABLE:
        try:
            pipeline = build_augraphy_pipeline(severity=severity)
            # Augraphy expects a numpy uint8 BGR image
            data = pipeline(img)
            degraded = data["output"] if isinstance(data, dict) else data
            method = f"augraphy_{severity}"
        except Exception as exc:
            augraphy_runtime_error = str(exc)
            print("[WARN] Augraphy failed; falling back to OpenCV degradation")
            print(f"       Reason: {augraphy_runtime_error}")
            degraded = opencv_fallback_degrade(img, rng)
            method = "opencv_fallback_after_augraphy_error"
    else:
        degraded = opencv_fallback_degrade(img, rng)
        method = "opencv_fallback"

    # ── Save result ───────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), degraded, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"  [Phase 3] Degraded image saved -> {output_path}")

    # ── Optional telemetry ────────────────────────────────────────────────
    if metadata_path:
        meta = {
            "degradation_method": method,
            "augraphy_available": AUGRAPHY_AVAILABLE,
            "augraphy_runtime_error": augraphy_runtime_error,
            "paper_texture_type": PAPER_TEXTURE_TYPE,
            "inkbleed_severity": list(INKBLEED_SEVERITY),
            "bleedthrough_alpha": list(BLEEDTHROUGH_ALPHA),
            "markup_coverage_ratio": MARKUP_COVERAGE_RATIO,
            "geometric_rotate_deg": list(GEOMETRIC_ROTATE_RANGE),
            "input_image": str(input_path),
            "output_image": str(output_path),
            "input_shape_hwc": list(img.shape),
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  [Phase 3] Degradation metadata -> {metadata_path}")

    return degraded


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_img = Path("output/phase2/rendered_page.png")
    output_img = Path("output/phase3/degraded_page.png")
    meta_file = Path("output/phase3/degradation_metadata.json")

    if not input_img.exists():
        print(f"[ERROR] Phase 2 output not found at {input_img}")
        print("        Run phase2_ink_synthesis.py first.")
        sys.exit(1)

    degrade_image(input_img, output_img, metadata_path=meta_file)

    print("\n======================================================")
    print("  Phase 3 - Environmental Degradation Complete")
    print(f"  Method used: {'Augraphy' if AUGRAPHY_AVAILABLE else 'OpenCV fallback'}")
    print("======================================================")
