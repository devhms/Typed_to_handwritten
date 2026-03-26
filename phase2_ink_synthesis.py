"""
Phase 2: Digital Ink Synthesis — Handwriting Rendering Pipeline
Objective: Render PakE-augmented text into realistic raster handwriting images
with calligraphic headings, blue body ink, and ruled notebook-style margins.

Stack:
  - Pillow      : raster canvas, image compositing
  - OpenCV      : double blue margin lines, post-processing
  - matplotlib  : stroke-trajectory debug visualisation
  - numpy       : per-glyph stroke noise (RNN-substitute approximation)

References:
  [7]  Graves (2013) - Generating Sequences With Recurrent Neural Networks
  [8]  Ha & Eck (2018) - A Neural Representation of Sketch Drawings (SketchRNN)
  [9]  Louloudis et al. (2008) - Block-based text extraction in handwritten pages
  [10] Gatos et al. (2004) - Automatic handwritten text segmentation
"""

import cv2
import json
import math
import numpy as np
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Canvas
PAGE_W, PAGE_H   = 2480, 3508          # A4 @ 300 dpi
MARGIN_LEFT      = 200                 # pixels from left edge
MARGIN_RIGHT     = PAGE_W - 120
TOP_MARGIN       = 280
LINE_HEIGHT      = 90                  # body text line height
HEADING_HEIGHT   = 130                 # calligraphic heading line height

# Ink colours
INK_HEADING      = (0,   0,   0)       # #000000 black  (BGR for OpenCV)
INK_BODY_PIL     = (8,   73,  127)     # #08497f blue   (RGB for Pillow)
INK_BODY_CV      = (127, 73,  8)       # #08497f blue   (BGR for OpenCV)
MARGIN_LINE_COL  = (200, 100, 50)      # double blue margin lines (BGR)

# Typography
FONT_DIR         = Path("fonts")       # place .ttf fonts here
HEADING_FONT_SZ  = 72
BODY_FONT_SZ     = 48

# Handwriting noise parameters (approximates RNN stroke variance, Ref [7][8])
NOISE_SIGMA_X    = 3.5                 # horizontal jitter σ (px) - Increased for more human look
NOISE_SIGMA_Y    = 4.0                 # vertical jitter σ (px) - Increased
SLANT_RANGE      = (-0.15, 0.12)       # italic shear range (radians) - Wider range
CHAR_SPACING_VAR = 10                  # ± pixel variance between characters - Increased
LINE_SPACING_VAR = 12                  # ± pixel variance between lines - Increased


# ─────────────────────────────────────────────────────────────────────────────
# FONT LOADING  (graceful fallback to default PIL font)
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        FONT_DIR / name,
        Path("/usr/share/fonts/truetype/liberation") / name,
        Path("/usr/share/fonts/truetype/dejavu") / "DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/freefont") / "FreeSerif.ttf",
    ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    # Last resort: PIL built-in bitmap font (no size control)
    return ImageFont.load_default()


def load_heading_font() -> ImageFont.FreeTypeFont:
    """
    Load a bold/heavy font to simulate 605 cut-marker calligraphy (Ref [7]).
    Prefers: DancingScript-Bold.ttf or Pacifico.ttf for chisel-tip look.
    """
    for name in ["DancingScript-Bold.ttf", "Pacifico-Regular.ttf",
                  "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"]:
        font = _load_font(name, HEADING_FONT_SZ)
        if not isinstance(font, ImageFont.ImageFont):
            return font
    return _load_font("DejaVuSans-Bold.ttf", HEADING_FONT_SZ)


def load_body_font() -> ImageFont.FreeTypeFont:
    """
    Load a handwriting-style font for body text rendered in blue ink.
    Prefers: Caveat-Regular.ttf, Satisfy-Regular.ttf, or DejaVuSans.
    """
    for name in ["Caveat-Regular.ttf", "Satisfy-Regular.ttf",
                  "LiberationSans-Regular.ttf", "DejaVuSans.ttf"]:
        font = _load_font(name, BODY_FONT_SZ)
        if not isinstance(font, ImageFont.ImageFont):
            return font
    return _load_font("DejaVuSans.ttf", BODY_FONT_SZ)


# ─────────────────────────────────────────────────────────────────────────────
# STROKE-LEVEL NOISE  (RNN/latent-diffusion approximation)
# ─────────────────────────────────────────────────────────────────────────────

class StrokeNoiseModel:
    """
    Approximates RNN-based handwriting variance (Ref [7], [8]) using:
      - Per-character positional jitter (Gaussian)
      - Global line-level slant / baseline drift
      - Brownian-motion cumulative x-drift across a word
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng(seed=42)

    def char_offset(self) -> tuple[int, int]:
        dx = int(self.rng.normal(0, NOISE_SIGMA_X))
        dy = int(self.rng.normal(0, NOISE_SIGMA_Y))
        return dx, dy

    def line_slant(self) -> float:
        return float(self.rng.uniform(*SLANT_RANGE))

    def char_spacing(self) -> int:
        return int(self.rng.integers(-CHAR_SPACING_VAR, CHAR_SPACING_VAR + 1))

    def line_spacing(self) -> int:
        return int(self.rng.integers(-LINE_SPACING_VAR, LINE_SPACING_VAR + 1))

    def word_drift(self, n_words: int) -> np.ndarray:
        """Cumulative Brownian drift across a line's words (Ref [7])."""
        steps = self.rng.normal(0, 4.0, n_words)
        return np.cumsum(steps).astype(int)

    def char_pressure(self) -> float:
        """Per-character ink opacity variation (0.85 - 1.0)."""
        return float(self.rng.uniform(0.85, 1.0))

    def ink_color_drift(self) -> tuple[int, int, int]:
        """Per-word ink saturation drift ±8 RGB."""
        dr = self.rng.integers(-8, 9)
        dg = self.rng.integers(-8, 9)
        db = self.rng.integers(-8, 9)
        return dr, dg, db

    def baseline_drift(self, n_chars: int) -> np.ndarray:
        """Cumulative Brownian drift over a word's characters."""
        steps = self.rng.normal(0, 1.2, n_chars)
        return np.cumsum(steps).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT WRAPPING  (word-level, respects right margin)
# ─────────────────────────────────────────────────────────────────────────────

def wrap_text(text: str, draw: ImageDraw.ImageDraw,
              font: ImageFont.FreeTypeFont,
              max_width: int) -> list[str]:
    words  = text.split()
    lines  = []
    line   = ""
    for word in words:
        test = (line + " " + word).strip()
        w    = draw.textlength(test, font=font)
        if w <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# DOUBLE MARGIN LINES  (OpenCV, Ref [9], [10])
# ─────────────────────────────────────────────────────────────────────────────

def draw_margin_lines(cv_img: np.ndarray) -> np.ndarray:
    """
    Draw two vertical blue margin lines on the left side of the page,
    4 pixels apart, simulating a standard ruled notebook (Ref [9], [10]).
    Also adds a book-binding crease shadow at x=60px.
    """
    x1 = MARGIN_LEFT - 12
    x2 = MARGIN_LEFT - 6
    thickness = 2
    cv2.line(cv_img, (x1, TOP_MARGIN - 60), (x1, PAGE_H - 100),
             MARGIN_LINE_COL, thickness)
    cv2.line(cv_img, (x2, TOP_MARGIN - 60), (x2, PAGE_H - 100),
             MARGIN_LINE_COL, thickness)
    
    # Book-binding crease shadow (Ref: Research update 2025)
    shadow_x = 60
    overlay = cv_img.copy()
    cv2.rectangle(overlay, (shadow_x - 30, 0), (shadow_x + 30, PAGE_H), (20, 20, 20), -1)
    cv_img = cv2.addWeighted(overlay, 0.15, cv_img, 0.85, 0)
    
    return cv_img


# ─────────────────────────────────────────────────────────────────────────────
# HORIZONTAL RULING LINES  (faint blue, notebook style)
# ─────────────────────────────────────────────────────────────────────────────

def draw_ruling_lines(cv_img: np.ndarray) -> np.ndarray:
    """Draw faint horizontal ruled lines across the page."""
    rule_col = (220, 170, 140)   # very faint blue (BGR)
    y = TOP_MARGIN
    while y < PAGE_H - 80:
        cv2.line(cv_img, (0, y), (PAGE_W, y), rule_col, 1)
        y += LINE_HEIGHT
    return cv_img


# ─────────────────────────────────────────────────────────────────────────────
# HEADING RENDERER  (chisel-tip calligraphy simulation)
# ─────────────────────────────────────────────────────────────────────────────

def render_heading(draw: ImageDraw.ImageDraw,
                   text: str,
                   x: int, y: int,
                   font: ImageFont.FreeTypeFont,
                   noise: StrokeNoiseModel) -> int:
    """
    Render a heading with thick black ink, simulating a 605 cut-marker
    chisel tip by drawing each character with a slight shadow/double stroke.
    Returns the new y position after the heading.
    """
    drift  = noise.baseline_drift(len(text))
    cursor = x
    for i, ch in enumerate(text):
        dx, dy = noise.char_offset()
        cy = y + drift[i] if i < len(drift) else y
        # Double-stroke for chisel-tip thickness simulation
        for offset in [(1, 1), (2, 1), (0, 0)]:
            draw.text(
                (cursor + dx + offset[0], cy + dy + offset[1]),
                ch, font=font, fill=(0, 0, 0)
            )
        cursor += int(draw.textlength(ch, font=font)) + noise.char_spacing()
    return y + HEADING_HEIGHT


# ─────────────────────────────────────────────────────────────────────────────
# BODY TEXT RENDERER  (blue ink, stroke-level jitter)
# ─────────────────────────────────────────────────────────────────────────────

def render_body_line(draw: ImageDraw.ImageDraw,
                      line: str,
                      x: int, y: int,
                      font: ImageFont.FreeTypeFont,
                      noise: StrokeNoiseModel,
                      ink_color: tuple[int, int, int]) -> None:
    """
    Render one body-text line character-by-character with:
      - per-character Gaussian jitter
      - Brownian baseline drift
      - variable character spacing
      - per-character pressure (opacity)
    """
    drift  = noise.baseline_drift(len(line))
    slant  = noise.line_slant()
    cursor = x
    for i, ch in enumerate(line):
        dx, dy = noise.char_offset()
        slant_offset = int(i * slant * 2)
        cy = y + (drift[i] if i < len(drift) else 0)
        
        # Per-character pressure (opacity)
        pressure = noise.char_pressure()
        color = tuple(int(c * pressure) for c in ink_color)
        
        draw.text(
            (cursor + dx + slant_offset, cy + dy),
            ch, font=font, fill=color
        )
        cursor += int(draw.textlength(ch, font=font)) + noise.char_spacing()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def draw_crossout(draw: ImageDraw.ImageDraw, x: int, y: int, length: int, line_height: int, noise: StrokeNoiseModel):
    """Draw a messy 'human' cross-out over a word."""
    y_center = y + line_height // 2
    for _ in range(3):  # Multiple strokes for messiness
        y1 = y_center + noise.rng.integers(-10, 11)
        y2 = y_center + noise.rng.integers(-10, 11)
        draw.line((x - 5, y1, x + length + 5, y2), fill=(30, 30, 30), width=4)

def render_page(title: str, body_text: str,
                output_path: Path,
                metadata_path: Path | None = None) -> np.ndarray:
    """Full page rendering pipeline with Authentic Assignment features."""
    
    # ── 1. Load Background Texture ────────────────────────────────────────
    bg_path = Path("assets/paper_texture.png")
    if bg_path.exists():
        pil_bg = Image.open(bg_path).resize((PAGE_W, PAGE_H))
    else:
        pil_bg = Image.new("RGB", (PAGE_W, PAGE_H), color=(252, 251, 245))
    
    # Text Layer (Transparent)
    text_layer = Image.new("RGBA", (PAGE_W, PAGE_H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(text_layer)

    # ── 2. Fonts & Noise ──────────────────────────────────────────────────
    heading_font = load_heading_font()
    body_font    = load_body_font()
    noise = StrokeNoiseModel()

    # ── 3. Rendering Logic ───────────────────────────────────────────────
    y = TOP_MARGIN
    y = render_heading(draw, title.upper(), MARGIN_LEFT, y,
                       heading_font, noise)
    y += 20

    max_w = MARGIN_RIGHT - MARGIN_LEFT
    paragraphs = body_text.split('\n')
    line_meta = []

    for para in paragraphs:
        if not para.strip():
            y += LINE_HEIGHT // 2
            continue
        
        current_x = MARGIN_LEFT + 100
        wrapped = wrap_text(para, draw, body_font, max_w - 100)
        
        # Consistent ink for the paragraph
        dr, dg, db = noise.ink_color_drift()
        current_ink = (
            max(0, min(255, INK_BODY_PIL[0] + dr)),
            max(0, min(255, INK_BODY_PIL[1] + dg)),
            max(0, min(255, INK_BODY_PIL[2] + db)),
            230 # Slight alpha for blending
        )

        for ln_idx, line_str in enumerate(wrapped):
            if y + LINE_HEIGHT > PAGE_H - 100: break
            
            lspacing = LINE_HEIGHT + noise.line_spacing()
            line_x = current_x if ln_idx == 0 else MARGIN_LEFT
            
            # ── [NEW] Human Error Model ──────────────────────────────────
            # 1% chance to cross out a word and rewrite it
            words = line_str.split()
            current_cursor = line_x
            for w_idx, word in enumerate(words):
                word_w = int(draw.textlength(word + " ", font=body_font))
                if noise.rng.random() < 0.015: # 1.5% chance
                    draw_crossout(draw, current_cursor, y, word_w, LINE_HEIGHT, noise)
                    current_cursor += word_w + 10 # small gap
                
                render_body_line(draw, word + " ", current_cursor, y,
                                 body_font, noise, current_ink)
                current_cursor += word_w

            line_meta.append({"y": y, "text": line_str[:50]})
            y += lspacing

    # ── 4. Composition (Multiply Blending) ───────────────────────────────
    # Combine text layer with background
    pil_bg.paste(text_layer, (0, 0), text_layer)
    cv_img = cv2.cvtColor(np.array(pil_bg), cv2.COLOR_RGB2BGR)

    # ── 5. Post-Processing ────────────────────────────────────────────────
    # We skip digital ruling lines because the texture already has them.
    # We only add a very subtle crease shadow if the texture is flat.
    cv_img = draw_margin_lines(cv_img)

    # ── 6. Save PNG ───────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    if metadata_path:
        meta = {"authentic_mode": True, "paper_texture": "ruled_notebook", "lines": len(line_meta)}
        metadata_path.write_text(json.dumps(meta, indent=2))

    return cv_img


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    aug_text_path = Path("output/phase1/augmented_text.txt")

    if aug_text_path.exists():
        body = aug_text_path.read_text(encoding="utf-8")
    else:
        body = (
            "Because, hence, the informations provided comprises of various "
            "softwares and equipments. Likewise, the staffs was informed about "
            "the updation of records. Moreover, he are responsible for all feedbacks "
            "received from the advices panel. Kindly do the needful and oblige."
        )
        print("[WARN] Phase 1 output not found. Using built-in sample.")

    render_page(
        title       = "Pakistani English OCR Corpus — Sample Page",
        body_text   = body,
        output_path = Path("output/phase2/rendered_page.png"),
        metadata_path = Path("output/phase2/render_metadata.json"),
    )

    print("\n══════════════════════════════════════════════════════")
    print("  Phase 2 — Ink Synthesis Complete")
    print("══════════════════════════════════════════════════════")
