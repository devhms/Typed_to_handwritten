"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    NOTEBOOK RENDERER v2.1 — MDN-Randomized (Stability Edition)             ║
║    Randomization: pytorch-handwriting-synthesis-toolkit (MDN mechanisms)    ║
║    Rendering: PIL font (Caveat-Regular.ttf) on ruled notebook paper        ║
║    Output: Flat A4 at 300 DPI. No perspective. No degradation.             ║
║                                                                            ║
║    v2.1 Fixes: paste_y formula, per-char wander, advance clamp,           ║
║    single RNG, ruled-line alignment, fast path, dead code removal.         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Phase 3/4 Dependencies
from fiber_generator import generate_fiber_map
from thixotropic_pbi import ThixotropicPBI
from bio_kinematic_engine import BioKinematicEngine

# MDN-derived randomization from pytorch-handwriting-synthesis-toolkit
from handwriting_randomizer import (
    BiasController,
    MixtureSampler,
    make_default_mixture,
    sample_bivariate_gaussian,
    generate_line_baseline_wander,
    perturb_glyph_mask,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NotebookConfig:
    """All rendering parameters in one place."""
    # Page geometry (A4 at 300 DPI)
    page_w: int = 2480
    page_h: int = 3508

    # Style Preset: 'neat' or 'messy'
    style: str = 'neat'
    
    # Parametric Variation (scaled by style)
    variation_magnitude: float = 0.02  # Deformation field strength
    drift_intensity: float = 0.6       # Baseline walk amplitude
    margin_panic_strength: float = 0.1 # Compression near right margin
    fatigue_slant_rate: float = 0.02   # Slant increase per line (%)

    # Paper colors
    paper_color: Tuple[int, int, int] = (255, 255, 252)
    ruled_line_color: Tuple[int, int, int, int] = (180, 200, 220, 90)
    margin_line_color: Tuple[int, int, int, int] = (210, 80, 80, 120)

    # Ruled line geometry (8.5mm at 300dpi ~ 100px)
    line_spacing: int = 100
    first_line_y: int = 350
    margin_x: int = 210

    # Text start
    text_start_x: int = 250
    text_max_x: int = 2300

    # Font (single font — MDN randomization handles variation)
    font_dir: str = "fonts"
    primary_font: str = "Caveat-Regular.ttf"
    body_font_size: int = 62
    header_font_size: int = 58
    title_font_size: int = 66

    # Ink color (blue ballpoint)
    ink_color: Tuple[int, int, int, int] = (18, 32, 168, 230)

    # MDN bias parameter
    bias: float = 0.0

    # Word spacing
    word_spacing_base: int = 16

    # Header spacing
    header_line_spacing: int = 100


# ─────────────────────────────────────────────────────────────────────────────
# PAPER GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class PaperGenerator:
    """Generates a clean ruled notebook paper background."""

    def __init__(self, cfg: NotebookConfig):
        self.cfg = cfg

    def generate(self) -> Image.Image:
        paper = Image.new("RGBA", (self.cfg.page_w, self.cfg.page_h),
                          (*self.cfg.paper_color, 255))

        # Apply subtle paper grain directly
        rng = np.random.default_rng(42)
        noise = rng.integers(-4, 5, (self.cfg.page_h, self.cfg.page_w), dtype=np.int16)
        paper_arr = np.array(paper)
        for c in range(3):
            paper_arr[:, :, c] = np.clip(
                paper_arr[:, :, c].astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)
        paper = Image.fromarray(paper_arr)

        draw = ImageDraw.Draw(paper)

        # Horizontal ruled lines
        y = self.cfg.first_line_y
        while y < self.cfg.page_h - 100:
            draw.line(
                [(0, y), (self.cfg.page_w, y)],
                fill=self.cfg.ruled_line_color,
                width=2
            )
            y += self.cfg.line_spacing

        # Red margin line
        draw.line(
            [(self.cfg.margin_x, 0), (self.cfg.margin_x, self.cfg.page_h)],
            fill=self.cfg.margin_line_color,
            width=3
        )

        return paper


# ─────────────────────────────────────────────────────────────────────────────
# HANDWRITING RENDERER
# ─────────────────────────────────────────────────────────────────────────────

class HandwritingRenderer:
    def __init__(self, cfg: NotebookConfig, seed: int = 42):
        self.cfg = cfg
        self._np_rng = np.random.default_rng(seed)
        self._fonts_cache = {}
        self._char_count = 0

        self._bc = BiasController(cfg.bias)
        self._mixture = make_default_mixture(bias=cfg.bias, rng=self._np_rng)
        self._base_font = self._load_font(cfg.body_font_size)
    
        # Phase 3 / v8.0 Engines
        self.pbi_engine = ThixotropicPBI()
        self.kinematic_engine = BioKinematicEngine(seed=seed)
    
    def _apply_unified_pbi(self, masterpiece_canvas: np.ndarray, mask: np.ndarray, 
                            offset_x: int, offset_y: int, 
                            groove_canvas: Optional[np.ndarray] = None,
                            fiber_map: Optional[np.ndarray] = None):
        """
        Phase 4: High-Precision Unified PBI Deposition.
        Converts mask to kinematic-like velocity/pressure distribution for the v8.0 PBI.
        """
        # Heuristic: map mask intensities to path points
        y_idx, x_idx = np.nonzero(mask > 50)
        if len(y_idx) == 0: return
        
        # Create a synthetic stroke density distribution
        # Since we have a mask (raster), we treat it as a dense cloud of points
        points = []
        for my, mx in zip(y_idx, x_idx):
            pressure = mask[my, mx] / 255.0
            # v is lower where mask is thicker (higher pressure)
            v = 1.0 / (0.5 + pressure)
            points.append([offset_x + mx, offset_y + my, v, pressure])
            
        self.pbi_engine.render_bio_stroke(np.array(points), masterpiece_canvas, groove_canvas, fiber_map)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        if size in self._fonts_cache:
            return self._fonts_cache[size]

        font_dir = Path(self.cfg.font_dir)
        font_path = font_dir / self.cfg.primary_font
        if font_path.exists():
            try:
                font = ImageFont.truetype(str(font_path), size)
                self._fonts_cache[size] = font
                return font
            except Exception:
                pass

        font = ImageFont.load_default()
        self._fonts_cache[size] = font
        return font

    def get_line_wander(self, n_chars: int) -> np.ndarray:
        return generate_line_baseline_wander(
            n_chars,
            font_size_px=self.cfg.body_font_size,
            bias=self.cfg.bias,
            rng=self._np_rng,
        )

    def _sample_ink_color(self) -> Tuple[int, int, int, int]:
        r, g, b, a = self.cfg.ink_color
        dr, dg = sample_bivariate_gaussian(
            sd1=self._bc.scale_sd(3.0),
            sd2=self._bc.scale_sd(2.0),
            ro=0.6,
            rng=self._np_rng,
        )
        db = self._np_rng.normal(0, self._bc.scale_sd(3.0))
        alpha = self._np_rng.normal(a, self._bc.scale_sd(10.0))
        return (
            int(np.clip(r + dr, 0, 255)),
            int(np.clip(g + dg, 0, 255)),
            int(np.clip(b + db, 0, 255)),
            int(np.clip(alpha, 180, 255)),
        )

    def render_char(self, canvas: Image.Image, char: str, x: int, y_baseline: int,
                    baseline_offset: float = 0.0,
                    font: Optional[ImageFont.FreeTypeFont] = None,
                    ink: Optional[Tuple] = None,
                    masterpiece_canvas: Optional[np.ndarray] = None,
                    groove_canvas: Optional[np.ndarray] = None,
                    fiber_map: Optional[np.ndarray] = None) -> int:
        if font is None:
            font = self._base_font
        if ink is None:
            ink = self._sample_ink_color()

        sample = self._mixture.sample()
        dx, dy = sample['dx'], sample['dy']
        rotation, scale = sample['rotation'], sample['scale']

        # Extra safety: clamp vertical jitter to prevent outliers jumping off rule
        dy = float(np.clip(dy, -2.5, 2.5))

        bbox = font.getbbox(char)
        char_w, char_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if char_w <= 0 or char_h <= 0:
            return round(font.getlength(" "))

        base_y = y_baseline + round(baseline_offset)

        # Fast path
        if abs(rotation) < 0.1 and abs(scale - 1.0) < 0.01 and abs(dx) < 0.5 and abs(dy) < 0.5:
            draw = ImageDraw.Draw(canvas)
            draw.text((x + round(dx), base_y + round(dy)), char, font=font, fill=ink, anchor='ls')
        else:
            # Slow path / Masterpiece Interceptor
            pad = 24
            tmp = Image.new("RGBA", (char_w + pad * 2, char_h + pad * 2), (0, 0, 0, 0))
            ImageDraw.Draw(tmp).text((pad - bbox[0], pad - bbox[1]), char, font=font, fill=ink)
            
            # Perturb with Elastic Warping and Precision Pivot Mapping (v8.1)
            variation_mag = self.cfg.variation_magnitude
            tmp, px, py = perturb_glyph_mask(
                tmp, dx=dx, dy=dy, rotation=rotation, scale=scale, 
                pivot_x=pad, pivot_y=pad,
                variation_magnitude=variation_mag, 
                rng=self._np_rng
            )
            
            # Accurate spatial anchoring (no more floating letters)
            paste_x = int(round(x + bbox[0] - pad + px))
            paste_y = int(round(base_y + bbox[1] - pad + py))
            
            if masterpiece_canvas is not None:
                mask_alpha = np.array(tmp)[:, :, 3]
                self._apply_unified_pbi(masterpiece_canvas, mask_alpha, paste_x, paste_y, groove_canvas, fiber_map)
            else:
                canvas.paste(tmp, (paste_x, paste_y), tmp)

        spacing_noise = float(np.clip(self._np_rng.normal(1.0, self._bc.spacing_jitter), 0.7, 1.4))
        advance = round(font.getlength(char) * float(np.clip(scale, 0.88, 1.12)) * spacing_noise)
        self._char_count += 1
        return max(1, advance)

    def render_word(self, canvas: Image.Image, word: str, x: int, y_baseline: int,
                    wander_offsets: Optional[np.ndarray] = None,
                    baseline_offset: float = 0.0,
                    font: Optional[ImageFont.FreeTypeFont] = None,
                    masterpiece_canvas: Optional[np.ndarray] = None,
                    groove_canvas: Optional[np.ndarray] = None,
                    fiber_map: Optional[np.ndarray] = None) -> int:
        cursor = x
        if font is None: font = self._base_font
        for i, ch in enumerate(word):
            char_wander = baseline_offset
            if wander_offsets is not None and i < len(wander_offsets):
                char_wander += float(wander_offsets[i])
            cursor += self.render_char(canvas, ch, cursor, y_baseline, 
                                     baseline_offset=char_wander, font=font,
                                     masterpiece_canvas=masterpiece_canvas,
                                     groove_canvas=groove_canvas,
                                     fiber_map=fiber_map)
        return cursor - x

    def word_spacing_noise(self) -> float:
        return float(np.clip(self._np_rng.normal(1.0, self._bc.spacing_jitter), 0.5, 2.0))

    def _get_jittered_ink(self) -> Tuple[int, int, int, int]:
        return self._sample_ink_color()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class TextLayoutEngine:
    def __init__(self, cfg: NotebookConfig): self.cfg = cfg
    def wrap_text(self, text, font, max_w):
        words = text.split()
        lines, current, current_w = [], [], 0
        sw = font.getlength(" ")
        for w in words:
            ww = font.getlength(w)
            if current_w + ww + (sw if current else 0) > max_w and current:
                lines.append(" ".join(current)); current, current_w = [w], ww
            else:
                current.append(w); current_w += ww + (sw if len(current) > 1 else 0)
        if current: lines.append(" ".join(current))
        return lines

class HeaderRenderer:
    def __init__(self, cfg, hw): self.cfg, self.hw = cfg, hw
    def render(self, canvas, lines, title, y):
        f, tf = self.hw._load_font(self.cfg.header_font_size), self.hw._load_font(self.cfg.title_font_size)
        cx = (self.cfg.text_start_x + self.cfg.text_max_x) // 2
        for l in lines:
            self.hw.render_word(canvas, l, cx - round(f.getlength(l)) // 2, y, font=f)
            y += self.cfg.header_line_spacing
        tx = cx - round(tf.getlength(title)) // 2
        self.hw.render_word(canvas, title, tx, y, font=tf)
        ink = self.hw._get_jittered_ink()
        pts = [(px, y + 8 + int(self.hw._np_rng.integers(-1, 2))) for px in range(tx-5, tx+round(tf.getlength(title))+5, 6)]
        if len(pts) >= 2: ImageDraw.Draw(canvas).line(pts, fill=ink, width=2)
        return y + self.cfg.header_line_spacing + 20

def _snap_to_ruled_line(y, cfg):
    if y <= cfg.first_line_y: return cfg.first_line_y
    return cfg.first_line_y + math.ceil((y - cfg.first_line_y) / cfg.line_spacing) * cfg.line_spacing

def forensic_post_process(img_bgr: np.ndarray, fiber_map: np.ndarray) -> np.ndarray:
    """
    Phase 4: Forensic Camera Simulation and Substrate Physics.
    Includes Capillary Feathering, Normal-Map Displacement (Pen Grooves), 
    and Digital Sensor Telemetry (Noise/Chromatic Aberration).
    """
    h, w = img_bgr.shape[:2]
    
    # Pass 1: Capillary Feathering (Simulated bleed into fibers)
    # Guided blur based on fiber map intensity (ink bleeds where fibers are dense)
    blur_map = (fiber_map.astype(float) / 255.0) * 0.8
    feathered = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    img_post = cv2.addWeighted(img_bgr, 0.7, feathered, 0.3, 0)

    # Pass 2: Normal-Map Displacement (Pen Grooves)
    # Generate heightmap from luminance variation (inverted ink is "down")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    heightmap = cv2.GaussianBlur(255 - gray, (5, 5), 0)
    
    # Sobel gradients as normals
    dx = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3)
    
    # Virtual Desk Lamp (Top-Left)
    lx, ly = -1.0, -1.0
    light_norm = math.sqrt(lx*lx + ly*ly + 1.0)
    lx /= light_norm; ly /= light_norm; lz = 1.0 / light_norm
    
    # Lambertian shading for the groove
    shading = (dx * lx + dy * ly + lz) * 0.15 # Strength of shadowing
    shading_img = np.clip(1.0 + shading, 0.85, 1.15)
    
    for c in range(3):
        img_post[:, :, c] = np.clip(img_post[:, :, c] * shading_img, 0, 255)

    # Pass 3: Digital Sensor Telemetry (Chromatic Aberration & Noise)
    # Split channels
    b, g, r = cv2.split(img_post)
    
    # Shift R/B slightly from center for CA (Reduced for 300DPI realism)
    center_y, center_x = h // 2, w // 2
    M_r = np.float32([[1.0002, 0, -center_x*0.0002], [0, 1.0002, -center_y*0.0002]])
    M_b = np.float32([[0.9998, 0, center_x*0.0002], [0, 0.9998, center_y*0.0002]])
    
    r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
    b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    img_post = cv2.merge([b, g, r])
    
    # Bayer Pattern Noise simulation
    noise = np.random.normal(0, 1.5, img_post.shape).astype(np.float32)
    img_post = np.clip(img_post.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Pass 4: Environmental Hand Shadow (Large-scale gradient)
    # Simulate the photographer or hand casting a soft shadow over the page
    shadow_mask = np.ones((h, w), dtype=np.float32)
    center_y, center_x = h // 2, w // 2
    # Gradient from top-left to bottom-right or similar
    yy, xx = np.mgrid[:h, :w]
    # Simple linear shadow from a corner
    shadow_grad = (yy / float(h) + xx / float(w)) * 0.05
    shadow_mask = np.clip(1.0 - shadow_grad, 0.90, 1.0) # Subtle 10% attenuation
    
    # Also add a "blobi" shadow (closer hand/phone)
    cv2.circle(shadow_mask, (int(w*0.8), int(h*0.2)), int(w*0.4), 0.95, -1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (151, 151), 0)
    
    for c in range(3):
        img_post[:, :, c] = np.clip(img_post[:, :, c] * shadow_mask, 0, 255)

    return img_post

def render_notebook_page(body_text, output_path="output/rendered_page.png", 
                         title="Assignment", header_lines=None, seed=42, 
                         config=None, masterpiece=False):
    cfg = config or NotebookConfig()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Preparations
    paper_canvas = PaperGenerator(cfg).generate()
    hw = HandwritingRenderer(cfg, seed=seed)
    layout = TextLayoutEngine(cfg)
    
    # Masterpiece Substrate setup
    masterpiece_canvas = None
    groove_canvas = None
    fiber_map = None
    if masterpiece:
        masterpiece_canvas = np.zeros((cfg.page_h, cfg.page_w, 4), dtype=np.uint8)
        groove_canvas = np.zeros((cfg.page_h, cfg.page_w), dtype=np.uint8)
        fiber_map = generate_fiber_map(cfg.page_w, cfg.page_h)

    # Apply style preset modifiers
    if cfg.style == 'messy':
        cfg.variation_magnitude = 0.06
        cfg.drift_intensity = 3.5
        cfg.margin_panic_strength = 0.4
        cfg.fatigue_slant_rate = 0.05
    elif cfg.style == 'neat':
        cfg.variation_magnitude = 0.015
        cfg.drift_intensity = 0.6
        cfg.margin_panic_strength = 0.1
        cfg.fatigue_slant_rate = 0.01

    y = _snap_to_ruled_line(HeaderRenderer(cfg, hw).render(paper_canvas, header_lines or [], title, cfg.first_line_y), cfg) + cfg.line_spacing
    f = hw._load_font(cfg.body_font_size)
    
    line_count = 0
    for p in [p.strip() for p in body_text.split("\n") if p.strip()]:
        for i, line in enumerate(layout.wrap_text(p, f, cfg.text_max_x - cfg.text_start_x)):
            if y > cfg.page_h - 200: break
            cx = cfg.text_start_x + (80 if i == 0 else 0)
            words = line.split()
            
            total_chars = sum(len(w) for w in words)
            # Apply Style-Driven drift intensity to wander
            wander = hw.get_line_wander(max(1, total_chars)) * cfg.drift_intensity
            
            # Word-level drift (accumulate over line)
            word_drift = 0.0
            
            c_idx = 0
            for w_idx, w in enumerate(words):
                if cx + f.getlength(w) * 1.10 > cfg.text_max_x: break
                
                # Right-Margin Panic: Compress spacing as we reach the right edge
                dist_to_margin = cfg.text_max_x - (cx + f.getlength(w))
                panic_factor = 1.0
                if dist_to_margin < 400:
                    panic_factor -= (1.0 - (dist_to_margin / 400.0)) * cfg.margin_panic_strength
                
                # Apply word-level dipping
                word_drift += hw._np_rng.normal(0, cfg.drift_intensity * 0.5)
                
                word_advance = hw.render_word(
                    paper_canvas, w, cx, y, 
                    wander_offsets=wander[c_idx:c_idx+len(w)],
                    baseline_offset=word_drift,
                    masterpiece_canvas=masterpiece_canvas,
                    groove_canvas=groove_canvas,
                    fiber_map=fiber_map
                )

                # Apply Panic to word spacing
                word_spacing = round(cfg.word_spacing_base * hw.word_spacing_noise() * panic_factor)
                cx += word_advance + word_spacing
                c_idx += len(w)
                
            y += cfg.line_spacing
            line_count += 1
            # Adjust global bias dynamically based on fatigue slant
            hw._mixture = make_default_mixture(bias=cfg.bias + (line_count * cfg.fatigue_slant_rate), rng=hw._np_rng)
            
        y = _snap_to_ruled_line(y + cfg.line_spacing, cfg)

    # FINAL COMPOSITION
    if masterpiece and masterpiece_canvas is not None:
        # v8.0 Post-Rendering: Capillary Wicking
        hw.pbi_engine.apply_capillary_wicking(masterpiece_canvas, fiber_map)
        
        # Convert paper to BGR for CV2
        img_bgr = cv2.cvtColor(np.array(paper_canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Alpha Blending of Particle Layer
        alpha = masterpiece_canvas[:, :, 3] / 255.0
        for c in range(3):
            img_bgr[:, :, c] = (alpha * masterpiece_canvas[:, :, c] + (1.0 - alpha) * img_bgr[:, :, c]).astype(np.uint8)
        
        # Phase 4 Post-Processing
        final_img = forensic_post_process(img_bgr, fiber_map)
        cv2.imwrite(output_path, final_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    else:
        # Fast Path
        paper_canvas.convert("RGB").save(output_path, "PNG")
        
    return output_path

if __name__ == "__main__":
    render_notebook_page("This is a stability test. No more flying letters.", output_path="output/test_v21_restore.png")
