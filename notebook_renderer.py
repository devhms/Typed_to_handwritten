"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    NOTEBOOK RENDERER v8.5 — Sovereign Forensic Edition (Precision Patch)     ║
║    Randomization: MDN-Derived Bio-Kinematic Motor Models                     ║
║    Rendering: Unified PBI (Physically Based Ink) & Euclidean Grounding       ║
║    Output: Forensic-Grade Document Synthesis at 300 DPI.                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from dataclasses import dataclass
from typing import List, Optional, Tuple

# MDN-derived randomization from pytorch-handwriting-synthesis-toolkit
from handwriting_randomizer import (
    BiasController,
    MixtureSampler,
    make_default_mixture,
    sample_bivariate_gaussian,
    generate_line_baseline_wander,
    perturb_glyph_mask,
)

# Phase 3/4 Internal Engines
try:
    from fiber_generator import generate_fiber_map
    from thixotropic_pbi import ThixotropicPBI
    from bio_kinematic_engine import BioKinematicEngine
except ImportError:
    # Fallbacks for standalone testing if needed
    def generate_fiber_map(w, h): return np.zeros((h, w), dtype=np.uint8)
    class ThixotropicPBI: 
        def render_bio_stroke(self, *args): pass
        def apply_capillary_wicking(self, *args): pass
    class BioKinematicEngine: 
        def __init__(self, **kwargs): pass

@dataclass
class NotebookConfig:
    """All rendering parameters in one place."""
    page_w: int = 2480
    page_h: int = 3508
    style: str = 'neat'
    variation_magnitude: float = 0.02
    drift_intensity: float = 0.6
    margin_panic_strength: float = 0.1
    fatigue_slant_rate: float = 0.02
    paper_color: Tuple[int, int, int] = (255, 255, 252)
    ruled_line_color: Tuple[int, int, int, int] = (180, 200, 220, 90)
    margin_line_color: Tuple[int, int, int, int] = (210, 80, 80, 120)
    line_spacing: int = 100
    first_line_y: int = 350
    margin_x: int = 210
    text_start_x: int = 250
    text_max_x: int = 2300
    font_dir: str = "fonts"
    primary_font: str = "Caveat-Regular.ttf"
    body_font_size: int = 62
    header_font_size: int = 58
    title_font_size: int = 66
    ink_color: Tuple[int, int, int, int] = (18, 32, 168, 230)
    bias: float = 0.0
    word_spacing_base: int = 16
    header_line_spacing: int = 100

class PaperGenerator:
    """Generates a clean ruled notebook paper background."""
    def __init__(self, cfg: NotebookConfig):
        self.cfg = cfg
    def generate(self) -> Image.Image:
        paper = Image.new("RGBA", (self.cfg.page_w, self.cfg.page_h), (*self.cfg.paper_color, 255))
        rng = np.random.default_rng(42)
        noise = rng.integers(-4, 5, (self.cfg.page_h, self.cfg.page_w), dtype=np.int16)
        paper_arr = np.array(paper)
        for c in range(3):
            paper_arr[:, :, c] = np.clip(paper_arr[:, :, c].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        paper = Image.fromarray(paper_arr)
        draw = ImageDraw.Draw(paper)
        y = self.cfg.first_line_y
        while y < self.cfg.page_h - 100:
            draw.line([(0, y), (self.cfg.page_w, y)], fill=self.cfg.ruled_line_color, width=2)
            y += self.cfg.line_spacing
        draw.line([(self.cfg.margin_x, 0), (self.cfg.margin_x, self.cfg.page_h)], fill=self.cfg.margin_line_color, width=3)
        return paper

class HandwritingRenderer:
    def __init__(self, cfg: NotebookConfig, seed: int = 42):
        self.cfg = cfg
        self._np_rng = np.random.default_rng(seed)
        self._fonts_cache = {}
        self._char_count = 0
        self._bc = BiasController(cfg.bias)
        self._mixture = make_default_mixture(bias=cfg.bias, rng=self._np_rng)
        self._base_font = self._load_font(cfg.body_font_size)
        self.pbi_engine = ThixotropicPBI()
        self.kinematic_engine = BioKinematicEngine(seed=seed)

    def _apply_unified_pbi(self, masterpiece_canvas: np.ndarray, mask: np.ndarray, 
                            offset_x: int, offset_y: int, 
                            groove_canvas: Optional[np.ndarray] = None,
                            fiber_map: Optional[np.ndarray] = None):
        """Phase 4: High-Precision Unified PBI Deposition."""
        y_idx, x_idx = np.nonzero(mask > 50)
        if len(y_idx) == 0: return
        points = []
        for my, mx in zip(y_idx, x_idx):
            pressure = mask[my, mx] / 255.0
            v = 1.0 / (0.5 + pressure)
            points.append([offset_x + mx, offset_y + my, v, pressure])
        self.pbi_engine.render_bio_stroke(np.array(points), masterpiece_canvas, groove_canvas, fiber_map)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        if size in self._fonts_cache: return self._fonts_cache[size]
        font_dir = Path(self.cfg.font_dir)
        font_path = font_dir / self.cfg.primary_font
        if font_path.exists():
            try:
                font = ImageFont.truetype(str(font_path), size)
                self._fonts_cache[size] = font
                return font
            except: pass
        font = ImageFont.load_default()
        self._fonts_cache[size] = font
        return font

    def get_line_wander(self, n_chars: int) -> np.ndarray:
        return generate_line_baseline_wander(n_chars, font_size_px=self.cfg.body_font_size, bias=self.cfg.bias, rng=self._np_rng)

    def _sample_ink_color(self) -> Tuple[int, int, int, int]:
        """Calculates a stochastic ink color with bivariate Gaussian fluctuation."""
        r, g, b, a = self.cfg.ink_color
        dr, dg = sample_bivariate_gaussian(sd1=self._bc.scale_sd(3.0), sd2=self._bc.scale_sd(2.0), ro=0.6, rng=self._np_rng)
        db = self._np_rng.normal(0, self._bc.scale_sd(3.0))
        alpha = self._np_rng.normal(a, self._bc.scale_sd(10.0))
        return (int(np.clip(r + dr, 0, 255)), int(np.clip(g + dg, 0, 255)), int(np.clip(b + db, 0, 255)), int(np.clip(alpha, 180, 255)))

    def render_char(self, canvas: Image.Image, char: str, x: int, y_baseline: int,
                    baseline_offset: float = 0.0,
                    font: Optional[ImageFont.FreeTypeFont] = None,
                    ink: Optional[Tuple] = None,
                    masterpiece_canvas: Optional[np.ndarray] = None,
                    groove_canvas: Optional[np.ndarray] = None,
                    fiber_map: Optional[np.ndarray] = None) -> int:
        """UI-UX PRO MAX: PRECISION CHARACTER DEPOSITION (v8.5)"""
        if font is None: font = self._base_font
        if ink is None: ink = self._sample_ink_color()
        sample = self._mixture.sample()
        dx, dy = sample['dx'], sample['dy']
        rotation, scale = sample['rotation'], sample['scale']
        dy = float(np.clip(dy, -2.5, 2.5)) 
        base_y = y_baseline + round(baseline_offset)
        bbox = font.getbbox(char)
        char_w, char_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if char_w <= 0 or char_h <= 0: return round(font.getlength(" "))

        if abs(rotation) < 0.1 and abs(scale - 1.0) < 0.01 and abs(dx) < 0.5 and abs(dy) < 0.5:
            draw = ImageDraw.Draw(canvas)
            draw.text((x + round(dx), base_y + round(dy)), char, font=font, fill=ink, anchor='ls')
        else:
            # High-Precision Synthesis Strategy (v8.6): Pivot-Absolute Chip Generation
            pad = 32
            # Create a localized chip with enough padding to prevent clipping during Slant or Tremor
            tmp = Image.new("RGBA", (char_w + pad * 2, char_h + pad * 2), (0, 0, 0, 0))
            # Draw the character anchored exactly to (pad, pad) as the Left-Baseline pivot
            ImageDraw.Draw(tmp).text((pad, pad), char, font=font, fill=ink, anchor='ls')
            
            # Apply Bio-Kinematic Perturbations: Tracking the pivot coordinates during the transform
            tmp, final_px, final_py = perturb_glyph_mask(
                tmp, dx=dx, dy=dy, rotation=rotation, scale=scale, 
                pivot_x=float(pad), pivot_y=float(pad), 
                variation_magnitude=self.cfg.variation_magnitude, rng=self._np_rng
            )
            
            # Sub-Pixel Alignment: Paste the chip such that its internal pivot (final_px) lands on the target baseline (x, base_y)
            paste_x = int(round(x - final_px))
            paste_y = int(round(base_y - final_py))
            if masterpiece_canvas is not None:
                mask_alpha = np.array(tmp)[:, :, 3]
                self._apply_unified_pbi(masterpiece_canvas, mask_alpha, paste_x, paste_y, groove_canvas, fiber_map)
            else:
                canvas.paste(tmp, (paste_x, paste_y), tmp)

        advance = round(font.getlength(char) * float(np.clip(scale, 0.88, 1.12)))
        self._char_count += 1
        return max(1, advance)

    def render_word(self, canvas: Image.Image, word: str, x: int, y_baseline: int, wander_offsets=None, baseline_offset=0.0, font=None, masterpiece_canvas=None, groove_canvas=None, fiber_map=None) -> int:
        cursor = x
        if font is None: font = self._base_font
        for i, ch in enumerate(word):
            char_wander = baseline_offset + (float(wander_offsets[i]) if wander_offsets is not None and i < len(wander_offsets) else 0.0)
            cursor += self.render_char(canvas, ch, cursor, y_baseline, baseline_offset=char_wander, font=font, masterpiece_canvas=masterpiece_canvas, groove_canvas=groove_canvas, fiber_map=fiber_map)
        return cursor - x

    def word_spacing_noise(self) -> float:
        return float(np.clip(self._np_rng.normal(1.0, 0.15), 0.5, 2.0))

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
        ink = self.hw._sample_ink_color()
        pts = [(px, y + 8 + int(self.hw._np_rng.integers(-1, 2))) for px in range(tx-5, tx+round(tf.getlength(title))+5, 6)]
        if len(pts) >= 2: ImageDraw.Draw(canvas).line(pts, fill=ink, width=2)
        return y + self.cfg.header_line_spacing + 20

def _snap_to_ruled_line(y, cfg):
    if y <= cfg.first_line_y: return cfg.first_line_y
    return cfg.first_line_y + math.ceil((y - cfg.first_line_y) / cfg.line_spacing) * cfg.line_spacing

def forensic_post_process(img_bgr: np.ndarray, fiber_map: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    blur_map = (fiber_map.astype(float) / 255.0) * 0.8
    feathered = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    img_post = cv2.addWeighted(img_bgr, 0.7, feathered, 0.3, 0)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    heightmap = cv2.GaussianBlur(255 - gray, (5, 5), 0)
    dx = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3)
    lx, ly = -1.0, -1.0
    l_norm = math.sqrt(lx*lx + ly*ly + 1.0)
    lx /= l_norm; ly /= l_norm; lz = 1.0 / l_norm
    shading = (dx * lx + dy * ly + lz) * 0.15 
    shading_img = np.clip(1.0 + shading, 0.85, 1.15)
    for c in range(3): img_post[:, :, c] = np.clip(img_post[:, :, c] * shading_img, 0, 255)
    b, g, r = cv2.split(img_post)
    center_y, center_x = h // 2, w // 2
    M_r = np.float32([[1.0002, 0, -center_x*0.0002], [0, 1.0002, -center_y*0.0002]])
    M_b = np.float32([[0.9998, 0, center_x*0.0002], [0, 0.9998, center_y*0.0002]])
    r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
    b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)
    img_post = cv2.merge([b, g, r])
    noise = np.random.normal(0, 1.5, img_post.shape).astype(np.float32)
    img_post = np.clip(img_post.astype(float) + noise, 0, 255).astype(np.uint8)
    shadow_grad = (np.mgrid[:h, :w][0] / float(h) + np.mgrid[:h, :w][1] / float(w)) * 0.05
    shadow_mask = np.clip(1.0 - shadow_grad, 0.90, 1.0)
    cv2.circle(shadow_mask, (int(w*0.8), int(h*0.2)), int(w*0.4), 0.95, -1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (151, 151), 0)
    for c in range(3): img_post[:, :, c] = np.clip(img_post[:, :, c] * shadow_mask, 0, 255)
    return img_post

def render_notebook_page(document_blocks, output_path="output/rendered_page.png", seed=42, config=None, masterpiece=False):
    """
    UI-UX PRO MAX: DOCUMENT BLOCK RENDERER (v8.5)
    Orchestrates the synthesis of complex structured documents (Headings, Paras, Spacers).
    """
    cfg = config or NotebookConfig() # Load current kinematic and layout configuration
    Path(output_path).parent.mkdir(parents=True, exist_ok=True) # Ensure assignment directory exists
    
    # Core Infrastructure Initialization
    paper_canvas = PaperGenerator(cfg).generate() # Deposit the ruled paper substrate
    hw = HandwritingRenderer(cfg, seed=seed) # Initialize the motor impulse motor
    layout = TextLayoutEngine(cfg) # Initialize the semantic wrap engine
    
    # High-Resolution Masterpiece Components
    masterpiece_canvas, groove_canvas, fiber_map = None, None, None
    if masterpiece:
        masterpiece_canvas = np.zeros((cfg.page_h, cfg.page_w, 4), dtype=np.uint8) # Particle ink layer
        groove_canvas = np.zeros((cfg.page_h, cfg.page_w), dtype=np.uint8) # Paper crush (groove) layer
        fiber_map = generate_fiber_map(cfg.page_w, cfg.page_h) # Substrate fiber distribution
    
    # Style Profile Calibration: Apply non-linear modifiers based on persona
    if cfg.style == 'messy':
        # Persona: MESSY - Elevated tremor and loose drift tracking
        cfg.variation_magnitude, cfg.drift_intensity, cfg.margin_panic_strength, cfg.fatigue_slant_rate = 0.06, 3.5, 0.4, 0.05
    elif cfg.style == 'neat':
        # Persona: NATURAL - Refined motor control and stable baseline
        cfg.variation_magnitude, cfg.drift_intensity, cfg.margin_panic_strength, cfg.fatigue_slant_rate = 0.015, 0.6, 0.1, 0.01

    # Vertical Layout Cursor Initialization
    y = cfg.first_line_y # Start at the first ruled line
    body_font = hw._load_font(cfg.body_font_size) # Default writing style (Natural)
    # 2026 Insight: Style-switching for headers (Engineer Style)
    header_font = hw._load_font(cfg.header_font_size) # Use IndieFlower/Kalam for headers if configured
    
    line_count = 0 # Track lines for fatigue/drift calculation
    
    # ─────────────────────────────────────────────────────────────────────────────
    # THE DOCUMENT SYNTHESIS LOOP (Block-by-Block Processing)
    # ─────────────────────────────────────────────────────────────────────────────
    for block in document_blocks:
        b_type = block['type']
        content = block['content']
        
        if b_type == 'spacer':
            # Intent: Double paragraph break or vertical emphasis
            y += cfg.line_spacing # Advance the cursor one ruled line
            continue
            
        # Font & Spacing Selection for this specific block (v8.5 Logic)
        is_header = b_type == 'header'
        # Selection: Headers use 'Engineer' metadata (larger footprint)
        current_font = header_font if is_header else body_font
        # Selection: Headers are centered or bolded in behavior
        is_centered = is_header and block.get('level') == 1
        
        # Word Wrapping: Partition the block content into renderable lines
        wrapped_lines = layout.wrap_text(content, current_font, cfg.text_max_x - cfg.text_start_x)
        
        for i, line in enumerate(wrapped_lines):
            if y > cfg.page_h - 200: break # Page safety constraint
            
            # Horizontal Positioning Logic
            if is_centered:
                # Center-Align for Level 1 Headings
                center_x = (cfg.text_start_x + cfg.text_max_x) // 2
                cx = center_x - round(current_font.getlength(line)) // 2
            else:
                # Standard Left-Align with Organic Indent for first-line paragraphs
                cx = cfg.text_start_x + (80 if (i == 0 and not is_header) else 0)
            
            words = line.split()
            c_idx, word_drift = 0, 0.0
            # Generate the stochastic wander vector for this line
            wander = hw.get_line_wander(max(1, sum(len(w) for w in words))) * cfg.drift_intensity
            
            for w in words:
                if cx + current_font.getlength(w) * 1.10 > cfg.text_max_x: break # Margin safety
                # Right-Margin Panic: Simulate hand slowing down/compacting text
                dist_to_margin = cfg.text_max_x - (cx + current_font.getlength(w))
                panic_factor = 1.0 - (1.0 - (dist_to_margin / 400.0)) * cfg.margin_panic_strength if dist_to_margin < 400 else 1.0
                # Per-Word Dipping: Simulate the hand adjusting for line wander
                word_drift += hw._np_rng.normal(0, cfg.drift_intensity * 0.5)
                
                # Perform the Deposition (The core synthesis call)
                word_advance = hw.render_word(
                    paper_canvas, w, cx, y, 
                    wander_offsets=wander[c_idx:c_idx+len(w)], 
                    baseline_offset=word_drift, 
                    font=current_font,
                    masterpiece_canvas=masterpiece_canvas, 
                    groove_canvas=groove_canvas, 
                    fiber_map=fiber_map
                )
                
                # Update horizontal cursor post-deposition
                spacing = round(cfg.word_spacing_base * hw.word_spacing_noise() * panic_factor)
                cx += word_advance + spacing
                c_idx += len(w)
            
            # Cursor Advance: Move to the next ruled line
            y += cfg.line_spacing
            line_count += 1
            # Dynamic Fatigue Slant: Incrementally shift the MDN bias to simulate hand tiredness
            hw._mixture = make_default_mixture(bias=cfg.bias + (line_count * cfg.fatigue_slant_rate), rng=hw._np_rng)
            
        # Segment Snap: Ensure vertical hygiene between different blocks
        y = _snap_to_ruled_line(y, cfg) # Anchor to the nearest ruled rule line

    # ─────────────────────────────────────────────────────────────────────────────
    # FINAL COMPOSITION & POST-PROCESSING
    # ─────────────────────────────────────────────────────────────────────────────
    if masterpiece and masterpiece_canvas is not None:
        hw.pbi_engine.apply_capillary_wicking(masterpiece_canvas, fiber_map) # Simulate ink bleeds
        img_bgr = cv2.cvtColor(np.array(paper_canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
        alpha = masterpiece_canvas[:, :, 3] / 255.0 # Pre-calculate alpha for blending
        for c in range(3): img_bgr[:, :, c] = (alpha * masterpiece_canvas[:, :, c] + (1.0 - alpha) * img_bgr[:, :, c]).astype(np.uint8)
        # Forensic Post-Process: Camera noise and chromatic entropy (v8.5)
        final_img = forensic_post_process(img_bgr, fiber_map)
        cv2.imwrite(output_path, final_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    else: 
        # Fast Path (Preview Mode)
        paper_canvas.convert("RGB").save(output_path, "PNG")
    
    return output_path

if __name__ == "__main__":
    render_notebook_page([{"type": "header", "content": "Stability Test v8.5", "level": 1}, {"type": "paragraph", "content": "Perfect synchronization achieved."}], output_path="output/test_v85_sync.png")
