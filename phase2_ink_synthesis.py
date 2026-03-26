import cv2
import numpy as np
import json
import math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = 2480, 3508 # A4 @ 300DPI
FONT_DIR = Path("fonts")

# ─────────────────────────────────────────────────────────────────────────────
# ORGANIC REALISM MODELS
# ─────────────────────────────────────────────────────────────────────────────

class StrokeNoiseModel:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.phase = self.rng.uniform(0, 2 * math.pi) # Baseline wander phase
        
    def char_offset(self):
        """Micro-jitter for each character."""
        return self.rng.integers(-3, 4), self.rng.integers(-3, 4)
    
    def char_spacing(self):
        """Variable tracking (organic hand spacing)."""
        return self.rng.integers(-2, 6)
    
    def char_pressure(self):
        """Simulate ink flow variance."""
        return self.rng.uniform(0.78, 1.0) # More range for "human" feel

    def baseline_wander(self, x_pos):
        """Organic sinusoidal drift on the ruling line."""
        amplitude = 6.0 # 6px drift
        frequency = 0.005
        return int(amplitude * math.sin(frequency * x_pos + self.phase))

# ─────────────────────────────────────────────────────────────────────────────
# RENDERING CORE
# ─────────────────────────────────────────────────────────────────────────────

def _load_font_stack(size: int):
    """Load multiple fonts for glyph blending."""
    stack = []
    # Try to load our 3 distinct handwriting styles
    names = ["Caveat-Regular.ttf", "GochiHand-Regular.ttf", "HomemadeApple-Regular.ttf"]
    for name in names:
        path = FONT_DIR / name
        if path.exists():
            stack.append(ImageFont.truetype(str(path), size))
    
    # Fallback if none found
    if not stack:
        stack.append(ImageFont.load_default())
    return stack

def wrap_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = " ".join(current_line + [word])
        w = draw.textlength(test_line, font=font)
        if w <= max_width:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def render_heading(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font: ImageFont.FreeTypeFont, noise: StrokeNoiseModel):
    """Render a calligraphic heading."""
    draw.text((x, y), text, font=font, fill=(5, 15, 60, 255), anchor="ls")
    return y

def render_body_line(draw: ImageDraw.ImageDraw,
                      line: str,
                      x: int, y: int,
                      fonts: list[ImageFont.FreeTypeFont],
                      noise: StrokeNoiseModel,
                      ink_color: tuple[int, int, int, int]) -> None:
    """[FULL POTENTIAL] Multi-Glyph Blending & Baseline Wander."""
    cursor = x
    for ch in line:
        # 1. Sinusoidal Baseline Wander
        wander_y = noise.baseline_wander(cursor)
        
        # 2. Glyph Randomization (pick from stack)
        font = random.choice(fonts)
        
        # 3. Micro-Jitter & Pressure
        dx, dy = noise.char_offset()
        pressure = noise.char_pressure()
        color = tuple(int(c * pressure) for c in ink_color)
        
        # 4. Render with 'ls' anchor
        draw.text((cursor + dx, y + dy + wander_y), ch, font=font, fill=color, anchor="ls")
        
        # 5. Organic spacing
        cursor += int(draw.textlength(ch, font=font)) + noise.char_spacing() + 2

def draw_margin_lines(img_cv: np.ndarray) -> np.ndarray:
    """Subtle red margin line if missing."""
    return img_cv # Texture usually has it

def render_page(title: str, body_text: str,
                output_path: Path,
                metadata_path: Path | None = None) -> np.ndarray:
    """[FULL POTENTIAL] Organic Rendering Engine."""
    W_A4, H_A4 = 2480, 3508
    bg_path = Path("assets/paper_texture.png")
    
    # Robust OpenCV Load
    cv_bg = cv2.imread(str(bg_path))
    if cv_bg is not None:
        cv_bg = cv2.cvtColor(cv_bg, cv2.COLOR_BGR2RGB)
        pil_bg = Image.fromarray(cv_bg).resize((W_A4, H_A4), Image.Resampling.LANCZOS)
    else:
        pil_bg = Image.new("RGB", (W_A4, H_A4), color=(252, 251, 245))
    
    text_layer = Image.new("RGBA", (W_A4, H_A4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    # 1. Load Calibration
    try:
        with open("assets/calibration.json", "r") as f:
            cal = json.load(f)
        line_ys = cal.get("all_line_ys", [])
        m_left  = cal.get("left_margin", 260)
        line_h  = cal.get("line_height", 134)
    except:
        line_ys, m_left, line_h = [], 260, 134

    class LineManager:
        def __init__(self, ys, default_h):
            self.ys = ys
            self.idx = 0
            self.last_y = 400
            self.h = default_h
        def next(self):
            if self.idx < len(self.ys):
                self.last_y = self.ys[self.idx]
                self.idx += 1
            else: self.last_y += self.h
            return self.last_y
        def skip(self, n=1):
            for _ in range(n): self.next()

    lm = LineManager(line_ys, line_h)
    body_sz = int(line_h * 0.72) 
    
    # [ORGANIC] Multi-Font Stack
    font_stack = _load_font_stack(body_sz)
    heading_font = font_stack[0] # Usually Caveat or similar
    noise = StrokeNoiseModel()

    # 2. Render Heading
    target_y = lm.next()
    render_heading(draw, title.upper(), m_left, target_y - 12, heading_font, noise)
    
    lm.skip(1)
    max_w = W_A4 - m_left - 150
    
    # 3. Dynamic Paragraph Splitting
    body_clean = body_text.strip()
    if body_clean.lower().startswith(title.lower()):
        body_clean = body_clean[len(title):].strip()
    
    if "\n\n" in body_clean: paragraphs = body_clean.split("\n\n")
    elif "\n" in body_clean: paragraphs = body_clean.split("\n")
    else:
        sentences = body_clean.split(". ")
        paragraphs = []
        for i in range(0, len(sentences), 4):
            paragraphs.append(". ".join(sentences[i:i+4]) + ".")

    # 4. Render Body
    for p_idx, para in enumerate(paragraphs):
        if not para.strip(): continue
        
        current_x = m_left + 150 # Indent
        wrapped = wrap_text(para, draw, font_stack[0], max_w - 150)
        ink = (0, 20, 80, 245) # Professional Blue

        for ln_idx, line_str in enumerate(wrapped):
            target_y = lm.next()
            if target_y > H_A4 - 200: break
            
            line_x = current_x if ln_idx == 0 else m_left
            # [Full Potential Sync]
            render_body_line(draw, line_str, line_x, target_y + 8,
                             font_stack, noise, ink)
        
        lm.skip(1)

    # 5. Composition
    pil_bg.paste(text_layer, (0, 0), text_layer)
    cv_img = cv2.cvtColor(np.array(pil_bg), cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv_img)
    return cv_img

if __name__ == "__main__":
    render_page(
        title       = "ORGANIC POTENTIAL TEST",
        body_text   = "This is a full potential test of the organic handwriting synthesis engine. No two characters should look identical because every glyph is sampled from a dynamic font stack.",
        output_path = Path("assignments/organic_test.png")
    )
