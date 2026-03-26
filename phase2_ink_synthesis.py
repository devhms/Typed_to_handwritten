import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = 2480, 3508 # A4 @ 300DPI
FONT_DIR = Path("fonts")

# ─────────────────────────────────────────────────────────────────────────────
# NOISE & REALISM MODELS
# ─────────────────────────────────────────────────────────────────────────────

class StrokeNoiseModel:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        
    def char_offset(self):
        """Micro-jitter for each character."""
        return self.rng.integers(-2, 3), self.rng.integers(-2, 3)
    
    def char_spacing(self):
        """Variable tracking."""
        return self.rng.integers(-1, 4)
    
    def char_pressure(self):
        """Simulate ink flow variance."""
        return self.rng.uniform(0.85, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# RENDERING CORE
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(name: str, size: int):
    path = FONT_DIR / name
    if not path.exists():
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size)

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
    draw.text((x, y), text, font=font, fill=(10, 30, 80, 255), anchor="ls")
    return y

def render_body_line(draw: ImageDraw.ImageDraw,
                      line: str,
                      x: int, y: int,
                      font: ImageFont.FreeTypeFont,
                      noise: StrokeNoiseModel,
                      ink_color: tuple[int, int, int, int]) -> None:
    """Render with Baseline Anchor for absolute line snapping."""
    cursor = x
    for ch in line:
        dx, dy = noise.char_offset()
        pressure = noise.char_pressure()
        color = tuple(int(c * pressure) for c in ink_color)
        # 'ls' = Left-Baseline
        draw.text((cursor + dx, y + dy), ch, font=font, fill=color, anchor="ls")
        cursor += int(draw.textlength(ch, font=font)) + noise.char_spacing()

def draw_crossout(draw, x, y, word_w, line_h, noise):
    """Messy human-like crossout."""
    y_mid = y - line_h // 3 # Base on baseline
    for _ in range(3):
        y1 = y_mid + noise.rng.integers(-5, 6)
        y2 = y_mid + noise.rng.integers(-5, 6)
        draw.line((x - 5, y1, x + word_w + 5, y2), fill=(10, 30, 80, 200), width=4)

def draw_margin_lines(img_cv: np.ndarray) -> np.ndarray:
    """Subtle red margin line if missing."""
    return img_cv # Texture usually has it

def render_page(title: str, body_text: str,
                output_path: Path,
                metadata_path: Path | None = None) -> np.ndarray:
    """Full page rendering pipeline with Baseline-Anchor Sync."""
    W_A4, H_A4 = 2480, 3508
    bg_path = Path("assets/paper_texture.png")
    
    # [ROBUST] Use OpenCV to load then convert to PIL
    cv_bg = cv2.imread(str(bg_path))
    if cv_bg is not None:
        cv_bg = cv2.cvtColor(cv_bg, cv2.COLOR_BGR2RGB)
        pil_bg = Image.fromarray(cv_bg).resize((W_A4, H_A4), Image.Resampling.LANCZOS)
    else:
        pil_bg = Image.new("RGB", (W_A4, H_A4), color=(252, 251, 245))
    
    text_layer = Image.new("RGBA", (W_A4, H_A4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    # ── 1. Load Calibration ───────────────────────────────────────────────
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
    # Font size should be ~75% of line height for natural look
    body_sz = int(line_h * 0.75) 
    body_font = _load_font("Caveat-Regular.ttf", body_sz)
    # [FIX] DancingScript-Bold is corrupted; fallback to Caveat
    heading_font = _load_font("Caveat-Regular.ttf", int(body_sz * 1.5))
    noise = StrokeNoiseModel()

    # ── 2. Render Heading ─────────────────────────────────────────────────
    target_y = lm.next()
    # Heading font is larger, so we need a different offset
    render_heading(draw, title.upper(), m_left, target_y - 8, heading_font, noise)
    
    lm.skip(1)
    max_w = W_A4 - m_left - 150
    
    # [CLEANUP] Deduplicate title
    body_clean = body_text.strip()
    if body_clean.lower().startswith(title.lower()):
        body_clean = body_clean[len(title):].strip()
    
    # [ROBUST] Dynamic Paragraph Splitting
    if "\n\n" in body_clean:
        paragraphs = body_clean.split("\n\n")
    elif "\n" in body_clean:
        paragraphs = body_clean.split("\n")
    else:
        sentences = body_clean.split(". ")
        paragraphs = []
        for i in range(0, len(sentences), 5):
            paragraphs.append(". ".join(sentences[i:i+5]) + ".")

    # ── 3. Render Body ───────────────────────────────────────────────────
    for p_idx, para in enumerate(paragraphs):
        if not para.strip(): continue
        
        current_x = m_left + 160 # Clear indent
        wrapped = wrap_text(para, draw, body_font, max_w - 160)
        ink = (10, 30, 90, 248) # Dark Academic Blue

        for ln_idx, line_str in enumerate(wrapped):
            target_y = lm.next()
            if target_y > H_A4 - 200: break
            
            line_x = current_x if ln_idx == 0 else m_left
            # anchor='ls' + offset makes characters 'kiss' the line
            render_body_line(draw, line_str, line_x, target_y + 5,
                             body_font, noise, ink)
        
        lm.skip(1) # Gap between paragraphs

    # ── 4. Composition ───────────────────────────────────────────────────
    pil_bg.paste(text_layer, (0, 0), text_layer)
    cv_img = cv2.cvtColor(np.array(pil_bg), cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv_img)
    return cv_img

if __name__ == "__main__":
    body = Path("my_history_assignment.txt").read_text(encoding="utf-8")
    render_page(
        title       = "History Assignment: The Industrial Revolution",
        body_text   = body,
        output_path = Path("assignments/my_history_assignment_handwritten_photo.png")
    )
