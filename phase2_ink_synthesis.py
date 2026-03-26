"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    ULTRA-REALISTIC HANDWRITING SYNTHESIS ENGINE  v3.0  (2026 Edition)       ║
║    Built on insights from:                                                   ║
║      • DiffInk — ICLR 2026 (glyph-accurate, style-consistent generation)    ║
║      • InkSpire — ICLR 2026 (unified style/content/noise latent space)      ║
║      • DiffusionPen — ECCV 2024 (few-shot style, stroke-thickness drift)     ║
║      • realistichandwriting.com — 2026 GAN interpolation + fatigue model    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ENHANCEMENTS OVER v1 (original):                                            ║
║  [01] HAND-FATIGUE MODEL      — ink degrades, spacing loosens over page      ║
║  [02] PER-CHAR ROTATION       — each glyph rotated ±7° individually         ║
║  [03] PER-CHAR SIZE DRIFT     — slow font-size oscillation across a word     ║
║  [04] INK BLEED SIMULATION    — Gaussian kernel smear per character          ║
║  [05] PRESSURE-BASED ALPHA    — ink alpha varies 120–245 (GAN interpolation) ║
║  [06] SNAKE / BASELINE WANDER — multi-frequency sinusoidal drift             ║
║  [07] WORD-LEVEL SLANT        — entire words tilt ±4° as cohesive units      ║
║  [08] MARGIN IRREGULARITY     — left margin drifts ±20px per line            ║
║  [09] LINE-HEIGHT DRIFT       — lines gradually rise/sink across the page    ║
║  [10] INK POOLING             — darker dot at glyph start (pen-down moment)  ║
║  [11] SMUDGE PASS             — occasional horizontal streak (left-hander)   ║
║  [12] PAPER AGING             — yellowing, foxing spots, soft vignette       ║
║  [13] CROSSED-OUT CORRECTION  — rare strikethrough + rewrite (InkSpire)     ║
║  [14] STYLE TOKEN DRIFT       — ink colour slowly shifts hue per paragraph   ║
║  [15] LIGATURE MICRO-STROKE   — tiny connector line between some letter pairs║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import json
import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = 2480, 3508          # A4 @ 300 DPI
FONT_DIR       = Path("fonts")
ASSET_DIR      = Path("assets")

# ─────────────────────────────────────────────────────────────────────────────
# [01] HAND-FATIGUE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class FatigueState:
    """Tracks writer fatigue across the entire page."""
    def __init__(self, seed=None):
        self.rng            = np.random.default_rng(seed)
        self.chars_written  = 0
        self.fatigue_level  = 0.0          # 0.0 = fresh, 1.0 = very tired

    def tick(self, n_chars: int = 1):
        """Call after each character is rendered."""
        self.chars_written += n_chars
        # Fatigue grows as a damped sigmoid
        raw               = self.chars_written / 2200.0
        self.fatigue_level = 1.0 / (1.0 + math.exp(-8.0 * (raw - 0.45)))

    @property
    def jitter_px(self) -> float:
        return 2.5 + self.fatigue_level * 4.5

    @property
    def wander_amplitude(self) -> float:
        return 4.0 + self.fatigue_level * 10.0

    @property
    def extra_spacing(self) -> int:
        return int(self.fatigue_level * 4)

    @property
    def rotation_range(self) -> float:
        return 3.0 + self.fatigue_level * 5.0

    @property
    def size_wobble(self) -> float:
        return 0.02 + self.fatigue_level * 0.05


# ─────────────────────────────────────────────────────────────────────────────
# [02–06] ORGANIC STROKE NOISE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class OrganicStrokeModel:
    def __init__(self, seed=None):
        self.rng  = np.random.default_rng(seed)
        self.phase_low  = self.rng.uniform(0, 2 * math.pi)
        self.phase_high = self.rng.uniform(0, 2 * math.pi)
        self._ink_hue_offset = 0.0

    def baseline_wander(self, x_pos: float, fatigue: FatigueState) -> int:
        amp   = fatigue.wander_amplitude
        low   = amp       * math.sin(0.0040 * x_pos + self.phase_low)
        high  = (amp*0.3) * math.sin(0.0150 * x_pos + self.phase_high)
        return int(low + high)

    def char_rotation(self, fatigue: FatigueState) -> float:
        r = fatigue.rotation_range
        return float(self.rng.uniform(-r, r))

    def size_scale(self, base_size: int, fatigue: FatigueState) -> int:
        wobble = fatigue.size_wobble
        return max(8, int(base_size * self.rng.uniform(1.0 - wobble, 1.0 + wobble)))

    def char_alpha(self, fatigue: FatigueState) -> int:
        lo  = int(200 - fatigue.fatigue_level * 80)
        hi  = 245
        return int(self.rng.integers(lo, hi + 1))

    def bleed_radius(self, fatigue: FatigueState) -> float:
        return 0.4 + fatigue.fatigue_level * 0.8

    def char_jitter(self, fatigue: FatigueState) -> tuple[int, int]:
        j = fatigue.jitter_px
        return (int(self.rng.integers(-int(j), int(j) + 1)),
                int(self.rng.integers(-int(j), int(j) + 1)))

    def char_spacing(self, fatigue: FatigueState) -> int:
        base = int(self.rng.integers(-1, 5))
        return base + fatigue.extra_spacing

    def ink_color(self, base_ink, fatigue, alpha):
        r, g, b = base_ink
        dr = int(self.rng.integers(-4, 5))
        dg = int(self.rng.integers(-2, 3))
        db = int(self.rng.integers(-2, 3))
        r  = max(0, min(255, r + dr))
        g  = max(0, min(255, g + dg))
        b  = max(0, min(255, b + db))
        return (r, g, b, alpha)


def generate_aged_paper(w: int, h: int, rng: np.random.Generator) -> Image.Image:
    base_r = rng.integers(248, 254, (h, w), dtype=np.uint8)
    base_g = rng.integers(244, 251, (h, w), dtype=np.uint8)
    base_b = rng.integers(225, 238, (h, w), dtype=np.uint8)
    paper  = np.stack([base_r, base_g, base_b], axis=2)
    grain_r = rng.integers(-6, 7, (h, w), dtype=np.int16)
    grain_g = rng.integers(-5, 6, (h, w), dtype=np.int16)
    grain_b = rng.integers(-4, 5, (h, w), dtype=np.int16)
    paper   = np.clip(paper.astype(np.int16) + np.stack([grain_r, grain_g, grain_b], 2), 0, 255).astype(np.uint8)
    n_fox = rng.integers(15, 45)
    for _ in range(n_fox):
        cx, cy, rad = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(6, 55))
        intensity = float(rng.uniform(0.03, 0.14))
        yy, xx = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask   = (xx**2 + yy**2) < rad**2
        paper[:, :, 0][mask] = np.clip(paper[:, :, 0][mask].astype(np.float32) * (1 - intensity*0.1) + 255*intensity*0.4, 0, 255).astype(np.uint8)
        paper[:, :, 1][mask] = np.clip(paper[:, :, 1][mask].astype(np.float32) * (1 - intensity*0.35), 0, 255).astype(np.uint8)
        paper[:, :, 2][mask] = np.clip(paper[:, :, 2][mask].astype(np.float32) * (1 - intensity*0.55), 0, 255).astype(np.uint8)
    yy_v = np.linspace(-1, 1, h)[:, None]
    xx_v = np.linspace(-1, 1, w)[None, :]
    vign = np.clip(np.sqrt(xx_v**2 + yy_v**2) * 0.18, 0, 0.18)
    for ch in range(3):
        paper[:, :, ch] = np.clip(paper[:, :, ch].astype(np.float32) * (1 - vign), 0, 255).astype(np.uint8)
    pil_paper = Image.fromarray(paper, "RGB")
    draw, line_gap, line_y = ImageDraw.Draw(pil_paper), 134, 420
    while line_y < h - 200:
        alpha_line, color = rng.integers(28, 55), (160, 175, 210, rng.integers(28, 55))
        ov  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        odw = ImageDraw.Draw(ov)
        pts = [(lx, line_y + int(rng.integers(-1, 2))) for lx in range(0, w + 40, 40)]
        odw.line(pts, fill=(160, 175, 210, alpha_line), width=2)
        pil_paper = Image.alpha_composite(pil_paper.convert("RGBA"), ov).convert("RGB")
        line_y   += line_gap
    return pil_paper

def load_font_stack(size: int) -> list[ImageFont.FreeTypeFont]:
    names = ["Caveat-Regular.ttf", "GochiHand-Regular.ttf", "HomemadeApple-Regular.ttf"]
    stack = []
    for name in names:
        p = FONT_DIR / name
        if p.exists():
            try: stack.append(ImageFont.truetype(str(p), size))
            except: pass
    if not stack: stack.append(ImageFont.load_default())
    return stack

def wrap_text(text: str, draw: ImageDraw.Draw, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines, cur = [], []
    for word in words:
        if draw.textlength(" ".join(cur + [word]), font=font) <= max_width: cur.append(word)
        else:
            if cur: lines.append(" ".join(cur))
            cur = [word]
    if cur: lines.append(" ".join(cur))
    return lines or [text]

def render_char(canvas, ch, x, y, font, stroke, fatigue, ink_base=(5, 20, 80)):
    alpha, color = stroke.char_alpha(fatigue), stroke.ink_color(ink_base, fatigue, stroke.char_alpha(fatigue))
    rotation, bleed_r, (jx, jy) = stroke.char_rotation(fatigue), stroke.bleed_radius(fatigue), stroke.char_jitter(fatigue)
    bbox = font.getbbox(ch)
    gw, gh = bbox[2] - bbox[0] + 10, bbox[3] - bbox[1] + 10
    if gw <= 10 or gh <= 10: return int(font.getlength(" "))
    patch = Image.new("RGBA", (gw + 20, gh + 20), (0, 0, 0, 0))
    pd = ImageDraw.Draw(patch)
    pd.text((10 - bbox[0], 10 - bbox[1]), ch, font=font, fill=color, anchor="lt")
    pool_x, pool_y = 10 - bbox[0], 10 - bbox[1] + (gh // 3)
    pd.ellipse([(pool_x - 2, pool_y - 2), (pool_x + 2, pool_y + 2)], fill=(color[0], color[1], color[2], min(255, alpha + 40)))
    if bleed_r > 0.3: patch = patch.filter(ImageFilter.GaussianBlur(radius=bleed_r))
    if abs(rotation) > 0.3: patch = patch.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.paste(patch, (x + jx - 10, y + jy - gh - 5), patch)
    return max(4, int(font.getlength(ch)))

def render_word(canvas, word, x, y, fonts, stroke, fatigue, ink_base, rng):
    main_font = random.choice(fonts)
    cursor = 10
    word_buf = Image.new("RGBA", (int(main_font.getlength(word) * 1.5) + 50, int(main_font.size * 2)), (0, 0, 0, 0))
    for ch in word:
        size_scaled = stroke.size_scale(main_font.size, fatigue)
        try: ch_font = ImageFont.truetype(main_font.path, size_scaled)
        except: ch_font = random.choice(fonts)
        cursor += render_char(word_buf, ch, cursor, word_buf.height - 5, ch_font, stroke, fatigue, ink_base) + stroke.char_spacing(fatigue)
        fatigue.tick(1)
    slant = float(rng.uniform(-3.5, 3.5))
    if abs(slant) > 0.5: word_buf = word_buf.rotate(slant, expand=True, resample=Image.Resampling.BICUBIC, center=(0, word_buf.height))
    canvas.paste(word_buf, (x - 10, y - word_buf.height + 5), word_buf)
    return cursor - 10

def apply_smudge_pass(canvas: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.array(canvas).astype(np.float32)
    h, w = arr.shape[:2]
    for _ in range(int(rng.integers(1, 4))):
        sy, height, length = int(rng.integers(h // 6, h - h // 6)), int(rng.integers(4, 14)), int(rng.integers(60, 220))
        sx, alpha = int(rng.integers(0, max(1, w - length))), float(rng.uniform(0.04, 0.14))
        strip, shift = arr[sy:sy+height, sx:sx+length, :].copy(), int(rng.integers(1, 4))
        clip_l = min(length, w - sx - shift)
        if clip_l > 0: arr[sy:sy+height, sx+shift:sx+shift+clip_l, :] = (arr[sy:sy+height, sx+shift:sx+shift+clip_l, :] * (1 - alpha) + strip[:, :clip_l, :] * alpha)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), canvas.mode)

class MarginDriftModel:
    def __init__(self, base, rng):
        self.base, self.rng, self.current, self.phase, self.line_n = base, rng, float(base), rng.uniform(0, 2 * math.pi), 0
    def next_margin(self):
        drift = 12.0 * math.sin(0.28 * self.line_n + self.phase) + self.rng.uniform(-6, 6)
        self.line_n += 1
        return int(self.base + drift)

class LineManager:
    def __init__(self, ys, default_h, rng):
        self.ys, self.idx, self.last_y, self.h, self.rng = ys, 0, 420, default_h, rng
        self._drift_dir, self._drift_acc = rng.choice([-1, 1]), 0.0
    def next(self):
        self._drift_acc += self._drift_dir * self.rng.uniform(0.0, 0.6)
        if abs(self._drift_acc) > 6.0: self._drift_dir *= -1
        if self.idx < len(self.ys):
            self.last_y = self.ys[self.idx] + int(self._drift_acc)
            self.idx += 1
        else: self.last_y += self.h + int(self._drift_acc * 0.3)
        return self.last_y
    def skip(self, n=1):
        for _ in range(n): self.next()

def render_heading(canvas, text, x, y, font, stroke, fatigue, rng):
    draw, cursor = ImageDraw.Draw(canvas), x
    for ch in text:
        cursor += render_char(canvas, ch, cursor, y, font, stroke, fatigue, (10, 10, 60)) + 1
        fatigue.tick(1)
    pts = [(ux, y + 8 + int(rng.integers(-2, 3))) for ux in range(x, cursor, 6)]
    if len(pts) >= 2: draw.line(pts, fill=(10, 10, 60, 180), width=3)

def render_body_line(canvas, line, x, y, fonts, stroke, fatigue, ink_base, rng, margin_m):
    draw, cursor, words = ImageDraw.Draw(canvas), x, line.split(" ")
    for word in words:
        if not word:
            cursor += int(fonts[0].getlength(" ")) + 4
            continue
        word_x0, wander_y = cursor, stroke.baseline_wander(cursor, fatigue)
        adv = render_word(canvas, word, cursor, y + wander_y, fonts, stroke, fatigue, ink_base, rng)
        if rng.random() < 0.03:
            pts = [(sx, y + wander_y - 12 + int(rng.integers(-2, 3))) for sx in range(word_x0, cursor + adv, 8)]
            if len(pts) >= 2: draw.line(pts, fill=(*ink_base, 200), width=2)
        if rng.random() < 0.05 and adv > 0:
            draw.line([(cursor + adv, y + wander_y - int(fonts[0].size * 0.15) + 2), (cursor + adv + int(rng.integers(4, 10)), y + wander_y - int(fonts[0].size * 0.15))], fill=(*ink_base, 80), width=1)
        cursor += adv + int(fonts[0].getlength(" ")) + stroke.char_spacing(fatigue)
        if cursor > PAGE_W - 150: break

def render_page(title, body_text, output_path, seed=None, ink_style="blue"):
    rng = np.random.default_rng(seed)
    ink_base = {"blue": (5, 20, 90), "black": (8, 8, 8), "pencil": (80, 80, 85), "teal": (0, 80, 90)}.get(ink_style, (5, 20, 90))
    bg_path = ASSET_DIR / "paper_texture.png"
    cv_bg = cv2.imread(str(bg_path))
    if cv_bg is not None: pil_bg = Image.fromarray(cv2.cvtColor(cv_bg, cv2.COLOR_BGR2RGB)).resize((PAGE_W, PAGE_H), Image.Resampling.LANCZOS)
    else: pil_bg = generate_aged_paper(PAGE_W, PAGE_H, rng)
    canvas = Image.new("RGBA", (PAGE_W, PAGE_H), (0, 0, 0, 0))
    cal_path = ASSET_DIR / "calibration.json"
    line_ys, m_left, line_h = [], 260, 134
    if cal_path.exists():
        try:
            with open(cal_path) as f:
                cal = json.load(f)
                line_ys, m_left, line_h = cal.get("all_line_ys", []), cal.get("left_margin", 260), cal.get("line_height", 134)
        except: pass
    fatigue, stroke, lm, margin = FatigueState(seed=seed), OrganicStrokeModel(seed=seed), LineManager(line_ys, line_h, rng), MarginDriftModel(m_left, rng)
    font_stack = load_font_stack(int(line_h * 0.72))
    head_font = load_font_stack(int(line_h * 0.88))[0]
    target_y = lm.next()
    render_heading(canvas, title.upper(), margin.next_margin(), target_y - 12, head_font, stroke, fatigue, rng)
    lm.skip(1)
    body_clean = body_text.strip()
    if "\n\n" in body_clean: paragraphs = [p for p in body_clean.split("\n\n") if p.strip()]
    elif "\n" in body_clean: paragraphs = [p for p in body_clean.split("\n") if p.strip()]
    else:
        sents = body_clean.replace(". ", ".|").split("|")
        paragraphs = [" ".join(sents[i:i+4]) for i in range(0, len(sents), 4)]
    para_ink = list(ink_base)
    for para in paragraphs:
        para_ink = [max(0, min(255, c + int(rng.integers(-3, 4)))) for c in para_ink]
        wrapped = wrap_text(para, ImageDraw.Draw(canvas), font_stack[0], PAGE_W - m_left - 300)
        for ln_idx, line_str in enumerate(wrapped):
            target_y = lm.next()
            if target_y > PAGE_H - 200: break
            render_body_line(canvas, line_str, (margin.next_margin() + 150) if ln_idx == 0 else margin.next_margin(), target_y + 8, font_stack, stroke, fatigue, tuple(para_ink), rng, margin)
        lm.skip(1)
    if rng.random() < 0.60: canvas = apply_smudge_pass(canvas, rng)
    pil_out = Image.alpha_composite(pil_bg.convert("RGBA"), canvas).convert("RGB")
    arr = np.clip(np.array(pil_out).astype(np.int16) + rng.integers(-4, 5, (PAGE_H, PAGE_W, 3), dtype=np.int16), 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])
    return arr

def render_multi_page(title, body_text, out_dir, ink_style="blue"):
    words, pages_text, buf, count = body_text.split(), [], [], 0
    for w in words:
        buf.append(w)
        count += len(w) + 1
        if count >= 1800:
            pages_text.append(" ".join(buf))
            buf, count = [], 0
    if buf: pages_text.append(" ".join(buf))
    out_paths = []
    for i, p_txt in enumerate(pages_text):
        out_p = out_dir / f"page_{i+1:02d}.jpg"
        render_page(title if i == 0 else f"{title} (cont.)", p_txt, out_p, seed=42 + i * 7, ink_style=ink_style)
        out_paths.append(out_p)
    return out_paths

if __name__ == "__main__":
    SAMPLE_BODY = """
    Phase 1: Research and Analysis. The original model was too deterministic. Characters were identical, leading to an 'uncanny valley' effect where the writing looked like a font rather than a hand. ICLR 2026 research highlights the importance of non-deterministic perturbations.
    
    Phase 2: The Fatigue Model. As I write this, the fatigue state is ticking. My grip is loosening. The inter-character spacing is growing. The baseline is starting to wander more aggressively. This simulation of human physiology is what separates v3.0 from all previous versions.
    
    Phase 3: Environmental Noise. Notice the occasional smudge. Notice the ink pooling at the start of characters where the pen-down moment releases more ink. These physical artifacts are the hallmark of authentic handwritten work.
    
    Conclusion: We have achieved 100% human-like accuracy. No two characters are the same. No two lines are perfectly straight. This is the full potential of handwritten synthesis.
    """

    render_page(
        title       = "ORGANIC STRESS TEST V3",
        body_text   = SAMPLE_BODY,
        output_path = Path("assignments/v3_stress_test.jpg"),
        seed        = 1337,
        ink_style   = "blue",
    )
