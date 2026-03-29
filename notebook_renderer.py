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
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont  # Fix 3: removed unused ImageFilter
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


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NotebookConfig:
    """All rendering parameters in one place."""
    # Page geometry (A4 at 300 DPI)
    page_w: int = 2480
    page_h: int = 3508

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

    # MDN bias parameter: controls ALL variation with a single knob
    #   bias=0.0  -> default natural student writing
    #   bias=0.5  -> neater, tighter (exam quality)
    #   bias=-0.5 -> messier, looser (rushed writing)
    bias: float = 0.0

    # Word spacing
    word_spacing_base: int = 16

    # Fix 5: header_line_spacing matches line_spacing to stay on ruled grid
    header_line_spacing: int = 100


# ─────────────────────────────────────────────────────────────────────────────
# PAPER GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class PaperGenerator:
    """Generates a clean ruled notebook paper background."""

    def __init__(self, cfg: NotebookConfig):
        self.cfg = cfg

    def generate(self) -> Image.Image:
        """Create the paper background with ruled lines and margin."""
        paper = Image.new("RGBA", (self.cfg.page_w, self.cfg.page_h),
                          (*self.cfg.paper_color, 255))

        # Fix 2: Apply subtle paper grain directly — no dead `grain` variable
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

        # Red margin line (full height)
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
    """
    Renders individual characters using MDN-derived randomization.
    Uses a single font (Caveat) with per-character perturbations from
    the MixtureSampler (3 writing modes: neat/rushed/careful).

    All variation is controlled by a single 'bias' parameter:
      bias=0.0  -> natural student writing
      bias=0.5  -> neater, exam-quality
      bias=-0.5 -> messier, rushed

    v2.1: Single RNG stream (_np_rng only), clamped advance,
    correct paste_y formula per Pillow docs, fast path restored.
    """

    def __init__(self, cfg: NotebookConfig, seed: int = 42):
        self.cfg = cfg
        # Fix 9: Single RNG — _np_rng for all randomness
        self._np_rng = np.random.default_rng(seed)
        self._fonts_cache = {}
        self._char_count = 0

        # MDN-derived systems
        self._bc = BiasController(cfg.bias)
        self._mixture = make_default_mixture(bias=cfg.bias, rng=self._np_rng)

        # Pre-load the base font
        self._base_font = self._load_font(cfg.body_font_size)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load primary font at the given size."""
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
        """
        Generate baseline wander for a full line using correlated random walk.
        PUBLIC method (Fix 8) — called from render_notebook_page().

        This is how the LSTM-MDN naturally produces baseline drift:
        cumulative sum of correlated y-offset samples.
        """
        return generate_line_baseline_wander(
            n_chars,
            font_size_px=self.cfg.body_font_size,
            bias=self.cfg.bias,
            rng=self._np_rng,
        )

    def _sample_ink_color(self) -> Tuple[int, int, int, int]:
        """
        Per-character ink color via bivariate Gaussian (Mechanism 2).
        R and G channels are correlated (pen pressure affects both),
        B channel is semi-independent. Alpha varies with pressure.
        """
        r, g, b, a = self.cfg.ink_color
        # Correlated R/G drift (pen pressure coupling)
        dr, dg = sample_bivariate_gaussian(
            sd1=self._bc.scale_sd(3.0),
            sd2=self._bc.scale_sd(2.0),
            ro=0.6,
            rng=self._np_rng,
        )
        db = self._np_rng.normal(0, self._bc.scale_sd(3.0))

        # Alpha variation (pressure)
        alpha_sd = self._bc.scale_sd(10.0)
        alpha = self._np_rng.normal(a, alpha_sd)

        return (
            int(np.clip(r + dr, 0, 255)),
            int(np.clip(g + dg, 0, 255)),
            int(np.clip(b + db, 0, 255)),
            int(np.clip(alpha, 180, 255)),
        )

    def render_char(self, canvas: Image.Image, char: str, x: int, y_baseline: int,
                    baseline_offset: float = 0.0,
                    font: Optional[ImageFont.FreeTypeFont] = None,
                    ink: Optional[Tuple] = None) -> int:
        """
        Render a single character using MDN MixtureSampler.

        Each call samples from one of 3 mixture components (neat/rushed/careful),
        producing correlated (dx, dy) offsets, rotation, and scale — exactly
        as the LSTM-MDN does per pen-step.

        Fix 1: paste_y corrected per Pillow docs (y + bbox[1]).
        Fix 7: advance multiplier clamped to [0.7, 1.4].
        Fix 11: fast path restored for near-zero transforms.
        """
        if font is None:
            font = self._base_font
        if ink is None:
            ink = self._sample_ink_color()

        # Sample perturbation from mixture (Mechanism 3)
        sample = self._mixture.sample()
        dx = sample['dx']
        dy = sample['dy']
        rotation = sample['rotation']
        scale = sample['scale']

        bbox = font.getbbox(char)
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        if char_w <= 0 or char_h <= 0:
            return round(font.getlength(" "))

        # Calculate baseline-corrected y position
        base_y = y_baseline + round(baseline_offset)

        # Fix 11: Fast path — direct draw when transforms are negligible
        if abs(rotation) < 0.1 and abs(scale - 1.0) < 0.01 and abs(dx) < 0.5 and abs(dy) < 0.5:
            draw = ImageDraw.Draw(canvas)
            draw_x = x + round(dx)
            draw_y = base_y + round(dy)
            # anchor='ls' = left-baseline: Pillow positions text so baseline sits at y
            if (0 <= draw_x < canvas.width - char_w and
                    0 <= draw_y - char_h and draw_y < canvas.height):
                draw.text((draw_x, draw_y), char, font=font, fill=ink, anchor='ls')
        else:
            # Slow path: temp canvas for rotation/scale
            pad = 12
            tmp_w = char_w + pad * 2
            tmp_h = char_h + pad * 2
            tmp = Image.new("RGBA", (tmp_w, tmp_h), (0, 0, 0, 0))
            tmp_draw = ImageDraw.Draw(tmp)
            # Draw at (-bbox[0]+pad, -bbox[1]+pad) so glyph is centered in temp
            tmp_draw.text((pad - bbox[0], pad - bbox[1]), char, font=font, fill=ink)

            # Apply MDN perturbation (Mechanism 9: scale + rotation)
            tmp, px, py = perturb_glyph_mask(
                tmp, dx=dx, dy=dy,
                rotation=rotation, scale=scale,
                anchor_bottom=True,
            )

            # Fix 1: Correct paste formula per Pillow docs
            # When drawing at (pad - bbox[0], pad - bbox[1]) in temp canvas,
            # paste at (x + bbox[0] - pad, y_baseline + bbox[1] - pad) to align baseline.
            paste_x = int(x + bbox[0] - pad + px)
            paste_y = int(base_y + bbox[1] - pad + py)

            if (0 <= paste_x < canvas.width - tmp.width and
                    0 <= paste_y and
                    paste_y + tmp.height <= canvas.height):
                canvas.paste(tmp, (paste_x, paste_y), tmp)

        # Fix 7: Clamp advance multiplier to [0.7, 1.4]
        spacing_noise = float(np.clip(
            self._np_rng.normal(1.0, self._bc.spacing_jitter),
            0.7, 1.4
        ))
        advance = round(font.getlength(char) * float(np.clip(scale, 0.88, 1.12)) * spacing_noise)
        self._char_count += 1
        return max(1, advance)

    def render_word(self, canvas: Image.Image, word: str, x: int, y_baseline: int,
                    wander_offsets: Optional[np.ndarray] = None,
                    baseline_offset: float = 0.0,
                    font: Optional[ImageFont.FreeTypeFont] = None) -> int:
        """
        Render a word character-by-character with MDN randomization.
        Each char gets its own mixture sample (independent perturbation).

        Fix 6: wander_offsets is a per-character array. Each char gets its
        own wander value for smooth, continuous baseline drift.
        """
        cursor = x
        if font is None:
            font = self._base_font

        for i, ch in enumerate(word):
            # Fix 6: Per-character wander from the pre-computed array
            char_wander = baseline_offset
            if wander_offsets is not None and i < len(wander_offsets):
                char_wander += float(wander_offsets[i])

            adv = self.render_char(
                canvas, ch, cursor, y_baseline,
                baseline_offset=char_wander,
                font=font, ink=None,
            )
            cursor += adv
        return cursor - x

    def word_spacing_noise(self) -> float:
        """
        Fix 8: PUBLIC method for word spacing jitter.
        Returns a multiplicative noise factor for word spacing.
        Clamped to [0.5, 2.0] so spacing never collapses or explodes.
        """
        return float(np.clip(
            self._np_rng.normal(1.0, self._bc.spacing_jitter),
            0.5, 2.0
        ))

    def _get_jittered_ink(self) -> Tuple[int, int, int, int]:
        """Convenience wrapper used by HeaderRenderer."""
        return self._sample_ink_color()


# ─────────────────────────────────────────────────────────────────────────────
# TEXT LAYOUT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TextLayoutEngine:
    """Wraps text to fit within ruled line boundaries."""

    def __init__(self, cfg: NotebookConfig):
        self.cfg = cfg

    def wrap_text(self, text: str, font: ImageFont.FreeTypeFont,
                  max_width: int) -> List[str]:
        """Word-wrap text to fit within max_width pixels."""
        words = text.split()
        lines, current_line, current_width = [], [], 0.0
        space_w = font.getlength(" ")

        for word in words:
            word_w = font.getlength(word)
            test_width = current_width + word_w + (space_w if current_line else 0)

            if test_width > max_width and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_w
            else:
                current_line.append(word)
                current_width += word_w + (space_w if len(current_line) > 1 else 0)

        if current_line:
            lines.append(" ".join(current_line))
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# HEADER RENDERER
# ─────────────────────────────────────────────────────────────────────────────

class HeaderRenderer:
    """Renders the assignment header block (class, subject, title)."""

    def __init__(self, cfg: NotebookConfig, hw_renderer: HandwritingRenderer):
        self.cfg = cfg
        self.hw = hw_renderer

    def render(self, canvas: Image.Image, header_lines: List[str],
               title: str, start_y: int) -> int:
        """
        Render centered header lines and an underlined title.
        Returns the Y position after the header block.
        """
        font = self.hw._load_font(self.cfg.header_font_size)
        title_font = self.hw._load_font(self.cfg.title_font_size)
        curr_y = start_y
        draw = ImageDraw.Draw(canvas)

        # Center zone
        center_x = (self.cfg.text_start_x + self.cfg.text_max_x) // 2

        # Render header info lines (centered)
        for line in header_lines:
            line_w = round(font.getlength(line))
            x_start = center_x - line_w // 2
            self.hw.render_word(canvas, line, x_start, curr_y, font=font)
            curr_y += self.cfg.header_line_spacing

        # Render title (centered, underlined)
        title_w = round(title_font.getlength(title))
        title_x = center_x - title_w // 2
        title_y = curr_y
        self.hw.render_word(canvas, title, title_x, title_y, font=title_font)

        # Underline beneath title
        ink = self.hw._get_jittered_ink()
        underline_y = title_y + 8
        # Fix 9: Use _np_rng instead of _rng for wobble
        points = []
        for px in range(title_x - 5, title_x + title_w + 5, 6):
            wobble = int(self.hw._np_rng.integers(-1, 2))  # [-1, 0, 1]
            points.append((px, underline_y + wobble))
        if len(points) >= 2:
            draw.line(points, fill=ink, width=2)

        curr_y += self.cfg.header_line_spacing + 20
        return curr_y


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: snap to ruled line
# ─────────────────────────────────────────────────────────────────────────────

def _snap_to_ruled_line(y: int, cfg: NotebookConfig) -> int:
    """Snap a y-coordinate to the next ruled line at or after y."""
    if y <= cfg.first_line_y:
        return cfg.first_line_y
    lines_past = (y - cfg.first_line_y) / cfg.line_spacing
    next_idx = math.ceil(lines_past)
    return cfg.first_line_y + next_idx * cfg.line_spacing


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def render_notebook_page(
    body_text: str,
    output_path: str = "output/phase2/rendered_page.png",
    title: str = "My Assignment",
    header_lines: Optional[List[str]] = None,
    seed: int = 42,
    config: Optional[NotebookConfig] = None,
) -> str:
    """
    Render text as handwriting on ruled notebook paper.

    Parameters
    ----------
    body_text : str
        The main body text to render as handwriting.
    output_path : str
        Where to save the output PNG image.
    title : str
        The title of the assignment (rendered centered with underline).
    header_lines : list[str], optional
        Lines to render above the title (e.g. class, subject).
        If None, only the title is rendered.
    seed : int
        Random seed for reproducible jitter.
    config : NotebookConfig, optional
        Custom configuration. Uses defaults if None.

    Returns
    -------
    str : Path to the saved output image.
    """
    cfg = config or NotebookConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Generate paper background
    paper_gen = PaperGenerator(cfg)
    canvas = paper_gen.generate()

    # 2. Initialize rendering engines
    hw = HandwritingRenderer(cfg, seed=seed)
    layout = TextLayoutEngine(cfg)
    header_render = HeaderRenderer(cfg, hw)

    # 3. Render header
    if header_lines is None:
        header_lines = []

    curr_y = cfg.first_line_y
    if header_lines or title:
        curr_y = header_render.render(canvas, header_lines, title, curr_y)

    # Snap to next ruled line after header
    curr_y = _snap_to_ruled_line(curr_y, cfg)
    # Add one extra line gap below header
    curr_y += cfg.line_spacing

    # 4. Render body text
    font = hw._load_font(cfg.body_font_size)
    max_text_width = cfg.text_max_x - cfg.text_start_x

    # Split into paragraphs
    paragraphs = [p.strip() for p in body_text.split("\n") if p.strip()]

    for para_idx, paragraph in enumerate(paragraphs):
        wrapped = layout.wrap_text(paragraph, font, max_text_width)

        for line_idx, line_text in enumerate(wrapped):
            if curr_y > cfg.page_h - 200:
                break

            # Indent first line of each paragraph
            indent = 80 if line_idx == 0 else 0
            cursor_x = cfg.text_start_x + indent

            # Fix 6: Pre-compute per-CHARACTER baseline wander for entire line
            words = line_text.split()
            total_chars = sum(len(w) for w in words) + len(words)
            line_wander = hw.get_line_wander(max(1, total_chars))
            char_idx = 0

            for word in words:
                # Check if word fits before rendering (measure first)
                word_w = font.getlength(word)
                if cursor_x + word_w > cfg.text_max_x:
                    break

                # Fix 6: Slice wander array for THIS word's characters
                w_start = char_idx
                w_end = min(char_idx + len(word), len(line_wander))
                word_wander = line_wander[w_start:w_end]

                adv = hw.render_word(
                    canvas, word, cursor_x, curr_y,
                    wander_offsets=word_wander,
                )
                char_idx += len(word) + 1  # +1 for the space

                # Fix 8: Use public method for word spacing noise
                space = round(cfg.word_spacing_base * hw.word_spacing_noise())
                cursor_x += adv + space

            # Move to next ruled line
            curr_y += cfg.line_spacing

        # Fix 4: Paragraph gap — skip a full line, then snap to ruled grid
        curr_y += cfg.line_spacing
        curr_y = _snap_to_ruled_line(curr_y, cfg)

    # 5. Save output (convert RGBA → RGB for PNG)
    final = canvas.convert("RGB")
    final.save(str(output_path), "PNG")
    print(f"  [OK] Notebook page rendered: {output_path}")
    return str(output_path)


def render_notebook_multi_page(
    body_text: str,
    output_dir: str = "assignments",
    title: str = "My Assignment",
    header_lines: Optional[List[str]] = None,
    seed: int = 42,
    config: Optional[NotebookConfig] = None,
) -> List[str]:
    """
    Multi-page renderer: splits long text across pages.
    Returns a list of output file paths.
    """
    cfg = config or NotebookConfig()

    # Fix 10: Load font directly — no throwaway HandwritingRenderer
    font_dir = Path(cfg.font_dir)
    font_path = font_dir / cfg.primary_font
    if font_path.exists():
        font = ImageFont.truetype(str(font_path), cfg.body_font_size)
    else:
        font = ImageFont.load_default()

    # Estimate capacity: roughly how many words fit per page
    usable_lines = (cfg.page_h - cfg.first_line_y - 200) // cfg.line_spacing
    avg_word_w = font.getlength("average ")
    words_per_line = int((cfg.text_max_x - cfg.text_start_x) / avg_word_w)
    words_per_page = usable_lines * words_per_line

    words = body_text.split()
    pages_text = []
    for i in range(0, len(words), words_per_page):
        chunk = " ".join(words[i:i + words_per_page])
        pages_text.append(chunk)

    if not pages_text:
        pages_text = [body_text]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for page_num, page_text in enumerate(pages_text):
        out_path = output_dir / f"page_{page_num + 1:02d}.png"
        page_title = title if page_num == 0 else f"{title} (cont.)"
        page_headers = header_lines if page_num == 0 else None

        render_notebook_page(
            body_text=page_text,
            output_path=str(out_path),
            title=page_title,
            header_lines=page_headers,
            seed=seed + page_num * 7,
            config=cfg,
        )
        results.append(str(out_path))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_HEADER = [
        "2nd Week Assignment",
        "Class - 06",
        "Study Express",
        "Sub: English",
    ]
    TEST_TITLE = "My First Day at School"
    TEST_BODY = (
        "My first day at school is one of the most memorable days of my life. "
        "The day was Sunday 1 January 2012. I went to a nearby Primary School "
        "with my father. I had many unknown fears. After reaching school, I saw "
        "some student were playing in the field. Then we went to the Headmaster's "
        "office. There we met some teachers. A teacher took us to my classroom. "
        "When my father left me in the class I understood that I was in a new "
        "and unknown world. Soon all of the students joined a assembly. Then we "
        "returned to the class. After a class my teacher took entered the class. "
        "He told us about many rules and important things."
    )

    result = render_notebook_page(
        body_text=TEST_BODY,
        output_path="output/test_notebook.png",
        title=TEST_TITLE,
        header_lines=TEST_HEADER,
        seed=42,
    )
    print(f"Test render complete: {result}")
