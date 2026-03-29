"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    SOVEREIGN FORENSIC ENGINE  v5.1  (Full Potential 2026)                  ║
║    "The Physics of Ink" — Sub-Pixel Fiber-Selective PBR Synthesis           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CORE PRINCIPLES:                                                            ║
║  1. STOCHASTIC PARTICLES: Strokes are probability-density fields of ink.    ║
║  2. FIBER-DEPTH SELECTIVITY: Ink skips over high-pulp 'fibers' in the map.   ║
║  3. BALLPOINT FRICTION: Friction-slip model for depletion & re-inking.      ║
║  4. CAPILLARY FEATHERING: Sub-pixel ink 'crawling' along paper fibers.      ║
║  5. LENS OPTICS: Digital sensor ISO noise and chromatic aberration.         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import json
import math
from pathlib import Path
from scipy.interpolate import CubicSpline
from PIL import Image, ImageDraw, ImageFilter, ImageChops

# ─────────────────────────────────────────────────────────────────────────────
# [1] SOVEREIGN MOTOR-PATH ATLAS
# ─────────────────────────────────────────────────────────────────────────────

def get_sovereign_atlas():
    """Load motor-path skeletons from the global atlas."""
    atlas_path = Path("assets/sovereign_atlas.json")
    if atlas_path.exists():
        with open(atlas_path, "r") as f:
            return json.load(f)
    print("[WARN] sovereign_atlas.json not found, using minimal fallback.")
    return {' ': [[0.5, 1.0]]}

# ─────────────────────────────────────────────────────────────────────────────
# [2] PHYSICALLY-BASED INK (PBI) ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SovereignEngine:
    def __init__(self, seed=None, params=None):
        self.rng = np.random.default_rng(seed)
        self.atlas = get_sovereign_atlas()
        # Default Optimal Params (from v5.1)
        self.params = {
            "jitter": 0.03,
            "slant": 0.12,
            "hand_waviness": 0.015,
            "waviness_freq": 12.0,
            "ink_depletion_rate": 0.00018,
            "ink_refill_rate": 0.0012,
            "fiber_skip_sens": 0.25,
            "thickness_base": 4.5,
            "alpha_base": 215
        }
        if params: self.params.update(params)
        
        # Ballpoint friction state
        self.ink_on_ball = 1.0 

    def generate_motor_stroke(self, char):
        points = self.atlas.get(char)
        if points is None:
            points = self.atlas.get('?', [[0.5, 0.5]])
            
        path = np.array(points, dtype=np.float32)
        
        # Forensic perturbation
        noise = self.rng.normal(0, self.params["jitter"], path.shape)
        p = path + noise
        p[:, 0] += p[:, 1] * self.params["slant"]
        
        # Natural Hand-Waviness (Frequency Noise)
        p[:, 0] += np.sin(p[:, 1] * self.params["waviness_freq"]) * self.params["hand_waviness"]
        
        if len(p) > 2:
            t = np.linspace(0, 1, len(p))
            fine_t = np.linspace(0, 1, 350) 
            cs_x = CubicSpline(t, p[:, 0])
            cs_y = CubicSpline(t, p[:, 1])
            return np.stack([cs_x(fine_t), cs_y(fine_t)], axis=1)
        return p

    def render_pbi_text(self, text, canvas, start_pos, char_size=80, ink_color=(20, 30, 120), fiber_map=None):
        """High-fidelity PBR rendering (Slow)."""
        x_cursor, y_cursor = start_pos
        
        for i, char in enumerate(text):
            if char == ' ':
                x_cursor += char_size * 0.6
                self.ink_on_ball = min(1.0, self.ink_on_ball + 0.35) 
                continue
            
            stroke = self.generate_motor_stroke(char)
            render_points = (stroke * char_size) + np.array([x_cursor, y_cursor])
            
            # PBR Particle Injection logic
            for j in range(len(render_points)-1):
                p1 = render_points[j]
                p2 = render_points[j+1]
                
                dist = np.linalg.norm(p2 - p1)
                num_particles = max(1, int(dist * 2.5))
                
                for k in range(num_particles):
                    interp = k / num_particles
                    pos = p1 * (1 - interp) + p2 * interp
                    py, px = int(pos[1]), int(pos[0])
                    
                    if 0 <= py < canvas.shape[0] and 0 <= px < canvas.shape[1]:
                        fiber_val = 1.0
                        if fiber_map is not None:
                            fh, fw = fiber_map.shape[:2]
                            fiber_val = fiber_map[py % fh, px % fw] / 255.0
                        
                        skip_prob = (fiber_val - 0.78) * self.params["fiber_skip_sens"] + (1.0 - self.ink_on_ball) * 0.12
                        if self.rng.random() > max(0, skip_prob):
                            base_alpha = self.params["alpha_base"] + self.rng.integers(0, 40)
                            # Dynamic Pressure Simulation (Velocity inverse)
                            press_mod = 1.0 / (1.0 + dist * 0.1)
                            
                            bleed_boost = (1.25 - fiber_val) * 1.4
                            clamped_alpha = int(min(255, base_alpha * self.ink_on_ball * bleed_boost * press_mod))
                            
                            ink_vibrant = (12, 42, 195) 
                            
                            thickness = max(2, int(self.params["thickness_base"] - dist * 0.4))
                            c = (*ink_vibrant, clamped_alpha)
                            cv2.circle(canvas, (px, py), thickness, c, -1, cv2.LINE_AA)
                            
                            self.ink_on_ball = max(0.35, self.ink_on_ball - self.params["ink_depletion_rate"])
                        else:
                            self.ink_on_ball = min(1.0, self.ink_on_ball + self.params["ink_refill_rate"])
                
            x_cursor += char_size * 0.82 + self.rng.uniform(-2, 4)
            # Ligatures
            if self.rng.random() < 0.65 and i < len(text)-1:
                p_end = render_points[-1].astype(int)
                cv2.line(canvas, tuple(p_end), (int(x_cursor), int(y_cursor + char_size*0.72)), (*ink_color, 45), 1, cv2.LINE_AA)

    def render_fast_text(self, text, canvas, start_pos, char_size=60, ink_color=(30, 20, 100)):
        """Fast deterministic rendering for live preview."""
        x, y = start_pos
        for i, char in enumerate(text):
            if char == ' ':
                x += char_size * 0.6
                continue
            
            stroke = self.generate_motor_stroke(char)
            render_points = (stroke * char_size) + np.array([x, y])
            
            for j in range(len(render_points)-1):
                p1 = tuple(render_points[j].astype(int))
                p2 = tuple(render_points[j+1].astype(int))
                dist = np.linalg.norm(render_points[j+1] - render_points[j])
                thickness = max(1, int(3 - dist * 0.8))
                alpha = int(180 + self.rng.integers(0, 75))
                cv2.line(canvas, p1, p2, (*ink_color, alpha), thickness, cv2.LINE_AA)
                
            x += char_size * 0.85 + self.rng.uniform(-2, 5)

# ─────────────────────────────────────────────────────────────────────────────
# [3] FORENSIC COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def apply_forensic_post_process(image):
    """Sub-pixel feathering and camera sensor simulation."""
    blur = cv2.GaussianBlur(image, (3, 3), 0.5)
    image = cv2.addWeighted(image, 0.7, blur, 0.3, 0)
    
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    r, g, b = pil_img.split()
    r = ImageChops.offset(r, 1, 0)
    b = ImageChops.offset(b, -1, 1)
    pil_img = Image.merge("RGB", (r, g, b))
    
    arr = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0, 4.5, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def compose_forensic_assignment(title, body, output_path, fast_mode=False, params=None):
    # Load assets
    bg_p = Path("assets/photorealistic_substrate.jpg")
    fb_p = Path("assets/fiber_map.png")
    
    if bg_p.exists():
        bg = cv2.imread(str(bg_p))
    else:
        bg = np.ones((2000, 2000, 3), dtype=np.uint8) * 245
        
    fiber_map = None
    if fb_p.exists():
        fiber_map = cv2.imread(str(fb_p), 0)
    
    h_bg, w_bg = bg.shape[:2]
    canvas_w, canvas_h = 2480, 3508 
    
    ink_layer = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    engine = SovereignEngine(seed=1337, params=params)
    
    margin_left = 320
    line_y_start = 280
    line_spacing = 108 
    
    # Render logic
    lines = [title] + [l.strip() for l in body.split('\n') if l.strip()]
    for i, txt in enumerate(lines):
        curr_y = line_y_start + i * line_spacing
        if curr_y > canvas_h - 300: break
        
        drift_x = np.cos(i * 0.3) * 20
        indent = 250 if (i > 0 and (i-1) % 4 == 0) else 0
        if i == 0: # Title
            engine.render_pbi_text(txt.upper(), ink_layer, (margin_left + 400, curr_y - 30), char_size=110, ink_color=(15, 25, 100), fiber_map=fiber_map)
            continue

        if fast_mode:
            engine.render_fast_text(txt, ink_layer, (margin_left + drift_x + indent, curr_y), char_size=82)
        else:
            engine.render_pbi_text(txt, ink_layer, (margin_left + drift_x + indent, curr_y), char_size=82, fiber_map=fiber_map)

    # Perspective projection
    f_w, f_h = w_bg / 1024.0, h_bg / 1024.0
    src_pts = np.float32([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]])
    dst_pts = np.float32([
        [410 * f_w, 130 * f_h],   # TL
        [930 * f_w, 260 * f_h],   # TR
        [670 * f_w, 910 * f_h],   # BR
        [45 * f_w, 680 * f_h]     # BL
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_ink = cv2.warpPerspective(ink_layer, M, (w_bg, h_bg), flags=cv2.INTER_LANCZOS4)
    
    # Blending
    alpha = warped_ink[:, :, 3] / 255.0
    final_bg = bg.copy().astype(float)
    
    for c in range(3):
        ink_pix = warped_ink[:, :, c].astype(float)
        final_bg[:, :, c] *= (1.0 - alpha * 0.05)
        final_bg[:, :, c] = final_bg[:, :, c] * (1.0 - alpha) + (final_bg[:, :, c] * ink_pix / 255.0) * alpha
        
    result = np.clip(final_bg, 0, 255).astype(np.uint8)
    if not fast_mode:
        result = apply_forensic_post_process(result)
    
    cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 98])
    return result

if __name__ == "__main__":
    TITLE = "SOVEREIGN UPGRADE V5.1"
    BODY = "This is a test of the comprehensive Sovereign Atlas.\nIt includes uppercase, lowercase, numbers 12345, and symbols !?&."
    out_p = Path("assignments/sovereign_v51_test.jpg")
    compose_forensic_assignment(TITLE, BODY, out_p)
    print(f"Test complete: {out_p}")
