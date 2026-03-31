"""
handwriting_randomizer.py

Complete extraction of every randomization mechanism from:
  https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit

Translated from PyTorch to pure Python + NumPy + PIL.
No PyTorch required. Drop this file next to your notebook_renderer.py.

SOURCES (exact file → line):
  models.py          → MixtureDensityLayer.forward(), get_mean_prediction(),
                        sample_from_bivariate_mixture()
  onnx_models.py     → MixtureDensityLayer.forward() [bias applied inside forward]
  sampling.py        → HandwritingSynthesizer.generate_handwriting(), bias param
  utils.py           → create_strokes_png(), get_strokes(), merge_images(),
                        split_into_lines(), draw_points()
  data.py            → to_offsets(), to_absolute_coordinates(), NormalizedDataset
"""

import math
import random
from typing import Optional, Tuple, List, Dict # UI-UX PRO MAX: Advanced Forensic Typing (v8.5)
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageOps


# ─────────────────────────────────────────────────────────────────────────────
# MECHANISM 1 — THE BIAS PARAMETER
# ─────────────────────────────────────────────────────────────────────────────

def apply_bias_to_variance(sd_base: float, bias: float) -> float:
    """
    Direct port of: sd = torch.exp(sd_hat - self.bias)
    
    When bias=0: sd = exp(sd_hat)          → natural variance
    When bias=1: sd = exp(sd_hat - 1)      → ~0.37x natural variance (much tighter)
    When bias=-1: sd = exp(sd_hat + 1)     → ~2.72x natural variance (much looser)
    """
    return sd_base * math.exp(-bias)


def apply_bias_to_weights(weights: list, bias: float) -> list:
    """
    Direct port of: pi = softmax(pi_hat * (1 + bias))
    """
    sharpened = [w * (1.0 + bias) for w in weights]
    max_w = max(sharpened)
    exps = [math.exp(w - max_w) for w in sharpened]
    total = sum(exps)
    return [e / total for e in exps]


class BiasController:
    """
    Implements the full bias system from sampling.py.
    
    Usage:
        neat   = BiasController(bias=1.0)   # exam-quality handwriting
        normal = BiasController(bias=0.0)   # default student writing
        messy  = BiasController(bias=-0.5)  # rushed/tired writing
    """
    def __init__(self, bias: float = 0.0):
        self.bias = bias

    def scale_sd(self, sd_base: float) -> float:
        return apply_bias_to_variance(sd_base, self.bias)

    def sharpen_weights(self, weights: list) -> list:
        return apply_bias_to_weights(weights, self.bias)

    @property
    def pos_jitter_x(self) -> float:
        return self.scale_sd(2.0)

    @property
    def pos_jitter_y(self) -> float:
        return self.scale_sd(1.3)

    @property
    def rotation_jitter(self) -> float:
        return self.scale_sd(1.8)

    @property
    def scale_jitter(self) -> float:
        return self.scale_sd(0.04)

    @property
    def spacing_jitter(self) -> float:
        return self.scale_sd(0.10)


# ─────────────────────────────────────────────────────────────────────────────
# MECHANISM 2 — BIVARIATE GAUSSIAN SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def sample_bivariate_gaussian(
    mu1: float = 0.0,
    mu2: float = 0.0,
    sd1: float = 2.0,
    sd2: float = 1.3,
    ro: float = 0.3,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Exact port of models.py sample_from_bivariate_mixture().
    Samples one (dx, dy) offset from a 2D correlated Gaussian.
    """
    cov_xy = ro * sd1 * sd2
    cov_matrix = np.array([
        [sd1 ** 2, cov_xy],
        [cov_xy,   sd2 ** 2],
    ])
    mean = np.array([mu1, mu2])
    if rng is not None:
        dx, dy = rng.multivariate_normal(mean, cov_matrix)
    else:
        dx, dy = np.random.multivariate_normal(mean, cov_matrix)
    return float(dx), float(dy)


# ─────────────────────────────────────────────────────────────────────────────
# MECHANISM 3 — MIXTURE COMPONENT SELECTION
# ─────────────────────────────────────────────────────────────────────────────

class MixtureSampler:
    """
    Implements the full mixture density sampling logic from get_mean_prediction().
    Multiple writing "modes" compete — one is randomly chosen per character.
    """

    def __init__(self, components: list, base_weights: list = None,
                 bias: float = 0.0, rng: np.random.Generator = None):
        self.components = components
        n = len(components)
        self.base_weights = base_weights or [1.0 / n] * n
        self.bias = bias
        self._prev_component = 0
        self._rng = rng or np.random.default_rng()

    def sample(self) -> dict:
        weights = apply_bias_to_weights(self.base_weights, self.bias)

        if self._prev_component != 0:
            weights[self._prev_component] *= 0.15
            total = sum(weights)
            weights = [w / total for w in weights]

        idx = int(self._rng.choice(len(self.components), p=weights))
        self._prev_component = idx
        comp = self.components[idx]

        dx, dy = sample_bivariate_gaussian(
            sd1=comp.get('sd_x', 1.5),
            sd2=comp.get('sd_y', 1.0),
            ro=comp.get('ro', 0.25),
            rng=self._rng,
        )

        rotation = self._rng.normal(
            comp.get('slant', 0.0),
            comp.get('rotation_sd', 1.5),
        )

        scale = self._rng.normal(
            comp.get('scale_mean', 1.0),
            comp.get('scale_sd', 0.03),
        )

        return {
            'dx': dx,
            'dy': dy,
            'rotation': float(rotation),
            'scale': float(np.clip(scale, 0.88, 1.12)),
            'component_idx': idx,
        }


def make_default_mixture(bias: float = 0.0, rng=None) -> MixtureSampler:
    """
    Ready-to-use MixtureSampler with 3 components:
      - Component 0: standard neat strokes (60% weight)
      - Component 1: slightly rushed strokes (30% weight)
      - Component 2: careful deliberate strokes (10% weight)
    """
    bc = BiasController(bias)
    components = [
        {   # Component 0: standard — most common writing mode
            'sd_x': bc.scale_sd(1.0),
            'sd_y': bc.scale_sd(0.7),
            'ro': 0.28,
            'slant': 0.0,
            'rotation_sd': bc.scale_sd(0.9),
            'scale_mean': 1.00,
            'scale_sd': bc.scale_sd(0.025),
        },
        {   # Component 1: rushed — slightly larger offsets
            'sd_x': bc.scale_sd(1.6),
            'sd_y': bc.scale_sd(1.1),
            'ro': 0.38,
            'slant': -0.5,
            'rotation_sd': bc.scale_sd(1.4),
            'scale_mean': 0.98,
            'scale_sd': bc.scale_sd(0.035),
        },
        {   # Component 2: careful — very tight, deliberate
            'sd_x': bc.scale_sd(0.5),
            'sd_y': bc.scale_sd(0.3),
            'ro': 0.12,
            'slant': 0.3,
            'rotation_sd': bc.scale_sd(0.5),
            'scale_mean': 1.01,
            'scale_sd': bc.scale_sd(0.012),
        },
    ]
    weights = [0.65, 0.25, 0.10]
    return MixtureSampler(components, weights, bias=bias, rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
# MECHANISM 7 — BASELINE WANDER (correlated random walk)
# ─────────────────────────────────────────────────────────────────────────────

def generate_baseline_wander(
    n_chars: int,
    sd_step: float = 0.4,
    ro: float = 0.7,
    bias: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simulates baseline wander as a Mean-Reverting (Ornstein-Uhlenbeck) process.
    Unlike a pure random walk (cumulative sum), this prevents infinite drift
    by "pulling" the baseline back to y=0 at each step.
    """
    if rng is None:
        rng = np.random.default_rng()
    sd_step = apply_bias_to_variance(sd_step, bias)

    # Mean-reversion strength (0.1 means 10% pull per char)
    reversion_strength = 0.15
    
    wander = np.zeros(n_chars)
    current = 0.0
    for i in range(n_chars):
        # Pull back towards center
        current -= reversion_strength * current
        # Add correlated noise
        noise = rng.normal(0.0, sd_step)
        current += noise
        wander[i] = current

    return wander


def generate_line_baseline_wander(
    n_chars: int,
    font_size_px: int = 72,
    bias: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Produces the final y-offset array for one line of text.
    """
    sd_step = font_size_px * 0.04  # Scaling up for the mean-reversion pull
    return generate_baseline_wander(n_chars, sd_step=sd_step, ro=0.72,
                                     bias=bias, rng=rng)


def apply_elastic_warp(image: Image.Image, magnitude: float = 0.02, rng=None) -> Image.Image:
    """
    Mechanism 10: Stochastic Mesh Warping.
    Applies a low-frequency, random displacement field to the glyph.
    """
    if magnitude < 0.001: return image
    if rng is None: rng = np.random.default_rng()
    
    img_arr = np.array(image)
    h, w = img_arr.shape[:2]
    
    grid_size = 4
    dx_grid = rng.normal(0, w * magnitude, (grid_size, grid_size))
    dy_grid = rng.normal(0, h * magnitude, (grid_size, grid_size))
    
    dx_map = cv2.resize(dx_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    dy_map = cv2.resize(dy_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx_map).astype(np.float32)
    map_y = (y + dy_map).astype(np.float32)
    
    warped = cv2.remap(img_arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Image.fromarray(warped)


# ─────────────────────────────────────────────────────────────────────────────
# MECHANISM 11 — PER-CHARACTER AFFINE PERTURBATION (Revised v8.1)
# ─────────────────────────────────────────────────────────────────────────────

def perturb_glyph_mask(
    glyph: Image.Image,
    dx: float,
    dy: float,
    rotation: float,
    scale: float,
    pivot_x: Optional[float] = None,
    pivot_y: Optional[float] = None,
    variation_magnitude: float = 0.0,
    rng=None,
) -> tuple:
    """
    Mechanism 11 (v8.5): High-Precision Geometric Grounding.
    Applies per-character affine perturbations while locking the pivot (baseline) point.
    """
    w, h = glyph.size # Extract current pixel dimensions of the glyph mask

    # 1. Biological Elastic Warping: Simulate non-linear motor tremors (bio-kinematics)
    if variation_magnitude > 0:
        glyph = apply_elastic_warp(glyph, magnitude=variation_magnitude, rng=rng)

    # 2. Euclidean Scaling: Simulate variable handwriting size/pressure
    if abs(scale - 1.0) > 0.005: 
        new_w = max(1, int(w * scale)) # Calculate new width constraint
        new_h = max(1, int(h * scale)) # Calculate new height constraint
        glyph = glyph.resize((new_w, new_h), Image.LANCZOS) # Perform high-fidelity resampling
        
        # Proportional Pivot Update: Ensure the baseline anchor scales with the glyph
        if pivot_x is not None: pivot_x *= scale 
        if pivot_y is not None: pivot_y *= scale
        w, h = glyph.size # Update working dimensions post-scale

    # 3. Precision Rotation: Min-Corner Mapping (v8.6 Forensic Standard)
    new_pivot_x, new_pivot_y = pivot_x, pivot_y # Baseline: Pivot is fixed if no rotation occurs
    if abs(rotation) > 0.05:
        # Determine the Pivot Point (P) around which rotation occurs
        px = pivot_x if pivot_x is not None else w / 2.0 
        py = pivot_y if pivot_y is not None else h - 2.0 
        
        # 3.1 Trace the Bounding Box Shift: Calculate the rotated footprint relative to the pivot
        rad = math.radians(rotation) 
        cos_t, sin_t = math.cos(rad), math.sin(rad) 
        
        # Corner Verification: Trace all four vertices of the original chip
        corners = [(0,0), (w,0), (w,h), (0,h)]
        rot_corners = []
        for cx_c, cy_c in corners:
            # Vector from pivot to corner
            vx, vy = cx_c - px, cy_c - py
            # Rotated Vector (V')
            rx = vx * cos_t - vy * sin_t
            ry = vx * sin_t + vy * cos_t
            rot_corners.append((rx, ry))
        
        # Min-Corner Extraction: Find the new top-left relative to the pivot
        min_x = min(c[0] for c in rot_corners)
        min_y = min(c[1] for c in rot_corners)
        
        # 3.2 Affine Synthesis: Perform the actual bit-mask rotation
        glyph = glyph.rotate(-rotation, resample=Image.BICUBIC, center=(px, py), expand=True)
        
        # 3.3 Target Pivot Projection: Calculate exact pivot location in the NEW canvas
        new_pivot_x = -min_x
        new_pivot_y = -min_y
        
    # Final Telemetry: Return the transformed chip and its absolute internal pivot position
    return glyph, new_pivot_x + dx, new_pivot_y + dy
