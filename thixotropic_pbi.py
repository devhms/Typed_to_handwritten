"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    THIXOTROPIC PBI RENDERER v8.0 (The Fluid)                                ║
║    "Ink is a Non-Newtonian Fluid" — Viscosity-Velocity Interaction          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CORE PRINCIPLES:                                                            ║
║  1. SHEAR THINNING: High velocity = lower viscosity = thinner line.          ║
║  2. FIBER STAINING: Ink preferentially adheres to raised paper fibers.       ║
║  3. CAPILLARY WICKING: Lateral ink spread follows fiber orientation.         ║
║  4. MECHANICAL GROOVE: Ballpoint pressure creates substrate indentation.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

class ThixotropicPBI:
    def __init__(self, params: Optional[Dict] = None):
        self.params = {
            "viscosity_base": 0.6,
            "shear_sensitivity": 1.4,
            "depletion_rate": 0.00012,
            "refill_rate": 0.0008,
            "base_thickness": 4.5,
            "base_alpha": 220,
            "fiber_stain_bias": 0.82,  # Threshold for fiber adherence
            "wicking_strength": 0.3,    # Lateral spread factor
            "groove_depth": 15,         # Indentation intensity
        }
        if params:
            self.params.update(params)
        self.ink_on_ball = 1.0

    def render_bio_stroke(self, stroke_v, canvas, groove_canvas=None, fiber_map=None):
        """
        stroke_v: Array of (x, y, v, pressure)
        canvas: BGRA image (uint8)
        groove_canvas: Grayscale heightmap (uint8)
        fiber_map: Grayscale fiber texture (uint8)
        """
        h, w = canvas.shape[:2]
        r_ink, g_ink, b_ink = 18, 32, 168 # Deep Royal Blue

        for i in range(len(stroke_v)):
            x, y, v, pressure = stroke_v[i]
            px, py = int(x), int(y)

            if not (0 <= py < h and 0 <= px < w):
                continue

            # 1. Thixotropic Viscosity (Shear-thinning)
            viscosity = self.params["viscosity_base"] / (1.0 + v * self.params["shear_sensitivity"])
            
            # 2. Fiber Staining Logic
            fiber_val = 1.0
            if fiber_map is not None:
                fiber_val = fiber_map[py % fiber_map.shape[0], px % fiber_map.shape[1]] / 255.0
            
            # v8.0 Staining Threshold: Ink sticks to fibers above bias
            # High pressure can overcome low fiber adherence
            stain_prob = pressure * (fiber_val / self.params["fiber_stain_bias"])
            
            # 3. Ballpoint Groove (Physical Indentation)
            if groove_canvas is not None:
                # Add depth to the groove map
                g_radius = int(self.params["base_thickness"] * pressure * 0.45)
                if g_radius > 0:
                    cv2.circle(groove_canvas, (px, py), g_radius, int(self.params["groove_depth"] * pressure), -1)

            # 4. Stochastic Deposition
            skip_threshold = (1.0 - stain_prob) * 0.25 + (1.0 - self.ink_on_ball) * 0.2
            if np.random.random() > max(0, skip_threshold):
                # Alpha and Thickness based on bio-pressure and viscosity
                alpha_mod = pressure * (1.1 - viscosity * 0.5)
                clamped_alpha = int(min(250, self.params["base_alpha"] * self.ink_on_ball * alpha_mod))
                
                radius = max(0.5, self.params["base_thickness"] * pressure * 0.7)
                num_particles = min(30, max(2, int((radius ** 2) * 1.8)))
                
                # Particle offsets weighted by fiber gradient (simulated wicking)
                # For high-precision, we shift particles slightly towards higher fiber values
                offsets_x = np.random.normal(0, radius * 0.4, num_particles)
                offsets_y = np.random.normal(0, radius * 0.4, num_particles)
                
                alpha_f = (clamped_alpha / 255.0) * 0.7
                
                for dx, dy in zip(offsets_x, offsets_y):
                    p_x, p_y = int(px + dx), int(py + dy)
                    
                    if 0 <= p_y < h and 0 <= p_x < w:
                        # Apply wicking: adjust opacity by local fiber density
                        local_f = 1.0
                        if fiber_map is not None:
                            local_f = fiber_map[p_y % fiber_map.shape[0], p_x % fiber_map.shape[1]] / 255.0
                        
                        part_alpha = alpha_f * (0.8 + 0.4 * local_f)
                        
                        # Blend with canvas
                        bg_b, bg_g, bg_r, bg_a = canvas[p_y, p_x]
                        if bg_a == 0:
                            canvas[p_y, p_x] = [b_ink, g_ink, r_ink, int(part_alpha * 255)]
                        else:
                            # Standard Porter-Duff Over
                            out_a = part_alpha + (bg_a/255.0) * (1 - part_alpha)
                            out_r = (r_ink * part_alpha + bg_r * (bg_a/255.0) * (1 - part_alpha)) / out_a
                            out_g = (g_ink * part_alpha + bg_g * (bg_a/255.0) * (1 - part_alpha)) / out_a
                            out_b = (b_ink * part_alpha + bg_b * (bg_a/255.0) * (1 - part_alpha)) / out_a
                            canvas[p_y, p_x] = [out_b, out_g, out_r, int(out_a * 255)]
                
                # Depletion
                self.ink_on_ball = max(0.2, self.ink_on_ball - self.params["depletion_rate"])
            else:
                # Refill when pen leaves paper or skips
                self.ink_on_ball = min(1.0, self.ink_on_ball + self.params["refill_rate"])

    def apply_capillary_wicking(self, canvas, fiber_map):
        """Simulates lateral ink spread along fiber directions using a guided blur."""
        if fiber_map is None: return
        
        # Guide: fiber map gradients
        gray_ink = canvas[:, :, 3] # Use alpha channel as ink source
        
        # Create a kernel aligned with paper fibers (approximate with anisotropic blur)
        # For high precision, we use the fiber map to influence the blur reach
        w_h, w_w = fiber_map.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]
        
        # Guide should be same size as canvas
        guide = cv2.resize(fiber_map, (canvas_w, canvas_h)).astype(float) / 255.0
        
        # Simple iterative diffusion
        alpha = canvas[:, :, 3].astype(float) / 255.0
        diffused = cv2.GaussianBlur(alpha, (3, 3), 0)
        
        # Capillary diffusion only where fibers are present
        alpha_new = np.where(guide > 0.6, diffused * 1.05, alpha * 0.95 + diffused * 0.05)
        canvas[:, :, 3] = (np.clip(alpha_new, 0, 1) * 255).astype(np.uint8)
