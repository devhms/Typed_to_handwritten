"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    THIXOTROPIC PBI RENDERER v7.0 (The Fluid)                                ║
║    "Ink is a Non-Newtonian Fluid" — Viscosity-Velocity Interaction          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CORE PRINCIPLES:                                                            ║
║  1. SHEAR THINNING: High velocity = lower viscosity = thinner line.          ║
║  2. INK POOLING: Low velocity turns/stops = higher ink saturation.           ║
║  3. CAPILLARY DRAG: Ink 'hangs' on the ball longer at high speeds.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from bio_kinematic_engine import BioKinematicEngine
from typing import Dict

class ThixotropicPBI:
    def __init__(self, params: Dict = None):
        self.params = {
            "viscosity_base": 0.5,
            "shear_sensitivity": 1.2,
            "depletion_rate": 0.00015,
            "refill_rate": 0.001,
            "base_thickness": 4.2,
            "base_alpha": 210
        }
        if params: self.params.update(params)
        self.ink_on_ball = 1.0

    def render_bio_stroke(self, stroke_v, canvas, char_size, fiber_map=None, fiber_sens=0.25):
        """
        stroke_v: Array of (x, y, v)
        """
        for i in range(len(stroke_v)):
            # 4nd channel is the Pre-calculated Bio-Pressure (v10.0)
            x, y, v, pressure_bio = stroke_v[i]
            px, py = int(x), int(y)
            
            if 0 <= py < canvas.shape[0] and 0 <= px < canvas.shape[1]:
                # 1. Thixotropic Viscosity Calculation
                viscosity = self.params["viscosity_base"] / (1.0 + v * self.params["shear_sensitivity"])
                
                # 2. Fiber Interaction
                fiber_val = 1.0
                if fiber_map is not None:
                    fh, fw = fiber_map.shape[:2]
                    fiber_val = fiber_map[py % fh, px % fw] / 255.0
                
                # 3. Dynamic Deposition using BIO pressure
                # We use the pressure_bio which includes curvature + velocity dynamics
                pressure = pressure_bio * self.params.get("pressure_bias", 1.0)
                
                skip_prob = (fiber_val - 0.78) * fiber_sens + (1.0 - self.ink_on_ball) * 0.15
                if np.random.random() > max(0, skip_prob):
                    # Ink alpha controlled by viscosity and bio-pressure
                    alpha_mod = pressure * (1.0 + viscosity)
                    clamped_alpha = int(min(255, self.params["base_alpha"] * self.ink_on_ball * alpha_mod * (1.25 - fiber_val)))
                    
                    # Thickness controlled by pressure and viscosity
                    thickness = max(1, int(self.params["base_thickness"] * pressure * 0.6))
                    
                    # Royal Blue Ink Profile
                    ink_color = (12, 42, 195, clamped_alpha)
                    cv2.circle(canvas, (px, py), thickness, ink_color, -1, cv2.LINE_AA)
                    
                    # Depletion logic
                    self.ink_on_ball = max(0.3, self.ink_on_ball - self.params["depletion_rate"])
                else:
                    self.ink_on_ball = min(1.0, self.ink_on_ball + self.params["refill_rate"])
                    
    def compose_masterpiece(self, render_func, text, canvas, start_pos, char_size=80, fiber_map=None):
        """Orchestrates Bio Engine + Thixotropic Render."""
        engine = BioKinematicEngine()
        x, y = start_pos
        
        for char in text:
            if char == ' ':
                x += char_size * 0.6
                continue
            
            # 1. Bio-Kinematic Path Generation
            # We get points from the atlas (using fallback if needed)
            from phase5_forensic_synthesis import get_sovereign_atlas
            atlas = get_sovereign_atlas()
            raw_pts = atlas.get(char, atlas.get('?', [[0.5, 0.5]]))
            
            stroke_3d = engine.generate_human_stroke(raw_pts)
            # Scale and Offset
            stroke_3d[:, 0] = stroke_3d[:, 0] * char_size + x
            stroke_3d[:, 1] = stroke_3d[:, 1] * char_size + y
            
            # 2. Render
            self.render_bio_stroke(stroke_3d, canvas, char_size, fiber_map)
            
            x += char_size * 0.82 + np.random.uniform(-2, 4)
