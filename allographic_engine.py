"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    ALLOGRAPHIC SELECTION ENGINE v9.0                                         ║
║    Objective: Elimination of Character Uniformity                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import numpy as np
from pathlib import Path

class AllographicEngine:
    def __init__(self, atlas_path="assets/sovereign_atlas_v9.json", seed=101):
        with open(atlas_path, 'r') as f:
            self.atlas = json.load(f)
        self.rng = np.random.default_rng(seed)
        
    def get_variant(self, char):
        """Randomly picks an allograph and applies micro-mutation."""
        variants = self.atlas.get(char, self.atlas.get('?', []))
        if not variants:
            return [[0.5, 0.5]]
        
        # 1. Stochastic Selection (Robust to variable-length skeletons)
        idx = self.rng.integers(0, len(variants))
        base_skel = variants[idx]
        pts = np.array(base_skel, dtype=np.float32)
        
        # 2. Dynamic Morphing (Global)
        morph_scale = self.rng.uniform(0.97, 1.03, size=(2,))
        pts *= morph_scale
        
        # 3. Non-Linear Point Warping (Local "Muscle" Noise)
        # Prevents perfectly straight digital lines
        noise = self.rng.normal(0, 0.015, pts.shape)
        pts += noise
        
        # 4. Perspective Shear (Micro-drift)
        shear = self.rng.uniform(-0.04, 0.04)
        pts[:, 0] += pts[:, 1] * shear
        
        return pts.tolist()

class StyleDriftState:
    def __init__(self, seed=202):
        self.rng = np.random.default_rng(seed)
        self.slant = self.rng.uniform(-0.05, 0.05)
        self.scale = 1.0
        self.y_offset = 0.0
        
    def update(self):
        """Random walk for style drift."""
        self.slant += self.rng.normal(0, 0.005)
        self.scale += self.rng.normal(0, 0.002)
        self.y_offset += self.rng.normal(0, 0.3)
        
        # Clamp to realistic ranges
        self.slant = np.clip(self.slant, -0.2, 0.2)
        self.scale = np.clip(self.scale, 0.85, 1.15)
        self.y_offset = np.clip(self.y_offset, -10, 10)
