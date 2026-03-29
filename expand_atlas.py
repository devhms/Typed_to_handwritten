"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    ALLOGRAPHIC ATLAS EXPANDER v9.0                                           ║
║    Objective: Procedural Generation of Morphological Variants                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import numpy as np
from pathlib import Path

def mutate_skeleton(points, rng):
    pts = np.array(points)
    # 1. Random Scaling (Individual glyph size variation)
    scale_x = rng.uniform(0.92, 1.08)
    scale_y = rng.uniform(0.92, 1.08)
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    
    # 2. Random Slant (Italicization)
    slant = rng.uniform(-0.15, 0.15)
    pts[:, 0] += pts[:, 1] * slant
    
    # 3. Micro-Warping (Local point noise)
    noise = rng.normal(0, 0.02, pts.shape)
    pts += noise
    
    return pts.tolist()

def expand_atlas(input_path, output_path, variants_per_char=5, seed=999):
    with open(input_path, 'r') as f:
        base_atlas = json.load(f)
    
    rng = np.random.default_rng(seed)
    expanded_atlas = {}
    
    for char, points in base_atlas.items():
        variants = []
        # Add the original as the first variant
        variants.append(points)
        # Generate N-1 mutated variants
        for _ in range(variants_per_char - 1):
            variants.append(mutate_skeleton(points, rng))
        
        expanded_atlas[char] = variants

    with open(output_path, 'w') as f:
        json.dump(expanded_atlas, f, indent=2)
    
    print(f"Expanded Atlas created with {len(expanded_atlas)} characters and {variants_per_char} variants each.")

if __name__ == "__main__":
    src = Path("assets/sovereign_atlas.json")
    dst = Path("assets/sovereign_atlas_v9.json")
    expand_atlas(src, dst)
