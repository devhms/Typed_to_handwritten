"""
fiber_generator.py

Generates a procedural fiber map based on correlated noise (Simplex/Perlin approximation)
for the Thixotropic PBI Engine to simulate microscopic paper grain for ink skipping.
"""

import cv2
import numpy as np
from pathlib import Path

def generate_fiber_map(width: int, height: int, scale: float = 4.0) -> np.ndarray:
    """
    Simulates organic paper fibrous texture using multi-octave pulp-clump logic.
    """
    print("Generating Organic Pulp Fiber Map...")
    rng = np.random.default_rng(42)
    
    # 1. Base High-Frequency Fiber Noise
    noise = rng.standard_normal((height, width)).astype(np.float32)
    fiber_map_h = cv2.GaussianBlur(noise, (15, 3), 0)
    fiber_map_v = cv2.GaussianBlur(noise, (3, 15), 0)
    fmap = (fiber_map_h + fiber_map_v)
    
    # 2. Add Low-Frequency "Pulp Clumping" (Perlin-ish)
    pulp_noise = np.zeros_like(fmap)
    for s in [20, 100, 300]:
        # Generate small noise grid and scale it up to create clumps
        small = rng.standard_normal((height // s + 1, width // s + 1)).astype(np.float32)
        clumps = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        pulp_noise += clumps * (5.0 / s)
    
    # Combine and normalize
    fiber_map = (fmap * scale) + (pulp_noise * 1.5)
    fiber_map_norm = cv2.normalize(fiber_map, None, 0, 255, cv2.NORM_MINMAX)
    return fiber_map_norm.astype(np.uint8)

if __name__ == "__main__":
    # Test generation
    fmap = generate_fiber_map(1000, 1000)
    out_path = Path("assets/generated_fiber_map.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), fmap)
    print(f"Test fiber map saved to {out_path} (Mean density: {np.mean(fmap):.1f})")
