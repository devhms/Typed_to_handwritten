import numpy as np
import cv2
from scipy.interpolate import CubicSpline

def render_hershey_sovereign(text, scale=1.2, jitter=0.08):
    # Hershey Simplex is a good base
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Create a large canvas for high-res paths
    canvas = np.ones((800, 2400, 3), dtype=np.uint8) * 255
    
    # We'll use a trick: draw each char, find its stroke, and jitter it.
    # Actually, Hershey fonts are hardcoded in OpenCV. Let's use a simpler way:
    # Just use the coordinates if we had them. Since we don't have the raw Hershey coords easily,
    # I'll use the 'a-e' atlas I built previously but expanded, OR I'll use a 'Skeleton Font' method.
    
    # REAL SOVEREIGN METHOD: Use Font outlines + Skeletonization
    # But for a quick 'WOW' test, I'll use the Bezier Atlas for 'Hello'
    
    atlas = {
        'h': [(0.2, 0.0), (0.2, 1.0), (0.2, 0.5), (0.5, 0.4), (0.7, 0.6), (0.7, 1.0)],
        'e': [(0.2, 0.6), (0.7, 0.6), (0.7, 0.3), (0.4, 0.2), (0.1, 0.5), (0.1, 0.8), (0.4, 1.0), (0.7, 0.9)],
        'l': [(0.4, 0.0), (0.4, 0.9), (0.5, 1.0), (0.7, 1.0)],
        'o': [(0.5, 0.3), (0.2, 0.5), (0.2, 0.8), (0.5, 1.0), (0.8, 0.8), (0.8, 0.5), (0.5, 0.3)],
        ' ': [(0.5, 0.5)]
    }
    
    cursor = 100
    for char in text.lower():
        if char not in atlas: continue
        skeleton = np.array(atlas[char])
        
        # Add 'Wander' to the baseline
        baseline_offset = np.sin(cursor * 0.01) * 10
        
        # Jitter the skeleton
        noise = np.random.normal(0, jitter, skeleton.shape)
        p = (skeleton + noise) * 150 # Scale to 150px
        p[:, 0] += cursor
        p[:, 1] += 300 + baseline_offset
        
        # Spline it
        t = np.linspace(0, 1, len(p))
        if len(p) > 2:
            cs_x = CubicSpline(t, p[:, 0])
            cs_y = CubicSpline(t, p[:, 1])
            fine_t = np.linspace(0, 1, 200)
            points = np.stack([cs_x(fine_t), cs_y(fine_t)], axis=1)
        else:
            points = p
            
        # Draw with 'Ink Flow'
        for i in range(len(points)-1):
            # Speed/Curvature based thickness
            dist = np.linalg.norm(points[i+1] - points[i])
            thickness = max(1, int(3 - dist * 0.5 + np.random.normal(0, 0.5)))
            alpha = int(200 + np.random.randint(0, 55))
            color = (80, 20, 10) # Dark Blue Ink
            
            cv2.line(canvas, tuple(points[i].astype(int)), tuple(points[i+1].astype(int)), color, thickness, cv2.LINE_AA)
            
        cursor += 150
        
    cv2.imwrite("sovereign_hello.png", canvas)

if __name__ == "__main__":
    render_hershey_sovereign("hello")
