from PIL import Image, ImageDraw
import numpy as np
import math

def test_rotation_offset(w, h, angle, cx, cy):
    # Create original
    img = Image.new("L", (w, h), 255)
    # Rotate with expansion
    rotated = img.rotate(-angle, resample=Image.BICUBIC, center=(cx, cy), expand=True)
    nw, nh = rotated.size
    
    # Calculate where the center (cx, cy) moved to in the new image (nw, nh)
    # 1. The expand=True moves the rotation center to the center of the NEW image (nw/2, nh/2)
    # OR DOES IT? Let's check.
    
    print(f"Angle: {angle}, Center: ({cx}, {cy})")
    print(f"Original: ({w}, {h}), Rotated: ({nw}, {nh})")
    
    # Let's find the shift (ox, oy) for the top-left (0,0) of the original relative to the new (0,0)
    # In expand=True, the center of the original image is moved to the center of the new image.
    ox = (nw - w) / 2.0
    oy = (nh - h) / 2.0
    print(f"Calculated Shift (Top-Left): ({ox}, {oy})")

test_rotation_offset(50, 80, 15, 25, 78) # Typical "k" rotation
