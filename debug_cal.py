import cv2
import json
import numpy as np
from pathlib import Path

def debug_calibration():
    img = cv2.imread("assets/paper_texture.png")
    with open("assets/calibration.json", "r") as f:
        cal = json.load(f)
    
    line_ys = cal.get("all_line_ys", [])
    m_left = cal.get("left_margin", 100)
    
    # Draw horizontal lines
    for y in line_ys:
        cv2.line(img, (0, y), (img.shape[1], y), (255, 0, 0), 2)
    
    # Draw margin
    cv2.line(img, (m_left, 0), (m_left, img.shape[0]), (0, 0, 255), 2)
    
    cv2.imwrite("assets/calibration_debug.png", img)
    print(f"Debug image saved: {img.shape}")

if __name__ == "__main__":
    debug_calibration()
