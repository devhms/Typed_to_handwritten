import cv2
import numpy as np
import json
from pathlib import Path

def get_paper_corners(img):
    """Detect the 4 corners of the paper sheet in the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours:
        perc = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perc, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def calibrate_paper(image_path: str):
    img = cv2.imread(image_path)
    if img is None: return {"error": "No image"}
    H_target, W_target = 3508, 2480
    img = cv2.resize(img, (W_target, H_target))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # [ROBUST] Morphological Line Detection
    # Use a horizontal kernel to highlight lines
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 15)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    
    # Projection profile on MORPHED lines
    proj = np.sum(detected_lines, axis=1)
    threshold = np.max(proj) * 0.2
    peaks = []
    for y in range(2, len(proj) - 2):
        if proj[y] > threshold and proj[y] == np.max(proj[y-2:y+3]):
            peaks.append(int(y))
    
    unique_ys = []
    if peaks:
        unique_ys.append(peaks[0])
        for p in peaks[1:]:
            if p - unique_ys[-1] > 60:
                unique_ys.append(p)
    
    # Vertical Margin (Red line) - use Red balance
    b, g, r = cv2.split(img)
    red_mask = cv2.subtract(r, g)
    margin_x = int(np.argmax(np.sum(red_mask, axis=0)))
    if margin_x < 100 or margin_x > 1000: margin_x = 420

    calibration = {
        "top_margin": unique_ys[0] if unique_ys else 400,
        "line_height": int(np.median(np.diff(unique_ys))) if len(unique_ys) > 5 else 130,
        "left_margin": margin_x + 80,
        "all_line_ys": unique_ys
    }
    
    with open("assets/calibration.json", "w") as f:
        json.dump(calibration, f, indent=4)
    print(f"Aggressive Calibration: {len(unique_ys)} lines found.")
    return calibration

if __name__ == "__main__":
    calibrate_paper("assets/paper_texture.png")
