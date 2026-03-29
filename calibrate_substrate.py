import cv2
import numpy as np
import json
from pathlib import Path

def calibrate():
    img_path = Path("assets/photorealistic_substrate.jpg")
    if not img_path.exists():
        print(f"Error: {img_path} not found.")
        return

    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screen_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        print("Failed to detect paper corners automatically.")
        # Fallback to manual heuristics if needed, or just warn
        return

    # Order corners: TL, TR, BR, BL
    pts = screen_cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    calibration = {
        "src_corners": rect.tolist(),
        "substrate_path": str(img_path),
        "target_size": [2480, 3508] # A4 @ 300DPI
    }

    with open("assets/calibration.json", "w") as f:
        json.dump(calibration, f, indent=4)

    # Debug image
    for pt in rect:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)
    cv2.imwrite("assets/calibration_debug.png", img)
    print("Calibration saved to assets/calibration.json")

if __name__ == "__main__":
    calibrate()
