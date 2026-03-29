import numpy as np
import cv2
from scipy.interpolate import CubicSpline

def get_skeleton_path(char):
    # Base skeleton paths for a few test characters
    # Normalized to [0,1] space
    atlas = {
        'a': [(0.8, 0.4), (0.5, 0.2), (0.2, 0.4), (0.2, 0.7), (0.5, 0.9), (0.8, 0.7), (0.8, 0.4), (0.8, 1.0)],
        'b': [(0.3, 0.0), (0.3, 1.0), (0.3, 0.7), (0.6, 0.6), (0.8, 0.75), (0.6, 1.0), (0.3, 1.0)],
        'c': [(0.8, 0.3), (0.5, 0.2), (0.2, 0.5), (0.2, 0.8), (0.5, 1.0), (0.8, 0.9)],
        'd': [(0.8, 0.0), (0.8, 1.0), (0.8, 0.7), (0.5, 0.6), (0.2, 0.8), (0.5, 1.0), (0.8, 1.0)],
        'e': [(0.2, 0.6), (0.8, 0.6), (0.8, 0.3), (0.5, 0.2), (0.2, 0.5), (0.2, 0.8), (0.5, 1.0), (0.8, 0.9)],
    }
    return atlas.get(char.lower(), [(0.5, 0.5)])

def generate_motor_stroke(skeleton_path, jitter=0.04):
    path = np.array(skeleton_path)
    # Add random jitter to control points
    noise = np.random.normal(0, jitter, path.shape)
    jittered_path = path + noise
    
    # Parametric spline interpolation
    t = np.linspace(0, 1, len(jittered_path))
    cs_x = CubicSpline(t, jittered_path[:, 0])
    cs_y = CubicSpline(t, jittered_path[:, 1])
    
    fine_t = np.linspace(0, 1, 100)
    return np.stack([cs_x(fine_t), cs_y(fine_t)], axis=1)

def render_sovereign_char(canvas, char, pos, size, color=(0,0,0)):
    skeleton = get_skeleton_path(char)
    stroke = generate_motor_stroke(skeleton)
    
    # Scale and move
    draw_points = (stroke * size + pos).astype(np.int32)
    
    for i in range(len(draw_points)-1):
        # Pressure-modulated thickness
        thickness = 1 + int(np.random.randint(0, 2))
        cv2.line(canvas, tuple(draw_points[i]), tuple(draw_points[i+1]), color, thickness, cv2.LINE_AA)

if __name__ == "__main__":
    # Test canvas
    canvas = np.ones((500, 1000, 3), dtype=np.uint8) * 255
    text = "abcde"
    for i, char in enumerate(text):
        render_sovereign_char(canvas, char, [100 + i*150, 100], 100)
    
    cv2.imwrite("sovereign_test.png", canvas)
